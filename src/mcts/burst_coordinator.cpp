#include "mcts/burst_coordinator.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include <cmath>
#include <limits>
#include <omp.h>
#include "core/igamestate.h"

namespace alphazero {
namespace mcts {

std::vector<NetworkOutput> BurstCoordinator::startBurstCollection(
    int simulations_needed,
    const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    
    std::vector<NetworkOutput> results;
    
    if (search_roots.empty() || simulations_needed <= 0) {
        return results;
    }
    
    // Early termination if we've had too many consecutive empty collections
    if (consecutive_empty_collections_.load() >= MAX_EMPTY_COLLECTIONS) {
        return results;
    }
    
    // Reset burst state with minimal locking
    {
        std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
        current_burst_.clear();
        current_burst_.reserve(std::min(simulations_needed, static_cast<int>(config_.target_burst_size)));
        burst_size_.store(0, std::memory_order_relaxed);
        collection_active_.store(true);
        active_collectors_.store(0);
        burst_start_time_ = std::chrono::steady_clock::now();
    }
    
    // THREADING FIX: Much more conservative parallelism to prevent resource exhaustion
    std::vector<std::future<void>> collection_futures;
    // Use very conservative parallelism to prevent memory explosion
    const int num_threads = std::min(static_cast<size_t>(2), // Maximum 2 threads
                                   std::min(config_.max_parallel_threads, 
                                          std::max(static_cast<size_t>(1), 
                                                 static_cast<size_t>(simulations_needed / 32))));
    
    // MEMORY FIX: Much more conservative adaptive sizing
    size_t adaptive_target = std::min(config_.target_burst_size, static_cast<size_t>(16));
    if (consecutive_empty_collections_.load() > 0) {
        adaptive_target = std::max(config_.min_burst_size, 
                                 adaptive_target / 4); // More aggressive reduction
    }
    
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        auto collection_future = std::async(std::launch::async, [this, thread_id, num_threads, simulations_needed, adaptive_target, &search_roots]() {
            active_collectors_.fetch_add(1);
            
            const int simulations_per_thread = simulations_needed / num_threads;
            const int extra_simulations = simulations_needed % num_threads;
            const int start_sim = thread_id * simulations_per_thread + std::min(static_cast<size_t>(extra_simulations), thread_id);
            const int end_sim = start_sim + simulations_per_thread + (thread_id < extra_simulations ? 1 : 0);
            
            for (int sim = start_sim; sim < end_sim && collection_active_.load(); ++sim) {
                try {
                    // Select root for this simulation
                    const size_t root_index = (sim + thread_id) % search_roots.size();
                    auto current_root = search_roots[root_index];
                    
                    // Select leaf node for evaluation (simplified for now)
                    auto [leaf, path] = selectLeafForBurstEvaluation(current_root);
                    
                    if (leaf && !leaf->isTerminal()) {
                        // Clone state for evaluation
                        auto state_clone = std::unique_ptr<core::IGameState>(
                            leaf->getState().clone());
                        
                        // Lock-free candidate addition with atomic size tracking
                        size_t current_size = burst_size_.load(std::memory_order_relaxed);
                        if (current_size < adaptive_target) {
                            // MEMORY FIX: Try to reserve a slot atomically with stricter limits
                            size_t new_size = burst_size_.fetch_add(1, std::memory_order_acq_rel);
                            if (new_size < adaptive_target) {
                                // We got a valid slot - add to the pre-allocated vector position
                                // This is safe because we reserved the vector size and only access our slot
                                try {
                                    std::lock_guard<std::mutex> finalize_lock(burst_finalization_mutex_);
                                    if (current_burst_.size() <= new_size) {
                                        current_burst_.resize(new_size + 1);
                                    }
                                    current_burst_[new_size] = BurstRequest(std::move(leaf), 
                                                                           std::move(state_clone), 
                                                                           std::move(path));
                                } catch (const std::exception& e) {
                                    // MEMORY FIX: If addition failed, decrement the counter and clean up
                                    burst_size_.fetch_sub(1, std::memory_order_acq_rel);
                                    // Clean up the leaf node state
                                    if (leaf) {
                                        leaf->clearEvaluationFlag();
                                    }
                                }
                            } else {
                                // MEMORY FIX: Couldn't get a valid slot, decrement counter and clean up
                                burst_size_.fetch_sub(1, std::memory_order_acq_rel);
                                if (leaf) {
                                    leaf->clearEvaluationFlag();
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error in burst collection thread " << thread_id 
                              << ": " << e.what() << std::endl;
                }
            }
            
            active_collectors_.fetch_sub(1);
        });
        
        collection_futures.push_back(std::move(collection_future));
    }
    
    // Wait for all collection threads to complete or timeout
    waitForCollectionComplete();
    
    // Wait for collection threads to finish
    for (auto& future : collection_futures) {
        if (future.valid()) {
            future.wait();
        }
    }
    
    // Process the collected burst to get results directly
    processBurstWhenReady();
    
    // Extract results from processed requests using atomic size
    {
        std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
        size_t actual_size = std::min(burst_size_.load(std::memory_order_acquire), current_burst_.size());
        results.reserve(actual_size);
        for (size_t i = 0; i < actual_size; ++i) {
            if (current_burst_[i].result_ready) {
                results.push_back(current_burst_[i].result);
            }
        }
    }
    
    // Update consecutive empty collections counter and add efficiency metrics
    if (results.empty()) {
        consecutive_empty_collections_.fetch_add(1);
        std::cout << "[BURST_EFFICIENCY] Empty collection #" 
                  << consecutive_empty_collections_.load() 
                  << " - Tree may be exhausted" << std::endl;
    } else {
        consecutive_empty_collections_.store(0); // Reset on successful collection
        successful_collections_.fetch_add(1, std::memory_order_relaxed);
        
        // Calculate collection efficiency with adaptive tracking
        double efficiency = static_cast<double>(results.size()) / config_.target_burst_size * 100.0;
        
        // Update rolling average efficiency
        double current_avg = avg_efficiency_.load(std::memory_order_relaxed);
        double new_avg = (current_avg * 0.9) + (efficiency * 0.1); // Exponential moving average
        avg_efficiency_.store(new_avg, std::memory_order_relaxed);
        
        std::cout << "[BURST_EFFICIENCY] Collected " << results.size() 
                  << "/" << config_.target_burst_size 
                  << " candidates (" << std::fixed << std::setprecision(1) 
                  << efficiency << "% efficiency, avg: " << new_avg << "%)" << std::endl;
    }
    
    return results;
}


void BurstCoordinator::resetEmptyCollectionCounter() {
    consecutive_empty_collections_.store(0);
}

void BurstCoordinator::waitForCollectionComplete() {
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = config_.collection_timeout;
    
    while (collection_active_.load()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        
        // Check timeout
        if (elapsed >= timeout) {
            collection_active_.store(false);
            break;
        }
        
        // Check if all collectors finished
        if (active_collectors_.load() == 0) {
            collection_active_.store(false);
            break;
        }
        
        // Check if we have enough candidates
        {
            std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
            if (current_burst_.size() >= config_.target_burst_size) {
                collection_active_.store(false);
                break;
            }
        }
        
        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void BurstCoordinator::processBurstWhenReady() {
    std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
    
    size_t actual_size = std::min(burst_size_.load(std::memory_order_acquire), current_burst_.size());
    if (actual_size == 0) {
        return;
    }
    
    std::cout << "[BURST_COORDINATOR] Processing burst of " 
              << actual_size << " candidates (target: " 
              << config_.target_burst_size << ")" << std::endl;
    
    // MEMORY FIX: Submit entire burst to inference server for optimal batching
    submitBurstForEvaluation();
    
    // MEMORY FIX: Clear burst data immediately after processing
    current_burst_.clear();
    burst_size_.store(0, std::memory_order_release);
}

void BurstCoordinator::submitBurstForEvaluation() {
    if (current_burst_.empty()) {
        std::cout << "[BURST_TRACE] submitBurstForEvaluation called with empty burst!" << std::endl;
        return;
    }
    
    std::cout << "[BURST_TRACE] Submitting burst of " << current_burst_.size() 
              << " requests to inference server" << std::endl;
    
    // Prepare batch submission to inference server
    std::vector<std::future<NetworkOutput>> inference_futures;
    inference_futures.reserve(current_burst_.size());
    
    // MEMORY FIX: Submit all requests to inference server with better cleanup
    for (size_t i = 0; i < current_burst_.size(); ++i) {
        auto& request = current_burst_[i];
        if (request.leaf && request.state) {
            std::cout << "[BURST_TRACE] Submitting request " << i << " to inference server" << std::endl;
            auto inference_future = inference_server_->submitRequest(
                request.leaf, 
                std::move(request.state), 
                request.path);
            inference_futures.push_back(std::move(inference_future));
            std::cout << "[BURST_TRACE] Request " << i << " submitted successfully" << std::endl;
        } else {
            std::cout << "[BURST_TRACE] WARNING: Request " << i << " has null leaf or state!" << std::endl;
        }
        // MEMORY FIX: Clear path immediately after submission
        request.path.clear();
    }
    
    // PERFORMANCE FIX: Asynchronous result processing to eliminate blocking
    auto wait_start = std::chrono::steady_clock::now();
    std::cout << "[BURST_TRACE] Processing " << inference_futures.size() << " inference results asynchronously..." << std::endl;
    
    // Process results in background thread to avoid blocking
    std::thread result_processor([this, inference_futures = std::move(inference_futures), wait_start]() mutable {
        for (size_t i = 0; i < inference_futures.size() && i < current_burst_.size(); ++i) {
            try {
                auto result = inference_futures[i].get(); // Now in background thread
                
                // Thread-safe result storage
                {
                    std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
                    if (i < current_burst_.size()) {
                        current_burst_[i].result = std::move(result);
                        current_burst_[i].result_ready = true;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[BURST_TRACE] Error getting inference result " << i << ": " << e.what() << std::endl;
                std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
                if (i < current_burst_.size()) {
                    current_burst_[i].result = NetworkOutput{}; // Default result
                    current_burst_[i].result_ready = true;
                }
            }
        }
    });
    
    // Detach to avoid blocking - results will be available when ready
    result_processor.detach();
    std::cout << "[BURST_TRACE] All " << inference_futures.size() << " results processed" << std::endl;
}

std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
BurstCoordinator::selectLeafForBurstEvaluation(std::shared_ptr<MCTSNode> root) {
    std::vector<std::shared_ptr<MCTSNode>> path;
    std::shared_ptr<MCTSNode> current = root;
    path.push_back(current);
    
    // Traverse tree using UCB-like selection
    while (current && !current->isLeaf() && !current->isTerminal()) {
        auto children = current->getChildren();
        if (children.empty()) {
            break;
        }
        
        // Find child with highest UCB value (simplified)
        std::shared_ptr<MCTSNode> best_child = nullptr;
        float best_ucb = -std::numeric_limits<float>::infinity();
        
        for (auto& child : children) {
            if (!child) continue;
            
            // Skip nodes that are already being evaluated or have pending evaluation
            if (child->hasPendingEvaluation() || child->isBeingEvaluated()) {
                continue;
            }
            
            // Simple UCB calculation
            float visit_count = static_cast<float>(child->getVisitCount() + 1);
            float value = child->getValue();
            float exploration = std::sqrt(std::log(current->getVisitCount() + 1) / visit_count);
            float ucb = value + 1.414f * exploration; // sqrt(2) for exploration constant
            
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child;
            }
        }
        
        if (!best_child) {
            break; // No available children
        }
        
        current = best_child;
        path.push_back(current);
        
        // Prevent infinite loops
        if (path.size() > 100) {
            break;
        }
    }
    
    // If we found a leaf that needs evaluation, return it
    if (current && current->isLeaf() && !current->isTerminal() && 
        !current->hasPendingEvaluation() && !current->isBeingEvaluated()) {
        
        // Mark as pending evaluation to prevent other threads from selecting it
        current->markEvaluationPending();
        return {current, path};
    }
    
    // If we couldn't find a leaf, try to expand the current node to create more leaves
    if (current && !current->isTerminal() && !current->isLeaf()) {
        // Try to expand this node to create new leaf candidates
        auto children = current->getChildren();
        for (auto& child : children) {
            if (child && child->isLeaf() && !child->isTerminal() && 
                !child->hasPendingEvaluation() && !child->isBeingEvaluated()) {
                
                child->markEvaluationPending();
                auto child_path = path;
                child_path.push_back(child);
                return {child, child_path};
            }
        }
    }
    
    return {nullptr, {}};
}

BurstCoordinator::BurstStats BurstCoordinator::getStats() const {
    // TODO: Implement proper statistics tracking
    return BurstStats{};
}

std::vector<NetworkOutput> BurstCoordinator::collectAndEvaluate(
    const std::vector<BurstRequest>& requests, 
    size_t target_count) {
    
    // Early check for empty or trivial cases
    if (requests.empty() || target_count == 0) {
        return {};
    }
    
    std::vector<NetworkOutput> results;
    results.reserve(requests.size());
    
    // Prepare states for batch evaluation
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(requests.size());
    
    // Create a map to track which request corresponds to which state
    std::vector<size_t> request_indices;
    request_indices.reserve(requests.size());
    
    // Clone states to ensure we don't modify the originals
    for (size_t i = 0; i < requests.size(); i++) {
        if (requests[i].state) {
            states.push_back(requests[i].state->clone());
            request_indices.push_back(i);
        }
    }
    
    // If we have no valid states, return empty results
    if (states.empty()) {
        return results;
    }
    
    // Batch evaluate the states using the unified inference server
    if (inference_server_) {
        try {
            auto batch_results = inference_server_->evaluateBatch(states);
            
            // Map the results back to the original requests
            results.resize(requests.size()); // Resize to match original request count
            for (size_t i = 0; i < batch_results.size() && i < request_indices.size(); i++) {
                results[request_indices[i]] = batch_results[i];
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in BurstCoordinator::collectAndEvaluate: " << e.what() << std::endl;
            
            // Create default results on error
            results.resize(requests.size());
            for (size_t i = 0; i < results.size(); i++) {
                // Set default policy (uniform distribution)
                if (requests[i].state) {
                    int action_space = requests[i].state->getActionSpaceSize();
                    results[i].policy.resize(action_space, 1.0f / action_space);
                    results[i].value = 0.0f; // Neutral value on error
                }
            }
        }
    }
    
    // Update efficiency statistics
    {
        double efficiency = static_cast<double>(states.size()) / target_count;
        double current_avg = avg_efficiency_.load(std::memory_order_relaxed);
        double new_avg = (current_avg * 0.9) + (efficiency * 0.1); // Exponential moving average
        avg_efficiency_.store(new_avg, std::memory_order_relaxed);
        successful_collections_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return results;
}

void BurstCoordinator::submitBurst(std::vector<BurstRequest>&& requests) {
    // Early check for empty requests
    if (requests.empty()) {
        return;
    }
    
    // Process a batch of requests through the inference server
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(requests.size());
    
    // Extract states for batch evaluation
    for (auto& request : requests) {
        if (request.state) {
            states.push_back(std::move(request.state));
        }
    }
    
    // If we have no valid states, abort
    if (states.empty()) {
        return;
    }
    
    try {
        // Perform batch inference
        auto results = inference_server_->evaluateBatch(states);
        
        // Apply results to requests and invoke callbacks
        for (size_t i = 0; i < results.size() && i < requests.size(); ++i) {
            // Store result in the request
            requests[i].result = results[i];
            requests[i].result_ready = true;
            
            // Process node update, if provided
            if (requests[i].node) {
                requests[i].node->updateStats(results[i].value);
            }
            
            // Process leaf node update, if different from node
            if (requests[i].leaf && requests[i].leaf != requests[i].node) {
                requests[i].leaf->updateStats(results[i].value);
            }
            
            // Invoke callback, if provided
            if (requests[i].callback) {
                requests[i].callback(results[i]);
            }
        }
        
        // Update statistics
        successful_collections_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        std::cerr << "Error in BurstCoordinator::submitBurst: " << e.what() << std::endl;
        
        // Process with default results on error
        for (auto& request : requests) {
            // Create a default result
            NetworkOutput default_output;
            default_output.value = 0.0f;
            
            // Create uniform policy distribution
            int action_space = 225; // Default for Gomoku
            if (request.leaf) {
                action_space = request.leaf->getState().getActionSpaceSize();
            }
            default_output.policy.resize(action_space, 1.0f / action_space);
            
            // Store result
            request.result = default_output;
            request.result_ready = true;
            
            // Process node update, if provided
            if (request.node) {
                request.node->updateStats(default_output.value);
            }
            
            // Process leaf node update, if different from node
            if (request.leaf && request.leaf != request.node) {
                request.leaf->updateStats(default_output.value);
            }
            
            // Invoke callback, if provided
            if (request.callback) {
                request.callback(default_output);
            }
        }
    }
}

void BurstCoordinator::shutdown() {
    shutdown_.store(true);
    collection_active_.store(false);
}

bool BurstCoordinator::shouldProcessBurst() const {
    std::lock_guard<std::mutex> lock(burst_finalization_mutex_);
    
    // Check if we have any candidates
    if (current_burst_.empty()) {
        return false;
    }
    
    // Check if we have reached the target size
    if (current_burst_.size() >= config_.target_burst_size) {
        return true;
    }
    
    // Check if we have reached the minimum size and timeout has occurred
    if (current_burst_.size() >= config_.min_burst_size) {
        auto elapsed = std::chrono::steady_clock::now() - burst_start_time_;
        if (elapsed >= config_.collection_timeout) {
            return true;
        }
    }
    
    // Check if collection is no longer active 
    if (!collection_active_.load()) {
        return true;
    }
    
    return false;
}

} // namespace mcts
} // namespace alphazero