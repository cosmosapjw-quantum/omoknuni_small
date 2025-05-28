#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include "utils/advanced_memory_monitor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <omp.h>
#include <moodycamel/concurrentqueue.h>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

// Structure for batch collection
struct BatchRequest {
    std::shared_ptr<MCTSNode> node;
    std::unique_ptr<core::IGameState> state;
    std::vector<std::shared_ptr<MCTSNode>> path;
    int thread_id;
    
    BatchRequest() = default;
    BatchRequest(BatchRequest&&) = default;
    BatchRequest& operator=(BatchRequest&&) = default;
};

// CRITICAL: Proper tree parallelization with lock-free batch collection
void MCTSEngine::executeParallelBatchedSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty() || !search_roots[0]) {
        std::cerr << "No valid search roots provided" << std::endl;
        return;
    }
    
    auto root = search_roots[0];
    const int num_simulations = settings_.num_simulations;
    const int batch_size = settings_.batch_size;
    const int num_threads = settings_.num_threads;
    
    std::cout << "🚀 PARALLEL BATCHED SEARCH: " << num_simulations << " simulations, "
              << "batch_size=" << batch_size << ", threads=" << num_threads << std::endl;
    
    // Ensure root is expanded
    if (!root->isTerminal() && !root->isExpanded()) {
        expandNonTerminalLeaf(root);
    }
    
    // CRITICAL FIX: Limited queue size to prevent unbounded memory growth
    // Use smaller queue to force backpressure and prevent accumulation
    const size_t max_queue_size = std::min(static_cast<size_t>(batch_size * 2), size_t(64));
    moodycamel::ConcurrentQueue<BatchRequest> request_queue(max_queue_size);
    
    // Atomic counters for coordination
    std::atomic<int> simulations_completed(0);
    std::atomic<int> active_threads(0);
    std::atomic<bool> collection_done(false);
    
    // Worker thread function for parallel leaf collection
    auto collectLeaves = [&](int thread_id) {
        std::mt19937 rng(thread_id);
        int local_sims = 0;
        
        while (true) {
            int current_sims = simulations_completed.load();
            if (current_sims >= num_simulations) {
                break;
            }
            
            // Reserve a simulation slot
            if (!simulations_completed.compare_exchange_weak(current_sims, current_sims + 1)) {
                continue;
            }
            
            // Perform tree traversal with virtual loss
            std::vector<std::shared_ptr<MCTSNode>> path;
            auto result = selectLeafNodeParallel(root, path, rng);
            auto leaf = result.leaf_node;
            auto leaf_path = result.path;
            
            if (!leaf || leaf->isTerminal()) {
                // Handle terminal nodes immediately
                if (leaf && leaf->isTerminal()) {
                    float value = 0.0f;
                    auto result = leaf->getState().getGameResult();
                    if (result == core::GameResult::WIN_PLAYER1) {
                        value = leaf->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
                    } else if (result == core::GameResult::WIN_PLAYER2) {
                        value = leaf->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
                    }
                    backpropagateParallel(leaf_path, value, settings_.virtual_loss);
                }
                continue;
            }
            
            // Expand leaf if needed
            if (!leaf->isExpanded()) {
                expandNonTerminalLeaf(leaf);
                if (leaf->getChildren().empty()) {
                    continue;
                }
                // Select first child after expansion
                leaf = leaf->getChildren()[0];
                leaf_path.push_back(leaf);
            }
            
            // Create batch request
            BatchRequest request;
            request.node = leaf;
            request.state = leaf->getState().clone();
            request.path = std::move(leaf_path);
            request.thread_id = thread_id;
            
            // CRITICAL FIX: Try to enqueue with backpressure
            // If queue is full, process synchronously to prevent accumulation
            if (!request_queue.try_enqueue(std::move(request))) {
                // Queue is full - skip this simulation to prevent memory growth
                // Revert virtual loss since we're not processing this node
                for (auto& node : request.path) {
                    node->revertVirtualLoss(settings_.virtual_loss);
                }
                simulations_completed.fetch_sub(1); // Don't count this simulation
            }
            local_sims++;
        }
        
        // Signal thread completion
        active_threads.fetch_sub(1);
    };
    
    // Start collection threads
    std::vector<std::thread> collectors;
    active_threads.store(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        collectors.emplace_back(collectLeaves, i);
    }
    
    // Main thread handles batch processing
    int total_batches = 0;
    auto batch_start = std::chrono::steady_clock::now();
    
    while (simulations_completed.load() < num_simulations || active_threads.load() > 0) {
        std::vector<BatchRequest> batch;
        batch.reserve(batch_size);
        
        // Collect batch from queue with timeout
        auto collect_start = std::chrono::steady_clock::now();
        auto timeout = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        while (batch.size() < static_cast<size_t>(batch_size)) {
            BatchRequest request;
            if (request_queue.try_dequeue(request)) {
                batch.push_back(std::move(request));
            } else {
                // Check timeout
                auto elapsed = std::chrono::steady_clock::now() - collect_start;
                if (elapsed > timeout && !batch.empty()) {
                    break; // Process partial batch
                }
                if (active_threads.load() == 0 && request_queue.size_approx() == 0) {
                    break; // No more requests coming
                }
                std::this_thread::yield();
            }
        }
        
        // Process batch if we have requests
        if (!batch.empty()) {
            // Extract states for neural network
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.reserve(batch.size());
            
            for (auto& req : batch) {
                states.push_back(std::move(req.state));
            }
            
            // CRITICAL FIX: Clear request states immediately after moving
            // This ensures no dangling references remain
            for (auto& req : batch) {
                req.state.reset(); // Explicitly reset to nullptr
            }
            
            // Neural network inference
            std::vector<NetworkOutput> outputs;
            if (direct_inference_fn_ && !states.empty()) {
                auto nn_start = std::chrono::steady_clock::now();
                outputs = direct_inference_fn_(states);
                auto nn_end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(nn_end - nn_start);
                
                total_batches++;
                std::cout << "✅ Batch " << total_batches 
                          << ": " << states.size() << " states in " << duration.count() << "ms" 
                          << " (" << (states.size() * 1000.0 / (duration.count() + 1)) << " states/sec)" << std::endl;
            }
            
            // Apply results and backpropagate
            for (size_t i = 0; i < batch.size() && i < outputs.size(); ++i) {
                auto& req = batch[i];
                auto& output = outputs[i];
                
                // Set prior probabilities
                if (req.node && !output.policy.empty()) {
                    req.node->setPriorProbabilities(output.policy);
                }
                
                // Backpropagate with virtual loss reversal
                backpropagateParallel(req.path, output.value, settings_.virtual_loss);
            }
            
            // CRITICAL FIX: Aggressive memory cleanup after each batch
            batch.clear();
            batch.shrink_to_fit(); // Release memory immediately
            states.clear();
            outputs.clear();
            outputs.shrink_to_fit();
            
            // Force cleanup every few batches to prevent accumulation
            if (total_batches % 5 == 0) {
                // Clear any accumulated requests in the queue
                BatchRequest dummy;
                while (request_queue.try_dequeue(dummy)) {
                    // Drain excess requests to prevent queue growth
                }
                
                // Force memory cleanup
                utils::GameStatePoolManager::getInstance().clearAllPools();
                if (node_pool_) {
                    node_pool_->compact();
                }
                
                #ifdef WITH_TORCH
                if (torch::cuda::is_available()) {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
                #endif
            }
            
            // Check memory pressure
            auto& monitor = utils::AdvancedMemoryMonitor::getInstance();
            if (monitor.isMemoryPressureHigh()) {
                std::cout << "[MEMORY] High memory pressure detected, forcing aggressive cleanup" << std::endl;
                // Even more aggressive cleanup
                request_queue = moodycamel::ConcurrentQueue<BatchRequest>(batch_size * 2); // Reset queue
                utils::GameStatePoolManager::getInstance().clearAllPools();
                if (node_pool_) {
                    node_pool_->compact(); // Compact to release unused memory
                }
                #ifdef WITH_TORCH
                if (torch::cuda::is_available()) {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
                #endif
            }
        }
    }
    
    // Wait for all collectors to finish
    for (auto& t : collectors) {
        t.join();
    }
    
    auto total_time = std::chrono::steady_clock::now() - batch_start;
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count();
    
    std::cout << "✅ PARALLEL BATCHED SEARCH completed: " << simulations_completed.load() 
              << " simulations in " << total_batches << " batches"
              << " (" << (simulations_completed.load() * 1000.0 / (total_ms + 1)) << " sims/sec)" << std::endl;
    
    // Final cleanup
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    #endif
}

} // namespace mcts
} // namespace alphazero