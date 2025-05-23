#include "mcts/concurrent_request_aggregator.h"
#include <iostream>
#include <algorithm>
#include <future>

namespace alphazero {
namespace mcts {

ConcurrentRequestAggregator::ConcurrentRequestAggregator(
    std::shared_ptr<nn::NeuralNetwork> neural_net, 
    const AggregatorConfig& config)
    : neural_network_(neural_net), config_(config) {
    
    std::cout << "ConcurrentRequestAggregator: Initializing with:" << std::endl;
    std::cout << "  - Target batch size: " << config_.target_batch_size << std::endl;
    std::cout << "  - Max batch size: " << config_.max_batch_size << std::endl;
    std::cout << "  - Batch timeout: " << config_.batch_timeout.count() << "ms" << std::endl;
    std::cout << "  - Aggregator threads: " << config_.num_aggregator_threads << std::endl;
}

ConcurrentRequestAggregator::~ConcurrentRequestAggregator() {
    stop();
}

void ConcurrentRequestAggregator::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    
    // Start aggregator worker threads
    aggregator_threads_.reserve(config_.num_aggregator_threads);
    for (size_t i = 0; i < config_.num_aggregator_threads; ++i) {
        aggregator_threads_.emplace_back(&ConcurrentRequestAggregator::aggregatorWorkerLoop, this);
    }
    
    std::cout << "ConcurrentRequestAggregator: Started with " 
              << config_.num_aggregator_threads << " worker threads" << std::endl;
}

void ConcurrentRequestAggregator::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Wait for all worker threads to complete
    for (auto& thread : aggregator_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    aggregator_threads_.clear();
    
    std::cout << "ConcurrentRequestAggregator: Stopped" << std::endl;
}

std::vector<NetworkOutput> ConcurrentRequestAggregator::evaluateBatch(
    std::vector<std::unique_ptr<core::IGameState>> states) {
    
    if (states.empty()) {
        return {};
    }

    // Create request with promise for result
    uint64_t request_id = next_request_id_.fetch_add(1);
    PendingRequest request(std::move(states), request_id);
    
    // Get future for the result
    auto future = request.promise.get_future();
    
    // Submit to lock-free queue
    pending_request_count_.fetch_add(1);
    if (!request_queue_.enqueue(std::move(request))) {
        pending_request_count_.fetch_sub(1);
        stats_.requests_dropped.fetch_add(1);
        std::cerr << "ConcurrentRequestAggregator: Failed to enqueue request " << request_id << std::endl;
        return {};
    }
    
    stats_.total_requests.fetch_add(1);
    
    // Wait for result with timeout
    auto status = future.wait_for(config_.max_wait_time * 2);  // Allow extra time for batching
    if (status == std::future_status::timeout) {
        std::cerr << "ConcurrentRequestAggregator: Request " << request_id 
                  << " timed out after " << (config_.max_wait_time * 2).count() << "ms" << std::endl;
        return {};
    }
    
    try {
        return future.get();
    } catch (const std::exception& e) {
        std::cerr << "ConcurrentRequestAggregator: Exception in request " << request_id 
                  << ": " << e.what() << std::endl;
        return {};
    }
}

void ConcurrentRequestAggregator::aggregatorWorkerLoop() {
    std::vector<PendingRequest> current_batch;
    current_batch.reserve(config_.max_batch_size);
    
    while (running_.load()) {
        current_batch.clear();
        
        // PHASE 1: Collect requests using lock-free bulk operations
        size_t collected = collectRequests(current_batch);
        
        if (collected > 0) {
            // PHASE 2: Process the collected batch
            processBatch(current_batch);
            
            // Update statistics
            stats_.total_batches_processed.fetch_add(1);
            stats_.total_states_evaluated.fetch_add(collected);
            
            // Update average batch size
            double current_avg = stats_.average_batch_size.load();
            uint64_t batch_count = stats_.total_batches_processed.load();
            double new_avg = ((current_avg * (batch_count - 1)) + collected) / batch_count;
            stats_.average_batch_size.store(new_avg);
            
        } else {
            // Brief sleep when no requests available
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }
}

size_t ConcurrentRequestAggregator::collectRequests(std::vector<PendingRequest>& batch) {
    size_t collected = 0;
    
    // PHASE 1: Bulk collection for maximum efficiency
    std::array<PendingRequest, 32> bulk_buffer;
    size_t bulk_collected = request_queue_.try_dequeue_bulk(bulk_buffer.begin(), 
        std::min(static_cast<size_t>(32), config_.max_batch_size));
    
    for (size_t i = 0; i < bulk_collected; ++i) {
        batch.push_back(std::move(bulk_buffer[i]));
        collected++;
    }
    
    // PHASE 2: Additional single-item collection if space available
    PendingRequest single_request;
    while (collected < config_.max_batch_size && request_queue_.try_dequeue(single_request)) {
        batch.push_back(std::move(single_request));
        collected++;
    }
    
    // PHASE 3: Dynamic wait for more requests if we haven't reached target
    if (collected > 0 && collected < config_.target_batch_size) {
        auto start_wait = std::chrono::steady_clock::now();
        auto max_wait = collected >= config_.target_batch_size / 2 ? 
            config_.batch_timeout / 2 : config_.batch_timeout;
            
        while (collected < config_.target_batch_size && 
               std::chrono::steady_clock::now() - start_wait < max_wait) {
            
            if (request_queue_.try_dequeue(single_request)) {
                batch.push_back(std::move(single_request));
                collected++;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }
    
    pending_request_count_.fetch_sub(collected);
    return collected;
}

bool ConcurrentRequestAggregator::shouldProcessBatch(const std::vector<PendingRequest>& batch) const {
    if (batch.empty()) {
        return false;
    }
    
    // Always process if we have a good batch size
    if (batch.size() >= config_.target_batch_size) {
        return true;
    }
    
    // Check if oldest request is approaching timeout
    auto now = std::chrono::steady_clock::now();
    auto oldest_age = now - batch[0].submit_time;
    
    return oldest_age >= config_.batch_timeout;
}

void ConcurrentRequestAggregator::processBatch(std::vector<PendingRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    active_batch_count_.fetch_add(1);
    
    auto batch_start = std::chrono::steady_clock::now();
    
    try {
        // PHASE 1: Collect all states from all requests in the batch
        std::vector<std::unique_ptr<core::IGameState>> all_states;
        std::vector<size_t> request_state_counts;
        
        size_t total_states = 0;
        for (const auto& request : batch) {
            request_state_counts.push_back(request.states.size());
            total_states += request.states.size();
        }
        
        all_states.reserve(total_states);
        for (auto& request : batch) {
            for (auto& state : request.states) {
                all_states.push_back(std::move(state));
            }
        }
        
        std::cout << "[AGGREGATOR] Processing batch: " << batch.size() 
                  << " requests, " << total_states << " total states" << std::endl;
        
        // PHASE 2: Single neural network call with all states
        auto nn_results = neural_network_->inference(all_states);
        
        auto batch_end = std::chrono::steady_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            batch_end - batch_start);
            
        std::cout << "[AGGREGATOR] Batch completed in " << batch_duration.count() 
                  << "μs, " << (total_states * 1000000.0 / batch_duration.count()) 
                  << " states/sec" << std::endl;
        
        // PHASE 3: Distribute results back to individual requests
        size_t result_offset = 0;
        for (size_t req_idx = 0; req_idx < batch.size(); ++req_idx) {
            size_t state_count = request_state_counts[req_idx];
            
            std::vector<NetworkOutput> request_results;
            request_results.reserve(state_count);
            
            for (size_t i = 0; i < state_count; ++i) {
                if (result_offset + i < nn_results.size()) {
                    request_results.push_back(nn_results[result_offset + i]);
                }
            }
            
            // Fulfill the promise
            batch[req_idx].promise.set_value(std::move(request_results));
            result_offset += state_count;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[AGGREGATOR] Exception processing batch: " << e.what() << std::endl;
        
        // Fulfill all promises with empty results
        for (auto& request : batch) {
            try {
                request.promise.set_value({});
            } catch (...) {
                // Promise might already be set
            }
        }
    }
    
    active_batch_count_.fetch_sub(1);
}

} // namespace mcts
} // namespace alphazero