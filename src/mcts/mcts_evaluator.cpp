// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>

namespace alphazero {
namespace mcts {

MCTSEvaluator::MCTSEvaluator(InferenceFunction inference_fn, 
                           size_t batch_size, 
                           std::chrono::milliseconds timeout)
    : inference_fn_(std::move(inference_fn)),
      batch_size_(batch_size),
      timeout_(timeout),
      shutdown_flag_(false),
      total_batches_(0),
      total_evaluations_(0),
      cumulative_batch_size_(0),
      cumulative_batch_time_ms_(0) {
}

MCTSEvaluator::~MCTSEvaluator() {
    stop();
}

void MCTSEvaluator::start() {
    if (worker_thread_.joinable()) {
        return; // Already started
    }
    
    shutdown_flag_ = false;
    worker_thread_ = std::thread(&MCTSEvaluator::processBatches, this);
}

void MCTSEvaluator::stop() {
    if (!worker_thread_.joinable()) {
        return; // Already stopped
    }
    
    shutdown_flag_ = true;
    cv_.notify_all(); // Wake up worker if it's waiting
    worker_thread_.join();
}

std::future<NetworkOutput> MCTSEvaluator::evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state) {
    EvaluationRequest request(node, std::move(state));
    std::future<NetworkOutput> future = request.promise.get_future();
    
    request_queue_.enqueue(std::move(request));
    
    // Notify worker thread that a new request is available
    cv_.notify_one();
    
    return future;
}

size_t MCTSEvaluator::getQueueSize() const {
    return request_queue_.size_approx();
}

float MCTSEvaluator::getAverageBatchSize() const {
    size_t batches = total_batches_.load();
    if (batches == 0) return 0.0f;
    return static_cast<float>(cumulative_batch_size_.load()) / batches;
}

std::chrono::milliseconds MCTSEvaluator::getAverageBatchLatency() const {
    size_t batches = total_batches_.load();
    if (batches == 0) return std::chrono::milliseconds(0);
    return std::chrono::milliseconds(cumulative_batch_time_ms_.load() / batches);
}

size_t MCTSEvaluator::getTotalEvaluations() const {
    return total_evaluations_.load();
}

void MCTSEvaluator::processBatches() {
    std::cout << "Starting MCTSEvaluator processing thread" << std::endl;
    
    // Performance stats tracking
    size_t total_batch_count = 0;
    size_t total_batch_size = 0;
    auto last_status_time = std::chrono::steady_clock::now();

    while (!shutdown_flag_) {
        // Collect a batch of states
        auto batch = collectBatch();

        if (!batch.empty()) {
            // Process the batch
            processBatch(batch);

            // Update batch statistics
            total_batch_count++;
            total_batch_size += batch.size();
            
            // Log periodic status (every ~5 seconds)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time);
            if (elapsed.count() >= 5) {
                float avg_batch_size = total_batch_count > 0 ? 
                    static_cast<float>(total_batch_size) / total_batch_count : 0.0f;
                
                std::cout << "Neural network evaluator status: "
                          << total_batch_count << " batches processed, "
                          << "avg batch size: " << avg_batch_size << ", "
                          << "queue size: " << request_queue_.size_approx() << ", "
                          << "full/partial/timeouts: " 
                          << full_batches_.load() << "/" 
                          << partial_batches_.load() << "/" 
                          << timeouts_.load() << std::endl;
                          
                last_status_time = now;
            }
        } else {
            // Wait a short time to avoid tight loops when queue is empty
            // but don't sleep too long to maintain responsiveness
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    std::cout << "MCTSEvaluator thread shutting down..." << std::endl;
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch() {
    std::vector<EvaluationRequest> batch;
    batch.reserve(batch_size_);

    // Start with a timestamp for timeout calculation
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_point = start_time + timeout_;

    // Try to collect up to batch_size_ requests
    EvaluationRequest request(nullptr, nullptr);

    // First quick pass to collect immediately available items
    for (size_t i = 0; i < batch_size_; ++i) {
        if (request_queue_.try_dequeue(request)) {
            batch.push_back(std::move(request));
        } else {
            break; // No more available immediately
        }
    }

    // If we already have a full batch, return it
    if (batch.size() >= batch_size_) {
        std::cout << "Full batch collected immediately: " << batch.size() << " requests" << std::endl;
        full_batches_.fetch_add(1, std::memory_order_relaxed);
        return batch;
    }

    // We don't have a full batch yet. There are two possible scenarios:
    // 1. We have some requests (0 < batch.size() < batch_size_)
    // 2. We have no requests (batch.size() == 0)

    // Use condition variable for efficient waiting
    std::unique_lock<std::mutex> lock(cv_mutex_);
    
    // Wait for notification with timeout
    // When we wake up, check if we got enough requests or timed out
    bool notification = cv_.wait_until(lock, timeout_point, [this, &batch_size = batch_size_, &batch] {
        return request_queue_.size_approx() + batch.size() >= batch_size || shutdown_flag_;
    });

    // Check for shutdown first
    if (shutdown_flag_) {
        return batch;
    }
    
    // Try to collect all available requests
    while (batch.size() < batch_size_) {
        if (request_queue_.try_dequeue(request)) {
            batch.push_back(std::move(request));
        } else {
            break;
        }
    }
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
    
    // Important: Always process whatever we have collected, even if it's a partial batch
    // Log diagnostic information
    if (batch.size() > 0) {
        std::cout << "Collected batch of " << batch.size() << "/" << batch_size_
                  << " requests in " << elapsed.count() << "ms" << std::endl;
        partial_batches_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // This shouldn't happen often - if the queue is completely empty after timeout
        std::cout << "Warning: No requests collected after " << elapsed.count() << "ms timeout" << std::endl;
        timeouts_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return batch;
}

void MCTSEvaluator::processBatch(std::vector<EvaluationRequest>& batch) {
    // Don't process empty batches
    if (batch.empty()) {
        return;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Prepare states for inference
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    // Track indices of valid requests
    std::vector<size_t> valid_indices;
    valid_indices.reserve(batch.size());

    // Collect valid states and track their indices
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& request = batch[i];
        if (request.state) {
            valid_indices.push_back(i);
            states.push_back(std::move(request.state));
        } else {
            std::cerr << "ERROR: Null game state in batch request at index " << i << std::endl;
            // Immediately fulfill invalid request with default output
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                // Set an empty policy vector of appropriate size
                request.promise.set_value(std::move(default_output));
            } catch (const std::exception& e) {
                std::cerr << "Error setting default promise for null state: " << e.what() << std::endl;
            }
        }
    }

    // If no valid states remain, we're done
    if (states.empty()) {
        std::cerr << "Warning: No valid states in batch!" << std::endl;
        return;
    }

    // Run batch inference with enhanced error handling
    std::vector<NetworkOutput> outputs;
    try {
        std::cout << "Running neural network inference for " << states.size() << " states" << std::endl;
        outputs = inference_fn_(states);
        std::cout << "Neural network returned " << outputs.size() << " outputs" << std::endl;

        if (outputs.size() != states.size()) {
            std::cerr << "Warning: neural network returned " << outputs.size()
                      << " outputs for " << states.size() << " inputs!" << std::endl;
            
            // Resize outputs to match states if we got more outputs than expected
            if (outputs.size() > states.size()) {
                outputs.resize(states.size());
            }
            // If we got fewer outputs, we'll handle that below by providing defaults
        }
    } catch (const std::bad_alloc& e) {
        // Special handling for memory errors
        std::cerr << "MEMORY ERROR during neural network inference: " << e.what() << std::endl;
        // Create empty output vector - we'll provide defaults below
        outputs.clear();
    } catch (const std::exception& e) {
        // Handle other inference errors
        std::cerr << "Neural network inference error: " << e.what() << std::endl;
        // Create empty output vector - we'll provide defaults below
        outputs.clear();
    }

    // Distribute results to requesters via promises
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        size_t batch_idx = valid_indices[i];
        
        try {
            if (i < outputs.size()) {
                // We have a valid output from the neural network
                batch[batch_idx].promise.set_value(std::move(outputs[i]));
            } else {
                // Provide default output for this state
                NetworkOutput default_output;
                default_output.value = 0.0f;
                
                // Determine appropriate policy size
                if (batch[batch_idx].node) {
                    try {
                        int action_size = batch[batch_idx].node->getState().getActionSpaceSize();
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    } catch (...) {
                        // If we can't determine action space, use empty policy
                        default_output.policy.clear();
                    }
                }
                
                batch[batch_idx].promise.set_value(std::move(default_output));
            }
        } catch (const std::exception& e) {
            // This is a critical error - if we can't fulfill the promise,
            // MCTS threads waiting on these futures will deadlock
            std::cerr << "CRITICAL: Error setting promise result: " << e.what() << std::endl;
            try {
                // Last desperate attempt to prevent deadlock
                NetworkOutput emergency_output;
                emergency_output.value = 0.0f;
                batch[batch_idx].promise.set_value(std::move(emergency_output));
            } catch (...) {
                // Nothing more we can do
                std::cerr << "FATAL: Could not fulfill promise for request " << batch_idx << std::endl;
            }
        }
    }

    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Update counters
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(elapsed_ms, std::memory_order_relaxed);
    
    std::cout << "Batch processed: " << states.size() << " states in " << elapsed_ms << "ms" << std::endl;
}

} // namespace mcts
} // namespace alphazero