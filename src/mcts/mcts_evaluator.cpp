// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>

namespace alphazero {
namespace mcts {

    // Extension method for std::thread to join with timeout
    inline bool join_for(std::thread& thread, std::chrono::duration<long long, std::micro> timeout) {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < timeout) {
            if (thread.joinable()) {
                try {
                    thread.join();
                    return true;
                } catch (...) {
                    return false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

MCTSEvaluator::MCTSEvaluator(InferenceFunction inference_fn, 
                           size_t batch_size, 
                           std::chrono::milliseconds timeout)
    : inference_fn_(std::move(inference_fn)),
      batch_size_(batch_size),
      timeout_(timeout),
      shutdown_flag_(true), // Start as shutdown
      total_batches_(0),
      total_evaluations_(0),
      cumulative_batch_size_(0),
      cumulative_batch_time_ms_(0),
      timeouts_(0),
      full_batches_(0),
      partial_batches_(0) {
    
    // Validate parameters
    if (batch_size_ < 1) {
        batch_size_ = 1;
    }
    if (timeout_ < std::chrono::milliseconds(1)) {
        timeout_ = std::chrono::milliseconds(1);
    }
}

MCTSEvaluator::~MCTSEvaluator() {
    stop();
}

void MCTSEvaluator::start() {
    // Use mutex to ensure clean thread start/stop
    std::lock_guard<std::mutex> lock(cv_mutex_);
    
    if (worker_thread_.joinable()) {
        return; // Already started
    }
    
    // Reset shutdown flag before starting thread
    shutdown_flag_.store(false, std::memory_order_release);
    
    // Clear the queue in case of previous leftover requests
    int cleared_items = 0;
    while (true) {
        // Create a fresh dummy request each time to avoid use-after-move issues
        EvaluationRequest dummy(nullptr, nullptr, 10);
        
        if (!request_queue_.try_dequeue(dummy)) {
            break; // No more items in queue
        }
        
        cleared_items++;
        
        // Fulfill any pending promises to avoid memory leaks
        try {
            NetworkOutput empty_output;
            empty_output.value = 0.0f;
            int action_size = dummy.action_space_size > 0 ? dummy.action_space_size : 10;
            empty_output.policy.resize(action_size, 1.0f / action_size);
            dummy.promise.set_value(std::move(empty_output));
        } catch (...) {
            // Promise might have been fulfilled or broken already
        }
    }
    
    try {
        worker_thread_ = std::thread(&MCTSEvaluator::processBatches, this);
    } catch (const std::exception& e) {
        shutdown_flag_.store(true, std::memory_order_release);
        std::cerr << "Failed to start MCTSEvaluator thread: " << e.what() << std::endl;
        throw;
    }
}

void MCTSEvaluator::stop() {
    // Graceful shutdown procedure
    bool need_join = false;
    
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        
        if (!worker_thread_.joinable() || shutdown_flag_.load(std::memory_order_acquire)) {
            return; // Already stopped or stopping
        }
        
        shutdown_flag_.store(true, std::memory_order_release);
        need_join = true;
        cv_.notify_all(); // Wake up worker if it's waiting
    }
    
    if (need_join) {
        // First, try joining directly with a short timeout
        if (worker_thread_.joinable()) {
            // Use a direct future-based approach for the main join attempt
            auto join_future = std::async(std::launch::async, [this]() {
                if (worker_thread_.joinable()) {
                    worker_thread_.join();
                    return true;
                }
                return false;
            });
            
            // Wait for joining to complete with a timeout
            if (join_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
                // Thread joined successfully
            } else {
                std::cerr << "Warning: Timeout waiting for MCTSEvaluator thread to join" << std::endl;
                
                // Secondary approach: create a dedicated joiner thread with longer timeout
                std::thread joiner([this]() {
                    if (worker_thread_.joinable()) {
                        worker_thread_.join();
                    }
                });
                
                if (joiner.joinable()) {
                    if (join_for(joiner, std::chrono::seconds(3))) {
                        // Thread joined on second attempt
                    } else {
                        // Still failed, detach the joiner thread
                        joiner.detach();
                        std::cerr << "ERROR: Timeout waiting for MCTSEvaluator thread to join" << std::endl;
                        std::cerr << "WARNING: MCTSEvaluator thread may be permanently blocked" << std::endl;
                    }
                }
            }
        }
        
        // Drain the queue to ensure no pending requests
        int cleared_requests = 0;
        while (true) {
            // Create a fresh request object for each iteration to avoid use-after-move issues
            EvaluationRequest request(nullptr, nullptr, 10);
            
            if (!request_queue_.try_dequeue(request)) {
                break; // Queue is empty
            }
            
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                default_output.policy.resize(action_size, 1.0f / action_size);
                request.promise.set_value(std::move(default_output));
                cleared_requests++;
            } catch (...) {
                // Promise might be broken already, just continue
            }
        }
    }
}

std::future<NetworkOutput> MCTSEvaluator::evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state) {
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        // Create a default response if evaluator is shutting down
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        default_output.value = 0.0f;
        
        try {
            if (node) {
                int action_size = node->getState().getActionSpaceSize();
                default_output.policy.resize(action_size, 1.0f / action_size);
            } else {
                default_output.policy.resize(10, 0.1f);
            }
        } catch (...) {
            // Default size if we can't get action space
            default_output.policy.resize(10, 0.1f);
        }
        
        promise.set_value(std::move(default_output));
        return promise.get_future();
    }
    
    // Create a safe policy size before potentially moving the state
    int policy_size = 10; // Default fallback
    if (node && state) {
        try {
            policy_size = state->getActionSpaceSize();
        } catch (...) {
            // Keep default size if we can't access action space
        }
    }
    
    if (!node || !state) {
        // Handle invalid input with immediate default response
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        default_output.value = 0.0f;
        default_output.policy.resize(policy_size, 1.0f / policy_size);
        
        promise.set_value(std::move(default_output));
        return promise.get_future();
    }
    
    // Create the request with a copy of the policy_size for safety
    EvaluationRequest request(node, std::move(state), policy_size);
    std::future<NetworkOutput> future = request.promise.get_future();
    
    // Enqueue the request
    bool enqueued = request_queue_.enqueue(std::move(request));
    
    if (!enqueued) {
        // Handle enqueue failure - should not happen with moodycamel queue
        std::promise<NetworkOutput> error_promise;
        NetworkOutput error_output;
        error_output.value = 0.0f;
        error_output.policy.resize(policy_size, 1.0f / policy_size);
        
        error_promise.set_value(std::move(error_output));
        return error_promise.get_future();
    }
    
    // Notify worker thread that a new request is available
    cv_.notify_all(); // Notify all waiting threads for redundancy
    
    return future;
}

size_t MCTSEvaluator::getQueueSize() const {
    return request_queue_.size_approx();
}

float MCTSEvaluator::getAverageBatchSize() const {
    size_t batches = total_batches_.load(std::memory_order_relaxed);
    if (batches == 0) return 0.0f;
    return static_cast<float>(cumulative_batch_size_.load(std::memory_order_relaxed)) / batches;
}

std::chrono::milliseconds MCTSEvaluator::getAverageBatchLatency() const {
    size_t batches = total_batches_.load(std::memory_order_relaxed);
    if (batches == 0) return std::chrono::milliseconds(0);
    return std::chrono::milliseconds(cumulative_batch_time_ms_.load(std::memory_order_relaxed) / batches);
}

size_t MCTSEvaluator::getTotalEvaluations() const {
    return total_evaluations_.load(std::memory_order_relaxed);
}

void MCTSEvaluator::processBatches() {
    // Thread name setting code (platform-specific, omitted for brevity)
    
    const int max_consecutive_errors = 5;
    int consecutive_errors = 0;
    int idle_count = 0;  // Track consecutive idle iterations
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        try {
            // Check shutdown flag at beginning of loop
            if (shutdown_flag_.load(std::memory_order_acquire)) {
                break;
            }
            
            // Get the current approximate queue size once to avoid repeated calls
            size_t queue_size = request_queue_.size_approx();
            
            // Process any requests in the queue
            if (queue_size > 0) {
                // Collect a batch of states with the timeout
                auto batch = collectBatch();
                
                if (!batch.empty()) {
                    // Process the batch safely
                    processBatch(batch);
                    consecutive_errors = 0; // Reset error counter on success
                    idle_count = 0;        // Reset idle counter on activity
                }
            } else {
                // If queue is empty, wait for a short period to avoid busy-waiting
                std::unique_lock<std::mutex> lock(cv_mutex_);
                
                // Adaptive wait timeout: use longer waits when consistently idle
                auto wait_timeout = (idle_count > 10) ? 
                    std::chrono::milliseconds(50) : std::chrono::milliseconds(10);
                    
                auto wait_result = cv_.wait_for(lock, wait_timeout, [this]() {
                    return request_queue_.size_approx() > 0 || shutdown_flag_.load(std::memory_order_acquire);
                });
                
                // Check shutdown flag immediately after wait
                if (shutdown_flag_.load(std::memory_order_acquire)) {
                    break;
                }
                
                // Handle idle state
                if (!wait_result) {
                    idle_count++;
                    
                    // Apply exponential backoff for sleep duration based on idle count
                    // to reduce CPU usage during prolonged idle periods
                    if (idle_count > 100) { // Very long idle time - back off significantly
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    } else if (idle_count > 10) { // Moderate idle time
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                } else {
                    // Reset idle count if we were woken up for a reason
                    idle_count = 0;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in MCTSEvaluator processing thread: " << e.what() << std::endl;
            consecutive_errors++;
            
            // If too many consecutive errors, sleep to avoid thrashing
            if (consecutive_errors >= max_consecutive_errors) {
                std::cerr << "Too many consecutive errors, sleeping to avoid thrashing" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                consecutive_errors = 0; // Reset after sleeping
            }
        } catch (...) {
            std::cerr << "Unknown error in MCTSEvaluator processing thread" << std::endl;
            consecutive_errors++;
            
            // If too many consecutive errors, sleep to avoid thrashing
            if (consecutive_errors >= max_consecutive_errors) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                consecutive_errors = 0; // Reset after sleeping
            }
        }
        
        // Extra check for shutdown flag at end of each loop iteration
        if (shutdown_flag_.load(std::memory_order_acquire)) {
            break;
        }
    }

    // Process any remaining items in the queue before exiting
    int cleaned_up_count = 0;
    try {
        // Drain the queue and fulfill any pending promises
        while (true) {
            // Create a fresh request object for each iteration
            EvaluationRequest request(nullptr, nullptr);
            
            if (!request_queue_.try_dequeue(request)) {
                break; // Queue is empty
            }
            
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                
                // Be extra careful about accessing node and state
                if (request.node) {
                    try {
                        int action_size = request.node->getState().getActionSpaceSize();
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    } catch (...) {
                        // Use stored action space size as fallback
                        int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    }
                } else {
                    // Use stored action space size as fallback
                    int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                }
                
                request.promise.set_value(std::move(default_output));
                cleaned_up_count++;
            } catch (...) {
                // Promise might be broken already, just continue
            }
        }
    } catch (...) {
        // Ignore any errors during final cleanup
    }
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch() {
    std::vector<EvaluationRequest> batch;
    batch.reserve(batch_size_);
    
    // Track the start time for timeout calculation
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_point = start_time + timeout_;
    
    // First try to dequeue as many requests as available immediately, up to batch_size_
    // Create a fresh request object each iteration to prevent reusing a moved-from object
    
    // Get an estimate of current queue size
    size_t initial_queue_size = request_queue_.size_approx();
    size_t to_dequeue = std::min(initial_queue_size, batch_size_);
    
    // If we have a good number of requests, grab them immediately
    if (to_dequeue >= 1) {
        for (size_t i = 0; i < to_dequeue; ++i) {
            // Check shutdown flag
            if (shutdown_flag_.load(std::memory_order_acquire)) {
                break;
            }
            
            // Create a fresh request object for each dequeue attempt
            EvaluationRequest request(nullptr, nullptr);
            
            if (request_queue_.try_dequeue(request)) {
                if (request.node && request.state) {
                    batch.push_back(std::move(request));
                } else {
                    // Handle invalid request with default output
                    try {
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        // Use the stored action_space_size for safety
                        int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        request.promise.set_value(std::move(default_output));
                    } catch (...) {
                        // Promise might be broken, ignore
                    }
                }
            } else {
                break; // Queue emptied faster than expected
            }
            
            // Process immediately if we have a full batch
            if (batch.size() >= batch_size_) {
                full_batches_.fetch_add(1, std::memory_order_relaxed);
                return batch;
            }
        }
    }
    
    // If batch isn't full but not empty, and we haven't reached the timeout yet,
    // wait for more requests
    if (!batch.empty() && batch.size() < batch_size_ && 
        std::chrono::steady_clock::now() < timeout_point && 
        !shutdown_flag_.load(std::memory_order_acquire)) {
        
        std::unique_lock<std::mutex> lock(cv_mutex_);
        
        // Wait for more requests with timeout
        auto remaining_time = timeout_point - std::chrono::steady_clock::now();
        if (remaining_time > std::chrono::milliseconds::zero()) {
            cv_.wait_for(lock, remaining_time, [this, &batch, this_batch_size = batch_size_]() {
                return request_queue_.size_approx() + batch.size() >= this_batch_size || 
                       shutdown_flag_.load(std::memory_order_acquire);
            });
        }
        
        // Try to collect more requests if shutdown hasn't been signaled
        if (!shutdown_flag_.load(std::memory_order_acquire)) {
            size_t remaining = batch_size_ - batch.size();
            for (size_t i = 0; i < remaining; ++i) {
                // Create a fresh request object for each dequeue attempt
                EvaluationRequest request(nullptr, nullptr);
                
                if (request_queue_.try_dequeue(request)) {
                    if (request.node && request.state) {
                        batch.push_back(std::move(request));
                    } else {
                        // Handle invalid request with default output
                        try {
                            NetworkOutput default_output;
                            default_output.value = 0.0f;
                            // Set empty policy vector
                            if (request.node) {
                                int action_size = request.node->getState().getActionSpaceSize();
                                default_output.policy.resize(action_size, 1.0f / action_size);
                            } else {
                                // Use the stored action_space_size for safety
                                int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                                default_output.policy.resize(action_size, 1.0f / action_size);
                            }
                            request.promise.set_value(std::move(default_output));
                        } catch (...) {
                            // Promise might be broken, ignore
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }
    
    // Update metrics
    if (batch.size() >= batch_size_) {
        full_batches_.fetch_add(1, std::memory_order_relaxed);
    } else if (!batch.empty()) {
        partial_batches_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // This was a timeout with no requests
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
    
    // Track indices of valid requests and store action space sizes
    std::vector<size_t> valid_indices;
    std::vector<int> action_space_sizes;
    valid_indices.reserve(batch.size());
    action_space_sizes.reserve(batch.size());
    
    // Collect valid states and track their indices
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& request = batch[i];
        
        // Double-check that node and state are valid
        if (request.node && request.state) {
            // Add a memory barrier to ensure node pointer visibility across threads
            std::atomic_thread_fence(std::memory_order_acquire);
            
            // Store the action space size before moving the state
            int action_size = 0;
            try {
                action_size = request.state->getActionSpaceSize();
            } catch (const std::exception& e) {
                std::cerr << "Error getting action space size: " << e.what() << std::endl;
                action_size = 10; // Default if we can't get actual size
            }
            
            valid_indices.push_back(i);
            action_space_sizes.push_back(action_size);
            states.push_back(std::move(request.state));
        } else {
            // Handle invalid request with default output
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                // Set empty policy vector
                if (request.node) {
                    try {
                        int action_size = request.node->getState().getActionSpaceSize();
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    } catch (...) {
                        // If we can't get action size, use a small default
                        default_output.policy.resize(10, 0.1f);
                    }
                } else {
                    default_output.policy.resize(10, 0.1f);
                }
                request.promise.set_value(std::move(default_output));
            } catch (...) {
                // Promise might be broken, ignore
            }
        }
    }
    
    // If no valid states remain, we're done
    if (states.empty()) {
        return;
    }
    
    // Run batch inference with exception handling
    std::vector<NetworkOutput> outputs;
    bool inference_succeeded = false;
    
    try {
        // Critical fix: ensure the inference_fn_ is valid
        if (!inference_fn_) {
            std::cerr << "ERROR: Neural network inference function is null during batch processing" << std::endl;
            throw std::runtime_error("Neural network inference function is null");
        }
        
        // Run the actual inference with additional safety checks
        try {
            outputs = inference_fn_(states);
        } catch (const std::exception& e) {
            std::cerr << "Exception during inference call: " << e.what() << std::endl;
            throw; // Re-throw after logging
        } catch (...) {
            std::cerr << "Unknown exception during inference call" << std::endl;
            throw; // Re-throw after logging
        }
        inference_succeeded = (outputs.size() == states.size());
        
        // Check if we got valid outputs
        if (!inference_succeeded) {
            std::cerr << "Error: Neural network returned " << outputs.size() 
                     << " outputs for " << states.size() << " inputs" << std::endl;
            
            // Resize outputs vector if we got more than expected
            if (outputs.size() > states.size()) {
                outputs.resize(states.size());
                inference_succeeded = true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during neural network inference: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error during neural network inference" << std::endl;
    }
    
    // Fulfill promises for valid requests
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        size_t batch_idx = valid_indices[i];
        
        // Check index validity
        if (batch_idx >= batch.size()) {
            std::cerr << "Invalid batch index " << batch_idx << " (batch size: " << batch.size() << ")" << std::endl;
            continue;
        }
        
        auto& request = batch[batch_idx];
        
        // Check for null request
        if (!request.node) {
            std::cerr << "Null node in request at index " << batch_idx << std::endl;
            continue;
        }
        
        // Add a memory barrier to ensure node pointer visibility across threads
        std::atomic_thread_fence(std::memory_order_acquire);
        
        try {
            if (inference_succeeded && i < outputs.size()) {
                // We have a valid output from the neural network
                request.promise.set_value(std::move(outputs[i]));
            } else {
                // Create default output on inference failure
                NetworkOutput default_output;
                default_output.value = 0.0f;
                
                // Set uniform policy using cached action space size
                if (i < action_space_sizes.size()) {
                    int action_size = action_space_sizes[i];
                    default_output.policy.resize(action_size, 1.0f / action_size);
                } else {
                    // Fallback if we don't have the cached size
                    try {
                        int action_size = request.node->getState().getActionSpaceSize();
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    } catch (...) {
                        // If we can't get action size, use a small default
                        default_output.policy.resize(10, 0.1f);
                    }
                }
                
                request.promise.set_value(std::move(default_output));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error fulfilling promise at index " << batch_idx << ": " << e.what() << std::endl;
            
            // Critical: Still need to fulfill the promise to prevent hanging threads
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                default_output.policy.resize(10, 0.1f);
                request.promise.set_value(std::move(default_output));
            } catch (...) {
                // Last resort - promise might be broken beyond repair
            }
        } catch (...) {
            std::cerr << "Unknown error fulfilling promise at index " << batch_idx << std::endl;
            
            // Critical: Still need to fulfill the promise to prevent hanging threads
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                default_output.policy.resize(10, 0.1f);
                request.promise.set_value(std::move(default_output));
            } catch (...) {
                // Last resort - promise might be broken beyond repair
            }
        }
    }
    
    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(elapsed_ms, std::memory_order_relaxed);
}

} // namespace mcts
} // namespace alphazero