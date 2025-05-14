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
        std::cerr << "MCTSEvaluator::start - Thread already running" << std::endl;
        return; // Already started
    }
    
    // Reset shutdown flag before starting thread
    shutdown_flag_.store(false, std::memory_order_release);
    
    // Clear the queue in case of previous leftover requests
    EvaluationRequest dummy(nullptr, nullptr);
    while (request_queue_.try_dequeue(dummy)) {
        // Fulfill any pending promises to avoid memory leaks
        try {
            NetworkOutput empty_output;
            dummy.promise.set_value(std::move(empty_output));
        } catch (...) {
            // Promise might have been fulfilled or broken already
        }
    }
    
    try {
        worker_thread_ = std::thread(&MCTSEvaluator::processBatches, this);
        std::cout << "MCTSEvaluator started successfully" << std::endl;
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
        // Set timeout for joining the thread
        auto join_start = std::chrono::steady_clock::now();
        
        // Try to join with timeout (up to 5 seconds)
        if (worker_thread_.joinable()) {
            std::thread joiner([this]() {
                if (worker_thread_.joinable()) {
                    worker_thread_.join();
                }
            });
            
            if (joiner.joinable()) {
                if (join_for(joiner, std::chrono::seconds(5))) {
                    // Joined successfully
                } else {
                    // Timed out, detach the joiner
                    joiner.detach();
                    std::cerr << "Warning: Timed out waiting for evaluator thread to join" << std::endl;
                }
            }
        }
        
        // Clear any remaining requests with default responses
        EvaluationRequest request(nullptr, nullptr);
        while (request_queue_.try_dequeue(request)) {
            try {
                NetworkOutput default_output;
                if (request.node) {
                    int action_size = request.node->getState().getActionSpaceSize();
                    default_output.policy.resize(action_size, 1.0f / action_size);
                }
                request.promise.set_value(std::move(default_output));
            } catch (...) {
                // Promise might be broken already, just continue
            }
        }
        
        std::cout << "MCTSEvaluator stopped" << std::endl;
    }
}

std::future<NetworkOutput> MCTSEvaluator::evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state) {
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        throw std::runtime_error("MCTSEvaluator is not running");
    }
    
    if (!node || !state) {
        // Handle invalid input with immediate default response
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        
        if (node) {
            try {
                int action_size = node->getState().getActionSpaceSize();
                default_output.policy.resize(action_size, 1.0f / action_size);
            } catch (...) {
                // Default size if we can't get action space
                default_output.policy.resize(10, 0.1f);
            }
        }
        
        promise.set_value(std::move(default_output));
        return promise.get_future();
    }
    
    // Create the request
    EvaluationRequest request(node, std::move(state));
    std::future<NetworkOutput> future = request.promise.get_future();
    
    // Enqueue the request
    bool enqueued = request_queue_.enqueue(std::move(request));
    
    if (!enqueued) {
        // Handle enqueue failure - should not happen with moodycamel queue
        std::promise<NetworkOutput> error_promise;
        NetworkOutput error_output;
        error_output.value = 0.0f;
        
        try {
            if (node) {
                int action_size = node->getState().getActionSpaceSize();
                error_output.policy.resize(action_size, 1.0f / action_size);
            } else {
                error_output.policy.resize(10, 0.1f);
            }
        } catch (...) {
            // Default size if we can't get action space
            error_output.policy.resize(10, 0.1f);
        }
        
        error_promise.set_value(std::move(error_output));
        return error_promise.get_future();
    }
    
    // Notify worker thread that a new request is available
    cv_.notify_one();
    
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
    
    std::cout << "Starting MCTSEvaluator processing thread" << std::endl;
    
    const int max_consecutive_errors = 5;
    int consecutive_errors = 0;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        try {
            // Process any requests in the queue
            if (request_queue_.size_approx() > 0) {
                // Collect a batch of states with the timeout
                auto batch = collectBatch();
                
                if (!batch.empty()) {
                    // Process the batch safely
                    processBatch(batch);
                    consecutive_errors = 0; // Reset error counter on success
                }
            } else {
                // If queue is empty, wait for a short period to avoid busy-waiting
                std::unique_lock<std::mutex> lock(cv_mutex_);
                auto wait_result = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                    return request_queue_.size_approx() > 0 || shutdown_flag_.load(std::memory_order_acquire);
                });
                
                // Prevent tight spinning with a small sleep if we've been woken up
                // but there are still no requests and no shutdown signal
                if (!wait_result && !shutdown_flag_.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in MCTSEvaluator processing thread: " << e.what() << std::endl;
            consecutive_errors++;
            
            // If too many consecutive errors, sleep to avoid thrashing
            if (consecutive_errors >= max_consecutive_errors) {
                std::cerr << "Too many consecutive errors, sleeping..." << std::endl;
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
        
        // Prevent tight loops
        if (request_queue_.size_approx() == 0 && !shutdown_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    std::cout << "MCTSEvaluator processing thread exiting" << std::endl;
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch() {
    std::vector<EvaluationRequest> batch;
    batch.reserve(batch_size_);
    
    // Track the start time for timeout calculation
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_point = start_time + timeout_;
    
    // First try to dequeue as many requests as available immediately, up to batch_size_
    EvaluationRequest request(nullptr, nullptr);
    
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
                        }
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
        std::cout << "Running neural network inference for " << states.size() << " states" << std::endl;
        outputs = inference_fn_(states);
        inference_succeeded = (outputs.size() == states.size());
        
        // Check if we got valid outputs
        if (!inference_succeeded) {
            std::cerr << "Error: neural network returned " << outputs.size()
                      << " outputs for " << states.size() << " inputs!" << std::endl;
            
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
            std::cerr << "Invalid batch index: " << batch_idx << std::endl;
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
            std::cerr << "Error fulfilling promise: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error fulfilling promise" << std::endl;
        }
    }
    
    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(states.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(elapsed_ms, std::memory_order_relaxed);
    
    std::cout << "Batch processed: " << states.size() << " states in " << elapsed_ms << "ms" << std::endl;
}

} // namespace mcts
} // namespace alphazero