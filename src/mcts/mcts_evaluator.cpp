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
      shutdown_flag_(true), // Start as shutdown
      total_batches_(0),
      total_evaluations_(0),
      cumulative_batch_size_(0),
      cumulative_batch_time_ms_(0),
      timeouts_(0),
      full_batches_(0),
      partial_batches_(0) {
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
    shutdown_flag_ = false;
    
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
        shutdown_flag_ = true;
        std::cerr << "Failed to start MCTSEvaluator thread: " << e.what() << std::endl;
        throw;
    }
}

void MCTSEvaluator::stop() {
    // Graceful shutdown procedure
    bool need_join = false;
    
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        
        if (!worker_thread_.joinable() || shutdown_flag_) {
            return; // Already stopped or stopping
        }
        
        shutdown_flag_ = true;
        need_join = true;
        cv_.notify_all(); // Wake up worker if it's waiting
    }
    
    if (need_join) {
        // Wait for worker thread outside of the lock
        if (worker_thread_.joinable()) {
            worker_thread_.join();
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
        
        std::cout << "MCTSEvaluator stopped cleanly" << std::endl;
    }
}

std::future<NetworkOutput> MCTSEvaluator::evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state) {
    if (shutdown_flag_) {
        throw std::runtime_error("MCTSEvaluator is not running");
    }
    
    if (!node || !state) {
        // Handle invalid input with immediate default response
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        
        if (node) {
            int action_size = node->getState().getActionSpaceSize();
            default_output.policy.resize(action_size, 1.0f / action_size);
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
        
        if (node) {
            int action_size = node->getState().getActionSpaceSize();
            error_output.policy.resize(action_size, 1.0f / action_size);
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
    // Set a descriptive thread name if supported by the platform
    #ifdef _MSC_VER
    const DWORD MS_VC_EXCEPTION = 0x406D1388;
    #pragma pack(push, 8)
    struct THREADNAME_INFO {
        DWORD dwType;     // Must be 0x1000
        LPCSTR szName;    // Pointer to name (in user addr space)
        DWORD dwThreadID; // Thread ID (-1=caller thread)
        DWORD dwFlags;    // Reserved for future use, must be zero
    };
    #pragma pack(pop)
    
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = "MCTSEvaluator";
    info.dwThreadID = GetCurrentThreadId();
    info.dwFlags = 0;
    
    __try {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // Continue execution
    }
    #endif
    
    std::cout << "Starting MCTSEvaluator processing thread" << std::endl;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        try {
            // Collect a batch of states with the timeout
            auto batch = collectBatch();
            
            if (!batch.empty()) {
                // Process the batch safely
                processBatch(batch);
            } else if (request_queue_.size_approx() == 0) {
                // If queue is empty, wait for a short period to avoid busy-waiting
                std::unique_lock<std::mutex> lock(cv_mutex_);
                auto wait_result = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                    return request_queue_.size_approx() > 0 || shutdown_flag_.load(std::memory_order_acquire);
                });
                
                if (wait_result && !shutdown_flag_.load(std::memory_order_acquire)) {
                    // Request arrived, try to collect immediately
                    continue;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in MCTSEvaluator processing thread: " << e.what() << std::endl;
            // Continue running, don't crash the thread
        } catch (...) {
            std::cerr << "Unknown error in MCTSEvaluator processing thread" << std::endl;
            // Continue running, don't crash the thread
        }
        
        // Small yield to prevent tight loops
        std::this_thread::yield();
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
    if (to_dequeue >= batch_size_ / 2) {
        for (size_t i = 0; i < to_dequeue; ++i) {
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
        }
        
        if (batch.size() >= batch_size_) {
            full_batches_.fetch_add(1, std::memory_order_relaxed);
            return batch;
        }
    }
    
    // If we don't have enough requests yet, wait with timeout for more
    if (batch.size() < batch_size_ && !shutdown_flag_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        
        // Wait for more requests with timeout
        bool got_notification = cv_.wait_until(lock, timeout_point, [this, &batch, this_batch_size = batch_size_]() {
            return request_queue_.size_approx() + batch.size() >= this_batch_size || 
                   shutdown_flag_.load(std::memory_order_acquire);
        });
        
        // If we got a notification or timed out, try to collect more requests
        if ((got_notification && !shutdown_flag_.load(std::memory_order_acquire)) || 
            std::chrono::steady_clock::now() >= timeout_point) {
            
            // Try to fill the batch up to batch_size_
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
    
    // Update metrics based on what we collected
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
    
    // Track indices of valid requests
    std::vector<size_t> valid_indices;
    valid_indices.reserve(batch.size());
    
    // Collect valid states and track their indices
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& request = batch[i];
        
        // Double-check that node and state are valid
        if (request.node && request.state) {
            valid_indices.push_back(i);
            states.push_back(std::move(request.state));
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
        auto& request = batch[batch_idx];
        
        try {
            if (inference_succeeded && i < outputs.size()) {
                // We have a valid output from the neural network
                request.promise.set_value(std::move(outputs[i]));
            } else {
                // Create default output on inference failure
                NetworkOutput default_output;
                default_output.value = 0.0f;
                
                // Set uniform policy
                if (request.node) {
                    try {
                        int action_size = request.node->getState().getActionSpaceSize();
                        default_output.policy.resize(action_size, 1.0f / action_size);
                    } catch (...) {
                        // If we can't get action size, use a small default
                        default_output.policy.resize(10, 0.1f);
                    }
                } else {
                    // Fallback for null node
                    default_output.policy.resize(10, 0.1f);
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