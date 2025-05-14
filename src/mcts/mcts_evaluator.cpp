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
    // Set a descriptive thread name if supported by the platform
    #ifdef _MSC_VER
    // Visual Studio specific thread naming
    const char* threadName = "MCTSEvaluator";
    typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
    SetThreadDescriptionFunc setThreadDescription = 
        (SetThreadDescriptionFunc)GetProcAddress(
            GetModuleHandleA("kernel32.dll"), 
            "SetThreadDescription");
            
    if (setThreadDescription) {
        wchar_t wThreadName[100];
        swprintf(wThreadName, 100, L"%S", threadName);
        setThreadDescription(GetCurrentThread(), wThreadName);
    }
    #endif
    
    std::cout << "Starting MCTSEvaluator processing thread" << std::endl;
    
    while (!shutdown_flag_) {
        // Collect a batch of states
        auto batch = collectBatch();

        if (!batch.empty()) {
            // Process the batch
            processBatch(batch);
        } else {
            // If queue is empty, wait for a short period to avoid busy-waiting
            std::unique_lock<std::mutex> lock(cv_mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() {
                return request_queue_.size_approx() != 0 || shutdown_flag_;
            });
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

    // First dequeue as many requests as are immediately available, up to batch_size_
    EvaluationRequest request(nullptr, nullptr);
    
    size_t initial_queue_size = request_queue_.size_approx();
    size_t to_dequeue = std::min(initial_queue_size, batch_size_);
    
    // If we have a good number of requests, grab them immediately
    if (to_dequeue >= batch_size_ / 2) {
        for (size_t i = 0; i < to_dequeue; ++i) {
            if (request_queue_.try_dequeue(request)) {
                batch.push_back(std::move(request));
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
    if (batch.size() < batch_size_) {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        
        // Wait for more requests with timeout
        bool got_notification = cv_.wait_until(lock, timeout_point, [this, &batch, this_batch_size = batch_size_]() {
            return request_queue_.size_approx() + batch.size() >= this_batch_size || shutdown_flag_;
        });
        
        // If we got a notification or timed out, try to collect more requests
        if (got_notification || std::chrono::steady_clock::now() >= timeout_point) {
            // Try to fill the batch up to batch_size_
            size_t remaining = batch_size_ - batch.size();
            for (size_t i = 0; i < remaining; ++i) {
                if (request_queue_.try_dequeue(request)) {
                    batch.push_back(std::move(request));
                } else {
                    break;
                }
            }
        }
    }

    // Check if we have a full batch
    if (batch.size() >= batch_size_) {
        full_batches_.fetch_add(1, std::memory_order_relaxed);
    } else if (!batch.empty()) {
        partial_batches_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // This was a timeout with no requests
        timeouts_.fetch_add(1, std::memory_order_relaxed);
        
        // Only log the timeout if it's relatively infrequent, to avoid console spam
        static int timeout_log_counter = 0;
        if ((timeout_log_counter++ % 10) == 0) {
            std::cout << "Warning: No requests collected after " << timeout_.count() 
                     << "ms timeout" << std::endl;
        }
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
            } catch (const std::exception& e) {
                std::cerr << "Error setting default promise: " << e.what() << std::endl;
            }
        }
    }

    // If no valid states remain, we're done
    if (states.empty()) {
        return;
    }

    // Run batch inference
    std::vector<NetworkOutput> outputs;
    try {
        std::cout << "Running neural network inference for " << states.size() << " states" << std::endl;
        outputs = inference_fn_(states);
        
        // Check if we got valid outputs
        if (outputs.size() != states.size()) {
            std::cerr << "Error: neural network returned " << outputs.size()
                      << " outputs for " << states.size() << " inputs!" << std::endl;
            
            // Resize outputs vector if we got more than expected
            if (outputs.size() > states.size()) {
                outputs.resize(states.size());
            }
            // If we got fewer, we'll handle that by providing defaults below
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during neural network inference: " << e.what() << std::endl;
        // Continue to fulfill promises with defaults
    }

    // Fulfill promises for valid requests
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        size_t batch_idx = valid_indices[i];
        auto& request = batch[batch_idx];
        
        try {
            if (i < outputs.size()) {
                // We have a valid output from the neural network
                request.promise.set_value(std::move(outputs[i]));
            } else {
                // Create default output
                NetworkOutput default_output;
                default_output.value = 0.0f;
                
                // Set uniform policy if we can determine action space size
                if (request.node) {
                    int action_size = request.node->getState().getActionSpaceSize();
                    default_output.policy.resize(action_size, 1.0f / action_size);
                }
                
                request.promise.set_value(std::move(default_output));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error fulfilling promise: " << e.what() << std::endl;
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