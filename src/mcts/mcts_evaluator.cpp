// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
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
    
    // Validate parameters and set reasonable defaults
    if (batch_size_ < 1) {
        batch_size_ = 1;
    }
    
    if (timeout_ < std::chrono::milliseconds(1)) {
        timeout_ = std::chrono::milliseconds(1);
    }
    
    // Set min batch size based on total batch size
    // For GPU efficiency, we want at least 25% of the batch to be filled
    min_batch_size_ = std::max(static_cast<size_t>(1), batch_size_ / 4);
    
    // Additional wait time should be much shorter than the main timeout
    // to avoid stalling when we already have enough work to do
    additional_wait_time_ = std::chrono::milliseconds(std::min(static_cast<long>(25), timeout_.count() / 10));
    
    std::cout << "MCTSEvaluator: Created with batch_size=" << batch_size_ 
              << ", timeout=" << timeout_.count() << "ms"
              << ", min_batch_size=" << min_batch_size_
              << ", additional_wait_time=" << additional_wait_time_.count() << "ms" << std::endl;
}

MCTSEvaluator::~MCTSEvaluator() {
    // Ensure proper shutdown and cleanup
    try {
        // First, stop the worker thread
        stop();
        
        // Clear any remaining requests to prevent memory leaks
        int cleared_count = 0;
        while (true) {
            EvaluationRequest req(nullptr, nullptr);
            if (!request_queue_.try_dequeue(req)) {
                break;
            }
            
            // Fulfill any pending promises
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                default_output.policy.resize(req.action_space_size > 0 ? req.action_space_size : 10, 
                                            1.0f / (req.action_space_size > 0 ? req.action_space_size : 10));
                req.promise.set_value(std::move(default_output));
                cleared_count++;
            } catch (...) {
                // Promise might already be fulfilled or broken
            }
        }
        
        if (cleared_count > 0) {
            std::cout << "MCTSEvaluator::~MCTSEvaluator - Cleared " << cleared_count << " residual requests" << std::endl;
        }
        
        // Release any resources and memory
        inference_fn_ = nullptr;
    } catch (const std::exception& e) {
        std::cerr << "MCTSEvaluator::~MCTSEvaluator - Error during cleanup: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "MCTSEvaluator::~MCTSEvaluator - Unknown error during cleanup" << std::endl;
    }
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
    
    if (cleared_items > 0) {
       std::cout << "MCTSEvaluator::start - Cleared " << cleared_items << " leftover items from queue before start." << std::endl;
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
    // Check for shutdown condition
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
        } catch (const std::exception& e) {
            // Default size if we can't get action space
            default_output.policy.resize(10, 0.1f);
        } catch (...) {
            // Default size if we can't get action space
            default_output.policy.resize(10, 0.1f);
        }
        
        promise.set_value(std::move(default_output));
        return promise.get_future();
    }
    
    // Create a safe policy size before potentially moving the state
    int policy_size = 10; // Default fallback
    if (state) {
        try {
            policy_size = state->getActionSpaceSize();
        } catch (const std::exception& e) {
            std::cerr << "MCTSEvaluator::evaluateState - Exception getting action space size: " << e.what() << std::endl;
            // Keep default size if we can't access action space
        } catch (...) {
            std::cerr << "MCTSEvaluator::evaluateState - Unknown exception getting action space size" << std::endl;
            // Keep default size if we can't access action space
        }
    } else {
        std::cerr << "MCTSEvaluator::evaluateState - Warning: Null state passed to evaluateState" << std::endl;
    }
    
    if (!node) {
        std::cerr << "MCTSEvaluator::evaluateState - Warning: Null node passed to evaluateState" << std::endl;
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
    
    // Validate state before enqueueing
    try {
        if (!state->validate()) {
            std::cerr << "MCTSEvaluator::evaluateState - Warning: State validation failed" << std::endl;
            std::promise<NetworkOutput> promise;
            NetworkOutput default_output;
            default_output.value = 0.0f;
            default_output.policy.resize(policy_size, 1.0f / policy_size);
            promise.set_value(std::move(default_output));
            return promise.get_future();
        }
    } catch (const std::exception& e) {
        std::cerr << "MCTSEvaluator::evaluateState - Exception validating state: " << e.what() << std::endl;
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        default_output.value = 0.0f;
        default_output.policy.resize(policy_size, 1.0f / policy_size);
        promise.set_value(std::move(default_output));
        return promise.get_future();
    } catch (...) {
        std::cerr << "MCTSEvaluator::evaluateState - Unknown exception validating state" << std::endl;
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
        std::cerr << "MCTSEvaluator::evaluateState - Failed to enqueue request (should never happen)" << std::endl;
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
    std::cout << "[EVAL_THREAD] MCTSEvaluator::processBatches worker thread started." << std::endl;
    
    // Variables for error handling and cleanup
    const int max_consecutive_errors = 5;
    int consecutive_errors = 0;
    
    // Periodic cleanup timer
    auto last_cleanup_time = std::chrono::steady_clock::now();
    const auto cleanup_interval = std::chrono::minutes(2); // More frequent cleanup
    
    // Memory stats tracking
    size_t peak_batch_size = 0;
    size_t total_processed_states = 0;
    size_t cleanup_counter = 0;
    
    // Track performance to adapt batch sizes
    auto last_performance_check = std::chrono::steady_clock::now();
    const auto performance_check_interval = std::chrono::minutes(1);
    std::deque<float> recent_batch_times;
    std::deque<size_t> recent_batch_sizes;
    const size_t max_history = 20;

    try {
        while (!shutdown_flag_.load(std::memory_order_acquire)) {
            try {
                // Periodic cleanup to prevent memory leaks - more frequent than before
                auto current_time = std::chrono::steady_clock::now();
                bool is_cleanup_time = current_time - last_cleanup_time > cleanup_interval;
                bool is_idle = request_queue_.size_approx() == 0;
                
                // Clean up more aggressively during idle periods or periodically
                if (is_cleanup_time || (is_idle && cleanup_counter % 100 == 0)) {
                    cleanup_counter = 0;
                    
                    // Force a small GC in PyTorch CUDA allocator if using CUDA
                    if (torch::cuda::is_available()) {
                        try {
                            // First synchronize all CUDA streams to ensure operations are complete
                            torch::cuda::synchronize();
                            
                            // Clear CUDA cache to free memory
                            c10::cuda::CUDACachingAllocator::emptyCache();
                            
                            std::cout << "[EVAL_THREAD] CUDA memory cache emptied (" 
                                    << (is_cleanup_time ? "periodic" : "idle") << ")" << std::endl;
                            
                            // Print memory stats for debugging during periodic cleanup
                            for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                                size_t free, total;
                                cudaSetDevice(dev);
                                cudaMemGetInfo(&free, &total);
                                std::cout << "[EVAL_THREAD] GPU " << dev << " memory: " << (total - free) / 1048576 
                                        << "MB used, " << free / 1048576 << "MB free of " 
                                        << total / 1048576 << "MB total" << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[EVAL_THREAD] Error during CUDA cleanup: " << e.what() << std::endl;
                        }
                    }
                    
                    if (is_cleanup_time) {
                        last_cleanup_time = current_time;
                        
                        // Reset the state tracking counters periodically
                        // to avoid using stale data for decisions
                        peak_batch_size = 0;
                        recent_batch_times.clear();
                        recent_batch_sizes.clear();
                    }
                }
                cleanup_counter++;

                // Performance check to dynamically adjust batch parameters
                if (current_time - last_performance_check > performance_check_interval) {
                    if (!recent_batch_times.empty() && !recent_batch_sizes.empty()) {
                        // Calculate average batch processing time and size
                        float avg_time = 0.0f;
                        float avg_size = 0.0f;
                        
                        for (float t : recent_batch_times) avg_time += t;
                        for (size_t s : recent_batch_sizes) avg_size += static_cast<float>(s);
                        
                        avg_time /= recent_batch_times.size();
                        avg_size /= recent_batch_sizes.size();
                        
                        // See if we need to adjust min_batch_size
                        if (avg_size < batch_size_ * 0.5f && min_batch_size_ > 1) {
                            // Batches are usually small, reduce min_batch_size
                            min_batch_size_ = std::max(static_cast<size_t>(1), min_batch_size_ - 1);
                            std::cout << "[EVAL_THREAD] Reducing min_batch_size to " << min_batch_size_ 
                                      << " (avg batch: " << avg_size << ", avg time: " << avg_time << "ms)" << std::endl;
                        } else if (avg_size >= batch_size_ * 0.8f && avg_time < timeout_.count() * 0.3f) {
                            // Batches are filling well and processing is fast, increase min_batch_size
                            min_batch_size_ = std::min(batch_size_, min_batch_size_ + 1);
                            std::cout << "[EVAL_THREAD] Increasing min_batch_size to " << min_batch_size_ 
                                      << " (avg batch: " << avg_size << ", avg time: " << avg_time << "ms)" << std::endl;
                        }
                    }
                    
                    last_performance_check = current_time;
                }

                auto batch_collection_start_time = std::chrono::high_resolution_clock::now();
                std::vector<EvaluationRequest> batch = collectBatch(batch_size_); 
                auto batch_collection_end_time = std::chrono::high_resolution_clock::now();
                auto batch_collection_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_collection_end_time - batch_collection_start_time).count();

                if (batch.empty()) {
                    // For very frequent timeouts, minimize logging to reduce overhead
                    if (timeouts_.load(std::memory_order_relaxed) % 100 == 0) {
                        std::cout << "[EVAL_THREAD] Batch collection returned empty. Collection time: " 
                                << batch_collection_duration_us << " us. Total timeouts: " 
                                << timeouts_.load(std::memory_order_relaxed) << std::endl;
                    }
                    
                    // Sleep briefly to avoid busy-waiting
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    // More aggressive cleanup during extended idle periods
                    if (timeouts_.load(std::memory_order_relaxed) % 500 == 0 && torch::cuda::is_available()) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                            std::cout << "[EVAL_THREAD] Extended idle period detected, performing memory cleanup" << std::endl;
                        } catch (...) {
                            // Ignore cleanup errors
                        }
                    }
                    
                    continue;
                }

                bool should_log_detail = total_batches_.load(std::memory_order_relaxed) % 20 == 0;
                
                if (should_log_detail) {
                    std::cout << "[EVAL_THREAD] Collected batch of size: " << batch.size() 
                            << ". Collection time: " << batch_collection_duration_us << " us." << std::endl;
                }
                
                auto entire_batch_processing_start_time = std::chrono::high_resolution_clock::now();

                std::vector<std::unique_ptr<core::IGameState>> states_for_eval;
                std::vector<EvaluationRequest*> original_requests_in_batch; 
                states_for_eval.reserve(batch.size());
                original_requests_in_batch.reserve(batch.size());

                // Track the largest batch we've processed
                peak_batch_size = std::max(peak_batch_size, batch.size());

                for (EvaluationRequest& req_ref : batch) { 
                    if (req_ref.node && req_ref.state) { 
                        states_for_eval.push_back(std::move(req_ref.state)); 
                        original_requests_in_batch.push_back(&req_ref);
                    } else {
                        std::cerr << "[EVAL_THREAD] Null state or node in a request. Action_space_size: " 
                                << req_ref.action_space_size << ". Fulfilling with default." << std::endl;
                        try {
                            NetworkOutput default_output;
                            default_output.value = 0.0f; 
                            default_output.policy.resize(req_ref.action_space_size > 0 ? req_ref.action_space_size : 10, 
                                                        1.0f / (req_ref.action_space_size > 0 ? req_ref.action_space_size : 10.0f));
                            req_ref.promise.set_value(std::move(default_output));
                        } catch (const std::future_error& fe) {
                            std::cerr << "[EVAL_THREAD] Future error setting default for null state: " << fe.what() << std::endl;
                        }
                    }
                }
                
                if (states_for_eval.empty()) { // This check is after processing the batch for null states
                    if (!batch.empty()) { // Original batch was not empty, but all states were null
                        std::cout << "[EVAL_THREAD] All states in the collected batch (orig size " << batch.size() << ") were null. Skipping model evaluation." << std::endl;
                        
                        auto entire_batch_processing_end_time = std::chrono::high_resolution_clock::now();
                        auto entire_batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(entire_batch_processing_end_time - entire_batch_processing_start_time).count();
                        
                        std::cout << "[EVAL_THREAD] Finished processing (all null) batch of original size: " << batch.size() 
                                << ". Total batch processing time: " << entire_batch_duration_us << " us." << std::endl;
                    }
                    continue;
                }

                if (should_log_detail) {
                    std::cout << "[EVAL_THREAD] Calling inference_fn_ for effective batch size: " << states_for_eval.size() << std::endl;
                }
                
                auto model_eval_start_time = std::chrono::high_resolution_clock::now();
                std::vector<NetworkOutput> results;
                
                try {
                    if (!inference_fn_) {
                        std::cerr << "[EVAL_THREAD] CRITICAL ERROR: inference_fn_ is null! Fulfilling " 
                                << original_requests_in_batch.size() << " promises with error." << std::endl;
                        
                        for (EvaluationRequest* req_ptr : original_requests_in_batch) {
                            try {
                                NetworkOutput error_output; 
                                error_output.value = 0.0f;
                                error_output.policy.resize(req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10, 
                                                        1.0f / (req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10.0f));
                                req_ptr->promise.set_value(std::move(error_output));
                            } catch (const std::future_error& ) { /* ignore if promise already gone */ }
                        }
                        consecutive_errors++;
                        
                        if (consecutive_errors >= max_consecutive_errors) {
                            std::cerr << "[EVAL_THREAD] Too many consecutive errors, sleeping to avoid thrashing" << std::endl;
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            consecutive_errors = 0;
                        }
                        
                        continue; 
                    }
                    
                    // Clear CUDA caches for large batches before inference to reduce fragmentation
                    if (torch::cuda::is_available() && states_for_eval.size() >= batch_size_ * 0.9) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        } catch (...) {
                            // Ignore cleanup errors
                        }
                    }
                    
                    results = inference_fn_(states_for_eval);
                    consecutive_errors = 0; // Reset error counter on success
                    
                    // Update performance tracking
                    total_processed_states += states_for_eval.size();
                } catch (const std::exception& e) {
                    std::cerr << "[EVAL_THREAD] Exception during inference_fn_ call for batch size " << states_for_eval.size() << ": " << e.what() << std::endl;
                    
                    for (EvaluationRequest* req_ptr : original_requests_in_batch) {
                        try {
                            NetworkOutput error_output; error_output.value = 0.0f;
                            error_output.policy.resize(req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10, 
                                                        1.0f / (req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10.0f));
                            req_ptr->promise.set_value(std::move(error_output));
                        } catch (const std::future_error& fe) {
                            std::cerr << "[EVAL_THREAD] Future error (post-inference-exception): " << fe.what() << std::endl;
                        }
                    }
                    
                    consecutive_errors++;
                    
                    if (consecutive_errors >= max_consecutive_errors) {
                        std::cerr << "[EVAL_THREAD] Too many consecutive errors, sleeping to avoid thrashing" << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        consecutive_errors = 0;
                        
                        // Aggressive cleanup after multiple errors - free all memory
                        if (torch::cuda::is_available()) {
                            try {
                                torch::cuda::synchronize();
                                c10::cuda::CUDACachingAllocator::emptyCache();
                                std::cout << "[EVAL_THREAD] Emergency CUDA memory cleanup due to errors" << std::endl;
                                
                                // Print memory stats for debugging after emergency cleanup
                                for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                                    size_t free, total;
                                    cudaSetDevice(dev);
                                    cudaMemGetInfo(&free, &total);
                                    std::cout << "[EVAL_THREAD] GPU " << dev << " memory: " << (total - free) / 1048576 
                                            << "MB used, " << free / 1048576 << "MB free of " 
                                            << total / 1048576 << "MB total" << std::endl;
                                }
                            } catch (...) {
                                // Ignore cleanup errors
                            }
                        }
                    }
                    
                    continue; 
                }
                
                auto model_eval_end_time = std::chrono::high_resolution_clock::now();
                auto model_eval_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(model_eval_end_time - model_eval_start_time).count();
                
                if (should_log_detail) {
                    std::cout << "[EVAL_THREAD] inference_fn_ took: " << model_eval_duration_us << " us for effective batch size: " << states_for_eval.size() << std::endl;
                }

                if (results.size() != original_requests_in_batch.size()) {
                    std::cerr << "[EVAL_THREAD] Mismatch! Results size: " << results.size() 
                            << ", Expected (valid requests): " << original_requests_in_batch.size() 
                            << ". Fulfilling based on available results and defaults." << std::endl;
                    
                    // Handle case where result size might be less or more (though more is odd)
                    for (size_t i = 0; i < original_requests_in_batch.size(); ++i) {
                        try {
                            if (i < results.size()) { 
                                original_requests_in_batch[i]->promise.set_value(std::move(results[i]));
                            } else { // Not enough results, fulfill with default
                                NetworkOutput default_output; default_output.value = 0.0f;
                                default_output.policy.resize(original_requests_in_batch[i]->action_space_size > 0 ? original_requests_in_batch[i]->action_space_size : 10, 
                                                            1.0f / (original_requests_in_batch[i]->action_space_size > 0 ? original_requests_in_batch[i]->action_space_size : 10.0f));
                                original_requests_in_batch[i]->promise.set_value(std::move(default_output));
                            }
                        } catch (const std::future_error& fe) {
                            std::cerr << "[EVAL_THREAD] Future error (mismatch handling): " << fe.what() << std::endl;
                        }
                    }
                } else {
                    for (size_t i = 0; i < results.size(); ++i) {
                        try {
                            // Set the value and explicitly clear state pointers to help memory management
                            original_requests_in_batch[i]->promise.set_value(std::move(results[i]));
                        } catch (const std::future_error& e) {
                            std::cerr << "[EVAL_THREAD] Future error (setting value): " << e.what() << " for request " << i << std::endl;
                        }
                    }
                }
                
                // Make sure we remove any pointers to the moved states
                states_for_eval.clear();
                original_requests_in_batch.clear();
                results.clear();
                
                auto entire_batch_processing_end_time = std::chrono::high_resolution_clock::now();
                auto entire_batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(entire_batch_processing_end_time - entire_batch_processing_start_time).count();
                
                if (should_log_detail) {
                    std::cout << "[EVAL_THREAD] Finished processing batch. Original collected size: " << batch.size() 
                            << ". Total batch processing time (excl collection): " << entire_batch_duration_us << " us." << std::endl;
                }
                
                // Update metrics
                total_batches_.fetch_add(1, std::memory_order_relaxed);
                size_t batch_count = total_batches_.load(std::memory_order_relaxed);
                total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
                cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
                cumulative_batch_time_ms_.fetch_add(entire_batch_duration_us / 1000, std::memory_order_relaxed);
                
                // Keep track of recent performance for auto-tuning
                recent_batch_times.push_back(entire_batch_duration_us / 1000.0f); // ms
                recent_batch_sizes.push_back(batch.size());
                
                if (recent_batch_times.size() > max_history) {
                    recent_batch_times.pop_front();
                    recent_batch_sizes.pop_front();
                }
                
                // Periodic CUDA memory management - more frequent for large or frequent batches
                if (torch::cuda::is_available() && 
                    ((batch_count % 25 == 0) || (batch.size() >= batch_size_ * 0.9))) {
                    try {
                        torch::cuda::synchronize();
                        c10::cuda::CUDACachingAllocator::emptyCache();
                        
                        if (should_log_detail) {
                            std::cout << "[EVAL_THREAD] Post-batch CUDA cleanup, batch: " << batch_count 
                                      << ", size: " << batch.size() << std::endl;
                        }
                    } catch (...) {
                        // Ignore cleanup errors
                    }
                }
                
                // Explicitly empty the batch vector to release memory
                batch.clear();
                batch.shrink_to_fit();
            } catch (const std::exception& e) {
                std::cerr << "[EVAL_THREAD] Exception in batch processing loop: " << e.what() << std::endl;
                consecutive_errors++;
                
                if (consecutive_errors >= max_consecutive_errors) {
                    std::cerr << "[EVAL_THREAD] Too many consecutive exceptions, sleeping to avoid thrashing" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    consecutive_errors = 0;
                }
            } catch (...) {
                std::cerr << "[EVAL_THREAD] Unknown exception in batch processing loop" << std::endl;
                consecutive_errors++;
                
                if (consecutive_errors >= max_consecutive_errors) {
                    std::cerr << "[EVAL_THREAD] Too many consecutive unknown exceptions, sleeping to avoid thrashing" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    consecutive_errors = 0;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[EVAL_THREAD] Fatal exception in main worker loop: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[EVAL_THREAD] Unknown fatal exception in main worker loop" << std::endl;
    }
    
    // Process any remaining items in the queue before exiting
    int cleaned_up_count = 0;
    std::cout << "[EVAL_THREAD] Cleaning up remaining queue items..." << std::endl;
    
    try {
        // Use a limited size vector for each batch of cleanup to avoid excessive memory growth
        const size_t cleanup_batch_size = 256;
        std::vector<EvaluationRequest> cleanup_batch;
        cleanup_batch.reserve(cleanup_batch_size);
        
        while (true) {
            // Reset the vector but keep its capacity to avoid repeated allocations
            cleanup_batch.clear();
            
            // Try to get a batch of items from the queue
            size_t items_dequeued = request_queue_.try_dequeue_bulk(cleanup_batch.data(), cleanup_batch_size);
            
            if (items_dequeued == 0) {
                break; // Queue is empty
            }
            
            // Process all items in the batch
            for (size_t i = 0; i < items_dequeued; ++i) {
                try {
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    
                    int action_size = cleanup_batch[i].action_space_size > 0 ? cleanup_batch[i].action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                    
                    cleanup_batch[i].promise.set_value(std::move(default_output));
                    cleaned_up_count++;
                } catch (...) {
                    // Promise might be broken already, just continue
                }
            }
        }
        
        std::cout << "[EVAL_THREAD] Cleaned up " << cleaned_up_count << " pending requests during shutdown" << std::endl;
        
        // Final CUDA cleanup
        if (torch::cuda::is_available()) {
            try {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
                std::cout << "[EVAL_THREAD] Final CUDA memory cleanup completed" << std::endl;
                
                // Print final memory stats
                for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                    size_t free, total;
                    cudaSetDevice(dev);
                    cudaMemGetInfo(&free, &total);
                    std::cout << "[EVAL_THREAD] Final GPU " << dev << " memory: " << (total - free) / 1048576 
                            << "MB used, " << free / 1048576 << "MB free of " 
                            << total / 1048576 << "MB total" << std::endl;
                }
            } catch (...) {
                // Ignore cleanup errors
            }
        }
        
        // Print statistics for analysis
        std::cout << "[EVAL_THREAD] MCTSEvaluator statistics:" << std::endl;
        std::cout << "[EVAL_THREAD] - Total batches processed: " << total_batches_.load(std::memory_order_relaxed) << std::endl;
        std::cout << "[EVAL_THREAD] - Total states evaluated: " << total_evaluations_.load(std::memory_order_relaxed) << std::endl;
        std::cout << "[EVAL_THREAD] - Peak batch size: " << peak_batch_size << std::endl;
        std::cout << "[EVAL_THREAD] - Avg batch size: " << getAverageBatchSize() << std::endl;
        std::cout << "[EVAL_THREAD] - Avg batch latency: " << getAverageBatchLatency().count() << "ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[EVAL_THREAD] Error during final cleanup: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[EVAL_THREAD] Unknown error during final cleanup" << std::endl;
    }
    
    std::cout << "[EVAL_THREAD] MCTSEvaluator::processBatches worker thread finished." << std::endl;
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch(size_t target_batch_size) {
    // If target_batch_size is 0, use the default batch_size_
    if (target_batch_size == 0) {
        target_batch_size = batch_size_;
    }
    
    // Enforce minimum batch size for GPU efficiency, but only if the configured batch_size_ allows it.
    if (target_batch_size < 4 && batch_size_ >= 4 && target_batch_size > 0) { // Ensure target_batch_size > 0
        target_batch_size = std::min(batch_size_, static_cast<size_t>(4));
    } else if (target_batch_size == 0 && batch_size_ >=4) { // If target_batch_size was 0 initially
        target_batch_size = std::max(batch_size_, static_cast<size_t>(4));
    }
    
    // Reduce verbosity by only logging every Nth batch start (or when queue is growing)
    size_t current_queue_size = request_queue_.size_approx();
    bool should_log = (total_batches_.load(std::memory_order_relaxed) % 100 == 0) || 
                      (current_queue_size > last_queue_size_.load(std::memory_order_relaxed));
    
    last_queue_size_.store(current_queue_size, std::memory_order_relaxed);
    
    if (should_log) {
        std::cout << "[BATCH_COLLECT] Starting. Target batch size: " << target_batch_size 
                  << ", Configured timeout: " << timeout_.count() << "ms" 
                  << ", Queue size approx: " << current_queue_size << std::endl;
    }

    std::vector<EvaluationRequest> batch;
    batch.reserve(target_batch_size);
    
    auto start_time = std::chrono::steady_clock::now();
    auto deadline = start_time + timeout_;
    
    if (should_log) {
        std::cout << "[BATCH_COLLECT] Deadline for collection: " << std::chrono::duration_cast<std::chrono::milliseconds>(deadline - start_time).count() << "ms from now." << std::endl;
    }

    // --- First attempt to fill the batch using bulk dequeue ---
    size_t num_to_dequeue_first_pass = std::min(current_queue_size, target_batch_size);
    
    if (should_log) {
        std::cout << "[BATCH_COLLECT] First pass. Queue size: " << current_queue_size << ", Attempting to dequeue: " << num_to_dequeue_first_pass << std::endl;
    }

    if (num_to_dequeue_first_pass > 0) {
        std::vector<EvaluationRequest> temp_batch(num_to_dequeue_first_pass);
        size_t actual_dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), num_to_dequeue_first_pass);
        
        if (should_log) {
            std::cout << "[BATCH_COLLECT] First pass. Actually dequeued: " << actual_dequeued << std::endl;
        }
        
        for (size_t i = 0; i < actual_dequeued; ++i) {
            if (temp_batch[i].node && temp_batch[i].state) {
                batch.push_back(std::move(temp_batch[i]));
            } else {
                std::cerr << "[BATCH_COLLECT] Warning: Invalid request (null node/state) dequeued in first pass at index " << i << std::endl;
                try {
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    int action_size = temp_batch[i].action_space_size > 0 ? temp_batch[i].action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                    temp_batch[i].promise.set_value(std::move(default_output));
                } catch (...) { /* Ignore if promise is already broken */ }
            }
        }
        
        if (should_log) {
            std::cout << "[BATCH_COLLECT] First pass. Batch size after dequeue: " << batch.size() << std::endl;
        }
    }

    // If we have enough items to start processing but not a full batch,
    // wait a little longer to see if we can fill the batch more
    if (batch.size() >= min_batch_size_ && batch.size() < target_batch_size && 
        !shutdown_flag_.load(std::memory_order_acquire)) {
        
        // Start a short additional wait to see if we can get more items
        auto short_deadline = std::chrono::steady_clock::now() + additional_wait_time_;
        
        if (should_log) {
            std::cout << "[BATCH_COLLECT] Have minimum batch (" << batch.size() 
                    << "), waiting extra " << additional_wait_time_.count() 
                    << "ms for more items" << std::endl;
        }
        
        while (std::chrono::steady_clock::now() < short_deadline && 
               batch.size() < target_batch_size && 
               !shutdown_flag_.load(std::memory_order_acquire)) {
            
            // Check if we can get more items
            size_t current_size = request_queue_.size_approx();
            if (current_size == 0) {
                break; // No more items, stop waiting
            }
            
            // Try to get more items
            size_t remaining = target_batch_size - batch.size();
            size_t to_get = std::min(remaining, current_size);
            
            if (to_get > 0) {
                std::vector<EvaluationRequest> temp_batch(to_get);
                size_t got = request_queue_.try_dequeue_bulk(temp_batch.data(), to_get);
                
                for (size_t i = 0; i < got; ++i) {
                    if (temp_batch[i].node && temp_batch[i].state) {
                        batch.push_back(std::move(temp_batch[i]));
                    } else {
                        try {
                            NetworkOutput default_output;
                            default_output.value = 0.0f;
                            int action_size = temp_batch[i].action_space_size > 0 ? temp_batch[i].action_space_size : 10;
                            default_output.policy.resize(action_size, 1.0f / action_size);
                            temp_batch[i].promise.set_value(std::move(default_output));
                        } catch (...) { /* Ignore if promise is already broken */ }
                    }
                }
                
                // If we got a full batch, we're done
                if (batch.size() >= target_batch_size) {
                    break;
                }
            }
            
            // Small sleep to avoid spinning
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // If batch is full or shutdown, return immediately
    if (batch.size() >= target_batch_size) {
        // To reduce log spam in high-throughput scenarios
        if (full_batches_.load(std::memory_order_relaxed) % 100 == 0) {
            std::cout << "[BATCH_COLLECT] Returning full batch. Size: " << batch.size() 
                    << ", Total full batches: " << full_batches_.load(std::memory_order_relaxed) << std::endl;
        }
        full_batches_.fetch_add(1, std::memory_order_relaxed);
        return batch;
    }
    
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        if (should_log) {
            std::cout << "[BATCH_COLLECT] Shutdown detected. Batch size: " << batch.size() << std::endl;
        }
        if (!batch.empty()) partial_batches_.fetch_add(1, std::memory_order_relaxed);
        return batch;
    }
    
    // If we already have the minimum batch size, process it rather than waiting longer
    if (batch.size() >= min_batch_size_) {
        if (should_log) {
            std::cout << "[BATCH_COLLECT] Processing partial batch (>= min_batch_size). Size: " << batch.size() << std::endl;
        }
        partial_batches_.fetch_add(1, std::memory_order_relaxed);
        return batch;
    }

    // --- If not enough items, wait for more items up to the deadline ---
    auto current_time_before_wait = std::chrono::steady_clock::now();
    auto time_to_wait = deadline - current_time_before_wait;
    
    if (should_log) {
        std::cout << "[BATCH_COLLECT] Before wait. Batch size: " << batch.size() 
                << ", Queue size approx: " << request_queue_.size_approx() 
                << ", Time to wait: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_to_wait).count() << "ms" << std::endl;
    }

    // Adjust the wait condition to trigger on meeting minimum batch size rather than target batch size
    bool woken_by_condition = false;
    if (time_to_wait > std::chrono::milliseconds::zero()) {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        woken_by_condition = cv_.wait_for(lock, time_to_wait, [this, &batch, should_log]() {
            size_t queue_size = request_queue_.size_approx();
            bool has_items = queue_size > 0;
            bool min_batch_can_be_filled = (queue_size + batch.size()) >= min_batch_size_;
            bool is_shutting_down = shutdown_flag_.load(std::memory_order_acquire);
            
            if ((has_items || is_shutting_down) && should_log) {
                std::cout << "Evaluator thread woke up. Has items: " 
                          << has_items << ", Shutting down: " << is_shutting_down << std::endl;
            }
            
            return min_batch_can_be_filled || is_shutting_down;
        });
    }
    
    if (should_log) {
        std::cout << "[BATCH_COLLECT] After wait. Woken by condition: " << (woken_by_condition ? "Yes" : "No (timeout)") 
                << ", Batch size: " << batch.size() 
                << ", Queue size approx: " << request_queue_.size_approx() << std::endl;
    }
    
    // --- Second attempt to fill the batch after waiting ---
    if (!shutdown_flag_.load(std::memory_order_acquire) && batch.size() < target_batch_size) {
        size_t remaining_capacity = target_batch_size - batch.size();
        size_t current_queue_size_after_wait = request_queue_.size_approx();
        size_t num_to_dequeue_second_pass = std::min(remaining_capacity, current_queue_size_after_wait);
        
        if (should_log) {
            std::cout << "[BATCH_COLLECT] Second pass. Remaining capacity: " << remaining_capacity 
                    << ", Queue size: " << current_queue_size_after_wait 
                    << ", Attempting to dequeue: " << num_to_dequeue_second_pass << std::endl;
        }

        if (num_to_dequeue_second_pass > 0) {
            std::vector<EvaluationRequest> temp_batch(num_to_dequeue_second_pass);
            size_t actual_dequeued_second_pass = request_queue_.try_dequeue_bulk(temp_batch.data(), num_to_dequeue_second_pass);
            
            if (should_log) {
                std::cout << "[BATCH_COLLECT] Second pass. Actually dequeued: " << actual_dequeued_second_pass << std::endl;
            }

            for (size_t i = 0; i < actual_dequeued_second_pass; ++i) {
                 if (temp_batch[i].node && temp_batch[i].state) {
                    batch.push_back(std::move(temp_batch[i]));
                } else {
                    std::cerr << "[BATCH_COLLECT] Warning: Invalid request (null node/state) dequeued in second pass at index " << i << std::endl;
                    try {
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        int action_size = temp_batch[i].action_space_size > 0 ? temp_batch[i].action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        temp_batch[i].promise.set_value(std::move(default_output));
                    } catch (...) { /* Ignore if promise is already broken */ }
                }
            }
            
            if (should_log) {
                std::cout << "[BATCH_COLLECT] Second pass. Batch size after dequeue: " << batch.size() << std::endl;
            }
        }
    }
    
    // Update metrics and log reason for returning
    if (batch.size() >= target_batch_size) {
        // To reduce log spam in high-throughput scenarios
        if (full_batches_.load(std::memory_order_relaxed) % 100 == 0) {
            std::cout << "[BATCH_COLLECT] Returning full batch. Size: " << batch.size() 
                    << ", Total full batches: " << full_batches_.load(std::memory_order_relaxed) << std::endl;
        }
        full_batches_.fetch_add(1, std::memory_order_relaxed);
    } else if (!batch.empty()) {
        if (should_log) {
            std::cout << "[BATCH_COLLECT] Returning partial batch. Size: " << batch.size() 
                    << ". Shutdown: " << shutdown_flag_.load(std::memory_order_acquire) 
                    << ", Timeout: " << (std::chrono::steady_clock::now() >= deadline) << std::endl;
        }
        partial_batches_.fetch_add(1, std::memory_order_relaxed);
    } else {
        if (std::chrono::steady_clock::now() >= deadline) {
            // To reduce log spam for common timeouts
            if (timeouts_.load(std::memory_order_relaxed) % 100 == 0) {
                std::cout << "[BATCH_COLLECT] Returning empty batch due to TIMEOUT. Total timeouts: " 
                        << timeouts_.load(std::memory_order_relaxed) << std::endl;
            }
            timeouts_.fetch_add(1, std::memory_order_relaxed);
        } else if (shutdown_flag_.load(std::memory_order_acquire)){
            std::cout << "[BATCH_COLLECT] Returning empty batch due to SHUTDOWN." << std::endl;
        } else {
            std::cout << "[BATCH_COLLECT] Returning empty batch for UNKNOWN REASON (should not happen often)." << std::endl;
        }
    }
    
    return batch;
}

} // namespace mcts
} // namespace alphazero