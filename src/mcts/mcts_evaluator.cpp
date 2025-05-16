// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "mcts/mcts_engine.h"
#include "utils/debug_monitor.h"
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
    // For GPU efficiency, we want at least 75% of the batch to be filled
    // This ensures we get better GPU throughput
    min_batch_size_ = std::max(static_cast<size_t>(16), (batch_size_ * 3) / 4);
    
    // Additional wait time should be significant to accumulate larger batches
    // Use full timeout to maximize batching opportunity
    additional_wait_time_ = timeout_;
    
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
        const size_t bulk_size = 100;
        std::vector<EvaluationRequest> bulk_requests(bulk_size);
        
        while (true) {
            size_t dequeued = request_queue_.try_dequeue_bulk(bulk_requests.data(), bulk_size);
            if (dequeued == 0) {
                break;
            }
            
            // Fulfill any pending promises
            for (size_t i = 0; i < dequeued; ++i) {
                try {
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    int action_size = bulk_requests[i].action_space_size > 0 ? 
                                      bulk_requests[i].action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                    bulk_requests[i].promise.set_value(std::move(default_output));
                    cleared_count++;
                } catch (...) {
                    // Promise might already be fulfilled or broken
                }
            }
        }
        
        if (cleared_count > 0) {
            std::cout << "MCTSEvaluator::~MCTSEvaluator - Cleared " << cleared_count << " residual requests" << std::endl;
        }
        
        // Release any resources and memory
        inference_fn_ = nullptr;
        
        // Final CUDA memory cleanup
        if (torch::cuda::is_available()) {
            try {
                c10::cuda::CUDACachingAllocator::emptyCache();
            } catch (...) {
                // Ignore cleanup errors
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in MCTSEvaluator destructor: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in MCTSEvaluator destructor" << std::endl;
    }
}

void MCTSEvaluator::start() {
    if (!shutdown_flag_.load(std::memory_order_acquire)) {
        std::cerr << "MCTSEvaluator::start - Already started. Cannot start twice." << std::endl;
        return;
    }
    
    shutdown_flag_.store(false, std::memory_order_release);
    
    // Clear any leftover items in the queue from previous runs
    int cleared_items = 0;
    
    while (true) {
        // Create a fresh request object for each iteration
        EvaluationRequest dummy(nullptr, nullptr, 10);
        
        // Attempt to dequeue
        if (!request_queue_.try_dequeue(dummy)) {
            break;
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
        worker_thread_ = std::thread(&MCTSEvaluator::evaluationLoop, this);
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

std::future<NetworkOutput> MCTSEvaluator::evaluateState(std::shared_ptr<MCTSNode> node, std::unique_ptr<core::IGameState> state) {
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

void MCTSEvaluator::evaluationLoop() {
    std::cout << "[EVALUATOR] Starting evaluation loop thread. use_external_queues_=" << use_external_queues_ << std::endl;
    
    const int min_batch_wait_ms = 50;   // Reduced wait to prevent deadlock
    const int max_wait_ms = 500;        // Reduced max wait to ensure progress
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        if (use_external_queues_) {
            // When using external queues, don't enforce minimum batch size
            processBatch();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for enough items to accumulate or timeout
        auto wait_pred = [this]() {
            return shutdown_flag_.load(std::memory_order_acquire) || 
                   request_queue_.size_approx() >= min_batch_size_;
        };
        
        batch_ready_cv_.wait_for(lock, std::chrono::milliseconds(max_wait_ms), wait_pred);
        lock.unlock();
        
        if (shutdown_flag_.load(std::memory_order_acquire)) {
            break;
        }
        
        // Force minimum wait time for batch accumulation
        std::this_thread::sleep_for(std::chrono::milliseconds(min_batch_wait_ms));
        
        if (!processBatch()) {
            // Only sleep if we truly have nothing to process
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::cout << "[EVALUATOR] Evaluation loop thread exiting" << std::endl;
}

bool MCTSEvaluator::processBatch() {
    if (use_external_queues_) {
        // Handle external queue processing
        using BatchInfo = MCTSEngine::BatchInfo;
        using NetworkOutput = mcts::NetworkOutput;
        using PendingEvaluation = MCTSEngine::PendingEvaluation;
        
        if (!batch_queue_ptr_ || !result_queue_ptr_) {
            return false;
        }
        
        auto* external_batch_queue = static_cast<moodycamel::ConcurrentQueue<BatchInfo>*>(batch_queue_ptr_);
        auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
        
        BatchInfo batch_info;
        if (!external_batch_queue->try_dequeue(batch_info) || batch_info.evaluations.empty()) {
            return false;
        }
        
        std::cout << "[EVALUATOR] Got batch from external queue. Size: " << batch_info.evaluations.size() << std::endl;
        
        try {
            // Extract states for inference
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.reserve(batch_info.evaluations.size());
            
            // Move evaluations to preserve them
            std::vector<PendingEvaluation> evaluations = std::move(batch_info.evaluations);
            
            for (auto& eval : evaluations) {
                states.push_back(std::move(eval.state));
            }
            
            // Perform inference
            std::vector<NetworkOutput> results = inference_fn_(states);
            
            // Pair results with original evaluations and enqueue to result queue
            if (results.size() == evaluations.size()) {
                for (size_t i = 0; i < results.size(); ++i) {
                    external_result_queue->enqueue(std::make_pair(
                        std::move(results[i]), 
                        std::move(evaluations[i])
                    ));
                }
                
                // Update statistics
                total_batches_.fetch_add(1);
                total_evaluations_.fetch_add(results.size());
                cumulative_batch_size_.fetch_add(results.size());
                
                return true;
            } else {
                std::cerr << "[EVALUATOR] Mismatch in result count: expected " 
                          << evaluations.size() << ", got " << results.size() << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "[EVALUATOR] Error processing external batch: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Internal queue processing
    auto batch = collectBatch(batch_size_);
    if (batch.empty()) {
        return false;
    }
    
    processBatch(batch);
    return true;
}

void MCTSEvaluator::processBatches() {
    std::cout << "[EVAL_THREAD] MCTSEvaluator::processBatches worker thread started." << std::endl;
    
    // For MCTSEngine integration with batch queue - use proper types
    using BatchInfo = MCTSEngine::BatchInfo;
    using NetworkOutput = mcts::NetworkOutput;
    using PendingEvaluation = MCTSEngine::PendingEvaluation;
    
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
    
    // Track if we've printed external queue state
    bool printed_queue_info = false;
    
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

                // Check external queue pointers dynamically each iteration
                moodycamel::ConcurrentQueue<BatchInfo>* external_batch_queue = nullptr;
                moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* external_result_queue = nullptr;
                
                if (batch_queue_ptr_ != nullptr) {
                    external_batch_queue = static_cast<moodycamel::ConcurrentQueue<BatchInfo>*>(batch_queue_ptr_);
                }
                if (result_queue_ptr_ != nullptr) {
                    external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
                }
                
                // Check if we're using external queues
                bool use_external_queue = (use_external_queues_ && external_batch_queue && external_result_queue);
                
                // Print detailed debugging info first time and periodically
                static int external_queue_check_counter = 0;
                if ((!printed_queue_info || external_queue_check_counter++ % 1000 == 0) && use_external_queue) {
                    std::cout << "[EVAL_THREAD] External queue status (check #" << external_queue_check_counter << "):" << std::endl;
                    std::cout << "  use_external_queues_=" << use_external_queues_ << std::endl;
                    std::cout << "  batch_queue_ptr_=" << batch_queue_ptr_ << std::endl;
                    std::cout << "  result_queue_ptr_=" << result_queue_ptr_ << std::endl;
                    std::cout << "  external_batch_queue=" << external_batch_queue << std::endl;
                    std::cout << "  external_result_queue=" << external_result_queue << std::endl;
                    std::cout << "  use_external_queue=" << use_external_queue << std::endl;
                    if (external_batch_queue) {
                        std::cout << "  external_batch_queue size_approx=" << external_batch_queue->size_approx() << std::endl;
                    }
                    if (!printed_queue_info) {
                        printed_queue_info = true;
                    }
                }
                
                if (use_external_queue) {
                    // Use external queue for batch processing
                    BatchInfo batch_info;
                    bool got_batch = external_batch_queue->try_dequeue(batch_info);
                    
                    if (!got_batch || batch_info.evaluations.empty()) {
                        // Wait for work with timeout
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }
                    
                    std::cout << "[EVAL_THREAD] Got batch from external queue. Size: " 
                              << batch_info.evaluations.size() << std::endl;
                    
                    auto batch_start_time = std::chrono::high_resolution_clock::now();
                    
                    try {
                        // First, move all evaluations to preserve them
                        std::vector<PendingEvaluation> evaluations;
                        evaluations.reserve(batch_info.evaluations.size());
                        for (auto& eval : batch_info.evaluations) {
                            evaluations.push_back(std::move(eval));
                        }
                        
                        // Extract states for inference
                        std::vector<std::unique_ptr<core::IGameState>> states;
                        states.reserve(evaluations.size());
                        for (auto& eval : evaluations) {
                            states.push_back(std::move(eval.state));
                        }
                        
                        // Perform inference
                        std::vector<NetworkOutput> results = inference_fn_(states);
                        
                        // Pair results with original evaluations and enqueue to result queue
                        if (results.size() == evaluations.size()) {
                            for (size_t i = 0; i < results.size(); ++i) {
                                external_result_queue->enqueue(std::make_pair(
                                    std::move(results[i]), 
                                    std::move(evaluations[i])
                                ));
                            }
                        } else {
                            std::cerr << "[EVAL_THREAD] Mismatch in result count: expected " 
                                      << evaluations.size() << ", got " << results.size() << std::endl;
                        }
                        
                        // Track stats
                        total_batches_.fetch_add(1);
                        total_evaluations_.fetch_add(results.size());
                        cumulative_batch_size_.fetch_add(results.size());
                        
                        auto batch_end_time = std::chrono::high_resolution_clock::now();
                        auto batch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            batch_end_time - batch_start_time).count();
                        cumulative_batch_time_ms_.fetch_add(batch_duration_ms);
                        
                        std::cout << "[EVAL_THREAD] External batch processed. Size: " << results.size() 
                                  << ", Time: " << batch_duration_ms << "ms" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[EVAL_THREAD] Error processing external batch: " << e.what() << std::endl;
                    }
                    
                    continue;
                }
                
                // Original internal queue processing
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
                size_t batch_count = total_batches_.load(std::memory_order_relaxed) + 1;
                
                if (should_log_detail) {
                    std::cout << "[EVAL_THREAD] Processing batch #" << batch_count 
                              << ". Size: " << batch.size() 
                              << ", Collection time: " << batch_collection_duration_us << "us" << std::endl;
                }
                
                peak_batch_size = std::max(peak_batch_size, batch.size());
                
                // Reset consecutive errors on successful batch
                consecutive_errors = 0;
                
                // Track states
                std::vector<std::unique_ptr<core::IGameState>> states_for_eval;
                std::vector<EvaluationRequest*> original_requests_in_batch; // Pointers to keep references
                
                states_for_eval.reserve(batch.size());
                original_requests_in_batch.reserve(batch.size());
                
                // Prepare states for batch evaluation
                for (size_t idx = 0; idx < batch.size(); ++idx) {
                    EvaluationRequest& req = batch[idx];
                    if (req.node && req.state) {
                        // Store a pointer to the original request
                        original_requests_in_batch.push_back(&req);
                        
                        // Move the state into the vector
                        states_for_eval.push_back(std::move(req.state));
                    } else {
                        // Handle invalid request
                        if (!req.node) {
                            std::cerr << "[EVAL_THREAD] Warning: Invalid request - null node" << std::endl;
                        }
                        if (!req.state) {
                            std::cerr << "[EVAL_THREAD] Warning: Invalid request - null state" << std::endl;
                        }
                        
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        int action_size = req.action_space_size > 0 ? req.action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        
                        try {
                            req.promise.set_value(std::move(default_output));
                        } catch (const std::exception& e) {
                            std::cerr << "[EVAL_THREAD] Error setting promise value: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "[EVAL_THREAD] Unknown error setting promise value" << std::endl;
                        }
                    }
                }
                
                // Check if we have any valid states to process
                if (states_for_eval.empty()) {
                    std::cerr << "[EVAL_THREAD] All requests in batch were invalid, skipping evaluation" << std::endl;
                    continue;
                }
                
                auto inference_start_time = std::chrono::high_resolution_clock::now();
                
                try {
                    // Perform batch inference
                    std::vector<NetworkOutput> results = inference_fn_(states_for_eval);
                    
                    // Record inference time
                    auto inference_end_time = std::chrono::high_resolution_clock::now();
                    auto inference_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time).count();
                    
                    if (should_log_detail) {
                        std::cout << "[EVAL_THREAD] Inference completed. Time: " << inference_duration_us << "us"
                                  << " for batch size " << states_for_eval.size() << std::endl;
                    }
                    
                    // Verify the results count
                    if (results.size() != original_requests_in_batch.size()) {
                        std::cerr << "[EVAL_THREAD] Mismatch in inference results! Expected: " 
                                  << original_requests_in_batch.size() << " Got: " << results.size() << std::endl;
                        
                        // Fulfill as many promises as we can
                        size_t min_size = std::min(results.size(), original_requests_in_batch.size());
                        for (size_t i = 0; i < min_size; ++i) {
                            try {
                                original_requests_in_batch[i]->promise.set_value(std::move(results[i]));
                            } catch (...) {
                                // Promise already fulfilled or broken
                            }
                        }
                        
                        // Provide default outputs for remaining requests
                        for (size_t i = min_size; i < original_requests_in_batch.size(); ++i) {
                            NetworkOutput default_output;
                            default_output.value = 0.0f;
                            int action_size = original_requests_in_batch[i]->action_space_size > 0 ? 
                                              original_requests_in_batch[i]->action_space_size : 10;
                            default_output.policy.resize(action_size, 1.0f / action_size);
                            
                            try {
                                original_requests_in_batch[i]->promise.set_value(std::move(default_output));
                            } catch (...) {
                                // Promise already fulfilled or broken
                            }
                        }
                    } else {
                        // Normal case - send all results
                        for (size_t i = 0; i < results.size(); ++i) {
                            if (i >= original_requests_in_batch.size()) {
                                std::cerr << "[EVAL_THREAD] Error: Result index " << i 
                                          << " out of bounds for request array size " 
                                          << original_requests_in_batch.size() << std::endl;
                                break;
                            }
                            
                            EvaluationRequest* req_ptr = original_requests_in_batch[i];
                            if (!req_ptr) {
                                std::cerr << "[EVAL_THREAD] Error: Null request pointer at index " << i << std::endl;
                                continue;
                            }
                            
                            try {
                                req_ptr->promise.set_value(std::move(results[i]));
                                total_evaluations_.fetch_add(1, std::memory_order_relaxed);
                            } catch (const std::exception& e) {
                                std::cerr << "[EVAL_THREAD] Error setting promise result: " << e.what() << std::endl;
                            } catch (...) {
                                std::cerr << "[EVAL_THREAD] Unknown error setting promise result" << std::endl;
                            }
                        }
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "[EVAL_THREAD] Exception during inference: " << e.what() << std::endl;
                    
                    // Cleanup CUDA memory on error
                    if (torch::cuda::is_available()) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        } catch (...) {}
                    }
                    
                    // Provide default outputs on error
                    for (EvaluationRequest* req_ptr : original_requests_in_batch) {
                        if (!req_ptr) continue;
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        int action_size = req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        
                        try {
                            req_ptr->promise.set_value(std::move(default_output));
                        } catch (...) {
                            // Promise already fulfilled or broken
                        }
                    }
                } catch (...) {
                    std::cerr << "[EVAL_THREAD] Unknown exception during inference" << std::endl;
                    
                    // Cleanup CUDA memory on error
                    if (torch::cuda::is_available()) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        } catch (...) {}
                    }
                    
                    // Provide default outputs on error
                    for (EvaluationRequest* req_ptr : original_requests_in_batch) {
                        if (!req_ptr) continue;
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        int action_size = req_ptr->action_space_size > 0 ? req_ptr->action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        
                        try {
                            req_ptr->promise.set_value(std::move(default_output));
                        } catch (...) {
                            // Promise already fulfilled or broken
                        }
                    }
                }
                
                // Update statistics
                total_batches_.fetch_add(1, std::memory_order_relaxed);
                cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
                
                // Calculate total batch processing time
                auto batch_end_time = std::chrono::high_resolution_clock::now();
                auto entire_batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_end_time - batch_collection_start_time).count();
                
                // Track for adaptive batch sizing (in milliseconds)
                cumulative_batch_time_ms_.fetch_add(entire_batch_duration_us / 1000, std::memory_order_relaxed);
                
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
    
    std::vector<EvaluationRequest> batch;
    batch.reserve(target_batch_size);
    
    auto start_time = std::chrono::steady_clock::now();
    auto deadline = start_time + timeout_;
    
    // Aggressive batch collection - always wait for minimum batch size
    const size_t ABSOLUTE_MIN_BATCH = 16;  // Reduced to prevent deadlock with small eval counts
    
    // First, collect everything available
    size_t queue_size = request_queue_.size_approx();
    if (queue_size > 0) {
        size_t to_dequeue = std::min(queue_size, target_batch_size);
        std::vector<EvaluationRequest> temp_batch(to_dequeue);
        size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), to_dequeue);
        
        for (size_t i = 0; i < dequeued; ++i) {
            if (temp_batch[i].node && temp_batch[i].state) {
                batch.push_back(std::move(temp_batch[i]));
            }
        }
    }
    
    // If we haven't reached minimum, wait aggressively
    if (batch.size() < ABSOLUTE_MIN_BATCH && !shutdown_flag_.load(std::memory_order_acquire)) {
        // Wait the full timeout period to accumulate more items
        auto extended_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);  // Longer wait for better batching
        
        while (std::chrono::steady_clock::now() < extended_deadline && 
               batch.size() < target_batch_size &&
               !shutdown_flag_.load(std::memory_order_acquire)) {
            
            size_t remaining = target_batch_size - batch.size();
            std::vector<EvaluationRequest> temp_batch(remaining);
            size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), remaining);
            
            for (size_t i = 0; i < dequeued; ++i) {
                if (temp_batch[i].node && temp_batch[i].state) {
                    batch.push_back(std::move(temp_batch[i]));
                }
            }
            
            if (dequeued == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    // If still below absolute minimum, don't process
    if (batch.size() < ABSOLUTE_MIN_BATCH && !shutdown_flag_.load(std::memory_order_acquire)) {
        // Put items back in queue for later processing
        for (auto& item : batch) {
            request_queue_.enqueue(std::move(item));
        }
        
        timeouts_.fetch_add(1, std::memory_order_relaxed);
        return std::vector<EvaluationRequest>();  // Return empty
    }
    
    return batch;
}

void MCTSEvaluator::processBatch(std::vector<EvaluationRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // Extract states from requests
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    for (auto& req : batch) {
        if (req.state) {
            states.push_back(std::move(req.state));
        }
    }
    
    // Perform inference
    try {
        auto results = inference_fn_(states);
        
        // Set results back to requests
        for (size_t i = 0; i < std::min(results.size(), batch.size()); ++i) {
            try {
                batch[i].promise.set_value(std::move(results[i]));
            } catch (...) {
                // Promise might already be fulfilled
            }
        }
        
        // Update statistics
        total_batches_.fetch_add(1);
        total_evaluations_.fetch_add(batch.size());
        cumulative_batch_size_.fetch_add(batch.size());
        
    } catch (const std::exception& e) {
        std::cerr << "[EVALUATOR] Error during batch inference: " << e.what() << std::endl;
        
        // Provide default outputs on error
        for (auto& req : batch) {
            NetworkOutput default_output;
            default_output.value = 0.0f;
            int action_size = req.action_space_size > 0 ? req.action_space_size : 10;
            default_output.policy.resize(action_size, 1.0f / action_size);
            
            try {
                req.promise.set_value(std::move(default_output));
            } catch (...) {
                // Promise already fulfilled
            }
        }
    }
}

} // namespace mcts
} // namespace alphazero