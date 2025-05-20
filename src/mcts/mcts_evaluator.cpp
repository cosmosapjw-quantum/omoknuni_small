// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "mcts/adaptive_backoff.h"
#include "utils/debug_monitor.h"
#include "utils/memory_tracker.h"
#include "utils/debug_logger.h"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace alphazero::mcts; // Added this line

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
                           std::chrono::milliseconds timeout,
                           size_t num_inference_threads)
    : inference_fn_(std::move(inference_fn)),
      batch_size_(batch_size),
      original_batch_size_(batch_size),
      timeout_(timeout),
      shutdown_flag_(true), // Start as shutdown
      total_batches_(0),
      total_evaluations_(0),
      cumulative_batch_size_(0),
      cumulative_batch_time_ms_(0),
      timeouts_(0),
      full_batches_(0),
      partial_batches_(0),
      pipeline_buffer_(batch_size), // Initialize pipeline buffer with batch size
      num_inference_threads_(num_inference_threads) {
    
    // Validate parameters and set reasonable defaults
    if (batch_size_ < 1) {
        batch_size_ = 1;
    }
    
    if (timeout_ < std::chrono::milliseconds(1)) {
        timeout_ = std::chrono::milliseconds(1);
    }
    
    if (num_inference_threads_ < 1) {
        num_inference_threads_ = 1;
    }
    
    // Initialize the standardized batch parameters
    batch_params_.optimal_batch_size = batch_size_;
    batch_params_.minimum_viable_batch_size = std::max(size_t(batch_size_ * 0.75), size_t(64));
    batch_params_.minimum_fallback_batch_size = std::max(size_t(batch_size_ * 0.3), size_t(16));
    batch_params_.max_wait_time = timeout_;
    batch_params_.additional_wait_time = std::chrono::milliseconds(10);
    
    // Initialize the pipeline buffer with optimal batch size
    // Configure pipeline buffer target batch size instead of recreating
    pipeline_buffer_.setTargetBatchSize(batch_params_.optimal_batch_size);
    
    // Set min batch size to optimize GPU usage (reduced from 75% to 25% for faster batching)
    min_batch_size_ = std::max(size_t(batch_size_ * 0.25), size_t(16));  // Wait for 25% batch or min 16
    
    // Set optimal batch size for GPU efficiency
    optimal_batch_size_ = std::min(size_t(batch_size_ * 0.75), size_t(128));
    
    // Reduce wait time for more responsive batching
    additional_wait_time_ = std::chrono::milliseconds(5);
    
    // Create the batch accumulator with appropriate sizes
    batch_accumulator_ = std::make_unique<BatchAccumulator>(
        batch_params_.optimal_batch_size,
        batch_params_.minimum_viable_batch_size,
        batch_params_.max_wait_time
    );
    
    // CRITICAL FIX: Start the batch accumulator immediately in the constructor
    // This ensures it's ready to receive items as soon as the evaluator is created
    if (batch_accumulator_) {
        std::cout << "MCTSEvaluator::constructor - Starting batch accumulator with optimal_size=" 
                 << batch_params_.optimal_batch_size << ", min_viable=" 
                 << batch_params_.minimum_viable_batch_size << ", max_wait=" 
                 << batch_params_.max_wait_time.count() << "ms" << std::endl;
        batch_accumulator_->start();
        
        if (!batch_accumulator_->isRunning()) {
            std::cout << "ERROR: MCTSEvaluator::constructor - Failed to start batch accumulator" << std::endl;
            // Try again with more explicit error handling
            try {
                batch_accumulator_->stop(); // Make sure it's not in an inconsistent state
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                batch_accumulator_->start();
                std::cout << "MCTSEvaluator::constructor - Second attempt to start batch accumulator: " 
                         << (batch_accumulator_->isRunning() ? "SUCCESS" : "FAILED") << std::endl;
            } catch (const std::exception& e) {
                std::cout << "ERROR: MCTSEvaluator::constructor - Exception during batch accumulator restart: " 
                         << e.what() << std::endl;
            }
        } else {
            std::cout << "SUCCESS: MCTSEvaluator::constructor - BatchAccumulator started successfully" << std::endl;
        }
    } else {
        std::cout << "ERROR: MCTSEvaluator::constructor - Failed to create batch accumulator" << std::endl;
    }
}

MCTSEvaluator::~MCTSEvaluator() {
    // Ensure proper shutdown and cleanup
    try {
        // First, stop the worker thread
        stop();
        
        // Force immediate GPU memory cleanup
        if (torch::cuda::is_available()) {
            try {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            } catch (...) {}
        }
        
        // Clear any remaining requests to prevent memory leaks
        int cleared_count = 0;
        
        while (true) {
            EvaluationRequest req;
            if (!request_queue_.try_dequeue(req)) {
                break;
            }
            
            // Fulfill pending promise
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                int action_size = req.action_space_size > 0 ? 
                                  req.action_space_size : 10;
                default_output.policy.resize(action_size, 1.0f / action_size);
                req.promise.set_value(std::move(default_output));
                cleared_count++;
            } catch (...) {
                // Promise might already be fulfilled or broken
            }
        }
        
        if (cleared_count > 0) {
            // Cleared residual requests
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
        // Error in MCTSEvaluator destructor
    } catch (...) {
        // Unknown error in MCTSEvaluator destructor
    }
}

void MCTSEvaluator::start() {
    // Don't hold the lock while creating threads to avoid potential deadlocks
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        
        // Check if already started (shutdown_flag_ is false when running)
        if (!shutdown_flag_.load(std::memory_order_acquire)) {
            // Already started. Cannot start twice.
            return;
        }
        
        shutdown_flag_.store(false, std::memory_order_release);
    }
    
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
    
    // Also clear the inference and result queues
    BatchForInference dummy_inf;
    while (inference_queue_.try_dequeue(dummy_inf)) {}
    
    BatchInferenceResult dummy_res;
    while (result_queue_internal_.try_dequeue(dummy_res)) {}
    
    try {
        // Ensure the batch accumulator is running (already started in constructor)
        if (batch_accumulator_ && !batch_accumulator_->isRunning()) {
            batch_accumulator_->start();
            std::cout << "MCTSEvaluator::start - Started batch accumulator" << std::endl;
        }
        
        // Initialize pipeline buffer
        // pipeline_buffer_ is already initialized in the constructor
        
        // Start the batch collector thread first - handles legacy path
        batch_collector_thread_ = std::thread(&MCTSEvaluator::batchCollectorLoop, this);
        
        // Start the inference worker threads
        for (size_t i = 0; i < num_inference_threads_; ++i) {
            // Assign one thread for pipeline processing, the rest for regular inference
            if (i == 0) {
                inference_worker_threads_.emplace_back(&MCTSEvaluator::pipelineProcessorLoop, this);
            } else {
                inference_worker_threads_.emplace_back(&MCTSEvaluator::inferenceWorkerLoop, this);
            }
        }
        
        // Start the result distributor thread (for internal queue mode)
        if (!use_external_queues_) {
            result_distributor_thread_ = std::thread(&MCTSEvaluator::resultDistributorLoop, this);
        }
    } catch (const std::exception& e) {
        shutdown_flag_.store(true, std::memory_order_release);
        
        // Stop batch accumulator
        if (batch_accumulator_) {
            batch_accumulator_->stop();
        }
        
        // Attempt to clean up any threads that may have been started
        if (batch_collector_thread_.joinable()) {
            batch_collector_thread_.join();
        }
        for (auto& thread : inference_worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        if (result_distributor_thread_.joinable()) {
            result_distributor_thread_.join();
        }
        throw;
    }
}

void MCTSEvaluator::stop() {
    // Graceful shutdown procedure
    bool need_join = false;
    
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        
        if (shutdown_flag_.load(std::memory_order_acquire)) {
            return; // Already stopped or stopping
        }
        
        shutdown_flag_.store(true, std::memory_order_release);
        need_join = true;
    }
    
    // Stop the batch accumulator if it exists
    if (batch_accumulator_) {
        batch_accumulator_->stop();
    }
    
    // Notify all condition variables to wake up threads
    cv_.notify_all();
    batch_ready_cv_.notify_all();
    inference_cv_.notify_all();
    result_cv_.notify_all();
    
    if (need_join) {
        // Join the batch collector thread
        if (batch_collector_thread_.joinable()) {
            try {
                batch_collector_thread_.join();
            } catch (...) {
                // Error joining batch collector thread
            }
        }
        
        // Join the inference worker threads
        for (auto& thread : inference_worker_threads_) {
            if (thread.joinable()) {
                try {
                    thread.join();
                } catch (...) {
                    // Error joining inference worker thread
                }
            }
        }
        
        // Join the result distributor thread
        if (result_distributor_thread_.joinable()) {
            try {
                result_distributor_thread_.join();
            } catch (...) {
                // Error joining result distributor thread
            }
        }
        
        // Drain the internal queue to ensure no pending requests
        int cleared_requests = 0;
        while (true) {
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
        
        // Clear any items in the inference queue
        BatchForInference dummy_inf;
        while (inference_queue_.try_dequeue(dummy_inf)) {}
        
        // Clear any items in the result queue
        BatchInferenceResult dummy_res;
        while (result_queue_internal_.try_dequeue(dummy_res)) {}
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
            // Exception getting action space size
            // Keep default size if we can't access action space
        } catch (...) {
            // Unknown exception getting action space size
            // Keep default size if we can't access action space
        }
    } else {
        // Warning: Null state passed to evaluateState
    }
    
    if (!node) {
        // Warning: Null node passed to evaluateState
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
            // Warning: State validation failed
            std::promise<NetworkOutput> promise;
            NetworkOutput default_output;
            default_output.value = 0.0f;
            default_output.policy.resize(policy_size, 1.0f / policy_size);
            promise.set_value(std::move(default_output));
            return promise.get_future();
        }
    } catch (const std::exception& e) {
        // Exception validating state
        std::promise<NetworkOutput> promise;
        NetworkOutput default_output;
        default_output.value = 0.0f;
        default_output.policy.resize(policy_size, 1.0f / policy_size);
        promise.set_value(std::move(default_output));
        return promise.get_future();
    } catch (...) {
        // Unknown exception validating state
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
        // Failed to enqueue request (should never happen)
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

void MCTSEvaluator::evaluateStateAsync(std::shared_ptr<MCTSNode> node,
                                     std::unique_ptr<core::IGameState> state,
                                     std::shared_ptr<std::promise<NetworkOutput>> promise) {
    // Check for shutdown condition
    if (shutdown_flag_.load(std::memory_order_acquire)) {
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
            default_output.policy.resize(10, 0.1f);
        }
        
        promise->set_value(std::move(default_output));
        return;
    }
    
    // Create a safe policy size before potentially moving the state
    int policy_size = 10; // Default fallback
    if (state) {
        try {
            policy_size = state->getActionSpaceSize();
        } catch (...) {
            // Keep default size if we can't access action space
        }
    }
    
    // Create the request with external promise
    EvaluationRequest request(node, std::move(state), policy_size);
    request.promise = std::move(*promise); // Move the promise content
    
    // Enqueue the request
    bool enqueued = request_queue_.enqueue(std::move(request));
    
    if (!enqueued) {
        // Failed to enqueue request - provide default
        NetworkOutput error_output;
        error_output.value = 0.0f;
        error_output.policy.resize(policy_size, 1.0f / policy_size);
        promise->set_value(std::move(error_output));
        return;
    }
    
    // Notify worker thread that a new request is available
    cv_.notify_all();
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

void MCTSEvaluator::notifyLeafAvailable() {
    static std::atomic<long long> notify_count_evaluator(0);
    long long current_notify_count = notify_count_evaluator.fetch_add(1, std::memory_order_relaxed);
    bool is_external = use_external_queues_; // Direct access for non-atomic bool

    if (current_notify_count < 20 || current_notify_count % 100 == 0) { // Log frequently at start, then less often
        std::cout << "MCTSEVALUATOR (notifyLeafAvailable @" << static_cast<void*>(this) 
                  << "): Called (count: " << current_notify_count << ", external_mode: " << is_external 
                  << "). Notifying internal CV." << std::endl;
    }

    // For external queues, we should immediately notify all waiting threads
    // to ensure responsive processing
    if (use_external_queues_) {
        // Get current queue sizes for diagnostics
        size_t leaf_queue_size = 0;
        if (leaf_queue_ptr_) {
            auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
            leaf_queue_size = external_leaf_queue->size_approx();
        }
        
        // Always notify all waiting threads when using external queues
        cv_.notify_all();
        batch_ready_cv_.notify_all();
        inference_cv_.notify_all();
        result_cv_.notify_all();
        return;
    }
    
    // For internal queues, batch notifications for efficiency
    static std::atomic<int> notification_count{0};
    static std::atomic<std::chrono::steady_clock::time_point> last_notify_time{std::chrono::steady_clock::now()};
    
    notification_count.fetch_add(1, std::memory_order_relaxed);
    
    // Only notify if we have accumulated enough notifications or enough time has passed
    auto now = std::chrono::steady_clock::now();
    auto last_time = last_notify_time.load(std::memory_order_relaxed);
    auto time_since_last = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time).count();
    
    int count = notification_count.load(std::memory_order_relaxed);
    bool should_notify = false;
    
    // Get the configured batch size for better notification batching
    size_t target_batch = batch_size_ / 2;  // Notify when we have half a batch
    
    // Notify if we have enough pending notifications or if enough time has passed
    if (count >= static_cast<int>(target_batch) || time_since_last > 2000) { // Target batch size or 2ms
        // Try to reset the counter
        if (notification_count.compare_exchange_strong(count, 0, std::memory_order_relaxed)) {
            last_notify_time.store(now, std::memory_order_relaxed);
            should_notify = true;
        }
    }
    
    if (should_notify) {
        cv_.notify_all(); // Notify all waiting threads
    }
}

// NOTE: The evaluationLoop method has been moved to mcts_evaluator_concurrent.cpp
// This eliminates duplicate implementation and centralizes the concurrent processing logic

bool MCTSEvaluator::processBatch() {
    if (use_external_queues_) {
        // Handle external queue processing
        using NetworkOutput = mcts::NetworkOutput;
        using PendingEvaluation = alphazero::mcts::PendingEvaluation;
        
        if (!leaf_queue_ptr_ || !result_queue_ptr_) {
            static int null_queue_count = 0;
            if (null_queue_count < 10) {}
            return false;
        }
        
        auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
        auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
        
        // Collect a batch of pending evaluations - optimized for batching
        std::vector<PendingEvaluation> evaluations;
        evaluations.reserve(batch_size_);
        
        // Adaptive batch collection for GPU efficiency
        // Dynamically adjust batch size based on queue depth
        size_t queue_size = external_leaf_queue->size_approx();
        
        // Adaptive minimum batch size based on queue depth
        size_t MIN_BATCH;
        auto max_wait = std::chrono::milliseconds(10);  // Default wait time
        
        if (queue_size < 10) {
            // Very few items - process even single items to avoid stalling
            MIN_BATCH = 1;
            max_wait = std::chrono::milliseconds(2);  // Very short wait
        } else if (queue_size < 50) {
            // Moderate queue - balance between latency and throughput
            MIN_BATCH = std::min(size_t(2), batch_size_ / 32);  // Keep very low
            max_wait = std::chrono::milliseconds(5);
        } else if (queue_size < 200) {
            // Larger queue - prefer bigger batches for efficiency
            MIN_BATCH = std::min(size_t(4), batch_size_ / 16);
            max_wait = std::chrono::milliseconds(10);
        } else {
            // Very large queue - maximize GPU utilization  
            MIN_BATCH = std::min(size_t(8), batch_size_ / 8);
            max_wait = std::chrono::milliseconds(15);
        }
        
        // Removed periodic logging for performance
        
        const size_t OPTIMAL_BATCH = batch_size_;  // Target batch size
        
        // Phase 1: Bulk dequeue everything available
        // Pre-allocate vector with the desired size to avoid issues with data() on empty vector
        evaluations.resize(OPTIMAL_BATCH);
        size_t initial_dequeued = external_leaf_queue->try_dequeue_bulk(evaluations.data(), OPTIMAL_BATCH);
        evaluations.resize(initial_dequeued);  // Shrink to actual size dequeued
        // Dequeued initial batch
        
        // Phase 2: If below minimum, wait briefly for more
        if (evaluations.size() < MIN_BATCH) {
            auto deadline = std::chrono::steady_clock::now() + max_wait;
            // Wait briefly for more items if below minimum
            // std::cout << "BATCH: Batch size " << evaluations.size() << " < min " << MIN_BATCH << ", waiting up to " << max_wait.count() << "ms" << std::endl;
            
            while (std::chrono::steady_clock::now() < deadline && 
                   evaluations.size() < OPTIMAL_BATCH &&
                   !shutdown_flag_.load(std::memory_order_acquire)) {
                
                // Try bulk dequeue again
                size_t remaining = OPTIMAL_BATCH - evaluations.size();
                std::vector<PendingEvaluation> temp_batch(remaining);
                size_t dequeued = external_leaf_queue->try_dequeue_bulk(temp_batch.data(), remaining);
                
                if (dequeued > 0) {
                    // evaluations.insert(evaluations.end(), std::make_move_iterator(temp_batch.begin()), std::make_move_iterator(temp_batch.begin() + dequeued));
                    // size_t old_size = evaluations.size(); // Unused variable
                    for (size_t i = 0; i < dequeued; ++i) {
                        evaluations.push_back(std::move(temp_batch[i]));
                    }
                    // Added more items to batch
                } else {
                    // CRITICAL FIX: Break early if queue is empty and we have at least one item
                    // This prevents deadlock when engine stops producing
                    if (external_leaf_queue->size_approx() == 0 && evaluations.size() > 0) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            
            // Final batch ready for processing
        }
        
        // Only process if we have items
        if (evaluations.empty()) {
            // CRITICAL DEBUG: Log when we return false due to empty queue
            static int empty_count = 0;
            if (empty_count < 10) {
                // int evaluator_id = reinterpret_cast<uintptr_t>(this) & 0xFFFF; // Unused
                }
            return false;
        }
        
        // Report batch size for external queue processing
        static int external_batch_count = 0;
        if (external_batch_count % 100 == 0) {
            // [BATCH] Track external batch size and count
        }
        external_batch_count++;
        
        // Track memory periodically
        static int batch_count = 0;
        if (batch_count % 10 == 0) {
            // alphazero::utils::trackMemory("Evaluator batch #" + std::to_string(batch_count));
        }
        batch_count++;
        
        try {
            // Extract states for inference and track valid indices
            std::vector<std::unique_ptr<core::IGameState>> states;
            std::vector<size_t> valid_indices;
            states.reserve(evaluations.size());
            valid_indices.reserve(evaluations.size());
            
            // Use OpenMP for parallel state extraction when beneficial
            if (evaluations.size() > 64) {
                // For parallel processing, first identify valid states
                std::vector<bool> is_valid(evaluations.size(), false);
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < evaluations.size(); ++i) {
                    is_valid[i] = (evaluations[i].state != nullptr);
                }
                
                // Now collect valid states sequentially to maintain order
                for (size_t i = 0; i < evaluations.size(); ++i) {
                    if (is_valid[i]) {
                        // Convert shared_ptr to unique_ptr for the neural network
                        auto unique_clone = evaluations[i].state->clone();
                        if (unique_clone) {
                            states.push_back(std::move(unique_clone));
                            valid_indices.push_back(i);
                        } else {
                            // WARNING: Failed to clone state at index i, skipping
                        }
                    } else {
                        // WARNING: Null state in parallel evaluation at index i, skipping
                    }
                }
            } else {
                for (size_t i = 0; i < evaluations.size(); ++i) {
                    if (evaluations[i].state) {
                        // Convert shared_ptr to unique_ptr for the neural network
                        auto unique_clone = evaluations[i].state->clone();
                        if (unique_clone) {
                            states.push_back(std::move(unique_clone));
                            valid_indices.push_back(i);
                        } else {
                            // WARNING: Failed to clone state at index i, skipping
                        }
                    } else {
                        // WARNING: Null state in evaluation at index i, skipping
                    }
                }
            }
            
            // Check if we have any valid states to process
            if (states.empty()) {
                return false;
            }
            
            // Perform inference
            std::vector<NetworkOutput> results = inference_fn_(states);
            
            // Log successful inference with actual batch size
            static int total_inferences = 0;
            total_inferences++;
            if (total_inferences % 100 == 0) {
                // [BATCH] Track neural net inference count
            }
            
            // Pair results with original evaluations based on valid indices
            if (results.size() == states.size()) {
                // Use bulk enqueue for better performance when possible
                if (results.size() > 1) {
                    std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs;
                    result_pairs.reserve(results.size());
                    
                    for (size_t i = 0; i < results.size(); ++i) {
                        size_t original_index = valid_indices[i];
                        result_pairs.emplace_back(
                            std::move(results[i]), 
                            std::move(evaluations[original_index])
                        );
                    }
                    
                    external_result_queue->enqueue_bulk(
                        std::make_move_iterator(result_pairs.begin()),
                        result_pairs.size()
                    );
                } else {
                    // Single item, use regular enqueue
                    size_t original_index = valid_indices[0];
                    external_result_queue->enqueue(std::make_pair(
                        std::move(results[0]), 
                        std::move(evaluations[original_index])
                    ));
                }
                
                // Update statistics
                total_batches_.fetch_add(1, std::memory_order_relaxed);
                total_evaluations_.fetch_add(results.size(), std::memory_order_relaxed);
                cumulative_batch_size_.fetch_add(results.size(), std::memory_order_relaxed);
                
                
                // Notify the engine that results are available
                if (result_notify_callback_) {
                    result_notify_callback_();
                }
                
                return true;
            } else {
                // Mismatch in result count
                return false;
            }
        } catch (const std::exception& e) {
            // Error processing external batch
            return false;
        }
    }
    
    // Internal queue processing
    auto batch = collectInternalBatch(batch_size_); // Corrected this call
    if (batch.empty()) {
        return false;
    }
    
    processInternalBatch(batch);
    return true;
}

void MCTSEvaluator::processBatches() {
    // MCTSEvaluator::processBatches worker thread started
    
    // For MCTSEngine integration with batch queue - use proper types
    using NetworkOutput = mcts::NetworkOutput;
    using PendingEvaluation = alphazero::mcts::PendingEvaluation;
    
    // Variables for error handling and cleanup
    const int max_consecutive_errors = 5;
    int consecutive_errors = 0;
    
    // Periodic cleanup timer
    auto last_cleanup_time = std::chrono::steady_clock::now();
    const auto cleanup_interval = std::chrono::minutes(2); // More frequent cleanup
    
    // Memory stats tracking
    size_t peak_batch_size = 0;
    size_t cleanup_counter = 0;
    
    // Track if we've printed external queue state
    bool printed_queue_info = false;
    
    // Track performance to adapt batch sizes
    auto last_performance_check = std::chrono::steady_clock::now();
    const auto performance_check_interval = std::chrono::minutes(1);
    std::deque<float> recent_batch_times;
    std::deque<size_t> recent_batch_sizes;
    const size_t max_history = 20;

    auto batch_collection_start_time = std::chrono::high_resolution_clock::now(); // Define higher up

    try {
        while (!shutdown_flag_.load(std::memory_order_acquire)) {
            try {
                // Periodic cleanup to prevent memory leaks - more frequent than before
                auto current_time = std::chrono::steady_clock::now();
                bool is_cleanup_time = current_time - last_cleanup_time > cleanup_interval;
                bool is_idle = request_queue_.size_approx() == 0;
                bool is_external_idle = false;
                
                // Also check external queue status if using external queues
                if (use_external_queues_ && leaf_queue_ptr_) {
                    auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
                    is_external_idle = external_leaf_queue->size_approx() == 0;
                }
                
                // Clean up more aggressively during idle periods or periodically
                if (is_cleanup_time || (is_idle && cleanup_counter % 100 == 0) || (is_external_idle && cleanup_counter % 50 == 0)) {
                    cleanup_counter = 0;
                    
                    // Force GPU memory cleanup if using CUDA
                    if (torch::cuda::is_available()) {
                        try {
                            // First synchronize all CUDA streams to ensure operations are complete
                            torch::cuda::synchronize();
                            
                            // Clear CUDA cache to free memory
                            c10::cuda::CUDACachingAllocator::emptyCache();
                            
                            // CUDA memory cache emptied
                            
                            // Print memory stats for debugging during periodic cleanup
                            if (is_cleanup_time || is_external_idle) {
                                for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                                    size_t free, total;
                                    cudaSetDevice(dev);
                                    cudaMemGetInfo(&free, &total);
                                    size_t used_mb = (total - free) / 1048576;
                                    
                                    // MEMORY LEAK WARNING: if used memory exceeds threshold, force aggressive cleanup
                                    if (used_mb > 7500) { // 7.5GB threshold for 8GB GPUs
                                        // WARNING: GPU memory usage critical
                                        
                                        // Force synchronize all streams and empty cache
                                        for (int i = 0; i < 3; i++) {
                                            torch::cuda::synchronize();
                                            c10::cuda::CUDACachingAllocator::emptyCache();
                                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                                        }
                                    } else {
                                        // GPU memory status logged
                                    }
                                }
                            }
                        } catch (const std::exception& e) {
                            // Error during CUDA cleanup
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
                            // Reducing min_batch_size
                        } else if (avg_size >= batch_size_ * 0.8f && avg_time < timeout_.count() * 0.3f) {
                            // Batches are filling well and processing is fast, increase min_batch_size
                            min_batch_size_ = std::min(batch_size_, min_batch_size_ + 1);
                            // Increasing min_batch_size
                        }
                    }
                    
                    last_performance_check = current_time;
                }

                // Check external queue pointers dynamically each iteration
                moodycamel::ConcurrentQueue<PendingEvaluation>* external_leaf_queue = nullptr;
                moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* external_result_queue = nullptr;
                
                if (leaf_queue_ptr_ != nullptr) {
                    external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
                }
                if (result_queue_ptr_ != nullptr) {
                    external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
                }
                
                // Check if we're using external queues
                bool use_external_queue = use_external_queues_ && (leaf_queue_ptr_ != nullptr) && (result_queue_ptr_ != nullptr);
                
                // Print detailed debugging info first time and periodically
                static int external_queue_check_counter = 0;
                if ((!printed_queue_info || external_queue_check_counter++ % 1 == 0)) {
                    if (!printed_queue_info) {
                        printed_queue_info = true;
                    }
                }
                
                if (use_external_queue) {
                    // Debug print when using external queue
                    static int external_queue_usage_count = 0;
                    if (external_queue_usage_count++ < 10) {
                        // [EVALUATOR] Using external queue with queue pointer and leaf_queue_ptr_
                    }
                    
                    // Collect batch directly from leaf queue
                    std::vector<PendingEvaluation> evaluations;
                    evaluations.reserve(batch_size_);
                    // auto batch_start_collect = std::chrono::steady_clock::now(); // Unused-but-set
                    
                    // Aggressive batch collection - wait for larger batches
                    const auto start_time = std::chrono::steady_clock::now();
                    const auto max_wait_time = std::chrono::milliseconds(50); // Increased timeout for better batching
                    const size_t min_batch_for_processing = std::max<size_t>(1, batch_size_ / 8); // Lower minimum to prevent deadlock
                    
                    // Phase 1: Quick bulk collection
                    if (external_leaf_queue->size_approx() > 0) {
                        size_t queue_size = external_leaf_queue->size_approx();
                        size_t bulk_size = std::min(queue_size, batch_size_);
                        // [EVALUATOR] Starting bulk dequeue with queue size and bulk size
                        
                        std::vector<PendingEvaluation> bulk_items(bulk_size);
                        
                        // Debug: Check queue before dequeue
                        // [EVALUATOR] Before bulk dequeue: logging queue size and bulk size
                        
                        size_t dequeued = external_leaf_queue->try_dequeue_bulk(bulk_items.data(), bulk_size);
                        // [EVALUATOR] Bulk dequeue result: log number of items dequeued from queue
                        
                        // Debug output for bulk dequeue
                        int null_count = 0;
                        // [EVALUATOR] Checking bulk dequeued items
                        
                        for (size_t i = 0; i < dequeued; ++i) {
                            if (!bulk_items[i].state) {
                                null_count++;
                                // [EVALUATOR] Item has null state immediately after bulk dequeue
                            } else {
                                // [EVALUATOR] Item has valid state at specific memory address
                            }
                        }
                        
                        // Move items to evaluations vector
                        for (size_t i = 0; i < dequeued; ++i) {
                            void* state_ptr_before = bulk_items[i].state.get();
                            evaluations.push_back(std::move(bulk_items[i]));
                            if (state_ptr_before && !evaluations.back().state) {
                                // [EVALUATOR] ERROR: State became null after move to evaluations
                            }
                        }
                        
                        if (null_count > 0) {
                            // [EVALUATOR] Bulk dequeue: tracking number of null states
                        }
                    }
                    
                    // Phase 2: Wait for more items if batch is small
                    while (evaluations.size() < batch_size_ && 
                           std::chrono::steady_clock::now() - start_time < max_wait_time) {
                        PendingEvaluation eval;
                        if (external_leaf_queue->try_dequeue(eval)) {
                            if (!eval.state) {
                                // [EVALUATOR] Single dequeue: state is null
                            }
                            evaluations.push_back(std::move(eval));
                        } else if (evaluations.size() < min_batch_for_processing) {
                            // If we don't have minimum batch size, wait longer
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        } else {
                            // Have enough for a reasonable batch, can stop collecting
                            break;
                        }
                    }
                    
                    // Don't process very small batches unless we're shutting down
                    if (evaluations.size() < min_batch_for_processing && 
                        !shutdown_flag_.load(std::memory_order_acquire)) {
                        // Put items back for later processing to avoid tiny batches
                        for (auto& eval : evaluations) {
                            external_leaf_queue->enqueue(std::move(eval));
                        }
                        evaluations.clear();
                    }
                    
                    if (evaluations.empty()) {
                        // Wait for work using the evaluator's internal condition variable
                        std::unique_lock<std::mutex> lock(cv_mutex_);
                        /* auto wait_result = */ cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() { // Unused variable wait_result
                            return shutdown_flag_.load(std::memory_order_acquire);
                        });
                        lock.unlock(); // Release lock after waiting
                        if (shutdown_flag_.load(std::memory_order_acquire)) break; // Exit loop if shutdown
                        continue; // Re-try collecting after being woken up
                    }
                    
                    // Log batch size periodically
                    size_t batch_num = total_batches_.load(std::memory_order_relaxed);
                    bool should_log = batch_num % 10 == 0;
                    if (should_log) {
                        // alphazero::utils::trackMemory("Evaluator batch #" + std::to_string(batch_num));
                    }
                    
                    // Collected batch from leaf queue
                    
                    auto batch_start_time = std::chrono::high_resolution_clock::now();
                    
                    try {
                        // Extract states for inference and track valid indices
                        std::vector<std::unique_ptr<core::IGameState>> states;
                        std::vector<size_t> valid_indices;
                        states.reserve(evaluations.size());
                        valid_indices.reserve(evaluations.size());
                        
                        for (size_t i = 0; i < evaluations.size(); ++i) {
                            if (evaluations[i].state) {
                                // Convert shared_ptr to unique_ptr for the neural network
                                auto unique_clone = evaluations[i].state->clone();
                                if (unique_clone) {
                                    states.push_back(std::move(unique_clone));
                                    valid_indices.push_back(i);
                                } else {
                                    // Failed to clone state
                                }
                            } else {
                                // Null state in evaluation
                            }
                        }
                        
                        // Check if we have any valid states to process
                        if (states.empty()) {
                            // ERROR: No valid states to process in batch
                            continue;
                        }
                        
                        // Perform inference
                        std::vector<NetworkOutput> results = inference_fn_(states);
                        
                        // Pair results with original evaluations based on valid indices
                        if (results.size() == states.size()) {
                            for (size_t i = 0; i < results.size(); ++i) {
                                size_t original_index = valid_indices[i];
                                external_result_queue->enqueue(std::make_pair(
                                    std::move(results[i]), 
                                    std::move(evaluations[original_index])
                                ));
                            }
                            
                            // Notify result distributor that results are available
                            if (result_notify_callback_) {
                                result_notify_callback_();
                            }
                            // Also directly notify any waiting threads
                            cv_.notify_all();
                        } else {
                            // ERROR: Mismatch in result count between expected and actual
                        }
                        
                        // Track stats
                        total_batches_.fetch_add(1, std::memory_order_relaxed);
                        total_evaluations_.fetch_add(results.size(), std::memory_order_relaxed);
                        cumulative_batch_size_.fetch_add(results.size(), std::memory_order_relaxed);
                        
                        
                        auto batch_end_time = std::chrono::high_resolution_clock::now();
                        auto batch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            batch_end_time - batch_start_time).count();
                        cumulative_batch_time_ms_.fetch_add(batch_duration_ms, std::memory_order_relaxed);
                        
                        // External batch processed
                        
                        // Adaptive batch size based on latency
                        if (total_batches_.load(std::memory_order_relaxed) % 10 == 0) {  // Adjust every 10 batches
                            float avg_latency_ms = cumulative_batch_time_ms_.load(std::memory_order_relaxed) / 
                                                     std::max(1.0f, static_cast<float>(total_batches_.load(std::memory_order_relaxed)));
                            float avg_batch_s = getAverageBatchSize(); // Use existing method

                            // Target batch size for comparison is the current adaptive batch_size_
                            float target_b_size = static_cast<float>(batch_size_); 

                            if (avg_batch_s < target_b_size * 0.5f && avg_latency_ms < static_cast<float>(timeout_.count()) * 0.3f) {
                                // Batches are consistently small and fast, reduce timeout
                                timeout_ = std::max(std::chrono::milliseconds(5),  // Min timeout 5ms
                                                  timeout_ - std::chrono::milliseconds(5));
                                // Reducing timeout (small, fast batches)
                            } else if (avg_batch_s >= target_b_size * 0.9f) {
                                // Batches are filling well, slightly increase timeout for better filling
                                timeout_ += std::chrono::milliseconds(5);
                                // Increasing timeout (good batch fill)
                            }
                            // Cap the timeout to reasonable bounds
                            timeout_ = std::min(timeout_, std::chrono::milliseconds(100)); // Max timeout 100ms
                            timeout_ = std::max(timeout_, std::chrono::milliseconds(5));   // Ensure min timeout

                            if (avg_latency_ms < 10.0f && batch_size_ < original_batch_size_ * 2) {
                                // If very fast, increase batch size for better throughput
                                batch_size_ = std::min(batch_size_ + 16, original_batch_size_ * 2);
                                // Increasing batch_size
                            } else if (avg_latency_ms > 50.0f && batch_size_ > original_batch_size_ / 2) {
                                // If too slow, decrease batch size for lower latency
                                batch_size_ = std::max(batch_size_ - 16, original_batch_size_ / 2);
                                // Decreasing batch_size
                            }
                            
                            // Adjust min batch size proportionally
                            min_batch_size_ = std::max(static_cast<size_t>(32), batch_size_ / 2);
                        }
                    } catch (const std::exception& e) {
                        // Error processing external batch
                    }
                    
                    continue;
                }
                
                // Original internal queue processing
                batch_collection_start_time = std::chrono::high_resolution_clock::now(); // Re-assign for internal path
                std::vector<EvaluationRequest> batch = collectInternalBatch(batch_size_); 
                // auto batch_collection_end_time = std::chrono::high_resolution_clock::now(); // Unused-but-set
                // auto batch_collection_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_collection_end_time - batch_collection_start_time).count(); // Unused

                if (batch.empty()) {
                    // For very frequent timeouts, minimize logging to reduce overhead
                    // Batch collection returned empty
                    
                    // Sleep briefly to avoid busy-waiting
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    // More aggressive cleanup during extended idle periods
                    if (timeouts_.load(std::memory_order_relaxed) % 500 == 0 && torch::cuda::is_available()) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                            // Extended idle period detected, performing memory cleanup
                        } catch (...) {
                            // Ignore cleanup errors
                        }
                    }
                    
                    continue;
                }

                // bool should_log_detail = total_batches_.load(std::memory_order_relaxed) % 20 == 0; // Unused
                size_t batch_count_stat = total_batches_.load(std::memory_order_relaxed) + 1; // Renamed from batch_count to avoid conflict
                
                // Log internal batch size periodically  
                if (total_batches_.load(std::memory_order_relaxed) % 10 == 0) {
                    }
                
                // Processing batch
                
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
                            // Warning: Invalid request - null node
                        }
                        if (!req.state) {
                            // Warning: Invalid request - null state
                        }
                        
                        NetworkOutput default_output;
                        default_output.value = 0.0f;
                        int action_size = req.action_space_size > 0 ? req.action_space_size : 10;
                        default_output.policy.resize(action_size, 1.0f / action_size);
                        
                        try {
                            req.promise.set_value(std::move(default_output));
                        } catch (const std::exception& e) {
                            // Error setting promise value
                        } catch (...) {
                            // Unknown error setting promise value
                        }
                    }
                }
                
                // Check if we have any valid states to process
                if (states_for_eval.empty()) {
                    // All requests in batch were invalid, skipping evaluation
                    continue;
                }
                
                // Perform inference with proper error handling
                try {
                    // Perform batch inference
                    std::vector<NetworkOutput> results = inference_fn_(states_for_eval);
                    
                    // Record inference time
                    // auto inference_end_time = std::chrono::high_resolution_clock::now(); // Unused-but-set
                    // auto inference_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time).count(); // Unused
                    
                    // Inference completed
                    
                    // Verify the results count
                    if (results.size() != original_requests_in_batch.size()) {
                        // Mismatch in inference results
                        
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
                                // Error: Result index out of bounds
                                break;
                            }
                            
                            EvaluationRequest* req_ptr = original_requests_in_batch[i];
                            if (!req_ptr) {
                                // Error: Null request pointer
                                continue;
                            }
                            
                            try {
                                // If we have a node with NodeTracker, use that path for better batching
                                if (req_ptr->node && result_queue_ptr_) {
                                    // This path should enable better batching
                                    PendingEvaluation pending_eval;
                                    pending_eval.node = req_ptr->node;
                                    // Note: path is not available here, would need to be passed through
                                    
                                    auto* result_queue = static_cast<moodycamel::ConcurrentQueue<mcts::EvaluationResult>*>(result_queue_ptr_);
                                    mcts::EvaluationResult eval_result;
                                    eval_result.output = std::move(results[i]);
                                    // Use a unique batch id based on time
                                    static std::atomic<int> batch_counter{0};
                                    eval_result.batch_id = batch_counter.fetch_add(1);
                                    eval_result.request_id = i;  // Use index as request_id
                                    result_queue->enqueue(std::move(eval_result));
                                    
                                    // Notify result distributor
                                    if (result_notify_callback_) {
                                        result_notify_callback_();
                                    }
                                } else {
                                    // Fallback: direct promise setting (less efficient)
                                    req_ptr->promise.set_value(std::move(results[i]));
                                }
                                
                                total_evaluations_.fetch_add(1, std::memory_order_relaxed);
                            } catch (const std::exception& e) {
                                // Error setting promise result
                            } catch (...) {
                                // Unknown error setting promise result
                            }
                        }
                    }
                    
                } catch (const std::exception& e) {
                    // Exception during inference
                    
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
                    // Unknown exception during inference
                    
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
                long long entire_batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_end_time - batch_collection_start_time).count();
                
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
                    ((batch_count_stat % 25 == 0) || (batch.size() >= batch_size_ * 0.9))) {
                    try {
                        torch::cuda::synchronize();
                        c10::cuda::CUDACachingAllocator::emptyCache();
                        
                        // Post-batch CUDA cleanup performed
                    } catch (...) {
                        // Ignore cleanup errors
                    }
                }
                
                // Explicitly empty the batch vector to release memory
                batch.clear();
                batch.shrink_to_fit();
            } catch (const std::exception& e) {
                // Exception in batch processing loop
                consecutive_errors++;
                
                if (consecutive_errors >= max_consecutive_errors) {
                    // Too many consecutive exceptions, sleeping to avoid thrashing
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    consecutive_errors = 0;
                }
            } catch (...) {
                // Unknown exception in batch processing loop
                consecutive_errors++;
                
                if (consecutive_errors >= max_consecutive_errors) {
                    // Too many consecutive unknown exceptions, sleeping to avoid thrashing
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    consecutive_errors = 0;
                }
            }
        }
    } catch (const std::exception& e) {
        // Fatal exception in main worker loop
    } catch (...) {
        // Unknown fatal exception in main worker loop
    }
    
    // Process any remaining items in the queue before exiting
    int cleaned_up_count = 0;
    // Cleaning up remaining queue items...
    
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
        
        // Cleaned up pending requests during shutdown
        
        // Final CUDA cleanup
        if (torch::cuda::is_available()) {
            try {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
                // Final CUDA memory cleanup completed
                
                // Print final memory stats
                for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                    size_t free, total;
                    cudaSetDevice(dev);
                    cudaMemGetInfo(&free, &total);
                    // Final GPU memory status logged
                }
            } catch (...) {
                // Ignore cleanup errors
            }
        }
        
        // Print statistics for analysis
        // MCTSEvaluator statistics logged
    } catch (const std::exception& e) {
        // Error during final cleanup
    } catch (...) {
        // Unknown error during final cleanup
    }
    
    // MCTSEvaluator::processBatches worker thread finished
}

std::vector<EvaluationRequest> MCTSEvaluator::collectInternalBatch(size_t target_batch_size) {
    // If target_batch_size is 0, use the default batch_size_
    if (target_batch_size == 0) {
        target_batch_size = batch_size_;
    }
    
    std::vector<EvaluationRequest> batch;
    batch.reserve(target_batch_size);
    
    // Configure batch collection parameters for more aggressive batching
    const size_t OPTIMAL_BATCH_SIZE = std::max(size_t(64), target_batch_size / 2);  // Prefer larger batches
    const size_t MIN_ACCEPTABLE_BATCH = std::max(size_t(16), target_batch_size / 8); // Minimum batch size to accept
    const std::chrono::milliseconds MAX_WAIT_TIME(20); // Maximum time to wait for batch formation
    
    // Phase 1: Fast bulk collection - try to get as many as possible immediately
    size_t queue_size = request_queue_.size_approx();
    if (queue_size > 0) {
        // Attempt to dequeue up to target_batch_size items at once
        size_t to_dequeue = std::min(queue_size, target_batch_size);
        
        std::vector<EvaluationRequest> temp_batch(to_dequeue);
        size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), to_dequeue);
        temp_batch.resize(dequeued); // Resize to actual number dequeued
        
        // Process valid requests
        for (auto& req : temp_batch) {
            if (req.node && req.state) {
                batch.push_back(std::move(req));
            } else {
                // Set default promise for invalid requests
                try {
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    int action_size = req.action_space_size > 0 ? req.action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                    req.promise.set_value(std::move(default_output));
                } catch (...) {
                    // Promise might already be set
                }
            }
        }
    }
    
    // Phase 2: Wait for optimal batch size if below target
    if (batch.size() < OPTIMAL_BATCH_SIZE && !shutdown_flag_.load(std::memory_order_acquire)) {
        auto optimal_deadline = std::chrono::steady_clock::now() + MAX_WAIT_TIME;
        
        while (std::chrono::steady_clock::now() < optimal_deadline && 
               batch.size() < target_batch_size &&
               !shutdown_flag_.load(std::memory_order_acquire)) {
            
            // Check how many more items we can accept
            size_t remaining = target_batch_size - batch.size();
            if (remaining <= 0) break;
            
            // Try to dequeue in bulk for efficiency
            std::vector<EvaluationRequest> additional_batch(remaining);
            size_t additional_dequeued = request_queue_.try_dequeue_bulk(additional_batch.data(), remaining);
            
            if (additional_dequeued > 0) {
                // Process valid requests from this additional batch
                for (size_t i = 0; i < additional_dequeued; ++i) {
                    if (additional_batch[i].node && additional_batch[i].state) {
                        batch.push_back(std::move(additional_batch[i]));
                    } else {
                        // Set default promise for invalid requests
                        try {
                            NetworkOutput default_output;
                            default_output.value = 0.0f;
                            int action_size = additional_batch[i].action_space_size > 0 ? additional_batch[i].action_space_size : 10;
                            default_output.policy.resize(action_size, 1.0f / action_size);
                            additional_batch[i].promise.set_value(std::move(default_output));
                        } catch (...) {
                            // Promise might already be set
                        }
                    }
                }
            } else {
                // Wait briefly with condition variable instead of yielding
                std::unique_lock<std::mutex> lock(cv_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(2), [this] {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() > 0;
                });
            }
        }
    }
    
    // Process whatever we have if we reached the minimum acceptable batch size or timeout
    if (batch.size() < MIN_ACCEPTABLE_BATCH) {
        timeouts_.fetch_add(1, std::memory_order_relaxed);
        
        if (batch.empty()) {
            return std::vector<EvaluationRequest>();  // Return empty if no items at all
        }
    }
    
    // Return the collected batch
    return batch;
}

std::vector<PendingEvaluation> MCTSEvaluator::collectExternalBatch(size_t target_batch_size) {
    // If not using external queues or no queue is set, return empty
    if (!use_external_queues_ || !leaf_queue_ptr_ || !result_queue_ptr_) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            std::cout << "ERROR: MCTSEvaluator::collectExternalBatch - External queues not configured" << std::endl;
            utils::debug_logger().log("EVALUATOR: Cannot collect batch - external queues not configured");
        }
        return std::vector<PendingEvaluation>();
    }
    
    auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
    
    // Log queue size more frequently for better debugging
    static int call_count = 0;
    call_count++;
    
    // Print more frequently at startup
    bool should_log = (call_count <= 20 || call_count % 50 == 0);
    size_t current_queue_size = external_leaf_queue->size_approx();
    
    if (should_log) {
        std::cout << "MCTSEvaluator::collectExternalBatch - [Call #" << call_count 
                 << "] Queue size: " << current_queue_size 
                 << ", Target size: " << target_batch_size 
                 << ", Batch accumulator running: " << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no") 
                 << std::endl;
        
        utils::debug_logger().log("EVALUATOR: collectExternalBatch - Queue size: " + 
                                 std::to_string(current_queue_size) + ", Target size: " + std::to_string(target_batch_size));
    }
    
    // Check if batch accumulator is running - if not, start it
    if (batch_accumulator_ && !batch_accumulator_->isRunning()) {
        std::cout << "MCTSEvaluator::collectExternalBatch - Batch accumulator not running, restarting it" << std::endl;
        batch_accumulator_->start();
    }
    
    // Configure batch collection parameters for more aggressive batching
    const size_t OPTIMAL_BATCH_SIZE = batch_params_.optimal_batch_size;
    const size_t MIN_VIABLE_BATCH = batch_params_.minimum_viable_batch_size;
    const size_t MIN_FALLBACK_BATCH = batch_params_.minimum_fallback_batch_size;
    const std::chrono::milliseconds MAX_WAIT_TIME = batch_params_.max_wait_time;
    const std::chrono::milliseconds ADDITIONAL_WAIT = batch_params_.additional_wait_time;
    
    // Prepare result vector
    std::vector<PendingEvaluation> batch;
    batch.reserve(OPTIMAL_BATCH_SIZE);
    
    // Phase 1: Fast bulk collection - try to get as many as possible immediately
    size_t queue_size = external_leaf_queue->size_approx();
    
    if (queue_size > 0) {
        // Attempt to dequeue up to OPTIMAL_BATCH_SIZE items at once
        size_t to_dequeue = std::min(queue_size, OPTIMAL_BATCH_SIZE);
        
        // Create a vector with preallocated storage to prevent allocation
        // during dequeue_bulk operation
        batch.resize(to_dequeue);
        size_t bulk_dequeued = external_leaf_queue->try_dequeue_bulk(batch.data(), to_dequeue);
        batch.resize(bulk_dequeued); // Resize to actual count
        
        // Validate the batch - remove any items with null state
        if (!batch.empty()) {
            size_t valid_count = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                if (batch[i].state) {
                    if (i != valid_count) {
                        batch[valid_count] = std::move(batch[i]);
                    }
                    valid_count++;
                }
            }
            batch.resize(valid_count);
        }
    }
    
    // Phase 2: Wait for optimal batch size if below minimum viable batch
    if (batch.size() < MIN_VIABLE_BATCH && !shutdown_flag_.load(std::memory_order_acquire)) {
        auto optimal_deadline = std::chrono::steady_clock::now() + MAX_WAIT_TIME;
        
        while (std::chrono::steady_clock::now() < optimal_deadline && 
               batch.size() < OPTIMAL_BATCH_SIZE &&
               !shutdown_flag_.load(std::memory_order_acquire)) {
            
            // Check how many more items we can accept
            size_t remaining = OPTIMAL_BATCH_SIZE - batch.size();
            if (remaining <= 0) break;
            
            // Additional items vector - preallocated for optimal performance
            std::vector<PendingEvaluation> additional(remaining);
            size_t additional_dequeued = external_leaf_queue->try_dequeue_bulk(additional.data(), remaining);
            
            // Break early if we got nothing
            if (additional_dequeued == 0) {
                // Check if queue is truly empty and we have a viable batch
                if (external_leaf_queue->size_approx() == 0 && batch.size() >= MIN_FALLBACK_BATCH) {
                    break;
                }
                // Sleep briefly to avoid tight busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Process and add additional items
            for (size_t i = 0; i < additional_dequeued; ++i) {
                if (additional[i].state) {
                    batch.push_back(std::move(additional[i]));
                }
            }
            
            // Check if we've reached MIN_VIABLE_BATCH but not yet OPTIMAL_BATCH_SIZE
            // If so, wait a bit more for more items to arrive, but with shorter timeout
            if (batch.size() >= MIN_VIABLE_BATCH && batch.size() < OPTIMAL_BATCH_SIZE) {
                auto additional_deadline = std::chrono::steady_clock::now() + ADDITIONAL_WAIT;
                
                // Quick wait for more items to reach optimal batch size
                while (std::chrono::steady_clock::now() < additional_deadline && 
                       batch.size() < OPTIMAL_BATCH_SIZE &&
                       std::chrono::steady_clock::now() < optimal_deadline &&
                       !shutdown_flag_.load(std::memory_order_acquire)) {
                    
                    remaining = OPTIMAL_BATCH_SIZE - batch.size();
                    if (remaining <= 0) break;
                    
                    std::vector<PendingEvaluation> final_items(remaining);
                    size_t final_dequeued = external_leaf_queue->try_dequeue_bulk(final_items.data(), remaining);
                    
                    if (final_dequeued == 0) {
                        // Sleep very briefly
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                        continue;
                    }
                    
                    // Add valid items
                    for (size_t i = 0; i < final_dequeued; ++i) {
                        if (final_items[i].state) {
                            batch.push_back(std::move(final_items[i]));
                        }
                    }
                }
                
                // After additional wait, break the main loop regardless
                break;
            }
        }
    }
    
    // Handle shutdown case
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        // Don't return batches when shutting down
        batch.clear();
    }
    
    // Metrics for debugging
    if (!batch.empty()) {
        if (batch.size() >= OPTIMAL_BATCH_SIZE) {
            full_batches_.fetch_add(1, std::memory_order_relaxed);
        } else if (batch.size() >= MIN_VIABLE_BATCH) {
            // Good batch, but not full
        } else {
            // Small batch
            timeouts_.fetch_add(1, std::memory_order_relaxed);
        }
        
        if (batch.size() < MIN_FALLBACK_BATCH) {
            // Apply minimum batch size constraint to prevent very small batches
            // that are inefficient on GPU
            if (!batch.empty() && batch.size() < MIN_FALLBACK_BATCH) {
                // This is an extremely small batch, but we'll process it anyway to make progress
                partial_batches_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    
    return batch;
}

void MCTSEvaluator::waitWithBackoff(std::function<bool()> predicate, std::chrono::milliseconds max_wait_time) {
    // Use the adaptive backoff utility for efficient polling
    static AdaptiveBackoff backoff(10, 100, 5000);
    backoff.wait_for(predicate, max_wait_time);
}

void MCTSEvaluator::clearPendingBatches() {
    // Clear all pending batches in the evaluation pipeline
    
    // First, reset the pipeline buffer (if using pipeline parallelism)
    pipeline_buffer_.reset();
    
    // Clear the batch accumulator if it exists
    if (batch_accumulator_) {
        batch_accumulator_->reset();
    }
    
    // Clear the request queue (for internal queue mode)
    EvaluationRequest dummy_request;
    while (request_queue_.try_dequeue(dummy_request)) {
        // Just empty the queue
    }
    
    // Clear inference jobs queue
    BatchForInference dummy_job;
    while (inference_queue_.try_dequeue(dummy_job)) {
        // Just empty the queue
    }
    
    // Clear result queue
    BatchInferenceResult dummy_result;
    while (result_queue_internal_.try_dequeue(dummy_result)) {
        // Just empty the queue
    }
    
    // For external queues, we can't directly clear them
    // We would need the owner to clear them
    
    // Reset metrics
    pending_inference_batches_.store(0, std::memory_order_release);
    pending_result_batches_.store(0, std::memory_order_release);
    
    // Notify all threads to check queue state
    {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        cv_.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(inference_mutex_);
        inference_cv_.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(result_mutex_);
        result_cv_.notify_all();
    }
    
    // Log the action
    std::cout << "MCTSEvaluator::clearPendingBatches - Cleared all pending evaluations" << std::endl;
}

void MCTSEvaluator::processInternalBatch(std::vector<EvaluationRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // Track batches processed
    static int internal_batch_count = 0;
    internal_batch_count++;
    
    // Create a timer to measure batch processing time
    auto batch_start = std::chrono::steady_clock::now();
    
    // Extract states from requests and track valid indices - use more efficient approach
    std::vector<std::unique_ptr<core::IGameState>> states;
    std::vector<size_t> valid_indices;
    states.reserve(batch.size());
    valid_indices.reserve(batch.size());
    
    // Use OpenMP for parallel state extraction for larger batches
    if (batch.size() > 64) {
        // For parallel processing, first identify valid states
        std::vector<bool> is_valid(batch.size(), false);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < batch.size(); ++i) {
            is_valid[i] = (batch[i].state != nullptr);
        }
        
        // Now collect valid states sequentially to maintain order
        for (size_t i = 0; i < batch.size(); ++i) {
            if (is_valid[i]) {
                // Move the unique_ptr directly since we're the only owner
                states.push_back(std::move(batch[i].state));
                valid_indices.push_back(i);
            }
        }
    } else {
        // For smaller batches, simply process sequentially
        for (size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].state) {
                states.push_back(std::move(batch[i].state));
                valid_indices.push_back(i);
            }
        }
    }
    
    // Check if we have any valid states to process
    if (states.empty()) {
        // No valid states, return default outputs for all
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
        return;
    }
    
    // Perform inference with proper error handling
    try {
        // Call the neural network inference function with our batch
        auto results = inference_fn_(states);
        
        // Set results back to requests using valid indices
        if (results.size() == states.size()) {
            // Use bulk or parallelized result processing for larger batches
            if (results.size() > 32) {
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < results.size(); ++i) {
                    size_t original_index = valid_indices[i];
                    try {
                        // Use atomic operation for thread safety in the promise
                        NetworkOutput result_copy = results[i]; // Copy for thread safety
                        batch[original_index].promise.set_value(std::move(result_copy));
                    } catch (...) {
                        // Promise already fulfilled - ignore
                    }
                }
            } else {
                // For smaller batches, process sequentially
                for (size_t i = 0; i < results.size(); ++i) {
                    size_t original_index = valid_indices[i];
                    try {
                        batch[original_index].promise.set_value(std::move(results[i]));
                    } catch (...) {
                        // Promise already fulfilled
                    }
                }
            }
        } else {
            // Mismatch in result count, provide default outputs
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
        
        // Update statistics
        total_batches_.fetch_add(1, std::memory_order_relaxed);
        total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
        cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
        
        
        // Calculate processing time for this batch
        auto batch_end = std::chrono::steady_clock::now();
        auto batch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            batch_end - batch_start).count();
        cumulative_batch_time_ms_.fetch_add(batch_duration_ms, std::memory_order_relaxed);
        
    } catch (const std::exception& e) {
        // Error during batch inference - provide default outputs
        
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

// These methods are already implemented with equivalent functionality
// in inferenceWorkerLoop and resultDistributorLoop classes

} // namespace mcts
} // namespace alphazero