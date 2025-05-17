// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "mcts/mcts_engine.h"
#include "utils/debug_monitor.h"
#include "utils/memory_tracker.h"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <iostream>
#include <omp.h>

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
      original_batch_size_(batch_size),
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
    
    // Set min batch size for immediate processing
    min_batch_size_ = 1;
    
    // Reduce wait time for faster response
    additional_wait_time_ = std::chrono::milliseconds(5);
    
    std::cout << "[EVALUATOR] Created with batch_size=" << batch_size_ 
              << ", timeout=" << timeout_.count() << "ms" << std::endl;
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
    if (!shutdown_flag_.load(std::memory_order_acquire)) {
        // Already started. Cannot start twice.
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
       // Cleared leftover items from queue before start
    }
    
    try {
        worker_thread_ = std::thread(&MCTSEvaluator::evaluationLoop, this);
    } catch (const std::exception& e) {
        shutdown_flag_.store(true, std::memory_order_release);
        // Failed to start MCTSEvaluator thread
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
                // Warning: Timeout waiting for MCTSEvaluator thread to join
                
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
                        // ERROR: Timeout waiting for MCTSEvaluator thread to join
                        // WARNING: MCTSEvaluator thread may be permanently blocked
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
    // Batch notifications to avoid excessive thread wake-ups
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
        std::lock_guard<std::mutex> lock(cv_mutex_);
        cv_.notify_one();
    }
}

void MCTSEvaluator::evaluationLoop() {
    // Only log once at startup if using external queues
    if (use_external_queues_) {
        std::cerr << "[EVALUATOR] Started with external queues enabled" << std::endl;
    }
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        if (use_external_queues_) {
            // When using external queues, don't wait long - check queue immediately
            if (!leaf_queue_ptr_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<MCTSEngine::PendingEvaluation>*>(leaf_queue_ptr_);
            
            // Check if we have a reasonable number of items to process
            size_t queue_size = external_leaf_queue->size_approx();
            
            // Only process if we have a good batch or have waited long enough
            if (queue_size >= batch_size_ / 4) {
                processBatch();
            } else if (queue_size > 0) {
                // Wait a bit for more items to accumulate
                std::unique_lock<std::mutex> lock(cv_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(5), [this, external_leaf_queue]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           external_leaf_queue->size_approx() >= batch_size_ / 4;
                });
                
                // Process whatever we have after waiting
                if (external_leaf_queue->size_approx() > 0) {
                    processBatch();
                }
            } else {
                // Only wait briefly if no items at all
                std::unique_lock<std::mutex> lock(cv_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(2), [this, external_leaf_queue]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           external_leaf_queue->size_approx() > 0;
                });
            }
            continue;
        }
        
        // Internal queue processing
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for items or timeout
        auto wait_pred = [this]() {
            return shutdown_flag_.load(std::memory_order_acquire) || 
                   request_queue_.size_approx() >= 1; // Process any available items immediately
        };
        
        batch_ready_cv_.wait_for(lock, std::chrono::milliseconds(1), wait_pred);
        lock.unlock();
        
        if (shutdown_flag_.load(std::memory_order_acquire)) {
            break;
        }
        
        if (!processBatch()) {
            // Only sleep if we truly have nothing to process
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    // Evaluation loop thread exiting
}

bool MCTSEvaluator::processBatch() {
    if (use_external_queues_) {
        // Handle external queue processing
        using NetworkOutput = mcts::NetworkOutput;
        using PendingEvaluation = MCTSEngine::PendingEvaluation;
        
        if (!leaf_queue_ptr_ || !result_queue_ptr_) {
            return false;
        }
        
        auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
        auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
        
        // Collect a batch of pending evaluations
        std::vector<PendingEvaluation> evaluations;
        PendingEvaluation pending_eval;
        
        // Improved batch collection strategy with adaptive waiting
        const auto start_time = std::chrono::steady_clock::now();
        const auto initial_wait_time = std::chrono::milliseconds(5);  // Start with longer wait
        const auto max_wait_time = std::chrono::milliseconds(50);     // Longer maximum wait time
        const size_t target_batch_size = batch_size_;
        const size_t min_batch_for_processing = std::max<size_t>(batch_size_ / 4, 4); // At least 25% of target or 4 items
        
        // Phase 1: Quick bulk collection
        if (external_leaf_queue->size_approx() > 0) {
            size_t bulk_size = std::min(external_leaf_queue->size_approx(), batch_size_);
            std::vector<PendingEvaluation> bulk_items(bulk_size);
            size_t dequeued = external_leaf_queue->try_dequeue_bulk(bulk_items.data(), bulk_size);
            for (size_t i = 0; i < dequeued; ++i) {
                evaluations.push_back(std::move(bulk_items[i]));
            }
        }
        
        // Phase 2: Adaptive waiting strategy for better batching
        auto current_wait = initial_wait_time;
        while (evaluations.size() < target_batch_size && 
               std::chrono::steady_clock::now() - start_time < max_wait_time) {
            
            // Try bulk dequeue first if queue has items
            size_t queue_size = external_leaf_queue->size_approx();
            if (queue_size > 0 && evaluations.size() < target_batch_size) {
                size_t to_dequeue = std::min(queue_size, target_batch_size - evaluations.size());
                std::vector<PendingEvaluation> temp_items(to_dequeue);
                size_t dequeued = external_leaf_queue->try_dequeue_bulk(temp_items.data(), to_dequeue);
                for (size_t i = 0; i < dequeued; ++i) {
                    evaluations.push_back(std::move(temp_items[i]));
                }
            }
            
            // If we have enough items for a reasonable batch, can stop
            if (evaluations.size() >= min_batch_for_processing) {
                // Check if we should wait a bit more for a fuller batch
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed < initial_wait_time || evaluations.size() >= target_batch_size * 0.75) {
                    break;  // We have enough items or waited enough
                }
            }
            
            // If queue is growing rapidly, wait a bit for more items
            size_t current_queue_size = external_leaf_queue->size_approx();
            if (current_queue_size > queue_size * 1.5) {
                // Queue is growing fast, wait for more items to accumulate
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            
            // Adaptive sleep - increase wait time if queue is empty
            if (evaluations.empty()) {
                std::this_thread::sleep_for(current_wait);
                current_wait = std::min(current_wait * 2, std::chrono::milliseconds(5));
            } else {
                std::this_thread::yield();  // Just yield if we have some items
            }
        }
        
        // Only process if we have a reasonable batch or waited long enough
        if (evaluations.size() < min_batch_for_processing && 
            !shutdown_flag_.load(std::memory_order_acquire)) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            // If we haven't waited long enough and have very few items, put them back
            if (elapsed < initial_wait_time && evaluations.size() < 2) {
                for (auto& eval : evaluations) {
                    external_leaf_queue->enqueue(std::move(eval));
                }
                return false;
            }
            // Otherwise process what we have to avoid deadlock
        }
        
        if (evaluations.empty()) {
            return false;
        }
        
        // Report batch size
        std::cout << "[EVALUATOR] Processing batch of " << evaluations.size() 
                  << " items (max: " << batch_size_ << ", total batches: " 
                  << total_batches_.load() << ")" << std::endl;
        
        // Track memory periodically
        static int batch_count = 0;
        if (batch_count % 10 == 0) {
            alphazero::utils::trackMemory("Evaluator batch #" + std::to_string(batch_count));
        }
        batch_count++;
        
        try {
            // Extract states for inference
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.reserve(evaluations.size());
            
            // Use OpenMP for parallel state extraction when beneficial
            if (evaluations.size() > 64) {
                std::vector<std::unique_ptr<core::IGameState>> temp_states(evaluations.size());
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < evaluations.size(); ++i) {
                    temp_states[i] = std::move(evaluations[i].state);
                }
                states = std::move(temp_states);
            } else {
                for (auto& eval : evaluations) {
                    states.push_back(std::move(eval.state));
                }
            }
            
            // Perform inference
            std::vector<NetworkOutput> results = inference_fn_(states);
            
            // Pair results with original evaluations and enqueue to result queue
            if (results.size() == evaluations.size()) {
                // Use bulk enqueue for better performance when possible
                if (results.size() > 1) {
                    std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs;
                    result_pairs.reserve(results.size());
                    
                    for (size_t i = 0; i < results.size(); ++i) {
                        result_pairs.emplace_back(
                            std::move(results[i]), 
                            std::move(evaluations[i])
                        );
                    }
                    
                    external_result_queue->enqueue_bulk(
                        std::make_move_iterator(result_pairs.begin()),
                        result_pairs.size()
                    );
                } else {
                    // Single item, use regular enqueue
                    external_result_queue->enqueue(std::make_pair(
                        std::move(results[0]), 
                        std::move(evaluations[0])
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
    auto batch = collectBatch(batch_size_);
    if (batch.empty()) {
        return false;
    }
    
    processBatch(batch);
    return true;
}

void MCTSEvaluator::processBatches() {
    // MCTSEvaluator::processBatches worker thread started
    
    // For MCTSEngine integration with batch queue - use proper types
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
                bool use_external_queue = (use_external_queues_ && external_leaf_queue && external_result_queue);
                
                // Print detailed debugging info first time and periodically
                static int external_queue_check_counter = 0;
                if ((!printed_queue_info || external_queue_check_counter++ % 1000 == 0) && use_external_queue) {
                    // External queue status logged
                    if (!printed_queue_info) {
                        printed_queue_info = true;
                    }
                }
                
                if (use_external_queue) {
                    // Collect batch directly from leaf queue
                    std::vector<PendingEvaluation> evaluations;
                    evaluations.reserve(batch_size_);
                    auto batch_start_collect = std::chrono::steady_clock::now();
                    
                    // Aggressive batch collection - wait for larger batches
                    const auto start_time = std::chrono::steady_clock::now();
                    const auto max_wait_time = std::chrono::milliseconds(50); // Increased timeout for better batching
                    const size_t min_batch_for_processing = std::max<size_t>(16, batch_size_ / 4); // At least 25% of target
                    
                    // Phase 1: Quick bulk collection
                    if (external_leaf_queue->size_approx() > 0) {
                        size_t bulk_size = std::min(external_leaf_queue->size_approx(), batch_size_);
                        std::vector<PendingEvaluation> bulk_items(bulk_size);
                        size_t dequeued = external_leaf_queue->try_dequeue_bulk(bulk_items.data(), bulk_size);
                        for (size_t i = 0; i < dequeued; ++i) {
                            evaluations.push_back(std::move(bulk_items[i]));
                        }
                    }
                    
                    // Phase 2: Wait for more items if batch is small
                    while (evaluations.size() < batch_size_ && 
                           std::chrono::steady_clock::now() - start_time < max_wait_time) {
                        PendingEvaluation eval;
                        if (external_leaf_queue->try_dequeue(eval)) {
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
                        auto wait_result = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
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
                        std::cout << "[EVALUATOR] Processing batch of " << evaluations.size() 
                                  << " items (max: " << batch_size_ << ")" << std::endl;
                        alphazero::utils::trackMemory("Evaluator batch #" + std::to_string(batch_num));
                    }
                    
                    // Collected batch from leaf queue
                    
                    auto batch_start_time = std::chrono::high_resolution_clock::now();
                    
                    try {
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
                            
                            // Notify result distributor that results are available
                            if (result_notify_callback_) {
                                result_notify_callback_();
                            }
                            // Also directly notify any waiting threads
                            cv_.notify_all();
                        } else {
                            // Mismatch in result count
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
                auto batch_collection_start_time = std::chrono::high_resolution_clock::now();
                std::vector<EvaluationRequest> batch = collectBatch(batch_size_); 
                auto batch_collection_end_time = std::chrono::high_resolution_clock::now();
                auto batch_collection_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_collection_end_time - batch_collection_start_time).count();

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

                bool should_log_detail = total_batches_.load(std::memory_order_relaxed) % 20 == 0;
                size_t batch_count = total_batches_.load(std::memory_order_relaxed) + 1;
                
                // Log internal batch size periodically  
                if (total_batches_.load(std::memory_order_relaxed) % 10 == 0) {
                    std::cout << "[EVALUATOR] Processing internal batch of " << batch.size() 
                              << " items (max: " << batch_size_ << ")" << std::endl;
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
                
                auto inference_start_time = std::chrono::high_resolution_clock::now();
                
                try {
                    // Perform batch inference
                    std::vector<NetworkOutput> results = inference_fn_(states_for_eval);
                    
                    // Record inference time
                    auto inference_end_time = std::chrono::high_resolution_clock::now();
                    auto inference_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time).count();
                    
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
                                    MCTSEngine::PendingEvaluation pending_eval;
                                    pending_eval.node = req_ptr->node;
                                    // Note: path is not available here, would need to be passed through
                                    
                                    auto* result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, MCTSEngine::PendingEvaluation>>*>(result_queue_ptr_);
                                    result_queue->enqueue(std::make_pair(std::move(results[i]), std::move(pending_eval)));
                                    
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

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch(size_t target_batch_size) {
    // If target_batch_size is 0, use the default batch_size_
    if (target_batch_size == 0) {
        target_batch_size = batch_size_;
    }
    
    std::vector<EvaluationRequest> batch;
    batch.reserve(target_batch_size);
    
    auto start_time = std::chrono::steady_clock::now();
    
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
    
    // Report batch size for internal queue
    std::cout << "[EVALUATOR-INT] Processing internal batch of " << batch.size() 
              << " items" << std::endl;
    
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
        total_batches_.fetch_add(1, std::memory_order_relaxed);
        total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
        cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
        
    } catch (const std::exception& e) {
        // Error during batch inference
        
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