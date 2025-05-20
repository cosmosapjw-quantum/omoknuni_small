// Implementation of thread pool functions for MCTSEvaluator
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/memory_tracker.h"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>

using namespace alphazero::mcts;

namespace alphazero {
namespace mcts {

void MCTSEvaluator::batchCollectorLoop() {
    // Main loop for collecting batches and submitting to inference workers
    auto last_status_time = std::chrono::steady_clock::now();
    int iteration_count = 0;
    int total_batches_collected = 0;
    int total_states_processed = 0;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        iteration_count++;
        
        // Different logic for external vs internal queues
        if (use_external_queues_) {
            // More frequent status reports during startup
            auto now = std::chrono::steady_clock::now();
            bool should_report = 
                (iteration_count < 100 && iteration_count % 10 == 0) || // Every 10 iterations during first 100 iterations
                (iteration_count % 100 == 0) || // Then every 100 iterations
                (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 1); // And at least every second
                
            // Periodically log detailed batch statistics
            if (should_report) {
                std::cout << "[BATCH_STATS] Total batches: " << total_batches_.load(std::memory_order_relaxed)
                          << ", Avg size: " << getAverageBatchSize() 
                          << ", Total states: " << total_evaluations_.load(std::memory_order_relaxed)
                          << ", Target batch: " << batch_size_ << std::endl;
            }
            
            if (!leaf_queue_ptr_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Get a reference to the external queue
            auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
            
            // Check queue size
            size_t queue_size = external_leaf_queue->size_approx();
            
            if (should_report) {
                last_status_time = now;
            }
            
            // Use adaptable batch size based on settings
            // This allows balancing between responsiveness and efficiency
            size_t target_batch_size = std::min(batch_size_, max_collection_batch_size_);
            
            // If there are items in the queue, always process them immediately
            if (queue_size > 0) {
                // Collect batch from external queue
                auto batch = collectExternalBatch(target_batch_size);
                
                if (!batch.empty()) {
                    total_batches_collected++;
                    
                    // Create a new batch for inference
                    BatchForInference inference_batch;
                    inference_batch.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                    inference_batch.created_time = std::chrono::steady_clock::now();
                    inference_batch.pending_evals = std::move(batch);
                    
                    // Extract states for inference
                    inference_batch.states.reserve(inference_batch.pending_evals.size());
                    
                    // Convert states for neural network
                    for (auto& eval : inference_batch.pending_evals) {
                        if (eval.state) {
                            inference_batch.states.push_back(eval.state->clone());
                            total_states_processed++;
                        }
                    }
                    
                    // Enqueue batch for inference if we have states
                    if (!inference_batch.states.empty()) {
                                  
                        inference_queue_.enqueue(std::move(inference_batch));
                        pending_inference_batches_.fetch_add(1, std::memory_order_relaxed);
                        
                        // Notify inference workers - IMPORTANT: notify ALL workers
                        {
                            std::lock_guard<std::mutex> lock(inference_mutex_);
                            inference_cv_.notify_all();  // Notify ALL waiting threads
                        }
                    }
                } else {
                    // No items collected, wait a very short time to avoid spinning
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            } else {
                // No items in queue, wait for notification
                std::unique_lock<std::mutex> lock(cv_mutex_);
                auto wait_pred = [this, external_leaf_queue]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           external_leaf_queue->size_approx() > 0;
                };
                
                // Use very short timeout to ensure we check frequently
                cv_.wait_for(lock, std::chrono::milliseconds(1), wait_pred);
            }
        } else {
            // Internal queue mode
            size_t queue_size = request_queue_.size_approx();
            
            
            // Same logic as external queue but using internal queue and internal batch collection
            if (queue_size >= min_batch_size_) {
                // Process immediately
                auto batch = collectInternalBatch(batch_size_);
                
                if (!batch.empty()) {
                    processInternalBatch(batch);
                }
            } else if (queue_size > 0) {
                // Wait for more items
                std::unique_lock<std::mutex> lock(cv_mutex_);
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() >= min_batch_size_;
                };
                
                cv_.wait_for(lock, timeout_, wait_pred);
                
                // Process whatever we have after timeout
                auto batch = collectInternalBatch(batch_size_);
                
                if (!batch.empty()) {
                    processInternalBatch(batch);
                }
            } else {
                // No items, wait for notification
                std::unique_lock<std::mutex> lock(cv_mutex_);
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() > 0;
                };
                
                cv_.wait_for(lock, std::chrono::milliseconds(5), wait_pred);
            }
        }
    }
}

void MCTSEvaluator::inferenceWorkerLoop() {
    // Neural network inference worker thread
    auto last_status_time = std::chrono::steady_clock::now();
    int iteration_count = 0;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        // Periodically report status (more frequent during startup)
        iteration_count++;
        auto now = std::chrono::steady_clock::now();
        bool should_report = 
            (iteration_count < 100 && iteration_count % 10 == 0) || // Every 10 iterations during first 100 iterations
            (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 1); // Then every second
            
        if (should_report) {
            last_status_time = now;
        }
        
        // Try to get a batch from the inference queue
        BatchForInference batch;
        bool dequeued = inference_queue_.try_dequeue(batch);
        
        if (!dequeued) {
            // No batch available, wait for notification but with a very short timeout
            std::unique_lock<std::mutex> lock(inference_mutex_);
            auto wait_pred = [this]() {
                return shutdown_flag_.load(std::memory_order_acquire) || 
                       inference_queue_.size_approx() > 0;
            };
            
            // Wait with a very short timeout - frequent checks to prevent deadlocks
            inference_cv_.wait_for(lock, std::chrono::milliseconds(1), wait_pred);
            continue;
        }
        
        // Check for valid states
        if (batch.states.empty()) {
            pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed);
            continue;
        }
        
        try {
            // Perform neural network inference
            auto start_time = std::chrono::steady_clock::now();
            std::vector<NetworkOutput> results = inference_fn_(batch.states);
            auto end_time = std::chrono::steady_clock::now();
            
            // Update metrics
            size_t batch_size = batch.states.size();
            auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            
            // Create result batch
            BatchInferenceResult result_batch;
            result_batch.batch_id = batch.batch_id;
            result_batch.outputs = std::move(results);
            result_batch.pending_evals = std::move(batch.pending_evals);
            result_batch.processed_time = end_time;
            
            // Process results
            if (use_external_queues_) {
                // For external queues, send results to the engine's result queue
                auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
                
                if (external_result_queue) {
                    // Pair results with evaluations and enqueue
                    if (result_batch.outputs.size() == result_batch.pending_evals.size()) {
                        std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs;
                        result_pairs.reserve(result_batch.outputs.size());
                        
                        for (size_t i = 0; i < result_batch.outputs.size(); ++i) {
                            result_pairs.emplace_back(
                                std::move(result_batch.outputs[i]),
                                std::move(result_batch.pending_evals[i])
                            );
                        }
                        
                        // Use bulk enqueue for better performance
                        external_result_queue->enqueue_bulk(
                            std::make_move_iterator(result_pairs.begin()),
                            result_pairs.size()
                        );
                        
                        // Notify result processor if callback is available
                        if (result_notify_callback_) {
                            result_notify_callback_();
                        }
                    }
                }
            } else {
                // For internal queue, send to our result distributor
                result_queue_internal_.enqueue(std::move(result_batch));
                pending_result_batches_.fetch_add(1, std::memory_order_relaxed);
                
                // Notify result distributor
                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    result_cv_.notify_one();
                }
            }
            
            // Update statistics
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_evaluations_.fetch_add(batch_size, std::memory_order_relaxed);
            cumulative_batch_size_.fetch_add(batch_size, std::memory_order_relaxed);
            cumulative_batch_time_ms_.fetch_add(batch_duration.count(), std::memory_order_relaxed);
            
            // Update complete
            pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed);
            
        } catch (const std::exception& e) {
            // Error in neural network inference
            pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed);
            
            // Only perform CUDA cache clearing in extreme cases
            static int error_count = 0;
            if (++error_count % 100 == 0 && torch::cuda::is_available()) {
                try {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                } catch (...) {
                    // Ignore cleanup errors
                }
            }
        }
    }
}

void MCTSEvaluator::resultDistributorLoop() {
    // Result distributor thread - only used for internal queue mode
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        // Try to get a result batch
        BatchInferenceResult result_batch;
        
        bool dequeued = result_queue_internal_.try_dequeue(result_batch);
        
        if (!dequeued) {
            // No results available, wait for notification
            std::unique_lock<std::mutex> lock(result_mutex_);
            auto wait_pred = [this]() {
                return shutdown_flag_.load(std::memory_order_acquire) || 
                       result_queue_internal_.size_approx() > 0;
            };
            
            // Wait with a timeout
            result_cv_.wait_for(lock, std::chrono::milliseconds(5), wait_pred);
            continue;
        }
        
        // Process results - fulfill promises
        if (result_batch.outputs.size() == result_batch.pending_evals.size()) {
            for (size_t i = 0; i < result_batch.outputs.size(); ++i) {
                auto& eval = result_batch.pending_evals[i];
                auto& output = result_batch.outputs[i];
                
                if (eval.node) {
                    // Apply evaluation to the node directly
                    eval.node->setPriorProbabilities(output.policy);
                    
                    // Clear evaluation flag to allow other operations
                    eval.node->clearEvaluationFlag();
                }
                
                // Will be used by MCTSEngine for internal tracking - nothing to do here
            }
        }
        
        // Update complete
        pending_result_batches_.fetch_sub(1, std::memory_order_relaxed);
    }
}

// These implementations are moved to mcts_evaluator.cpp to avoid duplicates
// See the full implementations there

// Declarations remain in the header, but implementations are in mcts_evaluator.cpp only
// std::vector<PendingEvaluation> MCTSEvaluator::collectExternalBatch(size_t target_batch_size);
// std::vector<EvaluationRequest> MCTSEvaluator::collectInternalBatch(size_t target_batch_size);
// void MCTSEvaluator::processInternalBatch(std::vector<EvaluationRequest>& batch);

} // namespace mcts
} // namespace alphazero