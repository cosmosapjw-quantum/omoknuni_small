/**
 * @file mcts_evaluator_concurrent.cpp
 * @brief Implementation of concurrent processing functions and methods for MCTSEvaluator
 * 
 * This file contains the concurrent processing logic for the MCTS evaluator, including:
 * - Thread management for batch collection, inference, and result distribution
 * - Concurrent queue processing with lock-free structures
 * - Adaptive backoff mechanisms for efficient thread synchronization
 * - Batch parameter configuration and management
 * - External and internal queue processing methods
 * 
 * It combines functionality previously split between thread functions and parameter methods
 * into a single coherent file focused on concurrent processing aspects of the evaluator.
 */
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

using namespace alphazero::mcts;

namespace alphazero {
namespace mcts {

//
// Thread management and concurrent processing functions
//

void MCTSEvaluator::batchCollectorLoop() {
    // Main loop for collecting batches and submitting to inference workers
    auto last_status_time = std::chrono::steady_clock::now();
    int iteration_count = 0;
    int total_batches_collected = 0;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        iteration_count++;
        
        // Different logic for external vs internal queues
        if (use_external_queues_) {
            auto now = std::chrono::steady_clock::now();
            bool should_report = 
                (iteration_count < 100 && iteration_count % 10 == 0) ||
                (iteration_count % 100 == 0) ||
                (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 1);
                
            if (should_report) {
                size_t leaf_queue_size = 0;
                if (leaf_queue_ptr_) {
                    auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
                    leaf_queue_size = external_leaf_queue->size_approx();
                }
                
                // Keep essential batch stats only - this is the one batch logging we preserve
                std::cout << "📈 [BATCH_STATS] " 
                          << "Batches: " << total_batches_.load(std::memory_order_relaxed)
                          << " | Avg size: " << std::fixed << std::setprecision(1) << getAverageBatchSize() 
                          << " | Total states: " << total_evaluations_.load(std::memory_order_relaxed)
                          << " | Target: " << batch_size_ 
                          << " | Queue size: " << leaf_queue_size
                          << " | Active: " << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no")
                          << std::endl;
            }
            
            if (!leaf_queue_ptr_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Get a reference to the external queue
            auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
            
            // Check queue size
            size_t queue_size = external_leaf_queue->size_approx();
            
            // Track queue activity
            static bool first_queue_items_seen = false;
            static int processing_cycle_count = 0;
            if (queue_size > 0) {
                if (!first_queue_items_seen) {
                    first_queue_items_seen = true;
                }
                processing_cycle_count++;
            }
            
            if (should_report) {
                last_status_time = now;
            }
            
            // CRITICAL FIX: Use optimal batch size for better throughput
            size_t target_batch_size = batch_params_.optimal_batch_size;  // Use 64 instead of 32
            
            // If there are items in the queue, collect efficiently 
            if (queue_size > 0) {
                // CRITICAL FIX: Directly collect from the external leaf queue and feed to BatchAccumulator
                // The previous approach of calling collectExternalBatch() was creating a redundant layer
                
                size_t effective_target = std::min(queue_size, target_batch_size);
                
                // Collect items directly from the external queue
                std::vector<PendingEvaluation> pending_eval_batch;
                pending_eval_batch.reserve(effective_target);
                
                // Use bulk dequeue for efficiency
                pending_eval_batch.resize(effective_target);
                size_t dequeued = external_leaf_queue->try_dequeue_bulk(pending_eval_batch.data(), effective_target);
                pending_eval_batch.resize(dequeued);
                
                // If we didn't get enough items, try individual dequeues for a short time
                if (dequeued < batch_params_.minimum_viable_batch_size && queue_size >= batch_params_.minimum_viable_batch_size) {
                    auto start_time = std::chrono::steady_clock::now();
                    auto max_wait = std::chrono::milliseconds(10);
                    
                    while (pending_eval_batch.size() < effective_target && 
                           std::chrono::steady_clock::now() - start_time < max_wait) {
                        PendingEvaluation eval;
                        if (external_leaf_queue->try_dequeue(eval)) {
                            pending_eval_batch.push_back(std::move(eval));
                        } else {
                            break; // No more items available
                        }
                    }
                }
                
                if (!pending_eval_batch.empty()) {
                    total_batches_collected++;

                    // CRITICAL FIX: If we already have a good sized batch, process it directly
                    // instead of breaking it apart and sending to BatchAccumulator
                    if (pending_eval_batch.size() >= batch_params_.minimum_viable_batch_size) {
                        // We have a decent sized batch - send it directly to inference
                        // This bypasses the BatchAccumulator to avoid fragmenting good batches
                        
                        // Clean invalid items first
                        auto end_iter = std::remove_if(pending_eval_batch.begin(), pending_eval_batch.end(),
                            [](const PendingEvaluation& eval) {
                                if (!eval.state || !eval.node) {
                                    if (eval.node) {
                                        eval.node->clearAllEvaluationFlags();
                                    }
                                    return true;
                                }
                                return false;
                            });
                        pending_eval_batch.erase(end_iter, pending_eval_batch.end());
                        
                        if (!pending_eval_batch.empty()) {
                            // Submit the pre-formed batch directly to completed queue
                            if (batch_accumulator_) {
                                batch_accumulator_->submitDirectBatch(std::move(pending_eval_batch));
                            }
                        }
                    } else {
                        // Small batch - add individual items to accumulator for better batching
                        if (batch_accumulator_) {
                            for (auto& eval : pending_eval_batch) {
                                if (eval.state && eval.node) {
                                    batch_accumulator_->addEvaluation(std::move(eval));
                                } else {
                                    // Clear flags for invalid items
                                    if (eval.node) {
                                        eval.node->clearAllEvaluationFlags();
                                    }
                                }
                            }
                        } else {
                            std::cerr << "ERROR: batchCollectorLoop - batch_accumulator_ is null!" << std::endl;
                        }
                    }
                } else {
                    // If we couldn't collect any items despite queue_size > 0, 
                    // the queue might have been drained by another thread
                    std::this_thread::yield();
                }
            } else {
                // No items - use adaptive polling for efficiency 
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           (leaf_queue_ptr_ && static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_)->size_approx() > 0);
                };
                
                // Use adaptive backoff with very short timeout for responsiveness
                static AdaptiveBackoff backoff(30, 10, 500);
                backoff.wait_for(wait_pred, std::chrono::milliseconds(1));
            }
        } else {
            // Internal queue mode
            size_t queue_size = request_queue_.size_approx();
            
            // Try to collect a batch if we have enough items or have timed out
            if (queue_size >= min_batch_size_) {
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() >= min_batch_size_;
                };
                
                // Use adaptive backoff with timeout from configuration
                static AdaptiveBackoff backoff(20, 100, 2000);
                backoff.wait_for(wait_pred, timeout_);
                
                // Process whatever we have after timeout
                auto batch = collectInternalBatch(batch_size_);
                if (!batch.empty()) {
                    processInternalBatch(batch);
                }
            } else if (queue_size > 0) {
                // We have some items but not enough - wait and see if more arrive
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() > 0;
                };
                
                // Use adaptive backoff with short timeout for empty queue case
                static AdaptiveBackoff backoff(10, 200, 1000);
                backoff.wait_for(wait_pred, std::chrono::milliseconds(5));
            }
        }
    }
}

void MCTSEvaluator::inferenceWorkerLoop() {
    // Neural network inference worker thread
    auto last_status_time = std::chrono::steady_clock::now();
    int iteration_count = 0;
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        iteration_count++;
        auto now = std::chrono::steady_clock::now();
        bool should_report = 
            (iteration_count < 100 && iteration_count % 10 == 0) ||
            (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 1);
            
        if (should_report) {
            last_status_time = now;
        }
        
        // Try to get a completed batch of PendingEvaluation from the BatchAccumulator
        std::vector<PendingEvaluation> pending_eval_batch;
        bool got_batch_from_accumulator = false;
        if (batch_accumulator_) {
            got_batch_from_accumulator = batch_accumulator_->getCompletedBatch(pending_eval_batch);
        } else {
            std::cerr << "ERROR: inferenceWorkerLoop - batch_accumulator_ is null! Cannot get completed batch." << std::endl;
            // If accumulator is null, this worker has nothing to do. Sleep to avoid busy loop.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (!got_batch_from_accumulator) {
            // CRITICAL FIX: More responsive waiting for batch accumulator
            // Use condition variable with timeout to ensure we wake up when batches are ready
            std::unique_lock<std::mutex> lock(cv_mutex_);
            
            // Check again under lock in case batch became available
            if (batch_accumulator_) {
                got_batch_from_accumulator = batch_accumulator_->getCompletedBatch(pending_eval_batch);
            }
            
            if (!got_batch_from_accumulator) {
                // Wait with condition variable for faster notification
                cv_.wait_for(lock, std::chrono::milliseconds(5), [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) ||
                           (batch_accumulator_ && std::get<1>(batch_accumulator_->getStats()) > 0); // Check if any batches exist
                });
            }
            
            if (!got_batch_from_accumulator) {
                continue;
            }
        }
        
        // Check for valid states (now in pending_eval_batch)
        if (pending_eval_batch.empty()) {
            // pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed); // This counter might need rethinking
            continue;
        }

        // Extract states for inference from pending_eval_batch
        std::vector<std::unique_ptr<core::IGameState>> states_for_inference;
        std::vector<PendingEvaluation> valid_pending_evals; // To keep track of original PendingEvaluations that are valid
        states_for_inference.reserve(pending_eval_batch.size());
        valid_pending_evals.reserve(pending_eval_batch.size());

        for (auto& eval : pending_eval_batch) {
            if (eval.state && eval.node) { // Basic validation
                auto state_clone = eval.state->clone();
                if (state_clone) {
                    states_for_inference.push_back(std::move(state_clone));
                    valid_pending_evals.push_back(std::move(eval)); // Move the valid eval
                } else {
                     if(eval.node) eval.node->clearEvaluationFlag();
                }
            } else {
                 if(eval.node) eval.node->clearEvaluationFlag();
            }
        }
        pending_eval_batch.clear(); // Original batch is processed or items moved/dropped

        if (states_for_inference.empty()) {
            continue; // No valid states to infer
        }
        
        try {
            // Run neural network inference on the batch
            auto inference_start_time = std::chrono::steady_clock::now();
            std::vector<NetworkOutput> results = inference_fn_(states_for_inference);
            auto inference_end_time = std::chrono::steady_clock::now();
            auto inference_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time).count();

            std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs_external;
            if (use_external_queues_) {
                result_pairs_external.reserve(results.size());
            }

            if (results.size() == valid_pending_evals.size()) {
                for (size_t i = 0; i < results.size(); ++i) {
                    if (use_external_queues_) {
                        result_pairs_external.emplace_back(std::move(results[i]), std::move(valid_pending_evals[i]));
                    } else {
                        // CRITICAL FIX: Handle internal path with BatchAccumulator
                        // When not using external queues, directly update the nodes with the results
                        auto& eval = valid_pending_evals[i];
                        auto& output = results[i];
                        
                        if (eval.node) {
                            try {
                                // Apply evaluation to the node
                                eval.node->setPriorProbabilities(output.policy);
                                
                                // Update the path with values
                                float value = output.value;
                                for (auto& path_node : eval.path) {
                                    if (path_node) {
                                        path_node->update(value);
                                        value = -value; // Flip value for next level
                                    }
                                }
                                
                                // Clear evaluation flag
                                eval.node->clearEvaluationFlag();
                            } catch (const std::exception& e) {
                                std::cerr << "Error updating node in internal path: " << e.what() << std::endl;
                                eval.node->clearEvaluationFlag();
                            }
                        }
                    }
                }
            } else {
                // Handle size mismatch - process as many as possible
                size_t smaller_size = std::min(results.size(), valid_pending_evals.size());
                
                for (size_t i = 0; i < smaller_size; ++i) {
                    if (use_external_queues_) {
                        result_pairs_external.emplace_back(std::move(results[i]), std::move(valid_pending_evals[i]));
                    } else {
                        // Handle internal path for partial results
                        auto& eval = valid_pending_evals[i];
                        auto& output = results[i];
                        
                        if (eval.node) {
                            try {
                                eval.node->setPriorProbabilities(output.policy);
                                
                                float value = output.value;
                                for (auto& path_node : eval.path) {
                                    if (path_node) {
                                        path_node->update(value);
                                        value = -value;
                                    }
                                }
                                
                                eval.node->clearEvaluationFlag();
                            } catch (const std::exception& e) {
                                eval.node->clearEvaluationFlag();
                            }
                        }
                    }
                }
                
                // Clear evaluation flags for any valid_pending_evals not getting a result
                for (size_t i = smaller_size; i < valid_pending_evals.size(); ++i) {
                    if (valid_pending_evals[i].node) {
                        valid_pending_evals[i].node->clearEvaluationFlag();
                    }
                }
            }
            
            if (use_external_queues_) {
                if (!result_pairs_external.empty() && result_queue_ptr_) {
                    auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
                    external_result_queue->enqueue_bulk(std::make_move_iterator(result_pairs_external.begin()), result_pairs_external.size());
                    
                    if (result_notify_callback_) {
                        result_notify_callback_();
                    }

                    total_batches_.fetch_add(1, std::memory_order_relaxed);
                    total_evaluations_.fetch_add(result_pairs_external.size(), std::memory_order_relaxed);
                    cumulative_batch_size_.fetch_add(result_pairs_external.size(), std::memory_order_relaxed);
                    cumulative_batch_time_ms_.fetch_add(inference_duration_ms, std::memory_order_relaxed);
                }
            } else {
                // CRITICAL FIX: Internal queue path with BatchAccumulator
                // Since we handled the node updates above, just update stats here
                total_batches_.fetch_add(1, std::memory_order_relaxed);
                total_evaluations_.fetch_add(valid_pending_evals.size(), std::memory_order_relaxed);
                cumulative_batch_size_.fetch_add(valid_pending_evals.size(), std::memory_order_relaxed);
                cumulative_batch_time_ms_.fetch_add(inference_duration_ms, std::memory_order_relaxed);
            }
            
            // pending_inference_batches_ counter might need to be removed or handled by accumulator itself.
            
        } catch (const std::exception& e) {
            pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed);
            
            if (torch::cuda::is_available()) {
                try {
                    c10::cuda::CUDACachingAllocator::emptyCache();
                } catch (...) {}
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
            // No results available, use adaptive polling instead of condition variables
            auto wait_pred = [this]() {
                return shutdown_flag_.load(std::memory_order_acquire) || 
                       result_queue_internal_.size_approx() > 0;
            };
            
            // Wait with adaptive backoff to reduce CPU usage while still being responsive
            static AdaptiveBackoff backoff(10, 100, 2000);
            backoff.wait_for(wait_pred, std::chrono::milliseconds(5));
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

void MCTSEvaluator::evaluationLoop() {
    if (use_external_queues_) {
        processExternalQueue();
    } else {
        processInternalQueue();
    }
}

//
// Batch parameter and configuration management
//

void MCTSEvaluator::setBatchParameters(const BatchParameters& params) {
    if (params.optimal_batch_size == 0) {
        batch_params_.optimal_batch_size = 128;
    } else {
        batch_params_.optimal_batch_size = params.optimal_batch_size;
    }

    if (params.minimum_viable_batch_size == 0) {
        batch_params_.minimum_viable_batch_size = std::max(size_t(batch_params_.optimal_batch_size * 0.75), size_t(64));
    } else {
        batch_params_.minimum_viable_batch_size = params.minimum_viable_batch_size;
    }

    batch_params_.minimum_fallback_batch_size = params.minimum_fallback_batch_size;
    if (batch_params_.minimum_fallback_batch_size == 0) {
        batch_params_.minimum_fallback_batch_size = std::max(size_t(batch_params_.optimal_batch_size * 0.3), size_t(16));
    }

    if (params.max_wait_time.count() < 1) {
        batch_params_.max_wait_time = std::chrono::milliseconds(50);
    } else {
        batch_params_.max_wait_time = params.max_wait_time;
    }

    batch_params_.additional_wait_time = params.additional_wait_time;
    if (batch_params_.additional_wait_time.count() < 1) {
        batch_params_.additional_wait_time = std::chrono::milliseconds(10);
    }
    
    // Update legacy parameters for backwards compatibility
    batch_size_ = batch_params_.optimal_batch_size;
    min_batch_size_ = batch_params_.minimum_fallback_batch_size;
    optimal_batch_size_ = batch_params_.minimum_viable_batch_size;
    timeout_ = batch_params_.max_wait_time;
    additional_wait_time_ = batch_params_.additional_wait_time;
    
    // Update batch accumulator if it exists
    if (batch_accumulator_) {
        batch_accumulator_->updateParameters(
            batch_params_.optimal_batch_size,
            batch_params_.minimum_viable_batch_size,
            batch_params_.max_wait_time
        );
        
        if (!batch_accumulator_->isRunning()) {
            try {
                batch_accumulator_->stop();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } catch (...) {}
            
            try {
                batch_accumulator_->start();
                
                // Check if it started successfully
                if (!batch_accumulator_->isRunning()) {
                    batch_accumulator_ = std::make_unique<BatchAccumulator>(
                        batch_params_.optimal_batch_size,
                        batch_params_.minimum_viable_batch_size,
                        batch_params_.max_wait_time
                    );
                    
                    // Start the new accumulator
                    if (batch_accumulator_) {
                        batch_accumulator_->start();
                    }
                }
            } catch (const std::exception& e) {
                try {
                    batch_accumulator_ = std::make_unique<BatchAccumulator>(
                        batch_params_.optimal_batch_size,
                        batch_params_.minimum_viable_batch_size,
                        batch_params_.max_wait_time
                    );
                    
                    if (batch_accumulator_) {
                        batch_accumulator_->start();
                    }
                } catch (...) {
                }
            }
        }
    } else {
        try {
            batch_accumulator_ = std::make_unique<BatchAccumulator>(
                batch_params_.optimal_batch_size,
                batch_params_.minimum_viable_batch_size,
                batch_params_.max_wait_time
            );
            
            if (batch_accumulator_) {
                batch_accumulator_->start();
            }
        } catch (const std::exception& e) {
        }
    }
    
    // Update pipeline buffer target batch size
    pipeline_buffer_.setTargetBatchSize(batch_params_.optimal_batch_size);
}

BatchParameters MCTSEvaluator::getBatchParameters() const {
    return batch_params_;
}

//
// External and internal queue processing methods
//

void MCTSEvaluator::processExternalQueue() {
    static int consecutive_empty_iterations = 0;
    
    // Main loop for external queue processing  
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        // Check if we have valid queues
        if (!leaf_queue_ptr_ || !result_queue_ptr_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
    
        auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
        size_t queue_size = external_leaf_queue->size_approx();
        
        // If using batch accumulator, process with it
        if (batch_accumulator_) {
            // Ensure accumulator is running
            if (!batch_accumulator_->isRunning()) {
                batch_accumulator_->start();
                
                // Verify that it actually started
                if (!batch_accumulator_->isRunning()) {
                    // Extra startup attempt with delay
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    batch_accumulator_->start();
                }
            }
            
            // Try to dequeue new items and add to accumulator
            int items_added = 0;
            const int MAX_ITEMS_PER_CYCLE = 512;
            
            // Only proceed if there are items in the queue
            if (queue_size > 0) {
                // Try to bulk dequeue items for better efficiency
                std::vector<PendingEvaluation> bulk_items(std::min(MAX_ITEMS_PER_CYCLE, static_cast<int>(queue_size)));
                size_t num_dequeued = external_leaf_queue->try_dequeue_bulk(bulk_items.data(), bulk_items.size());
                
                if (num_dequeued > 0) {
                    bulk_items.resize(num_dequeued);
                    
                    // Add all items to batch accumulator
                    for (auto& item : bulk_items) {
                        if (item.state && item.node) {
                            try {
                                if (item.state->validate()) {
                                    batch_accumulator_->addEvaluation(std::move(item));
                                    items_added++;
                                } else {
                                    if (item.node) item.node->clearEvaluationFlag();
                                }
                            } catch (const std::exception& e) {
                                if (item.node) item.node->clearEvaluationFlag();
                            }
                        } else {
                            if (item.node) item.node->clearEvaluationFlag();
                        }
                    }
                    
                    // Reset empty iterations counter when we add items
                    consecutive_empty_iterations = 0;
                } else {
                    // If bulk dequeue fails but queue_size > 0, try single dequeue
                    PendingEvaluation eval;
                    while (items_added < MAX_ITEMS_PER_CYCLE && external_leaf_queue->try_dequeue(eval)) {
                        if (eval.state && eval.node) {
                            try {
                                if (eval.state->validate()) {
                                    batch_accumulator_->addEvaluation(std::move(eval));
                                    items_added++;
                                } else {
                                    if (eval.node) eval.node->clearEvaluationFlag();
                                }
                            } catch (const std::exception& e) {
                                if (eval.node) eval.node->clearEvaluationFlag();
                            }
                        } else {
                            if (eval.node) eval.node->clearEvaluationFlag();
                        }
                    }
                    
                    if (items_added > 0) {
                        // Reset empty iterations counter when we add items
                        consecutive_empty_iterations = 0;
                    } else {
                        consecutive_empty_iterations++;
                    }
                }
                
                if (items_added > 0) {
                    if (result_notify_callback_) {
                        result_notify_callback_();
                    }
                }
            } else {
                consecutive_empty_iterations++;
            }
            
            // Get completed batch from accumulator and process it if available
            std::vector<PendingEvaluation> batch;
            if (batch_accumulator_->getCompletedBatch(batch)) {
                // Reset empty iterations counter when we process a batch
                consecutive_empty_iterations = 0;
                    
                // Process the batch
                processBatchWithAccumulator(std::move(batch));
            }
            
            // Sleep based on activity to optimize CPU usage
            if (items_added > 0 || !batch.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else if (consecutive_empty_iterations < 10) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            } else if (consecutive_empty_iterations < 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            
            continue; // Continue the main loop instead of returning
        }
        
        // Legacy path - process without batch accumulator
        // This is essentially the old code path for compatibility
        std::vector<PendingEvaluation> batch = collectExternalBatch(batch_size_);
        if (!batch.empty()) {
            consecutive_empty_iterations = 0;
            processBatch();
        } else {
            consecutive_empty_iterations++;
            
            if (consecutive_empty_iterations >= 10) {
                size_t fallback_batch_size = std::max(size_t(1), batch_size_ / 8);
                std::vector<PendingEvaluation> small_batch = collectExternalBatch(fallback_batch_size);
                
                if (!small_batch.empty()) {
                    consecutive_empty_iterations = 0;
                    processBatch();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    // End of main processing loop
}

void MCTSEvaluator::processInternalQueue() {
    // Main loop for internal queue processing
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        // Check if we have any requests to process
        if (request_queue_.size_approx() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        // If using batch accumulator, process with it
        if (batch_accumulator_) {
            // Internal request conversion is more complex due to different types
            // We need to convert EvaluationRequest to PendingEvaluation
            
            // Get up to 32 items per cycle
            std::vector<EvaluationRequest> requests;
            requests.reserve(32);
            
            EvaluationRequest req;
            int items_added = 0;
            
            while (items_added < 32 && request_queue_.try_dequeue(req)) {
                requests.push_back(std::move(req));
                items_added++;
            }
            
            // Convert requests to pending evaluations for batch accumulator
            for (auto& request : requests) {
                if (request.node && request.state) {
                    PendingEvaluation eval;
                    eval.node = request.node;
                    eval.state = std::move(request.state);
                    
                    // Create a promise/future pair for the result
                    auto promise_ptr = std::make_shared<std::promise<NetworkOutput>>();
                    request.promise = std::move(*promise_ptr.get());
                    
                    // Add to batch accumulator
                    batch_accumulator_->addEvaluation(std::move(eval));
                } else {
                    // Invalid request, fulfill with default output
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    int action_size = request.action_space_size > 0 ? request.action_space_size : 10;
                    default_output.policy.resize(action_size, 1.0f / action_size);
                    
                    try {
                        request.promise.set_value(std::move(default_output));
                    } catch (...) {
                        // Promise might be broken already
                    }
                }
            }
            
            // Get completed batch from accumulator
            std::vector<PendingEvaluation> batch;
            if (batch_accumulator_->getCompletedBatch(batch)) {
                // Process the batch
                processBatchWithAccumulator(std::move(batch));
            }
            
            // If no items added, sleep briefly to avoid busy waiting
            if (items_added == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            continue; // Continue the main loop instead of returning
        }
        
        // Legacy path - process without batch accumulator
        // This is essentially the old code path for compatibility
        auto batch = collectInternalBatch(batch_size_);
        if (!batch.empty()) {
            processInternalBatch(batch);
        } else {
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    // End of main processing loop
}

void MCTSEvaluator::processBatchWithAccumulator(std::vector<PendingEvaluation> batch) {
    if (batch.empty()) {
        return;
    }
    
    auto batch_start = std::chrono::steady_clock::now();
    
    // Extract states for inference
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    // Create a vector of valid indices
    std::vector<size_t> valid_indices;
    valid_indices.reserve(batch.size());
    
    for (size_t i = 0; i < batch.size(); ++i) {
        if (batch[i].state) {
            auto state_clone = batch[i].state->clone();
            if (state_clone) {
                states.push_back(std::move(state_clone));
                valid_indices.push_back(i);
            }
        }
    }
    
    // Skip if no valid states
    if (states.empty()) {
        return;
    }
    
    // Perform inference
    std::vector<NetworkOutput> results;
    try {
        results = inference_fn_(states);
    } catch (const std::exception& e) {
        std::cerr << "Exception during inference in processBatchWithAccumulator: " << e.what() << std::endl;
        // If inference fails, clear evaluation flags for all nodes in the batch
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            size_t original_index = valid_indices[i];
            auto& eval = batch[original_index];
            if (eval.node) {
                eval.node->clearEvaluationFlag();
            }
        }
        return;
    }
    
    // Check results vs valid indices
    if (results.size() != valid_indices.size()) {
        std::cerr << "Mismatch between results and valid indices in processBatchWithAccumulator: " 
                  << results.size() << " vs " << valid_indices.size() << std::endl;
        // Clear evaluation flags for all nodes in the batch
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            size_t original_index = valid_indices[i];
            auto& eval = batch[original_index];
            if (eval.node) {
                eval.node->clearEvaluationFlag();
            }
        }
        return;
    }
    
    // Process results with external or internal approach
    if (use_external_queues_ && result_queue_ptr_) {
        auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
        
        std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs;
        result_pairs.reserve(results.size());
        
        for (size_t i = 0; i < results.size(); ++i) {
            size_t original_index = valid_indices[i];
            result_pairs.emplace_back(
                std::move(results[i]), 
                std::move(batch[original_index])
            );
        }
        
        if (!result_pairs.empty()) {
            external_result_queue->enqueue_bulk(
                std::make_move_iterator(result_pairs.begin()),
                result_pairs.size()
            );
            
            // Notify callback if available
            if (result_notify_callback_) {
                result_notify_callback_();
            }
        }
    } else {
        // Internal case - update nodes directly (or fulfill promises if that was the old path)
        // This path is less likely if BatchAccumulator is primarily for external queues.
        // For simplicity, we'll assume if not external, then it's for internal promise fulfillment for now,
        // though ideally BatchAccumulator would be agnostic or configured for the output type.
        for (size_t i = 0; i < results.size(); ++i) {
            size_t original_index = valid_indices[i];
            auto& eval_request_wrapper = batch[original_index]; // This is PendingEvaluation
            auto& output = results[i];
            
            if (eval_request_wrapper.node) { // PendingEvaluation has .node
                try {
                    eval_request_wrapper.node->setPriorProbabilities(output.policy);
                    
                    float value = output.value;
                    for (auto& node_in_path : eval_request_wrapper.path) {
                        if (node_in_path) {
                            node_in_path->update(value);
                            value = -value; 
                        }
                    }
                    eval_request_wrapper.node->clearEvaluationFlag();
                } catch (const std::exception& e) {
                    std::cerr << "Exception processing node in processBatchWithAccumulator (internal path): " << e.what() << std::endl;
                }
            }
        }
    }
    
    // Update metrics
    auto batch_end = std::chrono::steady_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
    
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(results.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(results.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(batch_duration.count(), std::memory_order_relaxed);
    
    // Log successful batch processing
    static size_t processed_batch_counter = 0;
    processed_batch_counter++;
    
    // Log every batch for debugging
    utils::debug_logger().logBatchAccumulator(
        "Processed batch #" + std::to_string(processed_batch_counter),
        results.size(),
        batch_params_.optimal_batch_size
    );
    
}

} // namespace mcts
} // namespace alphazero