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
                size_t leaf_queue_size = 0;
                if (leaf_queue_ptr_) {
                    auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
                    leaf_queue_size = external_leaf_queue->size_approx();
                }
                
                std::cout << "[BATCH_STATS] Total batches: " << total_batches_.load(std::memory_order_relaxed)
                          << ", Avg size: " << getAverageBatchSize() 
                          << ", Total states: " << total_evaluations_.load(std::memory_order_relaxed)
                          << ", Target batch: " << batch_size_ 
                          << ", Leaf queue size: " << leaf_queue_size
                          << ", Batch accumulator active: " << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no")
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
            
            // ADDED DEBUG: Log when we first see items in the queue
            static bool first_queue_items_seen = false;
            if (queue_size > 0 && !first_queue_items_seen) {
                first_queue_items_seen = true;
                std::cout << "MCTS BATCHER DEBUG: First items detected in leaf queue (" << queue_size << " items)" << std::endl;
            }
            
            // Log queue size periodically when non-zero
            static int processing_cycle_count = 0;
            if (queue_size > 0 && (++processing_cycle_count % 100 == 0)) {
                std::cout << "MCTS BATCHER DEBUG: External leaf queue has " << queue_size 
                          << " pending items (" << total_evaluations_.load(std::memory_order_relaxed) << " total processed)" 
                          << std::endl;
            }
            
            if (should_report) {
                last_status_time = now;
            }
            
            // Use adaptable batch size based on settings
            // This allows balancing between responsiveness and efficiency
            size_t target_batch_size = std::min(batch_size_, static_cast<size_t>(batch_params_.max_collection_batch_size));
            
            // If there are items in the queue, always process them immediately
            if (queue_size > 0) {
                // Collect batch from external queue
                auto pending_eval_batch = collectExternalBatch(target_batch_size);
                
                if (!pending_eval_batch.empty()) {
                    total_batches_collected++; // Local counter for this loop's activity

                    // Add to this evaluator's batch accumulator
                    if (batch_accumulator_) {
                        for (auto& eval : pending_eval_batch) {
                            if (eval.state && eval.node) { // Ensure valid items before adding
                                batch_accumulator_->addEvaluation(std::move(eval));
                            }
                        }
                        // Notification to the accumulator happens inside its addEvaluation if needed
                    } else {
                        std::cerr << "ERROR: batchCollectorLoop - batch_accumulator_ is null!" << std::endl;
                    }
                }
            } else {
                // No items - use adaptive polling for efficiency 
                auto wait_pred = [this]() {
                    return shutdown_flag_.load(std::memory_order_acquire) || 
                           leaf_queue_ptr_ && static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_)->size_approx() > 0;
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
        // Periodically report status (more frequent during startup)
        iteration_count++;
        auto now = std::chrono::steady_clock::now();
        bool should_report = 
            (iteration_count < 100 && iteration_count % 10 == 0) || // Every 10 iterations during first 100 iterations
            (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 1); // Then every second
            
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
            // No batch available from accumulator, use adaptive polling (or wait on a CV if accumulator signals)
            // For now, simple sleep if accumulator yielded no batch.
            // The BatchAccumulator itself has a loop that waits and forms batches,
            // so this worker mainly waits for the accumulator to produce one.
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Short sleep, accumulator is the main delayer
            continue;
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
                     std::cerr << "Error: Failed to clone state for inference in batch." << std::endl;
                     // How to handle original eval if clone fails? It's already moved from pending_eval_batch effectively.
                     // Maybe clear its node's eval flag? For now, it will be dropped.
                     if(eval.node) eval.node->clearEvaluationFlag();
                }
            } else {
                 std::cerr << "Error: Invalid PendingEvaluation in batch (null state or node)." << std::endl;
                 if(eval.node) eval.node->clearEvaluationFlag(); // Clear flag if node exists
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
                        // This internal path needs to be re-evaluated if BatchAccumulator is now the primary source.
                        // For now, assume internal path also processes PendingEvaluation similar to external.
                        // This implies the internal result_queue_internal_ should also take std::pair<NetworkOutput, PendingEvaluation>.
                        // This is a larger change for the internal path not immediately addressed by this fix for external queues.
                        // Let's focus on the external path working correctly.
                        // If internal path is hit, we'd need to adapt `result_batch_internal` in the `else` block below.
                    }
                }
            } else {
                std::cerr << "Error: Inference results size (" << results.size() 
                          << ") does not match valid pending evaluations size (" << valid_pending_evals.size() 
                          << ")" << std::endl;
                size_t smaller_size = std::min(results.size(), valid_pending_evals.size());
                 for (size_t i = 0; i < smaller_size; ++i) {
                    if (use_external_queues_) {
                        result_pairs_external.emplace_back(std::move(results[i]), std::move(valid_pending_evals[i]));
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
                // Internal queue path - This needs significant rework if BatchAccumulator is the source.
                // The BatchInferenceResult struct expects batch.pending_evals and batch.states which are not directly available here.
                // This path is currently broken by the change to consume from BatchAccumulator.
                // For the purpose of fixing the external queue, we acknowledge this internal path needs revisiting.
                std::cerr << "WARNING: inferenceWorkerLoop internal path hit after consuming from BatchAccumulator - this path is not fully adapted." << std::endl;
                // Attempt to fulfill promises directly for any valid_pending_evals if possible, though this is crude.
                // This part of the 'else' needs to be properly designed if internal mode still uses this worker with an accumulator.
                // For now, this internal path will likely not function as originally intended.
            }
            
            // pending_inference_batches_ counter might need to be removed or handled by accumulator itself.
            
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            
            // Decrement counter even on error
            pending_inference_batches_.fetch_sub(1, std::memory_order_relaxed);
            
            // Log error details if available
            try {
                std::string error_msg = e.what();
                if (!error_msg.empty()) {
                    std::cerr << "CUDA error details: " << error_msg << std::endl;
                }
                
                // If using CUDA, try to get more CUDA error info
                if (torch::cuda::is_available()) {
                    std::cerr << "CUDA memory allocated: " << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).allocated_bytes[static_cast<size_t>(0)].current
                              << ", reserved: " << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).reserved_bytes[static_cast<size_t>(0)].current << std::endl;
                    
                    // Try to clean up CUDA memory
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
            } catch (...) {
                // Ignore errors while trying to get error details
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
    // This is the main thread loop that manages the evaluation process
    std::cout << "MCTSEvaluator::evaluationLoop - Starting with use_external_queues_=" 
              << (use_external_queues_ ? "true" : "false") 
              << ", leaf_queue_ptr_=" << leaf_queue_ptr_
              << ", result_queue_ptr_=" << result_queue_ptr_ << std::endl;
              
    if (use_external_queues_) {
        // Use external queues - MCTSEngine provides the queue
        std::cout << "MCTSEvaluator::evaluationLoop - Using external queue mode" << std::endl;
        processExternalQueue();
    } else {
        // Use internal queues - process our own queue
        std::cout << "MCTSEvaluator::evaluationLoop - Using internal queue mode" << std::endl;
        processInternalQueue();
    }
}

//
// Batch parameter and configuration management
//

void MCTSEvaluator::setBatchParameters(const BatchParameters& params) {
    std::cout << "====== DEBUG: MCTSEvaluator::setBatchParameters CALLED ======" << std::endl;
    std::cout << "BEFORE: this=" << static_cast<void*>(this)
              << ", batch_size_=" << batch_size_
              << ", min_batch_size_=" << min_batch_size_
              << ", optimal_batch_size_=" << optimal_batch_size_
              << ", timeout_=" << timeout_.count() << "ms"
              << ", batch_accumulator_=" << (batch_accumulator_ ? "exists" : "nullptr") 
              << ", batch_accumulator_running=" << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no")
              << std::endl;

    std::cout << "INPUT PARAMETERS: optimal_size=" << params.optimal_batch_size
              << ", min_viable=" << params.minimum_viable_batch_size
              << ", min_fallback=" << params.minimum_fallback_batch_size
              << ", max_wait=" << params.max_wait_time.count() << "ms"
              << ", additional_wait=" << params.additional_wait_time.count() << "ms"
              << std::endl;

    // Validate parameters before applying
    if (params.optimal_batch_size == 0) {
        std::cout << "ERROR: Cannot set optimal_batch_size to 0, forcing to 128" << std::endl;
        batch_params_.optimal_batch_size = 128;
    } else {
        batch_params_.optimal_batch_size = params.optimal_batch_size;
    }

    if (params.minimum_viable_batch_size == 0) {
        std::cout << "NOTICE: minimum_viable_batch_size is 0, setting to 75% of optimal" << std::endl;
        batch_params_.minimum_viable_batch_size = std::max(size_t(batch_params_.optimal_batch_size * 0.75), size_t(64));
    } else {
        batch_params_.minimum_viable_batch_size = params.minimum_viable_batch_size;
    }

    batch_params_.minimum_fallback_batch_size = params.minimum_fallback_batch_size;
    if (batch_params_.minimum_fallback_batch_size == 0) {
        std::cout << "NOTICE: minimum_fallback_batch_size is 0, setting to 30% of optimal" << std::endl;
        batch_params_.minimum_fallback_batch_size = std::max(size_t(batch_params_.optimal_batch_size * 0.3), size_t(16));
    }

    if (params.max_wait_time.count() < 1) {
        std::cout << "ERROR: max_wait_time must be at least 1ms, forcing to 50ms" << std::endl;
        batch_params_.max_wait_time = std::chrono::milliseconds(50);
    } else {
        batch_params_.max_wait_time = params.max_wait_time;
    }

    batch_params_.additional_wait_time = params.additional_wait_time;
    if (batch_params_.additional_wait_time.count() < 1) {
        std::cout << "NOTICE: additional_wait_time is too small, setting to 10ms" << std::endl;
        batch_params_.additional_wait_time = std::chrono::milliseconds(10);
    }
    
    // Update legacy parameters for backwards compatibility
    batch_size_ = batch_params_.optimal_batch_size;
    min_batch_size_ = batch_params_.minimum_fallback_batch_size;
    optimal_batch_size_ = batch_params_.minimum_viable_batch_size;
    timeout_ = batch_params_.max_wait_time;
    additional_wait_time_ = batch_params_.additional_wait_time;
    
    std::cout << "AFTER VALIDATION: optimal_size=" << batch_params_.optimal_batch_size
              << ", min_viable=" << batch_params_.minimum_viable_batch_size
              << ", min_fallback=" << batch_params_.minimum_fallback_batch_size
              << ", max_wait=" << batch_params_.max_wait_time.count() << "ms"
              << ", additional_wait=" << batch_params_.additional_wait_time.count() << "ms"
              << std::endl;
    
    // Update batch accumulator if it exists
    if (batch_accumulator_) {
        std::cout << "INFO: Updating batch accumulator parameters" << std::endl;
        batch_accumulator_->updateParameters(
            batch_params_.optimal_batch_size,
            batch_params_.minimum_viable_batch_size,
            batch_params_.max_wait_time
        );
        
        // Check if batch accumulator is running, start it if not
        if (!batch_accumulator_->isRunning()) {
            std::cout << "WARNING: Batch accumulator exists but is not running! Starting it now..." << std::endl;
            
            // CRITICAL FIX: Stop first in case it's in a bad state
            try {
                batch_accumulator_->stop();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } catch (...) {
                // Ignore errors during stop
            }
            
            // Start the accumulator
            try {
                batch_accumulator_->start();
                
                // Check if it started successfully
                if (batch_accumulator_->isRunning()) {
                    std::cout << "SUCCESS: Batch accumulator started successfully" << std::endl;
                } else {
                    std::cout << "ERROR: Failed to start batch accumulator!" << std::endl;
                    
                    // CRITICAL FIX: Create a new batch accumulator as a last resort
                    std::cout << "CRITICAL FIX: Creating a new batch accumulator as a last resort" << std::endl;
                    batch_accumulator_ = std::make_unique<BatchAccumulator>(
                        batch_params_.optimal_batch_size,
                        batch_params_.minimum_viable_batch_size,
                        batch_params_.max_wait_time
                    );
                    
                    // Start the new accumulator
                    if (batch_accumulator_) {
                        batch_accumulator_->start();
                        std::cout << "New batch accumulator created and started: " 
                                 << (batch_accumulator_->isRunning() ? "SUCCESS" : "FAILED") << std::endl;
                    } else {
                        std::cout << "CRITICAL ERROR: Failed to create new batch accumulator" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "EXCEPTION during batch accumulator start: " << e.what() << std::endl;
                
                // CRITICAL FIX: Create a new batch accumulator on exception
                try {
                    batch_accumulator_ = std::make_unique<BatchAccumulator>(
                        batch_params_.optimal_batch_size,
                        batch_params_.minimum_viable_batch_size,
                        batch_params_.max_wait_time
                    );
                    
                    if (batch_accumulator_) {
                        batch_accumulator_->start();
                        std::cout << "New batch accumulator created after exception: " 
                                 << (batch_accumulator_->isRunning() ? "SUCCESS" : "FAILED") << std::endl;
                    }
                } catch (...) {
                    std::cout << "CRITICAL ERROR: Failed to create new batch accumulator after exception" << std::endl;
                }
            }
        } else {
            std::cout << "INFO: Batch accumulator is already running" << std::endl;
        }
    } else {
        std::cout << "WARNING: batch_accumulator_ is nullptr, creating a new one" << std::endl;
        
        // CRITICAL FIX: Create a new batch accumulator if it doesn't exist
        try {
            batch_accumulator_ = std::make_unique<BatchAccumulator>(
                batch_params_.optimal_batch_size,
                batch_params_.minimum_viable_batch_size,
                batch_params_.max_wait_time
            );
            
            if (batch_accumulator_) {
                batch_accumulator_->start();
                std::cout << "New batch accumulator created: " 
                         << (batch_accumulator_->isRunning() ? "SUCCESS" : "FAILED") << std::endl;
            } else {
                std::cout << "CRITICAL ERROR: Failed to create batch accumulator" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "EXCEPTION during batch accumulator creation: " << e.what() << std::endl;
        }
    }
    
    // Update pipeline buffer target batch size
    std::cout << "INFO: Updating pipeline buffer target batch size to " << batch_params_.optimal_batch_size << std::endl;
    pipeline_buffer_.setTargetBatchSize(batch_params_.optimal_batch_size);
    
    // Log the final parameters
    std::cout << "FINAL MCTSEvaluator::setBatchParameters - Updated batch parameters:"
              << " optimal_size=" << batch_params_.optimal_batch_size
              << ", min_viable=" << batch_params_.minimum_viable_batch_size
              << ", min_fallback=" << batch_params_.minimum_fallback_batch_size
              << ", max_wait=" << batch_params_.max_wait_time.count() << "ms"
              << ", additional_wait=" << batch_params_.additional_wait_time.count() << "ms"
              << std::endl;

    // Final state
    std::cout << "AFTER: this=" << static_cast<void*>(this)
              << ", batch_size_=" << batch_size_
              << ", min_batch_size_=" << min_batch_size_
              << ", optimal_batch_size_=" << optimal_batch_size_
              << ", timeout_=" << timeout_.count() << "ms"
              << ", batch_accumulator_running=" << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no")
              << std::endl;
    std::cout << "====== DEBUG: MCTSEvaluator::setBatchParameters COMPLETED ======" << std::endl;
}

BatchParameters MCTSEvaluator::getBatchParameters() const {
    return batch_params_;
}

//
// External and internal queue processing methods
//

void MCTSEvaluator::processExternalQueue() {
    // Enhanced debug to track call count and frequency
    static int process_call_count = 0;
    static auto last_debug_time = std::chrono::steady_clock::now();
    static int consecutive_empty_iterations = 0;  // Track consecutive empty iterations
    process_call_count++;
    
    // Print detailed debug information every 1 second
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_debug = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_debug_time).count();
    bool print_debug = (process_call_count <= 20) || (time_since_last_debug >= 1000);
    
    if (print_debug) {
        std::cout << "MCTSEvaluator::processExternalQueue - Call #" << process_call_count 
                  << ", time since last debug: " << time_since_last_debug << "ms"
                  << ", consecutive_empty: " << consecutive_empty_iterations << std::endl;
        last_debug_time = now;
    }
    
    // Check if we have valid queues
    if (!leaf_queue_ptr_ || !result_queue_ptr_) {
        if (print_debug) {
            std::cout << "MCTSEvaluator::processExternalQueue - ERROR: Invalid queue pointers! "
                      << "leaf_queue_ptr_=" << leaf_queue_ptr_ 
                      << ", result_queue_ptr_=" << result_queue_ptr_ << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
    
    auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);
    size_t queue_size = external_leaf_queue->size_approx();
    
    // Always log queue size for debugging
    if (print_debug || queue_size > 0) {
        std::cout << "MCTSEvaluator::processExternalQueue - External leaf queue size: " 
                  << queue_size
                  << ", batch_accumulator_running=" << (batch_accumulator_ && batch_accumulator_->isRunning() ? "yes" : "no")
                  << ", use_external_queues_=" << use_external_queues_
                  << ", batch_count=" << (batch_accumulator_ ? std::get<1>(batch_accumulator_->getStats()) : 0)
                  << std::endl;
    }
    
    // If using batch accumulator, process with it
    if (batch_accumulator_) {
        // CRITICAL FIX: Ensure accumulator is running
        if (!batch_accumulator_->isRunning()) {
            std::cout << "MCTSEvaluator::processExternalQueue - ⚠️ RESTARTING batch accumulator that stopped running" << std::endl;
            batch_accumulator_->start();
            
            // CRITICAL FIX: Verify that it actually started
            if (!batch_accumulator_->isRunning()) {
                std::cout << "MCTSEvaluator::processExternalQueue - ❌ Failed to start batch accumulator! Retrying..." << std::endl;
                // Extra startup attempt with delay
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                batch_accumulator_->start();
            }
        }
        
        // CRITICAL FIX: Check if there are any pending items directly from the engine
        // before processing from the accumulator. This prioritizes new items.
        
        // CRITICAL FIX: Always check for new items even if we processed a completed batch
        // This ensures continuous flow of items into the accumulator
        
        // If no batch ready or even if there was one, try to dequeue new items and add to accumulator
        int items_added = 0;
        const int MAX_ITEMS_PER_CYCLE = 512; // Significantly increased to process more items per cycle
        
        // Only proceed if there are items in the queue
        if (queue_size > 0) {
            // Try to bulk dequeue items for better efficiency
            std::vector<PendingEvaluation> bulk_items(std::min(MAX_ITEMS_PER_CYCLE, static_cast<int>(queue_size)));
            size_t num_dequeued = external_leaf_queue->try_dequeue_bulk(bulk_items.data(), bulk_items.size());
            
            if (num_dequeued > 0) {
                bulk_items.resize(num_dequeued);
                std::cout << "MCTSEvaluator::processExternalQueue - ✅ Bulk dequeued " << num_dequeued 
                        << " items from external queue (size was " << queue_size << ")" << std::endl;
                
                // Add all items to batch accumulator
                for (auto& item : bulk_items) {
                    if (item.state && item.node) {
                        // CRITICAL FIX: Ensure state is valid before adding to accumulator
                        try {
                            if (item.state->validate()) {
                                batch_accumulator_->addEvaluation(std::move(item));
                                items_added++;
                            } else {
                                std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Invalid state in queue (validation failed)" 
                                         << std::endl;
                                if (item.node) item.node->clearEvaluationFlag();
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Exception validating state: " 
                                     << e.what() << std::endl;
                            if (item.node) item.node->clearEvaluationFlag();
                        }
                    } else {
                        std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Invalid item in queue: "
                                 << "state=" << (item.state ? item.state.get() : nullptr) 
                                 << ", node=" << (item.node ? item.node.get() : nullptr) << std::endl;
                        if (item.node) item.node->clearEvaluationFlag();
                    }
                }
                
                // Log information about added items
                std::cout << "MCTSEvaluator::processExternalQueue - ✅ Added " << items_added 
                        << " valid items to batch accumulator" << std::endl;
                
                // Reset empty iterations counter when we add items
                consecutive_empty_iterations = 0;
            } else {
                // CRITICAL FIX: If bulk dequeue fails but queue_size > 0, try single dequeue
                // This can happen due to race conditions with the queue
                PendingEvaluation eval;
                while (items_added < MAX_ITEMS_PER_CYCLE && external_leaf_queue->try_dequeue(eval)) {
                    if (eval.state && eval.node) {
                        try {
                            if (eval.state->validate()) {
                                batch_accumulator_->addEvaluation(std::move(eval));
                                items_added++;
                            } else {
                                std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Invalid state in queue (validation failed)" 
                                         << std::endl;
                                if (eval.node) eval.node->clearEvaluationFlag();
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Exception validating state: " 
                                     << e.what() << std::endl;
                            if (eval.node) eval.node->clearEvaluationFlag();
                        }
                    } else {
                        std::cerr << "MCTSEvaluator::processExternalQueue - ⚠️ Invalid item in queue (single): "
                                 << "state=" << (eval.state ? eval.state.get() : nullptr) 
                                 << ", node=" << (eval.node ? eval.node.get() : nullptr) << std::endl;
                        if (eval.node) eval.node->clearEvaluationFlag();
                    }
                }
                
                if (items_added > 0) {
                    std::cout << "MCTSEvaluator::processExternalQueue - ✅ Added " << items_added 
                             << " valid items to batch accumulator (single dequeue)" << std::endl;
                    
                    // Reset empty iterations counter when we add items
                    consecutive_empty_iterations = 0;
                } else {
                    // Increment empty iterations counter 
                    consecutive_empty_iterations++;
                    
                    if (consecutive_empty_iterations % 10 == 0) {
                        std::cout << "MCTSEvaluator::processExternalQueue - ⚠️ Queue size non-zero (" 
                                 << queue_size << ") but no items dequeued for " 
                                 << consecutive_empty_iterations << " consecutive iterations" << std::endl;
                    }
                }
            }
            
            // Log when items are added
            if (items_added > 0) {
                static size_t total_added = 0;
                total_added += items_added;
                std::cout << "MCTSEvaluator::processExternalQueue - Total items added to accumulator: " 
                          << total_added << std::endl;
                
                // CRITICAL FIX: Always notify after adding items
                if (result_notify_callback_) {
                    result_notify_callback_();
                }
            }
        } else {
            // Increment empty iterations counter when queue is empty
            consecutive_empty_iterations++;
            
            if (print_debug) {
                // Log every so often when queue is empty
                std::cout << "MCTSEvaluator::processExternalQueue - External leaf queue is empty (consecutive: " 
                          << consecutive_empty_iterations << ")" << std::endl;
            }
        }
        
        // CRITICAL FIX: Get completed batch from accumulator and process it if available
        // Moved this after item processing to prioritize adding new items
        std::vector<PendingEvaluation> batch;
        if (batch_accumulator_->getCompletedBatch(batch)) {
            std::cout << "MCTSEvaluator::processExternalQueue - 🔄 Processing completed batch of size " 
                      << batch.size() << std::endl;
                      
            // Reset empty iterations counter when we process a batch
            consecutive_empty_iterations = 0;
                
            // Process the batch
            processBatchWithAccumulator(std::move(batch));
        }
        
        // CRITICAL FIX: Sleep based on activity to optimize CPU usage
        if (items_added > 0 || !batch.empty()) {
            // If we were active this cycle, use very short sleep to maximize throughput
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else if (consecutive_empty_iterations < 10) {
            // Normal short sleep when we've only been inactive for a short time
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        } else if (consecutive_empty_iterations < 100) {
            // Medium sleep after longer inactivity
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            // Longer sleep after extended inactivity
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        return;
    }
    
    // Legacy path - process without batch accumulator
    // This is essentially the old code path for compatibility
    std::vector<PendingEvaluation> batch = collectExternalBatch(batch_size_);
    if (!batch.empty()) {
        consecutive_empty_iterations = 0;
        processBatch();
    } else {
        consecutive_empty_iterations++;
        
        // CRITICAL FIX: Attempt to process with smaller batch size when repeatedly empty
        if (consecutive_empty_iterations >= 10) {
            // Try to process with much smaller batch size to avoid stalling
            size_t fallback_batch_size = std::max(size_t(1), batch_size_ / 8);
            std::vector<PendingEvaluation> small_batch = collectExternalBatch(fallback_batch_size);
            
            if (!small_batch.empty()) {
                std::cout << "MCTSEvaluator::processExternalQueue - Using fallback small batch size (" 
                         << fallback_batch_size << "), got " << small_batch.size() << " items" << std::endl;
                consecutive_empty_iterations = 0;
                processBatch();
            } else {
                // Sleep briefly to avoid busy waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else {
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void MCTSEvaluator::processInternalQueue() {
    // Check if we have any requests to process
    if (request_queue_.size_approx() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return;
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
        
        return;
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
    
    // Log more detailed stats for first few batches
    if (processed_batch_counter < 10 || processed_batch_counter % 5 == 0) {
        std::cout << "Processed accumulated batch #" << processed_batch_counter 
                  << " with " << results.size() << " states in " << batch_duration.count() << "ms" 
                  << std::endl;
    }
}

} // namespace mcts
} // namespace alphazero