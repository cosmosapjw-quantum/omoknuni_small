#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <iostream>

namespace alphazero {
namespace mcts {

// Pipeline parallelism implementation for MCTSEvaluator
// This allows overlapping batch collection and inference

// Add an evaluation to the pipeline buffer
void MCTSEvaluator::addToPipelineBatch(PendingEvaluation&& eval) {
    // Add to the pipeline buffer using lock-free queue
    // The pipeline buffer will automatically swap buffers when the 
    // target batch size is reached
    pipeline_buffer_.add(std::move(eval));
}

// Swap active and inactive buffers, making the collected batch available for processing
void MCTSEvaluator::swapPipelineBuffers() {
    // Force a buffer swap in the pipeline buffer
    // This is a no-op if the collection buffer is empty
    pipeline_buffer_.forceBufferSwap();
}

// Get the active batch for processing
std::vector<PendingEvaluation> MCTSEvaluator::getActivePipelineBatch() {
    std::vector<PendingEvaluation> result;
    
    // Wait for the processing buffer to be ready with timeout
    auto timeout = batch_params_.max_wait_time;
    
    // Try to get items from the pipeline buffer
    pipeline_buffer_.getItems(result, timeout);
    
    // Check if shutdown was requested
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        result.clear(); // Return empty batch on shutdown
    }
    
    return result;
}

// Add a batch processing method that uses pipeline parallelism
bool MCTSEvaluator::processPipelineBatch() {
    // Only process if not shutting down
    if (shutdown_flag_.load(std::memory_order_acquire)) {
        return false;
    }
    
    // Try to collect a batch directly (no intermediate vector)
    std::vector<PendingEvaluation> batch;
    
    // Reserve space for optimal batch size
    batch.reserve(batch_params_.optimal_batch_size);
    
    // Use getItems which handles waiting and timeout internally
    bool got_items = pipeline_buffer_.getItems(batch, batch_params_.max_wait_time);
    if (!got_items || batch.empty()) {
        return false;
    }
    
    // Process the batch
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    // Track valid items and indices
    std::vector<size_t> valid_indices;
    valid_indices.reserve(batch.size());
    
    // Extract states for neural network evaluation - loop optimized with reserve
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
        return false;
    }
    
    // Run neural network inference
    std::vector<NetworkOutput> results;
    std::chrono::steady_clock::time_point inference_start_time;
    long long inference_duration_ms = 0;
    try {
        // Add timing metrics for inference
        inference_start_time = std::chrono::steady_clock::now();
        
        results = inference_fn_(states);
        
        auto inference_end_time = std::chrono::steady_clock::now();
        inference_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end_time - inference_start_time).count();
            
        // Update inference timing metrics (this was cumulative_batch_time_ms_ before, make sure it's appropriate here)
        // For pipeline, this specific batch's duration might be more relevant to log or use for adaptive logic for the pipeline itself.
        // The global cumulative_batch_time_ms_ will be updated when results are processed by the engine or if we add it here too.
        // For now, let's assume this duration is for this specific pipeline batch.

    } catch (const std::exception& e) {
        std::cout << "Exception during pipeline inference: " << e.what() << std::endl;
        // If inference fails, we should still mark nodes as not evaluating and potentially return default/error results.
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            size_t orig_idx = valid_indices[i];
            auto& eval = batch[orig_idx];
            if (eval.node) {
                eval.node->clearEvaluationFlag();
            }
        }
        return false;
    }
    
    // Check for valid results
    if (results.size() != valid_indices.size()) {
        std::cout << "Mismatch in pipeline result and valid indices counts: " << results.size() << " vs " << valid_indices.size() << std::endl;
        // Clear evaluation flags for nodes that were part of this failed batch processing
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            size_t orig_idx = valid_indices[i];
            auto& eval = batch[orig_idx];
            if (eval.node) {
                eval.node->clearEvaluationFlag();
            }
        }
        return false;
    }
    
    // Process the results
    if (use_external_queues_) {
        if (result_queue_ptr_) {
            auto* external_result_queue = static_cast<moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>*>(result_queue_ptr_);
            std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_pairs_external;
            result_pairs_external.reserve(results.size());

            for (size_t i = 0; i < results.size(); ++i) {
                size_t orig_idx = valid_indices[i];
                result_pairs_external.emplace_back(std::move(results[i]), std::move(batch[orig_idx]));
            }

            if (!result_pairs_external.empty()){
                external_result_queue->enqueue_bulk(std::make_move_iterator(result_pairs_external.begin()), result_pairs_external.size());
                if (result_notify_callback_) {
                    result_notify_callback_();
                }
            }
        } else {
             std::cout << "Error: use_external_queues_ is true but result_queue_ptr_ is null in processPipelineBatch" << std::endl;
        }
        // Update global/evaluator stats for batches processed via pipeline using external queues
        total_batches_.fetch_add(1, std::memory_order_relaxed);
        total_evaluations_.fetch_add(results.size(), std::memory_order_relaxed);
        cumulative_batch_size_.fetch_add(results.size(), std::memory_order_relaxed);
        // Note: cumulative_batch_time_ms_ was updated with inference_duration_ms from inferenceWorkerLoop.
        // To be consistent, we should also sum the duration_ms of this pipeline batch here.
        // auto current_batch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - batch_collection_start_time).count(); // This would be overall time
        // For now, let's use the inference duration of this specific batch, similar to inferenceWorkerLoop's update.
        // auto current_inference_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_placeholder).count(); // Placeholder for actual inference start time of this batch
        // The `start_time` used for `duration_ms` above is the correct one for this specific inference batch.
        // So, we use that `duration_ms`.
        // auto actual_inference_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_placeholder).count();
        try {
            cumulative_batch_time_ms_.fetch_add(inference_duration_ms, std::memory_order_relaxed);

        } catch(const std::exception& e) {
            std::cout << "Exception calculating duration for stats in pipeline: " << e.what() << std::endl;
        }

    } else {
        // Original internal processing logic (fulfilling promises directly or updating nodes)
        for (size_t i = 0; i < results.size(); ++i) {
            size_t orig_idx = valid_indices[i];
            auto& output = results[i];
            auto& eval = batch[orig_idx];
            
            // Apply results to the node with null checks
            if (eval.node) {
                try {
                    eval.node->setPriorProbabilities(output.policy);
                    
                    // Back-propagate value through the tree
                    float value = output.value;
                    for (auto& node : eval.path) {
                        if (node) {
                            node->update(value);
                            value = -value; // Flip value for alternating perspective
                        }
                    }
                    
                    // Clear evaluation flag
                    eval.node->clearEvaluationFlag();
                } catch (const std::exception& e) {
                    std::cout << "Exception during result processing: " << e.what() << std::endl;
                    // Continue processing other results
                }
            }
        }
        
        // Update statistics
        total_batches_.fetch_add(1, std::memory_order_relaxed);
        total_evaluations_.fetch_add(results.size(), std::memory_order_relaxed);
        cumulative_batch_size_.fetch_add(results.size(), std::memory_order_relaxed);
        
        // If we got a full batch, increment full_batches counter
        if (batch.size() >= batch_params_.optimal_batch_size) {
            full_batches_.fetch_add(1, std::memory_order_relaxed);
        } else {
            partial_batches_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    return true;
}

// A background thread function for pipeline processing
void MCTSEvaluator::pipelineProcessorLoop() {
    std::cout << "Starting pipeline processor thread" << std::endl;
    
    // Initialize the pipeline buffer with the optimal batch size
    pipeline_buffer_.setTargetBatchSize(batch_params_.optimal_batch_size);
    
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
        bool processed = processPipelineBatch();
        
        if (!processed) {
            // If no batch was processed, wait briefly but use less CPU
            // Adaptive backoff - longer wait time when fewer items are coming in
            auto size_approx = pipeline_buffer_.size_approx();
            if (size_approx == 0) {
                // If no items at all, wait longer
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } else if (size_approx < batch_params_.minimum_viable_batch_size / 2) {
                // If few items, wait a moderate amount
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } else {
                // If getting close to batch size, wait just a little
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    // Shutdown the pipeline buffer to release any waiting threads
    pipeline_buffer_.shutdown();
    
    std::cout << "Pipeline processor thread exiting" << std::endl;
}

} // namespace mcts
} // namespace alphazero