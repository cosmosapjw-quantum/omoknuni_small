#include "mcts/optimized_external_queue_processor.h"
#include "mcts/mcts_engine.h"
#include <algorithm>
#include <chrono>

namespace alphazero {
namespace mcts {

OptimizedExternalQueueProcessor::OptimizedExternalQueueProcessor(
    const OptimizedQueueConfig& config,
    std::shared_ptr<nn::NeuralNetwork> neural_network)
    : config_(config),
      neural_network_(neural_network),
      shutdown_(false),
      processing_thread_(&OptimizedExternalQueueProcessor::processingLoop, this) {
}

OptimizedExternalQueueProcessor::~OptimizedExternalQueueProcessor() {
    shutdown();
}

void OptimizedExternalQueueProcessor::shutdown() {
    if (!shutdown_.exchange(true)) {
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
}

void OptimizedExternalQueueProcessor::setQueues(
    moodycamel::ConcurrentQueue<PendingEvaluation>* input_queue,
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* output_queue) {
    input_queue_ = input_queue;
    output_queue_ = output_queue;
}

void OptimizedExternalQueueProcessor::processingLoop() {
    std::vector<PendingEvaluation> batch;
    batch.reserve(config_.batch_size);
    
    auto last_process_time = std::chrono::steady_clock::now();
    
    while (!shutdown_.load()) {
        if (!input_queue_ || !output_queue_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        collectBatch(batch);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_process_time;
        
        bool should_process = (!batch.empty() && batch.size() >= config_.min_batch_size) ||
                             (!batch.empty() && elapsed >= config_.max_wait_time);
        
        if (should_process) {
            processBatch(batch);
            batch.clear();
            last_process_time = now;
        } else if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    if (!batch.empty()) {
        processBatch(batch);
    }
}

void OptimizedExternalQueueProcessor::collectBatch(std::vector<PendingEvaluation>& batch) {
    PendingEvaluation eval;
    while (batch.size() < config_.batch_size && input_queue_->try_dequeue(eval)) {
        batch.push_back(std::move(eval));
    }
}

void OptimizedExternalQueueProcessor::processBatch(const std::vector<PendingEvaluation>& batch) {
    if (batch.empty() || !neural_network_) {
        return;
    }
    
    // Convert to states for neural network evaluation
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    for (const auto& eval : batch) {
        states.push_back(eval.state->clone());
    }
    
    try {
        // Perform batched neural network inference
        std::vector<NetworkOutput> results;
        
        // Use NeuralNetwork inference method if evaluateBatch is not available
        // This is a compatibility layer for the existing neural network implementations
        if (neural_network_->supportsEvaluateBatch()) {
            results = neural_network_->evaluateBatch(states);
        } else {
            // Legacy implementation - one by one evaluation
            results.reserve(states.size());
            for (auto& state : states) {
                results.push_back(neural_network_->evaluate(*state));
            }
        }
        
        // Send results back through output queue
        for (size_t i = 0; i < batch.size() && i < results.size(); ++i) {
            output_queue_->enqueue(std::make_pair(results[i], std::move(const_cast<PendingEvaluation&>(batch[i]))));
        }
        
        // Update statistics
        total_batches_processed_.fetch_add(1);
        total_evaluations_processed_.fetch_add(batch.size());
        
    } catch (const std::exception& e) {
        // Handle evaluation errors gracefully
        for (const auto& eval : batch) {
            NetworkOutput error_result;
            error_result.value = 0.0f;
            error_result.policy.resize(eval.state->getActionSpaceSize(), 1.0f / eval.state->getActionSpaceSize());
            
            output_queue_->enqueue(std::make_pair(error_result, std::move(const_cast<PendingEvaluation&>(eval))));
        }
    }
}

OptimizedQueueStats OptimizedExternalQueueProcessor::getStats() const {
    OptimizedQueueStats stats;
    stats.total_batches_processed = total_batches_processed_.load();
    stats.total_evaluations_processed = total_evaluations_processed_.load();
    
    if (stats.total_batches_processed > 0) {
        stats.average_batch_size = static_cast<float>(stats.total_evaluations_processed) / stats.total_batches_processed;
    }
    
    return stats;
}

} // namespace mcts
} // namespace alphazero
