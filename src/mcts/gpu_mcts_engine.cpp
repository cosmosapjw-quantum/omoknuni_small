#include "mcts/gpu_mcts_engine.h"
#include "utils/logger.h"

namespace alphazero {
namespace mcts {

GPUMCTSEngine::GPUMCTSEngine(
    std::shared_ptr<nn::NeuralNetwork> neural_net,
    const MCTSSettings& settings,
    const GPUConfig& gpu_config)
    : MCTSEngine(neural_net, settings),
      gpu_config_(gpu_config) {
    
    // Initialize GPU batch evaluator
    GPUBatchEvaluator::Config gpu_eval_config;
    gpu_eval_config.min_batch_size = gpu_config_.min_batch_size;
    gpu_eval_config.max_batch_size = gpu_config_.gpu_batch_size;
    gpu_eval_config.enable_cuda_graphs = gpu_config_.use_cuda_graph;
    gpu_eval_config.enable_adaptive_batching = true;
    
    gpu_evaluator_ = std::make_unique<GPUBatchEvaluator>(neural_net, gpu_eval_config);
    gpu_evaluator_->start();
    
    // LOG_SYSTEM_INFO("GPU MCTS Engine initialized with:");
    // LOG_SYSTEM_INFO("  - GPU evaluation: {}", gpu_config_.use_gpu_evaluation);
    // LOG_SYSTEM_INFO("  - CUDA graphs: {}", gpu_config_.use_cuda_graph);
    // LOG_SYSTEM_INFO("  - Min batch size: {}", gpu_config_.min_batch_size);
    // LOG_SYSTEM_INFO("  - GPU batch size: {}", gpu_config_.gpu_batch_size);
}

GPUMCTSEngine::~GPUMCTSEngine() {
    if (gpu_evaluator_) {
        gpu_evaluator_->stop();
    }
    
    // LOG_SYSTEM_INFO("GPU MCTS Engine stats:");
    // LOG_SYSTEM_INFO("  - GPU evaluations: {}", gpu_evaluations_.load());
    // LOG_SYSTEM_INFO("  - CPU evaluations: {}", cpu_evaluations_.load());
}

void GPUMCTSEngine::processGPUEvaluationQueue() {
    // Process evaluation queue using GPU batch evaluator
    if (gpu_evaluator_) {
        // This would interface with the existing MCTS evaluation pipeline
        // LOG_MCTS_DEBUG("Processing GPU evaluation queue");
    }
}

void GPUMCTSEngine::initializeGPU() {
    // Initialize GPU components if not already done
    if (!gpu_evaluator_ && gpu_config_.use_gpu_evaluation) {
        GPUBatchEvaluator::Config gpu_eval_config;
        gpu_eval_config.min_batch_size = gpu_config_.min_batch_size;
        gpu_eval_config.max_batch_size = gpu_config_.gpu_batch_size;
        gpu_eval_config.enable_cuda_graphs = gpu_config_.use_cuda_graph;
        
        gpu_evaluator_ = std::make_unique<GPUBatchEvaluator>(neural_network_, gpu_eval_config);
        gpu_evaluator_->start();
    }
}

void GPUMCTSEngine::evaluateBatchGPU(std::vector<PendingEvaluation>& batch) {
    // Convert PendingEvaluation to GPU batch format
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    for (auto& eval : batch) {
        states.push_back(eval.state->clone());
    }
    
    // Submit to GPU evaluator
    if (gpu_evaluator_) {
        auto future = gpu_evaluator_->submitBatch(std::move(states));
        auto outputs = future.get();
        
        // Apply results to nodes - PendingEvaluation doesn't store results directly
        for (size_t i = 0; i < batch.size() && i < outputs.size(); ++i) {
            if (batch[i].node) {
                // Apply neural network output to the node
                // Note: This would normally be handled by the MCTS engine's evaluation pipeline
                // The node expansion and value updates happen in the main MCTS loop
            }
        }
        
        gpu_evaluations_.fetch_add(batch.size());
    }
}

GPUMCTSEngine::GPUStats GPUMCTSEngine::getGPUStats() const {
    GPUStats stats;
    stats.total_gpu_evaluations = gpu_evaluations_.load();
    stats.total_cpu_evaluations = cpu_evaluations_.load();
    
    if (gpu_evaluator_) {
        // Get stats from GPU evaluator
        stats.avg_gpu_batch_size = static_cast<double>(gpu_config_.gpu_batch_size);
        stats.avg_gpu_inference_ms = 10.0; // Placeholder
        stats.gpu_utilization_percent = 75.0f; // Placeholder
    }
    
    return stats;
}

}  // namespace mcts
}  // namespace alphazero