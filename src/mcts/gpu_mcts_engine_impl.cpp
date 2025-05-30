#include "mcts/gpu_mcts_engine.h"
#include "utils/logger.h"
#include <chrono>

namespace alphazero {
namespace mcts {

GPUMCTSEngine::GPUMCTSEngine(std::shared_ptr<nn::NeuralNetwork> network,
                             const Config& config,
                             const GPUConfig& gpu_config)
    : MCTSEngine(network, config), gpu_config_(gpu_config) {
    LOG_MCTS_INFO("Initializing GPU-accelerated MCTS engine");
    LOG_MCTS_INFO("  GPU batch size: {}", gpu_config.gpu_batch_size);
    LOG_MCTS_INFO("  Use CUDA graph: {}", gpu_config.use_cuda_graph);
    LOG_MCTS_INFO("  Use half precision: {}", gpu_config.use_half_precision);
}

void GPUMCTSEngine::initialize() {
    // Call base class initialization first
    MCTSEngine::initialize();
    
    // Initialize GPU components
    initializeGPUComponents();
}

void GPUMCTSEngine::initializeGPUComponents() {
    if (!gpu_config_.use_gpu_evaluation) {
        LOG_MCTS_INFO("GPU evaluation disabled in config");
        return;
    }
    
    // Initialize GPU batch evaluator
    GPUBatchEvaluator::Config eval_config;
    eval_config.max_batch_size = gpu_config_.gpu_batch_size;
    eval_config.min_batch_size = gpu_config_.min_batch_size;
    eval_config.batch_timeout_ms = gpu_config_.batch_timeout_ms;
    eval_config.use_half_precision = gpu_config_.use_half_precision;
    eval_config.num_cuda_streams = 2;
    
    gpu_evaluator_ = std::make_unique<GPUBatchEvaluator>(
        network_, 
        config_.board_size, 
        eval_config
    );
    
    // Initialize CUDA graph manager if enabled
    if (gpu_config_.use_cuda_graph) {
        MultiGraphManager::Config graph_config;
        graph_config.batch_sizes = {32, 64, 128, gpu_config_.gpu_batch_size};
        graph_config.auto_capture = true;
        
        graph_manager_ = std::make_unique<MultiGraphManager>(graph_config);
        graph_manager_->initialize(*network_->getModel());
    }
    
    LOG_MCTS_INFO("GPU components initialized successfully");
}

void GPUMCTSEngine::processEvaluationQueue() {
    if (!gpu_config_.use_gpu_evaluation || !gpu_evaluator_) {
        // Fall back to CPU evaluation
        MCTSEngine::processEvaluationQueue();
        return;
    }
    
    // Collect pending evaluations into a batch
    std::vector<PendingEvaluation> batch;
    {
        std::unique_lock<std::mutex> lock(evaluation_mutex_);
        
        // Wait for evaluations or timeout
        auto timeout = std::chrono::microseconds(
            static_cast<int64_t>(gpu_config_.batch_timeout_ms * 1000)
        );
        
        evaluation_cv_.wait_for(lock, timeout, [this] {
            return !pending_evaluations_.empty() || 
                   pending_evaluations_.size() >= gpu_config_.gpu_batch_size ||
                   terminate_evaluation_thread_;
        });
        
        if (terminate_evaluation_thread_ && pending_evaluations_.empty()) {
            return;
        }
        
        // Collect up to gpu_batch_size evaluations
        size_t batch_size = std::min(pending_evaluations_.size(), 
                                    gpu_config_.gpu_batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(std::move(pending_evaluations_.front()));
            pending_evaluations_.pop();
        }
    }
    
    if (batch.empty()) {
        return;
    }
    
    // Process batch on GPU
    evaluateBatchGPU(batch);
}

void GPUMCTSEngine::evaluateBatchGPU(std::vector<PendingEvaluation>& batch) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert to GPU batch evaluator format
    std::vector<GPUBatchEvaluator::BatchRequest> gpu_batch;
    gpu_batch.reserve(batch.size());
    
    for (auto& eval : batch) {
        GPUBatchEvaluator::BatchRequest request;
        request.state = eval.state;
        request.node = eval.node;
        gpu_batch.push_back(request);
    }
    
    // Use CUDA graph if available and batch size matches
    if (graph_manager_) {
        // Create tensor from states
        auto input_tensor = createInputTensor(gpu_batch);
        auto [policy, value] = graph_manager_->execute(
            *network_->getModel(), 
            input_tensor, 
            gpu_batch.size()
        );
        
        // Process results
        auto policy_accessor = policy.accessor<float, 2>();
        auto value_accessor = value.accessor<float, 2>();
        
        for (size_t i = 0; i < gpu_batch.size(); ++i) {
            std::vector<float> policy_vec(config_.action_size);
            for (int a = 0; a < config_.action_size; ++a) {
                policy_vec[a] = policy_accessor[i][a];
            }
            
            auto& request = gpu_batch[i];
            request.policy_result = std::move(policy_vec);
            request.value_result = value_accessor[i][0];
            request.completed = true;
        }
    } else {
        // Use standard GPU batch evaluator
        gpu_evaluator_->processBatch(gpu_batch);
    }
    
    // Process results and update nodes
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& eval = batch[i];
        auto& gpu_request = gpu_batch[i];
        
        if (gpu_request.completed) {
            // Update node with results
            eval.node->clearEvaluationFlag();
            
            if (!eval.node->isExpanded()) {
                eval.node->expand();
                eval.node->setPriorProbabilities(gpu_request.policy_result);
            }
            
            // Backpropagate value
            eval.node->updateRecursive(gpu_request.value_result);
            
            // Set promise
            eval.promise.set_value(EvaluationResult{
                gpu_request.policy_result,
                gpu_request.value_result
            });
        }
    }
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double inference_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    gpu_evaluations_.fetch_add(batch.size());
    total_gpu_batch_size_.fetch_add(batch.size());
    total_gpu_inference_time_.fetch_add(inference_ms);
}

torch::Tensor GPUMCTSEngine::createInputTensor(
    const std::vector<GPUBatchEvaluator::BatchRequest>& batch) {
    
    // Create tensor for batch
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);
    
    // Assuming input shape is [batch, channels, height, width]
    int channels = 119;  // Game-specific
    torch::Tensor input = torch::zeros(
        {static_cast<int64_t>(batch.size()), channels, 
         config_.board_size, config_.board_size}, 
        options
    );
    
    // Fill tensor with state data
    for (size_t i = 0; i < batch.size(); ++i) {
        // Convert state to tensor representation
        // This is game-specific and should match your neural network input format
        auto state_tensor = batch[i].state->toTensor();
        input[i] = state_tensor;
    }
    
    return input;
}

GPUMCTSEngine::GPUStats GPUMCTSEngine::getGPUStats() const {
    GPUStats stats;
    stats.total_gpu_evaluations = gpu_evaluations_.load();
    stats.total_cpu_evaluations = cpu_evaluations_.load();
    
    if (stats.total_gpu_evaluations > 0) {
        stats.avg_gpu_batch_size = total_gpu_batch_size_.load() / 
                                  static_cast<double>(stats.total_gpu_evaluations);
        stats.avg_gpu_inference_ms = total_gpu_inference_time_.load() / 
                                    static_cast<double>(stats.total_gpu_evaluations);
    } else {
        stats.avg_gpu_batch_size = 0.0;
        stats.avg_gpu_inference_ms = 0.0;
    }
    
    double total_evals = stats.total_gpu_evaluations + stats.total_cpu_evaluations;
    if (total_evals > 0) {
        stats.gpu_utilization_percent = 100.0f * stats.total_gpu_evaluations / total_evals;
    } else {
        stats.gpu_utilization_percent = 0.0f;
    }
    
    return stats;
}

}  // namespace mcts
}  // namespace alphazero