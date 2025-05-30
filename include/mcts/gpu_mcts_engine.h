#pragma once

#include "mcts/mcts_engine.h"
#include "mcts/gpu_batch_evaluator.h"
#include "mcts/cuda_graph_capture.h"
#include "mcts/gpu_node.h"
#include <memory>

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH

/**
 * GPU-accelerated MCTS engine that extends the base MCTSEngine
 * 
 * This engine uses GPU for neural network evaluation while keeping
 * the tree search on CPU for compatibility.
 */
class GPUMCTSEngine : public MCTSEngine {
public:
    struct GPUConfig {
        bool use_gpu_evaluation = true;
        bool use_cuda_graph = true;
        size_t gpu_batch_size = 256;
        size_t min_batch_size = 64;
        float batch_timeout_ms = 5.0f;
        bool use_half_precision = false;
        
        // Future options (not implemented in Phase 1)
        bool use_gpu_tree_storage = false;
        bool use_gpu_tree_traversal = false;
    };
    
    // Constructor with GPU config
    GPUMCTSEngine(std::shared_ptr<nn::NeuralNetwork> network,
                  const MCTSSettings& settings,
                  const GPUConfig& gpu_config);
    
    virtual ~GPUMCTSEngine();
    
    // Get GPU statistics
    struct GPUStats {
        size_t total_gpu_evaluations;
        size_t total_cpu_evaluations;
        double avg_gpu_batch_size;
        double avg_gpu_inference_ms;
        float gpu_utilization_percent;
    };
    GPUStats getGPUStats() const;
    
protected:
    // Additional GPU-specific methods
    void processGPUEvaluationQueue();
    void initializeGPU();
    
private:
    GPUConfig gpu_config_;
    
    // GPU components
    std::unique_ptr<GPUBatchEvaluator> gpu_evaluator_;
    std::unique_ptr<MultiGraphManager> graph_manager_;
    
    // Statistics
    std::atomic<size_t> gpu_evaluations_{0};
    std::atomic<size_t> cpu_evaluations_{0};
    std::atomic<size_t> total_gpu_batch_size_{0};
    std::atomic<size_t> total_gpu_inference_time_ms_{0};
    
    // Helper methods
    void initializeGPUComponents();
    void evaluateBatchGPU(std::vector<PendingEvaluation>& batch);
};

#else // !WITH_TORCH
// Dummy class when torch is not available
class GPUMCTSEngine : public MCTSEngine {
public:
    struct GPUConfig {};
    GPUMCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, 
                  const MCTSSettings& settings, 
                  const GPUConfig& = {}) 
        : MCTSEngine(neural_net, settings) {}
};
#endif // WITH_TORCH

}  // namespace mcts
}  // namespace alphazero