#pragma once

#include "mcts/mcts_engine.h"
#include "mcts/gpu_batch_evaluator.h"
#include "mcts/gpu_tree_storage.h"
#include "mcts/cuda_graph_capture.h"
#include "mcts/gpu_node.h"
#include <memory>

namespace alphazero {
namespace mcts {

/**
 * GPU-enhanced MCTS engine that uses GPU for tree operations
 * 
 * This engine integrates:
 * - GPU batch evaluation with CUDA streams
 * - GPU tree storage for parallel tree operations  
 * - CUDA graph capture for optimized inference
 * - GPU node metadata management
 */
class GPUEnhancedMCTSEngine : public MCTSEngine {
public:
    struct GPUConfig {
        // Enable GPU components
        bool use_gpu_tree_storage;
        bool use_gpu_tree_traversal;
        bool use_cuda_graph;
        bool use_gpu_node_selection;
        
        // GPU tree storage settings
        size_t gpu_tree_max_nodes;
        size_t gpu_tree_max_actions;
        bool gpu_tree_half_precision;
        
        // CUDA graph settings
        std::vector<size_t> cuda_graph_batch_sizes;
        
        // GPU batch evaluation settings
        size_t gpu_batch_size;
        size_t gpu_min_batch_size;
        float gpu_batch_timeout_ms;
        size_t gpu_cuda_streams;
        
        GPUConfig() : use_gpu_tree_storage(true), use_gpu_tree_traversal(true),
                      use_cuda_graph(true), use_gpu_node_selection(true),
                      gpu_tree_max_nodes(100000), gpu_tree_max_actions(362),
                      gpu_tree_half_precision(false),
                      cuda_graph_batch_sizes({64, 128, 256, 512}),
                      gpu_batch_size(512), gpu_min_batch_size(256),
                      gpu_batch_timeout_ms(5.0f), gpu_cuda_streams(2) {}
    };
    
    GPUEnhancedMCTSEngine(std::shared_ptr<nn::NeuralNetwork> network,
                          const MCTSSettings& settings,
                          const GPUConfig& gpu_config = GPUConfig());
    
    virtual ~GPUEnhancedMCTSEngine();
    
    // GPU-enhanced search method
    SearchResult searchGPU(const core::IGameState& state);
    
    // Get GPU utilization statistics
    struct GPUStats {
        // Tree storage stats
        size_t gpu_tree_nodes;
        float gpu_tree_memory_mb;
        float tree_compression_ratio;
        
        // Evaluation stats
        size_t total_gpu_evaluations;
        double avg_gpu_batch_size;
        double avg_gpu_inference_ms;
        
        // Selection stats
        size_t gpu_selections;
        size_t cpu_selections;
        float gpu_selection_ratio;
        
        // CUDA graph stats
        size_t graph_hits;
        size_t graph_misses;
        float graph_speedup;
    };
    GPUStats getGPUStats() const;
    
protected:
    // GPU-specific implementation methods
    void runSimulations(std::shared_ptr<MCTSNode> root, int num_simulations);
    int selectBestMove(std::shared_ptr<MCTSNode> root) const;
    std::vector<int> getChildVisitCounts(std::shared_ptr<MCTSNode> root) const;
    std::vector<float> getPolicyDistribution(
        std::shared_ptr<MCTSNode> root, float temperature) const;
    float evaluateNode(std::shared_ptr<MCTSNode> node);
    
    // GPU-specific methods
    virtual void runSimulationsGPU(std::shared_ptr<MCTSNode> root, int num_simulations);
    virtual void expandNodeGPU(std::shared_ptr<MCTSNode> node);
    virtual void backpropagateGPU(const std::vector<std::shared_ptr<MCTSNode>>& path, float value);
    
private:
    GPUConfig gpu_config_;
    
    // GPU components
    std::unique_ptr<GPUBatchEvaluator> gpu_evaluator_;
    std::unique_ptr<GPUTreeStorage> gpu_tree_storage_;
    std::unique_ptr<MultiGraphManager> graph_manager_;
    
    // Statistics
    std::atomic<size_t> gpu_evaluations_{0};
    std::atomic<size_t> gpu_selections_{0};
    std::atomic<size_t> cpu_selections_{0};
    
    // Initialize GPU components
    void initializeGPUComponents();
    
    // GPU tree operations
    void uploadTreeToGPU(std::shared_ptr<MCTSNode> root);
    void downloadTreeFromGPU();
    
    // Batch GPU selection across multiple nodes
    std::vector<MCTSNode*> selectBestChildrenGPU(
        const std::vector<std::shared_ptr<MCTSNode>>& nodes,
        float exploration_constant);
        
    // Helper to create input tensor from game state
    torch::Tensor createInputTensor(const core::IGameState& state);
};

}  // namespace mcts
}  // namespace alphazero