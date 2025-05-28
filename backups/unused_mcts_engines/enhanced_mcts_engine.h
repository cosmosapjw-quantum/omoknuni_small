// include/mcts/enhanced_mcts_engine.h
#ifndef ALPHAZERO_ENHANCED_MCTS_ENGINE_H
#define ALPHAZERO_ENHANCED_MCTS_ENGINE_H

#include "mcts/mcts_engine.h"
#include "mcts/multi_instance_nn_manager.h"
#include "mcts/gpu_memory_pool.h"
#include "mcts/dynamic_batch_manager.h"
#include "mcts/advanced_transposition_table.h"
#include "core/export_macros.h"
#include <yaml-cpp/yaml.h>

namespace alphazero {
namespace mcts {

/**
 * @brief Enhanced MCTS Engine with all performance optimizations
 * 
 * This engine combines:
 * - GPU memory pooling for zero-copy operations
 * - Dynamic batch sizing based on queue depth
 * - Advanced transposition table with modern algorithms
 * - Multi-instance neural networks
 * - Thread-local memory management
 */
class ALPHAZERO_API EnhancedMCTSEngine : public MCTSEngine {
public:
    struct EnhancedSettings : public MCTSSettings {
        // GPU memory pool settings
        GPUMemoryPool::PoolConfig gpu_pool_config;
        
        // Dynamic batching settings
        DynamicBatchManager::Config batch_manager_config;
        
        // Advanced transposition table settings
        AdvancedTranspositionTable::Config tt_config;
        
        // Multi-instance NN settings
        int nn_instances_per_engine = 1;
        bool use_gpu_memory_pool = true;
        bool use_dynamic_batching = true;
        bool use_advanced_tt = true;
    };

    EnhancedMCTSEngine(
        const EnhancedSettings& settings,
        std::shared_ptr<nn::NeuralNetwork> network,
        core::GameType game_type
    );

    virtual ~EnhancedMCTSEngine();

    // Enhanced search method
    SearchResult search(const core::IGameState& root_state);

    // Get enhanced statistics
    struct EnhancedStats {
        MCTSStats base_stats;
        GPUMemoryPool::PoolStats gpu_pool_stats;
        DynamicBatchManager::BatchingStats batch_stats;
        AdvancedTranspositionTable::Stats tt_stats;
        float gpu_utilization;
        float memory_efficiency;
    };
    EnhancedStats getEnhancedStats() const;

protected:
    // Enhanced components
    std::shared_ptr<MultiInstanceNNManager> nn_manager_;
    std::shared_ptr<GPUMemoryPool> gpu_pool_;
    std::shared_ptr<DynamicBatchManager> batch_manager_;
    std::shared_ptr<AdvancedTranspositionTable> advanced_tt_;
    
    EnhancedSettings enhanced_settings_;

private:
    // Enhanced batch evaluation with GPU pooling
    struct PooledBatch {
        std::vector<torch::Tensor> input_tensors;
        std::vector<MCTSNode*> nodes;
        std::vector<std::shared_ptr<GPUMemoryPool::MemoryBlock>> memory_blocks;
        cudaStream_t stream;
    };
    
    void evaluateBatchWithPool(PooledBatch& batch);
    
    // Dynamic batch collection
    void collectDynamicBatch(std::vector<MCTSNode*>& batch, int optimal_size);
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_stats_update_;
    void updatePerformanceMetrics();
};

/**
 * @brief Factory for creating enhanced MCTS engines
 */
class ALPHAZERO_API EnhancedMCTSEngineFactory {
public:
    static std::unique_ptr<EnhancedMCTSEngine> create(
        const YAML::Node& config,
        std::shared_ptr<nn::NeuralNetwork> network,
        core::GameType game_type
    );
    
    static EnhancedMCTSEngine::EnhancedSettings parseConfig(const YAML::Node& config);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_ENHANCED_MCTS_ENGINE_H