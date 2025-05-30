#pragma once

#include "selfplay/self_play_manager.h"
#include "mcts/shared_evaluation_server.h"

namespace alphazero {
namespace selfplay {

/**
 * GPU-optimized self-play manager that uses a shared evaluation server
 * 
 * This manager creates a single evaluation server that all MCTS engines
 * share, enabling cross-game batching for better GPU utilization.
 */
class GPUSelfPlayManager : public SelfPlayManager {
public:
    struct GPUConfig {
        bool use_shared_evaluation = true;
        size_t evaluation_batch_size = 512;
        size_t evaluation_threads = 2;
        float evaluation_timeout_ms = 10.0f;
    };
    
    GPUSelfPlayManager(std::shared_ptr<nn::NeuralNetwork> neural_net,
                      const SelfPlaySettings& settings,
                      const GPUConfig& gpu_config = GPUConfig());
    
    virtual ~GPUSelfPlayManager();
    
    // Override to initialize shared evaluation server
    virtual void initialize() override;
    
    // Override to shutdown evaluation server
    virtual void shutdown() override;
    
    // Get evaluation server statistics
    mcts::SharedEvaluationServer::Stats getEvaluationStats() const;
    
protected:
    GPUConfig gpu_config_;
    
    // Override engine creation to use shared evaluation
    virtual std::unique_ptr<mcts::MCTSEngine> createEngine(
        const mcts::MCTSSettings& settings) override;
};

}  // namespace selfplay
}  // namespace alphazero