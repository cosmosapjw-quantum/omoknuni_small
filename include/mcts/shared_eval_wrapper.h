#pragma once

#include "mcts/mcts_engine.h"
#include "mcts/shared_evaluation_server.h"
#include <memory>

namespace alphazero {
namespace mcts {

/**
 * Wrapper that redirects MCTS engine evaluations to a shared server
 * 
 * This allows existing MCTS engines to use shared evaluation without
 * modifying their implementation.
 */
class SharedEvalWrapper {
public:
    /**
     * Initialize the shared evaluation system
     * Must be called before creating any wrapped engines
     */
    static void initialize(
        std::shared_ptr<nn::NeuralNetwork> network,
        size_t batch_size = 512,
        size_t min_batch_size = 256,
        float timeout_ms = 10.0f,
        size_t num_threads = 2
    );
    
    /**
     * Shutdown the shared evaluation system
     * Should be called after all engines are destroyed
     */
    static void shutdown();
    
    /**
     * Check if shared evaluation is enabled
     */
    static bool isEnabled();
    
    /**
     * Get evaluation server statistics
     */
    static SharedEvaluationServer::Stats getStats();
    
    /**
     * Create an inference function that uses the shared server
     */
    static MCTSEngine::InferenceFunction createSharedInferenceFunction();
    
private:
    static std::atomic<bool> enabled_;
    static std::mutex mutex_;
};

}  // namespace mcts
}  // namespace alphazero