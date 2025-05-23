#pragma once

#include "selfplay/self_play_manager.h"
#include "mcts/mcts_engine.h"
#include "mcts/evaluation_types.h"
#include <moodycamel/concurrentqueue.h>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

namespace alphazero {
namespace selfplay {

/**
 * Unified Parallel Self-Play Manager
 * 
 * This implementation uses a single shared evaluation queue and GPU inference
 * engine across all parallel games to maximize GPU utilization.
 */
class ALPHAZERO_API UnifiedParallelManager : public SelfPlayManager {
public:
    UnifiedParallelManager(std::shared_ptr<nn::NeuralNetwork> neural_net,
                          const SelfPlaySettings& settings);
    ~UnifiedParallelManager();
    
    // Generate games using unified batch collection
    std::vector<GameData> generateGamesUnified(
        core::GameType game_type,
        size_t num_games,
        int board_size = 15
    );

private:
    // Shared evaluation infrastructure
    struct SharedEvaluator {
        // Single queue for ALL games
        moodycamel::ConcurrentQueue<mcts::EvaluationRequest> request_queue;
        moodycamel::ConcurrentQueue<mcts::EvaluationResult> result_queue;
        
        // Evaluator thread
        std::thread evaluator_thread;
        std::atomic<bool> shutdown{false};
        
        // Neural network
        std::shared_ptr<nn::NeuralNetwork> neural_net;
        
        // Settings
        size_t batch_size;
        int batch_timeout_ms;
    };
    
    // Game worker data
    struct GameWorker {
        size_t worker_id;
        std::unique_ptr<mcts::MCTSEngine> engine;
        std::thread thread;
        GameData game_data;
        std::atomic<bool> completed{false};
    };
    
    // Unified evaluator thread function
    void unifiedEvaluatorThread(SharedEvaluator* evaluator);
    
    // Game worker thread function  
    void gameWorkerThread(GameWorker* worker, 
                         core::GameType game_type,
                         int board_size,
                         SharedEvaluator* evaluator);
    
    // Create custom MCTS engine that uses shared queues
    std::unique_ptr<mcts::MCTSEngine> createSharedQueueEngine(
        SharedEvaluator* evaluator);
    
    size_t num_parallel_games_;
};

} // namespace selfplay
} // namespace alphazero