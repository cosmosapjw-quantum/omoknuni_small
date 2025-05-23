#pragma once

#include "selfplay/self_play_manager.h"
#include "mcts/multi_instance_nn_manager.h"
#include "mcts/mcts_engine.h"
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace alphazero {
namespace selfplay {

// Game record structure for training data
struct GameRecord {
    std::string game_id;
    std::vector<std::shared_ptr<core::IGameState>> states;
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
};

/**
 * @brief Optimized self-play manager with true parallel MCTS engines
 * 
 * Key improvements:
 * - Each worker has independent neural network instance
 * - No shared state between workers
 * - Async game generation pipeline
 * - Efficient work distribution
 */
class ALPHAZERO_API OptimizedSelfPlayManager {
public:
    OptimizedSelfPlayManager(
        std::shared_ptr<mcts::MultiInstanceNNManager> nn_manager,
        const SelfPlaySettings& settings,
        const std::string& game_type = "gomoku");
    
    ~OptimizedSelfPlayManager();
    
    // Generate games using all available workers
    std::vector<GameRecord> generateGames(int num_games);
    
    // Start async game generation
    void startAsyncGeneration(int num_games);
    
    // Collect completed games (non-blocking)
    std::vector<GameRecord> collectCompletedGames();
    
    // Wait for all games to complete
    void waitForCompletion();
    
    // Get performance statistics
    void printStatistics() const;
    
private:
    struct WorkerContext {
        int worker_id;
        std::unique_ptr<mcts::MCTSEngine> engine;
        std::shared_ptr<nn::NeuralNetwork> neural_net;
        std::thread thread;
        
        // Performance metrics
        std::atomic<int> games_completed{0};
        std::atomic<double> total_game_time{0.0};
        std::atomic<int> total_moves{0};
    };
    
    // Worker function
    void workerLoop(WorkerContext* context);
    
    // Game generation
    GameRecord playOneGame(WorkerContext* context);
    
    // Work queue management
    std::queue<int> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> should_stop_{false};
    
    // Completed games
    std::vector<GameRecord> completed_games_;
    std::mutex completed_mutex_;
    
    // Workers
    std::vector<std::unique_ptr<WorkerContext>> workers_;
    
    // Configuration
    std::shared_ptr<mcts::MultiInstanceNNManager> nn_manager_;
    SelfPlaySettings settings_;
    std::string game_type_;  // Store the game type separately
    
    // Performance monitoring
    std::chrono::steady_clock::time_point start_time_;
    std::atomic<int> total_games_requested_{0};
    std::atomic<int> total_games_completed_{0};
};

} // namespace selfplay
} // namespace alphazero