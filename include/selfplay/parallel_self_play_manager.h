#pragma once

#include "selfplay/self_play_manager.h"
#include "core/export_macros.h"
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace alphazero {
namespace selfplay {

// Game record structure for parallel implementation
struct GameRecord {
    size_t game_id;
    std::string game_type;
    int board_size;
    std::vector<std::shared_ptr<core::IGameState>> states;
    std::vector<int> actions;
    std::vector<std::vector<float>> action_probabilities;
    float outcome;
};

class ALPHAZERO_API ParallelSelfPlayManager : public SelfPlayManager {
public:
    ParallelSelfPlayManager(std::shared_ptr<nn::NeuralNetwork> neural_net,
                           const SelfPlaySettings& settings);
    
    // Generate games in parallel using multiple MCTS engines
    std::vector<GameRecord> generateGamesParallel(
        const std::string& game_type,
        size_t num_games,
        int board_size = 15
    );

protected:
    // Override base class method
    int selectAction(const std::vector<float>& action_probs, float temperature);

private:
    struct WorkerTask {
        std::string game_type;
        size_t game_id;
        int board_size;
    };
    
    // Worker function for generating a single game
    GameRecord generateSingleGame(const WorkerTask& task, mcts::MCTSEngine* engine);
    
    // Worker thread function
    void workerThread(std::shared_ptr<nn::NeuralNetwork> neural_net);
    
    // Task queue management
    std::queue<WorkerTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool shutdown_ = false;
    
    // Result collection
    std::vector<GameRecord> completed_games_;
    std::mutex results_mutex_;
    
    // Worker pool
    std::vector<std::thread> workers_;
    size_t num_workers_;
};

} // namespace selfplay
} // namespace alphazero