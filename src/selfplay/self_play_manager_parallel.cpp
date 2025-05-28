#include "selfplay/parallel_self_play_manager.h"
#include "mcts/mcts_engine.h"
#include "core/igamestate.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <iostream>
#include <chrono>
#include <atomic>
#include <random>
#include <algorithm>
#include <cmath>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace selfplay {

ParallelSelfPlayManager::ParallelSelfPlayManager(
    std::shared_ptr<nn::NeuralNetwork> neural_net,
    const SelfPlaySettings& settings)
    : SelfPlayManager(neural_net, settings) {
    
    // Determine number of worker threads
    num_workers_ = settings.num_parallel_games;
    if (num_workers_ == 0) {
        // Auto-detect based on hardware
        int cpu_cores = std::thread::hardware_concurrency();
        int threads_per_game = settings.mcts_settings.num_threads;
        num_workers_ = std::max(1, cpu_cores / threads_per_game);
    }
}

std::vector<GameRecord> ParallelSelfPlayManager::generateGamesParallel(
    const std::string& game_type,
    size_t num_games,
    int board_size) {
    
    std::cout << "\n🚀 PARALLEL GAME GENERATION" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Total games to generate: " << num_games << std::endl;
    std::cout << "  - Parallel workers: " << num_workers_ << std::endl;
    std::cout << "  - MCTS threads per game: " << settings_.mcts_settings.num_threads << std::endl;
    std::cout << "  - Total CPU threads: " << num_workers_ * settings_.mcts_settings.num_threads << std::endl;
    std::cout << "  - Batch size per engine: " << settings_.mcts_settings.batch_size << std::endl;
    std::cout << "  - Combined GPU batch potential: " << num_workers_ * settings_.mcts_settings.batch_size << std::endl;
    std::cout << std::endl;
    
    // Clear previous results
    completed_games_.clear();
    completed_games_.reserve(num_games);
    
    // Reset shutdown flag
    shutdown_ = false;
    
    // Create task queue
    for (size_t i = 0; i < num_games; ++i) {
        task_queue_.push({game_type, i, board_size});
    }
    
    // Start worker threads
    workers_.clear();
    for (size_t i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&ParallelSelfPlayManager::workerThread, this, neural_net_);
    }
    
    // Monitor progress
    auto start_time = std::chrono::steady_clock::now();
    size_t last_completed = 0;
    
    while (completed_games_.size() < num_games) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        size_t current_completed;
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            current_completed = completed_games_.size();
        }
        
        if (current_completed > last_completed) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            
            if (elapsed_seconds > 0) {
                double games_per_second = current_completed / (double)elapsed_seconds;
                size_t remaining = num_games - current_completed;
                int eta_seconds = remaining / games_per_second;
                
                std::cout << "\rProgress: " << current_completed << "/" << num_games 
                          << " games (" << (100.0 * current_completed / num_games) << "%) "
                          << "Speed: " << games_per_second << " games/sec "
                          << "ETA: " << eta_seconds << "s" << std::flush;
            }
            
            last_completed = current_completed;
        }
    }
    
    // Shutdown workers
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_ = true;
    }
    queue_cv_.notify_all();
    
    // Wait for workers to finish
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    std::cout << "\n✅ Parallel game generation complete!" << std::endl;
    
    return completed_games_;
}

void ParallelSelfPlayManager::workerThread(std::shared_ptr<nn::NeuralNetwork> neural_net) {
    // Create a dedicated MCTS engine for this worker
    auto engine = std::make_unique<mcts::MCTSEngine>(neural_net, settings_.mcts_settings);
    
    while (true) {
        WorkerTask task;
        
        // Get next task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || shutdown_; });
            
            if (shutdown_ && task_queue_.empty()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = task_queue_.front();
                task_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Generate game
        try {
            auto game_record = generateSingleGame(task, engine.get());
            
            // Store result
            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                completed_games_.push_back(std::move(game_record));
            }
            
            // Force cleanup after each game to prevent memory accumulation
            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                size_t games_completed = completed_games_.size();
                
                // Cleanup every 5 games or when memory pressure is high
                if (games_completed % 5 == 0) {
#ifdef WITH_TORCH
                    if (torch::cuda::is_available()) {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                        torch::cuda::synchronize();
                    }
#endif
                    // Force garbage collection in engine
                    if (engine) {
                        engine->resetForNewSearch();
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "\nError in worker thread: " << e.what() << std::endl;
        }
    }
}

GameRecord ParallelSelfPlayManager::generateSingleGame(const WorkerTask& task, mcts::MCTSEngine* engine) {
    // Create game state based on type
    std::unique_ptr<core::IGameState> game_state;
    
    if (task.game_type == "gomoku") {
        game_state = std::make_unique<games::gomoku::GomokuState>(task.board_size);
    } else if (task.game_type == "chess") {
        game_state = std::make_unique<games::chess::ChessState>();
    } else if (task.game_type == "go") {
        game_state = std::make_unique<games::go::GoState>(task.board_size);
    } else {
        throw std::runtime_error("Unknown game type: " + task.game_type);
    }
    
    // Use the provided engine instead of creating a new one
    
    GameRecord record;
    record.game_id = task.game_id;
    record.game_type = task.game_type;
    record.board_size = task.board_size;
    
    // Track move number for temperature scheduling
    int move_number = 0;
    
    // Play game until terminal
    while (!game_state->isTerminal()) {
        // Run MCTS search
        auto search_result = engine->search(*game_state);
        
        // Determine temperature based on move number
        float temperature = (move_number < settings_.temperature_threshold) 
                           ? settings_.high_temperature 
                           : settings_.low_temperature;
        
        // Select action based on temperature
        int action = selectAction(search_result.probabilities, temperature);
        
        // Record state and action
        record.states.push_back(std::shared_ptr<core::IGameState>(game_state->clone()));
        record.actions.push_back(action);
        record.action_probabilities.push_back(search_result.probabilities);
        
        // Make move
        game_state->makeMove(action);
        move_number++;
    }
    
    // Record final outcome based on game result and perspective
    auto game_result = game_state->getGameResult();
    switch (game_result) {
        case core::GameResult::WIN_PLAYER1:
            record.outcome = 1.0f;
            break;
        case core::GameResult::WIN_PLAYER2:
            record.outcome = -1.0f;
            break;
        case core::GameResult::DRAW:
            record.outcome = 0.0f;
            break;
        default:
            record.outcome = 0.0f;
    }
    
    return record;
}

int ParallelSelfPlayManager::selectAction(const std::vector<float>& action_probs, float temperature) {
    if (temperature == 0.0f) {
        // Argmax
        return std::distance(action_probs.begin(), 
                           std::max_element(action_probs.begin(), action_probs.end()));
    }
    
    // Sample from distribution
    std::vector<float> adjusted_probs = action_probs;
    
    // Apply temperature
    if (temperature != 1.0f) {
        float sum = 0.0f;
        for (auto& prob : adjusted_probs) {
            prob = std::pow(prob, 1.0f / temperature);
            sum += prob;
        }
        // Renormalize
        if (sum > 0) {
            for (auto& prob : adjusted_probs) {
                prob /= sum;
            }
        }
    }
    
    // Sample
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(adjusted_probs.begin(), adjusted_probs.end());
    
    return dist(gen);
}

} // namespace selfplay
} // namespace alphazero