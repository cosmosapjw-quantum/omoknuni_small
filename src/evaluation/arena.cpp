#include "evaluation/arena.h"
#include "mcts/mcts_engine.h"
#include "core/game_export.h"
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

namespace alphazero {
namespace evaluation {

using namespace alphazero::mcts;
using namespace alphazero::core;

class Arena::Impl {
public:
    explicit Impl(const ArenaConfig& config) : config_(config) {
        // Validate configuration
        if (config_.num_games <= 0) {
            throw std::invalid_argument("Number of games must be positive");
        }
        if (config_.num_parallel_games <= 0) {
            config_.num_parallel_games = 1;
        }
    }
    
    GameResult playGame(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model,
        bool champion_plays_first
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create game state based on game type
        std::unique_ptr<core::IGameState> game_state;
        switch (config_.game_type) {
            case core::GameType::GOMOKU:
                game_state = std::make_unique<games::gomoku::GomokuState>(config_.board_size);
                break;
            case core::GameType::CHESS:
                game_state = std::make_unique<games::chess::ChessState>();
                break;
            case core::GameType::GO:
                game_state = std::make_unique<games::go::GoState>(config_.board_size);
                break;
            default:
                throw std::runtime_error("Unsupported game type for arena");
        }
        
        // Configure game-specific settings
        configureGameState(game_state.get());
        
        // Create MCTS engines for both players
        MCTSSettings mcts_settings;
        mcts_settings.num_simulations = config_.num_simulations;
        mcts_settings.num_threads = config_.num_threads;
        mcts_settings.batch_size = config_.batch_size;
        mcts_settings.exploration_constant = config_.exploration_constant;
        mcts_settings.temperature = config_.temperature;
        mcts_settings.add_dirichlet_noise = config_.add_dirichlet_noise;
        mcts_settings.virtual_loss = 3.0f;
        // mcts_settings.use_value_temperature = false; // Not used in current settings
        
        // Create engines with respective models
        auto champion_engine = std::make_unique<MCTSEngine>(champion_model, mcts_settings);
        auto contender_engine = std::make_unique<MCTSEngine>(contender_model, mcts_settings);
        
        // Track game progress
        std::vector<int> moves;
        int total_nodes = 0;
        int move_count = 0;
        int max_moves = config_.max_moves_per_game;
        if (max_moves == 0) {
            max_moves = config_.board_size * config_.board_size;
        }
        
        // Play the game
        while (!game_state->isTerminal() && move_count < max_moves) {
            // Determine current player's engine
            MCTSEngine* current_engine = nullptr;
            if (champion_plays_first) {
                current_engine = (move_count % 2 == 0) ? champion_engine.get() : contender_engine.get();
            } else {
                current_engine = (move_count % 2 == 0) ? contender_engine.get() : champion_engine.get();
            }
            
            // Run MCTS search
            auto search_result = current_engine->search(*game_state);
            
            // Get best move
            int best_move = search_result.action;
            if (best_move < 0) {
                // No valid move found
                break;
            }
            
            // Make the move
            game_state->makeMove(best_move);
            moves.push_back(best_move);
            total_nodes += search_result.stats.total_nodes;
            move_count++;
        }
        
        // Determine winner
        int winner = 0;
        if (game_state->isTerminal()) {
            auto game_result = game_state->getGameResult();
            if (game_result == core::GameResult::WIN_PLAYER1) {
                // Map to actual player based on who played first
                winner = champion_plays_first ? 1 : 2;
            } else if (game_result == core::GameResult::WIN_PLAYER2) {
                winner = champion_plays_first ? 2 : 1;
            }
            // GameResult::DRAW or ONGOING results in winner = 0
        }
        
        // Calculate duration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        GameResult result;
        result.winner = winner;
        result.moves = std::move(moves);
        result.duration_seconds = duration.count() / 1000.0;
        result.total_nodes_searched = total_nodes;
        
        return result;
    }
    
    MatchResult playMatch(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model
    ) {
        return playMatchWithProgress(champion_model, contender_model, nullptr);
    }
    
    MatchResult playMatchWithProgress(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model,
        std::function<void(int, int)> progress_callback
    ) {
        MatchResult result;
        
        // Play games with alternating colors
        for (int i = 0; i < config_.num_games; ++i) {
            bool champion_plays_first = true;
            if (config_.swap_colors) {
                champion_plays_first = (i % 2 == 0);
            }
            
            // Play single game
            auto game_result = playGame(champion_model, contender_model, champion_plays_first);
            
            // Update statistics
            if (game_result.winner == 1) {
                if (champion_plays_first) {
                    result.champion_wins++;
                } else {
                    result.contender_wins++;
                }
            } else if (game_result.winner == 2) {
                if (champion_plays_first) {
                    result.contender_wins++;
                } else {
                    result.champion_wins++;
                }
            } else {
                result.draws++;
            }
            
            // Call progress callback
            if (progress_callback) {
                progress_callback(i + 1, config_.num_games);
            }
            
            // Log progress
            if ((i + 1) % 10 == 0) {
                // LOG_SYSTEM_INFO("Arena progress: {}/{} games. Champion: {}, Contender: {}, Draws: {}",
                //                (i + 1), config_.num_games, result.champion_wins,
                //                result.contender_wins, result.draws);
            }
        }
        
        // LOG_SYSTEM_INFO("Arena match complete. Champion win rate: {:.1f}%, Contender win rate: {:.1f}%",
        //                result.champion_win_rate() * 100, result.contender_win_rate() * 100);
        
        return result;
    }
    
    MatchResult evaluateModels(
        const std::string& champion_path,
        const std::string& contender_path
    ) {
        // Load models
        // Use a simple approach - create models and load from disk
        std::shared_ptr<nn::NeuralNetwork> champion_model;
        std::shared_ptr<nn::NeuralNetwork> contender_model;
        
        // Create ResNet models (or use factory if available)
#ifdef WITH_TORCH
        champion_model = std::make_shared<nn::ResNetModel>(config_.board_size, 256, 20);
        champion_model->load(champion_path);
        
        contender_model = std::make_shared<nn::ResNetModel>(config_.board_size, 256, 20);
        contender_model->load(contender_path);
#else
        throw std::runtime_error("Neural network models require torch support");
#endif
        
        // Play match
        return playMatch(champion_model, contender_model);
    }
    
private:
    void configureGameState(IGameState* state) {
        // Configure game-specific settings
        if (config_.game_type == GameType::GOMOKU) {
            // Gomoku-specific configuration if needed
        } else if (config_.game_type == GameType::CHESS) {
            // Chess-specific configuration if needed
        } else if (config_.game_type == GameType::GO) {
            // Go-specific configuration if needed
        }
    }
    
    ArenaConfig config_;
};

// Arena public interface implementation

Arena::Arena(const ArenaConfig& config) 
    : impl_(std::make_unique<Impl>(config)) {
}

Arena::~Arena() = default;

GameResult Arena::playGame(
    std::shared_ptr<nn::NeuralNetwork> champion_model,
    std::shared_ptr<nn::NeuralNetwork> contender_model,
    bool champion_plays_first
) {
    return impl_->playGame(champion_model, contender_model, champion_plays_first);
}

MatchResult Arena::playMatch(
    std::shared_ptr<nn::NeuralNetwork> champion_model,
    std::shared_ptr<nn::NeuralNetwork> contender_model
) {
    return impl_->playMatch(champion_model, contender_model);
}

MatchResult Arena::playMatchWithProgress(
    std::shared_ptr<nn::NeuralNetwork> champion_model,
    std::shared_ptr<nn::NeuralNetwork> contender_model,
    std::function<void(int, int)> progress_callback
) {
    return impl_->playMatchWithProgress(champion_model, contender_model, progress_callback);
}

MatchResult Arena::evaluateModels(
    const std::string& champion_path,
    const std::string& contender_path
) {
    return impl_->evaluateModels(champion_path, contender_path);
}

// Parallel games implementation

std::vector<GameResult> playParallelGames(
    const ArenaConfig& config,
    std::shared_ptr<nn::NeuralNetwork> champion_model,
    std::shared_ptr<nn::NeuralNetwork> contender_model,
    int num_games,
    int num_workers
) {
    std::vector<GameResult> results;
    results.reserve(num_games);
    
    // Mutex for thread-safe result collection
    std::mutex results_mutex;
    
    // Atomic counter for game assignment
    std::atomic<int> game_counter(0);
    
    // Worker function
    auto worker = [&]() {
        Arena arena(config);
        
        while (true) {
            int game_id = game_counter.fetch_add(1);
            if (game_id >= num_games) {
                break;
            }
            
            // Alternate who plays first
            bool champion_plays_first = (game_id % 2 == 0);
            
            // Play game
            auto result = arena.playGame(champion_model, contender_model, champion_plays_first);
            
            // Store result
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                results.push_back(result);
            }
        }
    };
    
    // Launch worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    return results;
}

} // namespace evaluation
} // namespace alphazero