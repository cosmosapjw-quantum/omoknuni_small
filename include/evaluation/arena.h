#ifndef ALPHAZERO_EVALUATION_ARENA_H
#define ALPHAZERO_EVALUATION_ARENA_H

#include <memory>
#include <vector>
#include <future>
#include <functional>
#include "core/igamestate.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network.h"
#include "core/export_macros.h"

namespace alphazero {
namespace evaluation {

/**
 * @brief Result of a single arena game
 */
struct ALPHAZERO_API GameResult {
    int winner;  // 1 for player1, 2 for player2, 0 for draw
    std::vector<int> moves;
    double duration_seconds;
    int total_nodes_searched;
};

/**
 * @brief Aggregated results of an arena match
 */
struct ALPHAZERO_API MatchResult {
    int champion_wins = 0;
    int contender_wins = 0;
    int draws = 0;
    
    double champion_win_rate() const {
        int total = champion_wins + contender_wins + draws;
        return total > 0 ? static_cast<double>(champion_wins) / total : 0.0;
    }
    
    double contender_win_rate() const {
        int total = champion_wins + contender_wins + draws;
        return total > 0 ? static_cast<double>(contender_wins) / total : 0.0;
    }
    
    double draw_rate() const {
        int total = champion_wins + contender_wins + draws;
        return total > 0 ? static_cast<double>(draws) / total : 0.0;
    }
    
    int total_games() const {
        return champion_wins + contender_wins + draws;
    }
    
    bool contender_is_better(double threshold = 0.55) const {
        return contender_win_rate() >= threshold;
    }
};

/**
 * @brief Configuration for arena matches
 */
struct ALPHAZERO_API ArenaConfig {
    // Game settings
    core::GameType game_type = core::GameType::GOMOKU;
    int board_size = 15;
    
    // MCTS settings
    int num_simulations = 400;
    int num_threads = 4;
    int batch_size = 32;
    float exploration_constant = 1.5f;
    float temperature = 0.1f;  // Low temperature for deterministic play
    bool add_dirichlet_noise = false;  // No noise in evaluation
    
    // Match settings
    int num_games = 100;
    int num_parallel_games = 8;
    bool swap_colors = true;  // Play both colors for fairness
    int max_moves_per_game = 0;  // 0 = auto-calculate
    
    // Evaluation threshold
    double win_rate_threshold = 0.55;  // Contender needs 55% win rate
};

/**
 * @brief Arena class for model evaluation
 * 
 * Handles playing matches between two models to determine
 * which one is stronger.
 */
class ALPHAZERO_API Arena {
public:
    /**
     * @brief Constructor
     * 
     * @param config Arena configuration
     */
    explicit Arena(const ArenaConfig& config);
    
    /**
     * @brief Destructor
     */
    ~Arena();
    
    /**
     * @brief Play a single game between two models
     * 
     * @param champion_model Neural network for the champion
     * @param contender_model Neural network for the contender
     * @param champion_plays_first Whether champion plays first
     * @return Game result
     */
    GameResult playGame(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model,
        bool champion_plays_first
    );
    
    /**
     * @brief Play a match between two models
     * 
     * @param champion_model Neural network for the champion
     * @param contender_model Neural network for the contender
     * @return Match results
     */
    MatchResult playMatch(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model
    );
    
    /**
     * @brief Play a match with progress callback
     * 
     * @param champion_model Neural network for the champion
     * @param contender_model Neural network for the contender
     * @param progress_callback Called after each game with (games_played, total_games)
     * @return Match results
     */
    MatchResult playMatchWithProgress(
        std::shared_ptr<nn::NeuralNetwork> champion_model,
        std::shared_ptr<nn::NeuralNetwork> contender_model,
        std::function<void(int, int)> progress_callback
    );
    
    /**
     * @brief Evaluate models by playing a match
     * 
     * @param champion_path Path to champion model
     * @param contender_path Path to contender model
     * @return Match results
     */
    MatchResult evaluateModels(
        const std::string& champion_path,
        const std::string& contender_path
    );
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Play arena games in parallel
 * 
 * @param config Arena configuration
 * @param champion_model Champion model
 * @param contender_model Contender model
 * @param num_games Number of games to play
 * @param num_workers Number of parallel workers
 * @return Vector of game results
 */
ALPHAZERO_API std::vector<GameResult> playParallelGames(
    const ArenaConfig& config,
    std::shared_ptr<nn::NeuralNetwork> champion_model,
    std::shared_ptr<nn::NeuralNetwork> contender_model,
    int num_games,
    int num_workers
);

} // namespace evaluation
} // namespace alphazero

#endif // ALPHAZERO_EVALUATION_ARENA_H