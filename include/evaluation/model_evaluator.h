// include/evaluation/model_evaluator.h
#ifndef ALPHAZERO_EVALUATION_MODEL_EVALUATOR_H
#define ALPHAZERO_EVALUATION_MODEL_EVALUATOR_H

#include <memory>
#include <vector>
#include <string>
#include <random>
#include "nn/neural_network.h"
#include "mcts/mcts_engine.h"
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace evaluation {

/**
 * @brief Result of a match between two models
 */
struct ALPHAZERO_API MatchResult {
    // Match ID
    std::string match_id;
    
    // Game type
    core::GameType game_type;
    
    // Board size
    int board_size;
    
    // Moves played
    std::vector<int> moves;
    
    // Result: 0 = draw, 1 = first model wins, 2 = second model wins
    int result;
    
    // First model played as Player 1
    bool first_model_as_player1;
    
    // Total time taken (ms)
    int64_t total_time_ms;
};

/**
 * @brief Tournament result between two models
 */
struct ALPHAZERO_API TournamentResult {
    // Number of wins for first model
    int wins_first = 0;
    
    // Number of wins for second model
    int wins_second = 0;
    
    // Number of draws
    int draws = 0;
    
    // Total games played
    int total_games = 0;
    
    // ELO difference estimate
    float elo_diff = 0.0f;
    
    // All match results
    std::vector<MatchResult> matches;
};

/**
 * @brief Settings for model evaluation
 */
struct ALPHAZERO_API EvaluationSettings {
    // MCTS settings for first model
    mcts::MCTSSettings mcts_settings_first;
    
    // MCTS settings for second model
    mcts::MCTSSettings mcts_settings_second;
    
    // Number of games to play in the tournament
    int num_games = 100;
    
    // Number of parallel games to play
    int num_parallel_games = 1;
    
    // Maximum number of moves before forcing a draw
    int max_moves = 0;
    
    // Number of start positions (for chess, using Fischer Random)
    int num_start_positions = 1;
    
    // Random seed (-1 for time-based)
    int64_t random_seed = -1;
};

/**
 * @brief Model evaluator for tournament play
 */
class ALPHAZERO_API ModelEvaluator {
public:
    /**
     * @brief Constructor
     * 
     * @param model_first First model
     * @param model_second Second model
     * @param settings Evaluation settings
     */
    ModelEvaluator(std::shared_ptr<nn::NeuralNetwork> model_first,
                  std::shared_ptr<nn::NeuralNetwork> model_second,
                  const EvaluationSettings& settings = EvaluationSettings());

    // Add move constructor and assignment operator declarations
    ModelEvaluator(ModelEvaluator&& other) noexcept;
    ModelEvaluator& operator=(ModelEvaluator&& other) noexcept;

    // Delete copy constructor and copy assignment operator
    ModelEvaluator(const ModelEvaluator&) = delete;
    ModelEvaluator& operator=(const ModelEvaluator&) = delete;
    
    /**
     * @brief Run a tournament between the two models
     * 
     * @param game_type Game type
     * @param board_size Board size (if applicable)
     * @return Tournament result
     */
    TournamentResult runTournament(core::GameType game_type, int board_size = 0);
    
    /**
     * @brief Play a single match between the two models
     * 
     * @param game_type Game type
     * @param board_size Board size (if applicable)
     * @param position_id Start position ID
     * @param first_model_as_player1 Whether first model plays as Player 1
     * @return Match result
     */
    MatchResult playMatch(core::GameType game_type, int board_size, 
                          int position_id, bool first_model_as_player1);
    
    /**
     * @brief Calculate ELO difference from win/loss/draw counts
     * 
     * @param wins Number of wins
     * @param losses Number of losses
     * @param draws Number of draws
     * @return ELO difference
     */
    static float calculateEloDiff(int wins, int losses, int draws);
    
    /**
     * @brief Get the evaluation settings
     * 
     * @return Evaluation settings
     */
    const EvaluationSettings& getSettings() const;
    
    /**
     * @brief Update evaluation settings
     * 
     * @param settings New settings
     */
    void updateSettings(const EvaluationSettings& settings);
    
private:
    // Models
    std::shared_ptr<nn::NeuralNetwork> model_first_;
    std::shared_ptr<nn::NeuralNetwork> model_second_;
    
    // Evaluation settings
    EvaluationSettings settings_;
    
    // MCTS engines for first model
    std::vector<std::unique_ptr<mcts::MCTSEngine>> engines_first_;
    
    // MCTS engines for second model
    std::vector<std::unique_ptr<mcts::MCTSEngine>> engines_second_;
    
    // Random number generator
    std::mt19937 rng_;
    
    /**
     * @brief Create a game state
     * 
     * @param game_type Game type
     * @param board_size Board size
     * @param position_id Position ID (for varied starting positions)
     * @return Unique pointer to game state
     */
    std::unique_ptr<core::IGameState> createGame(core::GameType game_type, 
                                               int board_size, 
                                               int position_id);
};

} // namespace evaluation
} // namespace alphazero

#endif // ALPHAZERO_EVALUATION_MODEL_EVALUATOR_H