// include/selfplay/self_play_manager.h
#ifndef ALPHAZERO_SELFPLAY_MANAGER_H
#define ALPHAZERO_SELFPLAY_MANAGER_H

#include <vector>
#include <memory>
#include <string>
#include <random>
#include "core/igamestate.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network.h"
#include "core/export_macros.h"
#include <moodycamel/concurrentqueue.h>
#include "mcts/evaluation_types.h"

namespace alphazero {
namespace selfplay {

/**
 * @brief Structure for game data
 */
struct ALPHAZERO_API GameData {
    // Moves played in the game
    std::vector<int> moves;
    
    // MCTS policy distributions
    std::vector<std::vector<float>> policies;
    
    // Game result (0=draw, 1=player1, 2=player2)
    int winner;
    
    // Game type
    core::GameType game_type;
    
    // Board size (if applicable)
    int board_size;
    
    // Total time taken for the game (ms)
    int64_t total_time_ms;
    
    // Game ID
    std::string game_id;
};

/**
 * @brief Game-specific configuration structure
 */
struct ALPHAZERO_API GameConfig {
    // Gomoku settings
    bool gomoku_use_renju = true;
    bool gomoku_use_omok = false;
    bool gomoku_use_pro_long_opening = true;
    
    // Chess settings
    bool chess_use_chess960 = false;
    
    // Go settings
    float go_komi = 7.5f;
    bool go_chinese_rules = true;
    bool go_enforce_superko = true;
};

/**
 * @brief Settings for self-play
 */
struct ALPHAZERO_API SelfPlaySettings {
    // MCTS settings
    mcts::MCTSSettings mcts_settings;
    
    // Number of parallel games
    int num_parallel_games = 1;
    
    // IMPORTANT: num_mcts_engines is deprecated, use root parallelization instead
    // Configure mcts_settings.use_root_parallelization and mcts_settings.num_root_workers
    int num_mcts_engines = 1; // Deprecated - kept for backward compatibility
    
    // Maximum number of moves before forcing a draw
    int max_moves = 0;
    
    // Number of start positions (for chess, using Fischer Random)
    int num_start_positions = 1;
    
    // Temperature threshold (move number after which temperature is reduced)
    int temperature_threshold = 30;
    
    // High temperature (for early moves)
    float high_temperature = 1.0f;
    
    // Low temperature (for later moves)
    float low_temperature = 0.1f;
    
    // Add Dirichlet noise to root node
    bool add_dirichlet_noise = true;
    
    // Random seed (-1 for time-based)
    int64_t random_seed = -1;
    
    // Game-specific configuration
    GameConfig game_config;
};

/**
 * @brief Manager for self-play
 */
class ALPHAZERO_API SelfPlayManager {
public:
    /**
     * @brief Constructor
     * 
     * @param neural_net Neural network for MCTS
     * @param settings Self-play settings
     */
    SelfPlayManager(std::shared_ptr<nn::NeuralNetwork> neural_net, 
                   const SelfPlaySettings& settings = SelfPlaySettings());
    
    /**
     * @brief Destructor
     */
    ~SelfPlayManager();
    
    /**
     * @brief Generate self-play games
     * 
     * @param game_type Game type
     * @param num_games Number of games to generate
     * @param board_size Board size (if applicable)
     * @return Vector of game data
     */
    std::vector<GameData> generateGames(core::GameType game_type, 
                                        int num_games, 
                                        int board_size = 0);
    
    /**
     * @brief Save game data to files
     * 
     * @param games Vector of game data
     * @param output_dir Output directory
     * @param format Format ("json" or "binary")
     */
    void saveGames(const std::vector<GameData>& games, 
                  const std::string& output_dir, 
                  const std::string& format = "json");
    
    /**
     * @brief Load game data from files
     * 
     * @param input_dir Input directory
     * @param format Format ("json" or "binary")
     * @return Vector of game data
     */
    static std::vector<GameData> loadGames(const std::string& input_dir, 
                                          const std::string& format = "json");
    
    /**
     * @brief Convert game data to training examples
     * 
     * @param games Vector of game data
     * @return Pair of (states, targets) where states are tensor representations
     *         and targets are (policy, value) pairs
     */
    static std::pair<std::vector<std::vector<std::vector<std::vector<float>>>>, 
                   std::pair<std::vector<std::vector<float>>, std::vector<float>>> 
    convertToTrainingExamples(const std::vector<GameData>& games);
    
    /**
     * @brief Get a reference to the self-play settings
     * 
     * @return Self-play settings
     */
    const SelfPlaySettings& getSettings() const;
    
    /**
     * @brief Update self-play settings
     * 
     * @param settings New settings
     */
    void updateSettings(const SelfPlaySettings& settings);
    
private:
    /**
     * @brief Generate a single game
     * 
     * @param game_id Game ID
     * @return Game data
     */
    GameData generateGame(mcts::MCTSEngine& engine, core::GameType game_type, int board_size, const std::string& game_id);
    
    // Removed parallel gameWorker - Now using sequential generation only
    
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
    
protected:
    // Neural network
    std::shared_ptr<nn::NeuralNetwork> neural_net_;
    
    // Self-play settings
    SelfPlaySettings settings_;
    
private:
    
    // Single MCTS engine used for all games
    std::vector<std::unique_ptr<mcts::MCTSEngine>> engines_;
    int num_engines_resolved_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Sequential game counter (no longer using atomic for parallel operations)
    int game_counter_;
    
    // NOTE: Legacy shared queues - no longer used with new optimized architecture
    // Each MCTSEngine now has its own UnifiedInferenceServer + BurstCoordinator
    moodycamel::ConcurrentQueue<mcts::PendingEvaluation> shared_leaf_queue_;
    moodycamel::ConcurrentQueue<std::pair<mcts::NetworkOutput, mcts::PendingEvaluation>> shared_result_queue_;
};

} // namespace selfplay
} // namespace alphazero

#endif // ALPHAZERO_SELFPLAY_MANAGER_H