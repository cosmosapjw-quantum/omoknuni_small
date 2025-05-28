#ifndef ALPHAZERO_CLI_ALPHAZERO_PIPELINE_H
#define ALPHAZERO_CLI_ALPHAZERO_PIPELINE_H

#include <string>
#include <vector>
#include <memory>
#include "core/game_export.h"
#include "nn/neural_network.h"
#include "selfplay/self_play_manager.h"
#include "core/export_macros.h"

namespace alphazero {
namespace cli {

/**
 * @brief Configuration for AlphaZero pipeline
 */
struct ALPHAZERO_API AlphaZeroPipelineConfig {
    // General settings
    core::GameType game_type = core::GameType::GOMOKU;
    int board_size = 15;
    int input_channels = 17;  // Match Gomoku's enhanced tensor representation (17 planes)
    int policy_size = 0;  // Will be derived from board size if 0
    std::string model_dir = "models";
    std::string data_dir = "data";
    std::string log_dir = "logs";
    std::string network_type = "resnet";  // "resnet" or "ddw_randwire"
    bool use_gpu = true;
    int num_iterations = 10;
    
    // Neural network settings
    int num_res_blocks = 6;  // Reduced from 19 to fit on 8GB VRAM
    int num_filters = 64;    // Reduced from 256 to fit on 8GB VRAM
    
    // DDW-RandWire-ResNet specific settings
    struct DDWConfig {
        int num_blocks = 20;
        int channels = 128;
        int num_nodes = 32;
        std::string graph_method = "watts_strogatz";  // "watts_strogatz", "erdos_renyi", "barabasi_albert"
        double ws_p = 0.75;          // Watts-Strogatz rewiring probability
        double er_edge_prob = 0.1;   // Erdos-Renyi edge probability
        int ba_m = 5;                // Barabasi-Albert edges per new node
        int ws_k = 4;                // Watts-Strogatz initial neighbors
        bool use_dynamic_routing = true;
        int seed = -1;               // -1 for random seed
    } ddw_config;
    
    // Self-play settings
    int self_play_num_games = 500;
    int self_play_num_parallel_games = 8;
    // Note: self_play_num_mcts_engines is deprecated - root parallelization is used instead
    int self_play_num_mcts_engines = 8; // Deprecated - kept for backward compatibility
    int self_play_max_moves = 0;  // 0 means auto-calculate based on board size
    int self_play_temperature_threshold = 30;
    float self_play_high_temperature = 1.0f;
    float self_play_low_temperature = 0.1f;
    std::string self_play_output_format = "json";  // "json" or "binary"
    
    // MCTS settings
    int mcts_num_simulations = 800;
    int mcts_threads_per_engine = 8;
    
    // Batch parameters - AGGRESSIVE GPU OPTIMIZATION
    int mcts_batch_size = 1024;  // Large batches for RTX 3060 Ti
    int mcts_min_viable_batch_size = 512;  // Minimum batch size threshold (50% of optimal)
    int mcts_min_fallback_batch_size = 256;  // Minimum acceptable batch size (25% of optimal)
    int mcts_max_collection_batch_size = 128; // Collect many leaves at once
    int mcts_batch_timeout_ms = 50;  // Longer timeout for larger batches
    int mcts_additional_wait_ms = 20;  // More wait time for batch formation
    
    // Other MCTS parameters
    float mcts_exploration_constant = 1.5f;
    float mcts_temperature = 1.0f;
    bool mcts_add_dirichlet_noise = true;
    float mcts_dirichlet_alpha = 0.3f;
    float mcts_dirichlet_epsilon = 0.25f;
    
    // Training settings
    int train_epochs = 20;
    int train_batch_size = 128;  // Reduced from 1024 to lower memory usage
    int train_num_workers = 2;   // Reduced from 4 to lower memory usage
    float train_learning_rate = 0.001f;
    float train_weight_decay = 0.0001f;
    int train_lr_step_size = 10;
    float train_lr_gamma = 0.1f;
    
    // Arena/evaluation settings
    bool enable_evaluation = true;
    int arena_num_games = 50;
    int arena_num_parallel_games = 8;
    // Note: arena_num_mcts_engines is deprecated - root parallelization is used instead
    int arena_num_mcts_engines = 8; // Deprecated - kept for backward compatibility
    int arena_num_threads = 4;
    int arena_num_simulations = 400;
    float arena_temperature = 0.1f;
    float arena_win_rate_threshold = 0.55f;  // New model must win 55% to become champion
};

/**
 * @brief Main AlphaZero pipeline class
 * 
 * This class handles the complete AlphaZero self-play/train/evaluate loop.
 */
class ALPHAZERO_API AlphaZeroPipeline {
public:
    /**
     * @brief Constructor
     * 
     * @param config Pipeline configuration
     */
    explicit AlphaZeroPipeline(const AlphaZeroPipelineConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AlphaZeroPipeline();
    
    /**
     * @brief Run the complete AlphaZero pipeline
     */
    void run();
    
private:
    // Configuration
    AlphaZeroPipelineConfig config_;
    
    // State
    int current_iteration_;
    std::shared_ptr<nn::NeuralNetwork> current_model_;
    
    // Pipeline phases
    void initializeLogging();
    void createDirectories();
    void initializeNeuralNetwork();
    void initializeNewNeuralNetwork();
    std::string createIterationDirectory(int iteration);
    std::vector<selfplay::GameData> runSelfPlay(const std::string& iteration_dir);
    float trainNeuralNetwork(const std::vector<selfplay::GameData>& games, const std::string& iteration_dir);
    bool evaluateNewModel(const std::string& iteration_dir);
    std::vector<selfplay::GameData> playArenaGames(
        selfplay::SelfPlayManager& player1_manager,
        selfplay::SelfPlayManager& player2_manager,
        int num_games,
        const std::string& output_dir
    );
    void saveEvaluationResults(
        const std::string& file_path,
        int champion_wins,
        int contender_wins,
        int draws,
        int total_games
    );
    void updateBestModel();
    void saveBestModel();
    void logIterationSummary(int iteration, int num_games, float train_loss);
};

/**
 * @brief Parse a YAML configuration file into AlphaZeroPipelineConfig
 * 
 * @param config_path Path to YAML configuration file
 * @return AlphaZeroPipelineConfig
 */
ALPHAZERO_API AlphaZeroPipelineConfig parseConfigFile(const std::string& config_path);

/**
 * @brief Run AlphaZero pipeline from configuration file
 * 
 * @param config_path Path to configuration file
 * @return 0 on success, non-zero on error
 */
ALPHAZERO_API int runAlphaZeroPipelineFromConfig(const std::string& config_path);

/**
 * @brief CLI command handler for AlphaZero pipeline
 * 
 * @param args Command line arguments
 * @return 0 on success, non-zero on error
 */
ALPHAZERO_API int runPipelineCommand(const std::vector<std::string>& args);

} // namespace cli
} // namespace alphazero

#endif // ALPHAZERO_CLI_ALPHAZERO_PIPELINE_H