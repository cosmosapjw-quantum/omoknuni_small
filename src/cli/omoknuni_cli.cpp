// src/cli/omoknuni_cli.cpp
#include "cli/cli_manager.h"
#include "core/game_export.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <dlfcn.h>
#include <signal.h>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <map>
#include <chrono>
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "evaluation/model_evaluator.h"
#include "cli/alphazero_pipeline.h"
#include <optional>

// Configure PyTorch CUDA initialization
#define PYTORCH_NO_CUDA_INIT_OVERRIDE 0
#define USE_TORCH 1
#define C10_CUDA_DRIVER_INIT 1

// Watchdog timer variables
volatile sig_atomic_t g_watchdog_timer_expired = 0;
constexpr int WATCHDOG_TIMEOUT_SECONDS = 60; // 60 seconds timeout

// Signal handler for watchdog timer
void watchdog_handler(int /*sig*/) {
    std::cerr << "Timeout reached! Terminating program." << std::endl;
    // Cleanly exit, possibly logging state if feasible
    g_watchdog_timer_expired = 1;
}

// Setup watchdog timer
void setup_watchdog() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = watchdog_handler;
    sigaction(SIGALRM, &sa, NULL);
    alarm(WATCHDOG_TIMEOUT_SECONDS);
}

// Reset watchdog timer
void reset_watchdog() {
    alarm(WATCHDOG_TIMEOUT_SECONDS);
}

// Cancel watchdog timer
void cancel_watchdog() {
    alarm(0);
}

// Check if a specific library is available
bool check_library(const char* libname) {
    void* handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        return false;
    }
    dlclose(handle);
    return true;
}

// Helper function to safely get an integer config value
int getIntConfigValue(const std::map<std::string, std::string>& config, const std::string& key, int defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        try {
            return std::stoi(it->second);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid value for " << key << ": '" << it->second << "'. Using default value: " << defaultValue << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: Value for " << key << " out of range: '" << it->second << "'. Using default value: " << defaultValue << std::endl;
        }
    } else {
        std::cerr << "Warning: Config key '" << key << "' not found. Using default value: " << defaultValue << std::endl;
    }
    return defaultValue;
}

// Helper function to safely get a float config value
float getFloatConfigValue(const std::map<std::string, std::string>& config, const std::string& key, float defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        try {
            return std::stof(it->second);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid value for " << key << ": '" << it->second << "'. Using default value: " << defaultValue << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: Value for " << key << " out of range: '" << it->second << "'. Using default value: " << defaultValue << std::endl;
        }
    } else {
        std::cerr << "Warning: Config key '" << key << "' not found. Using default value: " << defaultValue << std::endl;
    }
    return defaultValue;
}

// Helper function to safely get a boolean config value
bool getBoolConfigValue(const std::map<std::string, std::string>& config, const std::string& key, bool defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        std::string val_str = it->second;
        std::transform(val_str.begin(), val_str.end(), val_str.begin(), ::tolower);
        if (val_str == "true" || val_str == "1") {
            return true;
        } else if (val_str == "false" || val_str == "0") {
            return false;
        } else {
            std::cerr << "Warning: Invalid boolean value for " << key << ": '" << it->second << "'. Using default value: " << defaultValue << std::endl;
        }
    } else {
        std::cerr << "Warning: Config key '" << key << "' not found. Using default value: " << defaultValue << std::endl;
    }
    return defaultValue;
}

// Helper function to safely get a string config value
std::string getStringConfigValue(const std::map<std::string, std::string>& config, const std::string& key, const std::string& defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        return it->second;
    } else {
        std::cerr << "Warning: Config key '" << key << "' not found. Using default value: '" << defaultValue << "'" << std::endl;
    }
    return defaultValue;
}

int main(int argc, char** argv) {
    // Setup watchdog timer
    setup_watchdog();

    // Try to find alphazero library
    std::vector<std::string> lib_paths = {
        "./build/lib/Release/libalphazero.so",
        "./build/lib/Debug/libalphazero.so",
        "../build/lib/Release/libalphazero.so",
        "../build/lib/Debug/libalphazero.so",
        "/usr/local/lib/libalphazero.so"
    };
    
    bool has_alphazero = false;
    for (const auto& path : lib_paths) {
        if (check_library(path.c_str())) {
            has_alphazero = true;
            break;
        }
    }

    // Display basic application information
    std::cout << "Omoknuni CLI - AlphaZero Implementation" << std::endl;
    std::cout << "Usage: omoknuni_cli <command> [options]" << std::endl;
    std::cout << "Available commands: self-play, train, eval, play" << std::endl;
    std::cout << "For more information, run 'omoknuni_cli <command> --help'" << std::endl;
    
    if (!has_alphazero) {
        std::cerr << "Warning: AlphaZero library not found. Some functionality may be limited." << std::endl;
    }

    try {
        // Reset watchdog timer
        reset_watchdog();

        // Register proper game implementations
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::GOMOKU,
            []() { return std::make_unique<alphazero::games::gomoku::GomokuState>(); }
        );
        
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::CHESS,
            []() { return std::make_unique<alphazero::games::chess::ChessState>(); }
        );
        
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::GO,
            []() { return std::make_unique<alphazero::games::go::GoState>(); }
        );

        // Create CLI manager
        alphazero::cli::CLIManager cli;

        // Add command handlers
        cli.addCommand("self-play", "Generate self-play games for training",
            [](const std::vector<std::string>& args) {
                std::cout << "Executing self-play..." << std::endl;
                
                // Parse arguments for configuration
                std::string config_path;
                for (size_t i = 0; i < args.size(); i++) {
                    if (args[i] == "--config" && i + 1 < args.size()) {
                        config_path = args[i + 1];
                        break;
                    }
                }
                
                if (config_path.empty()) {
                    std::cerr << "Error: Config file path not specified. Use --config <path>" << std::endl;
                    return 1;
                }
                
                try {
                    // Load configuration
                    std::ifstream config_file(config_path);
                    if (!config_file.is_open()) {
                        std::cerr << "Error: Could not open config file: " << config_path << std::endl;
                        return 1;
                    }
                    
                    // Parse YAML configuration
                    std::string line;
                    std::map<std::string, std::string> config;
                    std::string current_section;
                    
                    while (std::getline(config_file, line)) {
                        // Skip comments and empty lines
                        if (line.empty() || line[0] == '#') {
                            continue;
                        }
                        
                        // Check if this is a section header
                        if (line[line.size() - 1] == ':') {
                            current_section = line.substr(0, line.size() - 1);
                            continue;
                        }
                        
                        // Parse key-value pairs
                        size_t colon_pos = line.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string key = line.substr(0, colon_pos);
                            std::string value = line.substr(colon_pos + 1);
                            
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            
                            // Store with section prefix if we're in a section
                            if (!current_section.empty()) {
                                config[current_section + "." + key] = value;
                            } else {
                                config[key] = value;
                            }
                        }
                    }
                    
                    // Extract relevant configuration values
                    std::string game_type_str = getStringConfigValue(config, "game_type", "gomoku");
                    int board_size = getIntConfigValue(config, "board_size", 15);
                    std::string model_path = getStringConfigValue(config, "model_path", "models/model.pt");
                    int num_simulations = getIntConfigValue(config, "mcts_num_simulations", 800);
                    int num_threads = getIntConfigValue(config, "mcts_num_threads", 4);
                    int batch_size = getIntConfigValue(config, "mcts_batch_size", 16);
                    int max_collection_batch_size = getIntConfigValue(config, "mcts_max_collection_batch_size", 32);
                    float exploration_constant = getFloatConfigValue(config, "mcts_exploration_constant", 1.5f);
                    int virtual_loss = getIntConfigValue(config, "mcts_virtual_loss", 3);
                    bool add_dirichlet_noise = getBoolConfigValue(config, "mcts_add_dirichlet_noise", true);
                    float dirichlet_alpha = getFloatConfigValue(config, "mcts_dirichlet_alpha", 0.3f);
                    float dirichlet_epsilon = getFloatConfigValue(config, "mcts_dirichlet_epsilon", 0.25f);
                    float temperature = getFloatConfigValue(config, "mcts_temperature", 1.0f);
                    int batch_timeout_ms = getIntConfigValue(config, "mcts_batch_timeout_ms", 1000);
                    bool use_transposition_table = getBoolConfigValue(config, "mcts_use_transposition_table", true);
                    int transposition_table_size_mb = getIntConfigValue(config, "mcts_transposition_table_size_mb", 128);
                    
                    // Progressive widening settings
                    bool use_progressive_widening = getBoolConfigValue(config, "mcts_use_progressive_widening", true);
                    float progressive_widening_c = getFloatConfigValue(config, "mcts_progressive_widening_c", 1.0f);
                    float progressive_widening_k = getFloatConfigValue(config, "mcts_progressive_widening_k", 0.5f);
                    
                    // Root parallelization settings
                    bool use_root_parallelization = getBoolConfigValue(config, "mcts_use_root_parallelization", true);
                    int num_root_workers = getIntConfigValue(config, "mcts_num_root_workers", 4);
                    
                    // RAVE settings
                    bool use_rave = getBoolConfigValue(config, "mcts_use_rave", true);
                    float rave_constant = getFloatConfigValue(config, "mcts_rave_constant", 3000.0f);
                    
                    // Neural network configuration
                    int num_res_blocks = getIntConfigValue(config, "num_res_blocks", 10);
                    int num_filters = getIntConfigValue(config, "num_filters", 64);
                    
                    int num_games = getIntConfigValue(config, "self_play_num_games", 100);
                    int num_parallel_games = getIntConfigValue(config, "self_play_num_parallel_games", 4);
                    
                    // Root parallelization settings - already set above, no need to redefine
                    
                    std::cout << "  Number of parallel games: " << num_parallel_games << std::endl;
                    std::cout << "  Number of root workers per game: " << num_root_workers << std::endl;
                    int max_moves = getIntConfigValue(config, "self_play_max_moves", 0);
                    int temperature_threshold = getIntConfigValue(config, "self_play_temperature_threshold", 30);
                    float high_temperature = getFloatConfigValue(config, "self_play_high_temperature", 1.0f);
                    float low_temperature = getFloatConfigValue(config, "self_play_low_temperature", 0.1f);
                    std::string output_dir = getStringConfigValue(config, "self_play_output_dir", "data/self_play_games");
                    std::string output_format = getStringConfigValue(config, "self_play_output_format", "json");
                    
                    // Game-specific settings
                    // Gomoku settings
                    bool gomoku_use_renju = getBoolConfigValue(config, "gomoku_use_renju", true);
                    bool gomoku_use_omok = getBoolConfigValue(config, "gomoku_use_omok", false);
                    bool gomoku_use_pro_long_opening = getBoolConfigValue(config, "gomoku_use_pro_long_opening", true);
                    
                    // Chess settings
                    bool chess_use_chess960 = getBoolConfigValue(config, "chess_use_chess960", false);
                    
                    // Go settings
                    float go_komi = getFloatConfigValue(config, "go_komi", 7.5f);
                    bool go_chinese_rules = getBoolConfigValue(config, "go_chinese_rules", true);
                    bool go_enforce_superko = getBoolConfigValue(config, "go_enforce_superko", true);
                    
                    // Convert game type string to enum
                    alphazero::core::GameType game_type;
                    if (game_type_str == "gomoku") {
                        game_type = alphazero::core::GameType::GOMOKU;
                    } else if (game_type_str == "chess") {
                        game_type = alphazero::core::GameType::CHESS;
                    } else if (game_type_str == "go") {
                        game_type = alphazero::core::GameType::GO;
                    } else {
                        std::cerr << "Error: Unknown game type: " << game_type_str << std::endl;
                        return 1;
                    }
                    
                    // Create output directory if it doesn't exist
                    std::filesystem::create_directories(output_dir);
                    
                    // Setup MCTS settings
                    alphazero::mcts::MCTSSettings mcts_settings;
                    mcts_settings.num_simulations = num_simulations;
                    mcts_settings.num_threads = num_threads;
                    mcts_settings.batch_size = batch_size;
                    mcts_settings.max_collection_batch_size = max_collection_batch_size;
                    mcts_settings.batch_timeout = std::chrono::milliseconds(batch_timeout_ms);
                    mcts_settings.exploration_constant = exploration_constant;
                    mcts_settings.virtual_loss = virtual_loss;
                    mcts_settings.add_dirichlet_noise = add_dirichlet_noise;
                    mcts_settings.dirichlet_alpha = dirichlet_alpha;
                    mcts_settings.dirichlet_epsilon = dirichlet_epsilon;
                    mcts_settings.temperature = temperature;
                    mcts_settings.use_transposition_table = use_transposition_table;
                    mcts_settings.transposition_table_size_mb = transposition_table_size_mb;
                    mcts_settings.use_progressive_widening = use_progressive_widening;
                    mcts_settings.progressive_widening_c = progressive_widening_c;
                    mcts_settings.progressive_widening_k = progressive_widening_k;
                    mcts_settings.use_root_parallelization = use_root_parallelization;
                    mcts_settings.num_root_workers = num_root_workers;
                    mcts_settings.use_rave = use_rave;
                    mcts_settings.rave_constant = rave_constant;
                    
                    // Setup self-play settings
                    alphazero::selfplay::SelfPlaySettings self_play_settings;
                    self_play_settings.mcts_settings = mcts_settings;
                    self_play_settings.num_parallel_games = num_parallel_games;
                    
                    // Use root parallelization setting from config
                    self_play_settings.mcts_settings.use_root_parallelization = use_root_parallelization;
                    self_play_settings.mcts_settings.num_root_workers = num_root_workers;
                    self_play_settings.max_moves = max_moves;
                    self_play_settings.temperature_threshold = temperature_threshold;
                    self_play_settings.high_temperature = high_temperature;
                    self_play_settings.low_temperature = low_temperature;
                    
                    // Set game-specific config
                    self_play_settings.game_config.gomoku_use_renju = gomoku_use_renju;
                    self_play_settings.game_config.gomoku_use_omok = gomoku_use_omok;
                    self_play_settings.game_config.gomoku_use_pro_long_opening = gomoku_use_pro_long_opening;
                    self_play_settings.game_config.chess_use_chess960 = chess_use_chess960;
                    self_play_settings.game_config.go_komi = go_komi;
                    self_play_settings.game_config.go_chinese_rules = go_chinese_rules;
                    self_play_settings.game_config.go_enforce_superko = go_enforce_superko;
                    
                    // Print configuration summary
                    std::cout << "Configuration summary:" << std::endl;
                    std::cout << "  Game: " << game_type_str << std::endl;
                    std::cout << "  Board size: " << board_size << std::endl;
                    std::cout << "  MCTS simulations: " << num_simulations << std::endl;
                    std::cout << "  Number of games: " << num_games << std::endl;
                    std::cout << "  Output directory: " << output_dir << std::endl;
                    
                    // Debug model path
                    if (model_path.empty()) {
                        std::cerr << "Warning: Model path is empty!" << std::endl;
                        model_path = "/tmp/test_model.pt";
                    }
                    std::cout << "  Model path: '" << model_path << "'" << std::endl;
                    
                    // Convert relative paths to absolute if needed
                    std::filesystem::path model_file_path(model_path);
                    if (model_file_path.is_relative()) {
                        model_file_path = std::filesystem::absolute(model_file_path);
                        model_path = model_file_path.string();
                        std::cout << "  Absolute model path: " << model_path << std::endl;
                    }
                    
                    std::filesystem::path output_dir_path(output_dir);
                    if (output_dir_path.is_relative()) {
                        output_dir_path = std::filesystem::absolute(output_dir_path);
                        output_dir = output_dir_path.string();
                        std::cout << "  Absolute output directory: " << output_dir << std::endl;
                    }
                    
                    // Initialize neural network
                    std::cout << "Initializing neural network..." << std::endl;
                    int num_channels;
                    int policy_size;
                    
                    // Set input channels and policy size based on game type
                    switch (game_type) {
                        case alphazero::core::GameType::GOMOKU:
                            num_channels = 17;  // Enhanced representation with history (2*8 for history pairs + 1 turn plane)
                            policy_size = board_size * board_size;
                            break;
                        case alphazero::core::GameType::CHESS:
                            num_channels = 17;  // 12 piece planes + 3 special + 1 turn + 1 move count
                            policy_size = 64 * 73;  // Source, destination, promotions
                            break;
                        case alphazero::core::GameType::GO:
                            num_channels = 17;  // 2 player planes + 1 turn + 8 history
                            policy_size = board_size * board_size + 1;  // +1 for pass
                            break;
                        default:
                            std::cerr << "Error: Unknown game type" << std::endl;
                            return 1;
                    }
                    
                    // Load or create neural network model
                    std::shared_ptr<alphazero::nn::NeuralNetwork> neural_net;
                    try {
                        bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
                        
                        if (std::filesystem::exists(model_path)) {
                            std::cout << "Loading existing model from: " << model_path << std::endl;
                            neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
                                model_path, num_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
                        } else {
                            std::cout << "Creating new model and saving to: " << model_path << std::endl;
                            std::cout << "Full model path: " << std::filesystem::absolute(model_path).string() << std::endl;
                            neural_net = alphazero::nn::NeuralNetworkFactory::createResNet(
                                num_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
                            
                            // Create parent directory if it doesn't exist
                            try {
                                std::filesystem::path model_file_path(model_path);
                                auto parent_path = model_file_path.parent_path();
                                if (!parent_path.empty()) {
                                    std::cout << "Creating directory: " << parent_path << std::endl;
                                    std::filesystem::create_directories(parent_path);
                                }
                            } catch (const std::exception& e) {
                                std::cerr << "Error creating model directory: " << e.what() << std::endl;
                                // Continue anyway, it might be a permission issue but parent dir might already exist
                            }
                            
                            // Save initial model
                            std::cout << "Saving initial model to: " << model_path << std::endl;
                            neural_net->save(model_path);
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error initializing neural network: " << e.what() << std::endl;
                        return 1;
                    }
                    
                    // Initialize self-play manager
                    std::cout << "Initializing self-play manager..." << std::endl;
                    alphazero::selfplay::SelfPlayManager self_play_manager(neural_net, self_play_settings);
                    
                    // Generate games
                    std::cout << "Generating " << num_games << " self-play games..." << std::endl;
                    try {
                        auto games = self_play_manager.generateGames(game_type, num_games, board_size);
                        
                        // Save games
                        std::cout << "Saving games to " << output_dir << " in " << output_format << " format..." << std::endl;
                        self_play_manager.saveGames(games, output_dir, output_format);
                        
                        std::cout << "Self-play completed successfully." << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Error during self-play: " << e.what() << std::endl;
                        return 1;
                    }
                    
                    return 0;
                } catch (const std::exception& e) {
                    std::cerr << "Error during self-play: " << e.what() << std::endl;
                    return 1;
                }
            }
        );

        cli.addCommand("pipeline", "Run complete AlphaZero training pipeline",
            alphazero::cli::runPipelineCommand
        );

        cli.addCommand("train", "Train neural network from self-play data",
            [](const std::vector<std::string>& args) {
                std::cout << "Executing neural network training..." << std::endl;
                
                // Parse arguments for configuration
                std::string config_path;
                for (size_t i = 0; i < args.size(); i++) {
                    if (args[i] == "--config" && i + 1 < args.size()) {
                        config_path = args[i + 1];
                        break;
                    }
                }
                
                if (config_path.empty()) {
                    std::cerr << "Error: Config file path not specified. Use --config <path>" << std::endl;
                    return 1;
                }
                
                try {
                    // Load configuration
                    std::ifstream config_file(config_path);
                    if (!config_file.is_open()) {
                        std::cerr << "Error: Could not open config file: " << config_path << std::endl;
                        return 1;
                    }
                    
                    // Parse YAML configuration
                    std::string line;
                    std::map<std::string, std::string> config;
                    std::string current_section;
                    
                    while (std::getline(config_file, line)) {
                        // Skip comments and empty lines
                        if (line.empty() || line[0] == '#') {
                            continue;
                        }
                        
                        // Check if this is a section header
                        if (line[line.size() - 1] == ':') {
                            current_section = line.substr(0, line.size() - 1);
                            continue;
                        }
                        
                        // Parse key-value pairs
                        size_t colon_pos = line.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string key = line.substr(0, colon_pos);
                            std::string value = line.substr(colon_pos + 1);
                            
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            
                            // Store with section prefix if we're in a section
                            if (!current_section.empty()) {
                                config[current_section + "." + key] = value;
                            } else {
                                config[key] = value;
                            }
                        }
                    }
                    
                    // Extract relevant configuration values
                    std::string game_type_str = getStringConfigValue(config, "game_type", "gomoku");
                    int board_size = getIntConfigValue(config, "board_size", 15);
                    std::string model_path = getStringConfigValue(config, "model_path", "models/model.pt");
                    std::string selfplay_dir = getStringConfigValue(config, "self_play_output_dir", "data/self_play_games");
                    std::string selfplay_format = getStringConfigValue(config, "self_play_output_format", "json");
                    
                    // Get neural network parameters
                    int num_res_blocks = getIntConfigValue(config, "num_res_blocks", 10);
                    int num_filters = getIntConfigValue(config, "num_filters", 64);
                    
                    // Neural network configuration
                    int num_res_blocks_train = getIntConfigValue(config, "num_res_blocks", 10);
                    int num_filters_train = getIntConfigValue(config, "num_filters", 64);
                    
                    // Convert game type string to enum
                    alphazero::core::GameType game_type;
                    if (game_type_str == "gomoku") {
                        game_type = alphazero::core::GameType::GOMOKU;
                    } else if (game_type_str == "chess") {
                        game_type = alphazero::core::GameType::CHESS;
                    } else if (game_type_str == "go") {
                        game_type = alphazero::core::GameType::GO;
                    } else {
                        std::cerr << "Error: Unknown game type: " << game_type_str << std::endl;
                        return 1;
                    }
                    
                    // Set up neural network channel and policy sizes
                    int num_channels;
                    int policy_size;
                    
                    // Set input channels and policy size based on game type
                    switch (game_type) {
                        case alphazero::core::GameType::GOMOKU:
                            num_channels = 17;  // Enhanced representation with history (2*8 for history pairs + 1 turn plane)
                            policy_size = board_size * board_size;
                            break;
                        case alphazero::core::GameType::CHESS:
                            num_channels = 17;  // 12 piece planes + 3 special + 1 turn + 1 move count
                            policy_size = 64 * 73;  // Source, destination, promotions
                            break;
                        case alphazero::core::GameType::GO:
                            num_channels = 17;  // 2 player planes + 1 turn + 8 history
                            policy_size = board_size * board_size + 1;  // +1 for pass
                            break;
                        default:
                            std::cerr << "Error: Unknown game type" << std::endl;
                            return 1;
                    }
                    
                    // Load self-play games for training
                    std::cout << "Loading self-play games from " << selfplay_dir << "..." << std::endl;
                    auto games = alphazero::selfplay::SelfPlayManager::loadGames(selfplay_dir, selfplay_format);
                    
                    if (games.empty()) {
                        std::cerr << "Error: No self-play games found in " << selfplay_dir << std::endl;
                        return 1;
                    }
                    
                    std::cout << "Loaded " << games.size() << " self-play games." << std::endl;
                    
                    // Convert games to training examples
                    std::cout << "Converting games to training examples..." << std::endl;
                    auto examples = alphazero::selfplay::SelfPlayManager::convertToTrainingExamples(games);
                    
                    std::cout << "Generated " << examples.first.size() << " training examples." << std::endl;
                    
                    // Load or create neural network model
                    std::shared_ptr<alphazero::nn::ResNetModel> neural_net;
                    try {
                        bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
                        
                        if (std::filesystem::exists(model_path)) {
                            std::cout << "Loading existing model from: " << model_path << std::endl;
                            neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
                                model_path, num_channels, board_size, num_res_blocks_train, num_filters_train, policy_size, use_gpu);
                        } else {
                            std::cerr << "Error: Model file not found: " << model_path << std::endl;
                            return 1;
                        }
                        
                        // Train the model
                        std::cout << "Training neural network..." << std::endl;
                        
                        // Get training parameters
                        int batch_size = 256;
                        float learning_rate = 0.001f;
                        int num_epochs = 10;
                        
                        if (config.find("train_batch_size") != config.end()) {
                            batch_size = std::stoi(config["train_batch_size"]);
                        }
                        if (config.find("train_learning_rate") != config.end()) {
                            learning_rate = std::stof(config["train_learning_rate"]);
                        }
                        if (config.find("train_num_epochs") != config.end()) {
                            num_epochs = std::stoi(config["train_num_epochs"]);
                        }
                        
                        // Train the model using libtorch
                        std::cout << "Training with " << examples.first.size() << " examples..." << std::endl;
                        
                        // Convert C++ vectors to torch tensors
                        std::vector<torch::Tensor> state_tensors;
                        std::vector<torch::Tensor> policy_tensors;
                        std::vector<torch::Tensor> value_tensors;
                        
                        for (size_t i = 0; i < examples.first.size(); i++) {
                            // Process state tensor
                            auto state = examples.first[i];
                            std::vector<int64_t> dims = {
                                static_cast<int64_t>(state.size()),
                                static_cast<int64_t>(state[0].size()),
                                static_cast<int64_t>(state[0][0].size())
                            };
                            
                            auto options = torch::TensorOptions().dtype(torch::kFloat32);
                            torch::Tensor t_state = torch::zeros(dims, options);
                            
                            // Copy data
                            for (size_t c = 0; c < state.size(); ++c) {
                                for (size_t h = 0; h < state[c].size(); ++h) {
                                    for (size_t w = 0; w < state[c][h].size(); ++w) {
                                        t_state[c][h][w] = state[c][h][w];
                                    }
                                }
                            }
                            
                            // Process policy tensor
                            auto policy = examples.second.first[i];
                            torch::Tensor t_policy = torch::zeros({static_cast<int64_t>(policy.size())}, options);
                            for (size_t j = 0; j < policy.size(); j++) {
                                t_policy[j] = policy[j];
                            }
                            
                            // Process value
                            float value = examples.second.second[i];
                            torch::Tensor t_value = torch::tensor(value, options);
                            
                            state_tensors.push_back(t_state);
                            policy_tensors.push_back(t_policy);
                            value_tensors.push_back(t_value);
                        }
                        
                        // Stack tensors into batches
                        torch::Tensor states_batch = torch::stack(state_tensors);
                        torch::Tensor policies_batch = torch::stack(policy_tensors);
                        torch::Tensor values_batch = torch::stack(value_tensors).reshape({-1, 1});
                        
                        // Prepare for training
                        auto model = std::dynamic_pointer_cast<alphazero::nn::ResNetModel>(neural_net);
                        model->train(true);
                        
                        // Create optimizer
                        torch::optim::Adam optimizer(
                            model->parameters(),
                            torch::optim::AdamOptions(learning_rate).weight_decay(1e-4)
                        );
                        
                        // Training loop
                        auto device = model->parameters().begin()->device();
                        states_batch = states_batch.to(device);
                        policies_batch = policies_batch.to(device);
                        values_batch = values_batch.to(device);
                        
                        // Split into batches
                        int num_samples = states_batch.size(0);
                        int num_batches = (num_samples + batch_size - 1) / batch_size;
                        
                        for (int epoch = 0; epoch < num_epochs; epoch++) {
                            // Shuffle indices
                            auto indices = torch::randperm(num_samples);
                            float total_loss = 0.0f;
                            float policy_loss_sum = 0.0f;
                            float value_loss_sum = 0.0f;
                            
                            for (int batch = 0; batch < num_batches; batch++) {
                                // Get batch indices
                                int start_idx = batch * batch_size;
                                int end_idx = std::min(start_idx + batch_size, num_samples);
                                auto batch_indices = indices.slice(0, start_idx, end_idx);
                                
                                // Select batch data
                                auto batch_states = states_batch.index_select(0, batch_indices);
                                auto batch_policies = policies_batch.index_select(0, batch_indices);
                                auto batch_values = values_batch.index_select(0, batch_indices);
                                
                                // Forward pass
                                auto [policy_logits, value] = model->forward(batch_states);
                                
                                // Compute loss
                                auto policy_loss = -torch::sum(batch_policies * policy_logits) / batch_policies.size(0);
                                auto value_loss = torch::mean(torch::pow(value - batch_values, 2));
                                auto loss = policy_loss + value_loss;
                                
                                // Backward and optimize
                                optimizer.zero_grad();
                                loss.backward();
                                optimizer.step();
                                
                                // Track metrics
                                total_loss += loss.item<float>();
                                policy_loss_sum += policy_loss.item<float>();
                                value_loss_sum += value_loss.item<float>();
                            }
                            
                            // Print progress
                            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                                     << ": Loss=" << (total_loss / num_batches)
                                     << ", Policy=" << (policy_loss_sum / num_batches)
                                     << ", Value=" << (value_loss_sum / num_batches) << std::endl;
                        }
                        
                        // Switch back to evaluation mode
                        model->train(false);
                        
                        // Save the trained model
                        std::cout << "Saving trained model to: " << model_path << std::endl;
                        neural_net->save(model_path);
                        
                        std::cout << "Training completed successfully." << std::endl;
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error during training: " << e.what() << std::endl;
                        return 1;
                    }
                    
                    return 0;
                } catch (const std::exception& e) {
                    std::cerr << "Error during training: " << e.what() << std::endl;
                    return 1;
                }
            }
        );

        cli.addCommand("eval", "Evaluate model strength",
            [](const std::vector<std::string>& args) {
                std::cout << "Executing model evaluation..." << std::endl;
                
                // Parse arguments for configuration
                std::string config_path;
                for (size_t i = 0; i < args.size(); i++) {
                    if (args[i] == "--config" && i + 1 < args.size()) {
                        config_path = args[i + 1];
                        break;
                    }
                }
                
                if (config_path.empty()) {
                    std::cerr << "Error: Config file path not specified. Use --config <path>" << std::endl;
                    return 1;
                }
                
                try {
                    // Load configuration
                    std::ifstream config_file(config_path);
                    if (!config_file.is_open()) {
                        std::cerr << "Error: Could not open config file: " << config_path << std::endl;
                        return 1;
                    }
                    
                    // Parse YAML configuration
                    std::string line;
                    std::map<std::string, std::string> config;
                    std::string current_section;
                    
                    while (std::getline(config_file, line)) {
                        // Skip comments and empty lines
                        if (line.empty() || line[0] == '#') {
                            continue;
                        }
                        
                        // Check if this is a section header
                        if (line[line.size() - 1] == ':') {
                            current_section = line.substr(0, line.size() - 1);
                            continue;
                        }
                        
                        // Parse key-value pairs
                        size_t colon_pos = line.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string key = line.substr(0, colon_pos);
                            std::string value = line.substr(colon_pos + 1);
                            
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            
                            // Store with section prefix if we're in a section
                            if (!current_section.empty()) {
                                config[current_section + "." + key] = value;
                            } else {
                                config[key] = value;
                            }
                        }
                    }
                    
                    // Extract relevant configuration values
                    std::string game_type_str_eval = getStringConfigValue(config, "game_type", "gomoku");
                    int board_size_eval = getIntConfigValue(config, "board_size", 15);
                    std::string model_path_eval = getStringConfigValue(config, "model_path", "models/model.pt");
                    
                    // Neural network configuration
                    int num_res_blocks_eval = getIntConfigValue(config, "num_res_blocks", 10);
                    int num_filters_eval = getIntConfigValue(config, "num_filters", 64);
                    
                    // Extract evaluation specific settings
                    int num_eval_games = getIntConfigValue(config, "evaluation_num_games", 20);
                    int num_parallel_eval_games = getIntConfigValue(config, "evaluation_num_parallel_games", 4);
                    int max_moves_eval = getIntConfigValue(config, "evaluation_max_moves", 0);
                    
                    // MCTS settings for evaluation
                    int num_simulations_eval = getIntConfigValue(config, "mcts_num_simulations", 100);
                    int num_threads_eval = getIntConfigValue(config, "mcts_num_threads", 4);
                    int batch_size_eval = getIntConfigValue(config, "mcts_batch_size", 32);
                    float exploration_constant_eval = getFloatConfigValue(config, "mcts_exploration_constant", 1.5f);
                    int virtual_loss_eval = getIntConfigValue(config, "mcts_virtual_loss", 3);
                    bool add_dirichlet_noise_eval = getBoolConfigValue(config, "mcts_add_dirichlet_noise", false);
                    float dirichlet_alpha_eval = getFloatConfigValue(config, "mcts_dirichlet_alpha", 0.3f);
                    float dirichlet_epsilon_eval = getFloatConfigValue(config, "mcts_dirichlet_epsilon", 0.25f);
                    float temperature_eval = getFloatConfigValue(config, "mcts_temperature", 0.1f);
                    int batch_timeout_ms_eval = getIntConfigValue(config, "mcts_batch_timeout_ms", 25);
                    
                    // Convert game type string to enum
                    alphazero::core::GameType game_type;
                    if (game_type_str_eval == "gomoku") {
                        game_type = alphazero::core::GameType::GOMOKU;
                    } else if (game_type_str_eval == "chess") {
                        game_type = alphazero::core::GameType::CHESS;
                    } else if (game_type_str_eval == "go") {
                        game_type = alphazero::core::GameType::GO;
                    } else {
                        std::cerr << "Error: Unknown game type: " << game_type_str_eval << std::endl;
                        return 1;
                    }
                    
                    // Set up neural network channel and policy sizes
                    int num_channels;
                    int policy_size;
                    
                    // Set input channels and policy size based on game type
                    switch (game_type) {
                        case alphazero::core::GameType::GOMOKU:
                            num_channels = 17;  // Enhanced representation with history (2*8 for history pairs + 1 turn plane)
                            policy_size = board_size_eval * board_size_eval;
                            break;
                        case alphazero::core::GameType::CHESS:
                            num_channels = 17;  // 12 piece planes + 3 special + 1 turn + 1 move count
                            policy_size = 64 * 73;  // Source, destination, promotions
                            break;
                        case alphazero::core::GameType::GO:
                            num_channels = 17;  // 2 player planes + 1 turn + 8 history
                            policy_size = board_size_eval * board_size_eval + 1;  // +1 for pass
                            break;
                        default:
                            std::cerr << "Error: Unknown game type" << std::endl;
                            return 1;
                    }
                    
                    // Check if model exists
                    if (!std::filesystem::exists(model_path_eval)) {
                        std::cerr << "Error: Model file not found: " << model_path_eval << std::endl;
                        return 1;
                    }
                    
                    // Load model
                    std::cout << "Loading model from: " << model_path_eval << std::endl;
                    std::shared_ptr<alphazero::nn::NeuralNetwork> neural_net;
                    try {
                        bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
                        neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
                            model_path_eval, num_channels, board_size_eval, num_res_blocks_eval, num_filters_eval, policy_size, use_gpu);
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading model: " << e.what() << std::endl;
                        return 1;
                    }
                    
                    // Setup MCTS settings
                    alphazero::mcts::MCTSSettings mcts_settings;
                    mcts_settings.num_simulations = num_simulations_eval;
                    mcts_settings.num_threads = num_threads_eval;
                    mcts_settings.batch_size = batch_size_eval;
                    mcts_settings.batch_timeout = std::chrono::milliseconds(batch_timeout_ms_eval);
                    mcts_settings.exploration_constant = exploration_constant_eval;
                    mcts_settings.virtual_loss = virtual_loss_eval;
                    mcts_settings.add_dirichlet_noise = add_dirichlet_noise_eval;
                    mcts_settings.dirichlet_alpha = dirichlet_alpha_eval;
                    mcts_settings.dirichlet_epsilon = dirichlet_epsilon_eval;
                    mcts_settings.temperature = temperature_eval;
                    
                    // Set up self-play settings for evaluation
                    alphazero::selfplay::SelfPlaySettings self_play_settings;
                    self_play_settings.mcts_settings = mcts_settings;
                    self_play_settings.num_parallel_games = num_parallel_eval_games;
                    self_play_settings.num_mcts_engines = num_parallel_eval_games;
                    self_play_settings.max_moves = max_moves_eval;
                    
                    // Run evaluation
                    std::cout << "Running evaluation on model: " << model_path_eval << std::endl;
                    std::cout << "Playing " << num_eval_games << " games..." << std::endl;
                    
                    alphazero::selfplay::SelfPlayManager selfplay_manager(neural_net, self_play_settings);
                    auto games = selfplay_manager.generateGames(game_type, num_eval_games, board_size_eval);
                    
                    // Calculate statistics
                    int player1_wins = 0;
                    int player2_wins = 0;
                    int draws = 0;
                    int total_moves = 0;
                    
                    for (const auto& game : games) {
                        if (game.winner == 1) {
                            player1_wins++;
                        } else if (game.winner == 2) {
                            player2_wins++;
                        } else {
                            draws++;
                        }
                        total_moves += game.moves.size();
                    }
                    
                    // Print results
                    std::cout << "\nEvaluation Results:\n";
                    std::cout << "==================\n";
                    std::cout << "Total games: " << games.size() << std::endl;
                    std::cout << "Player 1 wins: " << player1_wins << " (" 
                             << (player1_wins * 100.0f / games.size()) << "%)" << std::endl;
                    std::cout << "Player 2 wins: " << player2_wins << " (" 
                             << (player2_wins * 100.0f / games.size()) << "%)" << std::endl;
                    std::cout << "Draws: " << draws << " (" 
                             << (draws * 100.0f / games.size()) << "%)" << std::endl;
                    std::cout << "Average moves per game: " << (total_moves / games.size()) << std::endl;
                    
                    return 0;
                } catch (const std::exception& e) {
                    std::cerr << "Error during evaluation: " << e.what() << std::endl;
                    return 1;
                }
            }
        );

        cli.addCommand("play", "Play against AI",
            [](const std::vector<std::string>& args) {
                std::cout << "Starting play mode..." << std::endl;
                
                // Parse arguments for configuration
                std::string config_path;
                for (size_t i = 0; i < args.size(); i++) {
                    if (args[i] == "--config" && i + 1 < args.size()) {
                        config_path = args[i + 1];
                        break;
                    }
                }
                
                if (config_path.empty()) {
                    std::cerr << "Error: Config file path not specified. Use --config <path>" << std::endl;
                    return 1;
                }
                
                try {
                    // Load configuration
                    std::ifstream config_file(config_path);
                    if (!config_file.is_open()) {
                        std::cerr << "Error: Could not open config file: " << config_path << std::endl;
                        return 1;
                    }
                    
                    // Parse YAML configuration
                    std::string line;
                    std::map<std::string, std::string> config;
                    std::string current_section;
                    
                    while (std::getline(config_file, line)) {
                        // Skip comments and empty lines
                        if (line.empty() || line[0] == '#') {
                            continue;
                        }
                        
                        // Check if this is a section header
                        if (line[line.size() - 1] == ':') {
                            current_section = line.substr(0, line.size() - 1);
                            continue;
                        }
                        
                        // Parse key-value pairs
                        size_t colon_pos = line.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string key = line.substr(0, colon_pos);
                            std::string value = line.substr(colon_pos + 1);
                            
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            
                            // Store with section prefix if we're in a section
                            if (!current_section.empty()) {
                                config[current_section + "." + key] = value;
                            } else {
                                config[key] = value;
                            }
                        }
                    }
                    
                    // Extract relevant configuration values
                    std::string game_type_str_play = getStringConfigValue(config, "game_type", "gomoku");
                    int board_size_play = getIntConfigValue(config, "board_size", 15);
                    std::string model_path_play = getStringConfigValue(config, "model_path", "models/best_model.pt");
                    
                    // Neural network configuration
                    int num_res_blocks_play = getIntConfigValue(config, "num_res_blocks", 10);
                    int num_filters_play = getIntConfigValue(config, "num_filters", 64);
                    
                    // MCTS settings for play
                    int num_simulations_play = getIntConfigValue(config, "mcts_num_simulations", 800);
                    int num_threads_play = getIntConfigValue(config, "mcts_num_threads", 2);
                    int batch_size_play = getIntConfigValue(config, "mcts_batch_size", 1);
                    float exploration_constant_play = getFloatConfigValue(config, "mcts_exploration_constant", 1.0f);
                    int virtual_loss_play = getIntConfigValue(config, "mcts_virtual_loss", 3);
                    bool add_dirichlet_noise_play = getBoolConfigValue(config, "mcts_add_dirichlet_noise", false);
                    float dirichlet_alpha_play = getFloatConfigValue(config, "mcts_dirichlet_alpha", 0.3f);
                    float dirichlet_epsilon_play = getFloatConfigValue(config, "mcts_dirichlet_epsilon", 0.25f);
                    float temperature_play = getFloatConfigValue(config, "mcts_temperature", 0.1f);
                    int batch_timeout_ms_play = getIntConfigValue(config, "mcts_batch_timeout_ms", 0);
                    
                    // Convert game type string to enum
                    alphazero::core::GameType game_type;
                    if (game_type_str_play == "gomoku") {
                        game_type = alphazero::core::GameType::GOMOKU;
                    } else if (game_type_str_play == "chess") {
                        game_type = alphazero::core::GameType::CHESS;
                    } else if (game_type_str_play == "go") {
                        game_type = alphazero::core::GameType::GO;
                    } else {
                        std::cerr << "Error: Unknown game type: " << game_type_str_play << std::endl;
                        return 1;
                    }
                    
                    // Set up neural network channel and policy sizes
                    int num_channels;
                    int policy_size;
                    
                    // Set input channels and policy size based on game type
                    switch (game_type) {
                        case alphazero::core::GameType::GOMOKU:
                            num_channels = 17;  // Enhanced representation with history (2*8 for history pairs + 1 turn plane)
                            policy_size = board_size_play * board_size_play;
                            break;
                        case alphazero::core::GameType::CHESS:
                            num_channels = 17;  // 12 piece planes + 3 special + 1 turn + 1 move count
                            policy_size = 64 * 73;  // Source, destination, promotions
                            break;
                        case alphazero::core::GameType::GO:
                            num_channels = 17;  // 2 player planes + 1 turn + 8 history
                            policy_size = board_size_play * board_size_play + 1;  // +1 for pass
                            break;
                        default:
                            std::cerr << "Error: Unknown game type" << std::endl;
                            return 1;
                    }
                    
                    // Check if model exists
                    if (!std::filesystem::exists(model_path_play)) {
                        std::cerr << "Error: Model file not found: " << model_path_play << std::endl;
                        return 1;
                    }
                    
                    // Load model
                    std::cout << "Loading model from: " << model_path_play << std::endl;
                    std::shared_ptr<alphazero::nn::NeuralNetwork> neural_net;
                    try {
                        bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
                        neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
                            model_path_play, num_channels, board_size_play, num_res_blocks_play, num_filters_play, policy_size, use_gpu);
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading model: " << e.what() << std::endl;
                        return 1;
                    }
                    
                    // Setup MCTS settings
                    alphazero::mcts::MCTSSettings mcts_settings;
                    mcts_settings.num_simulations = num_simulations_play;
                    mcts_settings.num_threads = num_threads_play;
                    mcts_settings.batch_size = batch_size_play;
                    mcts_settings.batch_timeout = std::chrono::milliseconds(batch_timeout_ms_play);
                    mcts_settings.exploration_constant = exploration_constant_play;
                    mcts_settings.virtual_loss = virtual_loss_play;
                    mcts_settings.add_dirichlet_noise = add_dirichlet_noise_play;
                    mcts_settings.dirichlet_alpha = dirichlet_alpha_play;
                    mcts_settings.dirichlet_epsilon = dirichlet_epsilon_play;
                    mcts_settings.temperature = temperature_play;
                    
                    // Create MCTS engine
                    alphazero::mcts::MCTSEngine mcts(neural_net, mcts_settings);
                    
                    // Create game state
                    std::unique_ptr<alphazero::core::IGameState> game_state;
                    switch (game_type) {
                        case alphazero::core::GameType::GOMOKU:
                            game_state = std::make_unique<alphazero::games::gomoku::GomokuState>(board_size_play);
                            break;
                        case alphazero::core::GameType::CHESS:
                            game_state = std::make_unique<alphazero::games::chess::ChessState>();
                            break;
                        case alphazero::core::GameType::GO:
                            game_state = std::make_unique<alphazero::games::go::GoState>(board_size_play);
                            break;
                        default:
                            std::cerr << "Error: Unknown game type" << std::endl;
                            return 1;
                    }
                    
                    std::cout << "\nStarting a new game of " << game_type_str_play << ".\n";
                    
                    // Game loop
                    int player = 1;  // Human is always player 1
                    // int ai_player = 2; // Unused variable
                    bool human_turn = true;
                    
                    while (!game_state->isTerminal()) {
                        // Display current state
                        std::cout << "\nCurrent board:\n";
                        std::cout << game_state->toString() << std::endl;
                        
                        // Check current player
                        int current_player = game_state->getCurrentPlayer();
                        human_turn = (current_player == player);
                        
                        if (human_turn) {
                            // Human's turn
                            std::cout << "Your move (enter 'help' for commands): ";
                            std::string input;
                            std::getline(std::cin, input);
                            
                            // Handle special commands
                            if (input == "help") {
                                std::cout << "Available commands:\n";
                                std::cout << "  help     - Show this help message\n";
                                std::cout << "  quit     - Quit the game\n";
                                std::cout << "  resign   - Resign the game\n";
                                std::cout << "  moves    - Show possible moves\n";
                                std::cout << "  <move>   - Make a move (format depends on the game)\n";
                                
                                switch (game_type) {
                                    case alphazero::core::GameType::GOMOKU:
                                        std::cout << "For Gomoku, enter moves as row,col (e.g. '7,8')\n";
                                        break;
                                    case alphazero::core::GameType::CHESS:
                                        std::cout << "For Chess, enter moves in algebraic notation (e.g. 'e2e4')\n";
                                        break;
                                    case alphazero::core::GameType::GO:
                                        std::cout << "For Go, enter moves as row,col (e.g. '3,4'), or 'pass'\n";
                                        break;
                                    default:
                                        break;
                                }
                                continue;
                            } else if (input == "quit") {
                                std::cout << "Quitting game...\n";
                                return 0;
                            } else if (input == "resign") {
                                std::cout << "You resigned. AI wins!\n";
                                break;
                            } else if (input == "moves") {
                                auto legal_moves = game_state->getLegalMoves();
                                std::cout << "Legal moves: ";
                                for (int move : legal_moves) {
                                    std::cout << game_state->actionToString(move) << " ";
                                }
                                std::cout << std::endl;
                                continue;
                            }
                            
                            // Try to parse and make move
                            try {
                                int move = -1;
                                
                                // Parse move based on game type
                                switch (game_type) {
                                    case alphazero::core::GameType::GOMOKU:
                                    case alphazero::core::GameType::GO: {
                                        // Parse row,col format
                                        if (input == "pass" && game_type == alphazero::core::GameType::GO) {
                                            // Special case for Go pass move
                                            move = board_size_play * board_size_play;  // Pass move index
                                        } else {
                                            size_t comma_pos = input.find(',');
                                            if (comma_pos == std::string::npos) {
                                                std::cout << "Invalid move format. Use 'row,col' (e.g. '7,8')" << std::endl;
                                                continue;
                                            }
                                            
                                            int row = std::stoi(input.substr(0, comma_pos));
                                            int col = std::stoi(input.substr(comma_pos + 1));
                                            
                                            // Convert to move index
                                            move = row * board_size_play + col;
                                        }
                                        break;
                                    }
                                    case alphazero::core::GameType::CHESS: {
                                        // For simplicity, we'll assume the move is already in the engine's format
                                        // In a real implementation, this would parse algebraic notation
                                        auto legal_moves = game_state->getLegalMoves();
                                        bool found = false;
                                        
                                        // Try to match the input with a legal move string
                                        for (int m : legal_moves) {
                                            if (game_state->actionToString(m) == input) {
                                                move = m;
                                                found = true;
                                                break;
                                            }
                                        }
                                        
                                        if (!found) {
                                            std::cout << "Invalid move. Try again or type 'moves' to see legal moves." << std::endl;
                                            continue;
                                        }
                                        break;
                                    }
                                    default:
                                        std::cerr << "Unknown game type" << std::endl;
                                        continue;
                                }
                                
                                // Check if move is legal
                                if (!game_state->isLegalMove(move)) {
                                    std::cout << "Illegal move. Try again or type 'moves' to see legal moves." << std::endl;
                                    continue;
                                }
                                
                                // Make the move
                                game_state->makeMove(move);
                                std::cout << "You played: " << game_state->actionToString(move) << std::endl;
                                
                            } catch (const std::exception& e) {
                                std::cout << "Error processing move: " << e.what() << std::endl;
                                continue;
                            }
                        } else {
                            // AI's turn
                            std::cout << "AI is thinking...\n";
                            
                            try {
                                // Run MCTS search
                                auto result = mcts.search(*game_state);
                                
                                // Get and make the move
                                int ai_move = result.action;
                                std::cout << "AI plays: " << game_state->actionToString(ai_move) << std::endl;
                                
                                // Make the move
                                game_state->makeMove(ai_move);
                                
                                // Print search statistics
                                std::cout << "Search stats: " << result.stats.total_nodes 
                                         << " nodes, " << result.stats.max_depth 
                                         << " max depth, " << result.stats.avg_batch_size 
                                         << " avg batch size" << std::endl;
                                
                            } catch (const std::exception& e) {
                                std::cerr << "Error during AI move: " << e.what() << std::endl;
                                return 1;
                            }
                        }
                        
                        // Check if game is over after the move
                        if (game_state->isTerminal()) {
                            auto result = game_state->getGameResult();
                            std::cout << "\nFinal board:\n";
                            std::cout << game_state->toString() << std::endl;
                            
                            if (result == alphazero::core::GameResult::WIN_PLAYER1) {
                                std::cout << "You win!" << std::endl;
                            } else if (result == alphazero::core::GameResult::WIN_PLAYER2) {
                                std::cout << "AI wins!" << std::endl;
                            } else {
                                std::cout << "Draw!" << std::endl;
                            }
                        }
                    }
                    
                    return 0;
                } catch (const std::exception& e) {
                    std::cerr << "Error during play: " << e.what() << std::endl;
                    return 1;
                }
            }
        );

        // Reset watchdog before running CLI
        reset_watchdog();
        
        // Run CLI with watchdog checking
        int result = cli.run(argc, argv);
        
        // Cancel watchdog timer since we've completed successfully
        cancel_watchdog();

        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 1;
    }
}