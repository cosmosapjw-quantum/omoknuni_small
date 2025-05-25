// src/cli/omoknuni_cli_enhanced.cpp
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <yaml-cpp/yaml.h>

#include "cli/cli_manager.h"
#include "mcts/enhanced_mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include "core/game_export.h"

using namespace alphazero;

/**
 * @brief Enhanced CLI for AlphaZero with all optimizations
 * 
 * This CLI uses:
 * - Enhanced MCTS Engine with GPU memory pooling
 * - Dynamic batch sizing
 * - Advanced transposition table
 * - Standard self-play manager (OptimizedSelfPlayManager requires MultiInstanceNNManager)
 */

// Parse game type from config
core::GameType parseGameType(const std::string& game_str) {
    if (game_str == "gomoku") return core::GameType::GOMOKU;
    if (game_str == "chess") return core::GameType::CHESS;
    if (game_str == "go") return core::GameType::GO;
    throw std::runtime_error("Unknown game type: " + game_str);
}

// Create game state based on type
std::unique_ptr<core::IGameState> createGameState(core::GameType game_type, const YAML::Node& config) {
    switch (game_type) {
        case core::GameType::GOMOKU:
            return std::make_unique<games::gomoku::GomokuState>(
                config["board_size"].as<int>(15)
            );
        case core::GameType::CHESS:
            return std::make_unique<games::chess::ChessState>();
        case core::GameType::GO:
            return std::make_unique<games::go::GoState>(
                config["board_size"].as<int>(19),
                config["komi"].as<float>(7.5f)
            );
        default:
            throw std::runtime_error("Unsupported game type");
    }
}

// Self-play command with enhanced engine
int runEnhancedSelfPlay(const std::vector<std::string>& args) {
    LOG_SYSTEM_INFO("Starting enhanced self-play with all optimizations");
    
    // Parse config file
    if (args.size() < 2) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_enhanced self-play <config.yaml>");
        return 1;
    }
    
    YAML::Node config;
    try {
        config = YAML::LoadFile(args[1]);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to load config file: {}", e.what());
        return 1;
    }
    
    // Parse game type
    auto game_type = parseGameType(config["game_type"].as<std::string>());
    
    // Load neural network model
    std::string model_path = config["model_path"].as<std::string>("models/model.pt");
    
    // Get model parameters from config
    int input_channels = config["input_channels"].as<int>(17);
    int board_size = config["board_size"].as<int>(19);
    int num_res_blocks = config["num_res_blocks"].as<int>(10);
    int num_filters = config["num_filters"].as<int>(128);
    int policy_size = config["policy_size"].as<int>(board_size * board_size + 1);
    
    LOG_SYSTEM_INFO("Loading neural network from: {}", model_path);
    
    // Load the network using standard factory method
    auto network = nn::NeuralNetworkFactory::loadResNet(
        model_path,
        input_channels,
        board_size,
        num_res_blocks,
        num_filters,
        policy_size,
        true // use_gpu
    );
    
    if (!network) {
        LOG_SYSTEM_ERROR("Failed to load neural network");
        return 1;
    }
    
    // Create enhanced MCTS settings
    mcts::EnhancedMCTSEngine::EnhancedSettings mcts_settings;
    mcts_settings.num_simulations = config["mcts_simulations"].as<int>(800);
    mcts_settings.num_threads = config["mcts_num_threads"].as<int>(12);
    mcts_settings.batch_size = config["mcts_batch_size"].as<int>(256);
    mcts_settings.batch_timeout = std::chrono::milliseconds(
        config["mcts_batch_timeout_ms"].as<int>(5)
    );
    mcts_settings.exploration_constant = config["mcts_c_puct"].as<float>(1.4f);
    mcts_settings.virtual_loss = config["mcts_virtual_loss"].as<int>(3);
    
    // Enhanced features configuration
    mcts_settings.use_gpu_memory_pool = config["use_gpu_memory_pool"].as<bool>(true);
    mcts_settings.use_dynamic_batching = config["use_dynamic_batching"].as<bool>(true);
    mcts_settings.use_advanced_tt = config["use_advanced_transposition_table"].as<bool>(true);
    mcts_settings.nn_instances_per_engine = config["nn_instances_per_engine"].as<int>(1);
    
    // GPU memory pool config
    mcts_settings.gpu_pool_config.initial_pool_size_mb = 
        config["gpu_pool_initial_size_mb"].as<size_t>(1024);
    mcts_settings.gpu_pool_config.max_pool_size_mb = 
        config["gpu_pool_max_size_mb"].as<size_t>(4096);
    
    // Dynamic batching config
    mcts_settings.batch_manager_config.min_batch_size = 
        config["dynamic_batch_min_size"].as<int>(64);
    mcts_settings.batch_manager_config.max_batch_size = 
        config["dynamic_batch_max_size"].as<int>(512);
    mcts_settings.batch_manager_config.preferred_batch_size = 
        config["dynamic_batch_preferred_size"].as<int>(256);
    
    // Advanced transposition table config
    mcts_settings.tt_config.size_mb = 
        config["tt_size_mb"].as<size_t>(256);
    mcts_settings.tt_config.num_buckets = 
        config["tt_num_buckets"].as<size_t>(4);
    mcts_settings.tt_config.enable_compression = 
        config["tt_enable_compression"].as<bool>(true);
    
    LOG_SYSTEM_INFO("Enhanced MCTS configuration:");
    LOG_SYSTEM_INFO("  - GPU memory pool: {} MB", mcts_settings.gpu_pool_config.initial_pool_size_mb);
    LOG_SYSTEM_INFO("  - Dynamic batching: {}-{} (preferred: {})", 
                 mcts_settings.batch_manager_config.min_batch_size,
                 mcts_settings.batch_manager_config.max_batch_size,
                 mcts_settings.batch_manager_config.preferred_batch_size);
    LOG_SYSTEM_INFO("  - Advanced TT: {} MB with {} buckets", 
                 mcts_settings.tt_config.size_mb,
                 mcts_settings.tt_config.num_buckets);
    
    // Create self-play settings
    selfplay::SelfPlaySettings sp_settings;
    sp_settings.mcts_settings = mcts_settings;
    sp_settings.num_parallel_games = config["num_parallel_games"].as<int>(4);
    sp_settings.max_moves = config["max_moves"].as<int>(500);
    sp_settings.high_temperature = config["temperature_start"].as<float>(1.0f);
    sp_settings.temperature_threshold = config["temperature_threshold"].as<int>(30);
    sp_settings.low_temperature = config["temperature_end"].as<float>(0.1f);
    
    // Create a self-play manager
    // Note: For true parallel optimization, we would need to implement a version
    // that creates multiple enhanced MCTS engines
    selfplay::SelfPlayManager manager(network, sp_settings);
    
    // Number of games to generate
    int num_games = config["num_games"].as<int>(100);
    
    LOG_SYSTEM_INFO("Starting enhanced self-play to generate {} games", num_games);
    auto start_time = std::chrono::steady_clock::now();
    
    // Generate games
    auto games = manager.generateGames(game_type, num_games, board_size);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // Save games
    std::string output_dir = config["output_dir"].as<std::string>("data/self_play_games");
    manager.saveGames(games, output_dir, "json");
    
    // Calculate statistics
    int total_moves = 0;
    for (const auto& game : games) {
        total_moves += game.moves.size();
    }
    
    LOG_SYSTEM_INFO("Enhanced self-play completed:");
    LOG_SYSTEM_INFO("  - Total games: {}", games.size());
    LOG_SYSTEM_INFO("  - Duration: {} seconds", duration.count());
    LOG_SYSTEM_INFO("  - Games/second: {:.2f}", 
                 static_cast<float>(games.size()) / duration.count());
    LOG_SYSTEM_INFO("  - Avg game length: {:.1f}", 
                 static_cast<float>(total_moves) / games.size());
    LOG_SYSTEM_INFO("  - Games saved to: {}", output_dir);
    
    return 0;
}

// Training command
int runTraining(const std::vector<std::string>& args) {
    LOG_SYSTEM_INFO("Training with enhanced features not yet implemented");
    LOG_SYSTEM_INFO("Use the Python training script for now");
    return 0;
}

// Evaluation command
int runEvaluation(const std::vector<std::string>& args) {
    LOG_SYSTEM_INFO("Starting enhanced model evaluation");
    
    if (args.size() < 2) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_enhanced eval <config.yaml>");
        return 1;
    }
    
    YAML::Node config;
    try {
        config = YAML::LoadFile(args[1]);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to load config file: {}", e.what());
        return 1;
    }
    
    // Parse game type
    auto game_type = parseGameType(config["game_type"].as<std::string>());
    
    // Load models
    std::string model1_path = config["model1_path"].as<std::string>();
    std::string model2_path = config["model2_path"].as<std::string>();
    
    LOG_SYSTEM_INFO("Loading models for evaluation:");
    LOG_SYSTEM_INFO("  Model 1: {}", model1_path);
    LOG_SYSTEM_INFO("  Model 2: {}", model2_path);
    
    // TODO: Implement enhanced evaluation
    LOG_SYSTEM_INFO("Enhanced evaluation not yet fully implemented");
    return 0;
}

// Interactive play command
int runInteractivePlay(const std::vector<std::string>& args) {
    LOG_SYSTEM_INFO("Starting enhanced interactive play");
    
    if (args.size() < 2) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_enhanced play <config.yaml>");
        return 1;
    }
    
    YAML::Node config;
    try {
        config = YAML::LoadFile(args[1]);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to load config file: {}", e.what());
        return 1;
    }
    
    // Parse game type
    auto game_type = parseGameType(config["game_type"].as<std::string>());
    
    // Create game state
    auto game_state = createGameState(game_type, config);
    
    // TODO: Implement enhanced interactive play
    LOG_SYSTEM_INFO("Enhanced interactive play not yet fully implemented");
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize logging
    utils::Logger::init("enhanced");
    
    try {
        // Create CLI manager
        cli::CLIManager cli_manager;
        
        // Add commands
        cli_manager.addCommand("self-play", "Run enhanced self-play with all optimizations", 
                             runEnhancedSelfPlay);
        cli_manager.addCommand("train", "Train model (not yet implemented)", 
                             runTraining);
        cli_manager.addCommand("eval", "Evaluate models", 
                             runEvaluation);
        cli_manager.addCommand("play", "Interactive play", 
                             runInteractivePlay);
        
        // Parse and execute
        return cli_manager.run(argc, argv);
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Fatal error: {}", e.what());
        return 1;
    }
}