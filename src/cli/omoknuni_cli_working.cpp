// src/cli/omoknuni_cli_working.cpp - Working CLI with correct APIs
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "cli/cli_manager.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include "core/game_export.h"

using namespace alphazero;

// Global shutdown flag
std::atomic<bool> g_shutdown_requested(false);

// Signal handler
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_SYSTEM_INFO("Shutdown signal received");
        g_shutdown_requested.store(true);
    }
}

// Parse game type from string
core::GameType parseGameType(const std::string& game_str) {
    if (game_str == "gomoku") return core::GameType::GOMOKU;
    if (game_str == "chess") return core::GameType::CHESS;
    if (game_str == "go") return core::GameType::GO;
    throw std::runtime_error("Unknown game type: " + game_str);
}

// Self-play command implementation
int runSelfPlay(const std::vector<std::string>& args) {
    LOG_SYSTEM_INFO("Starting self-play");
    
    if (args.empty()) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_working self-play <config.yaml>");
        return 1;
    }
    
    // Load configuration
    YAML::Node config;
    try {
        config = YAML::LoadFile(args[0]);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to load config file: {}", e.what());
        return 1;
    }
    
    // Parse configuration
    auto game_type = parseGameType(config["game_type"].as<std::string>());
    int board_size = config["board_size"].as<int>(15);
    int num_games = config["num_games"].as<int>(100);
    
    // Neural network configuration
    int input_channels = config["input_channels"].as<int>(17);
    int num_res_blocks = config["num_res_blocks"].as<int>(10);
    int num_filters = config["num_filters"].as<int>(64);
    std::string model_path = config["model_path"].as<std::string>("models/model.pt");
    
    // Self-play settings
    selfplay::SelfPlaySettings sp_settings;
    
    // MCTS settings
    sp_settings.mcts_settings.num_simulations = config["mcts_simulations"].as<int>(400);
    sp_settings.mcts_settings.num_threads = config["mcts_num_threads"].as<int>(1);
    sp_settings.mcts_settings.batch_size = config["mcts_batch_size"].as<int>(128);
    sp_settings.mcts_settings.batch_timeout = std::chrono::milliseconds(
        config["mcts_batch_timeout_ms"].as<int>(5)
    );
    sp_settings.mcts_settings.exploration_constant = config["mcts_c_puct"].as<float>(1.4f);
    sp_settings.mcts_settings.virtual_loss = config["mcts_virtual_loss"].as<int>(3);
    sp_settings.mcts_settings.use_transposition_table = 
        config["mcts_enable_transposition"].as<bool>(true);
    
    // Self-play specific settings
    sp_settings.num_parallel_games = config["num_parallel_workers"].as<int>(4);
    sp_settings.temperature_threshold = config["mcts_temp_threshold"].as<int>(30);
    sp_settings.high_temperature = config["mcts_temperature"].as<float>(1.0f);
    sp_settings.low_temperature = 0.1f;
    
    // Output directory
    std::string output_dir = config["output_dir"].as<std::string>("data/self_play_games");
    int save_interval = config["save_interval"].as<int>(10);
    
    LOG_SYSTEM_INFO("Configuration:");
    LOG_SYSTEM_INFO("  - Game: {} ({}x{})", config["game_type"].as<std::string>(), board_size, board_size);
    LOG_SYSTEM_INFO("  - Model: {}", model_path);
    LOG_SYSTEM_INFO("  - Parallel games: {}", sp_settings.num_parallel_games);
    LOG_SYSTEM_INFO("  - Total games: {}", num_games);
    LOG_SYSTEM_INFO("  - MCTS simulations: {}", sp_settings.mcts_settings.num_simulations);
    LOG_SYSTEM_INFO("  - Batch size: {}", sp_settings.mcts_settings.batch_size);
    LOG_SYSTEM_INFO("  - Output: {}", output_dir);
    
    try {
        // Create output directory
        std::filesystem::create_directories(output_dir);
        
        // Load neural network
        auto network = nn::NeuralNetworkFactory::loadResNet(
            model_path, input_channels, board_size, num_res_blocks, num_filters
        );
        
        if (!network) {
            LOG_SYSTEM_ERROR("Failed to load neural network from {}", model_path);
            return 1;
        }
        
        LOG_SYSTEM_INFO("Neural network loaded successfully");
        
        // Create self-play manager
        selfplay::SelfPlayManager manager(network, sp_settings);
        
        // Setup signal handler
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        
        // Progress tracking
        auto start_time = std::chrono::steady_clock::now();
        int games_generated = 0;
        
        // Generate games in batches
        while (games_generated < num_games && !g_shutdown_requested.load()) {
            int batch_size = std::min(save_interval, num_games - games_generated);
            
            LOG_SYSTEM_INFO("Generating batch of {} games...", batch_size);
            
            // Generate games
            auto games = manager.generateGames(game_type, batch_size, board_size);
            
            // Save games
            if (!games.empty()) {
                manager.saveGames(games, output_dir, "json");
                games_generated += games.size();
                
                // Progress update
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();
                
                if (elapsed > 0) {
                    float games_per_sec = static_cast<float>(games_generated) / elapsed;
                    LOG_SYSTEM_INFO("Progress: {} / {} games completed ({:.2f} games/sec)",
                                   games_generated, num_games, games_per_sec);
                }
            }
            
            // Check for shutdown
            if (g_shutdown_requested.load()) {
                LOG_SYSTEM_INFO("Shutdown requested, stopping self-play");
                break;
            }
        }
        
        // Final statistics
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        LOG_SYSTEM_INFO("Self-play completed!");
        LOG_SYSTEM_INFO("  - Total games: {}", games_generated);
        LOG_SYSTEM_INFO("  - Total time: {} seconds", total_elapsed);
        if (total_elapsed > 0) {
            LOG_SYSTEM_INFO("  - Average: {:.2f} games/sec", 
                           static_cast<float>(games_generated) / total_elapsed);
        }
        LOG_SYSTEM_INFO("  - Output directory: {}", output_dir);
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Self-play failed: {}", e.what());
        return 1;
    }
}

// Training command (placeholder)
int runTraining(const std::vector<std::string>&) {
    LOG_SYSTEM_INFO("Training not yet implemented");
    return 0;
}

// Evaluation command (placeholder)
int runEvaluation(const std::vector<std::string>&) {
    LOG_SYSTEM_INFO("Evaluation not yet implemented");
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize logging with SYNCHRONOUS mode to avoid segfault
    utils::Logger::init("logs", 
                       spdlog::level::info,  // console level
                       spdlog::level::debug, // file level
                       10485760,             // 10MB max file size
                       3,                    // max 3 files
                       false);               // SYNCHRONOUS logging
    
    try {
        // Create CLI manager
        cli::CLIManager cli_manager;
        
        // Add commands
        cli_manager.addCommand("self-play", "Run self-play game generation", runSelfPlay);
        cli_manager.addCommand("train", "Train model (not yet implemented)", runTraining);
        cli_manager.addCommand("eval", "Evaluate model (not yet implemented)", runEvaluation);
        
        // Run CLI
        int result = cli_manager.run(argc, argv);
        
        // Clean shutdown
        utils::Logger::shutdown();
        return result;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Fatal error: {}", e.what());
        utils::Logger::shutdown();
        return 1;
    }
}