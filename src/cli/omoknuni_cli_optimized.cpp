// src/cli/omoknuni_cli_optimized.cpp
#include "cli/cli_manager.h"
#include "core/game_export.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "mcts/multi_instance_nn_manager.h"
#include "selfplay/optimized_self_play_manager.h"
#include "nn/optimized_resnet_model.h"
#include "nn/neural_network_factory.h"
#include "utils/thread_local_memory_manager.h"
#include "utils/logger.h"

// Logger macros - using standard format
#undef LOG_INFO
#undef LOG_WARNING  
#undef LOG_ERROR
#undef LOG_DEBUG
#include "cli/alphazero_pipeline.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <map>
#include <chrono>
#include <thread>

// Parse config file
std::map<std::string, std::string> readConfig(const std::string& filepath) {
    std::map<std::string, std::string> config;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Find key-value separator
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config[key] = value;
        }
    }
    
    return config;
}

// Helper functions to safely get config values
int getIntConfigValue(const std::map<std::string, std::string>& config, 
                     const std::string& key, int defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            LOG_SYSTEM_WARN("Invalid value for {}: '{}'. Using default: {}", key, it->second, defaultValue);
        }
    }
    return defaultValue;
}

float getFloatConfigValue(const std::map<std::string, std::string>& config, 
                         const std::string& key, float defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            LOG_SYSTEM_WARN("Invalid value for {}: '{}'. Using default: {}", key, it->second, defaultValue);
        }
    }
    return defaultValue;
}

void runOptimizedSelfPlay(const std::string& config_path) {
    LOG_SYSTEM_INFO("Starting OPTIMIZED self-play with config: {}", config_path);
    
    // Register games first
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
    
    // Read configuration
    auto config = readConfig(config_path);
    
    // Extract settings
    std::string game_type = config["game_type"];
    int board_size = getIntConfigValue(config, "board_size", 15);
    
    // Neural network settings
    int input_channels = getIntConfigValue(config, "input_channels", 17);
    int num_res_blocks = getIntConfigValue(config, "num_res_blocks", 10);
    int num_filters = getIntConfigValue(config, "num_filters", 64);
    
    // Self-play settings
    alphazero::selfplay::SelfPlaySettings self_play_settings;
    // Note: game_type is passed separately to the manager
    int num_games = getIntConfigValue(config, "self_play_num_games", 100);
    self_play_settings.num_parallel_games = getIntConfigValue(config, "self_play_num_parallel_games", 8);
    self_play_settings.high_temperature = getFloatConfigValue(config, "self_play_temperature", 1.0f);
    self_play_settings.low_temperature = 0.0f;  // Greedy after threshold
    self_play_settings.temperature_threshold = getIntConfigValue(config, "self_play_temperature_moves", 20);
    
    // MCTS settings
    self_play_settings.mcts_settings.num_simulations = getIntConfigValue(config, "mcts_num_simulations", 800);
    self_play_settings.mcts_settings.num_threads = getIntConfigValue(config, "mcts_num_threads", 4);
    self_play_settings.mcts_settings.batch_size = getIntConfigValue(config, "mcts_batch_size", 8);
    self_play_settings.mcts_settings.virtual_loss = getFloatConfigValue(config, "mcts_virtual_loss", 3.0f);
    self_play_settings.mcts_settings.exploration_constant = getFloatConfigValue(config, "mcts_exploration_constant", 1.25f);
    self_play_settings.mcts_settings.batch_timeout = std::chrono::milliseconds(getIntConfigValue(config, "mcts_batch_timeout_ms", 5));
    self_play_settings.mcts_settings.progressive_widening_c = getFloatConfigValue(config, "mcts_progressive_widening_base", 1.0f);
    // Note: progressive_widening_alpha doesn't exist, using progressive_widening_c for both values
    // self_play_settings.mcts_settings.progressive_widening_alpha = getFloatConfigValue(config, "mcts_progressive_widening_scale", 0.0f);
    self_play_settings.mcts_settings.use_transposition_table = config["mcts_use_transposition_table"] == "true";
    // Virtual loss is always enabled, just set the value
    self_play_settings.mcts_settings.virtual_loss = getFloatConfigValue(config, "mcts_virtual_loss", 3.0f);
    self_play_settings.mcts_settings.add_dirichlet_noise = config["mcts_use_dirichlet_noise"] == "true";
    self_play_settings.mcts_settings.dirichlet_alpha = getFloatConfigValue(config, "mcts_dirichlet_alpha", 0.3f);
    self_play_settings.mcts_settings.dirichlet_epsilon = getFloatConfigValue(config, "mcts_dirichlet_epsilon", 0.25f);
    
    // Model path
    std::string model_path = config["model_path"];
    if (!std::filesystem::exists(model_path)) {
        LOG_SYSTEM_ERROR("Model file not found: {}", model_path);
        return;
    }
    
    // Create neural network configuration
    alphazero::nn::NeuralNetworkConfig nn_config;
    nn_config.input_channels = input_channels;
    nn_config.board_size = board_size;
    nn_config.num_res_blocks = num_res_blocks;
    nn_config.num_filters = num_filters;
    
    try {
        // Create multi-instance neural network manager
        int num_instances = self_play_settings.num_parallel_games;
        LOG_SYSTEM_INFO("Creating {} independent neural network instances", num_instances);
        
        auto nn_manager = std::make_shared<alphazero::mcts::MultiInstanceNNManager>(
            model_path, num_instances, nn_config
        );
        
        // Create optimized self-play manager
        LOG_SYSTEM_INFO("Initializing optimized self-play manager");
        alphazero::selfplay::OptimizedSelfPlayManager manager(nn_manager, self_play_settings, game_type);
        
        // Performance monitoring
        auto start_time = std::chrono::steady_clock::now();
        int total_games_generated = 0;
        
        // Main self-play loop
        while (total_games_generated < num_games) {
            int batch_size = std::min(
                self_play_settings.num_parallel_games * 2,
                num_games - total_games_generated
            );
            
            LOG_SYSTEM_INFO("Generating {} games...", batch_size);
            
            // Start async generation
            manager.startAsyncGeneration(batch_size);
            
            // Monitor progress
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
                // Collect completed games
                auto completed_games = manager.collectCompletedGames();
                if (!completed_games.empty()) {
                    total_games_generated += completed_games.size();
                    
                    // Save games
                    for (const auto& game : completed_games) {
                        // Save game record (implement based on your needs)
                        LOG_SYSTEM_DEBUG("Game completed: {} ({} moves)", game.game_id, game.states.size());
                    }
                    
                    // Print progress
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                    double games_per_sec = total_games_generated / elapsed;
                    
                    LOG_SYSTEM_INFO("Progress: {}/{} games ({:.2f} games/sec)", 
                                   total_games_generated, num_games, games_per_sec);
                    
                    // Check if batch is complete
                    if (completed_games.size() >= static_cast<size_t>(batch_size)) {
                        break;
                    }
                }
                
                // Print memory statistics
                auto mem_stats = alphazero::utils::ThreadLocalMemoryManager::getGlobalStats();
                LOG_SYSTEM_INFO("Memory - CPU: {} MB GPU: {} MB", 
                               mem_stats.cpu_allocated / (1024*1024), 
                               mem_stats.gpu_allocated / (1024*1024));
            }
        }
        
        // Final statistics
        manager.printStatistics();
        
        auto total_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        
        LOG_SYSTEM_INFO("=== Final Statistics ===");
        LOG_SYSTEM_INFO("Total games: {}", total_games_generated);
        LOG_SYSTEM_INFO("Total time: {} seconds", total_time);
        LOG_SYSTEM_INFO("Games/sec: {:.2f}", total_games_generated / total_time);
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Error during self-play: {}", e.what());
    }
}

int main(int argc, char* argv[]) {
    // Initialize logging
    alphazero::utils::Logger::init();
    
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <command> --config <config_file>" << std::endl;
        std::cout << "Commands: self-play-optimized" << std::endl;
        return 1;
    }
    
    std::string command = argv[1];
    std::string config_path;
    
    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }
    
    if (config_path.empty()) {
        LOG_SYSTEM_ERROR("Config file not specified");
        return 1;
    }
    
    if (command == "self-play-optimized") {
        runOptimizedSelfPlay(config_path);
    } else {
        LOG_SYSTEM_ERROR("Unknown command: {}", command);
        alphazero::utils::Logger::shutdown();
        return 1;
    }
    
    // Properly shutdown logging before exit
    alphazero::utils::Logger::shutdown();
    return 0;
}