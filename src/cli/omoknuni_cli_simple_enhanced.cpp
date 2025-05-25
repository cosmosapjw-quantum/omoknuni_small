// src/cli/omoknuni_cli_simple_enhanced.cpp
// Simplified enhanced version that integrates key optimizations with existing code

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
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <thread>
#include <chrono>

using namespace alphazero;

// Simple dynamic batch sizing based on queue depth
class SimpleDynamicBatchManager {
public:
    int calculateOptimalBatchSize(int queue_depth, int base_batch_size) {
        // Simple heuristic: increase batch size with queue depth
        if (queue_depth < 10) return std::max(1, base_batch_size / 2);
        if (queue_depth < 50) return base_batch_size;
        if (queue_depth < 100) return std::min(32, base_batch_size * 2);
        return std::min(64, base_batch_size * 4);
    }
};

void runSimpleEnhancedSelfPlay(const YAML::Node& config) {
    LOG_SYSTEM_INFO("Starting SIMPLE ENHANCED self-play with key optimizations");
    
    // Register games
    core::GameRegistry::instance().registerGame(
        core::GameType::GOMOKU,
        []() { return std::make_unique<games::gomoku::GomokuState>(); }
    );
    core::GameRegistry::instance().registerGame(
        core::GameType::CHESS,
        []() { return std::make_unique<games::chess::ChessState>(); }
    );
    core::GameRegistry::instance().registerGame(
        core::GameType::GO,
        []() { return std::make_unique<games::go::GoState>(); }
    );
    
    // Parse game type
    std::string game_type_str = config["game_type"].as<std::string>();
    core::GameType game_type;
    if (game_type_str == "gomoku") {
        game_type = core::GameType::GOMOKU;
    } else if (game_type_str == "chess") {
        game_type = core::GameType::CHESS;
    } else if (game_type_str == "go") {
        game_type = core::GameType::GO;
    } else {
        throw std::runtime_error("Unknown game type: " + game_type_str);
    }
    
    // Self-play settings
    selfplay::OptimizedSelfPlayManager::Settings settings;
    settings.num_actors = config["num_actors"].as<int>(8);
    settings.num_parallel_games = config["num_parallel_games"].as<int>(8);
    
    // MCTS settings
    settings.mcts_settings.num_simulations = config["mcts_num_simulations"].as<int>(800);
    settings.mcts_settings.num_threads = config["mcts_num_threads"].as<int>(8);
    settings.mcts_settings.batch_size = config["mcts_batch_size"].as<int>(16);
    settings.mcts_settings.batch_timeout = std::chrono::milliseconds(
        config["mcts_batch_timeout_ms"].as<int>(5)
    );
    settings.mcts_settings.virtual_loss = config["mcts_virtual_loss"].as<float>(3.0f);
    settings.mcts_settings.exploration_constant = config["mcts_exploration_constant"].as<float>(1.0f);
    settings.mcts_settings.dirichlet_alpha = config["mcts_dirichlet_alpha"].as<float>(0.3f);
    settings.mcts_settings.dirichlet_epsilon = config["mcts_dirichlet_epsilon"].as<float>(0.25f);
    
    // Enable advanced features
    settings.mcts_settings.use_true_parallel_search = true;
    settings.mcts_settings.use_direct_batching = true;
    settings.mcts_settings.enable_burst_mode = true;
    settings.mcts_settings.use_lock_free_queues = true;
    
    // Game settings
    settings.game_settings.high_temperature = config["high_temperature"].as<float>(1.0f);
    settings.game_settings.low_temperature = config["low_temperature"].as<float>(0.1f);
    settings.game_settings.temperature_threshold = config["temperature_threshold"].as<int>(30);
    
    // Neural network settings
    nn::ModelConfig model_config;
    model_config.input_channels = config["input_channels"].as<int>(17);
    model_config.board_size = config["board_size"].as<int>(15);
    model_config.num_res_blocks = config["num_res_blocks"].as<int>(10);
    model_config.num_filters = config["num_filters"].as<int>(64);
    model_config.policy_output_size = config["policy_output_size"].as<int>(225);
    model_config.model_path = config["model_path"].as<std::string>("models/model.pt");
    
    // Load optimized neural network
    auto network = nn::NeuralNetworkFactory::loadOptimizedResNet(model_config);
    
    // Key Enhancement 1: Use multi-instance NN manager
    auto nn_manager = std::make_shared<mcts::MultiInstanceNNManager>(
        settings.num_actors,  // One instance per actor
        network,
        true  // Enable CUDA graphs
    );
    
    // Set the multi-instance manager as the network provider
    settings.network_provider = [nn_manager](int actor_id) {
        return nn_manager->getInstance(actor_id % nn_manager->getNumInstances());
    };
    
    // Key Enhancement 2: Enable transposition table
    settings.mcts_settings.use_transposition_table = true;
    settings.mcts_settings.transposition_table_size_mb = 
        config["transposition_table_size_mb"].as<size_t>(1024);
    
    // Key Enhancement 3: Thread affinity
    settings.enable_thread_affinity = config["enable_thread_affinity"].as<bool>(true);
    
    // Create optimized self-play manager
    selfplay::OptimizedSelfPlayManager manager(settings, game_type);
    
    // Run self-play with progress monitoring
    int total_games = config["games_per_generation"].as<int>(10000);
    int batch_size = settings.num_parallel_games;
    int save_interval = config["save_every_n_games"].as<int>(100);
    
    LOG_SYSTEM_INFO("Starting enhanced self-play: {} total games", total_games);
    LOG_SYSTEM_INFO("Configuration:");
    LOG_SYSTEM_INFO("  - Actors: {}", settings.num_actors);
    LOG_SYSTEM_INFO("  - Parallel games: {}", settings.num_parallel_games);
    LOG_SYSTEM_INFO("  - MCTS simulations: {}", settings.mcts_settings.num_simulations);
    LOG_SYSTEM_INFO("  - Batch size: {}", settings.mcts_settings.batch_size);
    LOG_SYSTEM_INFO("  - NN instances: {}", nn_manager->getNumInstances());
    LOG_SYSTEM_INFO("  - Transposition table: {} MB", settings.mcts_settings.transposition_table_size_mb);
    
    auto start_time = std::chrono::steady_clock::now();
    int games_completed = 0;
    
    // Create simple batch manager
    SimpleDynamicBatchManager batch_manager;
    
    while (games_completed < total_games) {
        // Generate batch of games
        auto batch_start = std::chrono::steady_clock::now();
        auto games = manager.generateGames(
            std::min(batch_size, total_games - games_completed)
        );
        
        games_completed += games.size();
        
        // Calculate performance metrics
        auto current_time = std::chrono::steady_clock::now();
        auto batch_time = std::chrono::duration<float>(current_time - batch_start).count();
        auto total_time = std::chrono::duration<float>(current_time - start_time).count();
        
        float games_per_sec = games_completed / total_time;
        float batch_games_per_sec = games.size() / batch_time;
        
        LOG_SYSTEM_INFO("Progress: {}/{} games | {:.2f} games/sec (batch: {:.2f} games/sec)", 
                       games_completed, total_games, games_per_sec, batch_games_per_sec);
        
        // Log detailed statistics periodically
        if (games_completed % save_interval == 0) {
            manager.printStatistics();
            
            // Log NN instance statistics
            auto nn_stats = nn_manager->getStatistics();
            LOG_SYSTEM_INFO("Neural Network Statistics:");
            for (int i = 0; i < nn_manager->getNumInstances(); ++i) {
                LOG_SYSTEM_INFO("  Instance {}: {} inferences, {:.2f}ms avg latency",
                              i, nn_stats.instance_inference_counts[i],
                              nn_stats.instance_avg_latencies[i]);
            }
            LOG_SYSTEM_INFO("  Total GPU utilization: {:.1f}%", nn_stats.gpu_utilization * 100);
            
            // Save games if needed
            if (!games.empty()) {
                // TODO: Implement game saving
                LOG_SYSTEM_INFO("Saving {} games...", games.size());
            }
        }
        
        // Adjust batch size dynamically (simplified)
        if (games_per_sec < 50.0f && batch_size < 16) {
            batch_size = std::min(16, batch_size + 2);
            LOG_SYSTEM_INFO("Increasing parallel games to {}", batch_size);
        } else if (games_per_sec > 100.0f && batch_size > 4) {
            batch_size = std::max(4, batch_size - 2);
            LOG_SYSTEM_INFO("Decreasing parallel games to {}", batch_size);
        }
    }
    
    // Final statistics
    auto total_elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time
    ).count();
    
    LOG_SYSTEM_INFO("=== Enhanced Self-Play Complete ===");
    LOG_SYSTEM_INFO("Total games: {}", games_completed);
    LOG_SYSTEM_INFO("Total time: {:.1f} seconds", total_elapsed);
    LOG_SYSTEM_INFO("Average: {:.2f} games/second", games_completed / total_elapsed);
    
    manager.printStatistics();
}

int main(int argc, char* argv[]) {
    // Initialize logging
    utils::Logger::init();
    
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <command> --config <config_file>" << std::endl;
        std::cout << "Commands: self-play" << std::endl;
        return 1;
    }
    
    try {
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
        
        // Load configuration
        YAML::Node config = YAML::LoadFile(config_path);
        
        if (command == "self-play") {
            runSimpleEnhancedSelfPlay(config);
        } else {
            LOG_SYSTEM_ERROR("Unknown command: {}", command);
            utils::Logger::shutdown();
            return 1;
        }
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Fatal error: {}", e.what());
        utils::Logger::shutdown();
        return 1;
    }
    
    // Cleanup
    utils::Logger::shutdown();
    return 0;
}