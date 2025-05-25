// src/cli/omoknuni_cli_safe.cpp - Thread-safe optimized CLI
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "cli/cli_manager.h"
#include "mcts/mcts_engine.h"
#include "mcts/multi_instance_nn_manager.h"
#include "nn/neural_network_factory.h"
#include "nn/optimized_resnet_model.h"
#include "selfplay/self_play_manager.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include "utils/thread_local_memory_manager.h"
#include "core/game_export.h"

using namespace alphazero;

// Global flags for safe shutdown
std::atomic<bool> g_shutdown_requested(false);
std::atomic<bool> g_workers_finished(false);
std::atomic<int> g_active_workers(0);

// Signal handler
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_SYSTEM_INFO("Shutdown signal received. Initiating graceful shutdown...");
        g_shutdown_requested.store(true);
    }
}

// Safe logging wrapper that checks shutdown state
template<typename... Args>
void safeLogInfo(const std::string& format, Args&&... args) {
    if (!g_shutdown_requested.load()) {
        try {
            LOG_SYSTEM_INFO(format, std::forward<Args>(args)...);
        } catch (...) {
            // Ignore logging errors during shutdown
        }
    }
}

// Parse game type
core::GameType parseGameType(const std::string& game_str) {
    if (game_str == "gomoku") return core::GameType::GOMOKU;
    if (game_str == "chess") return core::GameType::CHESS;
    if (game_str == "go") return core::GameType::GO;
    throw std::runtime_error("Unknown game type: " + game_str);
}

// Worker thread with proper lifecycle management
class SafeSelfPlayWorker {
public:
    SafeSelfPlayWorker(int id, 
                      std::shared_ptr<nn::NeuralNetwork> network,
                      const mcts::MCTSSettings& mcts_settings,
                      core::GameType game_type,
                      int board_size)
        : id_(id), 
          network_(network),
          mcts_settings_(mcts_settings),
          game_type_(game_type),
          board_size_(board_size),
          games_completed_(0) {
        g_active_workers.fetch_add(1);
    }
    
    ~SafeSelfPlayWorker() {
        g_active_workers.fetch_sub(1);
    }
    
    void run(int num_games) {
        try {
            safeLogInfo("Worker {} starting with {} games", id_, num_games);
            
            // Create MCTS engine
            auto engine = std::make_unique<mcts::MCTSEngine>(network_, mcts_settings_);
            
            for (int game_idx = 0; game_idx < num_games && !g_shutdown_requested.load(); ++game_idx) {
                playOneGame(engine.get(), game_idx);
                games_completed_++;
                
                if (game_idx % 10 == 0) {
                    safeLogInfo("Worker {} progress: {}/{} games", id_, game_idx + 1, num_games);
                }
            }
            
            safeLogInfo("Worker {} completed {} games", id_, games_completed_);
            
        } catch (const std::exception& e) {
            std::cerr << "Worker " << id_ << " error: " << e.what() << std::endl;
        }
    }
    
    int getGamesCompleted() const { return games_completed_; }
    
private:
    void playOneGame(mcts::MCTSEngine* engine, int game_idx) {
        // Create initial game state
        std::unique_ptr<core::IGameState> state;
        
        switch (game_type_) {
            case core::GameType::GOMOKU:
                state = std::make_unique<games::gomoku::GomokuState>(board_size_);
                break;
            case core::GameType::CHESS:
                state = std::make_unique<games::chess::ChessState>();
                break;
            case core::GameType::GO:
                state = std::make_unique<games::go::GoState>(board_size_, 7.5f);
                break;
            default:
                throw std::runtime_error("Unsupported game type");
        }
        
        // Play game
        int move_count = 0;
        while (!state->isTerminal() && move_count < 500 && !g_shutdown_requested.load()) {
            auto search_result = engine->search(*state);
            state->makeMove(search_result.action);
            move_count++;
        }
        
        // Log completion only if not shutting down
        if (!g_shutdown_requested.load()) {
            safeLogInfo("Worker {} completed game {} with {} moves", id_, game_idx, move_count);
        }
    }
    
    int id_;
    std::shared_ptr<nn::NeuralNetwork> network_;
    mcts::MCTSSettings mcts_settings_;
    core::GameType game_type_;
    int board_size_;
    std::atomic<int> games_completed_;
};

// Optimized self-play with safe shutdown
void runOptimizedSelfPlay(const std::string& config_path) {
    safeLogInfo("Starting safe optimized self-play");
    
    // Load configuration
    YAML::Node config = YAML::LoadFile(config_path);
    
    // Parse configuration
    auto game_type = parseGameType(config["game_type"].as<std::string>());
    int board_size = config["board_size"].as<int>(15);
    int num_games = config["num_games"].as<int>(100);
    int num_workers = config["num_parallel_workers"].as<int>(4);
    int games_per_worker = num_games / num_workers;
    
    // Neural network configuration
    int input_channels = config["input_channels"].as<int>(17);
    int num_res_blocks = config["num_res_blocks"].as<int>(10);
    int num_filters = config["num_filters"].as<int>(64);
    std::string model_path = config["model_path"].as<std::string>("models/model.pt");
    
    // MCTS settings
    mcts::MCTSSettings mcts_settings;
    mcts_settings.num_simulations = config["mcts_simulations"].as<int>(400);
    mcts_settings.num_threads = config["mcts_num_threads"].as<int>(1);
    mcts_settings.batch_size = config["mcts_batch_size"].as<int>(128);
    mcts_settings.batch_timeout = std::chrono::milliseconds(
        config["mcts_batch_timeout_ms"].as<int>(5));
    mcts_settings.exploration_constant = config["mcts_c_puct"].as<float>(1.4f);
    mcts_settings.virtual_loss = config["mcts_virtual_loss"].as<int>(3);
    
    safeLogInfo("Configuration loaded:");
    safeLogInfo("  - Workers: {}", num_workers);
    safeLogInfo("  - Games per worker: {}", games_per_worker);
    safeLogInfo("  - Total games: {}", num_games);
    safeLogInfo("  - MCTS simulations: {}", mcts_settings.num_simulations);
    
    // Create multi-instance NN manager
    auto nn_manager = std::make_shared<mcts::MultiInstanceNNManager>(
        num_workers,
        [&]() {
            return nn::NeuralNetworkFactory::loadOptimizedResNet(
                model_path, input_channels, board_size, num_res_blocks, num_filters);
        }
    );
    
    // Start workers
    std::vector<std::thread> worker_threads;
    std::vector<std::unique_ptr<SafeSelfPlayWorker>> workers;
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (int i = 0; i < num_workers; ++i) {
        auto network = nn_manager->getInstance(i);
        auto worker = std::make_unique<SafeSelfPlayWorker>(
            i, network, mcts_settings, game_type, board_size);
        
        worker_threads.emplace_back([&worker, games_per_worker]() {
            worker->run(games_per_worker);
        });
        
        workers.push_back(std::move(worker));
        
        // Stagger worker starts
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Monitor progress
    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        int total_completed = 0;
        for (const auto& worker : workers) {
            total_completed += worker->getGamesCompleted();
        }
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - start_time).count();
        
        if (elapsed > 0) {
            float games_per_sec = static_cast<float>(total_completed) / elapsed;
            safeLogInfo("Progress: {} / {} games completed ({:.2f} games/sec)",
                       total_completed, num_games, games_per_sec);
        }
        
        // Check if all games completed
        if (total_completed >= num_games) {
            g_shutdown_requested.store(true);
            break;
        }
    }
    
    // Wait for all workers to finish
    safeLogInfo("Waiting for workers to complete...");
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Final statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    int total_completed = 0;
    for (const auto& worker : workers) {
        total_completed += worker->getGamesCompleted();
    }
    
    safeLogInfo("Self-play completed!");
    safeLogInfo("  - Total games: {}", total_completed);
    safeLogInfo("  - Total time: {} seconds", total_elapsed);
    if (total_elapsed > 0) {
        safeLogInfo("  - Average: {:.2f} games/sec", 
                   static_cast<float>(total_completed) / total_elapsed);
    }
}

// Main function with proper lifecycle management
int main(int argc, char* argv[]) {
    // Set up signal handlers FIRST
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Initialize logging with SYNCHRONOUS mode for safety
    utils::Logger::init("logs", 
                       spdlog::level::info,  // console level
                       spdlog::level::debug, // file level
                       10485760,             // 10MB max file size
                       3,                    // max 3 files
                       false);               // SYNCHRONOUS logging for safety
    
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " self-play <config.yaml>" << std::endl;
            return 1;
        }
        
        std::string command = argv[1];
        std::string config_path = argv[2];
        
        if (command == "self-play") {
            runOptimizedSelfPlay(config_path);
        } else {
            std::cerr << "Unknown command: " << command << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        
        // Ensure clean shutdown
        g_shutdown_requested.store(true);
        
        // Wait for workers
        while (g_active_workers.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Shutdown logging
        utils::Logger::shutdown();
        return 1;
    }
    
    // Clean shutdown
    utils::Logger::shutdown();
    return 0;
}