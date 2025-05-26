// src/cli/omoknuni_cli_final_parallel.cpp - True parallel implementation
#include <iostream>
#include <memory>
#include <thread>
#include <future>
#include <chrono>
#include <csignal>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <deque>

#include "cli/cli_manager.h"
#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include "utils/advanced_memory_monitor.h"
#include "core/game_export.h"

using namespace alphazero;

// Global shutdown flag
std::atomic<bool> g_shutdown_requested(false);

// Global mutex for GPU initialization
std::mutex g_gpu_init_mutex;

// Signal handler
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
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

/**
 * Thread-safe game collector
 * Collects completed games from multiple workers
 */
class GameCollector {
public:
    void addGames(const std::vector<selfplay::GameData>& games) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& game : games) {
            games_.push_back(game);
        }
        total_games_ += games.size();
        cv_.notify_all();
    }
    
    std::vector<selfplay::GameData> collectBatch(size_t batch_size, 
                                                  std::chrono::milliseconds timeout,
                                                  bool all_games_complete = false) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto deadline = std::chrono::steady_clock::now() + timeout;
        cv_.wait_until(lock, deadline, [this, batch_size, all_games_complete] {
            return games_.size() >= batch_size || g_shutdown_requested.load() || 
                   (all_games_complete && !games_.empty());
        });
        
        std::vector<selfplay::GameData> batch;
        size_t count = std::min(batch_size, games_.size());
        for (size_t i = 0; i < count; ++i) {
            batch.push_back(std::move(games_.front()));
            games_.pop_front();
        }
        
        return batch;
    }
    
    size_t getTotalGames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_games_;
    }
    
    bool hasGames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !games_.empty();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<selfplay::GameData> games_;
    std::atomic<size_t> total_games_{0};
};

/**
 * Worker thread for parallel self-play
 * Each worker has its own neural network instance
 */
class SelfPlayWorker {
public:
    SelfPlayWorker(int id,
                   const std::string& model_path,
                   int input_channels,
                   int board_size,
                   int num_res_blocks,
                   int num_filters,
                   const selfplay::SelfPlaySettings& settings,
                   core::GameType game_type,
                   std::shared_ptr<GameCollector> collector)
        : id_(id),
          model_path_(model_path),
          input_channels_(input_channels),
          board_size_(board_size),
          num_res_blocks_(num_res_blocks),
          num_filters_(num_filters),
          settings_(settings),
          game_type_(game_type),
          collector_(collector) {}
    
    void run(int games_per_worker) {
        try {
            // Stagger worker starts slightly to avoid GPU contention
            std::this_thread::sleep_for(std::chrono::milliseconds(id_ * 100));
            
            
            // Synchronized GPU initialization
            std::shared_ptr<nn::NeuralNetwork> network;
            {
                std::lock_guard<std::mutex> lock(g_gpu_init_mutex);
                
                // Synchronize CUDA to ensure clean state
                if (torch::cuda::is_available()) {
                    // Note: torch::cuda::set_device is not available in all PyTorch versions
                    // Just synchronize to ensure clean state
                    torch::cuda::synchronize();
                }
                
                // Each worker loads its own neural network instance
                network = nn::NeuralNetworkFactory::loadResNet(
                    model_path_, input_channels_, board_size_, 
                    num_res_blocks_, num_filters_
                );
                
                if (!network) {
                    LOG_SYSTEM_ERROR("Worker {} failed to load neural network", id_);
                    return;
                }
                
            }
            
            // Additional delay after GPU init
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Create worker's own self-play manager
            selfplay::SelfPlayManager manager(network, settings_);
            
            int games_generated = 0;
            
            // Generate games
            while (games_generated < games_per_worker && !g_shutdown_requested.load()) {
                // Generate one game at a time for better debugging
                int batch_size = 1;
                
                auto batch_start = std::chrono::steady_clock::now();
                
                try {
                    auto games = manager.generateGames(game_type_, batch_size, board_size_);
                    auto batch_end = std::chrono::steady_clock::now();
                
                if (!games.empty()) {
                    // Add to collector
                    collector_->addGames(games);
                    games_generated += games.size();
                    
                    auto batch_time = std::chrono::duration_cast<std::chrono::seconds>(
                        batch_end - batch_start).count();
                }
                } catch (const std::exception& e) {
                    LOG_SYSTEM_ERROR("Worker {} failed to generate game: {}", id_, e.what());
                    break;
                }
            }
            
            
        } catch (const std::exception& e) {
            LOG_SYSTEM_ERROR("Worker {} error: {}", id_, e.what());
        }
    }
    
private:
    int id_;
    std::string model_path_;
    int input_channels_;
    int board_size_;
    int num_res_blocks_;
    int num_filters_;
    selfplay::SelfPlaySettings settings_;
    core::GameType game_type_;
    std::shared_ptr<GameCollector> collector_;
};

// Parallel self-play command
int runSelfPlay(const std::vector<std::string>& args) {
    
    if (args.size() < 1) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_final self-play <config.yaml>");
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
    
    // Override board size for Chess
    if (game_type == core::GameType::CHESS) {
        board_size = 8;
    }
    
    // Neural network configuration
    int input_channels = config["input_channels"].as<int>(17);
    int num_res_blocks = config["num_res_blocks"].as<int>(10);
    int num_filters = config["num_filters"].as<int>(64);
    std::string model_path = config["model_path"].as<std::string>("models/model.pt");
    
    // MCTS settings
    mcts::MCTSSettings mcts_settings;
    mcts_settings.num_simulations = config["mcts_simulations"].as<int>(400);
    mcts_settings.num_threads = config["mcts_threads_per_engine"].as<int>(1); // Per-engine threads
    mcts_settings.batch_size = config["mcts_batch_size"].as<int>(128);
    mcts_settings.batch_timeout = std::chrono::milliseconds(
        config["mcts_batch_timeout_ms"].as<int>(5)
    );
    mcts_settings.exploration_constant = config["mcts_c_puct"].as<float>(1.4f);
    mcts_settings.virtual_loss = config["mcts_virtual_loss"].as<int>(3);
    mcts_settings.use_transposition_table = config["mcts_enable_transposition"].as<bool>(true);
    
    // Self-play settings
    selfplay::SelfPlaySettings sp_settings;
    sp_settings.mcts_settings = mcts_settings;
    sp_settings.num_parallel_games = 1; // Each worker handles 1 game at a time
    sp_settings.temperature_threshold = config["mcts_temp_threshold"].as<int>(30);
    sp_settings.high_temperature = config["mcts_temperature"].as<float>(1.0f);
    sp_settings.low_temperature = 0.1f;
    
    // Parallel settings
    int num_workers = config["num_parallel_workers"].as<int>(4);
    int num_games = config["num_games"].as<int>(100);
    int save_interval = config["save_interval"].as<int>(10);
    std::string output_dir = config["output_dir"].as<std::string>("data/self_play_games");
    
    // Calculate games per worker
    int games_per_worker = (num_games + num_workers - 1) / num_workers;
    
    
    try {
        // Create output directory
        std::filesystem::create_directories(output_dir);
        
        // Setup signal handler
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        
        // Pre-initialize the AggressiveMemoryManager before any workers start
        // This avoids contention when multiple threads try to get the singleton instance
        auto& memory_manager = mcts::AggressiveMemoryManager::getInstance();
        
        // Pre-warm neural network to avoid initial contention
        {
            auto warmup_network = nn::NeuralNetworkFactory::loadResNet(
                model_path, input_channels, board_size, num_res_blocks, num_filters
            );
            
            if (warmup_network) {
                // Create dummy state and do one inference
                std::unique_ptr<core::IGameState> dummy_state;
                if (game_type == core::GameType::GOMOKU) {
                    dummy_state = std::make_unique<games::gomoku::GomokuState>(board_size);
                }
                
                if (dummy_state) {
                    std::vector<std::unique_ptr<core::IGameState>> batch;
                    batch.push_back(std::move(dummy_state));
                    auto outputs = warmup_network->inference(batch);
                }
            }
        }
        
        // Create game collector
        auto collector = std::make_shared<GameCollector>();
        
        // Progress tracking
        auto start_time = std::chrono::steady_clock::now();
        
        // Create and start worker threads
        std::vector<std::thread> worker_threads;
        std::vector<std::unique_ptr<SelfPlayWorker>> workers;
        
        
        for (int i = 0; i < num_workers; ++i) {
            workers.push_back(std::make_unique<SelfPlayWorker>(
                i, model_path, input_channels, board_size,
                num_res_blocks, num_filters, sp_settings,
                game_type, collector
            ));
        }
        
        // Start threads after all workers are created
        for (int i = 0; i < num_workers; ++i) {
            try {
                worker_threads.emplace_back([worker = workers[i].get(), games_per_worker]() {
                    try {
                        worker->run(games_per_worker);
                    } catch (const std::exception& e) {
                        LOG_SYSTEM_ERROR("Worker thread exception: {}", e.what());
                    } catch (...) {
                        LOG_SYSTEM_ERROR("Worker thread unknown exception");
                    }
                });
            } catch (const std::exception& e) {
                LOG_SYSTEM_ERROR("Failed to create worker thread {}: {}", i, e.what());
                // Signal shutdown and clean up existing threads
                g_shutdown_requested.store(true);
                for (auto& t : worker_threads) {
                    if (t.joinable()) t.join();
                }
                throw;
            }
        }
        
        // Saver thread - saves games as they complete
        std::thread saver_thread([&]() {
            int total_saved = 0;
            
            while (total_saved < num_games && !g_shutdown_requested.load()) {
                // Collect a batch of games - use min(save_interval, remaining games) to avoid waiting forever
                int remaining_games = num_games - total_saved;
                int batch_size = std::min(save_interval, remaining_games);
                
                // Use shorter timeout and allow partial batch collection
                auto games = collector->collectBatch(batch_size, 
                                                    std::chrono::milliseconds(1000));
                
                if (!games.empty()) {
                    // Save games directly without SelfPlayManager
                    // Create a timestamp for the batch
                    auto now = std::chrono::system_clock::now();
                    auto time_t = std::chrono::system_clock::to_time_t(now);
                    std::stringstream ss;
                    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                    
                    // Save each game in the batch
                    for (size_t i = 0; i < games.size(); ++i) {
                        std::string filename = output_dir + "/game_" + ss.str() + 
                                             "_" + std::to_string(total_saved + i) + ".json";
                        
                        std::ofstream file(filename);
                        if (file.is_open()) {
                            // Simple JSON serialization
                            file << "{\n";
                            file << "  \"game_type\": " << static_cast<int>(games[i].game_type) << ",\n";
                            file << "  \"board_size\": " << games[i].board_size << ",\n";
                            file << "  \"winner\": " << games[i].winner << ",\n";
                            file << "  \"total_time_ms\": " << games[i].total_time_ms << ",\n";
                            file << "  \"game_id\": \"" << games[i].game_id << "\",\n";
                            file << "  \"moves\": [";
                            for (size_t j = 0; j < games[i].moves.size(); ++j) {
                                file << games[i].moves[j];
                                if (j < games[i].moves.size() - 1) file << ", ";
                            }
                            file << "],\n";
                            file << "  \"policies\": [\n";
                            for (size_t j = 0; j < games[i].policies.size(); ++j) {
                                file << "    [";
                                for (size_t k = 0; k < games[i].policies[j].size(); ++k) {
                                    file << games[i].policies[j][k];
                                    if (k < games[i].policies[j].size() - 1) file << ", ";
                                }
                                file << "]";
                                if (j < games[i].policies.size() - 1) file << ",";
                                file << "\n";
                            }
                            file << "  ]\n";
                            file << "}\n";
                            file.close();
                        }
                    }
                    
                    total_saved += games.size();
                    
                    // Progress update
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        current_time - start_time).count();
                    
                    if (elapsed > 0) {
                        float games_per_sec = static_cast<float>(total_saved) / elapsed;
                    }
                }
            }
        });
        
        // Wait for all workers to complete
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        
        // Signal saver to stop and wait
        g_shutdown_requested.store(true);
        if (saver_thread.joinable()) {
            saver_thread.join();
        }
        
        // Ensure all workers are properly destroyed before exiting
        workers.clear();
        
        // Give time for any background threads to finish
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Final statistics
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        size_t total_games = collector->getTotalGames();
        
        if (total_elapsed > 0) {
            // Calculate average games per second
            float avg_games_per_sec = static_cast<float>(total_games) / total_elapsed;
            std::cout << "\nFinal statistics: " << total_games << " games generated in " 
                      << total_elapsed << " seconds (" << avg_games_per_sec << " games/sec)" << std::endl;
        }
        
        // Ensure all background threads are properly cleaned up
        // Stop the AggressiveMemoryManager monitoring thread before exit
        mcts::AggressiveMemoryManager::getInstance().shutdown();
        
        // Give GPU time to finish any pending operations
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
        
        // Give a bit more time for any background threads to finish cleanly
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Self-play failed: {}", e.what());
        g_shutdown_requested.store(true);
        return 1;
    }
}

// Training command (placeholder)
int runTraining(const std::vector<std::string>&) {
    return 0;
}

// Evaluation command (placeholder)
int runEvaluation(const std::vector<std::string>&) {
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize logging with SYNCHRONOUS mode for thread safety
    utils::Logger::init("logs", 
                       spdlog::level::info,  // console level
                       spdlog::level::debug, // file level
                       10485760,             // 10MB max file size
                       3,                    // max 3 files
                       false);               // SYNCHRONOUS logging
    
    int result = 0;
    try {
        // Create CLI manager
        cli::CLIManager cli_manager;
        
        // Add commands
        cli_manager.addCommand("self-play", "Run true parallel self-play", runSelfPlay);
        cli_manager.addCommand("train", "Train model (not yet implemented)", runTraining);
        cli_manager.addCommand("eval", "Evaluate model (not yet implemented)", runEvaluation);
        
        // Execute
        result = cli_manager.run(argc, argv);
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Fatal error: {}", e.what());
        result = 1;
    }
    
    // Ensure all singletons are properly shut down before exit
    try {
        // Shutdown the AggressiveMemoryManager singleton
        mcts::AggressiveMemoryManager::getInstance().shutdown();
        
        // Give time for threads to finish
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } catch (...) {
        // Ignore exceptions during cleanup
    }
    
    // Clean shutdown of logger
    utils::Logger::shutdown();
    
    // Exit cleanly
    std::exit(result);
}