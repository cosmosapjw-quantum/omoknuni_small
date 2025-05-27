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
#include <torch/torch.h>

#include "cli/cli_manager.h"
#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "utils/progress_bar.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/logger.h"
#include "utils/advanced_memory_monitor.h"
#include "utils/shutdown_manager.h"
#include "core/game_export.h"

using namespace alphazero;

// Global mutex for GPU initialization
std::mutex g_gpu_init_mutex;

// Enhanced signal handling
namespace {
    std::atomic<int> g_signal_count(0);
    std::atomic<bool> g_force_exit(false);
    std::vector<std::thread::id> g_active_threads;
    std::mutex g_thread_registry_mutex;
    std::condition_variable g_shutdown_cv;
    
    void registerThread(std::thread::id id) {
        std::lock_guard<std::mutex> lock(g_thread_registry_mutex);
        g_active_threads.push_back(id);
    }
    
    void unregisterThread(std::thread::id id) noexcept {
        try {
            std::lock_guard<std::mutex> lock(g_thread_registry_mutex);
            g_active_threads.erase(
                std::remove(g_active_threads.begin(), g_active_threads.end(), id),
                g_active_threads.end()
            );
            g_shutdown_cv.notify_all();
        } catch (...) {
            // Ignore exceptions during thread cleanup
        }
    }
    
    size_t getActiveThreadCount() {
        std::lock_guard<std::mutex> lock(g_thread_registry_mutex);
        return g_active_threads.size();
    }
}

// Enhanced signal handler
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        int count = ++g_signal_count;
        
        if (count == 1) {
            LOG_SYSTEM_INFO("Shutdown requested. Gracefully stopping all workers...");
            std::cout << "\n*** Shutdown requested. Gracefully stopping all workers... ***" << std::endl;
            std::cout << "*** Press Ctrl+C again to force immediate exit ***" << std::endl;
            utils::requestShutdown();
            
            // Start a watchdog thread for forced shutdown
            std::thread watchdog([]() {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                if (!g_force_exit.load()) {
                    LOG_SYSTEM_ERROR("Graceful shutdown timeout. Force exiting...");
                    std::cout << "\n*** Graceful shutdown timeout after 10 seconds. Force exiting... ***" << std::endl;
                    
                    // Try to clean up GPU resources
                    if (torch::cuda::is_available()) {
                        torch::cuda::synchronize();
                        // Note: empty_cache() not available in all PyTorch versions
                    }
                    
                    std::_Exit(1);
                }
            });
            watchdog.detach();
            
        } else if (count == 2) {
            LOG_SYSTEM_INFO("Force shutdown requested. Terminating immediately...");
            std::cout << "\n*** Force shutdown requested. Terminating immediately... ***" << std::endl;
            g_force_exit.store(true);
            
            // Force flush output
            std::cout.flush();
            std::cerr.flush();
            
            // Use abort() for immediate termination
            std::abort();
        } else if (count >= 3) {
            // Third Ctrl+C - absolutely force exit
            _exit(1);
        }
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
            return games_.size() >= batch_size || utils::isShutdownRequested() || 
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
                   std::shared_ptr<GameCollector> collector,
                   const std::string& network_type = "resnet",
                   const nn::DDWRandWireResNetConfig& ddw_config = nn::DDWRandWireResNetConfig())
        : id_(id),
          model_path_(model_path),
          input_channels_(input_channels),
          board_size_(board_size),
          num_res_blocks_(num_res_blocks),
          num_filters_(num_filters),
          settings_(settings),
          game_type_(game_type),
          collector_(collector),
          network_type_(network_type),
          ddw_config_(ddw_config) {}
    
    void run(int games_per_worker) {
        // Register this thread
        registerThread(std::this_thread::get_id());
        
        try {
            // Stagger worker starts to avoid GPU contention and model creation races
            std::this_thread::sleep_for(std::chrono::milliseconds(id_ * 500));
            
            
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
                if (network_type_ == "ddw_randwire") {
                    network = nn::NeuralNetworkFactory::loadDDWRandWireResNet(
                        model_path_, ddw_config_
                    );
                } else {
                    network = nn::NeuralNetworkFactory::loadResNet(
                        model_path_, input_channels_, board_size_, 
                        num_res_blocks_, num_filters_
                    );
                }
                
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
            while (games_generated < games_per_worker && !utils::isShutdownRequested()) {
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
        
        // Unregister thread before exit
        unregisterThread(std::this_thread::get_id());
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
    std::string network_type_;
    nn::DDWRandWireResNetConfig ddw_config_;
};

// Parallel self-play command
int runSelfPlay(const std::vector<std::string>& args) {
    
    if (args.size() < 1) {
        LOG_SYSTEM_ERROR("Usage: omoknuni_cli_final self-play <config.yaml> [--verbose]");
        return 1;
    }
    
    // Check for verbose flag
    bool verbose = false;
    for (const auto& arg : args) {
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
            break;
        }
    }
    
    // Enable verbose logging if requested
    auto& progress_manager = utils::SelfPlayProgressManager::getInstance();
    progress_manager.setVerboseLogging(verbose);
    
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
    std::string network_type = config["network_type"].as<std::string>("resnet");
    
    // DDW-RandWire-ResNet configuration
    nn::DDWRandWireResNetConfig ddw_config;
    
    if (network_type == "ddw_randwire") {
        ddw_config.input_channels = input_channels;
        ddw_config.output_size = board_size * board_size;
        ddw_config.board_height = board_size;
        ddw_config.board_width = board_size;
        ddw_config.channels = config["ddw_channels"].as<int>(128);
        ddw_config.num_blocks = config["ddw_num_blocks"].as<int>(20);
        ddw_config.use_dynamic_routing = config["ddw_dynamic_routing"].as<bool>(true);
        
        // RandWire configuration
        ddw_config.randwire_config.num_nodes = config["ddw_num_nodes"].as<int>(32);
        ddw_config.randwire_config.seed = config["ddw_seed"].as<int>(-1);
        
        // Graph generation method
        std::string graph_method = config["ddw_graph_method"].as<std::string>("watts_strogatz");
        if (graph_method == "watts_strogatz") {
            ddw_config.randwire_config.method = nn::GraphGenMethod::WATTS_STROGATZ;
            ddw_config.randwire_config.p = config["ddw_ws_p"].as<double>(0.75);
            ddw_config.randwire_config.k = config["ddw_ws_k"].as<int>(4);
        } else if (graph_method == "erdos_renyi") {
            ddw_config.randwire_config.method = nn::GraphGenMethod::ERDOS_RENYI;
            ddw_config.randwire_config.edge_prob = config["ddw_er_edge_prob"].as<double>(0.1);
        } else if (graph_method == "barabasi_albert") {
            ddw_config.randwire_config.method = nn::GraphGenMethod::BARABASI_ALBERT;
            ddw_config.randwire_config.m = config["ddw_ba_m"].as<int>(5);
        }
        
        ddw_config.randwire_config.use_dynamic_routing = ddw_config.use_dynamic_routing;
    }
    
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
            std::shared_ptr<nn::NeuralNetwork> warmup_network;
            if (network_type == "ddw_randwire") {
                warmup_network = nn::NeuralNetworkFactory::loadDDWRandWireResNet(
                    model_path, ddw_config
                );
            } else {
                warmup_network = nn::NeuralNetworkFactory::loadResNet(
                    model_path, input_channels, board_size, num_res_blocks, num_filters
                );
            }
            
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
        
        // Pre-create and save the model in the main thread to ensure consistency
        if (network_type == "ddw_randwire") {
            std::ifstream model_check(model_path);
            if (!model_check.good()) {
                LOG_SYSTEM_INFO("Creating initial DDW-RandWire-ResNet model in main thread...");
                auto initial_model = nn::NeuralNetworkFactory::createDDWRandWireResNet(ddw_config);
                try {
                    std::dynamic_pointer_cast<nn::DDWRandWireResNet>(initial_model)->save(model_path);
                    LOG_SYSTEM_INFO("Initial model saved to: {}", model_path);
                } catch (const std::exception& e) {
                    LOG_SYSTEM_ERROR("Failed to save initial model: {}", e.what());
                }
            }
        }
        
        // Create and start worker threads
        std::vector<std::thread> worker_threads;
        std::vector<std::unique_ptr<SelfPlayWorker>> workers;
        
        
        for (int i = 0; i < num_workers; ++i) {
            workers.push_back(std::make_unique<SelfPlayWorker>(
                i, model_path, input_channels, board_size,
                num_res_blocks, num_filters, sp_settings,
                game_type, collector, network_type, ddw_config
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
                utils::requestShutdown();
                for (auto& t : worker_threads) {
                    if (t.joinable()) t.join();
                }
                throw;
            }
        }
        
        // Saver thread - saves games as they complete
        std::thread saver_thread([&]() {
            // Register saver thread
            registerThread(std::this_thread::get_id());
            
            int total_saved = 0;
            
            while (total_saved < num_games && !utils::isShutdownRequested()) {
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
            
            // Unregister saver thread
            unregisterThread(std::this_thread::get_id());
        });
        
        // Wait for all workers to complete
        if (utils::isShutdownRequested()) {
            // Enhanced thread cleanup with timeout only during shutdown
            auto cleanup_start = std::chrono::steady_clock::now();
            const auto max_wait_time = std::chrono::seconds(5);
            
            for (size_t i = 0; i < worker_threads.size(); ++i) {
                if (worker_threads[i].joinable()) {
                    // Calculate remaining time
                    auto elapsed = std::chrono::steady_clock::now() - cleanup_start;
                    auto remaining = max_wait_time - elapsed;
                    
                    if (remaining > std::chrono::seconds(0)) {
                        // Try to join with timeout using condition variable
                        std::unique_lock<std::mutex> lock(g_thread_registry_mutex);
                        if (g_shutdown_cv.wait_for(lock, remaining, [&]() {
                            return !worker_threads[i].joinable() || 
                                   std::find(g_active_threads.begin(), g_active_threads.end(), 
                                           worker_threads[i].get_id()) == g_active_threads.end();
                        })) {
                            if (worker_threads[i].joinable()) {
                                worker_threads[i].join();
                            }
                        } else {
                            LOG_SYSTEM_ERROR("Worker thread {} failed to stop within timeout", i);
                        }
                    } else {
                        LOG_SYSTEM_ERROR("Timeout waiting for worker threads to stop");
                        break;
                    }
                }
            }
        } else {
            // Normal operation - wait indefinitely for threads to complete
            for (auto& thread : worker_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        
        // Signal saver to stop and wait
        utils::requestShutdown();
        if (saver_thread.joinable()) {
            // For normal operation, just wait for saver thread to complete
            saver_thread.join();
        }
        
        // Ensure all workers are properly destroyed before exiting
        workers.clear();
        
        // Wait for all threads to unregister (only log warnings if this was an interrupted shutdown)
        {
            std::unique_lock<std::mutex> lock(g_thread_registry_mutex);
            if (!g_shutdown_cv.wait_for(lock, std::chrono::seconds(2), []() {
                return g_active_threads.empty();
            })) {
                // Only log as error if shutdown was requested but threads are still active
                if (g_signal_count.load() > 0) {
                    LOG_SYSTEM_ERROR("Warning: {} threads still active after cleanup", g_active_threads.size());
                }
            }
        }
        
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
        
        // Comprehensive cleanup sequence
        LOG_SYSTEM_INFO("Starting cleanup sequence...");
        
        // 1. Stop the AggressiveMemoryManager monitoring thread
        mcts::AggressiveMemoryManager::getInstance().shutdown();
        
        // 2. Force collection of any pending games
        collector.reset();
        
        // 3. Synchronize and clear GPU resources
        if (torch::cuda::is_available()) {
            try {
                torch::cuda::synchronize();
                // Note: empty_cache() not available in all PyTorch versions
                // Manual cleanup would require lower-level CUDA API
            } catch (const std::exception& e) {
                LOG_SYSTEM_ERROR("GPU cleanup error: {}", e.what());
            }
        }
        
        // 4. Give time for any remaining cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        LOG_SYSTEM_INFO("Cleanup sequence completed");
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Self-play failed: {}", e.what());
        utils::requestShutdown();
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
    LOG_SYSTEM_INFO("Starting final cleanup...");
    
    try {
        // Set global shutdown flag
        utils::requestShutdown();
        
        // Wait for any remaining threads
        {
            std::unique_lock<std::mutex> lock(g_thread_registry_mutex);
            if (!g_shutdown_cv.wait_for(lock, std::chrono::seconds(3), []() {
                return g_active_threads.empty();
            })) {
                LOG_SYSTEM_ERROR("Warning: {} threads still active at exit", g_active_threads.size());
                
                // Force exit if threads won't stop
                if (g_active_threads.size() > 0) {
                    std::cout << "\n*** Forcing exit due to stuck threads ***" << std::endl;
                    
                    // Try GPU cleanup one more time
                    if (torch::cuda::is_available()) {
                        torch::cuda::synchronize();
                        // Note: empty_cache() not available in all PyTorch versions
                    }
                    
                    std::_Exit(result);
                }
            }
        }
        
        // Shutdown the AggressiveMemoryManager singleton
        mcts::AggressiveMemoryManager::getInstance().shutdown();
        
        // Final GPU cleanup
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
            // Note: empty_cache() not available in all PyTorch versions
        }
        
        // Give time for final cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } catch (...) {
        // Ignore exceptions during cleanup
    }
    
    // Clean shutdown of logger
    utils::Logger::shutdown();
    
    // Exit cleanly
    std::exit(result);
}