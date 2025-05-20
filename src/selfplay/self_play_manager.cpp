// src/selfplay/self_play_manager.cpp
#include "selfplay/self_play_manager.h"
#include "mcts/mcts_engine.h"
#include "utils/debug_monitor.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"
#include "utils/memory_tracker.h"
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "core/game_export.h"

// PyTorch/LibTorch headers for CUDA memory management
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace alphazero {
namespace selfplay {

using json = nlohmann::json;

// Helper function to create a readable timestamp
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

SelfPlayManager::SelfPlayManager(std::shared_ptr<nn::NeuralNetwork> neural_net,
                               const SelfPlaySettings& settings)
    : neural_net_(neural_net),
      settings_(settings),
      game_counter_(0) {

    // Check for valid neural network
    if (!neural_net_) {
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Log key configuration info
    std::cout << "SelfPlayManager: Created with sequential game generation and " 
              << settings_.mcts_settings.num_simulations 
              << " MCTS simulations" << std::endl;
    
    // Track initial memory
    // alphazero::utils::trackMemory("SelfPlayManager created");
    
    // Initialize random number generator
    if (settings_.random_seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(settings_.random_seed));
    }

    // Configure root parallelization in MCTS settings
    // auto mcts_settings = settings_.mcts_settings; // This line is unused and will be removed.
    
    // FIXED: Enable root parallelization as specified in config.yaml for better batching
    // Root parallelization is critical for proper batch formation
    // Now correctly apply the config settings to settings_.mcts_settings
    settings_.mcts_settings.use_root_parallelization = true;
    
    // Use the num_root_workers from config or set a reasonable default based on available cores
    if (settings_.mcts_settings.num_root_workers <= 0) {
        int available_cores = omp_get_max_threads();
        settings_.mcts_settings.num_root_workers = std::max(1, available_cores / 2);
    }
    
    std::cout << "SelfPlayManager: Using root parallelization with " 
              << settings_.mcts_settings.num_root_workers << " root workers per game - CRITICAL for batch formation" << std::endl;
    
    // Create a single shared evaluator for all engines
    auto notify_fn = [this]() {
        // Wake up the shared evaluator when there's work to do
        static std::atomic<long long> notify_count_selfplay_manager(0);
        long long current_notify_count = notify_count_selfplay_manager.fetch_add(1, std::memory_order_relaxed);
        if (current_notify_count < 20 || current_notify_count % 100 == 0) { // Log frequently at start, then less often
            std::cout << "SELFPLAY_MANAGER (notify_fn): Called (count: " << current_notify_count 
                      << "). Notifying shared_evaluator_@" << static_cast<void*>(shared_evaluator_.get()) << std::endl;
        }
        if (shared_evaluator_) {
            shared_evaluator_->notifyLeafAvailable();
        }
    };
    
    // Create the shared evaluator with the neural network
    auto evaluator_inference_fn = [this](const std::vector<std::unique_ptr<core::IGameState>>& states) -> std::vector<mcts::NetworkOutput> {
        return neural_net_->inference(states);
    };
    shared_evaluator_ = std::make_shared<mcts::MCTSEvaluator>(
        evaluator_inference_fn,
        settings_.mcts_settings.batch_size,
        settings_.mcts_settings.batch_timeout
    );
    
    // Configure the shared evaluator to use the shared queues
    // CRITICAL FIX: Implement the correct queue setup for the shared evaluator
    std::cout << "SelfPlayManager: Setting up shared queues for evaluator at addresses leaf_queue=" 
              << (void*)&shared_leaf_queue_ << ", result_queue=" << (void*)&shared_result_queue_ << std::endl;
    
    // Configure batch parameters first before setting external queues
    // Configure batch parameters for efficient GPU utilization
    mcts::BatchParameters batch_params;
    // Use batch size from settings with reasonable minimums
    batch_params.optimal_batch_size = settings_.mcts_settings.batch_size;
    // Set min viable size to ~75% of target size, but at least 16
    batch_params.minimum_viable_batch_size = std::max(16UL, batch_params.optimal_batch_size * 3 / 4);
    // Set fallback size to ~25% of target size, but at least 8
    batch_params.minimum_fallback_batch_size = std::max(8UL, batch_params.optimal_batch_size / 4);
    // Use timeout from settings for responsiveness, with reasonable minimums
    batch_params.max_wait_time = std::max(
        std::chrono::milliseconds(50), 
        settings_.mcts_settings.batch_timeout
    );
    // Very short additional wait time to balance responsiveness and efficiency
    batch_params.additional_wait_time = std::chrono::milliseconds(10);
    
    // Set batch parameters first
    shared_evaluator_->setBatchParameters(batch_params);
    
    // Then set external queues
    shared_evaluator_->setExternalQueues(&shared_leaf_queue_, &shared_result_queue_, notify_fn);
    
    // Verify that batch accumulator is created
    if (auto* batch_accumulator = shared_evaluator_->getBatchAccumulator()) {
        std::cout << "SelfPlayManager: Batch accumulator exists, isRunning=" 
                  << (batch_accumulator->isRunning() ? "yes" : "no") << std::endl;
        
        // Ensure the accumulator is started
        if (!batch_accumulator->isRunning()) {
            std::cout << "SelfPlayManager: Starting batch accumulator..." << std::endl;
            batch_accumulator->start();
        }
    } else {
        std::cout << "SelfPlayManager: WARNING: Batch accumulator is null!" << std::endl;
    }
    
    // Start the shared evaluator
    shared_evaluator_->start();
    
    std::cout << "SelfPlayManager: Shared evaluator configured and started" << std::endl;
    
    // Create a single MCTS engine with root parallelization
    // try {
    //     engine_ = std::make_unique<mcts::MCTSEngine>(neural_net_, mcts_settings);
    //     
    //     // Set up shared external queues
    //     if (engine_) {
    //         engine_->setSharedExternalQueues(&shared_leaf_queue_, &shared_result_queue_, notify_fn);
    //         std::cout << "SelfPlayManager: Created MCTS engine with shared evaluator" << std::endl;
    //     } else {
    //         throw std::runtime_error("Engine creation failed - null engine");
    //     }
    // } catch (const std::exception& e) {
    //     std::cerr << "ERROR creating MCTS engine: " << e.what() << std::endl;
    //     throw std::runtime_error("Failed to create MCTS engine: " + std::string(e.what()));
    // }

    // Determine the number of engines based on parallel games setting or available cores
    num_engines_resolved_ = settings_.num_parallel_games > 0 ? settings_.num_parallel_games : omp_get_max_threads();
    num_engines_resolved_ = std::max(1, num_engines_resolved_); // Ensure at least one engine
    
    std::cout << "SelfPlayManager: Initializing " << num_engines_resolved_ << " MCTS engine(s) for parallel games." << std::endl;

    engines_.reserve(num_engines_resolved_);
    for (int i = 0; i < num_engines_resolved_; ++i) {
        try {
            auto engine = std::make_unique<mcts::MCTSEngine>(neural_net_, settings_.mcts_settings);
            if (engine) {
                // Set up batch parameters first
                // Use consistent and efficient batch parameters for each engine
                mcts::BatchParameters batch_params;
                // Use batch size from settings
                batch_params.optimal_batch_size = settings_.mcts_settings.batch_size;
                // Set min viable size to ~75% of target size, but at least 16
                batch_params.minimum_viable_batch_size = std::max(16UL, batch_params.optimal_batch_size * 3 / 4);
                // Set fallback size to ~25% of target size, but at least 8
                batch_params.minimum_fallback_batch_size = std::max(8UL, batch_params.optimal_batch_size / 4);
                // Use timeout from settings with reasonable minimum
                batch_params.max_wait_time = std::max(
                    std::chrono::milliseconds(50), 
                    settings_.mcts_settings.batch_timeout
                );
                // Moderate additional wait time
                batch_params.additional_wait_time = std::chrono::milliseconds(10);
                // Collection batch size set to optimal batch size
                batch_params.max_collection_batch_size = batch_params.optimal_batch_size;
                
                // Configure the evaluator with batch parameters
                if (auto* evaluator = engine->getEvaluator()) {
                    evaluator->setBatchParameters(batch_params);
                } else {
                    std::cerr << "Engine " << i << " has null evaluator" << std::endl;
                }
                
                // Set up shared external queues
                engine->setSharedExternalQueues(&shared_leaf_queue_, &shared_result_queue_, notify_fn);
                
                std::cout << "Engine " << i << " configured with batch params: optimal=" 
                          << batch_params.optimal_batch_size 
                          << ", min_viable=" << batch_params.minimum_viable_batch_size
                          << ", min_fallback=" << batch_params.minimum_fallback_batch_size
                          << ", max_wait=" << batch_params.max_wait_time.count() << "ms" 
                          << std::endl;
                engines_.push_back(std::move(engine));
            } else {
                throw std::runtime_error("Engine creation failed - null engine in loop");
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR creating MCTS engine instance " << i << ": " << e.what() << std::endl;
            throw std::runtime_error("Failed to create MCTS engine instance: " + std::string(e.what()));
        }
    }
    std::cout << "SelfPlayManager: Created " << engines_.size() << " MCTS engine(s) sharing one evaluator." << std::endl;

    // GPU Warm-up sequence on the engine
    try {
        core::GameType warmup_game_type = core::GameType::GOMOKU; 
        int warmup_board_size = 15; // Matches user's config.yaml

        std::unique_ptr<core::IGameState> warmup_state_template = createGame(warmup_game_type, warmup_board_size, 0);

        if (warmup_state_template && neural_net_) {
            // Warm-up with batch size 1
            std::vector<std::unique_ptr<core::IGameState>> warmup_batch_1;
            try {
                warmup_batch_1.push_back(warmup_state_template->clone());
                if (!warmup_batch_1.empty() && warmup_batch_1.front()) {
                    auto outputs1 = neural_net_->inference(warmup_batch_1);
                } else {
                    std::cerr << "SelfPlayManager: Failed to create or clone dummy state for batch 1 warm-up." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "SelfPlayManager: Warning - warm-up inference (batch 1) failed: " << e.what() << std::endl;
            }

            // Warm-up with a typical batch size (e.g., mcts.batch_size)
            size_t N_batch_size = static_cast<size_t>(settings_.mcts_settings.batch_size);
            if (N_batch_size > 0) {
                std::vector<std::unique_ptr<core::IGameState>> warmup_batch_N;
                try {
                    for(size_t j = 0; j < N_batch_size; ++j) {
                        warmup_batch_N.push_back(warmup_state_template->clone());
                    }
                    if (!warmup_batch_N.empty() && warmup_batch_N.size() == N_batch_size) {
                        auto outputsN = neural_net_->inference(warmup_batch_N);
                    } else {
                        std::cerr << "SelfPlayManager: Failed to create or clone dummy states for batch " << N_batch_size << " warm-up." << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "SelfPlayManager: Warning - warm-up inference (batch " << N_batch_size << ") failed: " << e.what() << std::endl;
                }
            }
        } else {
            if (!warmup_state_template) {
                std::cerr << "SelfPlayManager: Warning - could not create dummy state for warm-up. Game type: " << static_cast<int>(warmup_game_type) << ", Board size: " << warmup_board_size << std::endl;
            }
            if (!neural_net_) {
                std::cerr << "SelfPlayManager: Warning - neural_net_ is null, skipping warm-up" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during warm-up sequence: " << e.what() << std::endl;
        // Don't throw - warm-up is optional
    }
    
    
}

SelfPlayManager::~SelfPlayManager() {
    // Stop the shared evaluator first
    if (shared_evaluator_) {
        try {
            shared_evaluator_->stop();
        } catch (const std::exception& e) {
            std::cerr << "Warning: Exception during evaluator cleanup: " << e.what() << std::endl;
        }
    }
    
    // Reset game counter
    game_counter_ = 0;
    
    // Force GPU memory cleanup before engines are destroyed
    if (neural_net_) {
        neural_net_.reset();
    }
    
    // Clean up the MCTS engines
    for (auto& engine_ptr : engines_) { // Use a different variable name to avoid conflict if engine_ was a member
        try {
            if (engine_ptr) engine_ptr.reset();
        } catch (const std::exception& e) {
            std::cerr << "Warning: Exception during engine cleanup: " << e.what() << std::endl;
        }
    }
    engines_.clear();
}

std::vector<alphazero::selfplay::GameData> alphazero::selfplay::SelfPlayManager::generateGames(core::GameType game_type,
                                                    int num_games,
                                                    int board_size) {
    // Sanity check inputs
    if (num_games <= 0) {
        throw std::invalid_argument("Number of games must be positive");
    }
    
    if (board_size < 0) {
        throw std::invalid_argument("Board size cannot be negative");
    }
    
    if (engines_.empty()) {
        throw std::runtime_error("MCTS engines not initialized");
    }
    
    
    
    // Initialize result vector
    std::vector<alphazero::selfplay::GameData> games;
    games.reserve(num_games);
    
    // Create a timestamp base for game IDs
    std::string timestamp_base = getCurrentTimestamp();
    
    // Generate games in parallel with OpenMP for better CPU/GPU utilization
    std::cout << "Generating " << num_games << " self-play games using root parallelization..." << std::endl;
    // Thread-safe game collection
    std::mutex games_mutex;
    std::atomic<int> completed_games(0);
    
    // Use OpenMP for parallel game generation (one game per thread)
    const int max_threads = settings_.num_parallel_games > 0 ? 
                          std::min(settings_.num_parallel_games, omp_get_max_threads()) : 
                          omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
    for (int i = 0; i < num_games; i++) {
        // Thread ID for debugging
        int thread_id = omp_get_thread_num();
        mcts::MCTSEngine& current_engine = *engines_[thread_id % engines_.size()];
        
        #pragma omp critical(debug_print)
        {
            std::cout << "Thread " << thread_id << " starting game " << i << " using engine " << (thread_id % engines_.size()) << std::endl;
        }
        
        // Periodic memory cleanup per thread
        if (i > 0 && i % 50 == 0) {
            #pragma omp critical
            {
                utils::GameStatePoolManager::getInstance().clearAllPools();
                std::cout << "Progress: " << completed_games.load() << "/" << num_games << " games" << std::endl;
            }
        }
        
        std::string game_id = timestamp_base + "_" + std::to_string(i);
        
        try {
            auto game_data = generateGame(current_engine, game_type, board_size, game_id);
            
            // Add game to results if it has at least one move
            if (!game_data.moves.empty()) {
                #pragma omp critical(games_add)
                {
                    games.push_back(std::move(game_data));
                }
                
                int comp = completed_games.fetch_add(1, std::memory_order_relaxed) + 1;
                
                // Progress update and memory cleanup
                if (comp % 10 == 0) {
                    #pragma omp critical(progress_print)
                    {
                        std::cout << "Completed " << comp << "/" << num_games << " games" << std::endl;
                        
                        // More aggressive cleanup between games
                        utils::GameStatePoolManager::getInstance().clearAllPools();
                        if (torch::cuda::is_available()) {
                            try {
                                torch::cuda::synchronize();
                                c10::cuda::CUDACachingAllocator::emptyCache();
                            } catch (...) {}
                        }
                    }
                }
            } else {
                std::cerr << "Failed to generate game " << i << ": no moves were made" << std::endl;
                // Don't increment the counter for empty games
            }
        } catch (const std::exception& e) {
            std::cerr << "Error generating game " << i << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    std::cout << "Successfully generated " << games.size() << " games" << std::endl;
    return games;
}

alphazero::selfplay::GameData alphazero::selfplay::SelfPlayManager::generateGame(mcts::MCTSEngine& engine, core::GameType game_type, int board_size,
                                      const std::string& game_id) {
    
    // Verify engine exists (engine is now passed by reference)
    // No need to check engines_.empty() here as a valid reference is expected

    // Don't start evaluator here - it will be configured with external queues during search
    

    // Select a random starting position
    std::uniform_int_distribution<int> pos_dist(0, settings_.num_start_positions - 1);
    int position_id = pos_dist(rng_);
    

    // Create game
    auto game = createGame(game_type, board_size, position_id);
    if (!game) {
        std::cerr << "SelfPlayManager: Failed to create game state for game_id: " << game_id << std::endl;
        throw std::runtime_error("Failed to create game state");
    }
    
    
    // Verify game state is valid
    
    try {
        if (!game->validate()) {
            std::cerr << "SelfPlayManager::generateGame - Game state failed validation" << std::endl;
            throw std::runtime_error("Invalid game state after creation");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "SelfPlayManager::generateGame - Exception during game state validation: " << e.what() << std::endl;
        throw;
    }

    // Prepare game data
    alphazero::selfplay::GameData data;
    data.game_type = game_type;
    data.board_size = board_size;
    data.winner = 0;  // Default to draw
    data.game_id = game_id;

    // Log game creation once
    int thread_id = omp_get_thread_num();
    std::cout << "Game " << game_id << ": Started (thread " << thread_id << ")" << std::endl;
    
    // Calculate max moves if not specified
    int max_moves = settings_.max_moves;
    if (max_moves <= 0) {
        // Default to board size squared or 1000, whichever is smaller
        max_moves = std::min(board_size * board_size, 1000);
    }
    
    // Start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Debug initial state
    std::vector<int> legal_moves = game->getLegalMoves();
    if (game->isTerminal()) {
        std::cerr << "Game " << game_id << ": WARNING - Initial state is terminal!" << std::endl;
        std::cerr << "  Game result: " << static_cast<int>(game->getGameResult()) << std::endl;
        std::cerr << "  Legal moves: " << legal_moves.size() << std::endl;
        std::cerr << "  Current player: " << game->getCurrentPlayer() << std::endl;
    } else {
        std::cout << "Game " << game_id << ": Initial state has " << legal_moves.size() << " legal moves" << std::endl;
        // For Gomoku, let's check what the legal moves are
        if (game_type == core::GameType::GOMOKU && legal_moves.size() < 10) {
            std::cout << "  Legal moves: ";
            for (int move : legal_moves) {
                std::cout << move << " ";
            }
            std::cout << std::endl;
            
            // Check board size and center position
            int board_size = dynamic_cast<games::gomoku::GomokuState*>(game.get())->getBoardSize();
            int center = (board_size * board_size) / 2;
            std::cout << "  Board size: " << board_size << "x" << board_size 
                     << ", Center position: " << center << std::endl;
        }
    }
    
    // Play until terminal state or max moves
    int move_count = 0;
    
    try {
        while (!game->isTerminal() && move_count < max_moves) {
            // Set temperature based on move number
            float temperature;
            if (move_count < settings_.temperature_threshold) {
                temperature = settings_.high_temperature;
            } else {
                temperature = settings_.low_temperature;
            }
            
            // Update engine settings
            auto current_settings = engine.getSettings();
            current_settings.temperature = temperature;
            current_settings.add_dirichlet_noise = settings_.add_dirichlet_noise &&
                                                 move_count == 0;  // Only on first move
            engine.updateSettings(current_settings);

            // Run search with timeout detection and robust error handling to prevent stalling
            mcts::SearchResult result;
            
            // Don't start evaluator here - it will be configured with external queues in the search method
            
            
            // Validate the game state before search
            // Debug: Validating game state before search
            // std::cout << "SelfPlayManager::generateGame - Validating game state before search for move " 
            //          << move_count << " in game " << game_id << std::endl;
            try {
                if (!game->validate()) {
                    std::cerr << "SelfPlayManager::generateGame - ERROR: Game state validation failed before search in game " 
                             << game_id << std::endl;
                    throw std::runtime_error("Game state validation failed");
                }
                
            } catch (const std::exception& e) {
                std::cerr << "SelfPlayManager::generateGame - Exception validating game state before search: " 
                         << e.what() << std::endl;
                throw;
            }
            
            // Debug: Starting MCTS search
            // std::cout << "SelfPlayManager::generateGame - Starting MCTS search for move " << move_count 
            //          << " in game " << game_id << std::endl;
            
            
            // Debug: Creating search future
            // std::cout << "SelfPlayManager::generateGame - Creating search future for move " << move_count 
            //          << " in game " << game_id << std::endl;
            
            // Start the search directly - NO std::async
            try {
                
                // Track memory every 10 moves
                if (move_count % 10 == 0) {
                    // alphazero::utils::trackMemory("Before search - Game " + game_id + ", Move " + std::to_string(move_count));
                }
                
                std::cout << "=============================================================" << std::endl;
                std::cout << "SelfPlayManager::generateGame - Starting MCTS search for game " << game_id 
                         << ", move " << move_count << std::endl;
                
                // Log key configuration for debugging
                auto engine_settings = engine.getSettings();
                std::cout << "MCTS CONFIG:"
                         << "\n  - num_simulations: " << engine_settings.num_simulations
                         << "\n  - batch_size: " << engine_settings.batch_size
                         << "\n  - use_root_parallelization: " << (engine_settings.use_root_parallelization ? "true" : "false")
                         << "\n  - num_root_workers: " << engine_settings.num_root_workers
                         << "\n  - temperature: " << engine_settings.temperature
                         << std::endl;
                
                // Begin search
                auto search_start_time = std::chrono::steady_clock::now();
                
                // Add diagnostic logging before search to track batch metrics
                if (auto* evaluator = engine.getEvaluator()) {
                    if (auto* batch_acc = evaluator->getBatchAccumulator()) {
                        auto [avg_batch, total_batches, timeouts, optimal] = batch_acc->getStats();
                        std::cout << "📊 BATCH METRICS BEFORE SEARCH: avg_size=" << avg_batch
                                << ", total_batches=" << total_batches
                                << ", timeouts=" << timeouts
                                << ", optimal_batches=" << optimal << std::endl;
                    }
                }
                
                // Run the search
                result = engine.search(*game);
                
                auto search_end_time = std::chrono::steady_clock::now();
                auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    search_end_time - search_start_time).count();
                
                // Add diagnostic logging after search to track batch metrics
                if (auto* evaluator = engine.getEvaluator()) {
                    if (auto* batch_acc = evaluator->getBatchAccumulator()) {
                        auto [avg_batch, total_batches, timeouts, optimal] = batch_acc->getStats();
                        std::cout << "📊 BATCH METRICS AFTER SEARCH: avg_size=" << avg_batch
                                << ", total_batches=" << total_batches
                                << ", timeouts=" << timeouts
                                << ", optimal_batches=" << optimal << std::endl;
                    }
                    
                    // Get evaluator stats
                    std::cout << "🔢 EVALUATOR STATS: avg_batch_size=" << evaluator->getAverageBatchSize()
                            << ", avg_latency=" << evaluator->getAverageBatchLatency().count() << "ms"
                            << ", total_evals=" << evaluator->getTotalEvaluations() << std::endl;
                }
                
                std::cout << "SelfPlayManager::generateGame - Completed MCTS search for game " << game_id 
                         << ", move " << move_count << " in " << search_duration << "ms" << std::endl;
                std::cout << "=============================================================" << std::endl;
                
                // Log every 10th move to reduce verbosity
                if (move_count % 10 == 0) {
                    std::cout << "Game " << game_id << ": Move " << move_count 
                             << " completed (action: " << result.action << ")" << std::endl;
                    // alphazero::utils::trackMemory("After search - Game " + game_id + ", Move " + std::to_string(move_count));
                    
                    // Periodic memory cleanup
                    utils::GameStatePoolManager::getInstance().clearAllPools();
                    if (torch::cuda::is_available()) {
                        try {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        } catch (...) {}
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "SelfPlayManager::generateGame - Exception during direct search: " << e.what() << std::endl;
                throw;
            } catch (...) {
                std::cerr << "SelfPlayManager::generateGame - Unknown exception during direct search" << std::endl;
                throw;
            }
                
            // Verify the search result
            if (result.action < 0) {
                std::cerr << "WARNING: Game " << game_id << ": MCTS search returned invalid action: " 
                         << result.action << std::endl;
                
                // Try to recover - get a valid move from legal moves
                std::vector<int> legal_moves = game->getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                    std::cout << "Game " << game_id << ": Recovered with legal move: " << result.action << std::endl;
                } else {
                    throw std::runtime_error("No legal moves available");
                }
            }

            // Store move and policy
            data.moves.push_back(result.action);
            data.policies.push_back(result.probabilities);
            // std::cout << "Game " << game_id << ": Stored move and policy. Making move on game state..." << std::endl;

            // Make move
            game->makeMove(result.action);
            move_count++;
            // std::cout << "Game " << game_id << ": Move " << move_count << " made. Checking if terminal..." << std::endl;

            // Print progress for moves multiple of 10
            if (move_count % 10 == 0) {
                // std::cout << "Game " << game_id << ": Completed " << move_count 
                         // << " moves" << std::endl;
            }
            
            // Check if terminal
            if (game->isTerminal()) {
                // std::cout << "Game " << game_id << ": Game is terminal." << std::endl;
                auto game_result = game->getGameResult();
                if (game_result == core::GameResult::WIN_PLAYER1) {
                    data.winner = 1;
                } else if (game_result == core::GameResult::WIN_PLAYER2) {
                    data.winner = 2;
                } else {
                    data.winner = 0; // Draw
                }
            }
        }
    }
    catch (const std::exception& e) {
        // Log error but don't throw - return partial game data if we have moves
        std::cerr << "Error during game " << game_id << ": " << e.what() << std::endl;
        
        if (data.moves.empty()) {
            // If no moves were made, this is a critical error
            throw;
        }
    }
    
    // End time
    auto end_time = std::chrono::steady_clock::now();
    data.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Log completion
    // std::cout << "Game " << game_id << ": Completed with "
             // << data.moves.size() << " moves in "
             // << (data.total_time_ms / 1000.0) << " seconds" << std::endl;
    
    return data;
}

std::unique_ptr<core::IGameState> alphazero::selfplay::SelfPlayManager::createGame(core::GameType game_type, 
                                                            int board_size, 
                                                            int position_id) {
    // Use specific parameters from GameConfig
    
    switch (game_type) {
        case core::GameType::CHESS: {
            // Use Chess960 based on config and position_id
            bool use_chess960 = settings_.game_config.chess_use_chess960 && 
                              position_id >= 0 && position_id < 960;
            return std::make_unique<games::chess::ChessState>(use_chess960, "", position_id);
        }
        case core::GameType::GO: {
            // Standard board sizes for Go are 9x9, 13x13, or 19x19
            int go_board_size = 19; // Default
            if (board_size == 9 || board_size == 13 || board_size == 19) {
                go_board_size = board_size;
            }
            
            // Use config values for komi, rules, and superko
            float komi = settings_.game_config.go_komi;
            bool chinese_rules = settings_.game_config.go_chinese_rules;
            bool enforce_superko = settings_.game_config.go_enforce_superko;
            
            return std::make_unique<games::go::GoState>(go_board_size, komi, chinese_rules, enforce_superko);
        }
        case core::GameType::GOMOKU: {
            // Default Gomoku board size is 15x15
            int gomoku_board_size = (board_size > 0) ? board_size : 15;
            
            // Use config values for game rules
            bool use_renju = settings_.game_config.gomoku_use_renju;
            bool use_omok = settings_.game_config.gomoku_use_omok;
            bool use_pro_long_opening = settings_.game_config.gomoku_use_pro_long_opening;
            
            // Seed is used for any randomized aspects of the game
            int seed = position_id;
            
            return std::make_unique<games::gomoku::GomokuState>(
                gomoku_board_size, use_renju, use_omok, seed, use_pro_long_opening);
        }
        default:
            throw std::runtime_error("Unsupported game type: " + 
                                     core::gameTypeToString(game_type));
    }
}

void alphazero::selfplay::SelfPlayManager::saveGames(const std::vector<alphazero::selfplay::GameData>& games, 
                              const std::string& output_dir, 
                              const std::string& format) {
    if (games.empty()) {
        std::cerr << "Warning: No games to save" << std::endl;
        return;
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Count of successfully saved games
    int saved_count = 0;
    
    // Save each game
    for (size_t i = 0; i < games.size(); ++i) {
        const auto& game = games[i];
        
        // Skip empty games
        if (game.moves.empty()) {
            std::cerr << "Warning: Skipping empty game " << game.game_id << std::endl;
            continue;
        }
        
        // Create filename
        std::string filename = output_dir + "/game_" + game.game_id;
        
        bool success = false;
        
        try {
            if (format == "json") {
                filename += ".json";
                
                // Create JSON
                json j;
                j["game_type"] = static_cast<int>(game.game_type);
                j["board_size"] = game.board_size;
                j["winner"] = game.winner;
                j["moves"] = game.moves;
                j["policies"] = game.policies;
                j["total_time_ms"] = game.total_time_ms;
                j["game_id"] = game.game_id;
                
                // Write to file
                std::ofstream file(filename);
                if (file) {
                    file << j.dump(2);
                    success = !file.fail();
                }
            } else if (format == "binary") {
                filename += ".bin";
                
                // Write to binary file
                std::ofstream file(filename, std::ios::binary);
                if (file) {
                    // Write header
                    int32_t magic = 0x41525A47;  // "AZGM" (AlphaZero Game)
                    int32_t version = 1;
                    int32_t game_type = static_cast<int32_t>(game.game_type);
                    int32_t board_size = static_cast<int32_t>(game.board_size);
                    int32_t winner = static_cast<int32_t>(game.winner);
                    int32_t num_moves = static_cast<int32_t>(game.moves.size());
                    int64_t total_time_ms = game.total_time_ms;
                    
                    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
                    file.write(reinterpret_cast<const char*>(&game_type), sizeof(game_type));
                    file.write(reinterpret_cast<const char*>(&board_size), sizeof(board_size));
                    file.write(reinterpret_cast<const char*>(&winner), sizeof(winner));
                    file.write(reinterpret_cast<const char*>(&num_moves), sizeof(num_moves));
                    file.write(reinterpret_cast<const char*>(&total_time_ms), sizeof(total_time_ms));
                    
                    // Write game ID
                    int32_t game_id_length = static_cast<int32_t>(game.game_id.length());
                    file.write(reinterpret_cast<const char*>(&game_id_length), sizeof(game_id_length));
                    file.write(game.game_id.c_str(), game_id_length);
                    
                    // Write moves
                    for (int move : game.moves) {
                        int32_t move_int = static_cast<int32_t>(move);
                        file.write(reinterpret_cast<const char*>(&move_int), sizeof(move_int));
                    }
                    
                    // Write policies
                    for (const auto& policy : game.policies) {
                        int32_t policy_size = static_cast<int32_t>(policy.size());
                        file.write(reinterpret_cast<const char*>(&policy_size), sizeof(policy_size));
                        file.write(reinterpret_cast<const char*>(policy.data()), policy_size * sizeof(float));
                    }
                    
                    success = !file.fail();
                }
            } else {
                throw std::runtime_error("Unsupported format: " + format);
            }
            
            if (success) {
                saved_count++;
            } else {
                std::cerr << "Error writing game " << game.game_id << " to " << filename << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving game " << game.game_id << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "Successfully saved " << saved_count << " out of " 
             << games.size() << " games to " << output_dir 
             << " in " << format << " format" << std::endl;
}

std::vector<alphazero::selfplay::GameData> alphazero::selfplay::SelfPlayManager::loadGames(const std::string& input_dir, 
                                               const std::string& format) {
    std::vector<alphazero::selfplay::GameData> games;
    
    // Check if directory exists
    if (!std::filesystem::exists(input_dir)) {
        throw std::runtime_error("Input directory does not exist: " + input_dir);
    }
    
    // Count of successfully loaded games
    int loaded_count = 0;
    int error_count = 0;
    
    // Iterate through files in directory
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        std::string extension = entry.path().extension().string();
        
        if ((format == "json" && extension == ".json") || 
            (format == "binary" && extension == ".bin")) {
            
            std::string filename = entry.path().string();
            
            try {
                alphazero::selfplay::GameData game;
                bool success = false;
                
                if (format == "json") {
                    // Read JSON file
                    std::ifstream file(filename);
                    if (!file) {
                        throw std::runtime_error("Could not open file: " + filename);
                    }
                    
                    json j;
                    file >> j;
                    
                    // Parse game data
                    game.game_type = static_cast<core::GameType>(j["game_type"].get<int>());
                    game.board_size = j["board_size"].get<int>();
                    game.winner = j["winner"].get<int>();
                    game.moves = j["moves"].get<std::vector<int>>();
                    game.policies = j["policies"].get<std::vector<std::vector<float>>>();
                    game.total_time_ms = j["total_time_ms"].get<int64_t>();
                    game.game_id = j["game_id"].get<std::string>();
                    
                    success = true;
                } else if (format == "binary") {
                    // Read binary file
                    std::ifstream file(filename, std::ios::binary);
                    if (!file) {
                        throw std::runtime_error("Could not open file: " + filename);
                    }
                    
                    // Read header
                    int32_t magic, version, game_type, board_size, winner, num_moves;
                    int64_t total_time_ms;
                    
                    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    file.read(reinterpret_cast<char*>(&version), sizeof(version));
                    file.read(reinterpret_cast<char*>(&game_type), sizeof(game_type));
                    file.read(reinterpret_cast<char*>(&board_size), sizeof(board_size));
                    file.read(reinterpret_cast<char*>(&winner), sizeof(winner));
                    file.read(reinterpret_cast<char*>(&num_moves), sizeof(num_moves));
                    file.read(reinterpret_cast<char*>(&total_time_ms), sizeof(total_time_ms));
                    
                    // Validate magic number
                    if (magic != 0x41525A47) {
                        throw std::runtime_error("Invalid file format: " + filename);
                    }
                    
                    // Read game ID
                    int32_t game_id_length;
                    file.read(reinterpret_cast<char*>(&game_id_length), sizeof(game_id_length));
                    std::string game_id(game_id_length, '\0');
                    file.read(&game_id[0], game_id_length);
                    
                    // Create game data
                    game.game_type = static_cast<core::GameType>(game_type);
                    game.board_size = board_size;
                    game.winner = winner;
                    game.total_time_ms = total_time_ms;
                    game.game_id = game_id;
                    
                    // Read moves
                    game.moves.resize(num_moves);
                    for (int i = 0; i < num_moves; ++i) {
                        int32_t move;
                        file.read(reinterpret_cast<char*>(&move), sizeof(move));
                        game.moves[i] = move;
                    }
                    
                    // Read policies
                    game.policies.resize(num_moves);
                    for (int i = 0; i < num_moves; ++i) {
                        int32_t policy_size;
                        file.read(reinterpret_cast<char*>(&policy_size), sizeof(policy_size));
                        game.policies[i].resize(policy_size);
                        file.read(reinterpret_cast<char*>(game.policies[i].data()), 
                                 policy_size * sizeof(float));
                    }
                    
                    success = !file.fail();
                }
                
                if (success) {
                    games.push_back(std::move(game));
                    loaded_count++;
                } else {
                    std::cerr << "Error reading game data from " << filename << std::endl;
                    error_count++;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading game from " << filename << ": " << e.what() << std::endl;
                error_count++;
            }
        }
    }
    
    std::cout << "Successfully loaded " << loaded_count << " games from " 
             << input_dir << " (" << error_count << " errors)" << std::endl;
    
    return games;
}

std::pair<std::vector<std::vector<std::vector<std::vector<float>>>>, 
         std::pair<std::vector<std::vector<float>>, std::vector<float>>> 
alphazero::selfplay::SelfPlayManager::convertToTrainingExamples(const std::vector<alphazero::selfplay::GameData>& games) {
    // Count total training examples
    size_t total_examples = 0;
    for (const auto& game : games) {
        total_examples += game.moves.size();
    }
    
    // Prepare output vectors
    std::vector<std::vector<std::vector<std::vector<float>>>> states;
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    
    states.reserve(total_examples);
    policies.reserve(total_examples);
    values.reserve(total_examples);
    
    // Process each game
    for (const auto& game : games) {
        // Skip invalid games
        if (game.moves.empty() || game.policies.empty()) {
            std::cerr << "Warning: Skipping invalid game " << game.game_id 
                     << " during training data conversion" << std::endl;
            continue;
        }
        
        try {
            // Create game instance
            auto game_state = core::GameFactory::createGame(game.game_type);
            if (!game_state) {
                throw std::runtime_error("Failed to create game state for game type: " + 
                                        core::gameTypeToString(game.game_type));
            }
            
            // Process each move
            for (size_t i = 0; i < game.moves.size(); ++i) {
                // Get state tensor
                states.push_back(game_state->getEnhancedTensorRepresentation());
                
                // Get policy
                if (i < game.policies.size()) {
                    policies.push_back(game.policies[i]);
                } else {
                    // Handle missing policy (shouldn't happen with correct data)
                    std::vector<float> uniform_policy(game_state->getActionSpaceSize(), 
                                                    1.0f / game_state->getActionSpaceSize());
                    policies.push_back(uniform_policy);
                }
                
                // Calculate value based on winner
                float value;
                int current_player = game_state->getCurrentPlayer();
                
                if (game.winner == 0) {  // Draw
                    value = 0.0f;
                } else if (game.winner == current_player) {  // Win
                    value = 1.0f;
                } else {  // Loss
                    value = -1.0f;
                }
                
                values.push_back(value);
                
                // Make move
                game_state->makeMove(game.moves[i]);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error converting game " << game.game_id 
                     << " to training examples: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Converted " << games.size() << " games to " 
             << states.size() << " training examples" << std::endl;
    
    return {states, {policies, values}};
}

const alphazero::selfplay::SelfPlaySettings& alphazero::selfplay::SelfPlayManager::getSettings() const {
    return settings_;
}

void alphazero::selfplay::SelfPlayManager::updateSettings(const alphazero::selfplay::SelfPlaySettings& settings) {
    settings_ = settings;
    
    // Update MCTS settings in the engines
    // Make sure root parallelization is enabled for maximum performance
    auto mcts_settings_update = settings_.mcts_settings; // Use a new variable to avoid confusion
    mcts_settings_update.use_root_parallelization = true;
    
    // Adjust num_root_workers based on available cores and parallel games
    const int available_cores_update = omp_get_max_threads(); // Use a new variable
    const int parallel_games_update = settings_.num_parallel_games > 0 ? settings_.num_parallel_games : 1; // Use a new variable
    int cores_per_game_update = std::max(1, available_cores_update / parallel_games_update); // Use a new variable
    mcts_settings_update.num_root_workers = cores_per_game_update;
    
    for (auto& engine_ptr : engines_) { // Use a different variable name
        if (engine_ptr) {
            engine_ptr->updateSettings(mcts_settings_update);
        }
    }
}

} // namespace selfplay
} // namespace alphazero