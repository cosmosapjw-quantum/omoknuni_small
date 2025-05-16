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
#include <future>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <condition_variable>
#include <nlohmann/json.hpp>
#include "core/game_export.h"

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
      active_games_(0) {

    // Check for valid neural network
    if (!neural_net_) {
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Log key configuration info
    std::cout << "SelfPlayManager: Created with " << settings_.num_parallel_games 
              << " parallel games, " << settings_.mcts_settings.num_simulations 
              << " MCTS simulations" << std::endl;
    
    // Track initial memory
    alphazero::utils::trackMemory("SelfPlayManager created");
    
    // Initialize random number generator
    if (settings_.random_seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(settings_.random_seed));
    }

    // Create MCTS engines for each worker thread with proper exception safety
    engines_.reserve(settings_.num_parallel_games);
    try {
        for (int i = 0; i < settings_.num_parallel_games; ++i) {
            
            try {
                auto engine = std::make_unique<mcts::MCTSEngine>(neural_net_, settings_.mcts_settings);
                
                // Important: Start the evaluator immediately to ensure neural network is initialized properly
                if (engine) {
                    // First, explicitly ensure the evaluator is started
                    
                    if (!engine->ensureEvaluatorStarted()) {
                        std::cerr << "ERROR: Failed to start evaluator for engine " << i << std::endl;
                        throw std::runtime_error("Failed to start evaluator for engine " + std::to_string(i));
                    }
                    
                
                // GPU Warm-up sequence
                
                // We need game_type and board_size. Assuming GOMOKU and 15x15 based on user's current run.
                // A more robust solution would pass these through SelfPlaySettings.
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
                        std::cerr << "SelfPlayManager: Warning - warm-up inference (batch 1) failed for engine " << i << ": " << e.what() << std::endl;
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
                            std::cerr << "SelfPlayManager: Warning - warm-up inference (batch " << N_batch_size << ") failed for engine " << i << ": " << e.what() << std::endl;
                        }
                    }
                    
                } else {
                    if (!warmup_state_template) {
                        std::cerr << "SelfPlayManager: Warning - could not create dummy state for warm-up for engine " << i << ". Game type: " << static_cast<int>(warmup_game_type) << ", Board size: " << warmup_board_size << std::endl;
                    }
                    if (!neural_net_) {
                         std::cerr << "SelfPlayManager: Warning - neural_net_ is null, skipping warm-up for engine " << i << std::endl;
                    }
                }
            }
            
            engines_.emplace_back(std::move(engine));
            
        } catch (const std::exception& e) {
            std::cerr << "Error creating MCTS engine " << i << ": " << e.what() << std::endl;
            throw;
            }
        }
    } catch (const std::exception& e) {
        // Clean up already created engines
        std::cerr << "Error creating engines: " << e.what() << std::endl;
        std::cerr << "Cleaning up " << engines_.size() << " already created engines" << std::endl;
        
        // Engines will be automatically cleaned up through unique_ptr destructor
        engines_.clear();
        throw; // Re-throw the exception
    }
    
    
}

SelfPlayManager::~SelfPlayManager() {
    
    
    // Stop all active games and wait for them to complete
    active_games_.store(0);
    
    // Force GPU memory cleanup before engines are destroyed
    if (neural_net_) {
        
        neural_net_.reset();
    }
    
    // Clean up engines in reverse order of creation
    
    while (!engines_.empty()) {
        try {
            engines_.pop_back();  // This will destroy the unique_ptr and call engine destructor
        } catch (const std::exception& e) {
            std::cerr << "Warning: Exception during engine cleanup: " << e.what() << std::endl;
        }
    }
    
    
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
        throw std::runtime_error("No MCTS engines available");
    }
    
    std::cout << "Generating " << num_games << " games using full MCTS search..." << std::endl;
    
    // Initialize result vector
    std::vector<alphazero::selfplay::GameData> games;
    games.reserve(num_games);
    
    // Create a timestamp base for game IDs
    std::string timestamp_base = getCurrentTimestamp();
    
    // Determine the number of parallel games to run
    int num_parallel_games = settings_.num_parallel_games;
    if (num_parallel_games <= 0 || num_parallel_games > static_cast<int>(engines_.size())) {
        num_parallel_games = static_cast<int>(engines_.size());
    }
    
    std::cout << "Using " << num_parallel_games << " parallel game threads..." << std::endl;
    
    // Generate games in batches using parallel threads
    int games_generated = 0;
    while (games_generated < num_games) {
        int batch_size = std::min(num_parallel_games, num_games - games_generated);
        
        // Periodic memory cleanup every 100 games
        if (games_generated > 0 && games_generated % 100 == 0) {
            utils::GameStatePoolManager::getInstance().clearAllPools();
            
        }
        
        // Create futures for parallel game generation
        std::vector<std::future<alphazero::selfplay::GameData>> futures;
        futures.reserve(batch_size);
        
        for (int j = 0; j < batch_size; j++) {
            int game_index = games_generated + j;
            std::string game_id = timestamp_base + "_" + std::to_string(game_index);
            int engine_id = j % engines_.size();  // Deterministic assignment
            
            // Run game generation in a separate thread with specific engine  
            futures.push_back(std::async(std::launch::async, [this, game_type, board_size, game_id, engine_id]() {
                return generateGame(game_type, board_size, game_id, engine_id);
            }));
        }
        
        // Collect results from all threads
        for (int j = 0; j < batch_size; j++) {
            try {
                int game_index = games_generated + j;
                auto game_data = futures[j].get();
                
                // Add game to results if it has at least one move
                if (!game_data.moves.empty()) {
                    games.push_back(std::move(game_data));
                    std::cout << "Generated game " << games.size() << "/" << num_games
                              << " with " << game_data.moves.size() << " moves" << std::endl;
                } else {
                    std::cerr << "Failed to generate game " << game_index << ": no moves were made" << std::endl;
                    // Don't increment games_generated for this one, it will be retried
                    continue;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error generating game " << (games_generated + j) << ": " << e.what() << std::endl;
                throw;
            }
        }
        
        // Increment the counter for successful games
        games_generated += batch_size;
    }
    
    std::cout << "Successfully generated " << games.size() << " games" << std::endl;
    return games;
}

alphazero::selfplay::GameData alphazero::selfplay::SelfPlayManager::generateGame(core::GameType game_type, int board_size,
                                      const std::string& game_id, int engine_id) {
    
    
    // Use the provided engine_id
    int thread_id = engine_id;
    if (thread_id < 0 || thread_id >= engines_.size()) {
        std::cerr << "SelfPlayManager::generateGame - Invalid engine_id: " << engine_id << std::endl;
        thread_id = 0; // Fallback to first engine
    }
    
    
    
    // Check engine validity before use
    if (thread_id >= engines_.size()) {
        std::cerr << "SelfPlayManager::generateGame - ERROR: Invalid engine index " << thread_id 
                 << " (engines_.size() = " << engines_.size() << ")" << std::endl;
        throw std::runtime_error("Invalid MCTS engine index");
    }
    
    if (!engines_[thread_id]) {
        std::cerr << "SelfPlayManager::generateGame - ERROR: Engine at index " << thread_id << " is null" << std::endl;
        throw std::runtime_error("Null MCTS engine");
    }
    
    
    
    // Get a reference to the engine
    mcts::MCTSEngine& engine = *engines_[thread_id];
    
    // Ensure the evaluator is started before use
    
    if (!engine.ensureEvaluatorStarted()) {
        std::cerr << "SelfPlayManager::generateGame - ERROR: Failed to start evaluator for engine " << thread_id << std::endl;
        throw std::runtime_error("Failed to start evaluator for engine " + std::to_string(thread_id));
    }
    

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
    std::cout << "Game " << game_id << ": Started (thread " << thread_id << ")" << std::endl;
    
    // Calculate max moves if not specified
    int max_moves = settings_.max_moves;
    if (max_moves <= 0) {
        // Default to board size squared or 1000, whichever is smaller
        max_moves = std::min(board_size * board_size, 1000);
    }
    
    // Start time
    auto start_time = std::chrono::steady_clock::now();
    
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
            
            // First, ensure the evaluator is started
            // Debug: Re-ensuring evaluator is started before search
            // std::cout << "SelfPlayManager::generateGame - Re-ensuring evaluator is started before search for move " 
            //          << move_count << " in game " << game_id << std::endl;
            if (!engine.ensureEvaluatorStarted()) {
                std::cerr << "SelfPlayManager::generateGame - ERROR: Failed to start evaluator for engine before search in game " 
                         << game_id << std::endl;
                throw std::runtime_error("Failed to start evaluator before search");
            }
            
            
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
                    alphazero::utils::trackMemory("Before search - Game " + game_id + ", Move " + std::to_string(move_count));
                }
                
                result = engine.search(*game);
                
                // Log every 10th move to reduce verbosity
                if (move_count % 10 == 0) {
                    std::cout << "Game " << game_id << ": Move " << move_count 
                             << " completed (action: " << result.action << ")" << std::endl;
                    alphazero::utils::trackMemory("After search - Game " + game_id + ", Move " + std::to_string(move_count));
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
    // First try to use the GameRegistry to create the game
    try {
        auto& registry = core::GameRegistry::instance();
        if (registry.isRegistered(game_type)) {
            auto game = registry.createGame(game_type);
            return game;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error using GameRegistry: " << e.what() << ", falling back to direct instantiation" << std::endl;
    }
    
    // Direct instantiation fallback
    switch (game_type) {
        case core::GameType::CHESS: {
            // Position_id is used for Chess960 starting positions (0-959)
            bool use_chess960 = position_id >= 0 && position_id < 960;
            return std::make_unique<games::chess::ChessState>(use_chess960, "", position_id);
        }
        case core::GameType::GO: {
            // Standard board sizes for Go are 9x9, 13x13, or 19x19
            int go_board_size = 19; // Default
            if (board_size == 9 || board_size == 13 || board_size == 19) {
                go_board_size = board_size;
            }
            
            // Default komi is 7.5 for 19x19, 6.5 for 13x13, 5.5 for 9x9
            float komi = 7.5f;
            if (go_board_size == 13) komi = 6.5f;
            if (go_board_size == 9) komi = 5.5f;
            
            // Chinese rules is more common in modern Go
            bool chinese_rules = true;
            
            // Enforce super-ko rule to prevent board state repetition
            bool enforce_superko = true;
            
            return std::make_unique<games::go::GoState>(go_board_size, komi, chinese_rules, enforce_superko);
        }
        case core::GameType::GOMOKU: {
            // Default Gomoku board size is 15x15
            int gomoku_board_size = (board_size > 0) ? board_size : 15;
            
            // Use Renju rules which apply constraints for black (first player)
            // to balance the inherent first-player advantage
            bool use_renju = true;
            
            // Don't use Omok rules (generally exclusive with Renju)
            bool use_omok = false;
            
            // Use pro-long opening to prevent immediate strong opening patterns
            bool use_pro_long_opening = true;
            
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
    
    // Update MCTS settings in engines
    for (auto& engine : engines_) {
        engine->updateSettings(settings_.mcts_settings);
    }
}

} // namespace selfplay
} // namespace alphazero