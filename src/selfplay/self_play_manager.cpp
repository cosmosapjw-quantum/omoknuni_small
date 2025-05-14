// src/selfplay/self_play_manager.cpp
#include "selfplay/self_play_manager.h"
#include "mcts/mcts_engine.h"
#include "utils/debug_monitor.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"
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
    
    // Initialize random number generator
    if (settings_.random_seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(settings_.random_seed));
    }

    // Create MCTS engines for each worker thread
    engines_.reserve(settings_.num_parallel_games);
    for (int i = 0; i < settings_.num_parallel_games; ++i) {
        try {
            engines_.emplace_back(std::make_unique<mcts::MCTSEngine>(neural_net_, settings_.mcts_settings));
        } catch (const std::exception& e) {
            std::cerr << "Error creating MCTS engine " << i << ": " << e.what() << std::endl;
            throw;
        }
    }
}

std::vector<GameData> SelfPlayManager::generateGames(core::GameType game_type,
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
    
    // Initialize result vector
    std::vector<GameData> games;
    games.reserve(num_games);
    
    // Thread-safe access to the games vector
    std::mutex games_mutex;
    
    // Use a vector to store exceptions from worker threads
    std::vector<std::exception_ptr> thread_exceptions;
    std::mutex exceptions_mutex;
    
    // Semaphore for controlling the number of active games
    std::mutex semaphore_mutex;
    std::condition_variable semaphore_cv;
    int active_games = 0;
    
    // Generate games in batches with improved batch scheduling
    int remaining_games = num_games;
    int batch_number = 0;
    
    // Create a timestamp base for game IDs
    std::string timestamp_base = getCurrentTimestamp();
    
    // Initialize progress tracking
    std::mutex progress_mutex;
    std::atomic<int> completed_games{0};
    std::atomic<int> completed_moves{0};
    auto progress_start_time = std::chrono::steady_clock::now();
    
    std::cout << "Generating " << num_games << " self-play games..." << std::endl;
    
    while (remaining_games > 0 && thread_exceptions.empty()) {
        // Determine batch size for this iteration
        int batch_size = std::min(remaining_games, settings_.num_parallel_games);
        
        // Prepare worker futures
        std::vector<std::future<void>> futures;
        futures.reserve(batch_size);
        
        // Launch worker threads for this batch
        for (int i = 0; i < batch_size; ++i) {
            // Create a unique game ID
            std::string game_id = timestamp_base + "_" + std::to_string(batch_number * settings_.num_parallel_games + i);
            
            // Wait until we have an available slot
            {
                std::unique_lock<std::mutex> lock(semaphore_mutex);
                semaphore_cv.wait(lock, [&active_games, &settings = settings_]() {
                    return active_games < settings.num_parallel_games;
                });
                active_games++;
            }
            
            // Launch game worker
            futures.emplace_back(std::async(std::launch::async, [&, game_id, i]() {
                std::string worker_id = "Worker_" + game_id;
                
                try {
                    // Create game data
                    GameData game_data;
                    game_data = generateGame(game_type, board_size, game_id);
                    
                    // Add to result vector
                    {
                        std::lock_guard<std::mutex> lock(games_mutex);
                        games.push_back(std::move(game_data));
                    }
                    
                    // Update progress counters
                    int game_moves = game_data.moves.size();
                    completed_games.fetch_add(1);
                    completed_moves.fetch_add(game_moves);
                    
                    // Periodically print progress
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - progress_start_time).count();
                        
                        if (elapsed > 0 && (completed_games % 5 == 0 || completed_games.load() == num_games)) {
                            float games_per_second = static_cast<float>(completed_games) / elapsed;
                            float moves_per_second = static_cast<float>(completed_moves) / elapsed;
                            
                            std::cout << "Progress: " << completed_games << "/" << num_games 
                                     << " games (" << (100.0f * completed_games / num_games) << "%), "
                                     << games_per_second << " games/s, "
                                     << moves_per_second << " moves/s" << std::endl;
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error in game worker " << worker_id << ": " << e.what() << std::endl;
                    
                    // Store exception for later re-throwing in main thread
                    std::lock_guard<std::mutex> lock(exceptions_mutex);
                    thread_exceptions.push_back(std::current_exception());
                }
                catch (...) {
                    std::cerr << "Unknown error in game worker " << worker_id << std::endl;
                    
                    // Store exception for later re-throwing in main thread
                    std::lock_guard<std::mutex> lock(exceptions_mutex);
                    thread_exceptions.push_back(std::current_exception());
                }
                
                // Release the semaphore slot
                {
                    std::lock_guard<std::mutex> lock(semaphore_mutex);
                    active_games--;
                }
                semaphore_cv.notify_one();
            }));
        }
        
        // Wait for all futures in this batch
        for (auto& future : futures) {
            future.wait();
        }
        
        // Update remaining games
        remaining_games -= batch_size;
        batch_number++;
        
        // If any thread had an exception, stop processing
        if (!thread_exceptions.empty()) {
            break;
        }
    }
    
    // If any thread thrown an exception, rethrow the first one
    if (!thread_exceptions.empty()) {
        std::rethrow_exception(thread_exceptions[0]);
    }
    
    std::cout << "Successfully generated " << games.size() << " games with "
             << completed_moves.load() << " total moves" << std::endl;
    
    return games;
}

GameData SelfPlayManager::generateGame(core::GameType game_type, int board_size,
                                      const std::string& game_id) {
    // Select a thread-specific MCTS engine
    int thread_id = 0;
    {
        std::hash<std::thread::id> hasher;
        thread_id = hasher(std::this_thread::get_id()) % engines_.size();
    }
    
    mcts::MCTSEngine& engine = *engines_[thread_id];

    // Select a random starting position
    std::uniform_int_distribution<int> pos_dist(0, settings_.num_start_positions - 1);
    int position_id = pos_dist(rng_);

    // Create game
    auto game = createGame(game_type, board_size, position_id);
    if (!game) {
        throw std::runtime_error("Failed to create game state");
    }

    // Prepare game data
    GameData data;
    data.game_type = game_type;
    data.board_size = board_size;
    data.winner = 0;  // Default to draw
    data.game_id = game_id;

    // Log game information
    std::cout << "Game " << game_id << ": Created game of type "
             << static_cast<int>(game_type) << " with board size "
             << board_size << " (thread " << thread_id << ")" << std::endl;
    
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

            // Run search with timeout detection to prevent stalling
            mcts::SearchResult result;
            
            auto search_future = std::async(std::launch::async, [&]() {
                return engine.search(*game);
            });
            
            // Wait for search with a generous timeout
            auto search_status = search_future.wait_for(std::chrono::seconds(30));
            if (search_status != std::future_status::ready) {
                throw std::runtime_error("MCTS search timed out after 30 seconds");
            }
            
            result = search_future.get();

            // Store move and policy
            data.moves.push_back(result.action);
            data.policies.push_back(result.probabilities);

            // Make move
            game->makeMove(result.action);
            move_count++;

            // Print progress for moves multiple of 10
            if (move_count % 10 == 0) {
                std::cout << "Game " << game_id << ": Completed " << move_count 
                         << " moves" << std::endl;
            }
            
            // Check if terminal
            if (game->isTerminal()) {
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
    std::cout << "Game " << game_id << ": Completed with "
             << data.moves.size() << " moves in "
             << (data.total_time_ms / 1000.0) << " seconds" << std::endl;
    
    return data;
}

std::unique_ptr<core::IGameState> SelfPlayManager::createGame(core::GameType game_type, 
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
            return std::make_unique<games::chess::ChessState>();
        }
        case core::GameType::GO: {
            return std::make_unique<games::go::GoState>(board_size > 0 ? board_size : 19);
        }
        case core::GameType::GOMOKU: {
            return std::make_unique<games::gomoku::GomokuState>(board_size > 0 ? board_size : 15);
        }
        default:
            throw std::runtime_error("Unsupported game type: " + 
                                     core::gameTypeToString(game_type));
    }
}

void SelfPlayManager::saveGames(const std::vector<GameData>& games, 
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

std::vector<GameData> SelfPlayManager::loadGames(const std::string& input_dir, 
                                               const std::string& format) {
    std::vector<GameData> games;
    
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
                GameData game;
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
SelfPlayManager::convertToTrainingExamples(const std::vector<GameData>& games) {
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

const SelfPlaySettings& SelfPlayManager::getSettings() const {
    return settings_;
}

void SelfPlayManager::updateSettings(const SelfPlaySettings& settings) {
    settings_ = settings;
    
    // Update MCTS settings in engines
    for (auto& engine : engines_) {
        engine->updateSettings(settings_.mcts_settings);
    }
}

} // namespace selfplay
} // namespace alphazero