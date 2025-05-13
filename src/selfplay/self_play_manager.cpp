// src/selfplay/self_play_manager.cpp
#include "selfplay/self_play_manager.h"
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include "nlohmann/json.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"
#include "core/game_export.h"
#include "utils/debug_monitor.h"
#include "utils/memory_debug.h"

namespace alphazero {
namespace selfplay {

using json = nlohmann::json;

SelfPlayManager::SelfPlayManager(std::shared_ptr<nn::NeuralNetwork> neural_net,
                               const SelfPlaySettings& settings)
    : neural_net_(neural_net),
      settings_(settings),
      active_games_(0) {

    // Start memory monitoring for debugging
    debug::startMemoryMonitoring();
    debug::takeMemorySnapshot("SelfPlayManager_Constructor_Start");

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
        debug::takeMemorySnapshot("Before_Creating_MCTSEngine_" + std::to_string(i));
        try {
            engines_.emplace_back(std::make_unique<mcts::MCTSEngine>(neural_net_, settings_.mcts_settings));
            debug::takeMemorySnapshot("After_Creating_MCTSEngine_" + std::to_string(i));
        } catch (const std::exception& e) {
            std::cerr << "Error creating MCTS engine " << i << ": " << e.what() << std::endl;
            debug::takeMemorySnapshot("Error_Creating_MCTSEngine_" + std::to_string(i));
            throw;
        }
    }

    debug::takeMemorySnapshot("SelfPlayManager_Constructor_End");
}

std::vector<GameData> SelfPlayManager::generateGames(core::GameType game_type,
                                                    int num_games,
                                                    int board_size) {
    debug::takeMemorySnapshot("GenerateGames_Start");

    std::vector<GameData> games;
    games.reserve(num_games);

    // Generate games in batches
    int remaining_games = num_games;
    int batch_number = 0;

    while (remaining_games > 0) {
        debug::takeMemorySnapshot("GenerateGames_Batch_" + std::to_string(batch_number) + "_Start");

        int batch_size = std::min(remaining_games, settings_.num_parallel_games);
        std::vector<std::thread> threads;
        std::vector<GameData> batch_results(batch_size);

        // Create timestamp for game IDs
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        // Launch worker threads
        for (int i = 0; i < batch_size; ++i) {
            std::string game_id = std::to_string(timestamp) + "_" + std::to_string(i);
            threads.emplace_back(&SelfPlayManager::gameWorker, this,
                                game_type, board_size, game_id, &batch_results[i]);
            active_games_++;
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Print memory status after each batch
        DEBUG_PRINT_MEMORY_USAGE();

        // Add results to games vector
        games.insert(games.end(), batch_results.begin(), batch_results.end());

        // Update remaining games
        remaining_games -= batch_size;
        batch_number++;

        debug::takeMemorySnapshot("GenerateGames_Batch_" + std::to_string(batch_number) + "_End");
    }

    debug::takeMemorySnapshot("GenerateGames_End");
    return games;
}

void SelfPlayManager::gameWorker(core::GameType game_type, int board_size,
                               const std::string& game_id, GameData* result) {
    std::string worker_id = "Worker_" + game_id;
    debug::takeMemorySnapshot(worker_id + "_Start");

    try {
        *result = generateGame(game_type, board_size, game_id);
        debug::takeMemorySnapshot(worker_id + "_Success");
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error in game worker " << game_id
                 << ": " << e.what() << std::endl;
        debug::takeMemorySnapshot(worker_id + "_BadAlloc");
        // Create an empty result
        result->game_id = game_id + "_ERROR";
        result->game_type = game_type;
        result->board_size = board_size;
    } catch (const std::exception& e) {
        std::cerr << "Error in game worker " << game_id << ": " << e.what() << std::endl;
        debug::takeMemorySnapshot(worker_id + "_Exception");
        // Create an empty result
        result->game_id = game_id + "_ERROR";
        result->game_type = game_type;
        result->board_size = board_size;
    }

    active_games_--;
    debug::takeMemorySnapshot(worker_id + "_End");
}

GameData SelfPlayManager::generateGame(core::GameType game_type, int board_size,
                                      const std::string& game_id) {
    std::string game_prefix = "Game_" + game_id;
    debug::takeMemorySnapshot(game_prefix + "_Start");

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

    debug::takeMemorySnapshot(game_prefix + "_BeforeCreateGame");

    // Create game
    auto game = createGame(game_type, board_size, position_id);
    if (!game) {
        throw std::runtime_error("Failed to create game state");
    }

    debug::takeMemorySnapshot(game_prefix + "_AfterCreateGame");

    // Prepare game data
    GameData data;
    data.game_type = game_type;
    data.board_size = board_size;
    data.winner = 0;  // Default to draw
    data.game_id = game_id;

    // Record detailed debug info
    std::cout << "Game " << game_id << ": Created game of type "
             << static_cast<int>(game_type) << " with board size "
             << board_size << " (thread " << thread_id << ")" << std::endl;
    
    // Calculate max moves if not specified
    int max_moves = settings_.max_moves;
    if (max_moves <= 0) {
        // Default to 5 times the action space size or 1000, whichever is smaller
        max_moves = std::min(5 * game->getActionSpaceSize(), 1000);
    }
    
    // Start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Play until terminal state or max moves
    int move_count = 0;
    while (!game->isTerminal() && static_cast<int>(data.moves.size()) < max_moves) {
        std::string move_prefix = game_prefix + "_Move" + std::to_string(move_count);
        debug::takeMemorySnapshot(move_prefix + "_Start");

        // Set temperature based on move number
        auto current_settings = engine.getSettings();
        if (static_cast<int>(data.moves.size()) < settings_.temperature_threshold) {
            current_settings.temperature = settings_.high_temperature;  // Exploration
        } else {
            current_settings.temperature = settings_.low_temperature;  // Exploitation
        }
        current_settings.add_dirichlet_noise = settings_.add_dirichlet_noise &&
                                              data.moves.empty();  // Only on first move
        engine.updateSettings(current_settings);

        debug::takeMemorySnapshot(move_prefix + "_BeforeSearch");

        try {
            // Run search
            auto result = engine.search(*game);

            debug::takeMemorySnapshot(move_prefix + "_AfterSearch");

            // Store move and policy
            data.moves.push_back(result.action);
            data.policies.push_back(result.probabilities);

            // Make move
            game->makeMove(result.action);

            // Check if terminal
            if (game->isTerminal()) {
                auto game_result = game->getGameResult();
                if (game_result == core::GameResult::WIN_PLAYER1) {
                    data.winner = 1;
                } else if (game_result == core::GameResult::WIN_PLAYER2) {
                    data.winner = 2;
                }
                debug::takeMemorySnapshot(move_prefix + "_GameTerminal");
                break;
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation error during search in game " << game_id
                     << " at move " << move_count << ": " << e.what() << std::endl;
            debug::takeMemorySnapshot(move_prefix + "_BadAlloc");
            // Exit the loop and return partial game data
            break;
        } catch (const std::exception& e) {
            std::cerr << "Error during search in game " << game_id
                     << " at move " << move_count << ": " << e.what() << std::endl;
            debug::takeMemorySnapshot(move_prefix + "_Exception");
            // Exit the loop and return partial game data
            break;
        }

        move_count++;

        if (move_count % 10 == 0) {
            // Print progress every 10 moves
            std::cout << "Game " << game_id << ": Completed " << move_count
                     << " moves, memory usage: "
                     << (debug::MemoryTracker::instance().getCurrentUsage() / (1024 * 1024))
                     << " MB" << std::endl;
        }

        debug::takeMemorySnapshot(move_prefix + "_End");
    }
    
    // End time
    auto end_time = std::chrono::steady_clock::now();
    data.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    return data;
}

std::unique_ptr<core::IGameState> SelfPlayManager::createGame(core::GameType game_type, 
                                                            int board_size, 
                                                            int position_id) {
    // Create game based on type
    std::unique_ptr<core::IGameState> game;
    
    switch (game_type) {
        case core::GameType::CHESS: {
            game = std::make_unique<::alphazero::games::chess::ChessState>();
            // TODO: Implement Fischer Random starting positions if needed
            break;
        }
        case core::GameType::GO: {
            game = std::make_unique<::alphazero::games::go::GoState>(board_size > 0 ? board_size : 19);
            break;
        }
        case core::GameType::GOMOKU: {
            game = std::make_unique<::alphazero::games::gomoku::GomokuState>(board_size > 0 ? board_size : 15);
            break;
        }
        default:
            throw std::runtime_error("Unsupported game type");
    }
    
    return game;
}

void SelfPlayManager::saveGames(const std::vector<GameData>& games, 
                              const std::string& output_dir, 
                              const std::string& format) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Save each game
    for (const auto& game : games) {
        // Create filename
        std::string filename = output_dir + "/game_" + game.game_id;
        
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
            file << j.dump(2);
        } else if (format == "binary") {
            filename += ".bin";
            
            // Write to binary file
            std::ofstream file(filename, std::ios::binary);
            
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
        } else {
            throw std::runtime_error("Unsupported format: " + format);
        }
    }
}

std::vector<GameData> SelfPlayManager::loadGames(const std::string& input_dir, 
                                               const std::string& format) {
    std::vector<GameData> games;
    
    // Check if directory exists
    if (!std::filesystem::exists(input_dir)) {
        throw std::runtime_error("Input directory does not exist: " + input_dir);
    }
    
    // Iterate through files in directory
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        std::string extension = entry.path().extension().string();
        
        if ((format == "json" && extension == ".json") || 
            (format == "binary" && extension == ".bin")) {
            
            std::string filename = entry.path().string();
            
            if (format == "json") {
                // Read JSON file
                std::ifstream file(filename);
                json j;
                file >> j;
                
                // Parse game data
                GameData game;
                game.game_type = static_cast<core::GameType>(j["game_type"].get<int>());
                game.board_size = j["board_size"].get<int>();
                game.winner = j["winner"].get<int>();
                game.moves = j["moves"].get<std::vector<int>>();
                game.policies = j["policies"].get<std::vector<std::vector<float>>>();
                game.total_time_ms = j["total_time_ms"].get<int64_t>();
                game.game_id = j["game_id"].get<std::string>();
                
                games.push_back(std::move(game));
            } else if (format == "binary") {
                // Read binary file
                std::ifstream file(filename, std::ios::binary);
                
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
                    std::cerr << "Invalid file format: " << filename << std::endl;
                    continue;
                }
                
                // Read game ID
                int32_t game_id_length;
                file.read(reinterpret_cast<char*>(&game_id_length), sizeof(game_id_length));
                std::string game_id(game_id_length, '\0');
                file.read(&game_id[0], game_id_length);
                
                // Create game data
                GameData game;
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
                
                games.push_back(std::move(game));
            }
        }
    }
    
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
        // Create game instance
        auto game_state = ::alphazero::core::GameFactory::createGame(game.game_type);
        
        // Process each move
        for (size_t i = 0; i < game.moves.size(); ++i) {
            // Get state tensor
            states.push_back(game_state->getEnhancedTensorRepresentation());
            
            // Get policy
            policies.push_back(game.policies[i]);
            
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