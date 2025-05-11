// examples/self_play_libtorch.cpp
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"

using json = nlohmann::json;
using namespace alphazero;

// Game factory
std::unique_ptr<core::IGameState> createGame(const std::string& game_type, int board_size = 0) {
    if (game_type == "gomoku") {
        return std::make_unique<gomoku::GomokuState>(board_size > 0 ? board_size : 15);
    } else if (game_type == "chess") {
        return std::make_unique<chess::ChessState>();
    } else if (game_type == "go") {
        return std::make_unique<go::GoState>(board_size > 0 ? board_size : 19);
    } else {
        throw std::runtime_error("Unknown game type: " + game_type);
    }
}

// Save game to JSON
void saveGameToJson(const std::string& filename, 
                   const std::vector<int>& moves,
                   const std::vector<std::vector<float>>& policies,
                   int winner,
                   const std::string& game_type,
                   int board_size) {
    json game_data;
    game_data["game_type"] = game_type;
    game_data["board_size"] = board_size;
    game_data["moves"] = moves;
    game_data["policies"] = policies;
    game_data["winner"] = winner;
    
    std::ofstream file(filename);
    file << game_data.dump(2);
}

int main(int argc, char** argv) {
    try {
        // Parse arguments
        std::string game_type = "gomoku";
        int board_size = 15;
        int num_simulations = 800;
        int num_threads = 4;
        int batch_size = 8;
        std::string model_path = "";
        std::string output_dir = ".";
        int num_games = 1;
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--game" && i + 1 < argc) {
                game_type = argv[++i];
            } else if (arg == "--board-size" && i + 1 < argc) {
                board_size = std::stoi(argv[++i]);
            } else if (arg == "--simulations" && i + 1 < argc) {
                num_simulations = std::stoi(argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else if (arg == "--batch-size" && i + 1 < argc) {
                batch_size = std::stoi(argv[++i]);
            } else if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--games" && i + 1 < argc) {
                num_games = std::stoi(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --game TYPE          Game type (gomoku, chess, go) [default: gomoku]\n"
                          << "  --board-size SIZE    Board size [default: 15]\n"
                          << "  --simulations NUM    Number of MCTS simulations [default: 800]\n"
                          << "  --threads NUM        Number of threads [default: 4]\n"
                          << "  --batch-size SIZE    Neural network batch size [default: 8]\n"
                          << "  --model PATH         Path to model file\n"
                          << "  --output DIR         Output directory [default: .]\n"
                          << "  --games NUM          Number of games [default: 1]\n"
                          << "  --help               Show this help message\n";
                return 0;
            }
        }
        
        // Create a sample game to determine input channels
        auto sample_game = createGame(game_type, board_size);
        auto tensor = sample_game->getEnhancedTensorRepresentation();
        int input_channels = static_cast<int>(tensor.size());
        int policy_size = sample_game->getActionSpaceSize();
        
        // Create or load neural network model
        std::shared_ptr<nn::ResNetModel> model;
        if (!model_path.empty()) {
            std::cout << "Loading model from " << model_path << "...\n";
            model = nn::NeuralNetworkFactory::loadResNet(model_path, input_channels, board_size, policy_size);
        } else {
            std::cout << "Creating new model...\n";
            model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 10, 128, policy_size);
        }
        
        // Create MCTS settings
        mcts::MCTSSettings settings;
        settings.num_simulations = num_simulations;
        settings.num_threads = num_threads;
        settings.batch_size = batch_size;
        settings.add_dirichlet_noise = true;
        
        // Create MCTS engine
        mcts::MCTSEngine engine(model, settings);
        
        // Play games
        for (int game_num = 0; game_num < num_games; ++game_num) {
            std::cout << "Playing game " << (game_num + 1) << "/" << num_games << "...\n";
            
            // Create game
            auto game = createGame(game_type, board_size);
            std::vector<int> moves;
            std::vector<std::vector<float>> policies;
            int winner = 0;
            
            // Temperature threshold (30 moves or half the board size^2)
            int temp_threshold = std::min(30, (board_size * board_size) / 2);
            
            // Play until terminal state
            auto start_time = std::chrono::steady_clock::now();
            while (!game->isTerminal() && static_cast<int>(moves.size()) < board_size * board_size) {
                // Set temperature
                auto current_settings = engine.getSettings();
                if (static_cast<int>(moves.size()) < temp_threshold) {
                    current_settings.temperature = 1.0f;  // Exploration
                } else {
                    current_settings.temperature = 0.1f;  // Exploitation
                }
                engine.updateSettings(current_settings);
                
                // Show current board
                std::cout << game->toString() << std::endl;
                std::cout << "Player " << game->getCurrentPlayer() << "'s turn...\n";
                
                // Run search
                auto search_start = std::chrono::steady_clock::now();
                auto result = engine.search(*game);
                auto search_end = std::chrono::steady_clock::now();
                auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start);
                
                // Make move
                game->makeMove(result.action);
                moves.push_back(result.action);
                policies.push_back(result.probabilities);
                
                // Print search stats
                std::cout << "Move: " << game->actionToString(result.action) << "\n";
                std::cout << "Value: " << result.value << "\n";
                std::cout << "Nodes: " << result.stats.total_nodes << "\n";
                std::cout << "Time: " << search_time.count() << "ms\n";
                std::cout << "Nodes/s: " << result.stats.nodes_per_second << "\n";
                std::cout << "Avg batch: " << result.stats.avg_batch_size << "\n";
                std::cout << "----------------------------\n";
                
                // Check if terminal
                if (game->isTerminal()) {
                    auto game_result = game->getGameResult();
                    if (game_result == core::GameResult::WIN_PLAYER1) {
                        winner = 1;
                        std::cout << "Player 1 wins!\n";
                    } else if (game_result == core::GameResult::WIN_PLAYER2) {
                        winner = 2;
                        std::cout << "Player 2 wins!\n";
                    } else {
                        std::cout << "Draw!\n";
                    }
                    break;
                }
            }
            
            auto end_time = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            
            // Print game stats
            std::cout << "Game completed in " << total_time.count() << "s\n";
            std::cout << "Moves: " << moves.size() << "\n";
            std::cout << "Final board:\n";
            std::cout << game->toString() << std::endl;
            
            // Save game
            std::string timestamp = std::to_string(std::time(nullptr));
            std::string filename = output_dir + "/game_" + timestamp + ".json";
            saveGameToJson(filename, moves, policies, winner, game_type, board_size);
            std::cout << "Game saved to " << filename << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}