// src/cli/omoknuni_cli.cpp
#include "cli/cli_manager.h"
#include "core/game_export.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "training/training_data_manager.h"
#include "evaluation/model_evaluator.h"
#include "chess/chess_state.h"
#include "go/go_state.h"
#include "gomoku/gomoku_state.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <thread>

using namespace alphazero;

// Load configuration from YAML file
YAML::Node loadConfig(const std::string& config_path) {
    try {
        return YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        return YAML::Node();
    }
}

// Get command line argument value
std::string getArg(const std::vector<std::string>& args, 
                  const std::string& option, 
                  const std::string& default_value = "") {
    for (size_t i = 0; i < args.size() - 1; ++i) {
        if (args[i] == option) {
            return args[i + 1];
        }
    }
    return default_value;
}

// Check if command line argument exists
bool hasArg(const std::vector<std::string>& args, const std::string& option) {
    return std::find(args.begin(), args.end(), option) != args.end();
}

// Parse board size from arguments or config
int parseBoardSize(const std::vector<std::string>& args, const YAML::Node& config, 
                  core::GameType game_type) {
    // Default board sizes
    int default_size = 15;  // Default for Gomoku
    if (game_type == core::GameType::GO) {
        default_size = 19;  // Default for Go
    } else if (game_type == core::GameType::CHESS) {
        default_size = 8;  // Chess is always 8x8
    }
    
    // Check command line argument
    std::string size_str = getArg(args, "--board-size");
    if (!size_str.empty()) {
        try {
            return std::stoi(size_str);
        } catch (...) {
            std::cerr << "Invalid board size: " << size_str << std::endl;
        }
    }
    
    // Check config
    if (config["board_size"]) {
        return config["board_size"].as<int>();
    }
    
    return default_size;
}

// Parse game type from arguments or config
core::GameType parseGameType(const std::vector<std::string>& args, const YAML::Node& config) {
    // Check command line argument
    std::string type_str = getArg(args, "--game");
    if (type_str.empty() && config["game"]) {
        type_str = config["game"].as<std::string>();
    }
    
    if (type_str == "chess") {
        return core::GameType::CHESS;
    } else if (type_str == "go") {
        return core::GameType::GO;
    } else if (type_str == "gomoku") {
        return core::GameType::GOMOKU;
    } else {
        std::cerr << "Unknown game type: " << type_str << ", defaulting to gomoku" << std::endl;
        return core::GameType::GOMOKU;
    }
}

// Parse MCTS settings from config
mcts::MCTSSettings parseMctsSettings(const YAML::Node& config) {
    mcts::MCTSSettings settings;
    
    if (config["mcts"]) {
        auto mcts_config = config["mcts"];
        
        if (mcts_config["num_simulations"]) {
            settings.num_simulations = mcts_config["num_simulations"].as<int>();
        }
        
        if (mcts_config["num_threads"]) {
            settings.num_threads = mcts_config["num_threads"].as<int>();
        }
        
        if (mcts_config["batch_size"]) {
            settings.batch_size = mcts_config["batch_size"].as<int>();
        }
        
        if (mcts_config["exploration_constant"]) {
            settings.exploration_constant = mcts_config["exploration_constant"].as<float>();
        }
        
        if (mcts_config["virtual_loss"]) {
            settings.virtual_loss = mcts_config["virtual_loss"].as<int>();
        }
        
        if (mcts_config["add_dirichlet_noise"]) {
            settings.add_dirichlet_noise = mcts_config["add_dirichlet_noise"].as<bool>();
        }
        
        if (mcts_config["dirichlet_alpha"]) {
            settings.dirichlet_alpha = mcts_config["dirichlet_alpha"].as<float>();
        }
        
        if (mcts_config["dirichlet_epsilon"]) {
            settings.dirichlet_epsilon = mcts_config["dirichlet_epsilon"].as<float>();
        }
        
        if (mcts_config["temperature"]) {
            settings.temperature = mcts_config["temperature"].as<float>();
        }
        
        if (mcts_config["batch_timeout_ms"]) {
            settings.batch_timeout = std::chrono::milliseconds(
                mcts_config["batch_timeout_ms"].as<int>());
        }
    }
    
    return settings;
}

// Parse self-play settings from config
selfplay::SelfPlaySettings parseSelfPlaySettings(const YAML::Node& config) {
    selfplay::SelfPlaySettings settings;
    
    settings.mcts_settings = parseMctsSettings(config);
    
    if (config["self_play"]) {
        auto selfplay_config = config["self_play"];
        
        if (selfplay_config["num_parallel_games"]) {
            settings.num_parallel_games = selfplay_config["num_parallel_games"].as<int>();
        }
        
        if (selfplay_config["max_moves"]) {
            settings.max_moves = selfplay_config["max_moves"].as<int>();
        }
        
        if (selfplay_config["num_start_positions"]) {
            settings.num_start_positions = selfplay_config["num_start_positions"].as<int>();
        }
        
        if (selfplay_config["temperature_threshold"]) {
            settings.temperature_threshold = selfplay_config["temperature_threshold"].as<int>();
        }
        
        if (selfplay_config["high_temperature"]) {
            settings.high_temperature = selfplay_config["high_temperature"].as<float>();
        }
        
        if (selfplay_config["low_temperature"]) {
            settings.low_temperature = selfplay_config["low_temperature"].as<float>();
        }
        
        if (selfplay_config["add_dirichlet_noise"]) {
            settings.add_dirichlet_noise = selfplay_config["add_dirichlet_noise"].as<bool>();
        }
        
        if (selfplay_config["random_seed"]) {
            settings.random_seed = selfplay_config["random_seed"].as<int64_t>();
        }
    }
    
    return settings;
}

// Create a neural network from path or config
#ifdef WITH_TORCH
std::shared_ptr<nn::ResNetModel> createNeuralNetwork(
    const std::vector<std::string>& args, 
    const YAML::Node& config,
    core::GameType game_type,
    int board_size,
    const std::string& arg_name = "--model") {
    
    // Get model path from arguments or config
    std::string model_path = getArg(args, arg_name);
    if (model_path.empty() && config["model_path"]) {
        model_path = config["model_path"].as<std::string>();
    }
    
    // Create a sample game to determine input channels and policy size
    auto sample_game = core::GameFactory::createGame(game_type);
    
    // For games that support custom board sizes, we need to recreate a sample
    // with the correct size to get accurate tensor dimensions
    if (game_type == core::GameType::GOMOKU) {
        // First, we'll store the board representation and create a new instance
        sample_game = std::make_unique<games::gomoku::GomokuState>(board_size);
    } else if (game_type == core::GameType::GO) {
        // First, we'll store the board representation and create a new instance
        sample_game = std::make_unique<alphazero::go::GoState>(board_size);
    }
    
    auto tensor = sample_game->getEnhancedTensorRepresentation();
    int input_channels = static_cast<int>(tensor.size());
    int policy_size = sample_game->getActionSpaceSize();
    
    // Create or load model
    std::shared_ptr<nn::ResNetModel> model;
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        // Load existing model
        model = nn::NeuralNetworkFactory::loadResNet(
            model_path, input_channels, board_size, policy_size);
    } else {
        // Create new model
        int num_res_blocks = 10;
        int num_filters = 128;
        
        if (config["network"]) {
            auto net_config = config["network"];
            if (net_config["num_res_blocks"]) {
                num_res_blocks = net_config["num_res_blocks"].as<int>();
            }
            if (net_config["num_filters"]) {
                num_filters = net_config["num_filters"].as<int>();
            }
        }
        
        model = nn::NeuralNetworkFactory::createResNet(
            input_channels, board_size, num_res_blocks, num_filters, policy_size);
        
        if (!model_path.empty()) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            std::cerr << "Creating a new model" << std::endl;
        }
    }
    
    return model;
}
#else // WITH_TORCH
// Stub or alternative implementation if WITH_TORCH is not defined
std::shared_ptr<nn::NeuralNetwork> createNeuralNetwork(
    const std::vector<std::string>& args, 
    const YAML::Node& config,
    core::GameType game_type,
    int board_size,
    const std::string& arg_name = "--model") {
    std::cerr << "Warning: ResNetModel support is disabled (WITH_TORCH not defined)." 
              << " Returning nullptr for model." << std::endl;
    return nullptr; 
}
#endif // WITH_TORCH

// Handler function for self-play command
int handleSelfPlay(const std::vector<std::string>& args) {
    // Check if help was requested
    if (hasArg(args, "-h") || hasArg(args, "--help")) {
        std::cout << "Usage: self-play [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --config CONFIG   Configuration file path" << std::endl;
        std::cout << "  --game TYPE       Game type (gomoku, chess, go)" << std::endl;
        std::cout << "  --board-size N    Board size (for gomoku and go)" << std::endl;
        std::cout << "  --num-games N     Number of games to play" << std::endl;
        std::cout << "  --output DIR      Output directory" << std::endl;
        std::cout << "  --model PATH      Model file path" << std::endl;
        return 0;
    }
    
    // Load config file
    std::string config_path = getArg(args, "--config", "config.yaml");
    YAML::Node config = loadConfig(config_path);
    
    // Parse game type and board size
    core::GameType game_type = parseGameType(args, config);
    int board_size = parseBoardSize(args, config, game_type);
    
    // Create neural network
    auto model = createNeuralNetwork(args, config, game_type, board_size);
    if (!model) {
        std::cerr << "Error: Failed to create neural network. Exiting self-play." << std::endl;
        return 1;
    }
    
    // Parse self-play settings
    selfplay::SelfPlaySettings settings = parseSelfPlaySettings(config);
    
    // Create self-play manager
    selfplay::SelfPlayManager manager(model, settings);
    
    // Get number of games
    int num_games = 10;
    std::string num_games_str = getArg(args, "--num-games");
    if (!num_games_str.empty()) {
        try {
            num_games = std::stoi(num_games_str);
        } catch (...) {
            std::cerr << "Invalid number of games: " << num_games_str << std::endl;
        }
    } else if (config["self_play"] && config["self_play"]["num_games"]) {
        num_games = config["self_play"]["num_games"].as<int>();
    }
    
    // Get output directory
    std::string output_dir = getArg(args, "--output", "selfplay_data");
    if (config["self_play"] && config["self_play"]["output_dir"]) {
        output_dir = config["self_play"]["output_dir"].as<std::string>();
    }
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // Generate games
    std::cout << "Generating " << num_games << " self-play games..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    auto games = manager.generateGames(game_type, num_games, board_size);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    std::cout << "Generated " << games.size() << " games in " 
             << duration << " seconds" << std::endl;
    
    // Save games
    std::cout << "Saving games to " << output_dir << std::endl;
    std::string format = "json";
    if (config["self_play"] && config["self_play"]["output_format"]) {
        format = config["self_play"]["output_format"].as<std::string>();
    }
    
    manager.saveGames(games, output_dir, format);
    
    std::cout << "Self-play completed successfully" << std::endl;
    return 0;
}

// Handler function for train command
int handleTrain(const std::vector<std::string>& args) {
    // Check if help was requested
    if (hasArg(args, "-h") || hasArg(args, "--help")) {
        std::cout << "Usage: train [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --config CONFIG   Configuration file path" << std::endl;
        std::cout << "  --data DIR        Training data directory" << std::endl;
        std::cout << "  --model PATH      Input model file path" << std::endl;
        std::cout << "  --output PATH     Output model file path" << std::endl;
        std::cout << "  --epochs N        Number of training epochs" << std::endl;
        return 0;
    }
    
    std::cout << "Training functionality is implemented in Python." << std::endl;
    std::cout << "Please use the Python training script directly." << std::endl;
    
    // The C++ CLI doesn't directly implement the training loop because training is
    // typically done in Python with PyTorch or TensorFlow. The C++ CLI provides 
    // the self-play, evaluation, and interactive play functionality, while training
    // is handled by the Python bindings.
    
    return 0;
}

// Handler function for eval command
int handleEval(const std::vector<std::string>& args) {
    // Check if help was requested
    if (hasArg(args, "-h") || hasArg(args, "--help")) {
        std::cout << "Usage: eval [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --config CONFIG   Configuration file path" << std::endl;
        std::cout << "  --model1 PATH     First model file path" << std::endl;
        std::cout << "  --model2 PATH     Second model file path" << std::endl;
        std::cout << "  --game TYPE       Game type (gomoku, chess, go)" << std::endl;
        std::cout << "  --board-size N    Board size (for gomoku and go)" << std::endl;
        std::cout << "  --num-games N     Number of games to play" << std::endl;
        std::cout << "  --output PATH     Output file path" << std::endl;
        return 0;
    }
    
    // Load config file
    std::string config_path = getArg(args, "--config", "config.yaml");
    YAML::Node config = loadConfig(config_path);
    
    // Parse game type and board size
    core::GameType game_type = parseGameType(args, config);
    int board_size = parseBoardSize(args, config, game_type);
    
    // Create neural networks
    auto model1 = createNeuralNetwork(args, config, game_type, board_size, "--model1");
    auto model2 = createNeuralNetwork(args, config, game_type, board_size, "--model2");

    if (!model1 || !model2) {
        std::cerr << "Error: Failed to create one or both neural networks for evaluation. Exiting." << std::endl;
        return 1;
    }
    
    // Parse MCTS settings
    mcts::MCTSSettings mcts_settings = parseMctsSettings(config);
    
    // Create evaluation settings
    evaluation::EvaluationSettings eval_settings;
    eval_settings.mcts_settings_first = mcts_settings;
    eval_settings.mcts_settings_second = mcts_settings;
    
    // No Dirichlet noise for evaluation
    eval_settings.mcts_settings_first.add_dirichlet_noise = false;
    eval_settings.mcts_settings_second.add_dirichlet_noise = false;
    
    // Low temperature for deterministic play
    eval_settings.mcts_settings_first.temperature = 0.1f;
    eval_settings.mcts_settings_second.temperature = 0.1f;
    
    // Parse specific evaluation settings
    if (config["evaluation"]) {
        auto eval_config = config["evaluation"];
        
        if (eval_config["num_games"]) {
            eval_settings.num_games = eval_config["num_games"].as<int>();
        }
        
        if (eval_config["num_parallel_games"]) {
            eval_settings.num_parallel_games = eval_config["num_parallel_games"].as<int>();
        }
        
        if (eval_config["max_moves"]) {
            eval_settings.max_moves = eval_config["max_moves"].as<int>();
        }
    }
    
    // Override with command line arguments
    std::string num_games_str = getArg(args, "--num-games");
    if (!num_games_str.empty()) {
        try {
            eval_settings.num_games = std::stoi(num_games_str);
        } catch (...) {
            std::cerr << "Invalid number of games: " << num_games_str << std::endl;
        }
    }
    
    // Create model evaluator
    evaluation::ModelEvaluator evaluator(model1, model2, eval_settings);
    
    // Run tournament
    std::cout << "Evaluating models with " << eval_settings.num_games << " games..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    auto result = evaluator.runTournament(game_type, board_size);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    // Print results
    std::cout << "Evaluation completed in " << duration << " seconds" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Model 1 wins: " << result.wins_first << std::endl;
    std::cout << "  Model 2 wins: " << result.wins_second << std::endl;
    std::cout << "  Draws: " << result.draws << std::endl;
    std::cout << "  Total games: " << result.total_games << std::endl;
    std::cout << "  ELO difference: " << result.elo_diff << std::endl;
    
    // Save results to file if requested
    std::string output_path = getArg(args, "--output");
    if (!output_path.empty()) {
        std::ofstream file(output_path);
        if (file) {
            file << "Model 1 wins: " << result.wins_first << std::endl;
            file << "Model 2 wins: " << result.wins_second << std::endl;
            file << "Draws: " << result.draws << std::endl;
            file << "Total games: " << result.total_games << std::endl;
            file << "ELO difference: " << result.elo_diff << std::endl;
            
            // Save individual match results
            file << "\nMatch results:" << std::endl;
            for (const auto& match : result.matches) {
                file << "Match " << match.match_id << ": ";
                if (match.result == 0) {
                    file << "Draw";
                } else if (match.result == 1) {
                    file << "Model " << (match.first_model_as_player1 ? "1" : "2") << " wins";
                } else {
                    file << "Model " << (match.first_model_as_player1 ? "2" : "1") << " wins";
                }
                file << " (" << match.moves.size() << " moves)" << std::endl;
            }
            
            std::cout << "Results saved to " << output_path << std::endl;
        } else {
            std::cerr << "Failed to write to output file: " << output_path << std::endl;
        }
    }
    
    return 0;
}

// Handler function for play command
int handlePlay(const std::vector<std::string>& args) {
    // Check if help was requested
    if (hasArg(args, "-h") || hasArg(args, "--help")) {
        std::cout << "Usage: play [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --config CONFIG   Configuration file path" << std::endl;
        std::cout << "  --model PATH      Model file path" << std::endl;
        std::cout << "  --game TYPE       Game type (gomoku, chess, go)" << std::endl;
        std::cout << "  --board-size N    Board size (for gomoku and go)" << std::endl;
        std::cout << "  --simulations N   Number of MCTS simulations" << std::endl;
        return 0;
    }
    
    // Load config file
    std::string config_path = getArg(args, "--config", "config.yaml");
    YAML::Node config = loadConfig(config_path);
    
    // Parse game type and board size
    core::GameType game_type = parseGameType(args, config);
    int board_size = parseBoardSize(args, config, game_type);
    
    // Create neural network
    auto model = createNeuralNetwork(args, config, game_type, board_size);
    if (!model) {
        std::cerr << "Error: Failed to create neural network. Exiting interactive play." << std::endl;
        return 1;
    }
    
    // Parse MCTS settings
    mcts::MCTSSettings mcts_settings = parseMctsSettings(config);
    
    // Override simulations if specified
    std::string sims_str = getArg(args, "--simulations");
    if (!sims_str.empty()) {
        try {
            mcts_settings.num_simulations = std::stoi(sims_str);
        } catch (...) {
            std::cerr << "Invalid number of simulations: " << sims_str << std::endl;
        }
    }
    
    // No Dirichlet noise for interactive play
    mcts_settings.add_dirichlet_noise = false;
    
    // Low temperature for deterministic play
    mcts_settings.temperature = 0.1f;
    
    // Create MCTS engine
    mcts::MCTSEngine engine(model, mcts_settings);
    
    // Create game state with proper board size
    std::unique_ptr<core::IGameState> game;
    
    if (game_type == core::GameType::CHESS) {
        game = std::make_unique<alphazero::chess::ChessState>();
    } else if (game_type == core::GameType::GO) {
        game = std::make_unique<alphazero::go::GoState>(board_size);
    } else {
        game = std::make_unique<games::gomoku::GomokuState>(board_size);
    }
    
    // Interactive play loop
    std::cout << "Starting interactive play..." << std::endl;
    std::cout << "Game: " << core::gameTypeToString(game_type) << std::endl;
    std::cout << "Board size: " << board_size << std::endl;
    std::cout << "Simulations: " << mcts_settings.num_simulations << std::endl;
    std::cout << std::endl;
    
    bool human_player = 1;  // Human plays as player 1 by default
    if (hasArg(args, "--play-as-black") || hasArg(args, "--play-as-2")) {
        human_player = 2;
    }
    
    while (!game->isTerminal()) {
        // Display current board
        std::cout << game->toString() << std::endl;
        
        int current_player = game->getCurrentPlayer();
        std::cout << "Player " << current_player << "'s turn" << std::endl;
        
        if (current_player == human_player) {
            // Human's turn
            std::cout << "Enter your move: ";
            std::string move_str;
            std::getline(std::cin, move_str);
            
            // Check for quit command
            if (move_str == "quit" || move_str == "exit") {
                break;
            }
            
            // Parse move
            auto action = game->stringToAction(move_str);
            if (!action) {
                std::cout << "Invalid move. Try again." << std::endl;
                continue;
            }
            
            // Check if move is legal
            if (!game->isLegalMove(*action)) {
                std::cout << "Illegal move. Try again." << std::endl;
                continue;
            }
            
            // Make move
            game->makeMove(*action);
        } else {
            // AI's turn
            std::cout << "AI is thinking..." << std::endl;
            
            auto start_time = std::chrono::steady_clock::now();
            auto result = engine.search(*game);
            auto end_time = std::chrono::steady_clock::now();
            
            auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            std::cout << "AI plays: " << game->actionToString(result.action) << std::endl;
            std::cout << "Value: " << result.value << std::endl;
            std::cout << "Nodes: " << result.stats.total_nodes 
                     << " in " << time_ms << " ms"
                     << " (" << result.stats.nodes_per_second << " nodes/s)" << std::endl;
            
            // Make move
            game->makeMove(result.action);
        }
        
        // Check if game is over
        if (game->isTerminal()) {
            std::cout << game->toString() << std::endl;
            std::cout << "Game over!" << std::endl;
            
            auto result = game->getGameResult();
            if (result == core::GameResult::WIN_PLAYER1) {
                std::cout << "Player 1 wins!" << std::endl;
            } else if (result == core::GameResult::WIN_PLAYER2) {
                std::cout << "Player 2 wins!" << std::endl;
            } else {
                std::cout << "It's a draw!" << std::endl;
            }
        }
    }
    
    return 0;
}

// Main function
int main(int argc, char** argv) {
    try {
        // Create CLI manager
        cli::CLIManager cli;
        
        // Add commands
        cli.addCommand("self-play", "Generate self-play games for training", handleSelfPlay);
        cli.addCommand("train", "Train neural network from self-play data", handleTrain);
        cli.addCommand("eval", "Evaluate model strength", handleEval);
        cli.addCommand("play", "Play against AI", handlePlay);
        
        // Run CLI
        return cli.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}