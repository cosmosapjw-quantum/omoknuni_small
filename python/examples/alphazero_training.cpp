// examples/alphazero_training.cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "selfplay/self_play_manager.h"
#include "training/training_data_manager.h"
#include "evaluation/model_evaluator.h"

using json = nlohmann::json;
using namespace alphazero;

// Create a timestamp string for logging
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Log message to console and file
void log(const std::string& message, std::ofstream& log_file) {
    std::string log_message = getTimestamp() + " - " + message;
    std::cout << log_message << std::endl;
    log_file << log_message << std::endl;
    log_file.flush();
}

// Parse command line arguments
struct CommandLineArgs {
    std::string game_type = "gomoku";
    int board_size = 15;
    int num_iterations = 100;
    int games_per_iteration = 50;
    int evaluation_games = 20;
    int num_threads = 4;
    int batch_size = 8;
    int num_simulations = 800;
    std::string output_dir = "alphazero_training";
    bool resume = false;
    std::string initial_model = "";
};

CommandLineArgs parseArgs(int argc, char** argv) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--game" && i + 1 < argc) {
            args.game_type = argv[++i];
        } else if (arg == "--board-size" && i + 1 < argc) {
            args.board_size = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            args.num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--games-per-iteration" && i + 1 < argc) {
            args.games_per_iteration = std::stoi(argv[++i]);
        } else if (arg == "--evaluation-games" && i + 1 < argc) {
            args.evaluation_games = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--simulations" && i + 1 < argc) {
            args.num_simulations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--resume") {
            args.resume = true;
        } else if (arg == "--initial-model" && i + 1 < argc) {
            args.initial_model = argv[++i];
        } else if (arg == "--help") {
            std::cout << "AlphaZero Training Pipeline\n\n"
                     << "Usage: " << argv[0] << " [options]\n\n"
                     << "Options:\n"
                     << "  --game TYPE                Game type (gomoku, chess, go) [default: gomoku]\n"
                     << "  --board-size SIZE          Board size [default: 15]\n"
                     << "  --iterations NUM           Number of training iterations [default: 100]\n"
                     << "  --games-per-iteration NUM  Number of self-play games per iteration [default: 50]\n"
                     << "  --evaluation-games NUM     Number of evaluation games per iteration [default: 20]\n"
                     << "  --threads NUM              Number of threads [default: 4]\n"
                     << "  --batch-size SIZE          Neural network batch size [default: 8]\n"
                     << "  --simulations NUM          Number of MCTS simulations [default: 800]\n"
                     << "  --output DIR               Output directory [default: alphazero_training]\n"
                     << "  --resume                   Resume training from last checkpoint\n"
                     << "  --initial-model PATH       Path to initial model file\n"
                     << "  --help                     Show this help message\n";
            exit(0);
        }
    }
    
    return args;
}

int main(int argc, char** argv) {
    try {
        // Parse arguments
        CommandLineArgs args = parseArgs(argc, argv);
        
        // Create output directory
        std::filesystem::create_directories(args.output_dir);
        
        // Open log file
        std::ofstream log_file(args.output_dir + "/training.log", 
                              args.resume ? std::ios::app : std::ios::out);
        
        log("Starting AlphaZero training pipeline", log_file);
        log("Game type: " + args.game_type, log_file);
        log("Board size: " + std::to_string(args.board_size), log_file);
        
        // Get game type
        core::GameType game_type;
        if (args.game_type == "gomoku") {
            game_type = core::GameType::GOMOKU;
        } else if (args.game_type == "chess") {
            game_type = core::GameType::CHESS;
        } else if (args.game_type == "go") {
            game_type = core::GameType::GO;
        } else {
            throw std::runtime_error("Unknown game type: " + args.game_type);
        }
        
        // Create a sample game to determine input channels
        auto sample_game = core::GameFactory::createGame(game_type);
        if (game_type == core::GameType::GOMOKU || game_type == core::GameType::GO) {
            // Recreate with specified board size
            sample_game = core::GameFactory::createGame(game_type);
        }
        
        auto tensor = sample_game->getEnhancedTensorRepresentation();
        int input_channels = static_cast<int>(tensor.size());
        int policy_size = sample_game->getActionSpaceSize();
        
        log("Input channels: " + std::to_string(input_channels), log_file);
        log("Policy size: " + std::to_string(policy_size), log_file);
        
        // Create models directory
        std::string models_dir = args.output_dir + "/models";
        std::filesystem::create_directories(models_dir);
        
        // Create data directory
        std::string data_dir = args.output_dir + "/data";
        std::filesystem::create_directories(data_dir);
        
        // Determine the last iteration if resuming
        int start_iteration = 0;
        std::string best_model_path = "";
        
        if (args.resume) {
            // Find the highest iteration model
            for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
                std::string filename = entry.path().filename().string();
                if (filename.find("model_iter_") == 0) {
                    int iter = std::stoi(filename.substr(11, filename.find(".") - 11));
                    start_iteration = std::max(start_iteration, iter + 1);
                }
            }
            
            if (start_iteration > 0) {
                best_model_path = models_dir + "/model_iter_" + 
                                 std::to_string(start_iteration - 1) + ".pt";
                log("Resuming from iteration " + std::to_string(start_iteration), log_file);
                log("Using model: " + best_model_path, log_file);
            } else {
                log("No previous models found, starting from scratch", log_file);
            }
        }
        
        // Create or load best model
        std::shared_ptr<nn::ResNetModel> best_model;
        
        if (!best_model_path.empty()) {
            // Load from best model path
            log("Loading best model from " + best_model_path, log_file);
            best_model = nn::NeuralNetworkFactory::loadResNet(
                best_model_path, input_channels, args.board_size, policy_size);
        } else if (!args.initial_model.empty()) {
            // Load from initial model path
            log("Loading initial model from " + args.initial_model, log_file);
            best_model = nn::NeuralNetworkFactory::loadResNet(
                args.initial_model, input_channels, args.board_size, policy_size);
        } else {
            // Create new model
            log("Creating new model", log_file);
            best_model = nn::NeuralNetworkFactory::createResNet(
                input_channels, args.board_size, 10, 128, policy_size);
        }
        
        // Save initial model if starting from scratch
        if (start_iteration == 0 && best_model_path.empty()) {
            best_model_path = models_dir + "/model_iter_0.pt";
            best_model->save(best_model_path);
            log("Saved initial model to " + best_model_path, log_file);
        }
        
        // Create training data manager
        training::TrainingDataSettings training_data_settings;
        training_data_settings.max_examples = 500000;
        training_data_settings.batch_size = 2048;
        training_data_settings.sample_recent_iterations = 20;
        
        training::TrainingDataManager training_data(training_data_settings);
        
        // Load existing training data if resuming
        if (args.resume) {
            log("Loading existing training data", log_file);
            try {
                training_data.load(data_dir, "binary");
                log("Loaded " + std::to_string(training_data.getTotalExamples()) + 
                   " training examples", log_file);
            } catch (const std::exception& e) {
                log("Failed to load training data: " + std::string(e.what()), log_file);
                log("Starting with empty training data", log_file);
            }
        }
        
        // Training loop
        for (int iteration = start_iteration; iteration < args.num_iterations; ++iteration) {
            log("\n=== Iteration " + std::to_string(iteration + 1) + "/" + 
               std::to_string(args.num_iterations) + " ===", log_file);
            
            // Create a new model for this iteration
            std::shared_ptr<nn::ResNetModel> new_model;
            if (iteration > start_iteration) {
                // Clone the best model
                new_model = nn::NeuralNetworkFactory::createResNet(
                    input_channels, args.board_size, 10, 128, policy_size);
                
                // Copy parameters from best model
                for (size_t i = 0; i < best_model->parameters().size(); ++i) {
                    new_model->parameters()[i].data().copy_(best_model->parameters()[i].data());
                }
                
                log("Created new model based on best model", log_file);
            } else {
                // Use the existing best model
                new_model = best_model;
                log("Using existing model for first iteration", log_file);
            }
            
            // Train model
            
            // Self-play
            log("Generating self-play games", log_file);
            
            selfplay::SelfPlaySettings selfplay_settings;
            selfplay_settings.mcts_settings.num_simulations = args.num_simulations;
            selfplay_settings.mcts_settings.num_threads = 1;  // One thread per game
            selfplay_settings.mcts_settings.batch_size = args.batch_size;
            selfplay_settings.mcts_settings.add_dirichlet_noise = true;
            selfplay_settings.num_parallel_games = args.num_threads;
            selfplay_settings.temperature_threshold = 30;
            
            selfplay::SelfPlayManager selfplay(new_model, selfplay_settings);
            
            auto start_time = std::chrono::steady_clock::now();
            auto games = selfplay.generateGames(game_type, args.games_per_iteration, args.board_size);
            auto end_time = std::chrono::steady_clock::now();
            
            auto selfplay_time = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();
            
            log("Generated " + std::to_string(games.size()) + 
               " games in " + std::to_string(selfplay_time) + "s", log_file);
            
            // Save games
            std::string games_dir = data_dir + "/games_iter_" + std::to_string(iteration);
            std::filesystem::create_directories(games_dir);
            
            selfplay.saveGames(games, games_dir, "json");
            log("Saved games to " + games_dir, log_file);
            
            // Add to training data
            log("Adding games to training data", log_file);
            training_data.addGames(games, iteration);
            log("Total training examples: " + std::to_string(training_data.getTotalExamples()), log_file);
            
            // Save training data
            log("Saving training data", log_file);
            training_data.save(data_dir, "binary");
            
            // Save the new model
            std::string model_path = models_dir + "/model_iter_" + 
                                    std::to_string(iteration) + ".pt";
            new_model->save(model_path);
            log("Saved model to " + model_path, log_file);
            
            // Evaluate model (for iteration > 0)
            if (iteration > start_iteration) {
                log("Evaluating new model against best model", log_file);
                
                evaluation::EvaluationSettings eval_settings;
                eval_settings.mcts_settings_first = selfplay_settings.mcts_settings;
                eval_settings.mcts_settings_second = selfplay_settings.mcts_settings;
                eval_settings.mcts_settings_first.add_dirichlet_noise = false;
                eval_settings.mcts_settings_second.add_dirichlet_noise = false;
                eval_settings.num_games = args.evaluation_games;
                eval_settings.num_parallel_games = args.num_threads;
                
                evaluation::ModelEvaluator evaluator(new_model, best_model, eval_settings);
                
                start_time = std::chrono::steady_clock::now();
                auto tournament_result = evaluator.runTournament(game_type, args.board_size);
                end_time = std::chrono::steady_clock::now();
                
                auto eval_time = std::chrono::duration_cast<std::chrono::seconds>(
                    end_time - start_time).count();
                
                log("Evaluation results:", log_file);
                log("  New model wins: " + std::to_string(tournament_result.wins_first), log_file);
                log("  Best model wins: " + std::to_string(tournament_result.wins_second), log_file);
                log("  Draws: " + std::to_string(tournament_result.draws), log_file);
                log("  ELO difference: " + std::to_string(tournament_result.elo_diff), log_file);
                log("  Time: " + std::to_string(eval_time) + "s", log_file);
                
                // Save tournament results
                std::string eval_file = args.output_dir + "/evaluation_iter_" + 
                                      std::to_string(iteration) + ".json";
                
                json eval_json;
                eval_json["iteration"] = iteration;
                eval_json["wins_new"] = tournament_result.wins_first;
                eval_json["wins_best"] = tournament_result.wins_second;
                eval_json["draws"] = tournament_result.draws;
                eval_json["elo_diff"] = tournament_result.elo_diff;
                eval_json["time_seconds"] = eval_time;
                
                std::ofstream eval_out(eval_file);
                eval_out << eval_json.dump(2);
                
                // Update best model if new model is better
                if (tournament_result.elo_diff > 0) {
                    log("New model is better! Updating best model", log_file);
                    best_model = new_model;
                    
                    // Create a copy as best_model.pt
                    std::string best_path = models_dir + "/best_model.pt";
                    best_model->save(best_path);
                } else {
                    log("Best model is still stronger, keeping it", log_file);
                }
            }
        }
        
        log("Training complete!", log_file);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}