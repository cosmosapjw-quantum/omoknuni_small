// src/cli/alphazero_pipeline.cpp
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <map>
#include <iomanip>
#include <ctime>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// Extra include for Torch CUDA functions and our utilities
#include <torch/torch.h>
#include "utils/device_utils.h"
#include "utils/cuda_utils.h"
#include "training/dataset.h"
#include "training/data_loader.h"

#include "cli/alphazero_pipeline.h"
#include "core/game_export.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"
#include "selfplay/self_play_manager.h"
#include "evaluation/model_evaluator.h"

namespace alphazero {
namespace cli {

// Helper function to create a readable timestamp
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

// Parse configuration from a YAML-like file
PipelineConfig parsePipelineConfig(const std::string& config_path) {
    PipelineConfig config;
    
    // Load configuration
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }
    
    // Parse YAML configuration
    std::string line;
    std::map<std::string, std::string> config_map;
    std::string current_section;
    
    while (std::getline(config_file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check if this is a section header
        if (line[line.size() - 1] == ':') {
            current_section = line.substr(0, line.size() - 1);
            continue;
        }
        
        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // Store with section prefix if we're in a section
            if (!current_section.empty()) {
                config_map[current_section + "." + key] = value;
            } else {
                config_map[key] = value;
            }
        }
    }
    
    // Extract values from the parsed configuration
    config.game_type_str = config_map["game"];
    config.board_size = std::stoi(config_map["board_size"]);
    
    // Convert game type string to enum
    if (config.game_type_str == "gomoku") {
        config.game_type = core::GameType::GOMOKU;
    } else if (config.game_type_str == "chess") {
        config.game_type = core::GameType::CHESS;
    } else if (config.game_type_str == "go") {
        config.game_type = core::GameType::GO;
    } else {
        throw std::runtime_error("Unknown game type: " + config.game_type_str);
    }
    
    // Set input channels and policy size based on game type
    switch (config.game_type) {
        case core::GameType::GOMOKU:
            config.num_channels = 17;  // Enhanced representation with history
            config.policy_size = config.board_size * config.board_size;
            break;
        case core::GameType::CHESS:
            config.num_channels = 17;  // 12 piece planes + 3 special + 1 turn + 1 move count
            config.policy_size = 64 * 73;  // Source, destination, promotions
            break;
        case core::GameType::GO:
            config.num_channels = 17;  // 2 player planes + 1 turn + 8 history
            config.policy_size = config.board_size * config.board_size + 1;  // +1 for pass
            break;
        default:
            throw std::runtime_error("Unsupported game type");
    }
    
    // Neural network settings
    config.num_res_blocks = std::stoi(config_map["network.num_res_blocks"]);
    config.num_filters = std::stoi(config_map["network.num_filters"]);
    
    // Path settings
    config.model_dir = config_map.count("model_dir") ? config_map["model_dir"] : "models";
    config.data_dir = config_map.count("data_dir") ? config_map["data_dir"] : "data";
    config.log_dir = config_map.count("log_dir") ? config_map["log_dir"] : "logs";
    
    // Training settings
    config.num_iterations = config_map.count("train.iterations") ? std::stoi(config_map["train.iterations"]) : 5;
    config.epochs_per_iteration = config_map.count("train.epochs") ? std::stoi(config_map["train.epochs"]) : 10;
    config.batch_size = config_map.count("train.batch_size") ? std::stoi(config_map["train.batch_size"]) : 256;
    config.learning_rate = config_map.count("train.learning_rate") ? std::stof(config_map["train.learning_rate"]) : 0.001f;
    config.weight_decay = config_map.count("train.weight_decay") ? std::stof(config_map["train.weight_decay"]) : 1e-4f;
    
    // Self-play settings
    config.games_per_iteration = config_map.count("self_play.num_games") ? 
        std::stoi(config_map["self_play.num_games"]) : 100;
    config.num_parallel_games = config_map.count("self_play.num_parallel_games") ? 
        std::stoi(config_map["self_play.num_parallel_games"]) : 4;
    config.max_moves = config_map.count("self_play.max_moves") ? 
        std::stoi(config_map["self_play.max_moves"]) : 0;
    config.temperature_threshold = config_map.count("self_play.temperature_threshold") ? 
        std::stoi(config_map["self_play.temperature_threshold"]) : 30;
    config.high_temperature = config_map.count("self_play.high_temperature") ? 
        std::stof(config_map["self_play.high_temperature"]) : 1.0f;
    config.low_temperature = config_map.count("self_play.low_temperature") ? 
        std::stof(config_map["self_play.low_temperature"]) : 0.0f;
        
    // MCTS settings
    config.num_simulations = config_map.count("mcts.num_simulations") ? 
        std::stoi(config_map["mcts.num_simulations"]) : 200;
    config.num_threads = config_map.count("mcts.num_threads") ? 
        std::stoi(config_map["mcts.num_threads"]) : 8;
    config.mcts_batch_size = config_map.count("mcts.batch_size") ? 
        std::stoi(config_map["mcts.batch_size"]) : 64;
    config.exploration_constant = config_map.count("mcts.exploration_constant") ? 
        std::stof(config_map["mcts.exploration_constant"]) : 1.5f;
    config.virtual_loss = config_map.count("mcts.virtual_loss") ? 
        std::stoi(config_map["mcts.virtual_loss"]) : 3;
    config.add_dirichlet_noise = config_map.count("mcts.add_dirichlet_noise") ? 
        (config_map["mcts.add_dirichlet_noise"] == "true") : true;
    config.dirichlet_alpha = config_map.count("mcts.dirichlet_alpha") ? 
        std::stof(config_map["mcts.dirichlet_alpha"]) : 0.3f;
    config.dirichlet_epsilon = config_map.count("mcts.dirichlet_epsilon") ? 
        std::stof(config_map["mcts.dirichlet_epsilon"]) : 0.25f;
    config.temperature = config_map.count("mcts.temperature") ? 
        std::stof(config_map["mcts.temperature"]) : 1.0f;
    config.batch_timeout_ms = config_map.count("mcts.batch_timeout_ms") ? 
        std::stoi(config_map["mcts.batch_timeout_ms"]) : 100;
    
    // Evaluation settings
    config.num_eval_games = config_map.count("evaluation.num_games") ? 
        std::stoi(config_map["evaluation.num_games"]) : 40;
    config.elo_threshold = config_map.count("evaluation.elo_threshold") ? 
        std::stof(config_map["evaluation.elo_threshold"]) : 10.0f;
    
    // Arena settings
    config.use_arena = config_map.count("arena.enabled") ? 
        (config_map["arena.enabled"] == "true") : true;
    config.arena_games = config_map.count("arena.num_games") ? 
        std::stoi(config_map["arena.num_games"]) : 20;
    config.arena_threads = config_map.count("arena.num_threads") ? 
        std::stoi(config_map["arena.num_threads"]) : 4;
    
    return config;
}

// Create a timestamp-based directory
std::string createTimestampedDirectory(const std::string& base_dir, const std::string& prefix) {
    std::string timestamp = getCurrentTimestamp();
    std::string dir_name = base_dir + "/" + prefix + "_" + timestamp;
    std::filesystem::create_directories(dir_name);
    return dir_name;
}

// Simple ELO calculation based on win/loss results
float calculateElo(int wins, int losses, int draws) {
    if (wins + losses + draws == 0) return 0.0f;
    
    float score = (wins + 0.5f * draws) / (wins + losses + draws);
    // Convert to ELO difference using logistic conversion
    float elo_diff = -400.0f * std::log10(1.0f / score - 1.0f);
    return elo_diff;
}

// Function to run the complete AlphaZero pipeline
int runAlphaZeroPipeline(const PipelineConfig& config) {
    // Create log directory
    std::filesystem::create_directories(config.log_dir);
    std::string log_file_path = config.log_dir + "/alphazero_pipeline_" + getCurrentTimestamp() + ".log";
    std::ofstream log_file(log_file_path);
    
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not open log file at " << log_file_path << std::endl;
        return 1;
    }
    
    // Create data directories
    std::filesystem::create_directories(config.data_dir);
    std::filesystem::create_directories(config.model_dir);
    
    // Log start of pipeline
    auto log_message = [&](const std::string& message) {
        log_file << "[" << getCurrentTimestamp() << "] " << message << std::endl;
        std::cout << message << std::endl;
    };
    
    log_message("Starting AlphaZero training pipeline");
    log_message("Game: " + config.game_type_str + ", Board size: " + std::to_string(config.board_size));
    log_message("Training for " + std::to_string(config.num_iterations) + " iterations");
    
    // Current best model path
    std::string best_model_path = config.model_dir + "/best_model.pt";
    std::string latest_model_path = config.model_dir + "/latest_model.pt";
    
    // Check if best model exists, if not initialize it
    bool best_model_exists = std::filesystem::exists(best_model_path);
    
    if (!best_model_exists) {
        log_message("No best model found, creating initial model");
        
        try {
            bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
            log_message("Using GPU: " + std::string(use_gpu ? "Yes" : "No"));
            
            // Create initial model
            auto neural_net = alphazero::nn::NeuralNetworkFactory::createResNet(
                config.num_channels, config.board_size, config.num_res_blocks, 
                config.num_filters, config.policy_size, use_gpu);
            
            // Save initial model as best model
            neural_net->save(best_model_path);
            
            // Also save as latest model
            neural_net->save(latest_model_path);
            
            log_message("Created and saved initial model");
        } catch (const std::exception& e) {
            log_message("Error creating initial model: " + std::string(e.what()));
            return 1;
        }
    } else {
        log_message("Found existing best model at " + best_model_path);
    }
    
    // Set up self-play settings
    alphazero::selfplay::SelfPlaySettings self_play_settings;
    alphazero::mcts::MCTSSettings mcts_settings;
    
    mcts_settings.num_simulations = config.num_simulations;
    mcts_settings.num_threads = config.num_threads;
    mcts_settings.batch_size = config.mcts_batch_size;
    mcts_settings.batch_timeout = std::chrono::milliseconds(config.batch_timeout_ms);
    mcts_settings.exploration_constant = config.exploration_constant;
    mcts_settings.virtual_loss = config.virtual_loss;
    mcts_settings.add_dirichlet_noise = config.add_dirichlet_noise;
    mcts_settings.dirichlet_alpha = config.dirichlet_alpha;
    mcts_settings.dirichlet_epsilon = config.dirichlet_epsilon;
    mcts_settings.temperature = config.temperature;
    
    self_play_settings.mcts_settings = mcts_settings;
    self_play_settings.num_parallel_games = config.num_parallel_games;
    self_play_settings.max_moves = config.max_moves;
    self_play_settings.temperature_threshold = config.temperature_threshold;
    self_play_settings.high_temperature = config.high_temperature;
    self_play_settings.low_temperature = config.low_temperature;
    
    // Main training loop
    for (int iteration = 0; iteration < config.num_iterations; ++iteration) {
        log_message("Starting iteration " + std::to_string(iteration + 1) + "/" + 
                    std::to_string(config.num_iterations));
        
        std::string iteration_dir = createTimestampedDirectory(
            config.data_dir, "iteration_" + std::to_string(iteration + 1));
        
        // Load best model
        std::shared_ptr<alphazero::nn::NeuralNetwork> neural_net;
        try {
            bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
            neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
                best_model_path, config.num_channels, config.board_size, 
                config.num_res_blocks, config.num_filters,
                config.policy_size, use_gpu);
        } catch (const std::exception& e) {
            log_message("Error loading best model: " + std::string(e.what()));
            return 1;
        }
        
        // Step 1: Self-play
        log_message("Starting self-play generation...");
        std::string selfplay_dir = iteration_dir + "/selfplay";
        std::filesystem::create_directories(selfplay_dir);
        
        try {
            // Create self-play manager
            alphazero::selfplay::SelfPlayManager self_play_manager(neural_net, self_play_settings);
            
            // Generate games
            auto games = self_play_manager.generateGames(
                config.game_type, config.games_per_iteration, config.board_size);
            
            // Save games
            self_play_manager.saveGames(games, selfplay_dir, "json");
            
            log_message("Completed self-play: " + std::to_string(games.size()) + " games generated");
        } catch (const std::exception& e) {
            log_message("Error during self-play: " + std::string(e.what()));
            return 1;
        }
        
        // Step 2: Training
        log_message("Starting neural network training...");
        std::string model_checkpoint_path = iteration_dir + "/model_checkpoint.pt";
        
        try {
            // Load self-play games for training
            auto games = alphazero::selfplay::SelfPlayManager::loadGames(selfplay_dir, "json");
            
            if (games.empty()) {
                log_message("Error: No self-play games found for training");
                return 1;
            }
            
            log_message("Loaded " + std::to_string(games.size()) + " self-play games for training");
            
            // Convert games to training examples
            log_message("DEBUG: Starting convertToTrainingExamples...");
            auto examples = alphazero::selfplay::SelfPlayManager::convertToTrainingExamples(games);
            
            log_message("Generated " + std::to_string(examples.first.size()) + " training examples");
            
            // Debug: Output dimensions and ranges to verify data integrity
            if (!examples.first.empty()) {
                try {
                    log_message("DEBUG: First example dimensions: " + 
                               std::to_string(examples.first[0].size()) + " x " + 
                               std::to_string(examples.first[0][0].size()) + " x " + 
                               std::to_string(examples.first[0][0][0].size()));
                               
                    // Check policy size
                    log_message("DEBUG: First policy size: " + std::to_string(examples.second.first[0].size()));
                    
                    // Check value
                    log_message("DEBUG: First value: " + std::to_string(examples.second.second[0]));
                    
                    // Validate integrity of random samples
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> distrib(0, examples.first.size() - 1);
                    
                    for (int i = 0; i < 5; i++) {
                        int idx = distrib(gen);
                        log_message("DEBUG: Random sample #" + std::to_string(i) + " (index " + 
                                   std::to_string(idx) + ") - dimensions: " +
                                   std::to_string(examples.first[idx].size()) + " x " + 
                                   std::to_string(examples.first[idx][0].size()) + " x " + 
                                   std::to_string(examples.first[idx][0][0].size()) + 
                                   ", policy size: " + std::to_string(examples.second.first[idx].size()) +
                                   ", value: " + std::to_string(examples.second.second[idx]));
                    }
                } catch (const std::exception& e) {
                    log_message("DEBUG ERROR: Exception examining examples: " + std::string(e.what()));
                }
            }
            
            // Train the model
            log_message("Preparing model for training...");
            auto model = std::dynamic_pointer_cast<alphazero::nn::ResNetModel>(neural_net);
            
            if (!model) {
                log_message("ERROR: Failed to cast to ResNetModel - neural_net is not a valid ResNetModel");
                throw std::runtime_error("Failed to cast neural network to ResNetModel");
            }
            
            // Create optimizer - moved declaration to outer scope
            torch::optim::Adam optimizer(
                model->parameters(),
                torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay)
            );

            // Determine device for training - prefer GPU if available
            torch::Device device = torch::kCPU;
            try {
                if (alphazero::utils::DeviceUtils::isCudaAvailable()) {
                    log_message("CUDA is available, using GPU for training");
                    device = torch::Device(torch::kCUDA, 0);
                    
                    // Print CUDA device info
                    auto [used_mb, total_mb] = alphazero::cuda::get_memory_usage(0);
                    log_message("GPU memory: " + std::to_string(used_mb) + "MB used / " + 
                               std::to_string(total_mb) + "MB total");
                } else {
                    log_message("CUDA is not available, using CPU for training");
                }
            } catch (const std::exception& e) {
                log_message("WARNING: Error checking CUDA availability: " + std::string(e.what()));
                log_message("Falling back to CPU for training");
            }
            
            log_message("Using device for training: " + device.str());
            
            // Set model to training mode
            model->train(true);
            
            // Safely move the model to the device
            try {
                model->to(device);
                log_message("Successfully moved model to " + device.str());
            } catch (const std::exception& e) {
                log_message("WARNING: Could not move model to " + device.str() + ": " + e.what());
            }
            
            // Verify all example sizes match
            size_t num_samples = examples.first.size();
            if (examples.first.size() != examples.second.first.size() || 
                examples.first.size() != examples.second.second.size()) {
                log_message("ERROR: Mismatch in tensor dimensions - aborting training");
                log_message("ERROR: states: " + std::to_string(examples.first.size()) + 
                           ", policies: " + std::to_string(examples.second.first.size()) + 
                           ", values: " + std::to_string(examples.second.second.size()));
                throw std::runtime_error("Mismatch in training example dimensions");
            }
            
            // Get dimensions from the first example for batch size optimization
            size_t channels = examples.first[0].size();
            size_t height = examples.first[0][0].size();
            size_t width = examples.first[0][0][0].size();
            size_t policy_size = examples.second.first[0].size();
            
            log_message("Training data dimensions: " +
                std::to_string(channels) + "x" + std::to_string(height) + "x" + std::to_string(width) +
                " states, " + std::to_string(policy_size) + " policy size");
            
            // For GPU training, adjust batch size to prevent OOM errors
            size_t adjusted_batch_size = config.batch_size;
            if (device.is_cuda()) {
                // size_t tensor_size_per_example = channels * height * width * 4 + policy_size * 4 + 4; // Unused variable
                size_t optimal_batch = alphazero::cuda::get_optimal_batch_size(
                    channels, height, width, policy_size, 16, config.batch_size, 0.7);
                
                // Cap batch size based on optimal calculation
                adjusted_batch_size = std::min(static_cast<size_t>(config.batch_size), optimal_batch); // Cast config.batch_size to size_t
                if (adjusted_batch_size < config.batch_size) {
                    log_message("Adjusted batch size from " + std::to_string(config.batch_size) + 
                               " to " + std::to_string(adjusted_batch_size) + 
                               " to fit in GPU memory");
                }
            }
            
            // Create dataset using the new Dataset class
            std::shared_ptr<alphazero::training::AlphaZeroDataset> dataset;
            try {
                log_message("Creating dataset with " + std::to_string(num_samples) + " examples");
                // Create dataset on CPU first for safety
                dataset = std::make_shared<alphazero::training::AlphaZeroDataset>(
                    examples.first, examples.second.first, examples.second.second, torch::kCPU);
                
                log_message("Dataset created successfully with " + std::to_string(dataset->size()) + " examples");
                
                torch::Device dataset_actual_device = torch::kCPU; // Track the dataset's actual device

                // Once created, we can try moving it to the target device if it's not CPU
                if (device.is_cuda()) {
                    log_message("Attempting to move dataset to " + device.str());
                    try {
                        dataset->to(device); // This moves the TENSORS INSIDE the dataset to GPU
                        dataset_actual_device = device; // Update if move was successful
                        log_message("Dataset moved to " + device.str() + " successfully");
                    } catch (const std::exception& e) {
                        log_message("WARNING: Failed to move dataset to " + device.str() + 
                                   ", keeping on CPU: " + std::string(e.what()));
                        // dataset_actual_device remains torch::kCPU
                    }
                }
            } catch (const std::exception& e) {
                log_message("ERROR: Failed to create dataset: " + std::string(e.what()));
                throw;
            }
            
            // Determine number of worker threads for DataLoader
            size_t num_workers = 0;  // Default to single-threaded for stability
            
            // If we have at least 4 CPU cores, use multi-threading
            size_t num_cores = std::thread::hardware_concurrency();
            if (num_cores >= 4) {
                // Leave some cores for the main thread and other processes
                num_workers = std::min(num_cores / 2, size_t(4));
                log_message("Using " + std::to_string(num_workers) + " worker threads for data loading");
            } else {
                log_message("Using single-threaded data loading (detected " + 
                           std::to_string(num_cores) + " CPU cores)");
            }
            
            // Create DataLoader
            // Pin memory only if the target training device is CUDA AND the dataset's tensors are on CPU.
            bool pin_memory = device.is_cuda() && dataset_actual_device.is_cpu();
            std::unique_ptr<alphazero::training::DataLoader> dataloader;
            
            try {
                log_message("Creating DataLoader with batch size " + std::to_string(adjusted_batch_size) + 
                           " and pin_memory: " + (pin_memory ? "true" : "false"));
                dataloader = std::make_unique<alphazero::training::DataLoader>(
                    dataset, adjusted_batch_size, true, num_workers, pin_memory, false);
                
                log_message("DataLoader created successfully with " + 
                           std::to_string(dataloader->size()) + " batches");
            } catch (const std::exception& e) {
                log_message("ERROR: Failed to create DataLoader: " + std::string(e.what()));
                throw;
            }
            
            // Training loop
            log_message("Starting training for " + std::to_string(config.epochs_per_iteration) + " epochs");
            
            // Track statistics across all epochs
            float total_epoch_loss = 0.0f;
            float total_epoch_policy_loss = 0.0f;
            float total_epoch_value_loss = 0.0f;
            int total_epoch_batches = 0;
            
            for (int epoch = 0; epoch < config.epochs_per_iteration; epoch++) {
                // Track metrics for this epoch
                float epoch_loss = 0.0f;
                float epoch_policy_loss = 0.0f;
                float epoch_value_loss = 0.0f;
                int epoch_batches = 0;
                
                // Reset dataloader for the new epoch
                dataloader->reset();
                
                // Log epoch start
                log_message("Epoch " + std::to_string(epoch + 1) + "/" + 
                           std::to_string(config.epochs_per_iteration));
                
                // Train on batches
                size_t batch_count = 0;
                
                // Iterate through the dataset manually
                for (size_t batch_idx = 0; batch_idx < dataloader->size(); ++batch_idx) {
                    try {
                        // Get the batch
                        auto batch = dataloader->load_batch(batch_idx);
                        
                        // Move batch to model device if necessary
                        if (batch.device() != device) {
                            try {
                                batch.to(device);
                            } catch (const std::exception& e) {
                                log_message("WARNING: Failed to move batch to " + device.str() + 
                                          ", skipping batch: " + std::string(e.what()));
                                continue;
                            }
                        }
                        
                        // Log progress periodically
                        if (++batch_count % 10 == 0 || batch_count == 1) {
                            log_message("Processing batch " + std::to_string(batch_count) + "/" + 
                                      std::to_string(dataloader->size()));
                        }
                        
                        // Forward pass with safe device handling
                        torch::Tensor policy_logits, value;
                        try {
                            // Make sure batch is on the right device
                            if (batch.states.device() != device) {
                                batch.states = batch.states.to(device);
                                batch.policies = batch.policies.to(device);
                                batch.values = batch.values.to(device);
                            }
                            
                            // Perform forward pass with explicit device checking
                            if (device.is_cuda()) {
                                torch::cuda::synchronize();  // Ensure previous operations completed
                            }
                            
                            // Use torch::NoGradGuard no_grad;
                            std::tie(policy_logits, value) = model->forward(batch.states);
                            
                            if (device.is_cuda()) {
                                torch::cuda::synchronize();  // Ensure forward pass completed
                            }
                        } catch (const c10::Error& e) {
                            log_message("PyTorch error in forward pass: " + std::string(e.what()));
                            continue;  // Skip this batch
                        }
                        
                        // Compute loss with safe operations
                        auto policy_loss = -torch::sum(batch.policies * policy_logits) / batch.policies.size(0);
                        auto value_loss = torch::mean(torch::pow(value - batch.values, 2));
                        auto loss = policy_loss + value_loss;
                        
                        // Check for NaN values in loss
                        if (loss.isnan().any().item<bool>()) {
                            log_message("WARNING: NaN detected in loss. Skipping this batch.");
                            continue;
                        }
                        
                        // Backward and optimize
                        optimizer.zero_grad(); // Now in scope
                        loss.backward();
                        
                        // Check for NaN gradients
                        bool has_nan_grads = false;
                        for (const auto& param : model->parameters()) {
                            if (param.grad().defined() && param.grad().isnan().any().item<bool>()) {
                                has_nan_grads = true;
                                break;
                            }
                        }
                        
                        if (has_nan_grads) {
                            log_message("WARNING: NaN detected in gradients. Skipping optimizer step.");
                            optimizer.zero_grad();
                        } else {
                            // Step if gradients are valid
                            optimizer.step();
                            
                            // Track metrics
                            epoch_loss += loss.item<float>();
                            epoch_policy_loss += policy_loss.item<float>();
                            epoch_value_loss += value_loss.item<float>();
                            epoch_batches++;
                        }
                    } catch (const c10::Error& e) {
                        log_message("PyTorch error during training: " + std::string(e.what()));
                        // Continue to next batch
                    } catch (const std::exception& e) {
                        log_message("Error during training: " + std::string(e.what()));
                        // Continue to next batch
                    }
                    
                    // Periodically clean up memory
                    if (device.is_cuda() && batch_count % 10 == 0) {
                        // Use PyTorch's API for synchronization
                        torch::cuda::synchronize();
                        
                        // Synchronize CUDA device to flush operations
                        cudaDeviceSynchronize();
                    }
                }
                
                // Calculate average loss for the epoch
                if (epoch_batches > 0) {
                    float avg_loss = epoch_loss / epoch_batches;
                    float avg_policy_loss = epoch_policy_loss / epoch_batches;
                    float avg_value_loss = epoch_value_loss / epoch_batches;
                    
                    log_message("Epoch " + std::to_string(epoch + 1) + "/" + 
                               std::to_string(config.epochs_per_iteration) + 
                               ": Loss=" + std::to_string(avg_loss) + 
                               ", Policy Loss=" + std::to_string(avg_policy_loss) + 
                               ", Value Loss=" + std::to_string(avg_value_loss));
                    
                    // Update total statistics
                    total_epoch_loss += epoch_loss;
                    total_epoch_policy_loss += epoch_policy_loss;
                    total_epoch_value_loss += epoch_value_loss;
                    total_epoch_batches += epoch_batches;
                } else {
                    log_message("WARNING: No valid batches processed in epoch " + 
                               std::to_string(epoch + 1));
                }
                
                // Explicit cleanup after each epoch
                if (device.is_cuda()) {
                    // Use PyTorch's API for synchronization
                    torch::cuda::synchronize();
                    
                    // Synchronize CUDA device to flush operations
                    cudaDeviceSynchronize();
                }
            }
            
            // Calculate overall training statistics
            if (total_epoch_batches > 0) {
                float avg_total_loss = total_epoch_loss / total_epoch_batches;
                float avg_total_policy_loss = total_epoch_policy_loss / total_epoch_batches;
                float avg_total_value_loss = total_epoch_value_loss / total_epoch_batches;
                
                log_message("Training completed: Average Loss=" + std::to_string(avg_total_loss) + 
                           ", Policy Loss=" + std::to_string(avg_total_policy_loss) + 
                           ", Value Loss=" + std::to_string(avg_total_value_loss));
            }
            
            // Clean up DataLoader and Dataset
            dataloader.reset();
            dataset.reset();
            
            // Explicitly clean up CUDA memory
            if (device.is_cuda()) {
                // Use PyTorch's API for synchronization
                torch::cuda::synchronize();
                
                // Synchronize CUDA device to flush operations
                cudaDeviceSynchronize();
            }
            
            // Set to evaluation mode
            model->train(false);
            
            // Define model checkpoint path and save the model
            std::string model_checkpoint_path = iteration_dir + "/model_checkpoint.pt";
            neural_net->save(model_checkpoint_path);
            
            // Also save as latest model
            neural_net->save(latest_model_path);
            
            log_message("Completed training and saved model checkpoint");
            
        } catch (const std::exception& e) {
            log_message("Error during training: " + std::string(e.what()));
            return 1;
        }
        
        // Step 3: Evaluation against best model
        log_message("Starting model evaluation...");
        
        try {
            // Load best model and new model
            bool use_gpu = alphazero::nn::NeuralNetworkFactory::isCudaAvailable();
            
            auto best_model = alphazero::nn::NeuralNetworkFactory::loadResNet(
                best_model_path, config.num_channels, config.board_size, 
                config.num_res_blocks, config.num_filters,
                config.policy_size, use_gpu);
                
            auto new_model = alphazero::nn::NeuralNetworkFactory::loadResNet(
                model_checkpoint_path, config.num_channels, config.board_size, 
                config.num_res_blocks, config.num_filters,
                config.policy_size, use_gpu);
            
            // Create Arena settings
            alphazero::evaluation::EvaluationSettings eval_settings;
            eval_settings.mcts_settings_first = mcts_settings;  // Copy from self-play
            eval_settings.mcts_settings_second = mcts_settings;
            eval_settings.mcts_settings_first.add_dirichlet_noise = false; // No noise in evaluation
            eval_settings.mcts_settings_second.add_dirichlet_noise = false;
            eval_settings.num_parallel_games = config.arena_threads;
            eval_settings.num_games = config.arena_games;
            
            // Run arena matches (new model vs best model)
            alphazero::evaluation::ModelEvaluator evaluator(new_model, best_model, eval_settings);
            
            // Log starting the tournament
            log_message("Playing " + std::to_string(config.arena_games) + 
                       " evaluation games between new model and best model");
            
            // Run the tournament
            auto tournament_result = evaluator.runTournament(config.game_type, config.board_size);
            
            // Extract results
            int new_model_wins = tournament_result.wins_first;
            int best_model_wins = tournament_result.wins_second;
            int draws = tournament_result.draws;
            
            // Use ELO difference from the tournament result
            float elo_diff = tournament_result.elo_diff;
            
            log_message("Evaluation results: New model vs Best model");
            log_message("Games played: " + std::to_string(new_model_wins + best_model_wins + draws));
            log_message("New model wins: " + std::to_string(new_model_wins) + 
                       " (" + std::to_string(100.0f * new_model_wins / config.arena_games) + "%)");
            log_message("Best model wins: " + std::to_string(best_model_wins) + 
                       " (" + std::to_string(100.0f * best_model_wins / config.arena_games) + "%)");
            log_message("Draws: " + std::to_string(draws) + 
                       " (" + std::to_string(100.0f * draws / config.arena_games) + "%)");
            log_message("ELO difference: " + std::to_string(elo_diff));
            
            // Step 4: Model acceptance and rotation
            if (elo_diff >= config.elo_threshold) {
                log_message("New model accepted as best model (ELO gain: " + 
                           std::to_string(elo_diff) + ")");
                
                // Save new model as versioned checkpoint
                std::string versioned_path = config.model_dir + "/model_" + 
                                          getCurrentTimestamp() + "_elo" + 
                                          std::to_string(static_cast<int>(elo_diff)) + ".pt";
                std::filesystem::copy_file(model_checkpoint_path, versioned_path, 
                                         std::filesystem::copy_options::overwrite_existing);
                
                // Update best model
                std::filesystem::copy_file(model_checkpoint_path, best_model_path, 
                                         std::filesystem::copy_options::overwrite_existing);
            } else {
                log_message("New model rejected (ELO gain: " + 
                           std::to_string(elo_diff) + " < threshold: " + 
                           std::to_string(config.elo_threshold) + ")");
            }
            
        } catch (const std::exception& e) {
            log_message("Error during evaluation: " + std::string(e.what()));
            return 1;
        }
        
        log_message("Completed iteration successfully");
    }
    
    log_message("AlphaZero training pipeline completed successfully");
    return 0;
}

} // namespace cli
} // namespace alphazero