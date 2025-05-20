#include "cli/alphazero_pipeline.h"
#include "selfplay/self_play_manager.h"
#include "training/training_data_manager.h"
#include "training/dataset.h"
#include "training/data_loader.h"
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"
#include "nn/ddw_randwire_resnet.h"
#include "utils/debug_monitor.h"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

namespace alphazero {
namespace cli {

// Helper to get a formatted timestamp string
std::string getTimestampString() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

AlphaZeroPipeline::AlphaZeroPipeline(const AlphaZeroPipelineConfig& config)
    : config_(config), current_iteration_(0) {
    
    // Initialize logging
    initializeLogging();
    
    // Create necessary directories
    createDirectories();
    
    // Load or initialize neural network
    initializeNeuralNetwork();
}

AlphaZeroPipeline::~AlphaZeroPipeline() {
    // Clean up resources if needed
}

void AlphaZeroPipeline::run() {
    std::cout << "Starting AlphaZero pipeline with game: " 
              << core::gameTypeToString(config_.game_type) << std::endl;
    
    try {
        // Main AlphaZero iteration loop
        for (int i = 0; i < config_.num_iterations; ++i) {
            current_iteration_ = i;
            std::cout << "Starting iteration " << i + 1 << " of " << config_.num_iterations << std::endl;
            
            // Create iteration directory
            std::string iteration_dir = createIterationDirectory(i);
            
            // Step 1: Self-play
            std::cout << "Starting self-play phase for iteration " << i + 1 << std::endl;
            std::vector<selfplay::GameData> games = runSelfPlay(iteration_dir);
            std::cout << "Self-play phase completed with " << games.size() << " games" << std::endl;
            
            // Step 2: Train neural network
            std::cout << "Starting training phase for iteration " << i + 1 << std::endl;
            float train_loss = trainNeuralNetwork(games, iteration_dir);
            std::cout << "Training phase completed with final loss: " << train_loss << std::endl;
            
            // Step 3: Evaluate new model against previous best
            if (config_.enable_evaluation && i > 0) {
                std::cout << "Starting evaluation phase for iteration " << i + 1 << std::endl;
                bool new_model_is_better = evaluateNewModel(iteration_dir);
                
                if (new_model_is_better) {
                    std::cout << "New model from iteration " << i + 1 << " is better than previous best" << std::endl;
                    updateBestModel();
                } else {
                    std::cout << "Previous best model remains champion" << std::endl;
                }
            } else {
                // For the first iteration or if evaluation is disabled, always update best model
                updateBestModel();
            }
            
            // Log iteration summary
            logIterationSummary(i, games.size(), train_loss);
        }
        
        std::cout << "AlphaZero pipeline completed successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in AlphaZero pipeline: " << e.what() << std::endl;
        throw;
    }
}

void AlphaZeroPipeline::initializeLogging() {
    // Create log directory if it doesn't exist
    std::filesystem::create_directories(config_.log_dir);
    
    // Get timestamp for log file
    std::string timestamp = getTimestampString();
    std::string log_file = config_.log_dir + "/alphazero_pipeline_" + timestamp + ".log";
    
    // Note: Actual logging implementation would use a logging library like spdlog
    std::cout << "Logging initialized. Log file: " << log_file << std::endl;
}

void AlphaZeroPipeline::createDirectories() {
    // Create model directory
    std::filesystem::create_directories(config_.model_dir);
    
    // Create data directory
    std::filesystem::create_directories(config_.data_dir);
    
    // Create log directory
    std::filesystem::create_directories(config_.log_dir);
    
    std::cout << "Created directory structure: models, data, logs" << std::endl;
}

void AlphaZeroPipeline::initializeNeuralNetwork() {
    // Check if there's an existing best model to load
    std::string best_model_path = config_.model_dir + "/best_model.pt";
    
    if (std::filesystem::exists(best_model_path)) {
        std::cout << "Loading existing best model from: " << best_model_path << std::endl;
        try {
            if (config_.network_type == "resnet") {
                current_model_ = std::make_shared<nn::ResNetModel>(
                    config_.input_channels, 
                    config_.board_size,
                    config_.num_res_blocks,
                    config_.num_filters,
                    config_.policy_size
                );
                
                current_model_->load(best_model_path);
            } 
            else if (config_.network_type == "ddw_randwire") {
                auto model = std::make_shared<nn::DDWRandWireResNet>(
                    config_.input_channels,
                    config_.policy_size,
                    config_.num_filters,
                    config_.num_res_blocks
                );
                
                model->load(best_model_path);
                current_model_ = std::static_pointer_cast<nn::NeuralNetwork>(model);
            }
            else {
                throw std::runtime_error("Unknown network type: " + config_.network_type);
            }
            std::cout << "Model loaded successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to load existing model: " << e.what() << std::endl;
            std::cout << "Creating new model instead" << std::endl;
            initializeNewNeuralNetwork();
        }
    } 
    else {
        std::cout << "No existing model found, creating new model" << std::endl;
        initializeNewNeuralNetwork();
    }
}

void AlphaZeroPipeline::initializeNewNeuralNetwork() {
    try {
        if (config_.network_type == "resnet") {
            current_model_ = std::make_shared<nn::ResNetModel>(
                config_.input_channels, 
                config_.board_size,
                config_.num_res_blocks,
                config_.num_filters,
                config_.policy_size
            );
            std::cout << "Created new ResNet model" << std::endl;
        } 
        else if (config_.network_type == "ddw_randwire") {
            auto model = std::make_shared<nn::DDWRandWireResNet>(
                config_.input_channels,
                config_.policy_size,
                config_.num_filters,
                config_.num_res_blocks
            );
            current_model_ = std::static_pointer_cast<nn::NeuralNetwork>(model);
            std::cout << "Created new DDW-RandWire-ResNet model" << std::endl;
        }
        else {
            throw std::runtime_error("Unknown network type: " + config_.network_type);
        }
        
        // Save the initial model
        saveBestModel();
        std::cout << "Initial model saved as best model" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create neural network: " << e.what() << std::endl;
        throw;
    }
}

std::string AlphaZeroPipeline::createIterationDirectory(int iteration) {
    // Create timestamped directory for this iteration
    std::string timestamp = getTimestampString();
    std::string iter_dir = config_.data_dir + "/iteration_" + 
                           std::to_string(iteration + 1) + "_" + timestamp;
    
    // Create directory structure
    std::filesystem::create_directories(iter_dir + "/selfplay");
    std::filesystem::create_directories(iter_dir + "/training");
    std::filesystem::create_directories(iter_dir + "/evaluation");
    
    return iter_dir;
}

std::vector<selfplay::GameData> AlphaZeroPipeline::runSelfPlay(const std::string& iteration_dir) {
    try {
        // Setup self-play settings
        selfplay::SelfPlaySettings settings;
        settings.mcts_settings.num_simulations = config_.mcts_num_simulations;
        settings.mcts_settings.num_threads = config_.mcts_num_threads;
        settings.mcts_settings.exploration_constant = config_.mcts_exploration_constant;
        settings.mcts_settings.temperature = config_.mcts_temperature;
        settings.mcts_settings.add_dirichlet_noise = config_.mcts_add_dirichlet_noise;
        settings.mcts_settings.dirichlet_alpha = config_.mcts_dirichlet_alpha;
        settings.mcts_settings.dirichlet_epsilon = config_.mcts_dirichlet_epsilon;
        settings.mcts_settings.batch_size = config_.mcts_batch_size;
        settings.mcts_settings.max_collection_batch_size = config_.mcts_max_collection_batch_size;
        settings.mcts_settings.batch_timeout = std::chrono::milliseconds(config_.mcts_batch_timeout_ms);
        
        settings.num_parallel_games = config_.self_play_num_parallel_games;
        
        // Configure root parallelization instead of multiple engines
        settings.mcts_settings.use_root_parallelization = true;
        
        // Calculate reasonable number of root workers based on threads and parallel games
        int available_cores = std::thread::hardware_concurrency();
        int cores_per_game = std::max(1, available_cores / config_.self_play_num_parallel_games);
        settings.mcts_settings.num_root_workers = cores_per_game;
        settings.max_moves = config_.self_play_max_moves;
        settings.temperature_threshold = config_.self_play_temperature_threshold;
        settings.high_temperature = config_.self_play_high_temperature;
        settings.low_temperature = config_.self_play_low_temperature;
        settings.add_dirichlet_noise = config_.mcts_add_dirichlet_noise;
        
        // Create self-play manager with current best model
        selfplay::SelfPlayManager self_play_manager(current_model_, settings);
        
        // Generate games
        std::cout << "Generating " << config_.self_play_num_games << " self-play games..." << std::endl;
        std::vector<selfplay::GameData> games = self_play_manager.generateGames(
            config_.game_type, 
            config_.self_play_num_games,
            config_.board_size
        );
        
        // Save games to iteration directory
        std::string self_play_dir = iteration_dir + "/selfplay";
        self_play_manager.saveGames(games, self_play_dir, config_.self_play_output_format);
        
        return games;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during self-play: " << e.what() << std::endl;
        throw;
    }
}

float AlphaZeroPipeline::trainNeuralNetwork(const std::vector<selfplay::GameData>& games, const std::string& iteration_dir) {
    try {
        // Print CUDA memory info at start
        if (torch::cuda::is_available() && config_.use_gpu) {
            std::cout << "Training start - CUDA memory info:" << std::endl;
            for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                size_t free_memory_bytes = 0;
                size_t total_memory_bytes = 0;
                cudaError_t cuda_status = cudaSetDevice(dev);
                if (cuda_status != cudaSuccess) {
                    std::cerr << "Error setting CUDA device " << dev << ": " << cudaGetErrorString(cuda_status) << std::endl;
                    continue;
                }
                cuda_status = cudaMemGetInfo(&free_memory_bytes, &total_memory_bytes);
                if (cuda_status != cudaSuccess) {
                    std::cerr << "Error getting CUDA memory info for device " << dev << ": " << cudaGetErrorString(cuda_status) << std::endl;
                    continue;
                }

                std::cout << "GPU " << dev << " Memory: Free = " << free_memory_bytes / (1024 * 1024) << " MB, Total = " << total_memory_bytes / (1024*1024) << " MB" << std::endl;
            }
        }
    
        // Convert games to training examples
        std::cout << "Converting games to training examples..." << std::endl;
        auto [states, targets] = selfplay::SelfPlayManager::convertToTrainingExamples(games);
        auto [policies, values] = targets;
        
        std::cout << "Created " << states.size() << " training examples" << std::endl;
        
        // Determine dataset device based on available memory
        torch::Device dataset_device = torch::kCPU;
        
        // Only attempt GPU dataset if use_gpu is enabled
        if (config_.use_gpu && torch::cuda::is_available()) {
            try {
                // Calculate approximate size of training data
                size_t num_examples = states.size();
                size_t channels = states[0].size();
                size_t height = states[0][0].size();
                size_t width = states[0][0][0].size();
                size_t policy_size = policies[0].size();
                
                // Calculate approximate memory usage
                size_t states_elements = num_examples * channels * height * width;
                size_t policies_elements = num_examples * policy_size;
                size_t values_elements = num_examples;
                size_t total_elements = states_elements + policies_elements + values_elements;
                size_t approx_bytes = total_elements * 4; // float32 = 4 bytes
                
                std::cout << "Estimated dataset memory: " << approx_bytes << " bytes" << std::endl;
                std::cout << "CUDA memory allocated: " << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).allocated_bytes[static_cast<size_t>(0)].current
                          << ", reserved: " << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).reserved_bytes[static_cast<size_t>(0)].current << std::endl;
                    
                // Check if we have enough memory headroom
                for (int device_idx = 0; device_idx < torch::cuda::device_count(); device_idx++) {
                    const auto* props = at::cuda::getDeviceProperties(device_idx);
                    size_t total_memory = props->totalGlobalMem;
                    size_t memory_threshold = 0.8 * total_memory; // 80% threshold for safety
                    
                    std::cout << "CUDA device " << device_idx << " total memory: " << total_memory << " bytes" << std::endl;
                    
                    if (approx_bytes < memory_threshold) {
                        dataset_device = torch::Device(torch::kCUDA, device_idx);
                        std::cout << "Selected CUDA device " << device_idx << " for dataset" << std::endl;
                        break;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error checking GPU memory: " << e.what() << std::endl;
                std::cerr << "Defaulting to CPU for dataset" << std::endl;
                dataset_device = torch::kCPU;
            }
        }
        
        std::cout << "Creating dataset on device: " << dataset_device << std::endl;
        
        // Create dataset
        auto dataset = std::make_shared<training::AlphaZeroDataset>(
            states, policies, values, dataset_device
        );
        
        // Create data loader
        training::DataLoader data_loader(
            dataset,
            config_.train_batch_size,
            true,  // shuffle
            config_.train_num_workers,
            dataset->device().is_cpu(),  // only pin memory if dataset is on CPU
            false  // don't drop last
        );
        
        // Determine device (use CUDA if available)
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && config_.use_gpu) {
            device = torch::kCUDA;
            std::cout << "Using CUDA device for training" << std::endl;
            // Print CUDA device properties for debugging
            std::cout << "CUDA Device count: " << torch::cuda::device_count() << std::endl;
            std::cout << "CUDA Device name: " << at::cuda::getDeviceProperties(c10::cuda::current_device())->name << std::endl;
            std::cout << "CUDA Device memory: " << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).allocated_bytes[static_cast<size_t>(0)].current << " bytes allocated, "
                      << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).reserved_bytes[static_cast<size_t>(0)].current << " bytes reserved" << std::endl;
        } else {
            std::cout << "Using CPU for training (CUDA not available or disabled)" << std::endl;
            if (torch::cuda::is_available()) {
                std::cout << "Note: CUDA is available but disabled in config" << std::endl;
            } else {
                std::cout << "Note: CUDA is not available on this system" << std::endl;
            }
        }
        
        // Create and initialize training model based on network type
        torch::nn::Module* training_model = nullptr;
        
        if (config_.network_type == "resnet") {
            auto* model = dynamic_cast<nn::ResNetModel*>(current_model_.get());
            if (!model) {
                throw std::runtime_error("Failed to cast to ResNetModel");
            }
            training_model = model;
        }
        else if (config_.network_type == "ddw_randwire") {
            // Extract the DDWRandWireResNet from the current model
            // This is a simplification - actual implementation would need to handle this properly
            std::cerr << "DDW-RandWire-ResNet training not fully implemented" << std::endl;
            throw std::runtime_error("DDW-RandWire-ResNet training not yet supported");
        }
        else {
            throw std::runtime_error("Unknown network type: " + config_.network_type);
        }
        
        if (!training_model) {
            throw std::runtime_error("Failed to initialize training model");
        }
        
        // Move model to device with error handling
        std::cout << "Moving model to device: " << device << std::endl;
        try {
            // Check current model device
            torch::Device model_device = torch::kCPU;
            for (const auto& param : training_model->parameters()) {
                model_device = param.device();
                break;
            }
            std::cout << "Current model device: " << model_device << std::endl;
            
            // Move model to target device
            training_model->to(device);
            
            // Verify model device after movement
            bool model_on_correct_device = true;
            for (const auto& param : training_model->parameters()) {
                if (param.device() != device) {
                    model_on_correct_device = false;
                    std::cerr << "ERROR: Model parameter still on " << param.device()
                              << " instead of " << device << std::endl;
                }
                break; // Just check the first parameter
            }
            
            if (model_on_correct_device) {
                std::cout << "Model successfully moved to " << device << std::endl;
            } else {
                std::cerr << "Failed to directly move model parameters to " << device << std::endl;
                std::cerr << "Trying alternative approach to move model to GPU..." << std::endl;
                
                // Alternative approach - copy state dict and recreate model on correct device
                if (device.is_cuda()) {
                    try {
                        // Save model parameters
                        std::vector<torch::Tensor> params;
                        for (const auto& param : training_model->parameters()) {
                            params.push_back(param.clone());
                        }
                        
                        // Recreate model directly on GPU
                        if (config_.network_type == "resnet") {
                            auto* model = dynamic_cast<nn::ResNetModel*>(training_model);
                            if (model) {
                                std::cout << "Recreating ResNet model directly on GPU" << std::endl;
                                auto new_model = std::make_shared<nn::ResNetModel>(
                                    config_.input_channels, 
                                    config_.board_size,
                                    config_.num_res_blocks,
                                    config_.num_filters,
                                    config_.policy_size
                                );
                                
                                // Move to GPU first
                                new_model->to(device);
                                
                                // Copy parameters without in-place operations
                                auto new_params = new_model->parameters();
                                if (params.size() == new_params.size()) {
                                    for (size_t i = 0; i < params.size(); ++i) {
                                        // Clone parameter to avoid in-place operations on parameters that require grad
                                        torch::Tensor device_param = params[i].to(device).detach().clone();
                                        new_params[i].data().copy_(device_param);
                                    }
                                    
                                    // Replace model
                                    training_model = new_model.get();
                                    current_model_ = std::move(new_model);
                                    std::cout << "Successfully recreated model on GPU" << std::endl;
                                } else {
                                    std::cerr << "Parameter count mismatch during model recreation" << std::endl;
                                    device = torch::kCPU;
                                    training_model->to(device);
                                }
                            } else {
                                std::cerr << "Failed to cast model during recreation" << std::endl;
                                device = torch::kCPU;
                                training_model->to(device);
                            }
                        } else {
                            std::cerr << "Alternative GPU approach only implemented for ResNet" << std::endl;
                            device = torch::kCPU;
                            training_model->to(device);
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error during alternative GPU approach: " << e.what() << std::endl;
                        device = torch::kCPU;
                        training_model->to(device);
                    }
                } else {
                    // Not attempting to move to GPU, just use CPU directly
                    device = torch::kCPU;
                    training_model->to(device);
                }
            }
        } catch (const torch::Error& e) {
            std::cerr << "PyTorch error moving model to device: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU" << std::endl;
            device = torch::kCPU;
            try {
                training_model->to(device);
            } catch (...) {
                std::cerr << "Critical error moving model to CPU. This should not happen." << std::endl;
                throw;
            }
        } catch (const std::exception& e) {
            std::cerr << "Standard error moving model to device: " << e.what() << std::endl;
            throw;
        }
        
        // Set up optimizer
        torch::optim::Adam optimizer(
            training_model->parameters(),
            torch::optim::AdamOptions(config_.train_learning_rate)
                .weight_decay(config_.train_weight_decay)
        );
        
        // Learning rate scheduler
        torch::optim::StepLR scheduler(
            optimizer,
            config_.train_lr_step_size,
            config_.train_lr_gamma
        );
        
        // Training loop
        std::cout << "Starting training for " << config_.train_epochs << " epochs" << std::endl;
        float best_loss = std::numeric_limits<float>::max();
        float final_loss = 0.0f;
        
        for (int epoch = 0; epoch < config_.train_epochs; ++epoch) {
            // Track metrics
            float epoch_loss = 0.0f;
            float policy_loss_sum = 0.0f;
            float value_loss_sum = 0.0f;
            int batch_count = 0;
            
            // Set model to training mode
            training_model->train();
            
            // Ensure all model parameters require gradients
            for (auto& param : training_model->parameters()) {
                param.set_requires_grad(true);
            }
            
            // Iterate through batches
            for (const auto& batch : data_loader) {
                // Move batch to device with error handling
                
                torch::Tensor states_tensor, policies_tensor, values_tensor;
                
                try {
                    // Move tensors one by one with error checking
                    states_tensor = batch.states.to(device);
                    policies_tensor = batch.policies.to(device);
                    values_tensor = batch.values.to(device);
                } catch (const torch::Error& e) {
                    std::cerr << "PyTorch error moving batch to device: " << e.what() << std::endl;
                    std::cerr << "Attempting emergency fallback to CPU" << std::endl;
                    
                    device = torch::kCPU;
                    training_model->to(device);
                    
                    // Use original tensors if they're already on CPU
                    states_tensor = batch.states;
                    policies_tensor = batch.policies;
                    values_tensor = batch.values;
                } catch (const std::exception& e) {
                    std::cerr << "Standard error moving batch to device: " << e.what() << std::endl;
                    throw;
                }
                
                // Reset gradients
                optimizer.zero_grad();
                
                std::tuple<torch::Tensor, torch::Tensor> output;
                try {
                    if (config_.network_type == "resnet") {
                        auto* resnet_model_ptr = dynamic_cast<nn::ResNetModel*>(training_model);
                        if (!resnet_model_ptr) throw std::runtime_error("Failed to cast training_model to ResNetModel for forward pass");
                        
                        // Double-check that all tensors are on the same device
                        if (states_tensor.device() != device) {
                            states_tensor = states_tensor.to(device);
                        }
                        
                        // Ensure input tensor has requires_grad=true for backpropagation
                        if (!states_tensor.requires_grad()) {
                            states_tensor = states_tensor.detach().requires_grad_(true);
                        }
                        
                        output = resnet_model_ptr->forward(states_tensor);
                    } else {
                        // DDW-RandWire is not fully implemented for training here yet, this would crash.
                        // For now, to make it compile, let's assume a similar forward if we were to implement it.
                        // This part needs proper implementation for DDWRandWireResNet.
                        throw std::runtime_error("DDW-RandWire training forward pass not implemented in this class-based pipeline.");
                        // auto* ddw_model_ptr = dynamic_cast<nn::DDWRandWireResNet*>(training_model);
                        // if (!ddw_model_ptr) throw std::runtime_error("Failed to cast training_model to DDWRandWireResNet for forward pass");
                        // output = ddw_model_ptr->forward(states_tensor); // Assuming DDWRandWireResNet has such a forward
                    }
                } catch (const torch::Error& e) {
                    std::cerr << "PyTorch error during forward pass: " << e.what() << std::endl;
                    throw;
                } catch (const std::exception& e) {
                    std::cerr << "Standard exception during forward pass: " << e.what() << std::endl;
                    throw;
                } catch (...) {
                    std::cerr << "Unknown error during forward pass" << std::endl;
                    throw;
                }

                auto policy_output = std::get<0>(output);
                auto value_output = std::get<1>(output);
                
                // Calculate loss
                // Make sure policy_output and policies_tensor are on the same device
                if (policy_output.device() != policies_tensor.device()) {
                    policy_output = policy_output.to(policies_tensor.device());
                }
                
                // Ensure policy_output has requires_grad
                if (!policy_output.requires_grad()) {
                    policy_output = policy_output.detach().requires_grad_(true);
                }
                
                auto log_softmax_policy = torch::log_softmax(policy_output, 1);
                auto policy_loss = -torch::sum(policies_tensor * log_softmax_policy) / policies_tensor.size(0);
                
                // Value loss: MSE
                // Make sure value_output and values_tensor are on the same device
                if (value_output.device() != values_tensor.device()) {
                    value_output = value_output.to(values_tensor.device());
                }
                
                // Ensure value_output has requires_grad
                if (!value_output.requires_grad()) {
                    value_output = value_output.detach().requires_grad_(true);
                }
                
                // Fix value tensor shape to match exactly
                values_tensor = values_tensor.view(value_output.sizes());
                auto value_loss = torch::nn::functional::mse_loss(value_output, values_tensor);
                
                // Combined loss
                auto loss = policy_loss + value_loss;
                
                // Backward pass and optimize with error handling
                try {
                    loss.backward();
                    optimizer.step();
                } catch (const torch::Error& e) {
                    std::cerr << "PyTorch error during backward/optimize: " << e.what() << std::endl;
                    
                    // Try to recover if possible
                    if (device.is_cuda()) {
                        std::cerr << "Attempting to recover from CUDA error during backward/optimize..." << std::endl;
                        try {
                            c10::cuda::CUDACachingAllocator::emptyCache();
                            std::cerr << "CUDA cache cleared. Skipping problematic batch." << std::endl;
                        } catch (const std::exception& cache_exc) {
                            std::cerr << "Exception while clearing CUDA cache: " << cache_exc.what() << std::endl;
                        }
                        continue; // Skip this batch and continue with the next one on the original 'device'
                    } else {
                        // If already on CPU and still failing, this is a critical error
                        std::cerr << "Error occurred on CPU during backward/optimize. Rethrowing." << std::endl;
                        throw;
                    }
                }
                
                // Update metrics
                epoch_loss += loss.item<float>();
                policy_loss_sum += policy_loss.item<float>();
                value_loss_sum += value_loss.item<float>();
                batch_count++;
                
                // Log progress occasionally
                if (batch_count % 10 == 0) {
                    std::cout << "Epoch " << epoch + 1 << "/" << config_.train_epochs
                              << ", Batch " << batch_count << "/" << data_loader.size()
                              << ", Loss: " << loss.item<float>()
                              << " (Policy: " << policy_loss.item<float>()
                              << ", Value: " << value_loss.item<float>() << ")" << std::endl;
                }
            }
            
            // Step scheduler
            scheduler.step();
            
            // Calculate average loss for the epoch
            epoch_loss /= batch_count;
            policy_loss_sum /= batch_count;
            value_loss_sum /= batch_count;
            
            std::cout << "Epoch " << epoch + 1 << "/" << config_.train_epochs
                      << " completed. Avg Loss: " << epoch_loss
                      << " (Policy: " << policy_loss_sum
                      << ", Value: " << value_loss_sum << ")" << std::endl;
            
            // Print GPU memory stats after each epoch
            if (torch::cuda::is_available()) {
                try {
                    std::cout << "End of epoch " << (epoch + 1) << " - CUDA memory stats:" << std::endl;
                    for (int dev = 0; dev < torch::cuda::device_count(); dev++) {
                        std::cout << "  Device " << dev << ": " 
                                << "allocated=" << c10::cuda::CUDACachingAllocator::getDeviceStats(dev).allocated_bytes[static_cast<size_t>(0)].current << " bytes, "
                                << "reserved=" << c10::cuda::CUDACachingAllocator::getDeviceStats(dev).reserved_bytes[static_cast<size_t>(0)].current << " bytes" << std::endl;
                    }
                    
                    // Force CUDA memory cleanup after each epoch to prevent memory buildup
                    if (c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).reserved_bytes[static_cast<size_t>(0)].current > 0) {
                        std::cout << "Running CUDA garbage collection..." << std::endl;
                        c10::cuda::CUDACachingAllocator::emptyCache();
                        std::cout << "After GC: allocated=" << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).allocated_bytes[static_cast<size_t>(0)].current
                                << ", reserved=" << c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).reserved_bytes[static_cast<size_t>(0)].current << std::endl;
                    }
                } catch (...) {
                    std::cout << "Error getting CUDA memory stats at end of epoch" << std::endl;
                }
            }
            
            // Save model if loss improved
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                std::string model_path = iteration_dir + "/training/best_epoch_model.pt";
                torch::save(training_model->parameters(), model_path);
                std::cout << "Saved best epoch model with loss: " << best_loss << std::endl;
            }
            
            // Save final epoch result
            final_loss = epoch_loss;
        }
        
        // Save final model for this iteration
        std::string latest_model_path = config_.model_dir + "/latest_model.pt";
        torch::save(training_model->parameters(), latest_model_path);
        std::cout << "Saved latest model to: " << latest_model_path << std::endl;
        
        // Also save a copy in the iteration directory
        std::string iteration_model_path = iteration_dir + "/training/final_model.pt";
        torch::save(training_model->parameters(), iteration_model_path);
        
        return final_loss;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        throw;
    }
}

bool AlphaZeroPipeline::evaluateNewModel(const std::string& iteration_dir) {
    try {
        // Load latest model as contender
        std::shared_ptr<nn::NeuralNetwork> contender_model;
        std::string latest_model_path = config_.model_dir + "/latest_model.pt";
        
        if (config_.network_type == "resnet") {
            contender_model = std::make_shared<nn::ResNetModel>(
                config_.input_channels, 
                config_.board_size,
                config_.num_res_blocks,
                config_.num_filters,
                config_.policy_size
            );
            contender_model->load(latest_model_path);
        } 
        else if (config_.network_type == "ddw_randwire") {
            auto model = std::make_shared<nn::DDWRandWireResNet>(
                config_.input_channels,
                config_.policy_size,
                config_.num_filters,
                config_.num_res_blocks
            );
            model->load(latest_model_path);
            contender_model = std::static_pointer_cast<nn::NeuralNetwork>(model);
        }
        else {
            throw std::runtime_error("Unknown network type: " + config_.network_type);
        }
        
        // Load best model as champion
        std::shared_ptr<nn::NeuralNetwork> champion_model;
        std::string best_model_path = config_.model_dir + "/best_model.pt";
        
        if (config_.network_type == "resnet") {
            champion_model = std::make_shared<nn::ResNetModel>(
                config_.input_channels, 
                config_.board_size,
                config_.num_res_blocks,
                config_.num_filters,
                config_.policy_size
            );
            champion_model->load(best_model_path);
        } 
        else if (config_.network_type == "ddw_randwire") {
            auto model = std::make_shared<nn::DDWRandWireResNet>(
                config_.input_channels,
                config_.policy_size,
                config_.num_filters,
                config_.num_res_blocks
            );
            model->load(best_model_path);
            champion_model = std::static_pointer_cast<nn::NeuralNetwork>(model);
        }
        else {
            throw std::runtime_error("Unknown network type: " + config_.network_type);
        }
        
        // Set up arena settings
        selfplay::SelfPlaySettings arena_settings;
        arena_settings.mcts_settings.num_simulations = config_.arena_num_simulations;
        arena_settings.mcts_settings.num_threads = config_.arena_num_threads;
        arena_settings.mcts_settings.exploration_constant = config_.mcts_exploration_constant;
        arena_settings.mcts_settings.temperature = config_.arena_temperature;
        arena_settings.mcts_settings.add_dirichlet_noise = false; // No noise in arena games
        arena_settings.mcts_settings.batch_size = config_.mcts_batch_size;
        arena_settings.mcts_settings.batch_timeout = std::chrono::milliseconds(config_.mcts_batch_timeout_ms);
        
        arena_settings.num_parallel_games = config_.arena_num_parallel_games;
        arena_settings.num_mcts_engines = config_.arena_num_mcts_engines;
        arena_settings.max_moves = config_.self_play_max_moves; // Same as self-play
        arena_settings.temperature_threshold = 0; // No high temperature period
        arena_settings.high_temperature = 0.0f;
        arena_settings.low_temperature = config_.arena_temperature;
        arena_settings.add_dirichlet_noise = false;
        
        // Create self-play managers for both models
        selfplay::SelfPlayManager champion_manager(champion_model, arena_settings);
        selfplay::SelfPlayManager contender_manager(contender_model, arena_settings);
        
        // Play half the games with each model as first player
        int num_games = config_.arena_num_games;
        int half_games = num_games / 2;
        int champion_wins = 0;
        int contender_wins = 0;
        int draws = 0;
        
        // Champion as first player
        std::cout << "Playing " << half_games << " arena games with champion as first player..." << std::endl;
        std::vector<selfplay::GameData> first_half = playArenaGames(
            champion_manager, contender_manager, half_games, iteration_dir + "/evaluation/champion_first"
        );
        
        // Contender as first player
        std::cout << "Playing " << num_games - half_games << " arena games with contender as first player..." << std::endl;
        std::vector<selfplay::GameData> second_half = playArenaGames(
            contender_manager, champion_manager, num_games - half_games, iteration_dir + "/evaluation/contender_first"
        );
        
        // Combine and count results
        std::vector<selfplay::GameData> all_games;
        all_games.insert(all_games.end(), first_half.begin(), first_half.end());
        all_games.insert(all_games.end(), second_half.begin(), second_half.end());
        
        for (const auto& game : first_half) {
            if (game.winner == 1) champion_wins++;
            else if (game.winner == 2) contender_wins++;
            else draws++;
        }
        
        for (const auto& game : second_half) {
            if (game.winner == 1) contender_wins++;
            else if (game.winner == 2) champion_wins++;
            else draws++;
        }
        
        // Calculate win rate for the contender
        float contender_win_rate = static_cast<float>(contender_wins) / all_games.size();
        float draw_rate = static_cast<float>(draws) / all_games.size();
        
        std::cout << "Arena evaluation complete. Results:" << std::endl;
        std::cout << "  Champion wins: " << champion_wins << " (" 
                 << 100.0f * champion_wins / all_games.size() << "%)" << std::endl;
        std::cout << "  Contender wins: " << contender_wins << " (" 
                 << 100.0f * contender_win_rate << "%)" << std::endl;
        std::cout << "  Draws: " << draws << " (" 
                 << 100.0f * draw_rate << "%)" << std::endl;
        
        // Save evaluation results
        saveEvaluationResults(
            iteration_dir + "/evaluation/results.json",
            champion_wins,
            contender_wins,
            draws,
            all_games.size()
        );
        
        // Return true if contender is better (based on win rate threshold)
        return contender_win_rate > config_.arena_win_rate_threshold;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during evaluation: " << e.what() << std::endl;
        throw;
    }
}

std::vector<selfplay::GameData> AlphaZeroPipeline::playArenaGames(
    selfplay::SelfPlayManager& player1_manager,
    selfplay::SelfPlayManager& player2_manager,
    int num_games,
    const std::string& output_dir) {
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // In a real implementation, we would need to carefully orchestrate the MCTS engines
    // to use the appropriate neural networks at the right times. This is a simplified version.
    
    // For now, we'll just use player1_manager to play the games
    // In reality, we'd need to alternate between models during the game
    std::vector<selfplay::GameData> games = player1_manager.generateGames(
        config_.game_type, num_games, config_.board_size
    );
    
    // Save games
    player1_manager.saveGames(games, output_dir, "json");
    
    return games;
}

void AlphaZeroPipeline::saveEvaluationResults(
    const std::string& file_path,
    int champion_wins,
    int contender_wins,
    int draws,
    int total_games) {
    
    try {
        // Create JSON with results
        nlohmann::json results;
        results["timestamp"] = getTimestampString();
        results["iteration"] = current_iteration_ + 1;
        results["champion_wins"] = champion_wins;
        results["contender_wins"] = contender_wins;
        results["draws"] = draws;
        results["total_games"] = total_games;
        results["champion_win_rate"] = static_cast<float>(champion_wins) / total_games;
        results["contender_win_rate"] = static_cast<float>(contender_wins) / total_games;
        results["draw_rate"] = static_cast<float>(draws) / total_games;
        
        // Save to file
        std::ofstream file(file_path);
        if (file) {
            file << results.dump(2) << std::endl;
            std::cout << "Saved evaluation results to: " << file_path << std::endl;
        } else {
            std::cerr << "Failed to save evaluation results to: " << file_path << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving evaluation results: " << e.what() << std::endl;
    }
}

void AlphaZeroPipeline::updateBestModel() {
    try {
        std::string latest_model_path = config_.model_dir + "/latest_model.pt";
        std::string best_model_path = config_.model_dir + "/best_model.pt";
        
        // Copy latest model to best model
        std::cout << "Updating best model from: " << latest_model_path << " to: " << best_model_path << std::endl;
        
        // Load latest model
        if (config_.network_type == "resnet") {
            auto* model = dynamic_cast<nn::ResNetModel*>(current_model_.get());
            if (!model) {
                throw std::runtime_error("Failed to cast to ResNetModel");
            }
            model->load(latest_model_path);
        } 
        else if (config_.network_type == "ddw_randwire") {
            auto* model = dynamic_cast<nn::DDWRandWireResNet*>(current_model_.get());
            if (!model) {
                throw std::runtime_error("Failed to get DDWRandWireResNet model from current_model_");
            }
            
            model->load(latest_model_path);
        }
        else {
            throw std::runtime_error("Unknown network type: " + config_.network_type);
        }
        
        // Save as best model
        saveBestModel();
        std::cout << "Best model updated successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error updating best model: " << e.what() << std::endl;
        throw;
    }
}

void AlphaZeroPipeline::saveBestModel() {
    try {
        std::string best_model_path = config_.model_dir + "/best_model.pt";
        current_model_->save(best_model_path);
        std::cout << "Saved best model to: " << best_model_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving best model: " << e.what() << std::endl;
        throw;
    }
}

void AlphaZeroPipeline::logIterationSummary(int iteration, int num_games, float train_loss) {
    try {
        std::string summary_path = config_.log_dir + "/iteration_summary.csv";
        bool file_exists = std::filesystem::exists(summary_path);
        
        std::ofstream file(summary_path, std::ios::app);
        if (!file) {
            std::cerr << "Failed to open summary file: " << summary_path << std::endl;
            return;
        }
        
        // Write header if new file
        if (!file_exists) {
            file << "Iteration,Timestamp,NumGames,TrainLoss" << std::endl;
        }
        
        // Write data
        file << (iteration + 1) << ","
             << getTimestampString() << ","
             << num_games << ","
             << train_loss << std::endl;
             
        std::cout << "Logged iteration summary to: " << summary_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error logging iteration summary: " << e.what() << std::endl;
    }
}

AlphaZeroPipelineConfig parseConfigFile(const std::string& config_path) {
    AlphaZeroPipelineConfig config;
    
    // Load YAML file
    try {
        std::ifstream file(config_path);
        if (!file) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }
        
        // Use YAML-CPP to parse the file
        YAML::Node yaml = YAML::LoadFile(config_path);
        
        // Parse game settings
        if (yaml["game_type"]) {
            std::string game_str = yaml["game_type"].as<std::string>();
            if (game_str == "gomoku") {
                config.game_type = core::GameType::GOMOKU;
            } else if (game_str == "chess") {
                config.game_type = core::GameType::CHESS;
            } else if (game_str == "go") {
                config.game_type = core::GameType::GO;
            } else {
                throw std::runtime_error("Unknown game type: " + game_str);
            }
        }
        
        if (yaml["board_size"]) {
            config.board_size = yaml["board_size"].as<int>();
        }
        
        // Parse directory settings
        if (yaml["model_dir"]) {
            config.model_dir = yaml["model_dir"].as<std::string>();
        }
        
        if (yaml["data_dir"]) {
            config.data_dir = yaml["data_dir"].as<std::string>();
        }
        
        if (yaml["log_dir"]) {
            config.log_dir = yaml["log_dir"].as<std::string>();
        }
        
        // Parse neural network settings
        // First try the flat structure (as in config.yaml)
        if (yaml["num_res_blocks"]) {
            config.num_res_blocks = yaml["num_res_blocks"].as<int>();
        }
        
        if (yaml["num_filters"]) {
            config.num_filters = yaml["num_filters"].as<int>();
        }
        
        if (yaml["input_channels"]) {
            config.input_channels = yaml["input_channels"].as<int>();
        }
        
        // Also support nested structure for backward compatibility
        if (yaml["network"]) {
            auto network = yaml["network"];
            
            if (network["num_res_blocks"]) {
                config.num_res_blocks = network["num_res_blocks"].as<int>();
            }
            
            if (network["num_filters"]) {
                config.num_filters = network["num_filters"].as<int>();
            }
            
            if (network["input_channels"]) {
                config.input_channels = network["input_channels"].as<int>();
            }
        }
        
        // Parse training settings
        if (yaml["train"]) {
            auto train = yaml["train"];
            
            if (train["iterations"]) {
                config.num_iterations = train["iterations"].as<int>();
            }
            
            if (train["epochs"]) {
                config.train_epochs = train["epochs"].as<int>();
            }
            
            if (train["batch_size"]) {
                config.train_batch_size = train["batch_size"].as<int>();
            }
            
            if (train["learning_rate"]) {
                config.train_learning_rate = train["learning_rate"].as<float>();
            }
            
            if (train["weight_decay"]) {
                config.train_weight_decay = train["weight_decay"].as<float>();
            }
        }
        
        // Parse MCTS settings
        if (yaml["mcts"]) {
            auto mcts = yaml["mcts"];
            
            if (mcts["num_simulations"]) {
                config.mcts_num_simulations = mcts["num_simulations"].as<int>();
            }
            
            if (mcts["num_threads"]) {
                config.mcts_num_threads = mcts["num_threads"].as<int>();
            }
            
            if (mcts["batch_size"]) {
                config.mcts_batch_size = mcts["batch_size"].as<int>();
            }
            
            if (mcts["max_collection_batch_size"]) {
                config.mcts_max_collection_batch_size = mcts["max_collection_batch_size"].as<int>();
            }
            
            if (mcts["exploration_constant"]) {
                config.mcts_exploration_constant = mcts["exploration_constant"].as<float>();
            }
            
            if (mcts["add_dirichlet_noise"]) {
                config.mcts_add_dirichlet_noise = mcts["add_dirichlet_noise"].as<bool>();
            }
            
            if (mcts["dirichlet_alpha"]) {
                config.mcts_dirichlet_alpha = mcts["dirichlet_alpha"].as<float>();
            }
            
            if (mcts["dirichlet_epsilon"]) {
                config.mcts_dirichlet_epsilon = mcts["dirichlet_epsilon"].as<float>();
            }
            
            if (mcts["temperature"]) {
                config.mcts_temperature = mcts["temperature"].as<float>();
            }
            
            if (mcts["batch_timeout_ms"]) {
                config.mcts_batch_timeout_ms = mcts["batch_timeout_ms"].as<int>();
            }
        }
        
        // Parse self-play settings
        if (yaml["self_play"]) {
            auto self_play = yaml["self_play"];
            
            if (self_play["num_games"]) {
                config.self_play_num_games = self_play["num_games"].as<int>();
            }
            
            if (self_play["num_parallel_games"]) {
                config.self_play_num_parallel_games = self_play["num_parallel_games"].as<int>();
            }
            
            if (self_play["num_mcts_engines"]) {
                config.self_play_num_mcts_engines = self_play["num_mcts_engines"].as<int>();
            } else {
                // Default to same as parallel games if not specified
                config.self_play_num_mcts_engines = config.self_play_num_parallel_games;
            }
            
            if (self_play["max_moves"]) {
                config.self_play_max_moves = self_play["max_moves"].as<int>();
            }
            
            if (self_play["temperature_threshold"]) {
                config.self_play_temperature_threshold = self_play["temperature_threshold"].as<int>();
            }
            
            if (self_play["high_temperature"]) {
                config.self_play_high_temperature = self_play["high_temperature"].as<float>();
            }
            
            if (self_play["low_temperature"]) {
                config.self_play_low_temperature = self_play["low_temperature"].as<float>();
            }
            
            if (self_play["output_format"]) {
                config.self_play_output_format = self_play["output_format"].as<std::string>();
            }
        }
        
        // Parse evaluation settings
        if (yaml["evaluation"]) {
            auto eval = yaml["evaluation"];
            
            if (eval["num_games"]) {
                config.arena_num_games = eval["num_games"].as<int>();
            }
            
            if (eval["num_parallel_games"]) {
                config.arena_num_parallel_games = eval["num_parallel_games"].as<int>();
            }
            
            if (eval["num_mcts_engines"]) {
                config.arena_num_mcts_engines = eval["num_mcts_engines"].as<int>();
            } else {
                // Default to same as parallel games if not specified
                config.arena_num_mcts_engines = config.arena_num_parallel_games;
            }
            
            if (eval["elo_threshold"]) {
                float elo_threshold = eval["elo_threshold"].as<float>();
                // Convert ELO threshold to win rate threshold using the formula:
                // win_rate = 1 / (1 + 10^(-elo/400))
                config.arena_win_rate_threshold = 1.0f / (1.0f + std::pow(10.0f, -elo_threshold / 400.0f));
            }
        }
        
        // Parse arena settings
        if (yaml["arena"]) {
            auto arena = yaml["arena"];
            
            if (arena["enabled"]) {
                config.enable_evaluation = arena["enabled"].as<bool>();
            }
            
            if (arena["num_games"]) {
                config.arena_num_games = arena["num_games"].as<int>();
            }
            
            if (arena["num_parallel_games"]) {
                config.arena_num_parallel_games = arena["num_parallel_games"].as<int>();
            }
            
            if (arena["num_mcts_engines"]) {
                config.arena_num_mcts_engines = arena["num_mcts_engines"].as<int>();
            } else {
                // Default to same as parallel games if not specified
                config.arena_num_mcts_engines = config.arena_num_parallel_games;
            }
            
            if (arena["num_threads"]) {
                config.arena_num_threads = arena["num_threads"].as<int>();
            }
        }
        
        return config;
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
    }
}

int runAlphaZeroPipelineFromConfig(const std::string& config_path) {
    try {
        // Parse config file
        AlphaZeroPipelineConfig config = parseConfigFile(config_path);
        
        // Create and run pipeline
        AlphaZeroPipeline pipeline(config);
        pipeline.run();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error running AlphaZero pipeline: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace cli
} // namespace alphazero