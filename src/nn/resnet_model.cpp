// src/nn/resnet_model.cpp
#include "nn/resnet_model.h"
#include "nn/gpu_optimizer.h"
#include "utils/memory_tracker.h"
// #include "mcts/aggressive_memory_manager.h" // Removed

// Define empty macros for removed memory tracking
#define TRACK_MEMORY_ALLOC(tag, size) ((void)0)
#define TRACK_MEMORY_FREE(tag, size) ((void)0)

#include "utils/attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono> // For timing
#include <iostream> // For logging
#include <omp.h> // For OpenMP parallelization
#include <cstring> // For memcpy

namespace alphazero {
namespace nn {

// Memory tracking macros are defined globally in aggressive_memory_manager.h

ResNetResidualBlock::ResNetResidualBlock(int64_t channels) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                             .padding(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                             .padding(1).bias(false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
}

torch::Tensor ResNetResidualBlock::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    x = torch::relu(x + residual);
    return x;
}

ResNetModel::ResNetModel(int64_t input_channels, int64_t board_size,
                       int64_t num_res_blocks, int64_t num_filters,
                       int64_t policy_size)
    : input_channels_(input_channels),
      board_size_(board_size),
      policy_size_(policy_size) {
    
    if (policy_size_ == 0) {
        policy_size_ = board_size_ * board_size_;
    }
    
    // Input layers
    input_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels_, num_filters, 3)
                                  .padding(1).bias(false));
    input_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters));
    
    register_module("input_conv", input_conv_);
    register_module("input_bn", input_bn_);
    
    // Residual blocks
    res_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < num_res_blocks; ++i) {
        res_blocks_->push_back(std::make_shared<ResNetResidualBlock>(num_filters));
        // No need to register individual blocks if ModuleList itself is registered and holds them.
        // However, explicit registration is safer for direct access by name if needed.
        // register_module("res_block_" + std::to_string(i), res_blocks_[i]); 
    }
    register_module("res_blocks", res_blocks_); // Register the ModuleList
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    // Ensure board_size_ is positive to avoid negative dimensions
    int64_t policy_fc_input_features = 32 * (board_size_ > 0 ? board_size_ : 1) * (board_size_ > 0 ? board_size_ : 1);
    policy_fc_ = torch::nn::Linear(policy_fc_input_features, policy_size_);
    
    register_module("policy_conv", policy_conv_);
    register_module("policy_bn", policy_bn_);
    register_module("policy_fc", policy_fc_);
    
    // Value head
    value_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1).bias(false));
    value_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    int64_t value_fc1_input_features = 32 * (board_size_ > 0 ? board_size_ : 1) * (board_size_ > 0 ? board_size_ : 1);
    value_fc1_ = torch::nn::Linear(value_fc1_input_features, 256);
    value_fc2_ = torch::nn::Linear(256, 1);
    
    register_module("value_conv", value_conv_);
    register_module("value_bn", value_bn_);
    register_module("value_fc1", value_fc1_);
    register_module("value_fc2", value_fc2_);
}

std::tuple<torch::Tensor, torch::Tensor> ResNetModel::forward(torch::Tensor x) {
    auto forward_total_start_time = std::chrono::high_resolution_clock::now();
    // std::cout << "[FWD] Entered. Input x device: " << x.device() << ", shape: " << x.sizes() << std::endl;

    try {
        torch::Device model_device = (!this->parameters().empty() && this->parameters().front().defined()) ? this->parameters().front().device() : torch::kCPU;
        // std::cout << "[FWD] Model device: " << model_device << std::endl;

        if (x.device() != model_device) {
            // std::cout << "[FWD] WARN: Input on " << x.device() << ", model on " << model_device << ". Moving input." << std::endl;
            x = x.to(model_device);
        }

        if (x.size(1) != input_channels_) {
            // std::cout << "[FWD] Adapting channels from " << x.size(1) << " to " << input_channels_ << std::endl;
            auto batch_size = x.size(0);
            auto height = x.size(2);
            auto width = x.size(3);
            
            torch::Tensor adapted_x;
            
            // Try to use GPU memory pool for channel adaptation
            if (gpu_memory_pool_ && x.is_cuda()) {
                try {
                    std::vector<int64_t> shape = {batch_size, input_channels_, height, width};
                    adapted_x = gpu_memory_pool_->allocateTensor(
                        shape,
                        torch::kFloat32,
                        x.device().index(),
                        nullptr
                    );
                    adapted_x.zero_();  // Initialize to zeros
                } catch (const std::exception& e) {
                    // Fall back to regular allocation
                    adapted_x = torch::zeros({batch_size, input_channels_, height, width}, x.options());
                }
            } else {
                adapted_x = torch::zeros({batch_size, input_channels_, height, width}, x.options());
            }
            
            int64_t channels_to_copy = std::min(x.size(1), input_channels_);
            adapted_x.slice(/*dim=*/1, /*start=*/0, /*end=*/channels_to_copy) = x.slice(/*dim=*/1, /*start=*/0, /*end=*/channels_to_copy);
            x = adapted_x;
        }
        
        auto input_layer_start_time = std::chrono::high_resolution_clock::now();
        x = torch::relu(input_bn_(input_conv_(x)));
        auto input_layer_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - input_layer_start_time);
        // Input layer processed
        
        auto res_blocks_total_start_time = std::chrono::high_resolution_clock::now();
        int block_idx = 0;
        for (const auto& block_module : *res_blocks_) {
            auto res_block_single_start_time = std::chrono::high_resolution_clock::now();
            auto* block_ptr = block_module->as<ResNetResidualBlock>();
            if (block_ptr) {
                 x = block_ptr->forward(x);
            } else {
                 std::cerr << "[FWD] Error: Non-ResNetResidualBlock in res_blocks_" << std::endl;
            }
            auto res_block_single_end_time = std::chrono::high_resolution_clock::now();
            auto res_block_single_duration = std::chrono::duration_cast<std::chrono::microseconds>(res_block_single_end_time - res_block_single_start_time);
            if (block_idx < 2 || block_idx == res_blocks_->size() -1) { // Log first 2 and last block
                 // std::cout << "[FWD] ResBlock " << block_idx << ": " << res_block_single_duration.count() << " us. Device: " << x.device() << std::endl;
            }
            block_idx++;
        }
        auto res_blocks_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - res_blocks_total_start_time);
        // All ResBlocks processed
        
        auto policy_head_total_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor policy_out = x; // Start with output from res_blocks

        auto policy_conv_bn_relu_start_time = std::chrono::high_resolution_clock::now();
        policy_out = torch::relu(policy_bn_(policy_conv_(policy_out)));
        auto policy_conv_bn_relu_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - policy_conv_bn_relu_start_time);
        // std::cout << "[FWD] Policy Head Conv+BN+ReLU: " << policy_conv_bn_relu_duration.count() << " us. Shape: " << policy_out.sizes() << std::endl;

        auto policy_view_start_time = std::chrono::high_resolution_clock::now();
        policy_out = policy_out.view({policy_out.size(0), -1});
        auto policy_view_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - policy_view_start_time);
        // std::cout << "[FWD] Policy Head View: " << policy_view_duration.count() << " us. Shape: " << policy_out.sizes() << std::endl;

        auto policy_fc_start_time = std::chrono::high_resolution_clock::now();
        policy_out = policy_fc_(policy_out);
        auto policy_fc_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - policy_fc_start_time);
        // std::cout << "[FWD] Policy Head FC: " << policy_fc_duration.count() << " us. Shape: " << policy_out.sizes() << std::endl;
        
        auto policy_log_softmax_start_time = std::chrono::high_resolution_clock::now();
        policy_out = torch::log_softmax(policy_out, 1);
        auto policy_log_softmax_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - policy_log_softmax_start_time);
        // std::cout << "[FWD] Policy Head LogSoftmax: " << policy_log_softmax_duration.count() << " us. Device: " << policy_out.device() << std::endl;

        auto policy_head_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - policy_head_total_start_time);
        // std::cout << "[FWD] Policy head (Total): " << policy_head_total_duration.count() << " us. Device: " << policy_out.device() << std::endl;
        
        auto value_head_total_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor value_out = x; // Start with output from res_blocks

        auto value_conv_bn_relu_start_time = std::chrono::high_resolution_clock::now();
        value_out = torch::relu(value_bn_(value_conv_(value_out)));
        auto value_conv_bn_relu_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - value_conv_bn_relu_start_time);
        // std::cout << "[FWD] Value Head Conv+BN+ReLU: " << value_conv_bn_relu_duration.count() << " us. Shape: " << value_out.sizes() << std::endl;

        auto value_view_start_time = std::chrono::high_resolution_clock::now();
        value_out = value_out.view({value_out.size(0), -1});
        auto value_view_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - value_view_start_time);
        // std::cout << "[FWD] Value Head View: " << value_view_duration.count() << " us. Shape: " << value_out.sizes() << std::endl;

        auto value_fc1_relu_start_time = std::chrono::high_resolution_clock::now();
        value_out = torch::relu(value_fc1_(value_out));
        auto value_fc1_relu_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - value_fc1_relu_start_time);
        // std::cout << "[FWD] Value Head FC1+ReLU: " << value_fc1_relu_duration.count() << " us. Shape: " << value_out.sizes() << std::endl;
        
        auto value_fc2_tanh_start_time = std::chrono::high_resolution_clock::now();
        value_out = torch::tanh(value_fc2_(value_out));
        auto value_fc2_tanh_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - value_fc2_tanh_start_time);
        // std::cout << "[FWD] Value Head FC2+Tanh: " << value_fc2_tanh_duration.count() << " us. Device: " << value_out.device() << std::endl;
        
        auto value_head_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - value_head_total_start_time);
        // std::cout << "[FWD] Value head (Total): " << value_head_total_duration.count() << " us. Device: " << value_out.device() << std::endl;
        
        auto forward_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - forward_total_start_time);
        // std::cout << "[FWD] Exiting. Total time: " << forward_total_duration.count() << " us. PolicyDev: " << policy_out.device() << " ValDev: " << value_out.device() << std::endl;
        return {policy_out, value_out};

    } catch (const c10::Error& e) {
        std::cerr << "[FWD] PyTorch c10::Error: " << e.what() << ". Input x device: " << x.device() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "[FWD] Std::exception: " << e.what() << ". Input x device: " << x.device() << std::endl;
        throw;
    }
}

// New overloaded version of prepareInputTensor
torch::Tensor ResNetModel::prepareInputTensor(
    const std::vector<std::unique_ptr<core::IGameState>>& states, 
    torch::Device target_device) {

    if (states.empty()) {
        // std::cerr << "ResNetModel::prepareInputTensor - Empty states vector." << std::endl;
        return torch::Tensor(); // Return an empty tensor
    }

    const auto& first_state_ptr = states[0];
    if (!first_state_ptr) {
        std::cerr << "ERROR: First state in vector is null in prepareInputTensor" << std::endl;
        return torch::Tensor();
    }
    
    // Determine expected channels from model configuration for consistency
    std::vector<std::vector<std::vector<float>>> first_tensor_data;
    try {
         first_tensor_data = (input_channels_ == 3) ? 
            first_state_ptr->getTensorRepresentation() : 
            first_state_ptr->getEnhancedTensorRepresentation();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception getting tensor representation from first state: " << e.what() << std::endl;
        return torch::Tensor();
    }


    if (first_tensor_data.empty() || first_tensor_data[0].empty() || first_tensor_data[0][0].empty()) {
        std::cerr << "ERROR: Empty tensor data from first state in prepareInputTensor" << std::endl;
        return torch::Tensor();
    }

    int64_t actual_data_channels = static_cast<int64_t>(first_tensor_data.size());
    int64_t height = static_cast<int64_t>(first_tensor_data[0].size());
    int64_t width = static_cast<int64_t>(first_tensor_data[0][0].size());

    if (height != board_size_ || width != board_size_) {
        std::cerr << "ERROR: State tensor dimensions HxW (" << height << "x" << width 
                  << ") do not match model's expected board_size_ (" << board_size_ << ") in prepareInputTensor." << std::endl;
        return torch::Tensor();
    }
    // The number of channels in the actual data might differ from input_channels_ the model's first conv layer expects.
    // The forward() method is responsible for adapting this. So, we create the tensor with actual_data_channels.

    // Create a CPU tensor first, then move if needed.
    // This simplifies data population from std::vector.
    auto tensor_alloc_start_time = std::chrono::high_resolution_clock::now();
    
    // Always allocate CPU tensor first for data population
    auto cpu_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .memory_format(torch::MemoryFormat::Contiguous)
        .pinned_memory(true);
    torch::Tensor batch_tensor_cpu = torch::empty({static_cast<int64_t>(states.size()), actual_data_channels, height, width}, cpu_options);
    
    // Track CPU tensor allocation
    size_t cpu_tensor_size = batch_tensor_cpu.numel() * batch_tensor_cpu.element_size();
    // // TRACK_MEMORY_ALLOC("NNCPUTensor", cpu_tensor_size); // Removed
    
    // CRITICAL FIX: Create RAII guard to ensure CPU tensor is tracked as freed
    struct CPUTensorGuard {
        size_t size;
        bool active;
        CPUTensorGuard(size_t s) : size(s), active(true) {}
        ~CPUTensorGuard() {
            if (active) {
                // // TRACK_MEMORY_FREE("NNCPUTensor", size); // Removed
            }
        }
        void release() { active = false; }
    } cpu_guard(cpu_tensor_size);
    
    auto tensor_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
    // std::cout << "[PREP] CPU Tensor alloc: " << tensor_alloc_duration.count() << " us." << std::endl;
    
    // PERFORMANCE FIX: Use raw pointer access instead of accessor for speed
    float* data_ptr = batch_tensor_cpu.data_ptr<float>();
    const int64_t state_stride = actual_data_channels * height * width;
    const int64_t channel_stride = height * width;

    // PERFORMANCE FIX: Batch GPU attack/defense computation BEFORE parallel loop
    #ifdef WITH_TORCH
    torch::Tensor attack_defense_batch;
    bool use_gpu_attack_defense = false;
    std::string game_type = "unknown";
    
    auto gpu_attack_defense_start = std::chrono::high_resolution_clock::now();
    if (target_device.is_cuda() && alphazero::utils::AttackDefenseModule::isGPUAvailable()) {
        try {
            // Detect game type and prepare batch
            if (board_size_ == 15 && input_channels_ == 19) {
                // Gomoku
                std::vector<const games::gomoku::GomokuState*> gomoku_states;
                gomoku_states.reserve(states.size());
                
                bool all_gomoku = true;
                for (const auto& state : states) {
                    auto* gomoku_state = dynamic_cast<const games::gomoku::GomokuState*>(state.get());
                    if (gomoku_state) {
                        gomoku_states.push_back(gomoku_state);
                    } else {
                        all_gomoku = false;
                        break;
                    }
                }
                
                if (all_gomoku) {
                    game_type = "gomoku";
                    attack_defense_batch = alphazero::utils::AttackDefenseModule::computeGomokuAttackDefenseGPU(gomoku_states);
                    use_gpu_attack_defense = (attack_defense_batch.size(0) == static_cast<int64_t>(states.size()));
                }
            } else if (board_size_ == 8 && input_channels_ >= 17) {
                // Chess
                std::vector<const games::chess::ChessState*> chess_states;
                chess_states.reserve(states.size());
                
                bool all_chess = true;
                for (const auto& state : states) {
                    auto* chess_state = dynamic_cast<const games::chess::ChessState*>(state.get());
                    if (chess_state) {
                        chess_states.push_back(chess_state);
                    } else {
                        all_chess = false;
                        break;
                    }
                }
                
                if (all_chess) {
                    game_type = "chess";
                    attack_defense_batch = alphazero::utils::AttackDefenseModule::computeChessAttackDefenseGPU(chess_states);
                    use_gpu_attack_defense = (attack_defense_batch.size(0) == static_cast<int64_t>(states.size()));
                }
            } else if (board_size_ == 19 && input_channels_ >= 17) {
                // Go
                std::vector<const games::go::GoState*> go_states;
                go_states.reserve(states.size());
                
                bool all_go = true;
                for (const auto& state : states) {
                    auto* go_state = dynamic_cast<const games::go::GoState*>(state.get());
                    if (go_state) {
                        go_states.push_back(go_state);
                    } else {
                        all_go = false;
                        break;
                    }
                }
                
                if (all_go) {
                    game_type = "go";
                    attack_defense_batch = alphazero::utils::AttackDefenseModule::computeGoAttackDefenseGPU(go_states);
                    use_gpu_attack_defense = (attack_defense_batch.size(0) == static_cast<int64_t>(states.size()));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "GPU attack/defense batch computation failed: " << e.what() << std::endl;
            use_gpu_attack_defense = false;
        }
    }
    
    if (use_gpu_attack_defense) {
        auto gpu_attack_defense_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - gpu_attack_defense_start);
        // GPU attack/defense completed successfully
    }
    #endif
    
    auto data_retrieval_loop_start_time = std::chrono::high_resolution_clock::now();
    
    // Use OpenMP to parallelize tensor conversion across states
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i]) {
            // Zero out this state's data
            std::memset(data_ptr + i * state_stride, 0, state_stride * sizeof(float));
            continue;
        }
        const auto& current_state_ptr = states[i];
        std::vector<std::vector<std::vector<float>>> tensor_data;
        try {
            // For games with GPU attack/defense, get basic representation
            if (use_gpu_attack_defense) {
                // Get basic representation without attack/defense planes
                tensor_data = current_state_ptr->getTensorRepresentation();
                // Extend to include attack/defense channels (filled later from GPU batch)
                tensor_data.resize(actual_data_channels, std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f)));
            } else {
                tensor_data = (input_channels_ == 3) ? 
                    current_state_ptr->getTensorRepresentation() : 
                    current_state_ptr->getEnhancedTensorRepresentation();
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "ERROR: Exception getting tensor representation from state " << i << ": " << e.what() << ". Filling with zeros." << std::endl;
            }
            // Zero out this state's data
            std::memset(data_ptr + i * state_stride, 0, state_stride * sizeof(float));
            continue;
        }


        if (tensor_data.size() != static_cast<size_t>(actual_data_channels) || 
            (actual_data_channels > 0 && tensor_data[0].size() != static_cast<size_t>(height)) ||
            (actual_data_channels > 0 && height > 0 && tensor_data[0][0].size() != static_cast<size_t>(width))) {
            // std::cerr << "WARNING: Tensor dimension mismatch for state " << i << ". Filling with zeros." << std::endl;
            // Zero out this state's data
            std::memset(data_ptr + i * state_stride, 0, state_stride * sizeof(float));
            continue;
        }
        
        // PERFORMANCE FIX: Direct memory copy using raw pointers
        float* state_ptr = data_ptr + i * state_stride;
        
        // Copy main tensor data
        int64_t channels_to_copy = (use_gpu_attack_defense && game_type == "gomoku") ? 17 : 
                                  (use_gpu_attack_defense && game_type == "chess") ? (actual_data_channels - 2) :
                                  (use_gpu_attack_defense && game_type == "go") ? (actual_data_channels - 2) :
                                  actual_data_channels;
        
        for (int64_t c = 0; c < channels_to_copy; ++c) {
            float* channel_ptr = state_ptr + c * channel_stride;
            for (int64_t h = 0; h < height; ++h) {
                // Copy entire row at once
                std::memcpy(channel_ptr + h * width, 
                           tensor_data[c][h].data(), 
                           width * sizeof(float));
            }
        }
        
        // Copy GPU attack/defense planes if available
        #ifdef WITH_TORCH
        if (use_gpu_attack_defense && attack_defense_batch.defined()) {
            // Attack plane
            auto attack_tensor = attack_defense_batch[i][0];
            auto attack_accessor = attack_tensor.accessor<float, 2>();
            float* attack_ptr = state_ptr + (actual_data_channels - 2) * channel_stride;
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    attack_ptr[h * width + w] = attack_accessor[h][w];
                }
            }
            
            // Defense plane
            auto defense_tensor = attack_defense_batch[i][1];
            auto defense_accessor = defense_tensor.accessor<float, 2>();
            float* defense_ptr = state_ptr + (actual_data_channels - 1) * channel_stride;
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    defense_ptr[h * width + w] = defense_accessor[h][w];
                }
            }
        }
        #endif
    }
    auto data_retrieval_loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - data_retrieval_loop_start_time);
    // Data retrieval completed

    if (target_device == torch::kCPU) {
        auto prep_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
        // std::cout << "[PREP] Total prep (to CPU): " << prep_total_duration.count() << " us." << std::endl;
        // CPU tensor is being returned, so don't free it yet
        cpu_guard.release();
        return batch_tensor_cpu;
    } else {
        // For CUDA, use GPU memory pool if available
        if (gpu_memory_pool_ && target_device.is_cuda()) {
            try {
                // Allocate GPU tensor from pool
                std::vector<int64_t> shape = {
                    static_cast<int64_t>(states.size()), 
                    actual_data_channels, 
                    height, 
                    width
                };
                
                auto gpu_alloc_start = std::chrono::high_resolution_clock::now();
                torch::Tensor gpu_tensor = gpu_memory_pool_->allocateTensor(
                    shape, 
                    torch::kFloat32, 
                    target_device.index(),
                    nullptr  // Use default stream
                );
                
                auto gpu_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - gpu_alloc_start);
                
                // Copy from CPU to GPU tensor
                auto copy_start = std::chrono::high_resolution_clock::now();
                gpu_tensor.copy_(batch_tensor_cpu, /*non_blocking=*/true);
                
                auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - copy_start);
                
                // Track GPU tensor allocation
                size_t gpu_tensor_size = gpu_tensor.numel() * gpu_tensor.element_size();
                // TRACK_MEMORY_ALLOC("NNGPUTensor", gpu_tensor_size);
                
                auto prep_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
                
                return gpu_tensor;
            } catch (const std::exception& e) {
                // Fall back to regular allocation
                // Continue with standard path below
            }
        }
        
        // Standard GPU allocation path
        try {
            auto pin_memory_start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor pinned_cpu_tensor = batch_tensor_cpu.pin_memory();
            auto pin_memory_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - pin_memory_start_time);
            // std::cout << "[PREP] Pin memory: " << pin_memory_duration.count() << " us." << std::endl;

            auto to_device_start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor result_tensor = pinned_cpu_tensor.to(target_device, /*non_blocking=*/true);
            
            // Track GPU tensor allocation
            size_t gpu_tensor_size = result_tensor.numel() * result_tensor.element_size();
            // TRACK_MEMORY_ALLOC("NNGPUTensor", gpu_tensor_size);
            
            auto to_device_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - to_device_start_time);
            // std::cout << "[PREP] Move to " << target_device << ": " << to_device_duration.count() << " us." << std::endl;
            
            auto prep_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
            // std::cout << "[PREP] Total prep (to " << target_device << "): " << prep_total_duration.count() << " us." << std::endl;
            return result_tensor;
        } catch (const c10::Error& e) {
            std::cerr << "ResNetModel::prepareInputTensor - PyTorch error pinning or moving tensor to " << target_device << ": " << e.what() 
                      << ". Returning tensor on CPU." << std::endl;
            auto prep_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
            // std::cout << "[PREP] Total prep (to " << target_device << ", exception): " << prep_total_duration.count() << " us." << std::endl;
            return batch_tensor_cpu; // Fallback to CPU tensor
        }
    }
}

// Original signature, calls the overloaded one.
torch::Tensor ResNetModel::prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    torch::Device model_device = torch::kCPU; // Default
    try {
        if (!this->parameters().empty() && this->parameters().front().defined()) {
            model_device = this->parameters().front().device();
        }
    } catch (const std::exception& e) {
        std::cerr << "ResNetModel::prepareInputTensor (old signature) - Warning: Could not determine model device: " 
                  << e.what() << ". Defaulting to CPU for tensor preparation." << std::endl;
    }
    return prepareInputTensor(states, model_device);
}



std::vector<mcts::NetworkOutput> ResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    auto inference_total_start_time = std::chrono::high_resolution_clock::now();
    auto total_start = std::chrono::high_resolution_clock::now();
    // std::cout << "[NN_TRACE] ========== ResNetModel::inference() called with " << states.size() << " states at " 
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(total_start.time_since_epoch()).count() % 100000 << "ms ==========" << std::endl;
    
    // Track memory periodically
    static std::atomic<size_t> inference_count{0};
    // static const size_t cleanup_interval = 10;  // CRITICAL FIX: Clean every 10 inferences to prevent accumulation
    
    size_t count = inference_count.fetch_add(1);
    if (count % 100 == 0) {
        // alphazero::utils::trackMemory("ResNet inference #" + std::to_string(count) + ", batch=" + std::to_string(states.size()));
    }
    
    std::vector<mcts::NetworkOutput> default_outputs;
    if (states.empty()) return default_outputs;
    default_outputs.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput out;
        size_t actual_policy_size = policy_size_;
        if (actual_policy_size == 0 && states[i]) { 
             try { actual_policy_size = states[i]->getActionSpaceSize(); } catch(...) { actual_policy_size = 1; }
        }
        if (actual_policy_size == 0) actual_policy_size = 1; // Final fallback for policy size
        out.policy.resize(actual_policy_size, 1.0f / actual_policy_size);
        out.value = 0.0f;
        default_outputs.push_back(std::move(out));
    }

    try {
        torch::Device model_device = torch::kCPU;
        if (!this->parameters().empty() && this->parameters().front().defined()) {
            model_device = this->parameters().front().device();
        }
        // std::cout << "[INF] Model device: " << model_device << std::endl;

        this->eval(); 
        torch::NoGradGuard no_grad;
        
        // PERFORMANCE FIX: GPU warmup and tensor pool initialization
        static bool gpu_warmed_up = false;
        if (!gpu_warmed_up && model_device.is_cuda()) {
            // Performing GPU warmup
            auto warmup_start = std::chrono::high_resolution_clock::now();
            
            // CRITICAL FIX: Warm up with multiple batch sizes to pre-compile all CUDA kernels
            try {
                // Warm up common batch sizes to avoid kernel compilation during inference
                const std::vector<int> warmup_batch_sizes = {1, 8, 16, 32, 64};
                
                for (int batch_size : warmup_batch_sizes) {
                    torch::Tensor dummy_input = torch::randn({batch_size, input_channels_, board_size_, board_size_}, 
                                                   torch::TensorOptions().dtype(torch::kFloat32).device(model_device));
                    
                    torch::NoGradGuard no_grad_warmup;
                    
                    // Run forward pass to compile kernels
                    auto warmup_iter_start = std::chrono::high_resolution_clock::now();
                    auto [dummy_policy, dummy_value] = this->forward(dummy_input);
                    
                    // Sync on first and last to ensure compilation completes
                    if (batch_size == 1 || batch_size == warmup_batch_sizes.back()) {
                        torch::cuda::synchronize();
                        auto warmup_iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - warmup_iter_start);
                        // Warmup batch completed
                    }
                }
                
                gpu_warmed_up = true;
                
                auto warmup_end = std::chrono::high_resolution_clock::now();
                auto warmup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start);
                // GPU warmup completed
            } catch (const std::exception& e) {
                std::cerr << "GPU warmup failed: " << e.what() << std::endl;
            }
        }
        
        // Direct allocation is more predictable for memory management

        auto prep_start = std::chrono::high_resolution_clock::now();
        // std::cout << "[NN_TRACE] Starting input tensor preparation at +" 
        //           << std::chrono::duration_cast<std::chrono::microseconds>(prep_start - total_start).count() << "μs" << std::endl;
        auto prepare_input_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor input_tensor;
        
        // Direct tensor preparation
        input_tensor = prepareInputTensor(states, model_device);
        
        auto prepare_input_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - prepare_input_start_time);
        // Input tensor prepared

        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            std::cerr << "[INF] prepareInputTensor returned invalid tensor." << std::endl;
            return default_outputs;
        }
        
        auto forward_start = std::chrono::high_resolution_clock::now();
        // std::cout << "[NN_TRACE] Starting forward pass at +" 
        //           << std::chrono::duration_cast<std::chrono::microseconds>(forward_start - total_start).count() 
        //           << "μs with input tensor shape: " << input_tensor.sizes() << ", device: " << input_tensor.device() << std::endl;
        auto forward_pass_start_time = std::chrono::high_resolution_clock::now();
        
        torch::Tensor policy_batch, value_batch;
        
        // Use TorchScript model if available
        if (use_torch_script_ && torch_script_model_) {
            try {
                torch::NoGradGuard no_grad;
                auto outputs = torch_script_model_->forward({input_tensor}).toTuple();
                policy_batch = outputs->elements()[0].toTensor();
                value_batch = outputs->elements()[1].toTensor();
            } catch (const std::exception& e) {
                // Fallback to regular forward
                std::cerr << "TorchScript forward failed, falling back to regular forward: " << e.what() << std::endl;
                auto [p, v] = this->forward(input_tensor);
                policy_batch = p;
                value_batch = v;
            }
        } else {
            auto [p, v] = this->forward(input_tensor);
            policy_batch = p;
            value_batch = v;
        }
        
        auto forward_end = std::chrono::high_resolution_clock::now();
        auto forward_pass_duration = std::chrono::duration_cast<std::chrono::microseconds>(forward_end - forward_pass_start_time);
        // std::cout << "[NN_TRACE] Forward pass completed in " << forward_pass_duration.count() << "μs" << std::endl;
        // Forward pass completed

        auto outputs_to_cpu_detach_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor policy_cpu = policy_batch.to(torch::kCPU, /*non_blocking=*/model_device.is_cuda()).detach();
        torch::Tensor value_cpu = value_batch.to(torch::kCPU, /*non_blocking=*/model_device.is_cuda()).detach();
        auto outputs_to_cpu_detach_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - outputs_to_cpu_detach_start_time);
        // Outputs transferred to CPU
        
        // PERFORMANCE FIX: Remove synchronization after forward pass
        // GPU operations will complete asynchronously while CPU processes results

        if (policy_cpu.size(0) != static_cast<int64_t>(states.size()) || value_cpu.size(0) != static_cast<int64_t>(states.size()) ||
            (policy_size_ > 0 && policy_cpu.size(1) != policy_size_) || value_cpu.size(1) != 1) { // check policy_size_ only if it's set
            std::cerr << "[INF] Output tensor dimension mismatch. Policy: " << policy_cpu.sizes()
                      << " (expected [~" << states.size() << "," << policy_size_ << "]), Value: " << value_cpu.sizes()
                      << " (expected [~" << states.size() << ",1])." << std::endl;
            return default_outputs;
        }

        auto policy_accessor = policy_cpu.accessor<float, 2>();
        auto value_accessor = value_cpu.accessor<float, 2>();

        std::vector<mcts::NetworkOutput> final_outputs;
        final_outputs.reserve(states.size());

        // Convert log probabilities to probabilities using softmax
        // Note: policy_cpu contains log probabilities from log_softmax
        torch::Tensor policy_probs = torch::softmax(policy_batch, /*dim=*/1).to(torch::kCPU).detach();
        auto policy_probs_accessor = policy_probs.accessor<float, 2>();
        
        for (int64_t i = 0; i < policy_probs_accessor.size(0); ++i) {
            mcts::NetworkOutput out;
            out.value = value_accessor[i][0];
            out.policy.resize(policy_probs_accessor.size(1));
            for (int64_t j = 0; j < policy_probs_accessor.size(1); ++j) {
                out.policy[j] = policy_probs_accessor[i][j];
            }
            final_outputs.push_back(std::move(out));
        }
        
        auto inference_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inference_total_start_time);
        // Inference completed successfully
        
        // Only cleanup periodically to maintain memory efficiency without destroying throughput
        inference_count++;
        if (inference_count % 100 == 0) {  // Much less frequent cleanup            
            // Only empty cache occasionally, not after every inference
            if (torch::cuda::is_available()) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
        
        // CRITICAL FIX: Ensure proper cleanup of all tensors
        // Track tensor cleanup (tensors will be freed when they go out of scope)
        if (input_tensor.defined()) {
            size_t tensor_size = input_tensor.numel() * input_tensor.element_size();
            if (input_tensor.is_cuda()) {
                // TRACK_MEMORY_FREE("NNGPUTensor", tensor_size);
            } else {
                // TRACK_MEMORY_FREE("NNCPUTensor", tensor_size);
            }
        }
        
        // Explicitly reset tensors to ensure immediate cleanup
        input_tensor.reset();
        policy_batch.reset();
        value_batch.reset();
        policy_cpu.reset();
        value_cpu.reset();
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        // std::cout << "[NN_TRACE] ========== ResNetModel::inference() returning " << final_outputs.size() 
        //           << " outputs after " << total_duration.count() << "μs total ==========" << std::endl;
        return final_outputs;

    } catch (const c10::Error& e) {
        std::cerr << "[INF] PyTorch c10::Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INF] Std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[INF] Unknown error." << std::endl;
    }
    auto inference_total_duration_exception = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inference_total_start_time);
    // std::cout << "[INF] Exiting due to exception. Total time: " << inference_total_duration_exception.count() << " us." << std::endl;
    return default_outputs;
}

void ResNetModel::save(const std::string& path) {
    try {
        torch::serialize::OutputArchive archive;
        // Call the base torch::nn::Module's save method to serialize parameters and buffers.
        // This saves the state_dict implicitly.
        this->torch::nn::Module::save(archive);
        archive.save_to(path);
        // std::cout << "ResNetModel saved state_dict to " << path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error saving model state_dict: " << e.what() << std::endl;
        throw; // Rethrow
    } catch (const std::exception& e) {
        std::cerr << "Error saving model state_dict: " << e.what() << std::endl;
        throw; // Rethrow
    }
}

void ResNetModel::load(const std::string& path) {
    try {
        torch::NoGradGuard no_grad; // Disable gradients during loading
        torch::serialize::InputArchive archive;
        
        // Determine the device the model is currently on.
        // This device should have been set by the NeuralNetworkFactory before calling load.
        torch::Device model_device = torch::kCPU; // Default if not yet moved or no parameters
        if (!this->parameters().empty() && this->parameters().front().defined()) {
            model_device = this->parameters().front().device();
        }
        // std::cout << "ResNetModel::load - Loading model state_dict from " << path << " onto device: " << model_device << std::endl;
        
        // Load the archive from file, mapping tensors to the model's current device.
        archive.load_from(path, model_device); 
        
        // Call the base torch::nn::Module's load method to deserialize parameters and buffers from the archive.
        this->torch::nn::Module::load(archive); 
        this->eval(); // Set to evaluation mode after loading weights.

    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error loading model state_dict from " << path << ": " << e.what() << std::endl;
        throw; // Rethrow
    } catch (const std::exception& e) {
        std::cerr << "Error loading model state_dict from " << path << ": " << e.what() << std::endl;
        throw; // Rethrow
    }
}

std::vector<int64_t> ResNetModel::getInputShape() const {
    return {input_channels_, board_size_, board_size_};
}

int64_t ResNetModel::getPolicySize() const {
    return policy_size_;
}

// cleanupTensorPool method removed - no longer using tensor pools

void ResNetModel::enableGPUOptimizations(
    bool enable_cuda_graphs,
    bool enable_persistent_kernels, 
    bool enable_torch_script,
    int cuda_stream_priority
) {
    gpu_opt_status_.cuda_graphs_enabled = enable_cuda_graphs;
    gpu_opt_status_.cuda_graphs_supported = enable_cuda_graphs && torch::cuda::is_available();
    gpu_opt_status_.persistent_kernels_enabled = enable_persistent_kernels;
    gpu_opt_status_.torch_script_enabled = enable_torch_script;
    gpu_opt_status_.cuda_stream_priority = cuda_stream_priority;
    
    if (enable_torch_script && torch::cuda::is_available()) {
        // Check if pre-traced model exists
        std::string traced_model_path = "traced_resnet_" + std::to_string(board_size_) + ".pt";
        
        try {
            // Try to load pre-traced model
            auto& gpu_optimizer = nn::getGlobalGPUOptimizer();
            torch_script_model_ = std::make_shared<torch::jit::Module>(
                gpu_optimizer.loadTorchScriptModel(traced_model_path, true, torch::kCUDA)
            );
            use_torch_script_ = true;
            
            std::cout << "ResNet: Loaded pre-traced TorchScript model from " << traced_model_path << std::endl;
        } catch (const std::exception& e) {
            // Pre-traced model not found
            std::cerr << "TorchScript model not found at " << traced_model_path 
                      << ". Please trace the model in Python first using trace_model_for_cpp.py" << std::endl;
            use_torch_script_ = false;
            
            // Note: Direct tracing in C++ is not supported
            // Models must be traced in Python and saved, then loaded here
        }
    }
    
    // Update GPU memory stats
    if (torch::cuda::is_available()) {
        gpu_opt_status_.allocated_memory_mb = 
            c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current / (1024 * 1024);
        gpu_opt_status_.reserved_memory_mb = 
            c10::cuda::CUDACachingAllocator::getDeviceStats(0).reserved_bytes[0].current / (1024 * 1024);
    }
}

ResNetModel::GPUOptimizationStatus ResNetModel::getGPUOptimizationStatus() const {
    // Update current memory usage
    if (torch::cuda::is_available()) {
        gpu_opt_status_.allocated_memory_mb = 
            c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current / (1024 * 1024);
        gpu_opt_status_.reserved_memory_mb = 
            c10::cuda::CUDACachingAllocator::getDeviceStats(0).reserved_bytes[0].current / (1024 * 1024);
    }
    
    return gpu_opt_status_;
}


} // namespace nn
} // namespace alphazero