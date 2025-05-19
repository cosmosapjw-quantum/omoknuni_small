// src/nn/resnet_model.cpp
#include "nn/resnet_model.h"
#include "utils/memory_tracker.h"
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono> // For timing
#include <iostream> // For logging
#include <omp.h> // For OpenMP parallelization

namespace alphazero {
namespace nn {

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
            torch::Tensor adapted_x = torch::zeros({batch_size, input_channels_, height, width}, x.options());
            int64_t channels_to_copy = std::min(x.size(1), input_channels_);
            adapted_x.slice(/*dim=*/1, /*start=*/0, /*end=*/channels_to_copy) = x.slice(/*dim=*/1, /*start=*/0, /*end=*/channels_to_copy);
            x = adapted_x;
        }
        
        auto input_layer_start_time = std::chrono::high_resolution_clock::now();
        x = torch::relu(input_bn_(input_conv_(x)));
        auto input_layer_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - input_layer_start_time);
        // std::cout << "[FWD] Input layer: " << input_layer_duration.count() << " us. Device: " << x.device() << std::endl;
        
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
        // std::cout << "[FWD] All ResBlocks: " << res_blocks_total_duration.count() << " us. Device: " << x.device() << std::endl;
        
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
    auto options_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor batch_tensor_cpu = torch::empty({static_cast<int64_t>(states.size()), actual_data_channels, height, width}, options_cpu);
    auto tensor_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
    // std::cout << "[PREP] CPU Tensor alloc: " << tensor_alloc_duration.count() << " us." << std::endl;
    
    auto accessor = batch_tensor_cpu.accessor<float, 4>();

    auto data_retrieval_loop_start_time = std::chrono::high_resolution_clock::now();
    
    // Use OpenMP to parallelize tensor conversion across states
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i]) {
            // std::cerr << "WARNING: Null state at index " << i << " in prepareInputTensor. Filling with zeros." << std::endl;
            for(int64_t c = 0; c < actual_data_channels; ++c) 
                for(int64_t h = 0; h < height; ++h) 
                    for(int64_t w = 0; w < width; ++w) 
                        accessor[i][c][h][w] = 0.0f;
            continue;
        }
        const auto& current_state_ptr = states[i];
        std::vector<std::vector<std::vector<float>>> tensor_data;
        try {
            tensor_data = (input_channels_ == 3) ? 
                current_state_ptr->getTensorRepresentation() : 
                current_state_ptr->getEnhancedTensorRepresentation();
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "ERROR: Exception getting tensor representation from state " << i << ": " << e.what() << ". Filling with zeros." << std::endl;
            }
            for(int64_t c = 0; c < actual_data_channels; ++c) 
                for(int64_t h = 0; h < height; ++h) 
                    for(int64_t w = 0; w < width; ++w) 
                        accessor[i][c][h][w] = 0.0f;
            continue;
        }


        if (tensor_data.size() != static_cast<size_t>(actual_data_channels) || 
            (actual_data_channels > 0 && tensor_data[0].size() != static_cast<size_t>(height)) ||
            (actual_data_channels > 0 && height > 0 && tensor_data[0][0].size() != static_cast<size_t>(width))) {
            // std::cerr << "WARNING: Tensor dimension mismatch for state " << i << ". Filling with zeros." << std::endl;
            for(int64_t c = 0; c < actual_data_channels; ++c) 
                for(int64_t h = 0; h < height; ++h) 
                    for(int64_t w = 0; w < width; ++w) 
                        accessor[i][c][h][w] = 0.0f;
            continue;
        }
        
        // Copy tensor data - innermost loops can remain serial as they're cache-friendly
        for (int64_t c = 0; c < actual_data_channels; ++c) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    accessor[i][c][h][w] = tensor_data[c][h][w];
                }
            }
        }
    }
    auto data_retrieval_loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - data_retrieval_loop_start_time);
    // std::cout << "[PREP] Data retrieval loop: " << data_retrieval_loop_duration.count() << " us for " << states.size() << " states." << std::endl;

    if (target_device == torch::kCPU) {
        auto prep_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tensor_alloc_start_time);
        // std::cout << "[PREP] Total prep (to CPU): " << prep_total_duration.count() << " us." << std::endl;
        return batch_tensor_cpu;
    } else {
        // For CUDA, use pinned memory for potentially faster transfer.
        try {
            auto pin_memory_start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor pinned_cpu_tensor = batch_tensor_cpu.pin_memory();
            auto pin_memory_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - pin_memory_start_time);
            // std::cout << "[PREP] Pin memory: " << pin_memory_duration.count() << " us." << std::endl;

            auto to_device_start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor result_tensor = pinned_cpu_tensor.to(target_device, /*non_blocking=*/true);
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


// TensorPool implementation
void ResNetModel::TensorPool::init(int64_t batch_size, int64_t channels, int64_t height, int64_t width, torch::Device device) {
    if (initialized) return;
    
    cpu_tensors.reserve(pool_size);
    gpu_tensors.reserve(pool_size);
    
    // Pre-allocate tensors
    for (size_t i = 0; i < pool_size; ++i) {
        // Create pinned CPU tensors
        auto cpu_tensor = torch::zeros({batch_size, channels, height, width}, 
                                      torch::dtype(torch::kFloat32).pinned_memory(true));
        cpu_tensors.push_back(cpu_tensor);
        
        // Create GPU tensors if device is CUDA
        if (device.is_cuda()) {
            auto gpu_tensor = torch::zeros({batch_size, channels, height, width}, 
                                         torch::dtype(torch::kFloat32).device(device));
            gpu_tensors.push_back(gpu_tensor);
        }
    }
    
    initialized = true;
}

torch::Tensor ResNetModel::TensorPool::getCPUTensor(size_t batch_size) {
    if (!initialized || cpu_tensors.empty()) {
        return torch::Tensor();
    }
    
    size_t idx = current_idx.fetch_add(1) % pool_size;
    auto tensor = cpu_tensors[idx];
    
    // Resize if needed
    if (static_cast<int64_t>(batch_size) > tensor.size(0)) {
        // Requested batch size is too large for this pooled tensor
        return torch::Tensor(); // Return !defined, will cause fallback to non-pooled allocation
    }

    if (tensor.size(0) != static_cast<int64_t>(batch_size)) {
        tensor = tensor.narrow(/*dim=*/0, /*start=*/0, /*length=*/batch_size);
    }
    
    return tensor;
}

torch::Tensor ResNetModel::TensorPool::getGPUTensor(size_t batch_size) {
    if (!initialized || gpu_tensors.empty()) {
        return torch::Tensor();
    }
    
    size_t idx = current_idx.fetch_add(1) % pool_size;
    auto tensor = gpu_tensors[idx];
    
    // Resize if needed
    if (static_cast<int64_t>(batch_size) > tensor.size(0)) {
        // Requested batch size is too large for this pooled tensor
        return torch::Tensor(); // Return !defined, will cause fallback to non-pooled allocation
    }

    if (tensor.size(0) != static_cast<int64_t>(batch_size)) {
        tensor = tensor.narrow(/*dim=*/0, /*start=*/0, /*length=*/batch_size);
    }
    
    return tensor;
}

std::vector<mcts::NetworkOutput> ResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    auto inference_total_start_time = std::chrono::high_resolution_clock::now();
    // std::cout << "[INF] Entered. Num states: " << states.size() << std::endl;
    
    // Track memory periodically
    static std::atomic<size_t> inference_count{0};
    static const size_t cleanup_interval = 100;  // Clean every 100 inferences
    
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
        
        // Initialize tensor pool if not already done
        if (!tensor_pool_.initialized && !states.empty()) {
            const auto& first_state = states[0];
            if (first_state) {
                auto tensor_data = first_state->getTensorRepresentation();
                int64_t channels = static_cast<int64_t>(tensor_data.size());
                // Reduced pool size to 2 tensors for much lower memory footprint
                tensor_pool_.pool_size = 2;
                tensor_pool_.init(/*batch_size=*/4, channels, board_size_, board_size_, model_device);  // Much smaller pool
            }
        }

        auto prepare_input_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor input_tensor;
        
        // Try to use pre-allocated tensor if available
        if (tensor_pool_.initialized) {
            auto cpu_tensor = tensor_pool_.getCPUTensor(states.size());
            if (cpu_tensor.defined()) {
                // Fill the pre-allocated tensor with state data
                float* data_ptr = cpu_tensor.data_ptr<float>();
                for (size_t i = 0; i < states.size(); ++i) {
                    const auto& state = states[i];
                    if (!state) continue;
                    
                    auto tensor_data = (input_channels_ == 3) ? 
                        state->getTensorRepresentation() : 
                        state->getEnhancedTensorRepresentation();
                    
                    // Copy data into the tensor
                    size_t offset = i * tensor_data.size() * board_size_ * board_size_;
                    for (size_t c = 0; c < tensor_data.size(); ++c) {
                        for (size_t h = 0; h < tensor_data[c].size(); ++h) {
                            for (size_t w = 0; w < tensor_data[c][h].size(); ++w) {
                                data_ptr[offset++] = tensor_data[c][h][w];
                            }
                        }
                    }
                }
                
                // Move to GPU if needed
                if (model_device.is_cuda()) {
                    input_tensor = cpu_tensor.to(model_device, /*non_blocking=*/true);
                } else {
                    input_tensor = cpu_tensor;
                }
            }
        }
        
        // Fallback to original method if pre-allocated tensor failed
        if (!input_tensor.defined()) {
            input_tensor = prepareInputTensor(states, model_device);
        }
        
        auto prepare_input_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - prepare_input_start_time);
        // std::cout << "[INF] prepareInputTensor: " << prepare_input_duration.count() << " us. Input tensor device: " << input_tensor.device() << ", shape: " << input_tensor.sizes() << std::endl;

        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            std::cerr << "[INF] prepareInputTensor returned invalid tensor." << std::endl;
            return default_outputs;
        }
        
        auto forward_pass_start_time = std::chrono::high_resolution_clock::now();
        auto [policy_batch, value_batch] = this->forward(input_tensor);
        auto forward_pass_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - forward_pass_start_time);
        // std::cout << "[INF] Forward pass: " << forward_pass_duration.count() << " us. PolicyDev: " << policy_batch.device() << ", ValDev: " << value_batch.device() << std::endl;

        auto outputs_to_cpu_detach_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor policy_cpu = policy_batch.to(torch::kCPU, /*non_blocking=*/model_device.is_cuda()).detach();
        torch::Tensor value_cpu = value_batch.to(torch::kCPU, /*non_blocking=*/model_device.is_cuda()).detach();
        auto outputs_to_cpu_detach_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - outputs_to_cpu_detach_start_time);
        // std::cout << "[INF] Outputs to CPU & detach: " << outputs_to_cpu_detach_duration.count() << " us." << std::endl;
        
        if (model_device.is_cuda()) {
            auto cuda_sync_start_time = std::chrono::high_resolution_clock::now();
            try {
                at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(model_device.index());
                stream.synchronize();
            } catch (const c10::Error& e) {
                std::cerr << "[INF] CUDA sync error: " << e.what() << std::endl;
            }
            auto cuda_sync_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cuda_sync_start_time);
            // std::cout << "[INF] CUDA sync: " << cuda_sync_duration.count() << " us." << std::endl;
        }

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

        for (int64_t i = 0; i < policy_accessor.size(0); ++i) {
            mcts::NetworkOutput out;
            out.value = value_accessor[i][0];
            out.policy.resize(policy_accessor.size(1));
            for (int64_t j = 0; j < policy_accessor.size(1); ++j) {
                out.policy[j] = policy_accessor[i][j];
            }
            final_outputs.push_back(std::move(out));
        }
        
        auto inference_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inference_total_start_time);
        // std::cout << "[INF] Exiting. Total time: " << inference_total_duration.count() << " us for " << states.size() << " states. Final policy_size: " << (final_outputs.empty() ? 0 : final_outputs[0].policy.size()) << std::endl;
        
        // Periodic cleanup to prevent memory accumulation
        inference_count++;
        if (inference_count % cleanup_interval == 0) {
            tensor_pool_.cleanup();
            if (torch::cuda::is_available()) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
        
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

void ResNetModel::cleanupTensorPool() {
    tensor_pool_.cleanup();
}

void ResNetModel::TensorPool::cleanup() {
    // Clear all tensors to free memory
    cpu_tensors.clear();
    gpu_tensors.clear();
    initialized = false;
    current_idx = 0;
    
    // Force garbage collection
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

} // namespace nn
} // namespace alphazero