// src/nn/resnet_model_optimized.cpp
// Optimized version with async cleanup and no blocking operations during inference
#include "nn/resnet_model.h"
#include "utils/memory_tracker.h"
#include "mcts/aggressive_memory_manager.h"
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cstring>
#include <thread>
#include <atomic>

namespace alphazero {
namespace nn {

// Global flag for async cleanup
static std::atomic<bool> cleanup_requested{false};
static std::thread cleanup_thread;
static std::atomic<bool> cleanup_thread_running{false};

// Async cleanup function
void asyncCleanup() {
    if (torch::cuda::is_available()) {
        try {
            // Run cleanup in background without blocking inference
            c10::cuda::CUDACachingAllocator::emptyCache();
        } catch (...) {
            // Ignore cleanup errors
        }
    }
}

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
    }
    register_module("res_blocks", res_blocks_);
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
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

    try {
        torch::Device model_device = (!this->parameters().empty() && this->parameters().front().defined()) ? 
            this->parameters().front().device() : torch::kCPU;

        // Move input to model device if needed
        if (x.device() != model_device) {
            auto x_to_device_start_time = std::chrono::high_resolution_clock::now();
            x = x.to(model_device, /*non_blocking=*/true);
            auto x_to_device_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - x_to_device_start_time);
        }

        // Input block
        x = torch::relu(input_bn_(input_conv_(x)));

        // Residual blocks
        for (const auto& block : *res_blocks_) {
            x = block->as<ResNetResidualBlock>()->forward(x);
        }

        // Policy head
        torch::Tensor policy = torch::relu(policy_bn_(policy_conv_(x)));
        policy = policy.view({policy.size(0), -1});
        policy = policy_fc_(policy);

        // Value head
        torch::Tensor value = torch::relu(value_bn_(value_conv_(x)));
        value = value.view({value.size(0), -1});
        value = torch::relu(value_fc1_(value));
        value = torch::tanh(value_fc2_(value));

        auto forward_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - forward_total_start_time);

        return std::make_tuple(policy, value);
    } catch (const c10::Error& e) {
        std::cerr << "[FWD] PyTorch c10::Error: " << e.what() << ". Input x device: " << x.device() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "[FWD] Std::exception: " << e.what() << ". Input x device: " << x.device() << std::endl;
        throw;
    }
}

torch::Tensor ResNetModel::prepareInputTensor(
    const std::vector<std::unique_ptr<core::IGameState>>& states, 
    torch::Device target_device) {

    if (states.empty()) {
        return torch::Tensor();
    }

    const auto& first_state_ptr = states[0];
    if (!first_state_ptr) {
        std::cerr << "ERROR: First state in vector is null in prepareInputTensor" << std::endl;
        return torch::Tensor();
    }
    
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

    // PERFORMANCE: Use pinned memory for faster GPU transfer
    auto cpu_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .memory_format(torch::MemoryFormat::Contiguous)
        .pinned_memory(target_device.is_cuda()); // Only pin if using CUDA
    
    torch::Tensor batch_tensor_cpu = torch::empty({static_cast<int64_t>(states.size()), 
                                                   actual_data_channels, height, width}, cpu_options);
    
    // Track CPU tensor allocation
    size_t cpu_tensor_size = batch_tensor_cpu.numel() * batch_tensor_cpu.element_size();
    TRACK_MEMORY_ALLOC("NNCPUTensor", cpu_tensor_size);
    
    auto accessor = batch_tensor_cpu.accessor<float, 4>();

    // PERFORMANCE: Use OpenMP to parallelize tensor conversion
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i]) {
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
                std::cerr << "ERROR: Exception getting tensor representation from state " << i 
                         << ": " << e.what() << ". Filling with zeros." << std::endl;
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
            for(int64_t c = 0; c < actual_data_channels; ++c) 
                for(int64_t h = 0; h < height; ++h) 
                    for(int64_t w = 0; w < width; ++w) 
                        accessor[i][c][h][w] = 0.0f;
            continue;
        }
        
        // PERFORMANCE: Use memcpy for efficient row copying
        for (int64_t c = 0; c < actual_data_channels; ++c) {
            for (int64_t h = 0; h < height; ++h) {
                std::memcpy(&accessor[i][c][h][0], 
                           tensor_data[c][h].data(), 
                           width * sizeof(float));
            }
        }
    }

    if (target_device == torch::kCPU) {
        return batch_tensor_cpu;
    } else {
        // PERFORMANCE: Use non-blocking transfer for GPU
        try {
            torch::Tensor result_tensor = batch_tensor_cpu.to(target_device, /*non_blocking=*/true);
            
            // Track GPU tensor allocation
            size_t gpu_tensor_size = result_tensor.numel() * result_tensor.element_size();
            TRACK_MEMORY_ALLOC("NNGPUTensor", gpu_tensor_size);
            
            // Free CPU tensor memory tracking
            TRACK_MEMORY_FREE("NNCPUTensor", cpu_tensor_size);
            
            return result_tensor;
        } catch (const c10::Error& e) {
            std::cerr << "[PREP] CUDA error moving to device " << target_device << ": " << e.what() << std::endl;
            return torch::Tensor();
        }
    }
}

std::vector<mcts::NetworkOutput> ResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "[NN_TRACE] ========== ResNetModel::inference() called with " << states.size() << " states at " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_start.time_since_epoch()).count() % 100000 << "ms ==========" << std::endl;
    
    // Track memory periodically (but don't do cleanup during inference)
    static std::atomic<size_t> inference_count{0};
    static const size_t cleanup_interval = 100;  // Request cleanup less frequently
    
    std::vector<mcts::NetworkOutput> default_outputs;
    if (states.empty()) return default_outputs;
    default_outputs.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput out;
        size_t actual_policy_size = policy_size_;
        if (actual_policy_size == 0 && states[i]) { 
             try { actual_policy_size = states[i]->getActionSpaceSize(); } catch(...) { actual_policy_size = 1; }
        }
        if (actual_policy_size == 0) actual_policy_size = 1;
        out.policy.resize(actual_policy_size, 1.0f / actual_policy_size);
        out.value = 0.0f;
        default_outputs.push_back(std::move(out));
    }

    try {
        torch::Device model_device = torch::kCPU;
        if (!this->parameters().empty() && this->parameters().front().defined()) {
            model_device = this->parameters().front().device();
        }

        this->eval(); 
        torch::NoGradGuard no_grad;
        
        // GPU warmup (only once)
        static bool gpu_warmed_up = false;
        if (!gpu_warmed_up && model_device.is_cuda()) {
            std::cout << "[NN_TRACE] Performing GPU warmup..." << std::endl;
            auto warmup_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto dummy_input = torch::zeros({1, input_channels_, board_size_, board_size_}, 
                                               torch::TensorOptions().dtype(torch::kFloat32).device(model_device));
                
                size_t warmup_tensor_size = dummy_input.numel() * dummy_input.element_size();
                TRACK_MEMORY_ALLOC("GPUWarmupTensor", warmup_tensor_size);
                
                torch::NoGradGuard no_grad_warmup;
                auto [dummy_policy, dummy_value] = this->forward(dummy_input);
                torch::cuda::synchronize();
                gpu_warmed_up = true;
                
                TRACK_MEMORY_FREE("GPUWarmupTensor", warmup_tensor_size);
                
                auto warmup_end = std::chrono::high_resolution_clock::now();
                auto warmup_duration = std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start);
                std::cout << "[NN_TRACE] GPU warmup completed in " << warmup_duration.count() << "μs" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[NN_TRACE] GPU warmup failed: " << e.what() << std::endl;
            }
        }
        
        // Prepare input tensor
        auto prep_start = std::chrono::high_resolution_clock::now();
        std::cout << "[NN_TRACE] Starting input tensor preparation at +" 
                  << std::chrono::duration_cast<std::chrono::microseconds>(prep_start - total_start).count() << "μs" << std::endl;
        
        torch::Tensor input_tensor = prepareInputTensor(states, model_device);
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            std::cerr << "[INF] Failed to prepare input tensor." << std::endl;
            return default_outputs;
        }

        // Forward pass
        auto forward_pass_start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[NN_TRACE] Starting forward pass at +" 
                  << std::chrono::duration_cast<std::chrono::microseconds>(forward_pass_start_time - total_start).count() 
                  << "μs with input tensor shape: " << input_tensor.sizes() << ", device: " << input_tensor.device() << std::endl;
        
        auto [policy_batch, value_batch] = this->forward(input_tensor);
        
        auto forward_pass_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - forward_pass_start_time);
        std::cout << "[NN_TRACE] Forward pass completed in " << forward_pass_duration.count() << "μs" << std::endl;

        // PERFORMANCE: Use non-blocking transfers and avoid synchronization where possible
        torch::Tensor policy_cpu = policy_batch.to(torch::kCPU, /*non_blocking=*/true).detach();
        torch::Tensor value_cpu = value_batch.to(torch::kCPU, /*non_blocking=*/true).detach();
        
        // Only sync if necessary (when data is actually needed)
        if (model_device.is_cuda()) {
            // PERFORMANCE: Use stream synchronization instead of device synchronization
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(model_device.index());
            stream.synchronize();
        }

        if (policy_cpu.size(0) != static_cast<int64_t>(states.size()) || 
            value_cpu.size(0) != static_cast<int64_t>(states.size()) ||
            (policy_size_ > 0 && policy_cpu.size(1) != policy_size_) || 
            value_cpu.size(1) != 1) {
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
        
        // PERFORMANCE: Request async cleanup instead of blocking
        size_t count = inference_count.fetch_add(1);
        if (count % cleanup_interval == 0) {
            cleanup_requested.store(true);
            // Start cleanup thread if not running
            if (!cleanup_thread_running.exchange(true)) {
                if (cleanup_thread.joinable()) {
                    cleanup_thread.join();
                }
                cleanup_thread = std::thread([]() {
                    asyncCleanup();
                    cleanup_requested.store(false);
                    cleanup_thread_running.store(false);
                });
                cleanup_thread.detach();
            }
        }
        
        // Track tensor cleanup
        if (input_tensor.defined()) {
            size_t tensor_size = input_tensor.numel() * input_tensor.element_size();
            if (input_tensor.is_cuda()) {
                TRACK_MEMORY_FREE("NNGPUTensor", tensor_size);
            } else {
                TRACK_MEMORY_FREE("NNCPUTensor", tensor_size);
            }
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        std::cout << "[NN_TRACE] ========== ResNetModel::inference() returning " << final_outputs.size() 
                  << " outputs after " << total_duration.count() << "μs total ==========" << std::endl;
        return final_outputs;

    } catch (const c10::Error& e) {
        std::cerr << "[INF] PyTorch c10::Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INF] Std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[INF] Unknown error." << std::endl;
    }
    return default_outputs;
}

void ResNetModel::save(const std::string& path) {
    try {
        torch::serialize::OutputArchive archive;
        this->torch::nn::Module::save(archive);
        archive.save_to(path);
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error saving model state_dict: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        throw;
    }
}

void ResNetModel::load(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        this->torch::nn::Module::load(archive);
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error loading model state_dict: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}

// Implementation of getInputShape
std::vector<int64_t> ResNetModel::getInputShape() const {
    return {input_channels_, board_size_, board_size_};
}

// Implementation of getPolicySize
int64_t ResNetModel::getPolicySize() const {
    return policy_size_;
}

// Implementation of cleanupTensorPool
void ResNetModel::cleanupTensorPool() {
    tensor_pool_.cleanup();
}

// Implementation of prepareInputTensor (original signature)
torch::Tensor ResNetModel::prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    // Default to CUDA if available
    torch::Device target_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return prepareInputTensor(states, target_device);
}

// TensorPool methods
void ResNetModel::TensorPool::init(int64_t batch_size, int64_t channels, int64_t height, int64_t width, torch::Device device) {
    // Disable tensor pool - not used in optimized version
    initialized = false;
}

torch::Tensor ResNetModel::TensorPool::getCPUTensor(size_t batch_size) {
    // Disable tensor pool - direct allocation is more predictable
    return torch::Tensor();
}

torch::Tensor ResNetModel::TensorPool::getGPUTensor(size_t batch_size) {
    // Disable tensor pool - direct allocation is more predictable
    return torch::Tensor();
}

void ResNetModel::TensorPool::cleanup() {
    // No-op since pool is disabled
    cpu_tensors.clear();
    gpu_tensors.clear();
    current_idx = 0;
    initialized = false;
}

} // namespace nn
} // namespace alphazero