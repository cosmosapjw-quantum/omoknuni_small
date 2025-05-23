// src/nn/resnet_model_performance.cpp
// High-performance ResNet implementation optimized for RTX 3060 Ti
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

// Performance optimization: Pre-allocated tensor cache
static thread_local torch::Tensor tl_cpu_tensor_cache;
static thread_local torch::Tensor tl_gpu_tensor_cache;
static thread_local size_t tl_last_batch_size = 0;

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
    try {
        torch::Device model_device = (!this->parameters().empty() && this->parameters().front().defined()) ? 
            this->parameters().front().device() : torch::kCPU;

        // Move input to model device if needed
        if (x.device() != model_device) {
            x = x.to(model_device, /*non_blocking=*/true);
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

        return std::make_tuple(policy, value);
    } catch (const c10::Error& e) {
        std::cerr << "[FWD] PyTorch c10::Error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "[FWD] Std::exception: " << e.what() << std::endl;
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
                  << ") do not match model's expected board_size_ (" << board_size_ << ")" << std::endl;
        return torch::Tensor();
    }

    // PERFORMANCE: Reuse thread-local tensor cache if possible
    torch::Tensor batch_tensor_cpu;
    bool use_pinned = target_device.is_cuda();
    
    if (tl_cpu_tensor_cache.defined() && tl_last_batch_size == states.size()) {
        // Reuse existing tensor
        batch_tensor_cpu = tl_cpu_tensor_cache;
    } else {
        // Allocate new tensor with pinned memory if using CUDA
        auto cpu_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .memory_format(torch::MemoryFormat::Contiguous)
            .pinned_memory(use_pinned);
        
        batch_tensor_cpu = torch::empty({static_cast<int64_t>(states.size()), 
                                        actual_data_channels, height, width}, cpu_options);
        
        // Cache for reuse
        tl_cpu_tensor_cache = batch_tensor_cpu;
        tl_last_batch_size = states.size();
    }
    
    auto accessor = batch_tensor_cpu.accessor<float, 4>();

    // PERFORMANCE: Limit OpenMP threads to avoid contention
    #pragma omp parallel for schedule(static) num_threads(std::min(4, static_cast<int>(states.size())))
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i]) {
            // Zero-fill null states
            for(int64_t c = 0; c < actual_data_channels; ++c) {
                float* row_start = &accessor[i][c][0][0];
                std::memset(row_start, 0, height * width * sizeof(float));
            }
            continue;
        }
        
        const auto& current_state_ptr = states[i];
        std::vector<std::vector<std::vector<float>>> tensor_data;
        try {
            tensor_data = (input_channels_ == 3) ? 
                current_state_ptr->getTensorRepresentation() : 
                current_state_ptr->getEnhancedTensorRepresentation();
        } catch (const std::exception& e) {
            // Zero-fill on error
            for(int64_t c = 0; c < actual_data_channels; ++c) {
                float* row_start = &accessor[i][c][0][0];
                std::memset(row_start, 0, height * width * sizeof(float));
            }
            continue;
        }

        // Validate dimensions
        if (tensor_data.size() != static_cast<size_t>(actual_data_channels) || 
            (actual_data_channels > 0 && tensor_data[0].size() != static_cast<size_t>(height)) ||
            (actual_data_channels > 0 && height > 0 && tensor_data[0][0].size() != static_cast<size_t>(width))) {
            // Zero-fill on dimension mismatch
            for(int64_t c = 0; c < actual_data_channels; ++c) {
                float* row_start = &accessor[i][c][0][0];
                std::memset(row_start, 0, height * width * sizeof(float));
            }
            continue;
        }
        
        // PERFORMANCE: Optimized contiguous copy
        for (int64_t c = 0; c < actual_data_channels; ++c) {
            float* channel_start = &accessor[i][c][0][0];
            for (int64_t h = 0; h < height; ++h) {
                std::memcpy(channel_start + h * width, 
                           tensor_data[c][h].data(), 
                           width * sizeof(float));
            }
        }
    }

    if (target_device == torch::kCPU) {
        return batch_tensor_cpu;
    } else {
        // PERFORMANCE: Use non-blocking transfer
        try {
            torch::Tensor result_tensor = batch_tensor_cpu.to(target_device, /*non_blocking=*/true);
            
            // Don't synchronize here - let the forward pass handle sync
            return result_tensor;
        } catch (const c10::Error& e) {
            std::cerr << "[PREP] CUDA error moving to device: " << e.what() << std::endl;
            return torch::Tensor();
        }
    }
}

std::vector<mcts::NetworkOutput> ResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Only log in debug mode
    #ifdef DEBUG_NN_TRACE
    std::cout << "[NN_TRACE] ========== ResNetModel::inference() called with " << states.size() << " states ==========" << std::endl;
    #endif
    
    std::vector<mcts::NetworkOutput> default_outputs;
    if (states.empty()) return default_outputs;
    
    // Pre-allocate outputs
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
        static std::once_flag warmup_flag;
        std::call_once(warmup_flag, [this, model_device]() {
            if (model_device.is_cuda()) {
                try {
                    auto dummy_input = torch::zeros({1, input_channels_, board_size_, board_size_}, 
                                                   torch::TensorOptions().dtype(torch::kFloat32).device(model_device));
                    torch::NoGradGuard no_grad_warmup;
                    auto [dummy_policy, dummy_value] = this->forward(dummy_input);
                    torch::cuda::synchronize();
                    #ifdef DEBUG_NN_TRACE
                    std::cout << "[NN_TRACE] GPU warmup completed" << std::endl;
                    #endif
                } catch (const std::exception& e) {
                    std::cerr << "[NN_TRACE] GPU warmup failed: " << e.what() << std::endl;
                }
            }
        });
        
        // Prepare input tensor
        torch::Tensor input_tensor = prepareInputTensor(states, model_device);
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            std::cerr << "[INF] Failed to prepare input tensor." << std::endl;
            return default_outputs;
        }

        // Forward pass
        auto [policy_batch, value_batch] = this->forward(input_tensor);

        // PERFORMANCE: Async CPU transfer
        torch::Tensor policy_cpu = policy_batch.to(torch::kCPU, /*non_blocking=*/true);
        torch::Tensor value_cpu = value_batch.to(torch::kCPU, /*non_blocking=*/true);
        
        // Only sync when we need the data
        if (model_device.is_cuda()) {
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(model_device.index());
            stream.synchronize();
        }

        // Validate output dimensions
        if (policy_cpu.size(0) != static_cast<int64_t>(states.size()) || 
            value_cpu.size(0) != static_cast<int64_t>(states.size()) ||
            (policy_size_ > 0 && policy_cpu.size(1) != policy_size_) || 
            value_cpu.size(1) != 1) {
            std::cerr << "[INF] Output tensor dimension mismatch" << std::endl;
            return default_outputs;
        }

        // PERFORMANCE: Direct accessor for faster data extraction
        auto policy_data = policy_cpu.data_ptr<float>();
        auto value_data = value_cpu.data_ptr<float>();
        int64_t policy_stride = policy_cpu.size(1);

        std::vector<mcts::NetworkOutput> final_outputs;
        final_outputs.reserve(states.size());

        for (int64_t i = 0; i < states.size(); ++i) {
            mcts::NetworkOutput out;
            out.value = value_data[i];
            out.policy.resize(policy_stride);
            
            // Direct memory copy for policy
            std::memcpy(out.policy.data(), 
                       policy_data + i * policy_stride, 
                       policy_stride * sizeof(float));
            
            final_outputs.push_back(std::move(out));
        }
        
        // Periodic async cleanup (every 1000 inferences)
        static std::atomic<size_t> inference_count{0};
        if (++inference_count % 1000 == 0 && model_device.is_cuda()) {
            // Run cleanup in detached thread to avoid blocking
            std::thread([]() {
                try {
                    c10::cuda::CUDACachingAllocator::emptyCache();
                } catch (...) {}
            }).detach();
        }
        
        #ifdef DEBUG_NN_TRACE
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - total_start);
        std::cout << "[NN_TRACE] ========== inference() completed in " 
                  << total_duration.count() << "μs ==========" << std::endl;
        #endif
        
        return final_outputs;

    } catch (const c10::Error& e) {
        std::cerr << "[INF] PyTorch error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INF] Error: " << e.what() << std::endl;
    }
    return default_outputs;
}

void ResNetModel::save(const std::string& path) {
    try {
        torch::serialize::OutputArchive archive;
        this->torch::nn::Module::save(archive);
        archive.save_to(path);
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error saving model: " << e.what() << std::endl;
        throw;
    }
}

void ResNetModel::load(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        this->torch::nn::Module::load(archive);
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error loading model: " << e.what() << std::endl;
        throw;
    }
}

// Implementation of virtual methods
std::vector<int64_t> ResNetModel::getInputShape() const {
    return {input_channels_, board_size_, board_size_};
}

int64_t ResNetModel::getPolicySize() const {
    return policy_size_;
}

void ResNetModel::cleanupTensorPool() {
    tensor_pool_.cleanup();
}

// Implementation of prepareInputTensor (original signature)
torch::Tensor ResNetModel::prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    torch::Device target_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return prepareInputTensor(states, target_device);
}

// TensorPool methods (minimal implementation as we use thread-local caching)
void ResNetModel::TensorPool::init(int64_t batch_size, int64_t channels, int64_t height, int64_t width, torch::Device device) {
    initialized = false;
}

torch::Tensor ResNetModel::TensorPool::getCPUTensor(size_t batch_size) {
    return torch::Tensor();
}

torch::Tensor ResNetModel::TensorPool::getGPUTensor(size_t batch_size) {
    return torch::Tensor();
}

void ResNetModel::TensorPool::cleanup() {
    cpu_tensors.clear();
    gpu_tensors.clear();
    current_idx = 0;
    initialized = false;
    
    // Clear thread-local caches
    tl_cpu_tensor_cache = torch::Tensor();
    tl_gpu_tensor_cache = torch::Tensor();
    tl_last_batch_size = 0;
}

} // namespace nn
} // namespace alphazero