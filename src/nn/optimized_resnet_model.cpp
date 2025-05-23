#include "nn/optimized_resnet_model.h"
#include "utils/logger.h"
#include <torch/cuda.h>
#include <chrono>
#include <future>
#include <optional>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// Use neural network logger
#define LOG_INFO(...) LOG_NN_INFO(__VA_ARGS__)
#define LOG_WARNING(...) LOG_NN_WARN(__VA_ARGS__)
#define LOG_ERROR(...) LOG_NN_ERROR(__VA_ARGS__)
#define LOG_DEBUG(...) LOG_NN_DEBUG(__VA_ARGS__)

namespace alphazero {
namespace nn {

// Thread-local tensor buffers
thread_local OptimizedResNetModel::TensorBuffers OptimizedResNetModel::buffers_;

void OptimizedResNetModel::TensorBuffers::ensureCapacity(
    size_t batch_size, int channels, int board_size, torch::Device device) {
    
    if (capacity >= batch_size) return;
    
    // Allocate with 20% extra capacity to avoid frequent reallocation
    capacity = static_cast<size_t>(batch_size * 1.2);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    input_buffer = torch::empty({static_cast<int64_t>(capacity), 
                                channels, board_size, board_size}, options);
    
    // Pre-allocate output buffers
    int action_size = board_size * board_size;
    policy_buffer = torch::empty({static_cast<int64_t>(capacity), action_size}, options);
    value_buffer = torch::empty({static_cast<int64_t>(capacity), 1}, options);
}

OptimizedResNetModel::OptimizedResNetModel(
    const std::string& model_path,
    int input_channels,
    int board_size,
    int num_res_blocks,
    int num_filters,
    bool use_gpu)
    : device_(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    , input_channels_(input_channels)
    , board_size_(board_size)
    , num_res_blocks_(num_res_blocks)
    , num_filters_(num_filters) {
    
    LOG_INFO("Initializing optimized ResNet model");
    
    // Load model
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        
        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;
        
        // JIT optimize the model
        model_ = torch::jit::optimize_for_inference(model_);
        
    } catch (const c10::Error& e) {
        LOG_ERROR("Failed to load model: {}", e.what());
        throw;
    }
    
    if (device_.is_cuda()) {
        // Create dedicated CUDA stream if not provided
        if (!cuda_stream_) {
            cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
            owns_stream_ = true;
        }
        
        // Pre-allocate pinned memory for async transfers
        ensurePinnedMemory(64 * input_channels * board_size * board_size * sizeof(float));
        
        // Warm up GPU
        torch::cuda::synchronize();
        auto dummy_input = torch::randn({1, input_channels, board_size, board_size}, 
                                       torch::TensorOptions().device(device_));
        model_.forward({dummy_input});
        
        LOG_INFO("GPU initialization complete");
    }
}

OptimizedResNetModel::~OptimizedResNetModel() {
    if (pinned_input_memory_) {
        cudaFreeHost(pinned_input_memory_);
    }
    
    if (owns_stream_ && cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }
}

void OptimizedResNetModel::ensurePinnedMemory(size_t required_size) {
    if (pinned_memory_size_ >= required_size) return;
    
    if (pinned_input_memory_) {
        cudaFreeHost(pinned_input_memory_);
    }
    
    pinned_memory_size_ = required_size;
    cudaError_t err = cudaMallocHost(&pinned_input_memory_, pinned_memory_size_);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to allocate pinned memory: {}", cudaGetErrorString(err));
        pinned_input_memory_ = nullptr;
        pinned_memory_size_ = 0;
    }
}

torch::Tensor OptimizedResNetModel::statesToTensorAsync(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    const size_t batch_size = states.size();
    const int total_elements = batch_size * input_channels_ * board_size_ * board_size_;
    const size_t total_bytes = total_elements * sizeof(float);
    
    // Ensure we have enough pinned memory
    ensurePinnedMemory(total_bytes);
    
    // Ensure tensor buffers have capacity
    buffers_.ensureCapacity(batch_size, input_channels_, board_size_, device_);
    
    // Fill pinned memory in parallel
    float* pinned_data = static_cast<float*>(pinned_input_memory_);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        // Get tensor representation and copy to pinned memory
        auto tensor_rep = states[i]->getTensorRepresentation();
        float* dest = pinned_data + i * input_channels_ * board_size_ * board_size_;
        size_t idx = 0;
        for (const auto& channel : tensor_rep) {
            for (const auto& row : channel) {
                for (float val : row) {
                    dest[idx++] = val;
                }
            }
        }
    }
    
    // Create tensor view of the actual batch size
    auto input_slice = buffers_.input_buffer.slice(0, 0, batch_size);
    
    // Async copy to GPU using the dedicated stream
    if (device_.is_cuda() && cuda_stream_) {
        // Set stream for tensor operations
        c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(cuda_stream_, device_.index()));
        
        // Copy data asynchronously
        cudaMemcpyAsync(input_slice.data_ptr<float>(),
                       pinned_data,
                       total_bytes,
                       cudaMemcpyHostToDevice,
                       cuda_stream_);
    } else {
        // CPU path - direct copy
        memcpy(input_slice.data_ptr<float>(), pinned_data, total_bytes);
    }
    
    return input_slice;
}

std::vector<mcts::NetworkOutput> OptimizedResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    if (states.empty()) return {};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    const size_t batch_size = states.size();
    
    // Disable gradient computation
    torch::NoGradGuard no_grad;
    
    // Convert states to tensor asynchronously
    torch::Tensor input_tensor = statesToTensorAsync(states);
    
    // Set CUDA stream for operations
    std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
    if (device_.is_cuda() && cuda_stream_) {
        stream_guard.emplace(c10::cuda::getStreamFromExternal(cuda_stream_, device_.index()));
    }
    
    // Forward pass (no synchronization here!)
    std::vector<torch::jit::IValue> inputs{input_tensor};
    auto output = model_.forward(inputs).toTuple();
    
    torch::Tensor policy_logits = output->elements()[0].toTensor();
    torch::Tensor value_logits = output->elements()[1].toTensor();
    
    // Apply softmax to policy (in-place for efficiency)
    torch::Tensor policy_probs = torch::softmax(policy_logits, /*dim=*/1);
    torch::Tensor values = torch::tanh(value_logits);
    
    // Only synchronize once at the end for result retrieval
    if (device_.is_cuda()) {
        // This is the ONLY synchronization point
        cudaStreamSynchronize(cuda_stream_);
    }
    
    // Convert to CPU tensors if needed
    if (policy_probs.is_cuda()) {
        policy_probs = policy_probs.cpu();
        values = values.cpu();
    }
    
    // Extract results
    std::vector<mcts::NetworkOutput> results;
    results.reserve(batch_size);
    
    auto policy_accessor = policy_probs.accessor<float, 2>();
    auto value_accessor = values.accessor<float, 2>();
    
    for (size_t i = 0; i < batch_size; ++i) {
        mcts::NetworkOutput output;
        output.value = value_accessor[i][0];
        
        output.policy.reserve(board_size_ * board_size_);
        for (int j = 0; j < board_size_ * board_size_; ++j) {
            output.policy.push_back(policy_accessor[i][j]);
        }
        
        results.push_back(std::move(output));
    }
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    
    inference_count_++;
    double old_time = total_inference_time_.load();
    total_inference_time_.store(old_time + duration);
    
    return results;
}

std::future<std::vector<mcts::NetworkOutput>> OptimizedResNetModel::asyncInference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    return std::async(std::launch::async, [this, &states]() {
        return this->inference(states);
    });
}

void OptimizedResNetModel::saveModel(const std::string& path) const {
    model_.save(path);
}

void OptimizedResNetModel::loadModel(const std::string& path) {
    model_ = torch::jit::load(path, device_);
    model_.eval();
}

torch::Tensor OptimizedResNetModel::statesToTensor(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    // Fallback synchronous version
    const size_t batch_size = states.size();
    
    torch::Tensor input_tensor = torch::zeros(
        {static_cast<int64_t>(batch_size), input_channels_, board_size_, board_size_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    );
    
    auto accessor = input_tensor.accessor<float, 4>();
    
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        // Get tensor representation and copy to tensor
        auto tensor_rep = states[i]->getTensorRepresentation();
        for (int c = 0; c < input_channels_ && c < static_cast<int>(tensor_rep.size()); ++c) {
            for (int h = 0; h < board_size_ && h < static_cast<int>(tensor_rep[c].size()); ++h) {
                for (int w = 0; w < board_size_ && w < static_cast<int>(tensor_rep[c][h].size()); ++w) {
                    accessor[i][c][h][w] = tensor_rep[c][h][w];
                }
            }
        }
    }
    
    return input_tensor.to(device_);
}

} // namespace nn
} // namespace alphazero