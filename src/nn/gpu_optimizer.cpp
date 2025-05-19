#include "nn/gpu_optimizer.h"
#include "utils/logger.h"
#include "utils/profiler.h"
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>

namespace alphazero {
namespace nn {

GPUOptimizer::GPUOptimizer(const Config& config) : config_(config) {
    if (!torch::cuda::is_available()) {
        LOG_SYSTEM_WARN("GPU not available, GPU optimizer will run in CPU mode");
        return;
    }
    
    initializeCUDAStreams();
    if (config_.pre_allocate) {
        allocatePinnedMemory();
        preallocateTensors(config_.max_batch_size);
    }
    
    LOG_SYSTEM_INFO("GPU Optimizer initialized with {} streams, max batch {}", 
                   config_.num_streams, config_.max_batch_size);
}

GPUOptimizer::~GPUOptimizer() {
    cleanupResources();
}

void GPUOptimizer::initializeCUDAStreams() {
    if (!torch::cuda::is_available()) return;
    
    cuda_streams_.resize(config_.num_streams);
    for (size_t i = 0; i < config_.num_streams; ++i) {
        cudaError_t error = cudaStreamCreate(&cuda_streams_[i]);
        if (error != cudaSuccess) {
            LOG_SYSTEM_ERROR("Failed to create CUDA stream {}: {}", i, cudaGetErrorString(error));
        }
    }
}

void GPUOptimizer::allocatePinnedMemory() {
    PROFILE_SCOPE_N("GPUOptimizer::allocatePinnedMemory");
    
    if (!torch::cuda::is_available() || !config_.use_pinned_memory) return;
    
    tensor_cache_ = std::make_unique<TensorCache>();
    
    // Pre-allocate pinned memory tensors
    
    for (size_t i = 0; i < config_.tensor_cache_size; ++i) {
        // Create CPU tensor with pinned memory
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(true);
            
        // Allocate for max batch size
        auto pinned_tensor = torch::empty(
            {static_cast<long>(config_.max_batch_size), 
             static_cast<long>(config_.num_channels),
             static_cast<long>(config_.board_height), 
             static_cast<long>(config_.board_width)}, 
            options);
            
        tensor_cache_->cpu_pinned_tensors.push_back(pinned_tensor);
        
        // Pre-allocate corresponding GPU tensor
        auto gpu_tensor = torch::empty(
            {static_cast<long>(config_.max_batch_size), 
             static_cast<long>(config_.num_channels),
             static_cast<long>(config_.board_height), 
             static_cast<long>(config_.board_width)}, 
            torch::device(torch::kCUDA));
            
        tensor_cache_->gpu_tensors.push_back(gpu_tensor);
    }
    
    LOG_SYSTEM_INFO("Allocated {} pinned memory tensors of size {}x{}x{}x{}", 
                   config_.tensor_cache_size, config_.max_batch_size, 
                   config_.num_channels, config_.board_height, config_.board_width);
}

torch::Tensor GPUOptimizer::prepareStatesBatch(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    PROFILE_SCOPE_N("GPUOptimizer::prepareStatesBatch");
    
    if (states.empty()) {
        return torch::empty({0, static_cast<long>(config_.num_channels), 
                            static_cast<long>(config_.board_height), 
                            static_cast<long>(config_.board_width)});
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t batch_size = states.size();
    const size_t channels = config_.num_channels;
    const size_t height = config_.board_height;
    const size_t width = config_.board_width;
    
    // Get a pre-allocated tensor or create new one
    torch::Tensor cpu_tensor;
    torch::Tensor gpu_tensor;
    
    if (config_.use_pinned_memory && tensor_cache_ && 
        batch_size <= config_.max_batch_size) {
        // Use pre-allocated pinned memory
        size_t tensor_idx = tensor_cache_->next_tensor_.fetch_add(1) % config_.tensor_cache_size;
        cpu_tensor = tensor_cache_->cpu_pinned_tensors[tensor_idx].slice(0, 0, batch_size);
        gpu_tensor = tensor_cache_->gpu_tensors[tensor_idx].slice(0, 0, batch_size);
    } else {
        // Allocate new tensors
        auto cpu_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(config_.use_pinned_memory && torch::cuda::is_available());
            
        cpu_tensor = torch::zeros({static_cast<long>(batch_size), 
                                  static_cast<long>(channels),
                                  static_cast<long>(height), 
                                  static_cast<long>(width)}, 
                                 cpu_options);
                                 
        if (torch::cuda::is_available()) {
            gpu_tensor = torch::empty_like(cpu_tensor, torch::device(torch::kCUDA));
        }
    }
    
    // Convert states to tensor in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < states.size(); ++i) {
        if (states[i]) {
            stateToTensor(*states[i], cpu_tensor, i, channels, height, width);
        }
    }
    
    // Transfer to GPU if available
    if (torch::cuda::is_available()) {
        cudaStream_t stream = getCurrentStream();
        
        // Non-blocking copy from pinned memory to GPU
        gpu_tensor.copy_(cpu_tensor, /*non_blocking=*/true);
        
        // Record event for synchronization if needed
        cudaStreamSynchronize(stream);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        transfer_count_.fetch_add(1);
        total_transfer_time_us_.fetch_add(duration.count());
        
        return gpu_tensor;
    }
    
    return cpu_tensor;
}

void GPUOptimizer::stateToTensor(const core::IGameState& state, torch::Tensor& output,
                                size_t batch_idx, size_t channels, size_t height, size_t width) {
    // Convert game state to tensor representation
    // This is game-specific and should be overridden or specialized per game
    
    // Example implementation for board games
    std::string game_type = core::gameTypeToString(state.getGameType());
    if (game_type == "GOMOKU" || game_type == "GO") {
        // Get tensor representation from the state
        try {
            auto tensor_data = state.getTensorRepresentation();
            
            // Convert vector data to torch tensor
            torch::Tensor state_tensor = torch::zeros({static_cast<long>(channels), 
                                                     static_cast<long>(height), 
                                                     static_cast<long>(width)});
            
            // Copy data from vector representation
            for (size_t c = 0; c < channels && c < tensor_data.size(); ++c) {
                for (size_t h = 0; h < height && h < tensor_data[c].size(); ++h) {
                    for (size_t w = 0; w < width && w < tensor_data[c][h].size(); ++w) {
                        state_tensor[c][h][w] = tensor_data[c][h][w];
                    }
                }
            }
            
            // Copy to the output tensor at the correct batch index
            output[batch_idx].copy_(state_tensor);
        } catch (...) {
            // On error, fill with zeros
            output[batch_idx].zero_();
        }
    } else {
        // Default implementation - fill with zeros
        output[batch_idx].zero_();
    }
}

torch::Tensor GPUOptimizer::getPreallocatedTensor(size_t batch_size, size_t height, 
                                                 size_t width, size_t channels) {
    if (!tensor_cache_ || batch_size > config_.max_batch_size) {
        // Return empty tensor if no cache or batch too large
        return torch::empty({static_cast<long>(batch_size), 
                           static_cast<long>(channels),
                           static_cast<long>(height), 
                           static_cast<long>(width)}, 
                          torch::device(torch::kCUDA));
    }
    
    size_t tensor_idx = tensor_cache_->next_tensor_.fetch_add(1) % config_.tensor_cache_size;
    return tensor_cache_->gpu_tensors[tensor_idx].slice(0, 0, batch_size);
}

cudaStream_t GPUOptimizer::getCurrentStream() {
    if (cuda_streams_.empty()) {
        return cudaStreamDefault;
    }
    
    size_t stream_idx = current_stream_idx_.fetch_add(1) % cuda_streams_.size();
    return cuda_streams_[stream_idx];
}

void GPUOptimizer::synchronizeStreams() {
    for (auto stream : cuda_streams_) {
        cudaStreamSynchronize(stream);
    }
}

GPUOptimizer::MemoryStats GPUOptimizer::getMemoryStats() const {
    MemoryStats stats;
    
    if (torch::cuda::is_available()) {
        // For PyTorch 2.x, use the newer API
        auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        stats.allocated_gpu_memory = device_stats.allocated_bytes[0].current;
        stats.peak_gpu_memory = device_stats.allocated_bytes[0].peak;
    }
    
    stats.allocated_pinned_memory = 0;
    if (tensor_cache_) {
        for (const auto& tensor : tensor_cache_->cpu_pinned_tensors) {
            stats.allocated_pinned_memory += tensor.numel() * sizeof(float);
        }
    }
    
    stats.transfer_count = transfer_count_.load();
    if (stats.transfer_count > 0) {
        stats.avg_transfer_time = std::chrono::microseconds(
            total_transfer_time_us_.load() / stats.transfer_count);
    }
    
    return stats;
}

void GPUOptimizer::preallocateTensors(size_t batch_size) {
    if (!torch::cuda::is_available()) return;
    
    // This method is called during initialization
    // The actual allocation is done in allocatePinnedMemory()
    LOG_SYSTEM_INFO("Pre-allocated tensors for batch size {}", batch_size);
}

void GPUOptimizer::cleanupResources() {
    // Destroy CUDA streams
    for (auto stream : cuda_streams_) {
        cudaStreamDestroy(stream);
    }
    cuda_streams_.clear();
    
    // Clear tensor cache
    if (tensor_cache_) {
        tensor_cache_->gpu_tensors.clear();
        tensor_cache_->cpu_pinned_tensors.clear();
    }
    
    // Force GPU memory cleanup
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

// Global instance
GPUOptimizer& getGlobalGPUOptimizer() {
    static GPUOptimizer global_optimizer;
    return global_optimizer;
}

} // namespace nn
} // namespace alphazero