#ifndef ALPHAZERO_GPU_OPTIMIZER_H
#define ALPHAZERO_GPU_OPTIMIZER_H

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <cmath>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

/**
 * GPU Optimization utilities for MCTS batch inference
 * 
 * Features:
 * - Pinned memory for CPU-GPU transfers
 * - Pre-allocated GPU tensors
 * - CUDA stream management
 * - Double buffering
 * - Tensor batching optimization
 */
class ALPHAZERO_API GPUOptimizer {
public:
    struct Config {
        size_t max_batch_size;
        size_t num_streams;              // For overlapping transfers/computation
        bool use_pinned_memory;          // Use page-locked memory for fast transfers
        bool pre_allocate;               // Pre-allocate GPU tensors
        size_t tensor_cache_size;        // Number of pre-allocated tensor sets
        
        // Input dimensions
        size_t board_height;             // Default for Gomoku
        size_t board_width;
        size_t num_channels;             // Board representation channels
        
        // Constructor with default values
        Config() 
            : max_batch_size(256)
            , num_streams(2)
            , use_pinned_memory(true)
            , pre_allocate(true)
            , tensor_cache_size(4)
            , board_height(15)
            , board_width(15)
            , num_channels(4)
        {}
    };
    
    GPUOptimizer(const Config& config = Config());
    ~GPUOptimizer();
    
    // Convert game states to GPU tensors efficiently
    torch::Tensor prepareStatesBatch(const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    // Pre-allocate tensors for a specific batch size
    void preallocateTensors(size_t batch_size);
    
    // Get pre-allocated tensor for reuse
    torch::Tensor getPreallocatedTensor(size_t batch_size, size_t height, size_t width, size_t channels);
    
    // Stream management
    cudaStream_t getCurrentStream();
    void synchronizeStreams();
    
    // Memory statistics
    struct MemoryStats {
        size_t allocated_gpu_memory;
        size_t allocated_pinned_memory;
        size_t peak_gpu_memory;
        size_t transfer_count;
        std::chrono::microseconds avg_transfer_time;
    };
    
    MemoryStats getMemoryStats() const;
    
private:
    Config config_;
    
    // CUDA streams for overlapping operations
    std::vector<cudaStream_t> cuda_streams_;
    std::atomic<size_t> current_stream_idx_{0};
    
    // Pre-allocated tensors
    struct TensorCache {
        std::vector<torch::Tensor> gpu_tensors;
        std::vector<at::Tensor> cpu_pinned_tensors;
        std::atomic<size_t> next_tensor_{0};
    };
    std::unique_ptr<TensorCache> tensor_cache_;
    
    // Statistics
    mutable std::atomic<size_t> transfer_count_{0};
    mutable std::atomic<size_t> total_transfer_time_us_{0};
    
    // Helper methods
    void initializeCUDAStreams();
    void allocatePinnedMemory();
    void cleanupResources();
    
    // Convert single state to tensor (CPU side)
    void stateToTensor(const core::IGameState& state, torch::Tensor& output, 
                      size_t batch_idx, size_t channels, size_t height, size_t width);
};

// Global GPU optimizer instance
ALPHAZERO_API GPUOptimizer& getGlobalGPUOptimizer();

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_GPU_OPTIMIZER_H