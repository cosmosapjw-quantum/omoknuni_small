#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#include "core/export_macros.h"
#include "nn/neural_network.h"
#include "core/igamestate.h"
#include <memory>
#include <vector>
#include <atomic>

namespace alphazero {
namespace nn {

/**
 * @brief Optimized ResNet implementation with minimal synchronization
 * 
 * Key optimizations:
 * - Removed unnecessary CUDA synchronization calls
 * - Async memory transfers with pinned memory
 * - Persistent tensor buffers to avoid allocations
 * - Stream-based pipelining for overlapped computation
 */
class ALPHAZERO_API OptimizedResNetModel : public NeuralNetwork {
public:
    OptimizedResNetModel(const std::string& model_path,
                        int input_channels,
                        int board_size,
                        int num_res_blocks,
                        int num_filters,
                        bool use_gpu = true);
    
    ~OptimizedResNetModel() override;
    
    // Override base class methods
    std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override;
    
    void saveModel(const std::string& path) const;
    void loadModel(const std::string& path);
    
    // New method for async inference
    std::future<std::vector<mcts::NetworkOutput>> asyncInference(
        const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    // Set dedicated CUDA stream for this instance
    void setCudaStream(cudaStream_t stream) { cuda_stream_ = stream; }
    
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    cudaStream_t cuda_stream_ = nullptr;
    bool owns_stream_ = false;
    
    // Configuration
    int input_channels_;
    int board_size_;
    int num_res_blocks_;
    int num_filters_;
    
    // Persistent buffers for zero-copy inference
    struct TensorBuffers {
        torch::Tensor input_buffer;
        torch::Tensor policy_buffer;
        torch::Tensor value_buffer;
        size_t capacity = 0;
        
        void ensureCapacity(size_t batch_size, int channels, int board_size, torch::Device device);
    };
    
    // Thread-local buffers to avoid contention
    static thread_local TensorBuffers buffers_;
    
    // Pinned memory for async transfers
    void* pinned_input_memory_ = nullptr;
    size_t pinned_memory_size_ = 0;
    
    // Helper methods
    torch::Tensor statesToTensor(
        const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    torch::Tensor statesToTensorAsync(
        const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    void ensurePinnedMemory(size_t required_size);
    
    // Performance monitoring
    std::atomic<size_t> inference_count_{0};
    std::atomic<double> total_inference_time_{0.0};
};

} // namespace nn
} // namespace alphazero