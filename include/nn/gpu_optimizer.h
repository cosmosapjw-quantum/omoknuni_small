#ifndef ALPHAZERO_GPU_OPTIMIZER_H
#define ALPHAZERO_GPU_OPTIMIZER_H

#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <functional>
#include <string>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <functional>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

#ifdef WITH_TORCH
/**
 * GPU Optimization utilities for MCTS batch inference
 * 
 * Features:
 * - Pinned memory for CPU-GPU transfers
 * - Pre-allocated GPU tensors
 * - CUDA stream management
 * - Double buffering
 * - Tensor batching optimization
 * - CUDA graphs for deterministic models
 * - Persistent kernels
 * - TorchScript optimization
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
        
        // CUDA Graph settings
        bool enable_cuda_graphs;         // Use CUDA graphs for deterministic models
        int cuda_graph_warmup_steps;     // Warmup iterations before capturing
        
        // Optimization settings
        bool enable_persistent_kernels;  // Keep data on GPU between ops
        bool enable_torch_script;        // JIT compile models
        bool enable_tensor_cores;        // Use tensor cores if available
        
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
            , enable_cuda_graphs(true)
            , cuda_graph_warmup_steps(10)
            , enable_persistent_kernels(true)
            , enable_torch_script(true)
            , enable_tensor_cores(true)
        {}
    };
    
    // Double-buffering system for improved CPU-GPU parallelism
    struct BufferPair {
        torch::Tensor cpu_tensor;  // Pinned memory tensor
        torch::Tensor gpu_tensor;  // Device memory tensor
        cudaEvent_t copy_done;     // Event to signal copy completion
        cudaEvent_t compute_done;  // Event to signal inference completion
        std::atomic<bool> in_use;  // Whether this buffer is currently in use

        // Default constructor
        BufferPair() : copy_done(nullptr), compute_done(nullptr), in_use(false) {}

        // Move constructor
        BufferPair(BufferPair&& other) noexcept
            : cpu_tensor(std::move(other.cpu_tensor)),
              gpu_tensor(std::move(other.gpu_tensor)),
              copy_done(other.copy_done),
              compute_done(other.compute_done),
              in_use(other.in_use.load(std::memory_order_relaxed)) {
            other.copy_done = nullptr;
            other.compute_done = nullptr;
        }

        // Move assignment operator
        BufferPair& operator=(BufferPair&& other) noexcept {
            if (this != &other) {
                cpu_tensor = std::move(other.cpu_tensor);
                gpu_tensor = std::move(other.gpu_tensor);
                
                // Clean up existing events before overwriting (if they exist)
                if (copy_done) cudaEventDestroy(copy_done);
                if (compute_done) cudaEventDestroy(compute_done);

                copy_done = other.copy_done;
                compute_done = other.compute_done;
                in_use.store(other.in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
                
                other.copy_done = nullptr;
                other.compute_done = nullptr;
            }
            return *this;
        }

        // Prevent copying
        BufferPair(const BufferPair&) = delete;
        BufferPair& operator=(const BufferPair&) = delete;

        // Destructor to clean up CUDA events
        ~BufferPair() {
            if (copy_done) {
                cudaEventDestroy(copy_done);
                copy_done = nullptr;
            }
            if (compute_done) {
                cudaEventDestroy(compute_done);
                compute_done = nullptr;
            }
        }
    };
    
    GPUOptimizer(const Config& config = Config());
    ~GPUOptimizer();
    
    // Get current configuration
    const Config& getConfig() const { return config_; }
    
    // Convert game states to GPU tensors efficiently
    // Set synchronize=false to use asynchronous transfer with double-buffering
    torch::Tensor prepareStatesBatch(const std::vector<std::unique_ptr<core::IGameState>>& states, 
                                   bool synchronize = true);
                                   
    // Get a buffer pair for double-buffering
    BufferPair& getNextBufferPair(size_t batch_size);
    
    // Pre-allocate tensors for a specific batch size
    void preallocateTensors(size_t batch_size);
    
    // Get pre-allocated tensor for reuse
    torch::Tensor getPreallocatedTensor(size_t batch_size, size_t height, size_t width, size_t channels);
    
    // Stream management
    cudaStream_t getCurrentStream();
    void synchronizeStreams();
    
    // CUDA Graph support
    struct CudaGraphHandle {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        std::vector<int64_t> input_shape;
        bool is_valid = false;
        int warmup_count = 0;
    };
    
    // Create and execute CUDA graphs for fixed computation patterns
    bool captureCudaGraph(
        const std::string& graph_id,
        std::function<torch::Tensor()> forward_fn,
        const torch::Tensor& example_input
    );
    
    torch::Tensor executeCudaGraph(
        const std::string& graph_id,
        const torch::Tensor& input
    );
    
    bool isCudaGraphAvailable(const std::string& graph_id) const;
    
    // TorchScript optimization
    torch::jit::Module optimizeWithTorchScript(
        torch::nn::Module& model,
        const std::vector<int64_t>& example_input_shape,
        bool optimize_for_inference = true
    );
    
    // Load pre-traced TorchScript model
    torch::jit::Module loadTorchScriptModel(
        const std::string& model_path,
        bool optimize_for_inference = true,
        torch::Device device = torch::kCUDA
    );
    
    // Advanced batching with dynamic accumulation
    class DynamicBatchAccumulator {
    public:
        DynamicBatchAccumulator(GPUOptimizer* optimizer, int optimal_size, int max_size);
        
        void addInput(torch::Tensor input, size_t request_id);
        bool shouldProcess() const;
        std::pair<torch::Tensor, std::vector<size_t>> extractBatch();
        void reset();
        
        // Adaptive sizing based on queue pressure
        void updateOptimalSize(int queue_depth, float gpu_utilization);
        
    private:
        GPUOptimizer* optimizer_;
        std::vector<torch::Tensor> pending_inputs_;
        std::vector<size_t> request_ids_;
        int optimal_size_;
        int max_size_;
        int current_target_size_;
        std::chrono::steady_clock::time_point first_input_time_;
        static constexpr auto MAX_WAIT_TIME = std::chrono::microseconds(2000); // 2ms max wait
    };
    
    std::unique_ptr<DynamicBatchAccumulator> createBatchAccumulator(
        int optimal_size = 64,
        int max_size = 256
    );
    
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
    
    // Pre-allocated tensors and buffer pairs
    struct TensorCache {
        std::vector<torch::Tensor> gpu_tensors;
        std::vector<at::Tensor> cpu_pinned_tensors;
        std::atomic<size_t> next_tensor_{0};
        
        // Double-buffering support
        std::vector<BufferPair> buffer_pairs;
        std::atomic<size_t> next_buffer_{0};
    };
    std::unique_ptr<TensorCache> tensor_cache_;
    
    // CUDA Graph cache
    std::unordered_map<std::string, CudaGraphHandle> cuda_graphs_;
    mutable std::mutex cuda_graph_mutex_;
    
    // TorchScript models cache
    std::unordered_map<std::string, torch::jit::Module> torch_script_models_;
    mutable std::mutex torch_script_mutex_;
    
    // Statistics
    mutable std::atomic<size_t> transfer_count_{0};
    mutable std::atomic<size_t> total_transfer_time_us_{0};
    mutable std::atomic<size_t> cuda_graph_hits_{0};
    mutable std::atomic<size_t> cuda_graph_misses_{0};
    
    // Helper methods
    void initializeCUDAStreams();
    void allocatePinnedMemory();
    void cleanupResources();
    void setupPersistentKernels();
    
    // Convert single state to tensor (CPU side)
    void stateToTensor(const core::IGameState& state, torch::Tensor& output, 
                      size_t batch_idx, size_t channels, size_t height, size_t width);
};

// Global GPU optimizer instance
ALPHAZERO_API GPUOptimizer& getGlobalGPUOptimizer();

#else // !WITH_TORCH
// Dummy class when torch is not available
class ALPHAZERO_API GPUOptimizer {
public:
    struct Config {
        bool enable_cuda_graphs;
        Config() : enable_cuda_graphs(false) {}
    };
    
    class DynamicBatchAccumulator {
    public:
        void updateOptimalSize(int, float) {}
    };
    
    explicit GPUOptimizer(const Config& config = Config()) {}
    void warmup() {}
    void cleanup() {}
    
    const Config& getConfig() const { 
        static Config config;
        return config; 
    }
    
    std::unique_ptr<DynamicBatchAccumulator> createBatchAccumulator(int = 64, int = 256) {
        return std::make_unique<DynamicBatchAccumulator>();
    }
    
    bool isCudaGraphAvailable(const std::string&) const { return false; }
    void prepareStatesBatch(const std::vector<std::unique_ptr<core::IGameState>>&, bool = true) {}
    void captureCudaGraph(const std::string&, std::function<void()>, const void*) {}
};

inline GPUOptimizer& getGlobalGPUOptimizer() {
    static GPUOptimizer instance;
    return instance;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_GPU_OPTIMIZER_H