#pragma once

#include <memory>
#include <vector>
#include <atomic>

#ifdef WITH_TORCH
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/torch.h>
#endif

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH
// CUDA Graph for optimized neural network inference
class CUDAGraphCapture {
public:
    struct Config {
        size_t max_batch_size;
        bool enable_graph_capture;
        bool use_stream_capture;
        int capture_warmup_iterations;
        
        Config() : max_batch_size(256), 
                   enable_graph_capture(true),
                   use_stream_capture(true),
                   capture_warmup_iterations(3) {}
    };

    explicit CUDAGraphCapture(const Config& config);
    ~CUDAGraphCapture();

    // Capture neural network inference into CUDA graph
    bool captureInference(
        torch::jit::script::Module& model,
        const torch::Tensor& sample_input,
        size_t batch_size
    );

    // Execute captured graph with new input
    std::pair<torch::Tensor, torch::Tensor> executeGraph(
        const torch::Tensor& input,
        size_t batch_size
    );

    // Check if graph is captured and ready
    bool isGraphReady() const { return graph_captured_; }

    // Get graph execution statistics
    struct GraphStats {
        size_t total_executions;
        size_t graph_hits;
        size_t graph_misses;
        double avg_graph_exec_time_ms;
        double avg_regular_exec_time_ms;
        float speedup_ratio;
    };
    GraphStats getStats() const;

    // Clear captured graph
    void clearGraph();

    // Update input tensor in graph (for dynamic batch sizes)
    bool updateGraphInput(const torch::Tensor& new_input, size_t batch_size);

private:
    Config config_;
    
    // CUDA graph handles
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t graph_exec_;
    cudaStream_t capture_stream_;
    cudaStream_t exec_stream_;
    
    // Graph state
    std::atomic<bool> graph_captured_{false};
    std::atomic<bool> graph_executing_{false};
    size_t captured_batch_size_;
    
    // Cached tensors for graph execution
    torch::Tensor graph_input_buffer_;
    torch::Tensor graph_policy_output_;
    torch::Tensor graph_value_output_;
    
    // Statistics tracking
    std::atomic<size_t> total_executions_{0};
    std::atomic<size_t> graph_hits_{0};
    std::atomic<size_t> graph_misses_{0};
    std::atomic<double> total_graph_time_ms_{0.0};
    std::atomic<double> total_regular_time_ms_{0.0};
    
    // Helper methods
    void initializeStreams();
    void cleanupStreams();
    bool validateBatchSize(size_t batch_size) const;
    void recordGraphExecution(double exec_time_ms, bool used_graph);
};

// Multi-graph manager for different batch sizes
class MultiGraphManager {
public:
    struct Config {
        std::vector<size_t> batch_sizes;  // Pre-captured batch sizes
        size_t max_graphs;                // Maximum number of graphs to keep
        bool auto_capture;                // Automatically capture new sizes
        
        Config() : batch_sizes({32, 64, 128, 256}),
                   max_graphs(8),
                   auto_capture(true) {}
    };

    explicit MultiGraphManager(const Config& config);
    ~MultiGraphManager();

    // Initialize with model
    void initialize(torch::jit::script::Module& model);

    // Get or create graph for specific batch size
    CUDAGraphCapture* getGraphForBatchSize(size_t batch_size);

    // Execute with automatic graph selection
    std::pair<torch::Tensor, torch::Tensor> execute(
        torch::jit::script::Module& model,
        const torch::Tensor& input,
        size_t batch_size
    );

    // Get combined statistics
    struct CombinedStats {
        size_t total_graphs;
        size_t active_graphs;
        std::vector<std::pair<size_t, CUDAGraphCapture::GraphStats>> graph_stats;
        float overall_speedup;
    };
    CombinedStats getStats() const;

private:
    Config config_;
    torch::jit::script::Module* model_;
    
    // Map from batch size to graph
    std::unordered_map<size_t, std::unique_ptr<CUDAGraphCapture>> graphs_;
    std::mutex graphs_mutex_;
    
    // LRU tracking for graph eviction
    std::list<size_t> lru_list_;
    std::unordered_map<size_t, std::list<size_t>::iterator> lru_map_;
    
    void evictLRUGraph();
    void updateLRU(size_t batch_size);
};

#else // !WITH_TORCH
// Dummy class when torch is not available
class CUDAGraphCapture {
public:
    struct Config {};
    CUDAGraphCapture(const Config& = {}) {}
    bool isGraphCaptured(size_t) const { return false; }
    void captureGraph(size_t, std::function<void()>) {}
    void executeGraph(size_t) {}
};
#endif // WITH_TORCH

}  // namespace mcts
}  // namespace alphazero