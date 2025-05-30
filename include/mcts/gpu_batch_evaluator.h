#pragma once

#include <vector>
#include <memory>
#include <queue>
#include <future>
#include <atomic>
#include <chrono>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <cuda_runtime.h>
#endif
#include "core/igamestate.h"
#include "mcts/evaluation_types.h"
#include "nn/neural_network.h"
#include "nn/gpu_optimizer.h"
#include "moodycamel/concurrentqueue.h"
#include <future>

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH
/**
 * GPU-optimized batch evaluator for MCTS
 * 
 * Key improvements:
 * - Tensorized tree representation for GPU processing
 * - Batch UCB calculation on GPU
 * - Triple-buffering for CPU-GPU overlap
 * - Dynamic batch sizing based on tree characteristics
 * - CUDA kernels for tree operations
 */
class GPUBatchEvaluator {
public:
    struct Config {
        // Batch processing
        size_t min_batch_size;      // Minimum batch for GPU efficiency
        size_t max_batch_size;     // Maximum batch size
        size_t batch_timeout_us;  // Microseconds to wait for batch
        
        // GPU optimization
        bool enable_cuda_graphs;
        bool enable_tensor_cores;
        size_t num_cuda_streams;
        
        // Tree tensorization
        size_t max_nodes_per_tree;
        size_t max_actions;        // Maximum branching factor
        
        // Memory management
        bool use_unified_memory;  // For tree statistics
        size_t gpu_memory_limit_mb;
        
        // Dynamic batching
        bool enable_adaptive_batching;
        float gpu_utilization_target;
        
        // Large branching factor optimization
        bool enable_sparse_selection;  // Use sparse tensors for >256 actions
        size_t sparse_threshold;       // Threshold to switch to sparse
        bool enable_topk_filtering;    // Only process top-K actions
        size_t topk_actions;          // Number of top actions to consider
        
        // Constructor with default values
        Config() :
            min_batch_size(64),
            max_batch_size(256),
            batch_timeout_us(2000),
            enable_cuda_graphs(true),
            enable_tensor_cores(true),
            num_cuda_streams(3),
            max_nodes_per_tree(10000),
            max_actions(512),
            use_unified_memory(false),
            gpu_memory_limit_mb(4096),
            enable_adaptive_batching(true),
            gpu_utilization_target(0.85f),
            enable_sparse_selection(true),
            sparse_threshold(256),
            enable_topk_filtering(true),
            topk_actions(64) {}
    };
    
    // Tensorized node representation for GPU
    struct TensorNode {
        torch::Tensor Q_values;      // [num_actions] - action values
        torch::Tensor N_visits;      // [num_actions] - visit counts
        torch::Tensor P_priors;      // [num_actions] - prior probabilities
        torch::Tensor W_values;      // [num_actions] - cumulative values
        torch::Tensor virtual_loss;  // [num_actions] - virtual loss counts
        
        int32_t total_visits = 0;
        int32_t node_id = -1;
        int32_t parent_id = -1;
        bool is_expanded = false;
        
        // Children indices for tree structure
        std::vector<int32_t> children_ids;
    };
    
    // Batch request for evaluation
    struct BatchRequest {
        std::vector<std::unique_ptr<core::IGameState>> states;
        std::vector<int32_t> node_ids;
        std::promise<std::vector<NetworkOutput>> promise;
        std::chrono::steady_clock::time_point timestamp;
        
        BatchRequest() = default;
        BatchRequest(BatchRequest&&) = default;
        BatchRequest& operator=(BatchRequest&&) = default;
    };
    
    // GPU tensors for batch tree operations
    struct TreeTensors {
        // Tree statistics - all on GPU
        torch::Tensor Q_tensor;          // [batch_size, max_nodes, max_actions]
        torch::Tensor N_tensor;          // [batch_size, max_nodes, max_actions]
        torch::Tensor P_tensor;          // [batch_size, max_nodes, max_actions]
        torch::Tensor W_tensor;          // [batch_size, max_nodes, max_actions]
        torch::Tensor virtual_losses;    // [batch_size, max_nodes, max_actions]
        
        // Tree structure
        torch::Tensor parent_indices;    // [batch_size, max_nodes]
        torch::Tensor children_mask;     // [batch_size, max_nodes, max_actions]
        torch::Tensor node_visits;       // [batch_size, max_nodes]
        
        // Selection results
        torch::Tensor ucb_scores;        // [batch_size, max_nodes, max_actions]
        torch::Tensor selected_actions;  // [batch_size, max_depth]
        torch::Tensor selected_paths;    // [batch_size, max_depth, 2] (node, action)
    };
    
    GPUBatchEvaluator(
        std::shared_ptr<nn::NeuralNetwork> neural_net,
        const Config& config = Config()
    );
    
    ~GPUBatchEvaluator();
    
    // Get configuration
    const Config& getConfig() const { return config_; }
    
    // Submit states for batch evaluation
    std::future<std::vector<NetworkOutput>> submitBatch(
        std::vector<std::unique_ptr<core::IGameState>> states,
        const std::vector<int32_t>& node_ids = {}
    );
    
    // Batch tree operations
    void batchSelectPaths(
        const TreeTensors& tree_tensors,
        torch::Tensor& selected_paths,
        int batch_size,
        float c_puct = 1.41f
    );
    
    void batchUpdateStatistics(
        TreeTensors& tree_tensors,
        const torch::Tensor& values,
        const torch::Tensor& paths,
        int batch_size
    );
    
    // Start/stop processing
    void start();
    void stop();
    
    // Get current statistics
    struct Stats {
        std::atomic<uint64_t> total_batches{0};
        std::atomic<uint64_t> total_states{0};
        std::atomic<double> avg_batch_size{0};
        std::atomic<double> gpu_utilization{0};
        std::atomic<uint64_t> cuda_graph_hits{0};
        std::atomic<uint64_t> tensorized_operations{0};
    };
    
    const Stats& getStats() const { return stats_; }
    
    // Initialize tree tensors for a batch of trees
    TreeTensors initializeTreeTensors(int batch_size);
    
private:
    // Processing loop
    void processingLoop();
    void processBatchGPU(std::vector<BatchRequest>& batch);
    
    // Adaptive batch sizing
    int computeOptimalBatchSize();
    bool shouldProcessBatch(int current_size, int optimal_size, 
                          std::chrono::steady_clock::time_point first_timestamp);
    
    // CUDA kernel launchers
    void launchUCBKernel(
        const torch::Tensor& Q,
        const torch::Tensor& N, 
        const torch::Tensor& P,
        const torch::Tensor& N_total,
        torch::Tensor& UCB,
        float c_puct,
        cudaStream_t stream
    );
    
    void launchPathSelectionKernel(
        const torch::Tensor& UCB,
        const torch::Tensor& children_mask,
        torch::Tensor& selected_paths,
        cudaStream_t stream
    );
    
    void launchBackupKernel(
        torch::Tensor& W,
        torch::Tensor& N,
        const torch::Tensor& values,
        const torch::Tensor& paths,
        cudaStream_t stream
    );
    
    // Member variables
    Config config_;
    std::shared_ptr<nn::NeuralNetwork> neural_net_;
    std::unique_ptr<nn::GPUOptimizer> gpu_optimizer_;
    
    // Request queue
    moodycamel::ConcurrentQueue<BatchRequest> request_queue_;
    moodycamel::ProducerToken producer_token_;
    moodycamel::ConsumerToken consumer_token_;
    
    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
    
    // CUDA resources
    std::vector<cudaStream_t> cuda_streams_;
    std::atomic<size_t> current_stream_{0};
    
    // Pre-allocated tensors
    std::unique_ptr<TreeTensors> preallocated_trees_;
    
    // Statistics
    Stats stats_;
    
    // Dynamic batch sizing state
    std::atomic<int> current_optimal_batch_{128};
    std::chrono::steady_clock::time_point last_adjustment_;
};

// Global GPU batch evaluator
class GlobalGPUBatchEvaluator {
public:
    static GPUBatchEvaluator& getInstance() {
        if (!instance_) {
            throw std::runtime_error("GlobalGPUBatchEvaluator not initialized");
        }
        return *instance_;
    }
    
    static void initialize(
        std::shared_ptr<nn::NeuralNetwork> neural_net,
        const GPUBatchEvaluator::Config& config = GPUBatchEvaluator::Config()
    ) {
        if (!instance_) {
            instance_ = std::make_unique<GPUBatchEvaluator>(neural_net, config);
            instance_->start();
        }
    }
    
    static void shutdown() {
        if (instance_) {
            instance_->stop();
            instance_.reset();
        }
    }
    
private:
    static std::unique_ptr<GPUBatchEvaluator> instance_;
};

#else // !WITH_TORCH
// Dummy class when torch is not available
class GPUBatchEvaluator {
public:
    struct Config {
        size_t min_batch_size = 64;
        size_t max_batch_size = 256;
        bool enable_cuda_graphs = false;
        bool enable_adaptive_batching = false;
    };
    
    GPUBatchEvaluator(std::shared_ptr<nn::NeuralNetwork>, const Config& = {}) {}
    GPUBatchEvaluator(const Config& = {}) {}
    
    void start() {}
    void stop() {}
    
    std::future<std::vector<NetworkOutput>> submitBatch(
        std::vector<std::unique_ptr<core::IGameState>> states
    ) {
        std::promise<std::vector<NetworkOutput>> promise;
        promise.set_value(std::vector<NetworkOutput>());
        return promise.get_future();
    }
    
    static GPUBatchEvaluator& getInstance() {
        static GPUBatchEvaluator instance;
        return instance;
    }
    static void destroyInstance() {}
};
#endif // WITH_TORCH

} // namespace mcts
} // namespace alphazero