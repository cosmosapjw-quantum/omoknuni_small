#pragma once

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <cuda_runtime.h>
#endif
#include <atomic>
#include <vector>
#include <memory>

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH
/**
 * GPU-optimized virtual loss management
 * 
 * Key features:
 * - Batch virtual loss application/removal
 * - GPU tensor-based tracking
 * - Lock-free atomic operations
 * - Adaptive virtual loss values
 */
class GPUVirtualLoss {
public:
    struct Config {
        int base_virtual_loss;        // Base virtual loss value
        float adaptive_factor;     // Factor for adaptive adjustment
        bool enable_adaptive;      // Enable adaptive virtual loss
        int max_virtual_loss;        // Maximum virtual loss value
        bool use_gpu_atomics;      // Use GPU atomic operations
        
        // Constructor with default values
        Config() :
            base_virtual_loss(3),
            adaptive_factor(1.5f),
            enable_adaptive(true),
            max_virtual_loss(10),
            use_gpu_atomics(true) {}
    };
    
    GPUVirtualLoss(const Config& config = Config());
    ~GPUVirtualLoss();
    
    /**
     * Apply virtual loss to multiple paths in batch
     * @param paths Tensor of paths [batch_size, max_depth, 2] (node_idx, action_idx)
     * @param virtual_loss_tensor Tensor tracking virtual losses [batch, nodes, actions]
     * @param path_lengths Length of each path
     */
    void batchApplyVirtualLoss(
        const torch::Tensor& paths,
        torch::Tensor& virtual_loss_tensor,
        const std::vector<int>& path_lengths,
        cudaStream_t stream = nullptr
    );
    
    /**
     * Remove virtual loss from multiple paths in batch
     */
    void batchRemoveVirtualLoss(
        const torch::Tensor& paths,
        torch::Tensor& virtual_loss_tensor,
        const std::vector<int>& path_lengths,
        cudaStream_t stream = nullptr
    );
    
    /**
     * Compute adaptive virtual loss based on node statistics
     * Higher virtual loss for more visited nodes to encourage exploration
     */
    int computeAdaptiveVirtualLoss(
        int node_visits,
        int parent_visits,
        float prior_probability
    ) const;
    
    /**
     * Apply virtual loss during UCB calculation
     * Modifies Q values to discourage selection of nodes being evaluated
     */
    void applyVirtualLossToUCB(
        torch::Tensor& Q_tensor,
        const torch::Tensor& virtual_loss_tensor,
        float virtual_loss_weight = 1.0f
    );
    
    /**
     * Get statistics about virtual loss usage
     */
    struct Stats {
        std::atomic<uint64_t> total_applications{0};
        std::atomic<uint64_t> total_removals{0};
        std::atomic<uint64_t> max_concurrent_vl{0};
        std::atomic<double> avg_vl_per_node{0};
    };
    
    const Stats& getStats() const { return stats_; }
    
private:
    Config config_;
    Stats stats_;
    
    // Pre-allocated tensors for efficiency
    torch::Tensor adaptive_vl_factors_;
    
    // CUDA kernel launchers
    void launchApplyVirtualLossKernel(
        const torch::Tensor& paths,
        torch::Tensor& virtual_loss_tensor,
        const int* path_lengths,
        int batch_size,
        int max_depth,
        int virtual_loss_value,
        cudaStream_t stream
    );
    
    void launchRemoveVirtualLossKernel(
        const torch::Tensor& paths,
        torch::Tensor& virtual_loss_tensor,
        const int* path_lengths,
        int batch_size,
        int max_depth,
        cudaStream_t stream
    );
};

/**
 * Thread-safe virtual loss tracker for CPU-GPU coordination
 */
class HybridVirtualLossTracker {
public:
    HybridVirtualLossTracker(size_t max_nodes = 100000);
    
    /**
     * Apply virtual loss to a node (CPU side)
     */
    void applyVirtualLoss(size_t node_id, int amount = 1);
    
    /**
     * Remove virtual loss from a node (CPU side)
     */
    void removeVirtualLoss(size_t node_id, int amount = 1);
    
    /**
     * Get current virtual loss for a node
     */
    int getVirtualLoss(size_t node_id) const;
    
    /**
     * Sync virtual losses to GPU tensor
     */
    void syncToGPU(torch::Tensor& gpu_virtual_loss_tensor);
    
    /**
     * Sync virtual losses from GPU tensor
     */
    void syncFromGPU(const torch::Tensor& gpu_virtual_loss_tensor);
    
    /**
     * Clear all virtual losses (for new search)
     */
    void clear();
    
private:
    // Atomic counters for each node
    std::unique_ptr<std::atomic<int>[]> virtual_losses_;
    size_t max_nodes_;
    mutable std::atomic<uint64_t> total_applications_{0};
    mutable std::atomic<uint64_t> total_removals_{0};
};

#else // !WITH_TORCH
// Dummy classes when torch is not available
class GPUVirtualLoss {
public:
    struct Config {};
    GPUVirtualLoss(const Config& = {}) {}
};

class HybridVirtualLossTracker {
public:
    HybridVirtualLossTracker(size_t = 100000) {}
    void applyVirtualLoss(size_t, int = 1) {}
    void removeVirtualLoss(size_t, int = 1) {}
    int getVirtualLoss(size_t) const { return 0; }
    void clear() {}
};
#endif // WITH_TORCH

} // namespace mcts
} // namespace alphazero