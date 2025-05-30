#pragma once

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <cuda_runtime.h>
#endif
#include "mcts/half_precision_utils.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH
/**
 * GPU-optimized tree storage using Structure of Arrays (SoA) layout
 * 
 * Key optimizations:
 * - Coalesced memory access patterns
 * - Compact representation to minimize memory bandwidth
 * - Aligned data structures for GPU efficiency
 * - Zero-copy between CPU and GPU where possible
 */
class GPUTreeStorage {
public:
    struct Config {
        size_t max_nodes;      // Maximum nodes per tree
        size_t max_actions;       // Maximum branching factor
        size_t max_trees;         // Maximum parallel trees
        
        // Memory layout options
        bool use_half_precision; // Use FP16 for values/priors
        bool use_unified_memory; // Use CUDA unified memory
        bool compress_indices;    // Use 16-bit indices where possible
        
        // Alignment settings
        size_t alignment;         // Byte alignment for GPU access
        
        // Sparse storage threshold
        size_t sparse_threshold;   // Use sparse storage above this
        
        // Constructor with default values
        Config() :
            max_nodes(100000),
            max_actions(512),
            max_trees(128),
            use_half_precision(true),
            use_unified_memory(false),
            compress_indices(true),
            alignment(128),
            sparse_threshold(64) {}
    };
    
    /**
     * Compact node representation for GPU
     * Uses SoA layout for better memory access patterns
     */
    struct CompactNode {
        // Node metadata (aligned)
        int32_t parent_idx;      // Parent node index (-1 for root)
        int32_t first_child_idx; // Index of first child in children array
        uint16_t num_children;   // Number of children
        uint16_t depth;          // Depth in tree
        
        // State hash for transposition detection
        uint64_t state_hash;
        
        // Padding for alignment
        uint32_t padding;
    };
    
    /**
     * Edge representation (parent action to child node)
     */
    struct CompactEdge {
        uint16_t action;         // Action that led to this child
        uint16_t visit_count;    // Visit count (compressed)
        half prior;              // Prior probability (FP16)
        half q_value;            // Q value (FP16)
    };
    
    GPUTreeStorage(const Config& config = Config());
    ~GPUTreeStorage();
    
    /**
     * Allocate storage for a batch of trees
     */
    void allocateTrees(int batch_size);
    
    /**
     * Add a node to the tree
     * Returns the node index
     */
    int32_t addNode(
        int tree_idx,
        int32_t parent_idx,
        uint64_t state_hash,
        const std::vector<float>& priors
    );
    
    /**
     * Update node statistics
     */
    void updateNode(
        int tree_idx,
        int32_t node_idx,
        int32_t child_action,
        float value,
        int visit_increment = 1
    );
    
    /**
     * Get tensors for batch operations
     */
    struct BatchTensors {
        // Node data (SoA layout)
        torch::Tensor node_metadata;     // [batch, max_nodes, 4] int32
        torch::Tensor node_hashes;       // [batch, max_nodes] uint64
        
        // Edge data (compressed)
        torch::Tensor edge_actions;      // [batch, max_edges] uint16
        torch::Tensor edge_visits;       // [batch, max_edges] uint16
        torch::Tensor edge_priors;       // [batch, max_edges] half
        torch::Tensor edge_q_values;     // [batch, max_edges] half
        
        // Sparse storage for wide nodes
        torch::Tensor sparse_indices;    // CSR column indices
        torch::Tensor sparse_ptr;        // CSR row pointers
        torch::Tensor sparse_values;     // Sparse Q values
        
        // Active node tracking
        torch::Tensor active_nodes;      // [batch, max_nodes] bool
        torch::Tensor node_counts;       // [batch] int32
    };
    
    BatchTensors getBatchTensors() const { return batch_tensors_; }
    
    /**
     * Sync between CPU and GPU
     */
    void syncToGPU(cudaStream_t stream = nullptr);
    void syncFromGPU(cudaStream_t stream = nullptr);
    
    /**
     * Launch backpropagation kernel
     */
    void launchBackpropagation(
        const torch::Tensor& paths,
        const torch::Tensor& path_lengths,
        const torch::Tensor& leaf_values,
        cudaStream_t stream = nullptr
    );
    
    /**
     * Memory statistics
     */
    struct MemoryStats {
        size_t total_allocated_bytes;
        size_t active_nodes;
        size_t sparse_nodes;
        float compression_ratio;
        float gpu_memory_usage_mb;
    };
    
    MemoryStats getMemoryStats() const;
    
    /**
     * Optimized tree traversal on GPU
     */
    void launchTreeTraversal(
        const torch::Tensor& root_indices,
        torch::Tensor& paths,
        torch::Tensor& leaf_indices,
        float c_puct,
        cudaStream_t stream = nullptr
    );
    
private:
    Config config_;
    
    // CPU-side storage (for building trees)
    std::vector<std::vector<CompactNode>> cpu_nodes_;
    std::vector<std::vector<CompactEdge>> cpu_edges_;
    std::vector<std::unordered_map<uint64_t, int32_t>> transposition_tables_;
    
    // GPU tensors
    BatchTensors batch_tensors_;
    
    // Memory pools
    uint8_t* pinned_memory_pool_;
    size_t pinned_memory_size_;
    
    // Helper methods
    void initializeTensors(int batch_size);
    void compressVisitCount(uint16_t& compressed, int full_count);
    int decompressVisitCount(uint16_t compressed);
    
    // CUDA kernel launchers
    void launchCompactKernel(cudaStream_t stream);
    void launchExpandKernel(cudaStream_t stream);
};

/**
 * GPU-optimized node pool with memory coalescing
 */
class GPUNodePool {
public:
    GPUNodePool(size_t initial_capacity = 100000);
    ~GPUNodePool();
    
    /**
     * Allocate a batch of nodes
     * Returns indices of allocated nodes
     */
    std::vector<int32_t> allocateBatch(size_t count);
    
    /**
     * Free a batch of nodes
     */
    void freeBatch(const std::vector<int32_t>& indices);
    
    /**
     * Get GPU memory pointer for direct access
     */
    void* getGPUMemory() { return d_memory_; }
    
    /**
     * Defragment memory to improve locality
     */
    void defragment();
    
private:
    // GPU memory management
    void* d_memory_;
    size_t capacity_;
    size_t allocated_;
    
    // Free list management
    std::vector<int32_t> free_list_;
    std::mutex free_list_mutex_;
    
    // Allocation statistics
    std::atomic<uint64_t> total_allocations_{0};
    std::atomic<uint64_t> total_deallocations_{0};
};

/**
 * Transposition table optimized for GPU access
 */
class GPUTranspositionTable {
public:
    GPUTranspositionTable(size_t capacity = 1000000);
    ~GPUTranspositionTable();
    
    /**
     * Batch lookup of positions
     */
    void batchLookup(
        const torch::Tensor& hashes,     // [batch_size] uint64
        torch::Tensor& found_mask,        // [batch_size] bool
        torch::Tensor& values,            // [batch_size] float
        torch::Tensor& visits,            // [batch_size] int32
        cudaStream_t stream = nullptr
    );
    
    /**
     * Batch insert/update of positions
     */
    void batchInsert(
        const torch::Tensor& hashes,      // [batch_size] uint64
        const torch::Tensor& values,      // [batch_size] float
        const torch::Tensor& visits,      // [batch_size] int32
        cudaStream_t stream = nullptr
    );
    
    /**
     * Clear table
     */
    void clear();
    
    /**
     * Get utilization statistics
     */
    float getLoadFactor() const;
    
private:
    // GPU hash table storage
    torch::Tensor keys_;         // [capacity] uint64
    torch::Tensor values_;       // [capacity] float
    torch::Tensor visits_;       // [capacity] int32
    torch::Tensor occupied_;     // [capacity] bool
    
    size_t capacity_;
    std::atomic<size_t> size_{0};
    
    // Hash function for GPU
    __device__ inline uint32_t hash(uint64_t key) const;
};

#else // !WITH_TORCH
// Dummy classes when torch is not available
class GPUTreeStorage {
public:
    struct Config {};
    GPUTreeStorage(const Config& = {}) {}
};

class GPUNodePool {
public:
    GPUNodePool(size_t = 100000) {}
};

class GPUTranspositionTable {
public:
    GPUTranspositionTable(size_t = 1000000) {}
};
#endif // WITH_TORCH

} // namespace mcts
} // namespace alphazero