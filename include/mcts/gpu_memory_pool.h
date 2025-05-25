// include/mcts/gpu_memory_pool.h
#ifndef ALPHAZERO_GPU_MEMORY_POOL_H
#define ALPHAZERO_GPU_MEMORY_POOL_H

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include "alphazero_export.h"
#include "utils/logger.h"

namespace alphazero {
namespace mcts {

/**
 * @brief GPU Memory Pool for zero-copy tensor operations
 * 
 * This class provides a high-performance memory pool for GPU tensors,
 * eliminating allocation overhead and enabling zero-copy operations.
 * 
 * Features:
 * - Pre-allocated GPU memory blocks
 * - Lock-free allocation for common sizes
 * - Automatic memory reclamation
 * - CUDA stream-aware operations
 * - Pinned memory for CPU-GPU transfers
 */
class ALPHAZERO_API GPUMemoryPool {
public:
    struct MemoryBlock {
        void* device_ptr;
        void* host_pinned_ptr;
        size_t size;
        std::atomic<bool> in_use{false};
        cudaStream_t last_stream{nullptr};
        int device_id;
    };

    struct PoolConfig {
        size_t initial_pool_size_mb;
        size_t max_pool_size_mb;
        size_t pinned_memory_size_mb;
        std::vector<size_t> block_sizes;
        bool enable_peer_access;
        bool use_unified_memory;
        
        // Constructor with defaults
        PoolConfig() : 
            initial_pool_size_mb(512),
            max_pool_size_mb(2048),
            pinned_memory_size_mb(128),
            enable_peer_access(true),
            use_unified_memory(false) {
            // Initialize block sizes
            block_sizes = {
                1 * 1024 * 1024,    // 1MB blocks
                4 * 1024 * 1024,    // 4MB blocks
                16 * 1024 * 1024,   // 16MB blocks
                64 * 1024 * 1024    // 64MB blocks
            };
        }
    };

    explicit GPUMemoryPool(const PoolConfig& config = PoolConfig());
    ~GPUMemoryPool();

    // Allocate a tensor from the pool
    torch::Tensor allocateTensor(
        const std::vector<int64_t>& shape,
        torch::ScalarType dtype,
        int device_id,
        cudaStream_t stream = nullptr
    );

    // Allocate raw memory block
    std::shared_ptr<MemoryBlock> allocateBlock(
        size_t size,
        int device_id,
        cudaStream_t stream = nullptr
    );

    // Create tensor from existing memory (zero-copy)
    torch::Tensor tensorFromMemory(
        void* data,
        const std::vector<int64_t>& shape,
        torch::ScalarType dtype,
        int device_id
    );

    // Async copy with pinned memory
    void asyncCopyToDevice(
        const void* host_data,
        void* device_data,
        size_t size,
        cudaStream_t stream
    );

    // Get pool statistics
    struct PoolStats {
        size_t total_allocated;
        size_t total_in_use;
        size_t num_allocations;
        size_t num_reuses;
        float fragmentation_ratio;
        std::unordered_map<size_t, size_t> block_usage;
    };
    PoolStats getStats() const;

    // Defragment the pool
    void defragment();

    // Release unused memory back to system
    void trim(float keep_ratio = 0.8f);

private:
    struct BlockPool {
        size_t block_size;
        std::vector<std::unique_ptr<MemoryBlock>> blocks;
        std::queue<MemoryBlock*> free_blocks;
        mutable std::mutex mutex;
        std::atomic<size_t> allocations{0};
        std::atomic<size_t> reuses{0};
        
        // Delete copy constructor and assignment
        BlockPool(const BlockPool&) = delete;
        BlockPool& operator=(const BlockPool&) = delete;
        
        // Default constructor
        BlockPool() : block_size(0) {}
        
        // Move constructor
        BlockPool(BlockPool&& other) noexcept 
            : block_size(other.block_size),
              blocks(std::move(other.blocks)),
              free_blocks(std::move(other.free_blocks)),
              allocations(other.allocations.load()),
              reuses(other.reuses.load()) {}
    };

    PoolConfig config_;
    std::vector<BlockPool> block_pools_;
    std::unordered_map<size_t, size_t> size_to_pool_index_;
    
    // Large allocation fallback
    struct LargeAllocation {
        std::unique_ptr<MemoryBlock> block;
        std::chrono::steady_clock::time_point last_used;
    };
    std::vector<LargeAllocation> large_allocations_;
    mutable std::mutex large_alloc_mutex_;

    // Pinned memory management
    struct PinnedMemoryPool {
        std::vector<void*> pinned_buffers;
        std::queue<void*> free_buffers;
        size_t buffer_size;
        mutable std::mutex mutex;
    };
    PinnedMemoryPool pinned_pool_;

    // Statistics
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_in_use_{0};

    // Initialize pools
    void initializePools();
    void allocateBlocksForPool(BlockPool& pool, size_t count, int device_id);
    size_t findBestPoolIndex(size_t requested_size) const;
    
    // Memory management
    void* allocateDeviceMemory(size_t size, int device_id);
    void freeDeviceMemory(void* ptr, int device_id);
    void* allocatePinnedMemory(size_t size);
    void freePinnedMemory(void* ptr);
};

/**
 * @brief Global GPU memory pool instance
 */
class ALPHAZERO_API GPUMemoryPoolManager {
public:
    static GPUMemoryPool& getInstance() {
        static GPUMemoryPool instance;
        return instance;
    }

    static void initialize(const GPUMemoryPool::PoolConfig& config) {
        static std::once_flag init_flag;
        std::call_once(init_flag, [&config]() {
            // Force reconstruction with new config
            getInstance().~GPUMemoryPool();
            new (&getInstance()) GPUMemoryPool(config);
        });
    }
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_GPU_MEMORY_POOL_H