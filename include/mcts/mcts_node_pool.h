#ifndef ALPHAZERO_MCTS_NODE_POOL_H
#define ALPHAZERO_MCTS_NODE_POOL_H

#include <memory>
#include <queue>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include "mcts/mcts_node.h"
#include "utils/memory_allocator.h"
#include "utils/thread_local_allocator.h"
#include "utils/logger.h"
#include "utils/profiler.h"

namespace alphazero {
namespace mcts {

/**
 * Memory pool allocator for MCTS nodes
 * 
 * Benefits:
 * - Reduces allocation overhead
 * - Improves cache locality
 * - Prevents memory fragmentation
 * - Enables better memory reuse
 */
class MCTSNodePool {
public:
    // Configuration for the pool
    struct Config {
        size_t initial_pool_size;
        size_t grow_size;
        size_t max_pool_size;
        bool enable_memory_reuse;
        
        Config() 
            : initial_pool_size(100000)  // Initial nodes to pre-allocate
            , grow_size(50000)          // Nodes to allocate when growing
            , max_pool_size(10000000)   // Maximum nodes to keep in pool
            , enable_memory_reuse(true) // Whether to reuse freed nodes
        {}
    };
    
    MCTSNodePool(const Config& config = Config());
    ~MCTSNodePool();
    
    // Custom allocator interface
    class Allocator {
    public:
        using value_type = MCTSNode;
        
        Allocator(MCTSNodePool* pool) : pool_(pool) {}
        
        MCTSNode* allocate(size_t n);
        void deallocate(MCTSNode* ptr, size_t n);
        
        template<typename U>
        struct rebind {
            using other = Allocator;
        };
        
    private:
        MCTSNodePool* pool_;
    };
    
    // Allocate a new node
    std::shared_ptr<MCTSNode> allocateNode(
        std::unique_ptr<core::IGameState> state,
        std::shared_ptr<MCTSNode> parent = nullptr);
    
    // Free a node (return to pool)
    void freeNode(MCTSNode* node);
    
    // Clear all nodes in the pool
    void clear();
    
    // Memory management
    void compact();  // Release unused memory blocks
    void releaseMemory(size_t target_free_nodes);  // Release memory to reach target
    bool shouldCompact() const;  // Check if compaction is needed
    
    // Get pool statistics
    struct PoolStats {
        size_t total_allocated;
        size_t in_use;
        size_t available;
        size_t peak_usage;
        size_t allocations;
        size_t deallocations;
    };
    
    PoolStats getStats() const;
    
private:
    // Memory blocks for nodes
    struct MemoryBlock {
        std::unique_ptr<MCTSNode[], std::function<void(MCTSNode*)>> nodes;
        std::vector<MCTSNode*> raw_pointers;  // Individual pointers for efficient access
        size_t size;
        
        MemoryBlock(size_t block_size);
    };
    
    // Pool configuration
    Config config_;
    
    // Memory management
    std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
    std::queue<MCTSNode*> free_nodes_; // Global free list (used less frequently)
    mutable std::mutex pool_mutex_;    // Global mutex (used less frequently)
    
    // Thread-local free lists to reduce contention
    static constexpr int MAX_THREADS = 64;
    struct ThreadLocalPool {
        std::vector<MCTSNode*> free_nodes;
        alignas(64) char padding[64]; // Avoid false sharing between threads
    };
    std::array<ThreadLocalPool, MAX_THREADS> thread_pools_;
    
    // Statistics
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> in_use_{0};
    std::atomic<size_t> peak_usage_{0};
    std::atomic<size_t> allocations_{0};
    std::atomic<size_t> deallocations_{0};
    
    // Memory pressure handling
    mutable std::atomic<std::chrono::steady_clock::time_point> last_compaction_time_;
    static constexpr size_t COMPACTION_INTERVAL_MS = 30000;  // 30 seconds
    static constexpr double MEMORY_PRESSURE_THRESHOLD = 0.8;  // 80% of max
    static constexpr size_t MIN_FREE_NODES_AFTER_COMPACT = 10000;  // Keep at least this many free
    
    // Allocate a new memory block
    void allocateBlock(size_t size);
    
    // Custom deleter for shared_ptr that returns nodes to pool
    class NodeDeleter {
    public:
        NodeDeleter(MCTSNodePool* pool) : pool_(pool) {}
        void operator()(MCTSNode* node);
        
    private:
        MCTSNodePool* pool_;
    };
};

// Global node pool instance (optional, can create per-engine)
inline MCTSNodePool& getGlobalNodePool() {
    static MCTSNodePool global_pool;
    return global_pool;
}

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_NODE_POOL_H