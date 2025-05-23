#include "mcts/mcts_node_pool.h"
#include "mcts/aggressive_memory_manager.h"
#include "utils/debug_logger.h"
#include <algorithm>
#include <new>
#include <omp.h>

namespace alphazero {
namespace mcts {

MCTSNodePool::MemoryBlock::MemoryBlock(size_t block_size) 
    : size(block_size),
      nodes(static_cast<MCTSNode*>(::operator new(block_size * sizeof(MCTSNode))), 
            [](MCTSNode* ptr) { ::operator delete(ptr); }) {
    // Store individual pointers for placement new later
    raw_pointers.reserve(block_size);
    MCTSNode* base = nodes.get();
    for (size_t i = 0; i < block_size; ++i) {
        raw_pointers.push_back(&base[i]);
    }
}

MCTSNodePool::MCTSNodePool(const Config& config) 
    : config_(config) {
    LOG_SYSTEM_INFO("Initializing MCTS node pool with {} initial nodes", 
                   config.initial_pool_size);
    
    // Initialize last compaction time
    last_compaction_time_.store(std::chrono::steady_clock::now());
    
    // Pre-allocate initial pool
    allocateBlock(config_.initial_pool_size);
}

MCTSNodePool::~MCTSNodePool() {
    clear();
    LOG_SYSTEM_INFO("MCTS node pool destroyed. Total allocations: {}, Total deallocations: {}",
                   allocations_.load(), deallocations_.load());
}

void MCTSNodePool::allocateBlock(size_t size) {
    PROFILE_SCOPE_N("MCTSNodePool::allocateBlock");
    
    if (total_allocated_ + size > config_.max_pool_size) {
        size = std::min(size, config_.max_pool_size - total_allocated_);
        if (size == 0) {
            LOG_SYSTEM_WARN("Node pool reached maximum size of {} nodes", 
                           config_.max_pool_size);
            return;
        }
    }
    
    auto block = std::make_unique<MemoryBlock>(size);
    
    // Add all nodes to free list efficiently using bulk operation
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Use bulk addition for better performance
    for (MCTSNode* node : block->raw_pointers) {
        free_nodes_.push(node);
    }
    
    memory_blocks_.push_back(std::move(block));
    total_allocated_ += size;
    
    LOG_SYSTEM_DEBUG("Allocated new block of {} nodes. Total allocated: {}",
                    size, total_allocated_.load());
}

std::shared_ptr<MCTSNode> MCTSNodePool::allocateNode(
    std::unique_ptr<core::IGameState> state,
    std::shared_ptr<MCTSNode> parent) {
    
    PROFILE_SCOPE_N("MCTSNodePool::allocateNode");
    
    // Get the thread-local pool for the current thread
    int thread_id = omp_get_thread_num() % MAX_THREADS;
    auto& thread_pool = thread_pools_[thread_id];
    
    MCTSNode* raw_node = nullptr;
    
    // First try to get a node from the thread-local free list
    if (!thread_pool.free_nodes.empty()) {
        // Get a node from the thread-local pool (no mutex needed)
        raw_node = thread_pool.free_nodes.back();
        thread_pool.free_nodes.pop_back();
    } else {
        // Thread-local pool is empty, try the global pool
        bool allocated_from_global = false;
        
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            
            if (!free_nodes_.empty()) {
                // Get a node from the global pool
                raw_node = free_nodes_.front();
                free_nodes_.pop();
                allocated_from_global = true;
            } else {
                // Need to allocate more nodes
                allocateBlock(config_.grow_size);
                
                if (!free_nodes_.empty()) {
                    raw_node = free_nodes_.front();
                    free_nodes_.pop();
                    allocated_from_global = true;
                } else {
                    // Failed to allocate more nodes
                    LOG_SYSTEM_ERROR("Failed to allocate more nodes from pool");
                    throw std::bad_alloc();
                }
            }
        }
        
        // If we got a node from the global pool, also refill the thread-local pool
        if (allocated_from_global) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            
            // Transfer some nodes to the thread-local pool for future use
            const size_t refill_count = 32; // Reasonable batch size to refill
            
            for (size_t i = 0; i < refill_count && !free_nodes_.empty(); ++i) {
                thread_pool.free_nodes.push_back(free_nodes_.front());
                free_nodes_.pop();
            }
        }
    }
    
    // Placement new to construct the node in the allocated memory
    new (raw_node) MCTSNode(std::move(state), parent);
    
    // Update statistics
    allocations_++;
    size_t current_in_use = ++in_use_;
    size_t current_peak = peak_usage_.load();
    while (current_in_use > current_peak && 
           !peak_usage_.compare_exchange_weak(current_peak, current_in_use)) {
        // Loop to update peak usage atomically
    }
    
    // Return shared_ptr with custom deleter
    return std::shared_ptr<MCTSNode>(raw_node, NodeDeleter(this));
}

void MCTSNodePool::freeNode(MCTSNode* node) {
    PROFILE_SCOPE_N("MCTSNodePool::freeNode");
    
    if (!node) return;
    
    // Call destructor explicitly
    node->~MCTSNode();
    
    // Get the thread-local pool for the current thread
    int thread_id = omp_get_thread_num() % MAX_THREADS;
    auto& thread_pool = thread_pools_[thread_id];
    
    const size_t MAX_THREAD_LOCAL_NODES = 128; // Limit to avoid excessive memory usage
    
    // Check if thread-local pool is getting too large
    if (thread_pool.free_nodes.size() >= MAX_THREAD_LOCAL_NODES) {
        // Thread-local pool is full, move some nodes to the global pool
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Move half of the local nodes to the global pool
        size_t half_size = thread_pool.free_nodes.size() / 2;
        for (size_t i = 0; i < half_size; ++i) {
            free_nodes_.push(thread_pool.free_nodes.back());
            thread_pool.free_nodes.pop_back();
        }
    }
    
    // Add the node to the thread-local pool
    thread_pool.free_nodes.push_back(node);
    
    in_use_--;
    deallocations_++;
}

void MCTSNodePool::NodeDeleter::operator()(MCTSNode* node) {
    if (pool_) {
        pool_->freeNode(node);
    }
}

void MCTSNodePool::clear() {
    PROFILE_SCOPE_N("MCTSNodePool::clear");
    
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Clear global free nodes queue
        while (!free_nodes_.empty()) {
            free_nodes_.pop();
        }
        
        // Clear all thread-local free lists
        for (auto& thread_pool : thread_pools_) {
            thread_pool.free_nodes.clear();
        }
        
        // Deallocate all memory blocks
        memory_blocks_.clear();
        
        // Reset statistics
        total_allocated_ = 0;
        in_use_ = 0;
    }
    
    LOG_SYSTEM_INFO("Node pool cleared (global and thread-local)");
}

MCTSNodePool::PoolStats MCTSNodePool::getStats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Count free nodes in all thread-local pools
    size_t total_free_nodes = free_nodes_.size();
    for (const auto& thread_pool : thread_pools_) {
        total_free_nodes += thread_pool.free_nodes.size();
    }
    
    return {
        total_allocated_.load(),
        in_use_.load(),
        total_free_nodes,
        peak_usage_.load(),
        allocations_.load(),
        deallocations_.load()
    };
}

MCTSNode* MCTSNodePool::Allocator::allocate(size_t n) {
    if (n != 1) {
        throw std::bad_alloc(); // Only support single node allocation
    }
    
    // Use the pool to allocate, but return raw pointer for STL compatibility
    // This is typically not used directly since we use shared_ptr
    return nullptr; // Not implemented for direct use
}

void MCTSNodePool::Allocator::deallocate(MCTSNode* ptr, size_t n) {
    if (n != 1) return;
    pool_->freeNode(ptr);
}

bool MCTSNodePool::shouldCompact() const {
    // Check if we should compact based on memory pressure and time
    auto now = std::chrono::steady_clock::now();
    auto last_compact = last_compaction_time_.load();
    auto time_since_compact = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_compact).count();
    
    // Compact if:
    // 1. Enough time has passed since last compaction
    // 2. We have significant free nodes (more than we need)
    size_t free_nodes = free_nodes_.size();
    for (const auto& thread_pool : thread_pools_) {
        free_nodes += thread_pool.free_nodes.size();
    }
    
    bool time_triggered = time_since_compact > COMPACTION_INTERVAL_MS;
    bool memory_triggered = free_nodes > MIN_FREE_NODES_AFTER_COMPACT * 2;
    
    return time_triggered && memory_triggered;
}

void MCTSNodePool::compact() {
    PROFILE_SCOPE_N("MCTSNodePool::compact");
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Update compaction time
    last_compaction_time_.store(std::chrono::steady_clock::now());
    
    // Count total free nodes across all pools
    size_t total_free = free_nodes_.size();
    for (auto& thread_pool : thread_pools_) {
        total_free += thread_pool.free_nodes.size();
    }
    
    // If we have too many free nodes, release some memory
    if (total_free > MIN_FREE_NODES_AFTER_COMPACT) {
        size_t nodes_to_release = total_free - MIN_FREE_NODES_AFTER_COMPACT;
        releaseMemory(nodes_to_release);
    }
    
    LOG_SYSTEM_INFO("Node pool compacted: {} free nodes, {} in use",
                    total_free, in_use_.load());
}

void MCTSNodePool::releaseMemory(size_t nodes_to_release) {
    PROFILE_SCOPE_N("MCTSNodePool::releaseMemory");
    
    // This is called with pool_mutex_ already locked
    
    // First, clear thread-local pools to consolidate free nodes
    for (auto& thread_pool : thread_pools_) {
        while (!thread_pool.free_nodes.empty() && nodes_to_release > 0) {
            thread_pool.free_nodes.pop_back();
            nodes_to_release--;
            total_allocated_--;
        }
    }
    
    // Then clear from global pool
    while (!free_nodes_.empty() && nodes_to_release > 0) {
        free_nodes_.pop();
        nodes_to_release--;
        total_allocated_--;
    }
    
    // Find and remove empty memory blocks
    // Note: This is a simplified approach - in practice, we'd need to track
    // which nodes belong to which blocks for proper deallocation
    auto it = memory_blocks_.begin();
    while (it != memory_blocks_.end()) {
        // Check if this block has any allocated nodes
        // For now, we'll keep all blocks to avoid complex bookkeeping
        // A more sophisticated implementation would track node-to-block mapping
        ++it;
    }
    
    LOG_SYSTEM_DEBUG("Released memory for {} nodes, total allocated: {}",
                    nodes_to_release, total_allocated_.load());
}

} // namespace mcts
} // namespace alphazero