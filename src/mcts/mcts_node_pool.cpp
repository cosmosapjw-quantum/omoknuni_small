#include "mcts/mcts_node_pool.h"
#include <algorithm>
#include <new>

namespace alphazero {
namespace mcts {

MCTSNodePool::MemoryBlock::MemoryBlock(size_t block_size) 
    : size(block_size) {
    // Allocate raw memory for nodes using mimalloc if available
#ifdef USE_MIMALLOC
    void* raw_memory = alphazero::memory::aligned_alloc(
        alignof(MCTSNode), block_size * sizeof(MCTSNode));
    nodes.reset(static_cast<MCTSNode*>(raw_memory));
#else
    nodes = std::make_unique<MCTSNode[]>(block_size);
#endif
}

MCTSNodePool::MCTSNodePool(const Config& config) 
    : config_(config) {
    LOG_SYSTEM_INFO("Initializing MCTS node pool with {} initial nodes", 
                   config.initial_pool_size);
    
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
    
    // Add all nodes to free list
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (size_t i = 0; i < size; ++i) {
        free_nodes_.push(&block->nodes[i]);
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
    
    MCTSNode* raw_node = nullptr;
    
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (free_nodes_.empty()) {
            // Need to allocate more nodes
            allocateBlock(config_.grow_size);
            
            if (free_nodes_.empty()) {
                // Failed to allocate more nodes
                LOG_SYSTEM_ERROR("Failed to allocate more nodes from pool");
                throw std::bad_alloc();
            }
        }
        
        raw_node = free_nodes_.front();
        free_nodes_.pop();
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
    
    // Return to pool
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        free_nodes_.push(node);
    }
    
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
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Clear free nodes queue
    while (!free_nodes_.empty()) {
        free_nodes_.pop();
    }
    
    // Deallocate all memory blocks
    memory_blocks_.clear();
    
    // Reset statistics
    total_allocated_ = 0;
    in_use_ = 0;
    
    LOG_SYSTEM_INFO("Node pool cleared");
}

MCTSNodePool::PoolStats MCTSNodePool::getStats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    return {
        total_allocated_.load(),
        in_use_.load(),
        free_nodes_.size(),
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

} // namespace mcts
} // namespace alphazero