// src/core/lock_free_tensor_pool.cpp
#include "core/lock_free_tensor_pool.h"
#include <sstream>
#include <iomanip>
#include <functional>

namespace alphazero {
namespace core {

LockFreeTensorPool& LockFreeTensorPool::getInstance() {
    static LockFreeTensorPool instance;
    return instance;
}

LockFreeTensorPool::LockFreeTensorPool() {
    // Initialize all buckets
    for (size_t i = 0; i < POOL_BUCKETS; ++i) {
        buckets_[i] = std::make_unique<LockFreeStack>();
    }
}

std::unique_ptr<LockFreeTensorPool::PooledTensor> LockFreeTensorPool::getTensor(
    size_t channels, size_t height, size_t width) {
    
    size_t bucket_idx = getBucketIndex(channels, height, width);
    
    // Try to get from pool first
    if (auto tensor = buckets_[bucket_idx]->pop()) {
        stats_.pool_hits.fetch_add(1, std::memory_order_relaxed);
        
        // Clear the tensor for reuse
        for (auto& channel : *tensor) {
            for (auto& row : channel) {
                std::fill(row.begin(), row.end(), 0.0f);
            }
        }
        
        return std::make_unique<PooledTensor>(std::move(tensor), channels, height, width, bucket_idx);
    }
    
    // Pool miss - allocate new tensor
    stats_.pool_misses.fetch_add(1, std::memory_order_relaxed);
    stats_.total_allocations.fetch_add(1, std::memory_order_relaxed);
    
    auto new_tensor = std::make_shared<TensorType>(
        channels, std::vector<std::vector<float>>(
            height, std::vector<float>(width, 0.0f)));
    
    return std::make_unique<PooledTensor>(std::move(new_tensor), channels, height, width, bucket_idx);
}

void LockFreeTensorPool::returnTensorInternal(const PooledTensor& pooled_tensor) {
    if (!pooled_tensor.tensor) {
        return;  // Already returned or moved
    }
    
    stats_.returns.fetch_add(1, std::memory_order_relaxed);
    
    size_t bucket_idx = pooled_tensor.bucket_index;
    if (bucket_idx >= POOL_BUCKETS) {
        return;  // Invalid bucket
    }
    
    // Try to return to pool
    if (!buckets_[bucket_idx]->push(pooled_tensor.tensor)) {
        stats_.bucket_overflows.fetch_add(1, std::memory_order_relaxed);
        // Tensor will be deallocated when shared_ptr goes out of scope
    }
}

void LockFreeTensorPool::clear() {
    for (auto& bucket : buckets_) {
        bucket->clear();
    }
    
    // Reset statistics
    stats_.total_allocations.store(0, std::memory_order_relaxed);
    stats_.pool_hits.store(0, std::memory_order_relaxed);
    stats_.pool_misses.store(0, std::memory_order_relaxed);
    stats_.returns.store(0, std::memory_order_relaxed);
    stats_.bucket_overflows.store(0, std::memory_order_relaxed);
}

std::string LockFreeTensorPool::getDetailedStats() const {
    std::stringstream ss;
    
    ss << "=== LockFreeTensorPool Statistics ===\n";
    ss << "Total allocations: " << stats_.total_allocations.load() << "\n";
    ss << "Pool hits: " << stats_.pool_hits.load() << "\n";
    ss << "Pool misses: " << stats_.pool_misses.load() << "\n";
    ss << "Hit rate: " << std::fixed << std::setprecision(2) 
       << (stats_.getHitRate() * 100) << "%\n";
    ss << "Returns: " << stats_.returns.load() << "\n";
    ss << "Bucket overflows: " << stats_.bucket_overflows.load() << "\n\n";
    
    ss << "Bucket utilization:\n";
    size_t total_tensors = 0;
    for (size_t i = 0; i < POOL_BUCKETS; ++i) {
        size_t bucket_size = buckets_[i]->size.load(std::memory_order_relaxed);
        if (bucket_size > 0) {
            ss << "  Bucket " << i << ": " << bucket_size << " tensors\n";
            total_tensors += bucket_size;
        }
    }
    
    ss << "\nTotal pooled tensors: " << total_tensors << "\n";
    
    return ss.str();
}

size_t LockFreeTensorPool::getBucketIndex(size_t channels, size_t height, size_t width) const {
    // Simple hash function to distribute tensors across buckets
    std::hash<size_t> hasher;
    size_t hash = hasher(channels) ^ (hasher(height) << 1) ^ (hasher(width) << 2);
    return hash % POOL_BUCKETS;
}

// LockFreeStack implementation
bool LockFreeTensorPool::LockFreeStack::push(TensorPtr tensor) {
    // Check size limit
    if (size.load(std::memory_order_relaxed) >= MAX_POOL_SIZE_PER_BUCKET) {
        return false;
    }
    
    auto new_node = new StackNode(std::move(tensor));
    StackNode* expected = head.load(std::memory_order_relaxed);
    
    do {
        new_node->next.store(expected, std::memory_order_relaxed);
    } while (!head.compare_exchange_weak(expected, new_node,
                                         std::memory_order_release,
                                         std::memory_order_relaxed));
    
    size.fetch_add(1, std::memory_order_relaxed);
    return true;
}

LockFreeTensorPool::TensorPtr LockFreeTensorPool::LockFreeStack::pop() {
    StackNode* head_node = head.load(std::memory_order_acquire);
    
    while (head_node != nullptr) {
        StackNode* next = head_node->next.load(std::memory_order_relaxed);
        
        if (head.compare_exchange_weak(head_node, next,
                                       std::memory_order_release,
                                       std::memory_order_acquire)) {
            size.fetch_sub(1, std::memory_order_relaxed);
            TensorPtr tensor = std::move(head_node->tensor);
            delete head_node;
            return tensor;
        }
    }
    
    return nullptr;
}

void LockFreeTensorPool::LockFreeStack::clear() {
    StackNode* current = head.exchange(nullptr, std::memory_order_acquire);
    
    while (current != nullptr) {
        StackNode* next = current->next.load(std::memory_order_relaxed);
        delete current;
        current = next;
    }
    
    size.store(0, std::memory_order_relaxed);
}

} // namespace core
} // namespace alphazero