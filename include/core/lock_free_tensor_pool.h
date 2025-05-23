// include/core/lock_free_tensor_pool.h
#ifndef LOCK_FREE_TENSOR_POOL_H
#define LOCK_FREE_TENSOR_POOL_H

#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include <cstddef>
#include "core/export_macros.h"

namespace alphazero {
namespace core {

/**
 * @brief Lock-free tensor pool using atomic operations for high concurrency
 * 
 * This implementation uses a lock-free stack approach with atomic operations
 * to eliminate mutex contention in high-throughput scenarios.
 */
class ALPHAZERO_API LockFreeTensorPool {
public:
    static constexpr size_t MAX_DIMENSIONS = 3;
    static constexpr size_t POOL_BUCKETS = 64;  // Number of size buckets
    static constexpr size_t MAX_POOL_SIZE_PER_BUCKET = 512;  // Max tensors per bucket
    
    using TensorType = std::vector<std::vector<std::vector<float>>>;
    using TensorPtr = std::shared_ptr<TensorType>;
    
    /**
     * @brief Thread-safe tensor wrapper with reference counting
     */
    struct PooledTensor {
        TensorPtr tensor;
        std::array<size_t, MAX_DIMENSIONS> dimensions;
        size_t bucket_index;
        
        PooledTensor() : bucket_index(0) {
            dimensions.fill(0);
        }
        
        PooledTensor(TensorPtr t, size_t channels, size_t height, size_t width, size_t bucket) 
            : tensor(std::move(t)), bucket_index(bucket) {
            dimensions[0] = channels;
            dimensions[1] = height;
            dimensions[2] = width;
        }
        
        ~PooledTensor() {
            // Automatically return to pool when destroyed
            if (tensor) {
                LockFreeTensorPool::getInstance().returnTensorInternal(*this);
            }
        }
        
        // Disable copy to ensure single ownership
        PooledTensor(const PooledTensor&) = delete;
        PooledTensor& operator=(const PooledTensor&) = delete;
        
        // Enable move semantics
        PooledTensor(PooledTensor&& other) noexcept 
            : tensor(std::move(other.tensor)), 
              dimensions(other.dimensions),
              bucket_index(other.bucket_index) {
            other.tensor = nullptr;
        }
        
        PooledTensor& operator=(PooledTensor&& other) noexcept {
            if (this != &other) {
                tensor = std::move(other.tensor);
                dimensions = other.dimensions;
                bucket_index = other.bucket_index;
                other.tensor = nullptr;
            }
            return *this;
        }
        
        TensorType& operator*() { return *tensor; }
        const TensorType& operator*() const { return *tensor; }
        TensorType* operator->() { return tensor.get(); }
        const TensorType* operator->() const { return tensor.get(); }
    };
    
    /**
     * @brief Get singleton instance
     */
    static LockFreeTensorPool& getInstance();
    
    /**
     * @brief Get a tensor from the pool or create a new one
     */
    std::unique_ptr<PooledTensor> getTensor(size_t channels, size_t height, size_t width);
    
    /**
     * @brief Get pool statistics
     */
    struct PoolStats {
        std::atomic<size_t> total_allocations{0};
        std::atomic<size_t> pool_hits{0};
        std::atomic<size_t> pool_misses{0};
        std::atomic<size_t> returns{0};
        std::atomic<size_t> bucket_overflows{0};
        
        double getHitRate() const {
            size_t hits = pool_hits.load(std::memory_order_relaxed);
            size_t total = hits + pool_misses.load(std::memory_order_relaxed);
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };
    
    const PoolStats& getStats() const { return stats_; }
    
    /**
     * @brief Clear all pooled tensors
     */
    void clear();
    
    /**
     * @brief Get detailed statistics string
     */
    std::string getDetailedStats() const;

private:
    LockFreeTensorPool();
    ~LockFreeTensorPool() = default;
    LockFreeTensorPool(const LockFreeTensorPool&) = delete;
    LockFreeTensorPool& operator=(const LockFreeTensorPool&) = delete;
    
    /**
     * @brief Lock-free stack node
     */
    struct StackNode {
        TensorPtr tensor;
        std::atomic<StackNode*> next;
        
        StackNode(TensorPtr t) : tensor(std::move(t)), next(nullptr) {}
    };
    
    /**
     * @brief Lock-free stack for each bucket
     */
    struct LockFreeStack {
        std::atomic<StackNode*> head{nullptr};
        std::atomic<size_t> size{0};
        
        bool push(TensorPtr tensor);
        TensorPtr pop();
        void clear();
        ~LockFreeStack() { clear(); }
    };
    
    /**
     * @brief Calculate bucket index from dimensions
     */
    size_t getBucketIndex(size_t channels, size_t height, size_t width) const;
    
    /**
     * @brief Internal return function called by PooledTensor destructor
     */
    friend struct PooledTensor;
    void returnTensorInternal(const PooledTensor& pooled_tensor);
    
    // Bucket array for different tensor sizes
    std::array<std::unique_ptr<LockFreeStack>, POOL_BUCKETS> buckets_;
    
    // Statistics
    mutable PoolStats stats_;
};

} // namespace core
} // namespace alphazero

#endif // LOCK_FREE_TENSOR_POOL_H