#ifndef ALPHAZERO_MCTS_OBJECT_POOL_H
#define ALPHAZERO_MCTS_OBJECT_POOL_H

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <stack>
#include <type_traits>
#include <iomanip>
#include <cstdlib>
#include "core/export_macros.h"
#include "mcts/mcts_engine.h"

#include <moodycamel/concurrentqueue.h>

// SFINAE helper for detecting reset method
template<typename T, typename = void>
struct has_reset : std::false_type {};

template<typename T>
struct has_reset<T, std::void_t<decltype(std::declval<T>().reset())>> : std::true_type {};

template<typename T>
constexpr bool has_reset_v = has_reset<T>::value;

namespace alphazero {
namespace mcts {

// Forward declaration to allow incomplete types
template <typename T>
struct ObjectPoolTraits {
    static constexpr size_t size = 64;      // Default conservative size
    static constexpr size_t alignment = 8;  // Default conservative alignment
};

/**
 * @brief Lock-free object pool for MCTS-related objects
 * 
 * This implementation addresses memory allocation hotspots identified in mcts_fix.md:
 * 1. Reduces frequent allocation/deallocation of PendingEvaluation objects
 * 2. Improves memory locality and cache hit rates
 * 3. Uses lock-free operations for thread safety
 * 4. Pre-allocates objects to avoid runtime allocation overhead
 * 5. Avoids sizeof/alignof on incomplete types
 */
template <typename T>
class ALPHAZERO_API LockFreeObjectPool {
private:
    // Lock-free queue for available objects
    moodycamel::ConcurrentQueue<T*> available_objects_;
    
    // Storage for pre-allocated objects (never deleted)
    std::vector<void*> object_storage_;
    
    // Configuration
    size_t initial_pool_size_;
    size_t max_pool_size_;
    size_t object_size_;
    size_t object_alignment_;
    
    // Statistics
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> pool_hits_{0};
    std::atomic<size_t> pool_misses_{0};
    std::atomic<size_t> current_pool_size_{0};
    
    // Mutex for expanding the pool (rare operation)
    std::mutex expansion_mutex_;
    
public:
    /**
     * @brief Construct object pool
     * 
     * @param initial_size Initial number of objects to pre-allocate
     * @param max_size Maximum pool size (0 = unlimited)
     * @param obj_size Size of objects (auto-detected if possible)
     * @param obj_alignment Alignment of objects (auto-detected if possible)
     */
    LockFreeObjectPool(size_t initial_size = 256, size_t max_size = 1024,
                       size_t obj_size = 0, size_t obj_alignment = 0)
        : initial_pool_size_(initial_size), max_pool_size_(max_size) {
        
        // Use provided sizes or traits defaults
        object_size_ = (obj_size > 0) ? obj_size : ObjectPoolTraits<T>::size;
        object_alignment_ = (obj_alignment > 0) ? obj_alignment : ObjectPoolTraits<T>::alignment;
        
        // Pre-allocate initial objects only if size is known
        if (obj_size > 0) {
            expandPool(initial_pool_size_);
        }
        
        std::cout << "🏊 LockFreeObjectPool initialized with " << initial_pool_size_ 
                  << " objects of size " << object_size_ << std::endl;
    }
    
    ~LockFreeObjectPool() {
        // Free all allocated memory
        for (void* memory : object_storage_) {
            std::free(memory);
        }
        std::cout << "🏊 LockFreeObjectPool destroyed. Final stats - hits: " << pool_hits_.load()
                  << ", misses: " << pool_misses_.load()
                  << ", hit rate: " << getHitRate() << "%" << std::endl;
    }
    
    /**
     * @brief Acquire an object from the pool
     * 
     * @return Pointer to object (caller is responsible for returning it)
     */
    T* acquire() {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        
        T* obj = nullptr;
        if (available_objects_.try_dequeue(obj)) {
            // Pool hit - got object from pool
            pool_hits_.fetch_add(1, std::memory_order_relaxed);
            
            // Reset object to default state if it has a reset method
            // Use SFINAE instead of requires (C++20) for C++17 compatibility
            if constexpr (has_reset_v<T>) {
                obj->reset();
            }
            
            return obj;
        }
        
        // Pool miss - need to create new object or expand pool
        pool_misses_.fetch_add(1, std::memory_order_relaxed);
        
        // Try to expand pool if under limit
        if (max_pool_size_ == 0 || current_pool_size_.load() < max_pool_size_) {
            if (tryExpandPool(64)) { // Expand in chunks of 64
                // Try again after expansion
                if (available_objects_.try_dequeue(obj)) {
                    pool_hits_.fetch_add(1, std::memory_order_relaxed);
                    
                    if constexpr (has_reset_v<T>) {
                        obj->reset();
                    }
                    
                    return obj;
                }
            }
        }
        
        // Last resort - allocate raw memory using stored size/alignment
        void* raw_memory = std::aligned_alloc(object_alignment_, object_size_);
        if (!raw_memory) {
            std::cerr << "❌ Failed to allocate object in LockFreeObjectPool" << std::endl;
            return nullptr;
        }
        return reinterpret_cast<T*>(raw_memory);
    }
    
    /**
     * @brief Return an object to the pool
     * 
     * @param obj Object to return (must have been obtained from this pool)
     */
    void release(T* obj) {
        if (!obj) return;
        
        // Try to return to pool
        if (!available_objects_.enqueue(obj)) {
            // Pool queue is full - just delete the object
            delete obj;
        }
    }
    
    /**
     * @brief RAII wrapper for automatic object management
     */
    class PooledObject {
    private:
        LockFreeObjectPool* pool_;
        T* object_;
        
    public:
        PooledObject(LockFreeObjectPool* pool) : pool_(pool), object_(pool->acquire()) {}
        
        ~PooledObject() {
            if (pool_ && object_) {
                pool_->release(object_);
            }
        }
        
        // Move constructor
        PooledObject(PooledObject&& other) noexcept 
            : pool_(other.pool_), object_(other.object_) {
            other.pool_ = nullptr;
            other.object_ = nullptr;
        }
        
        // Move assignment
        PooledObject& operator=(PooledObject&& other) noexcept {
            if (this != &other) {
                if (pool_ && object_) {
                    pool_->release(object_);
                }
                pool_ = other.pool_;
                object_ = other.object_;
                other.pool_ = nullptr;
                other.object_ = nullptr;
            }
            return *this;
        }
        
        // Delete copy operations
        PooledObject(const PooledObject&) = delete;
        PooledObject& operator=(const PooledObject&) = delete;
        
        T* get() const { return object_; }
        T* operator->() const { return object_; }
        T& operator*() const { return *object_; }
        bool valid() const { return object_ != nullptr; }
    };
    
    /**
     * @brief Create a RAII-managed pooled object
     */
    PooledObject createPooledObject() {
        return PooledObject(this);
    }
    
    /**
     * @brief Get current pool statistics
     */
    struct PoolStats {
        size_t total_allocations;
        size_t pool_hits;
        size_t pool_misses;
        size_t current_pool_size;
        float hit_rate;
    };
    
    PoolStats getStats() const {
        PoolStats stats;
        stats.total_allocations = total_allocations_.load();
        stats.pool_hits = pool_hits_.load();
        stats.pool_misses = pool_misses_.load();
        stats.current_pool_size = current_pool_size_.load();
        stats.hit_rate = getHitRate();
        return stats;
    }
    
    float getHitRate() const {
        size_t total = total_allocations_.load();
        if (total == 0) return 0.0f;
        return (static_cast<float>(pool_hits_.load()) / total) * 100.0f;
    }
    
private:
    void expandPool(size_t count) {
        std::lock_guard<std::mutex> lock(expansion_mutex_);
        
        size_t start_index = object_storage_.size();
        object_storage_.reserve(start_index + count);
        
        for (size_t i = 0; i < count; ++i) {
            try {
                // Allocate raw memory using stored size/alignment
                void* raw_memory = std::aligned_alloc(object_alignment_, object_size_);
                if (!raw_memory) {
                    break;
                }
                T* raw_ptr = reinterpret_cast<T*>(raw_memory);
                object_storage_.push_back(raw_memory);
                
                if (!available_objects_.enqueue(raw_ptr)) {
                    // Queue full - break early
                    break;
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Failed to expand object pool: " << e.what() << std::endl;
                break;
            }
        }
        
        current_pool_size_.store(object_storage_.size(), std::memory_order_release);
    }
    
    bool tryExpandPool(size_t count) {
        // Try to acquire expansion lock without blocking
        std::unique_lock<std::mutex> lock(expansion_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return false; // Another thread is expanding
        }
        
        // Check if we're under the limit
        if (max_pool_size_ > 0 && current_pool_size_.load() >= max_pool_size_) {
            return false;
        }
        
        size_t actual_count = count;
        if (max_pool_size_ > 0) {
            size_t current_size = current_pool_size_.load();
            actual_count = std::min(count, max_pool_size_ - current_size);
        }
        
        if (actual_count == 0) {
            return false;
        }
        
        expandPool(actual_count);
        return true;
    }
};

// Specialized pools for MCTS objects
using PendingEvaluationPool = LockFreeObjectPool<PendingEvaluation>;

/**
 * @brief Global object pool manager for MCTS components
 */
class ALPHAZERO_API MCTSObjectPoolManager {
private:
    static MCTSObjectPoolManager instance_;
    
    std::unique_ptr<PendingEvaluationPool> pending_eval_pool_;
    std::unique_ptr<LockFreeObjectPool<MCTSNode>> node_pool_;
    
    MCTSObjectPoolManager() {
        pending_eval_pool_ = std::make_unique<PendingEvaluationPool>(512, 2048);
        // Note: node_pool_ is created lazily when MCTSNode is complete
    }
    
public:
    static MCTSObjectPoolManager& getInstance() {
        return instance_;
    }
    
    PendingEvaluationPool& getPendingEvaluationPool() {
        return *pending_eval_pool_;
    }
    
    // Create node pool with explicit size when MCTSNode is complete
    LockFreeObjectPool<MCTSNode>& getNodePool() {
        if (!node_pool_) {
            // Lazy initialization with explicit sizes to avoid incomplete type issues
            node_pool_ = std::make_unique<LockFreeObjectPool<MCTSNode>>(1000, 2000, 256, 16);
        }
        return *node_pool_;
    }
    
    void printStatistics() {
        auto stats = pending_eval_pool_->getStats();
        std::cout << "📊 [OBJECT_POOLS] PendingEvaluation pool - " 
                  << "allocations: " << stats.total_allocations
                  << ", hit rate: " << std::fixed << std::setprecision(1) << stats.hit_rate << "%"
                  << ", pool size: " << stats.current_pool_size
                  << std::endl;
    }
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_OBJECT_POOL_H