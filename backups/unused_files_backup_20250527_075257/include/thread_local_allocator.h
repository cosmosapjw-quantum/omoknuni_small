#ifndef ALPHAZERO_THREAD_LOCAL_ALLOCATOR_H
#define ALPHAZERO_THREAD_LOCAL_ALLOCATOR_H

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <cstddef>
#include <mimalloc.h>
#include <iostream>

namespace alphazero {
namespace utils {

// Thread-local memory pool for efficient allocation in OpenMP environments
template<typename T>
class ThreadLocalAllocator {
private:
    static constexpr size_t CACHE_SIZE_PER_THREAD = 128;
    static constexpr size_t MAX_THREADS = 256;
    static constexpr size_t BULK_ALLOC_SIZE = 64;
    
    struct ThreadCache {
        std::vector<T*> free_list;
        std::vector<T*> allocated;
        size_t hits = 0;
        size_t misses = 0;
        
        ThreadCache() {
            free_list.reserve(CACHE_SIZE_PER_THREAD);
            allocated.reserve(CACHE_SIZE_PER_THREAD);
        }
    };
    
    // Thread-local storage for each thread's cache
    std::vector<ThreadCache> thread_caches_;
    
    // Global pool for overflow
    std::vector<T*> global_pool_;
    mutable std::mutex global_mutex_;
    
    // Statistics
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_deallocated_{0};
    std::atomic<size_t> active_objects_{0};
    
    // Memory alignment for cache efficiency
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
public:
    ThreadLocalAllocator() : thread_caches_(MAX_THREADS) {
        global_pool_.reserve(1024);
        
        // Pre-allocate some objects
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& cache = thread_caches_[tid];
            
            // Pre-allocate objects for this thread
            for (size_t i = 0; i < CACHE_SIZE_PER_THREAD / 2; ++i) {
                T* obj = static_cast<T*>(mi_aligned_alloc(
                    std::max(alignof(T), CACHE_LINE_SIZE), sizeof(T)));
                cache.free_list.push_back(obj);
            }
        }
    }
    
    ~ThreadLocalAllocator() {
        // Clean up all allocated memory
        for (auto& cache : thread_caches_) {
            for (auto ptr : cache.free_list) {
                mi_free(ptr);
            }
            for (auto ptr : cache.allocated) {
                mi_free(ptr);
            }
        }
        
        std::lock_guard<std::mutex> lock(global_mutex_);
        for (auto ptr : global_pool_) {
            mi_free(ptr);
        }
    }
    
    T* allocate() {
        int tid = omp_get_thread_num();
        if (tid >= MAX_THREADS) tid = 0; // Fallback for safety
        
        auto& cache = thread_caches_[tid];
        
        // Try thread-local cache first
        if (!cache.free_list.empty()) {
            T* obj = cache.free_list.back();
            cache.free_list.pop_back();
            cache.allocated.push_back(obj);
            cache.hits++;
            active_objects_.fetch_add(1, std::memory_order_relaxed);
            return obj;
        }
        
        cache.misses++;
        
        // Try global pool
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (!global_pool_.empty()) {
                // Grab multiple objects at once to reduce contention
                size_t count = std::min(BULK_ALLOC_SIZE, global_pool_.size());
                for (size_t i = 0; i < count; ++i) {
                    cache.free_list.push_back(global_pool_.back());
                    global_pool_.pop_back();
                }
                
                if (!cache.free_list.empty()) {
                    T* obj = cache.free_list.back();
                    cache.free_list.pop_back();
                    cache.allocated.push_back(obj);
                    active_objects_.fetch_add(1, std::memory_order_relaxed);
                    return obj;
                }
            }
        }
        
        // Allocate new objects in bulk
        std::vector<T*> new_objects;
        new_objects.reserve(BULK_ALLOC_SIZE);
        
        for (size_t i = 0; i < BULK_ALLOC_SIZE; ++i) {
            T* obj = static_cast<T*>(mi_aligned_alloc(
                std::max(alignof(T), CACHE_LINE_SIZE), sizeof(T)));
            if (obj) {
                new_objects.push_back(obj);
                total_allocated_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        
        if (new_objects.empty()) {
            throw std::bad_alloc();
        }
        
        // Keep one for return, cache the rest
        T* result = new_objects.back();
        new_objects.pop_back();
        
        for (auto obj : new_objects) {
            cache.free_list.push_back(obj);
        }
        
        cache.allocated.push_back(result);
        active_objects_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        int tid = omp_get_thread_num();
        if (tid >= MAX_THREADS) tid = 0;
        
        auto& cache = thread_caches_[tid];
        
        // Remove from allocated list (for debugging/tracking)
        auto it = std::find(cache.allocated.begin(), cache.allocated.end(), ptr);
        if (it != cache.allocated.end()) {
            cache.allocated.erase(it);
        }
        
        // Add to free list if space available
        if (cache.free_list.size() < CACHE_SIZE_PER_THREAD) {
            cache.free_list.push_back(ptr);
        } else {
            // Overflow to global pool
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_pool_.push_back(ptr);
        }
        
        active_objects_.fetch_sub(1, std::memory_order_relaxed);
        total_deallocated_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Bulk operations for efficiency
    void allocate_bulk(std::vector<T*>& out, size_t count) {
        out.clear();
        out.reserve(count);
        
        int tid = omp_get_thread_num();
        if (tid >= MAX_THREADS) tid = 0;
        
        auto& cache = thread_caches_[tid];
        
        // Get from cache first
        size_t from_cache = std::min(count, cache.free_list.size());
        for (size_t i = 0; i < from_cache; ++i) {
            out.push_back(cache.free_list.back());
            cache.free_list.pop_back();
            cache.hits++;
        }
        
        // Get remaining from new allocations
        for (size_t i = from_cache; i < count; ++i) {
            T* obj = allocate();
            out.push_back(obj);
        }
        
        active_objects_.fetch_add(count, std::memory_order_relaxed);
    }
    
    void deallocate_bulk(const std::vector<T*>& ptrs) {
        int tid = omp_get_thread_num();
        if (tid >= MAX_THREADS) tid = 0;
        
        auto& cache = thread_caches_[tid];
        std::vector<T*> overflow;
        
        for (T* ptr : ptrs) {
            if (cache.free_list.size() < CACHE_SIZE_PER_THREAD) {
                cache.free_list.push_back(ptr);
            } else {
                overflow.push_back(ptr);
            }
        }
        
        if (!overflow.empty()) {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_pool_.insert(global_pool_.end(), overflow.begin(), overflow.end());
        }
        
        active_objects_.fetch_sub(ptrs.size(), std::memory_order_relaxed);
        total_deallocated_.fetch_add(ptrs.size(), std::memory_order_relaxed);
    }
    
    // Get statistics
    void get_stats(size_t& allocated, size_t& deallocated, size_t& active) const {
        allocated = total_allocated_.load(std::memory_order_relaxed);
        deallocated = total_deallocated_.load(std::memory_order_relaxed);
        active = active_objects_.load(std::memory_order_relaxed);
    }
    
    void print_stats() const {
        size_t allocated, deallocated, active;
        get_stats(allocated, deallocated, active);
        
        std::cout << "[Memory Stats] Allocated: " << allocated 
                  << ", Deallocated: " << deallocated
                  << ", Active: " << active << std::endl;
        
        size_t total_hits = 0, total_misses = 0;
        for (const auto& cache : thread_caches_) {
            total_hits += cache.hits;
            total_misses += cache.misses;
        }
        
        if (total_hits + total_misses > 0) {
            double hit_rate = 100.0 * total_hits / (total_hits + total_misses);
            std::cout << "[Cache Stats] Hit rate: " << hit_rate << "%" << std::endl;
        }
    }
};

// Specialized allocator for MCTS nodes
template<typename T>
class MCTSNodeAllocator {
private:
    static ThreadLocalAllocator<T>& get_allocator() {
        static ThreadLocalAllocator<T> allocator;
        return allocator;
    }
    
public:
    using value_type = T;
    
    MCTSNodeAllocator() = default;
    
    template<typename U>
    MCTSNodeAllocator(const MCTSNodeAllocator<U>&) noexcept {}
    
    T* allocate(std::size_t n) {
        if (n == 1) {
            return get_allocator().allocate();
        } else {
            // For arrays, use mimalloc directly
            return static_cast<T*>(mi_aligned_alloc(alignof(T), n * sizeof(T)));
        }
    }
    
    void deallocate(T* p, std::size_t n) noexcept {
        if (n == 1) {
            get_allocator().deallocate(p);
        } else {
            mi_free(p);
        }
    }
    
    template<typename U>
    bool operator==(const MCTSNodeAllocator<U>&) const noexcept {
        return true;
    }
    
    template<typename U>
    bool operator!=(const MCTSNodeAllocator<U>&) const noexcept {
        return false;
    }
    
    // Static methods for bulk operations
    static void allocate_bulk(std::vector<T*>& out, size_t count) {
        get_allocator().allocate_bulk(out, count);
    }
    
    static void deallocate_bulk(const std::vector<T*>& ptrs) {
        get_allocator().deallocate_bulk(ptrs);
    }
    
    static void print_stats() {
        get_allocator().print_stats();
    }
};

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_THREAD_LOCAL_ALLOCATOR_H