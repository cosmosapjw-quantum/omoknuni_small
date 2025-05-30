#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <mutex>
#include <algorithm>
#ifdef WITH_TORCH
#include <cuda_runtime.h>
#endif
#include "core/export_macros.h"
#include "utils/logger.h"

namespace alphazero {
namespace utils {

/**
 * @brief Thread-local memory manager to replace global singleton
 * 
 * Eliminates contention by using thread-local storage for memory tracking
 */
class ALPHAZERO_API ThreadLocalMemoryManager {
public:
    struct MemoryStats {
        size_t cpu_allocated{0};
        size_t gpu_allocated{0};
        size_t peak_cpu{0};
        size_t peak_gpu{0};
    };
    
    // Get thread-local instance
    static ThreadLocalMemoryManager& getInstance() {
        static thread_local ThreadLocalMemoryManager instance;
        return instance;
    }
    
    // Track allocations (lock-free)
    void trackCpuAllocation(size_t bytes) {
        cpu_allocated_ += bytes;
        size_t current = cpu_allocated_.load();
        size_t peak = peak_cpu_.load();
        while (current > peak && !peak_cpu_.compare_exchange_weak(peak, current));
    }
    
    void trackCpuDeallocation(size_t bytes) {
        cpu_allocated_ -= bytes;
    }
    
    void trackGpuAllocation(size_t bytes) {
        gpu_allocated_ += bytes;
        size_t current = gpu_allocated_.load();
        size_t peak = peak_gpu_.load();
        while (current > peak && !peak_gpu_.compare_exchange_weak(peak, current));
    }
    
    void trackGpuDeallocation(size_t bytes) {
        gpu_allocated_ -= bytes;
    }
    
    // Get current stats
    size_t getCpuAllocated() const { return cpu_allocated_.load(); }
    size_t getGpuAllocated() const { return gpu_allocated_.load(); }
    
    // Cleanup if needed (non-blocking)
    bool shouldCleanup(size_t threshold) const {
        return (cpu_allocated_.load() + gpu_allocated_.load()) > threshold;
    }
    
    // Aggregate stats from all threads
    static MemoryStats getGlobalStats() {
        MemoryStats global;
        for (const auto& manager : all_managers_) {
            if (manager) {
                global.cpu_allocated += manager->cpu_allocated_.load();
                global.gpu_allocated += manager->gpu_allocated_.load();
                global.peak_cpu = std::max(global.peak_cpu, manager->peak_cpu_.load());
                global.peak_gpu = std::max(global.peak_gpu, manager->peak_gpu_.load());
            }
        }
        return global;
    }
    
private:
    ThreadLocalMemoryManager() {
        // Register this instance
        static std::mutex registry_mutex;
        std::lock_guard<std::mutex> lock(registry_mutex);
        all_managers_.push_back(this);
    }
    
    ~ThreadLocalMemoryManager() {
        // Unregister
        static std::mutex registry_mutex;
        std::lock_guard<std::mutex> lock(registry_mutex);
        all_managers_.erase(
            std::remove(all_managers_.begin(), all_managers_.end(), this),
            all_managers_.end()
        );
    }
    
    // Thread-local counters
    std::atomic<size_t> cpu_allocated_{0};
    std::atomic<size_t> gpu_allocated_{0};
    std::atomic<size_t> peak_cpu_{0};
    std::atomic<size_t> peak_gpu_{0};
    
    // Global registry (only accessed during construction/destruction)
    static std::vector<ThreadLocalMemoryManager*> all_managers_;
};

// RAII wrapper for automatic tracking
template<typename T>
class TrackedAllocation {
public:
    TrackedAllocation(T* ptr, size_t size, bool is_gpu = false)
        : ptr_(ptr), size_(size), is_gpu_(is_gpu) {
        auto& manager = ThreadLocalMemoryManager::getInstance();
        if (is_gpu_) {
            manager.trackGpuAllocation(size_);
        } else {
            manager.trackCpuAllocation(size_);
        }
    }
    
    ~TrackedAllocation() {
        if (ptr_) {
            auto& manager = ThreadLocalMemoryManager::getInstance();
            if (is_gpu_) {
                manager.trackGpuDeallocation(size_);
#ifdef WITH_TORCH
                cudaFree(ptr_);
#endif
            } else {
                manager.trackCpuDeallocation(size_);
                delete[] reinterpret_cast<char*>(ptr_);
            }
        }
    }
    
    // Move semantics
    TrackedAllocation(TrackedAllocation&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), is_gpu_(other.is_gpu_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    TrackedAllocation& operator=(TrackedAllocation&& other) noexcept {
        if (this != &other) {
            // Clean up current allocation
            this->~TrackedAllocation();
            
            // Take ownership
            ptr_ = other.ptr_;
            size_ = other.size_;
            is_gpu_ = other.is_gpu_;
            
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // No copy
    TrackedAllocation(const TrackedAllocation&) = delete;
    TrackedAllocation& operator=(const TrackedAllocation&) = delete;
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    T* ptr_;
    size_t size_;
    bool is_gpu_;
};

} // namespace utils
} // namespace alphazero