#ifndef ALPHAZERO_GPU_MEMORY_MANAGER_H
#define ALPHAZERO_GPU_MEMORY_MANAGER_H

#include <memory>
#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <atomic>

#ifdef USE_RAPIDS_RMM
    #include <rmm/mr/device/cuda_memory_resource.hpp>
    #include <rmm/mr/device/pool_memory_resource.hpp>
    #include <rmm/mr/device/device_memory_resource.hpp>
    #include <rmm/device_buffer.hpp>
    #include <rmm/device_uvector.hpp>
#endif

#ifdef WITH_TORCH
#include <cuda_runtime.h>
#endif

namespace alphazero {
namespace utils {

#ifdef WITH_TORCH
/**
 * GPU Memory Manager using RAPIDS Memory Manager (RMM)
 * 
 * Provides:
 * - Fast GPU memory allocation
 * - Memory pooling for reduced fragmentation
 * - Stream-ordered allocation
 * - Device memory utilities
 */
class GPUMemoryManager {
public:
    static GPUMemoryManager& getInstance() {
        static GPUMemoryManager instance;
        return instance;
    }
    
    // Initialize the memory pool
    void initialize(size_t initial_pool_size = 1024 * 1024 * 1024,  // 1GB
                   size_t maximum_pool_size = 8ULL * 1024 * 1024 * 1024); // 8GB
    
    // Allocate device memory
    void* allocate(size_t size, cudaStream_t stream = 0);
    
    // Deallocate device memory
    void deallocate(void* ptr, size_t size, cudaStream_t stream = 0);
    
    // Get current memory statistics
    struct MemoryStats {
        size_t allocated_bytes;
        size_t free_bytes;
        size_t pool_size;
        size_t peak_allocated;
    };
    
    MemoryStats getStats() const;
    
    // Reset the memory pool
    void reset();
    
private:
    GPUMemoryManager();
    ~GPUMemoryManager();
    
    // Delete copy/move constructors and operators
    GPUMemoryManager(const GPUMemoryManager&) = delete;
    GPUMemoryManager& operator=(const GPUMemoryManager&) = delete;
    GPUMemoryManager(GPUMemoryManager&&) = delete;
    GPUMemoryManager& operator=(GPUMemoryManager&&) = delete;
    
#ifdef USE_RAPIDS_RMM
    std::unique_ptr<rmm::mr::cuda_memory_resource> cuda_mr_;
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_mr_;
#endif
    
    bool initialized_ = false;
    
    // Allocation tracking for leak detection
    mutable std::mutex registry_mutex_;
    std::unordered_map<void*, size_t> allocation_registry_;
    std::atomic<size_t> total_leaked_bytes_{0};
};

// RAII wrapper for GPU memory
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t count, cudaStream_t stream = 0)
        : size_(count * sizeof(T)), stream_(stream) {
        data_ = static_cast<T*>(GPUMemoryManager::getInstance().allocate(size_, stream));
    }
    
    ~DeviceBuffer() {
        if (data_) {
            GPUMemoryManager::getInstance().deallocate(data_, size_, stream_);
        }
    }
    
    // Delete copy constructor/assignment
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move constructor/assignment
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), stream_(other.stream_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                GPUMemoryManager::getInstance().deallocate(data_, size_, stream_);
            }
            data_ = other.data_;
            size_ = other.size_;
            stream_ = other.stream_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return size_ / sizeof(T); }
    
    // Copy to/from host
    void copyFromHost(const T* host_data, size_t count) {
        cudaMemcpyAsync(data_, host_data, count * sizeof(T), 
                       cudaMemcpyHostToDevice, stream_);
    }
    
    void copyToHost(T* host_data, size_t count) const {
        cudaMemcpyAsync(host_data, data_, count * sizeof(T), 
                       cudaMemcpyDeviceToHost, stream_);
    }
    
private:
    T* data_ = nullptr;
    size_t size_ = 0;
    cudaStream_t stream_ = 0;
};

// Pinned host memory for fast CPU-GPU transfers
template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer(size_t count) : size_(count) {
        cudaMallocHost(&data_, count * sizeof(T));
        if (!data_) {
            throw std::runtime_error("Failed to allocate pinned memory");
        }
    }
    
    ~PinnedBuffer() {
        if (data_) {
            cudaFreeHost(data_);
        }
    }
    
    // Delete copy constructor/assignment
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return size_; }
    
private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

#else // !WITH_TORCH
// Dummy classes when torch is not available
class GPUMemoryManager {
public:
    static GPUMemoryManager& getInstance() {
        static GPUMemoryManager instance;
        return instance;
    }
    
    void initialize(size_t = 1024 * 1024 * 1024, size_t = 8ULL * 1024 * 1024 * 1024) {}
    void cleanup() {}
    void* allocate(size_t, int = 0) { return nullptr; }
    void deallocate(void*, size_t, int = 0) {}
    size_t getPeakMemoryUsage() const { return 0; }
    size_t getCurrentMemoryUsage() const { return 0; }
    size_t getAvailableMemory() const { return 0; }
    bool isInitialized() const { return false; }
    void synchronize() {}
    void setMemoryPressureCallback(std::function<void()>) {}
    
    // Additional methods needed by engine
    struct Stats {
        size_t allocated = 0;
        size_t peak = 0;
        size_t available = 0;
        size_t allocated_bytes = 0;
        size_t peak_allocated = 0;
    };
    Stats getStats() const { return {}; }
    void reset() {}
};

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t, int = 0) {}
    ~DeviceBuffer() {}
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = default;
    DeviceBuffer& operator=(DeviceBuffer&&) = default;
    T* get() { return nullptr; }
    const T* get() const { return nullptr; }
    size_t size() const { return 0; }
    void copyFromHost(const T*, size_t) {}
    void copyToHost(T*, size_t) const {}
};

template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer(size_t) {}
    ~PinnedBuffer() {}
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() { return nullptr; }
    const T* get() const { return nullptr; }
    size_t size() const { return 0; }
};
#endif // WITH_TORCH

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_GPU_MEMORY_MANAGER_H