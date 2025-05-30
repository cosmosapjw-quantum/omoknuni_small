#include "utils/gpu_memory_manager.h"
#include "utils/logger.h"
#include "utils/profiler.h"
#include <iostream>

namespace alphazero {
namespace utils {

GPUMemoryManager::GPUMemoryManager() {
    // Constructor
}

GPUMemoryManager::~GPUMemoryManager() {
    // Check for memory leaks
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!allocation_registry_.empty()) {
            size_t total_leaked = 0;
            for (const auto& [ptr, size] : allocation_registry_) {
                total_leaked += size;
            }
            LOG_SYSTEM_ERROR("GPU Memory leak detected: {} allocations ({} bytes) not freed",
                           allocation_registry_.size(), total_leaked);
            total_leaked_bytes_.store(total_leaked, std::memory_order_relaxed);
        }
    }
    
    reset();
}

void GPUMemoryManager::initialize(size_t initial_pool_size, size_t maximum_pool_size) {
    if (initialized_) {
        LOG_SYSTEM_WARN("GPU Memory Manager already initialized");
        return;
    }
    
    PROFILE_SCOPE_N("GPUMemoryManager::initialize");
    
#ifdef USE_RAPIDS_RMM
    try {
        // Create CUDA memory resource
        cuda_mr_ = std::make_unique<rmm::mr::cuda_memory_resource>();
        
        // Create pool memory resource with the CUDA MR as upstream
        pool_mr_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            cuda_mr_.get(),
            initial_pool_size,
            maximum_pool_size
        );
        
        // Set as default memory resource
        rmm::mr::set_current_device_resource(pool_mr_.get());
        
        // LOG_SYSTEM_INFO("RAPIDS RMM initialized with pool size {} MB - {} MB",
        //                initial_pool_size / (1024 * 1024),
        //                maximum_pool_size / (1024 * 1024));
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to initialize RMM: {}", e.what());
        throw;
    }
#else
    // LOG_SYSTEM_INFO("GPU Memory Manager initialized (using standard CUDA allocation)");
    LOG_SYSTEM_WARN("RAPIDS RMM is not available. Using standard CUDA memory allocation.");
    // LOG_SYSTEM_INFO("To enable RMM, upgrade to CMake 3.26.4+ and rebuild with -DENABLE_RMM=ON");
#endif
    
    initialized_ = true;
    
    // Log debug info about CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int current_device = 0;
    cudaGetDevice(&current_device);
    
    // LOG_SYSTEM_INFO("GPU Memory Manager initialized on device {}/{}", 
    //                current_device + 1, device_count);
                   
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    // LOG_SYSTEM_INFO("  Device name: {}", prop.name);
    // LOG_SYSTEM_INFO("  Total memory: {} MB", prop.totalGlobalMem / (1024 * 1024));
}

void* GPUMemoryManager::allocate(size_t size, cudaStream_t stream) {
    PROFILE_SCOPE_N("GPUMemoryManager::allocate");
    
    if (!initialized_) {
        initialize();
    }
    
    void* ptr = nullptr;
    
#ifdef USE_RAPIDS_RMM
    try {
        ptr = rmm::mr::get_current_device_resource()->allocate(size, stream);
        PROFILE_ALLOC(ptr, size);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("RMM allocation failed for {} bytes: {}", size, e.what());
        throw;
    }
#else
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        LOG_SYSTEM_ERROR("CUDA allocation failed for {} bytes: {}", 
                        size, cudaGetErrorString(error));
        throw std::runtime_error("CUDA allocation failed");
    }
    PROFILE_ALLOC(ptr, size);
#endif
    
    // Track allocation
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        allocation_registry_[ptr] = size;
    }
    
    utils::ProfileMemoryUsage(size);
    return ptr;
}

void GPUMemoryManager::deallocate(void* ptr, size_t size, cudaStream_t stream) {
    PROFILE_SCOPE_N("GPUMemoryManager::deallocate");
    
    if (!ptr) return;
    
    // Remove from allocation registry
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = allocation_registry_.find(ptr);
        if (it != allocation_registry_.end()) {
            allocation_registry_.erase(it);
        } else {
            LOG_SYSTEM_WARN("Deallocating untracked pointer: {}", ptr);
        }
    }
    
#ifdef USE_RAPIDS_RMM
    try {
        rmm::mr::get_current_device_resource()->deallocate(ptr, size, stream);
        PROFILE_FREE(ptr);
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("RMM deallocation failed: {}", e.what());
    }
#else
    cudaError_t error = cudaFree(ptr);
    if (error != cudaSuccess) {
        LOG_SYSTEM_ERROR("CUDA deallocation failed: {}", cudaGetErrorString(error));
    }
    PROFILE_FREE(ptr);
#endif
}

GPUMemoryManager::MemoryStats GPUMemoryManager::getStats() const {
    MemoryStats stats = {0, 0, 0, 0};
    
    // Get CUDA memory info
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    stats.free_bytes = free_bytes;
    stats.allocated_bytes = total_bytes - free_bytes;
    
#ifdef USE_RAPIDS_RMM
    // RMM doesn't provide direct pool statistics in the current API
    // We can only get general CUDA memory info
    stats.pool_size = total_bytes;
    
    // Track peak allocated memory
    static size_t peak_allocated = 0;
    if (stats.allocated_bytes > peak_allocated) {
        peak_allocated = stats.allocated_bytes;
    }
    stats.peak_allocated = peak_allocated;
    
    // Log detailed memory info at debug level
    // LOG_SYSTEM_DEBUG("GPU Memory: Used {} MB, Free {} MB, Total {} MB, Peak {} MB",
    //                 stats.allocated_bytes / (1024 * 1024),
    //                 stats.free_bytes / (1024 * 1024),
    //                 total_bytes / (1024 * 1024),
    //                 peak_allocated / (1024 * 1024));
#else
    stats.pool_size = 0;
#endif
    
    return stats;
}

void GPUMemoryManager::reset() {
    if (!initialized_) return;
    
    PROFILE_SCOPE_N("GPUMemoryManager::reset");
    
#ifdef USE_RAPIDS_RMM
    // RMM automatically cleans up when destroyed
    pool_mr_.reset();
    cuda_mr_.reset();
#endif
    
    initialized_ = false;
    // LOG_SYSTEM_INFO("GPU Memory Manager reset");
}

} // namespace utils
} // namespace alphazero