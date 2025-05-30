// src/mcts/gpu_memory_pool.cpp
#include "mcts/gpu_memory_pool.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>

namespace alphazero {
namespace mcts {

GPUMemoryPool::GPUMemoryPool(const PoolConfig& config) : config_(config) {
    initializePools();
}

GPUMemoryPool::~GPUMemoryPool() {
    // Clean up all allocated memory
    for (auto& pool : block_pools_) {
        for (auto& block : pool.blocks) {
            if (block->device_ptr) {
                cudaSetDevice(block->device_id);
                cudaFree(block->device_ptr);
            }
            if (block->host_pinned_ptr) {
                cudaFreeHost(block->host_pinned_ptr);
            }
        }
    }

    for (auto& alloc : large_allocations_) {
        if (alloc.block->device_ptr) {
            cudaSetDevice(alloc.block->device_id);
            cudaFree(alloc.block->device_ptr);
        }
        if (alloc.block->host_pinned_ptr) {
            cudaFreeHost(alloc.block->host_pinned_ptr);
        }
    }

    for (void* ptr : pinned_pool_.pinned_buffers) {
        cudaFreeHost(ptr);
    }
}

void GPUMemoryPool::initializePools() {
    // Create pools for each block size
    block_pools_.resize(config_.block_sizes.size());
    
    for (size_t i = 0; i < config_.block_sizes.size(); ++i) {
        block_pools_[i].block_size = config_.block_sizes[i];
        size_to_pool_index_[config_.block_sizes[i]] = i;
    }

    // Get number of GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);

    // Pre-allocate blocks for each pool
    size_t total_size = config_.initial_pool_size_mb * 1024 * 1024;
    size_t size_per_pool = total_size / config_.block_sizes.size();

    for (size_t i = 0; i < block_pools_.size(); ++i) {
        size_t num_blocks = size_per_pool / block_pools_[i].block_size;
        
        // Distribute blocks across GPUs
        for (int device_id = 0; device_id < device_count; ++device_id) {
            size_t blocks_per_device = (num_blocks + device_count - 1) / device_count;
            allocateBlocksForPool(block_pools_[i], blocks_per_device, device_id);
        }
    }

    // Initialize pinned memory pool
    pinned_pool_.buffer_size = 4 * 1024 * 1024; // 4MB buffers
    size_t num_pinned_buffers = config_.pinned_memory_size_mb / 4;
    
    for (size_t i = 0; i < num_pinned_buffers; ++i) {
        void* ptr = allocatePinnedMemory(pinned_pool_.buffer_size);
        if (ptr) {
            pinned_pool_.pinned_buffers.push_back(ptr);
            pinned_pool_.free_buffers.push(ptr);
        }
    }

    // Enable peer access if requested
    if (config_.enable_peer_access && device_count > 1) {
        for (int i = 0; i < device_count; ++i) {
            cudaSetDevice(i);
            for (int j = 0; j < device_count; ++j) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
    }

    LOG_NN_INFO("GPU Memory Pool initialized: {} MB across {} devices", 
                config_.initial_pool_size_mb, device_count);
}

void GPUMemoryPool::allocateBlocksForPool(BlockPool& pool, size_t count, int device_id) {
    cudaSetDevice(device_id);
    
    for (size_t i = 0; i < count; ++i) {
        auto block = std::make_unique<MemoryBlock>();
        block->size = pool.block_size;
        block->device_id = device_id;
        
        // Allocate device memory
        block->device_ptr = allocateDeviceMemory(pool.block_size, device_id);
        if (!block->device_ptr) {
            LOG_NN_WARN("Failed to allocate GPU memory block of size {} MB", 
                       pool.block_size / (1024 * 1024));
            break;
        }

        // Optionally allocate pinned host memory for this block
        if (i < count / 4) { // Only for 25% of blocks to save host memory
            block->host_pinned_ptr = allocatePinnedMemory(pool.block_size);
        }

        total_allocated_.fetch_add(pool.block_size, std::memory_order_relaxed);
        
        // Add to free list
        pool.free_blocks.push(block.get());
        pool.blocks.push_back(std::move(block));
    }
}

torch::Tensor GPUMemoryPool::allocateTensor(
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    int device_id,
    cudaStream_t stream
) {
    // Calculate required size
    size_t element_size = torch::elementSize(dtype);
    size_t num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    size_t required_size = num_elements * element_size;

    // Find appropriate pool
    size_t pool_index = findBestPoolIndex(required_size);
    
    if (pool_index < block_pools_.size()) {
        // Try to get from pool
        auto& pool = block_pools_[pool_index];
        
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            if (!pool.free_blocks.empty()) {
                MemoryBlock* block = pool.free_blocks.front();
                pool.free_blocks.pop();
                
                // Check if block is from the right device
                if (block->device_id == device_id) {
                    block->in_use.store(true, std::memory_order_release);
                    block->last_stream = stream;
                    
                    total_in_use_.fetch_add(block->size, std::memory_order_relaxed);
                    pool.reuses.fetch_add(1, std::memory_order_relaxed);
                    
                    // Create tensor from the memory
                    return tensorFromMemory(block->device_ptr, shape, dtype, device_id);
                } else {
                    // Wrong device, put it back
                    pool.free_blocks.push(block);
                }
            }
        }
        
        // Pool was empty or wrong device, allocate new block
        allocateBlocksForPool(pool, 1, device_id);
        pool.allocations.fetch_add(1, std::memory_order_relaxed);
        
        // Try again
        std::lock_guard<std::mutex> lock(pool.mutex);
        if (!pool.free_blocks.empty()) {
            MemoryBlock* block = pool.free_blocks.front();
            pool.free_blocks.pop();
            
            block->in_use.store(true, std::memory_order_release);
            block->last_stream = stream;
            
            total_in_use_.fetch_add(block->size, std::memory_order_relaxed);
            
            return tensorFromMemory(block->device_ptr, shape, dtype, device_id);
        }
    }

    // Large allocation - use fallback
    auto block = allocateBlock(required_size, device_id, stream);
    if (block && block->device_ptr) {
        return tensorFromMemory(block->device_ptr, shape, dtype, device_id);
    }

    // Final fallback - let PyTorch allocate
    LOG_NN_WARN("GPU memory pool exhausted, falling back to PyTorch allocation");
    return torch::empty(shape, torch::dtype(dtype).device(torch::kCUDA, device_id));
}

std::shared_ptr<GPUMemoryPool::MemoryBlock> GPUMemoryPool::allocateBlock(
    size_t size,
    int device_id,
    cudaStream_t stream
) {
    // Check large allocations for reuse
    {
        std::lock_guard<std::mutex> lock(large_alloc_mutex_);
        
        auto now = std::chrono::steady_clock::now();
        for (auto& alloc : large_allocations_) {
            if (!alloc.block->in_use.load(std::memory_order_acquire) &&
                alloc.block->size >= size &&
                alloc.block->device_id == device_id) {
                
                // Reuse this allocation
                alloc.block->in_use.store(true, std::memory_order_release);
                alloc.block->last_stream = stream;
                alloc.last_used = now;
                
                total_in_use_.fetch_add(alloc.block->size, std::memory_order_relaxed);
                
                // Create shared_ptr with proper cleanup
                auto* pool_ptr = this;
                size_t block_size = alloc.block->size;
                return std::shared_ptr<MemoryBlock>(alloc.block.get(), 
                    [pool_ptr, block_size](MemoryBlock* block) {
                        // Mark as not in use and update stats
                        block->in_use.store(false, std::memory_order_release);
                        pool_ptr->total_in_use_.fetch_sub(block_size, std::memory_order_relaxed);
                    });
            }
        }
    }

    // Allocate new large block
    auto block = std::make_unique<MemoryBlock>();
    block->size = size;
    block->device_id = device_id;
    block->device_ptr = allocateDeviceMemory(size, device_id);
    block->last_stream = stream;
    
    if (!block->device_ptr) {
        return nullptr;
    }

    block->in_use.store(true, std::memory_order_release);
    
    total_allocated_.fetch_add(size, std::memory_order_relaxed);
    total_in_use_.fetch_add(size, std::memory_order_relaxed);

    // Store in large_allocations_ before creating shared_ptr
    LargeAllocation large_alloc;
    large_alloc.block = std::move(block);
    large_alloc.last_used = std::chrono::steady_clock::now();
    
    // Get raw pointer before moving
    MemoryBlock* raw_ptr = large_alloc.block.get();
    
    {
        std::lock_guard<std::mutex> lock(large_alloc_mutex_);
        large_allocations_.push_back(std::move(large_alloc));
    }
    
    // Create shared_ptr with proper cleanup
    auto* pool_ptr = this;
    return std::shared_ptr<MemoryBlock>(raw_ptr, 
        [pool_ptr, size](MemoryBlock* block) {
            // Mark as not in use and update stats
            block->in_use.store(false, std::memory_order_release);
            pool_ptr->total_in_use_.fetch_sub(size, std::memory_order_relaxed);
        });
}

torch::Tensor GPUMemoryPool::tensorFromMemory(
    void* data,
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    int device_id
) {
    // For now, we need to copy to ensure proper memory management
    // TODO: Implement a custom storage that can share ownership with our memory pool
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, device_id);
    
    // Create a properly managed tensor
    torch::Tensor tensor = torch::empty(shape, options);
    
    // Calculate total bytes
    size_t total_bytes = tensor.numel() * tensor.element_size();
    
    // Copy data to the tensor
    cudaError_t err = cudaMemcpy(tensor.data_ptr(), data, total_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        LOG_SYSTEM_ERROR("Failed to copy tensor data: {}", cudaGetErrorString(err));
    }
    
    return tensor;
}

void GPUMemoryPool::asyncCopyToDevice(
    const void* host_data,
    void* device_data,
    size_t size,
    cudaStream_t stream
) {
    // Try to get pinned buffer for staging
    void* pinned_buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(pinned_pool_.mutex);
        if (!pinned_pool_.free_buffers.empty() && size <= pinned_pool_.buffer_size) {
            pinned_buffer = pinned_pool_.free_buffers.front();
            pinned_pool_.free_buffers.pop();
        }
    }

    if (pinned_buffer) {
        // Copy to pinned memory first
        memcpy(pinned_buffer, host_data, size);
        
        // Async copy to device
        cudaMemcpyAsync(device_data, pinned_buffer, size, 
                       cudaMemcpyHostToDevice, stream);
        
        // Create callback data structure
        struct CallbackData {
            PinnedMemoryPool* pool;
            void* buffer;
        };
        
        auto* callback_data = new CallbackData{&pinned_pool_, pinned_buffer};
        
        // Return pinned buffer after stream completes
        cudaStreamAddCallback(stream, 
            [](cudaStream_t, cudaError_t, void* userData) {
                auto* data = static_cast<CallbackData*>(userData);
                
                std::lock_guard<std::mutex> lock(data->pool->mutex);
                data->pool->free_buffers.push(data->buffer);
                
                delete data;
            }, 
            callback_data, 0);
    } else {
        // Direct copy
        cudaMemcpyAsync(device_data, host_data, size, 
                       cudaMemcpyHostToDevice, stream);
    }
}

size_t GPUMemoryPool::findBestPoolIndex(size_t requested_size) const {
    // Find the smallest pool that can fit the request
    for (size_t i = 0; i < config_.block_sizes.size(); ++i) {
        if (config_.block_sizes[i] >= requested_size) {
            return i;
        }
    }
    return config_.block_sizes.size(); // Indicates need for large allocation
}

void* GPUMemoryPool::allocateDeviceMemory(size_t size, int device_id) {
    cudaSetDevice(device_id);
    
    void* ptr = nullptr;
    cudaError_t err;
    
    if (config_.use_unified_memory) {
        err = cudaMallocManaged(&ptr, size);
    } else {
        err = cudaMalloc(&ptr, size);
    }
    
    if (err != cudaSuccess) {
        LOG_NN_ERROR("CUDA allocation failed: {}", cudaGetErrorString(err));
        return nullptr;
    }
    
    return ptr;
}

void GPUMemoryPool::freeDeviceMemory(void* ptr, int device_id) {
    if (ptr) {
        cudaSetDevice(device_id);
        cudaFree(ptr);
    }
}

void* GPUMemoryPool::allocatePinnedMemory(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    
    if (err != cudaSuccess) {
        LOG_NN_WARN("Pinned memory allocation failed: {}", cudaGetErrorString(err));
        return nullptr;
    }
    
    return ptr;
}

void GPUMemoryPool::freePinnedMemory(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

GPUMemoryPool::PoolStats GPUMemoryPool::getStats() const {
    PoolStats stats;
    stats.total_allocated = total_allocated_.load(std::memory_order_relaxed);
    stats.total_in_use = total_in_use_.load(std::memory_order_relaxed);
    stats.num_allocations = 0;
    stats.num_reuses = 0;

    for (const auto& pool : block_pools_) {
        stats.num_allocations += pool.allocations.load(std::memory_order_relaxed);
        stats.num_reuses += pool.reuses.load(std::memory_order_relaxed);
        
        size_t in_use = pool.blocks.size() - pool.free_blocks.size();
        stats.block_usage[pool.block_size] = in_use;
    }

    if (stats.total_allocated > 0) {
        stats.fragmentation_ratio = 1.0f - (float(stats.total_in_use) / float(stats.total_allocated));
    } else {
        stats.fragmentation_ratio = 0.0f;
    }

    return stats;
}

void GPUMemoryPool::defragment() {
    // TODO: Implement defragmentation by consolidating free blocks
    LOG_NN_INFO("GPU memory pool defragmentation not yet implemented");
}

void GPUMemoryPool::trim(float keep_ratio) {
    std::lock_guard<std::mutex> lock(large_alloc_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::minutes(5); // Free blocks unused for 5+ minutes
    
    large_allocations_.erase(
        std::remove_if(large_allocations_.begin(), large_allocations_.end(),
            [&](LargeAllocation& alloc) {
                if (!alloc.block->in_use.load(std::memory_order_acquire) &&
                    alloc.last_used < cutoff) {
                    // Free this allocation
                    freeDeviceMemory(alloc.block->device_ptr, alloc.block->device_id);
                    if (alloc.block->host_pinned_ptr) {
                        freePinnedMemory(alloc.block->host_pinned_ptr);
                    }
                    total_allocated_.fetch_sub(alloc.block->size, std::memory_order_relaxed);
                    return true;
                }
                return false;
            }),
        large_allocations_.end()
    );
}

} // namespace mcts
} // namespace alphazero