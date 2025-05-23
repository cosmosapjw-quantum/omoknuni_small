// src/core/tensor_pool.cpp
#include "core/tensor_pool.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace alphazero {
namespace core {

GlobalTensorPool& GlobalTensorPool::getInstance() {
    static GlobalTensorPool instance;
    return instance;
}

std::vector<std::vector<std::vector<float>>> GlobalTensorPool::getTensor(int channels, int height, int width) {
    TensorKey key{channels, height, width};
    
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        auto& pool = tensor_pools_[key];
        
        if (!pool.empty()) {
            auto tensor = std::move(pool.back());
            pool.pop_back();
            pool_hits_.fetch_add(1, std::memory_order_relaxed);
            
            // Clear the tensor for reuse
            for (auto& channel : tensor) {
                for (auto& row : channel) {
                    std::fill(row.begin(), row.end(), 0.0f);
                }
            }
            
            return tensor;
        }
    }
    
    // Create new tensor if none available in pool
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    return std::vector<std::vector<std::vector<float>>>(
        channels, std::vector<std::vector<float>>(
            height, std::vector<float>(width, 0.0f)));
}

void GlobalTensorPool::returnTensor(std::vector<std::vector<std::vector<float>>>& tensor, 
                                   int channels, int height, int width) {
    total_returns_.fetch_add(1, std::memory_order_relaxed);
    
    // Validate tensor dimensions
    if (static_cast<int>(tensor.size()) != channels || 
        (channels > 0 && static_cast<int>(tensor[0].size()) != height) ||
        (channels > 0 && height > 0 && static_cast<int>(tensor[0][0].size()) != width)) {
        rejected_returns_.fetch_add(1, std::memory_order_relaxed);
        return; // Don't pool incorrectly sized tensors
    }
    
    TensorKey key{channels, height, width};
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto& pool = tensor_pools_[key];
    
    // Limit pool size to prevent excessive memory usage
    // Increased from 10 to 256 to support high-throughput MCTS with many parallel threads
    const size_t MAX_POOL_SIZE = 256;
    if (pool.size() < MAX_POOL_SIZE) {
        pool.emplace_back(std::move(tensor));
    } else {
        rejected_returns_.fetch_add(1, std::memory_order_relaxed);
    }
    
    periodicLogCheck();
}

void GlobalTensorPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    tensor_pools_.clear();
    total_allocations_.store(0, std::memory_order_relaxed);
    pool_hits_.store(0, std::memory_order_relaxed);
    total_returns_.store(0, std::memory_order_relaxed);
    rejected_returns_.store(0, std::memory_order_relaxed);
}

void GlobalTensorPool::getStats(size_t& total_tensors, size_t& available_tensors) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    total_tensors = total_allocations_.load(std::memory_order_relaxed);
    available_tensors = 0;
    
    for (const auto& [key, pool] : tensor_pools_) {
        available_tensors += pool.size();
    }
}

std::string GlobalTensorPool::getDetailedStats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    std::stringstream ss;
    
    size_t total_available = 0;
    size_t total_configs = tensor_pools_.size();
    
    ss << "=== GlobalTensorPool Statistics ===\n";
    ss << "Total allocations: " << total_allocations_.load() << "\n";
    ss << "Pool hits: " << pool_hits_.load() << " (" 
       << (total_allocations_ > 0 ? (100.0 * pool_hits_ / total_allocations_) : 0.0) 
       << "% hit rate)\n";
    ss << "Total returns: " << total_returns_.load() << "\n";
    ss << "Rejected returns: " << rejected_returns_.load() << "\n";
    ss << "Unique tensor configs: " << total_configs << "\n\n";
    
    ss << "Pool contents by configuration:\n";
    for (const auto& [key, pool] : tensor_pools_) {
        size_t memory_per_tensor = key.channels * key.height * key.width * sizeof(float);
        size_t total_memory = memory_per_tensor * pool.size();
        total_available += pool.size();
        
        ss << "  [" << key.channels << "x" << key.height << "x" << key.width << "]: " 
           << pool.size() << " tensors, " 
           << std::fixed << std::setprecision(2) << (total_memory / (1024.0 * 1024.0)) << " MB\n";
    }
    
    ss << "\nTotal available tensors: " << total_available << "\n";
    ss << "Estimated total memory: " << std::fixed << std::setprecision(2) 
       << getMemoryUsageMB() << " MB\n";
    
    return ss.str();
}

void GlobalTensorPool::logStats() const {
    std::cout << getDetailedStats() << std::endl;
}

double GlobalTensorPool::getMemoryUsageMB() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t total_bytes = 0;
    for (const auto& [key, pool] : tensor_pools_) {
        size_t memory_per_tensor = key.channels * key.height * key.width * sizeof(float);
        total_bytes += memory_per_tensor * pool.size();
    }
    
    return total_bytes / (1024.0 * 1024.0);
}

void GlobalTensorPool::setPeriodicLogging(bool enable, std::chrono::seconds interval) {
    periodic_logging_enabled_.store(enable, std::memory_order_relaxed);
    logging_interval_ = interval;
    last_log_time_ = std::chrono::steady_clock::now();
}

void GlobalTensorPool::periodicLogCheck() const {
    if (!periodic_logging_enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time_) >= logging_interval_) {
        logStats();
        last_log_time_ = now;
    }
}

} // namespace core  
} // namespace alphazero