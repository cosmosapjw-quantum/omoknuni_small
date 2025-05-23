// include/core/tensor_pool.h
#ifndef ALPHAZERO_CORE_TENSOR_POOL_H
#define ALPHAZERO_CORE_TENSOR_POOL_H

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "core/export_macros.h"

namespace alphazero {
namespace core {

/**
 * @brief Global tensor pool for efficient memory reuse across all games
 * 
 * This pool manages pre-allocated tensor memory to avoid expensive
 * allocations during MCTS tree search. Thread-safe with atomic operations.
 */
class ALPHAZERO_API GlobalTensorPool {
public:
    /**
     * @brief Get singleton instance
     */
    static GlobalTensorPool& getInstance();
    
    /**
     * @brief Get a pre-allocated tensor or create one if needed
     * 
     * @param channels Number of channels
     * @param height Height dimension  
     * @param width Width dimension
     * @return Reusable tensor memory
     */
    std::vector<std::vector<std::vector<float>>> getTensor(int channels, int height, int width);
    
    /**
     * @brief Return a tensor to the pool for reuse
     * 
     * @param tensor Tensor to return
     * @param channels Number of channels
     * @param height Height dimension
     * @param width Width dimension
     */
    void returnTensor(std::vector<std::vector<std::vector<float>>>& tensor, 
                     int channels, int height, int width);
    
    /**
     * @brief Clear all pooled tensors (useful for testing)
     */
    void clear();
    
    /**
     * @brief Get pool statistics
     */
    void getStats(size_t& total_tensors, size_t& available_tensors) const;
    
    /**
     * @brief Get detailed statistics as string
     */
    std::string getDetailedStats() const;
    
    /**
     * @brief Log statistics periodically
     */
    void logStats() const;
    
    /**
     * @brief Get estimated memory usage in MB
     */
    double getMemoryUsageMB() const;
    
    /**
     * @brief Enable/disable periodic logging
     */
    void setPeriodicLogging(bool enable, std::chrono::seconds interval = std::chrono::seconds(30));

private:
    GlobalTensorPool() = default;
    ~GlobalTensorPool() = default;
    GlobalTensorPool(const GlobalTensorPool&) = delete;
    GlobalTensorPool& operator=(const GlobalTensorPool&) = delete;
    
    struct TensorKey {
        int channels, height, width;
        
        bool operator==(const TensorKey& other) const {
            return channels == other.channels && height == other.height && width == other.width;
        }
    };
    
    struct TensorKeyHash {
        size_t operator()(const TensorKey& key) const {
            return std::hash<int>()(key.channels) ^ 
                   (std::hash<int>()(key.height) << 1) ^ 
                   (std::hash<int>()(key.width) << 2);
        }
    };
    
    mutable std::mutex pool_mutex_;
    std::unordered_map<TensorKey, std::vector<std::vector<std::vector<std::vector<float>>>>, TensorKeyHash> tensor_pools_;
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> pool_hits_{0};
    std::atomic<size_t> total_returns_{0};
    std::atomic<size_t> rejected_returns_{0};
    
    // Periodic logging
    std::atomic<bool> periodic_logging_enabled_{false};
    std::chrono::seconds logging_interval_{30};
    mutable std::chrono::steady_clock::time_point last_log_time_;
    
    void periodicLogCheck() const;
};

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_TENSOR_POOL_H