// include/mcts/dynamic_batch_adjuster.h
#ifndef DYNAMIC_BATCH_ADJUSTER_H
#define DYNAMIC_BATCH_ADJUSTER_H

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <algorithm>
#include <string>
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Dynamic batch size and timeout adjuster based on thread arrival patterns
 * 
 * This class monitors actual batch sizes achieved and thread arrival patterns
 * to dynamically adjust batch formation timeouts for optimal GPU utilization.
 */
class ALPHAZERO_API DynamicBatchAdjuster {
public:
    struct BatchStats {
        size_t batch_size;
        std::chrono::milliseconds wait_time;
        std::chrono::steady_clock::time_point timestamp;
        double gpu_utilization;  // Estimated based on batch size
    };
    
    struct AdjustmentParams {
        std::chrono::milliseconds min_timeout{5};
        std::chrono::milliseconds max_timeout{200};
        std::chrono::milliseconds current_timeout{50};
        size_t target_batch_size{32};
        size_t min_batch_size{8};
        double target_gpu_utilization{0.8};  // 80% target
        
        // Adaptive parameters
        double timeout_increase_factor{1.2};
        double timeout_decrease_factor{0.8};
        size_t history_window{100};  // Number of batches to consider
        
        // Thread arrival pattern detection
        double arrival_rate_ema_alpha{0.1};  // Exponential moving average
        double initial_arrival_rate{0.0};  // Initial requests per ms (moved from atomic)
    };
    
    DynamicBatchAdjuster();
    explicit DynamicBatchAdjuster(const AdjustmentParams& params);
    
    /**
     * @brief Record a completed batch and adjust timeouts
     */
    void recordBatch(size_t batch_size, std::chrono::milliseconds actual_wait_time);
    
    /**
     * @brief Record thread arrival for pattern analysis
     */
    void recordThreadArrival();
    
    /**
     * @brief Get current recommended timeout
     */
    std::chrono::milliseconds getCurrentTimeout() const {
        return params_.current_timeout;
    }
    
    /**
     * @brief Get current arrival rate estimate
     */
    double getArrivalRate() const {
        return current_arrival_rate_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get average batch size over recent history
     */
    double getAverageBatchSize() const;
    
    /**
     * @brief Get GPU utilization estimate
     */
    double getGPUUtilization() const;
    
    /**
     * @brief Force timeout adjustment based on external metrics
     */
    void forceAdjustTimeout(std::chrono::milliseconds new_timeout);
    
    /**
     * @brief Get statistics summary
     */
    std::string getStatsSummary() const;

private:
    AdjustmentParams params_;
    
    // Thread-safe statistics storage
    mutable std::mutex stats_mutex_;
    std::deque<BatchStats> batch_history_;
    
    // Thread arrival tracking
    std::chrono::steady_clock::time_point last_arrival_time_;
    std::atomic<size_t> arrival_count_{0};
    std::atomic<double> current_arrival_rate_{0.0};  // Moved from params
    
    // Performance metrics
    std::atomic<double> avg_batch_size_{0.0};
    std::atomic<double> avg_gpu_utilization_{0.0};
    
    /**
     * @brief Adjust timeout based on recent performance
     */
    void adjustTimeout();
    
    /**
     * @brief Estimate GPU utilization from batch size
     */
    double estimateGPUUtilization(size_t batch_size) const;
    
    /**
     * @brief Update arrival rate estimate
     */
    void updateArrivalRate();
};

} // namespace mcts
} // namespace alphazero

#endif // DYNAMIC_BATCH_ADJUSTER_H