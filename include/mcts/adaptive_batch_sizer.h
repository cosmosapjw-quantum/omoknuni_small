#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <deque>
#include <memory>

namespace mcts {

struct BatchPerformanceMetrics {
    size_t batch_size;
    std::chrono::microseconds inference_time;
    std::chrono::microseconds queue_wait_time;
    double gpu_utilization_percent;
    size_t memory_usage_mb;
    std::chrono::steady_clock::time_point timestamp;
};

class AdaptiveBatchSizer {
public:
    AdaptiveBatchSizer(size_t initial_batch_size = 32,
                      size_t min_batch_size = 4,
                      size_t max_batch_size = 256);

    size_t getCurrentBatchSize() const;
    
    void recordBatchPerformance(const BatchPerformanceMetrics& metrics);
    
    void adjustBatchSize();
    
    struct AdaptationConfig {
        double target_gpu_utilization = 0.85;
        double utilization_tolerance = 0.1;
        
        std::chrono::microseconds max_acceptable_latency{50000}; // 50ms
        std::chrono::microseconds target_queue_wait{10000};     // 10ms
        
        double performance_improvement_threshold = 0.05; // 5%
        size_t min_samples_for_adaptation = 10;
        size_t max_history_size = 100;
        
        double size_increase_factor = 1.2;
        double size_decrease_factor = 0.8;
        size_t max_consecutive_adjustments = 3;
        
        // Batch size constraints
        size_t min_batch_size = 4;
        size_t max_batch_size = 256;
    };
    
    void setConfig(const AdaptationConfig& config);
    const AdaptationConfig& getConfig() const;
    
    struct AdaptationStats {
        size_t total_adaptations;
        size_t successful_adaptations;
        size_t failed_adaptations;
        double average_gpu_utilization;
        std::chrono::microseconds average_inference_time;
        std::chrono::microseconds average_queue_wait;
        size_t current_batch_size;
        std::chrono::steady_clock::time_point last_adaptation;
    };
    
    AdaptationStats getStats() const;
    void resetStats();

private:
    mutable std::mutex mutex_;
    std::atomic<size_t> current_batch_size_;
    
    AdaptationConfig config_;
    
    std::deque<BatchPerformanceMetrics> performance_history_;
    
    mutable std::atomic<size_t> total_adaptations_{0};
    mutable std::atomic<size_t> successful_adaptations_{0};
    mutable std::atomic<size_t> failed_adaptations_{0};
    std::chrono::steady_clock::time_point last_adaptation_time_;
    
    size_t consecutive_adjustments_ = 0;
    bool last_adjustment_positive_ = false;
    
    double calculateAverageGpuUtilization() const;
    std::chrono::microseconds calculateAverageInferenceTime() const;
    std::chrono::microseconds calculateAverageQueueWait() const;
    
    bool shouldIncreaseBatchSize() const;
    bool shouldDecreaseBatchSize() const;
    
    size_t calculateOptimalBatchSize() const;
    
    double calculatePerformanceScore(const BatchPerformanceMetrics& metrics) const;
    double calculateThroughput(size_t batch_size, std::chrono::microseconds inference_time) const;
    
    void trimHistory();
};

class GPUMonitor {
public:
    GPUMonitor();
    ~GPUMonitor();
    
    double getCurrentUtilization();
    size_t getCurrentMemoryUsage();
    
    bool isAvailable() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace mcts