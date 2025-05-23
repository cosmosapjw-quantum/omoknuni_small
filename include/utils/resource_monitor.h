// include/utils/resource_monitor.h
#ifndef ALPHAZERO_UTILS_RESOURCE_MONITOR_H
#define ALPHAZERO_UTILS_RESOURCE_MONITOR_H

#include <atomic>
#include <chrono>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include "core/export_macros.h"

namespace alphazero {
namespace utils {

/**
 * @brief Real-time resource utilization monitor
 * 
 * Tracks CPU, GPU, memory usage and provides performance insights
 */
class ALPHAZERO_API ResourceMonitor {
public:
    struct ResourceStats {
        double cpu_usage_percent = 0.0;
        double gpu_usage_percent = 0.0;
        size_t memory_usage_mb = 0;
        size_t gpu_memory_usage_mb = 0;
        size_t active_threads = 0;
        double avg_batch_size = 0.0;
        double avg_inference_latency_ms = 0.0;
        size_t total_requests_processed = 0;
        size_t queue_depth = 0;
    };
    
    /**
     * @brief Get singleton instance
     */
    static ResourceMonitor& getInstance();
    
    /**
     * @brief Start monitoring in background thread
     */
    void startMonitoring(std::chrono::milliseconds interval = std::chrono::milliseconds(1000));
    
    /**
     * @brief Stop monitoring
     */
    void stopMonitoring();
    
    /**
     * @brief Get current resource statistics
     */
    ResourceStats getCurrentStats() const;
    
    /**
     * @brief Record batch processing metrics
     */
    void recordBatchProcessing(size_t batch_size, double latency_ms);
    
    /**
     * @brief Record queue depth
     */
    void recordQueueDepth(size_t depth);
    
    /**
     * @brief Print performance summary
     */
    void printSummary() const;

private:
    ResourceMonitor() = default;
    ~ResourceMonitor();
    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;
    
    void monitoringLoop();
    double getCPUUsage() const;
    double getGPUUsage() const;
    size_t getMemoryUsage() const;
    size_t getGPUMemoryUsage() const;
    
    std::atomic<bool> monitoring_active_{false};
    std::unique_ptr<std::thread> monitoring_thread_;
    
    // Performance metrics
    std::atomic<size_t> total_batches_{0};
    std::atomic<size_t> total_requests_{0};
    std::atomic<uint64_t> total_latency_ms_x1000_{0}; // Store latency * 1000 to avoid floating point atomic issues
    std::atomic<size_t> current_queue_depth_{0};
    
    mutable std::mutex stats_mutex_;
    ResourceStats cached_stats_;
    std::chrono::steady_clock::time_point last_update_;
};

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_UTILS_RESOURCE_MONITOR_H