#pragma once

#include <iostream>
#include <chrono>
#include <atomic>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

namespace alphazero {
namespace utils {

/**
 * @brief Advanced memory and resource monitoring system for performance diagnosis
 */
class AdvancedMemoryMonitor {
public:
    struct MemorySnapshot {
        size_t total_memory_mb = 0;
        size_t used_memory_mb = 0;
        size_t free_memory_mb = 0;
        size_t gpu_memory_mb = 0;
        size_t gpu_free_mb = 0;
        double cpu_usage_percent = 0.0;
        double gpu_usage_percent = 0.0;
        std::chrono::steady_clock::time_point timestamp;
        std::string context;
    };
    
    static AdvancedMemoryMonitor& getInstance() {
        static AdvancedMemoryMonitor instance;
        return instance;
    }
    
    void startMonitoring(const std::string& log_file = "resource_monitor.log");
    void stopMonitoring();
    void captureSnapshot(const std::string& context = "");
    void logEvent(const std::string& event);
    
    // Memory pressure detection
    bool isMemoryPressureHigh() const;
    bool isGPUMemoryPressureHigh() const;
    
    // Performance metrics
    double getAverageMemoryUsage() const;
    double getAverageGPUUsage() const;
    
private:
    AdvancedMemoryMonitor() = default;
    ~AdvancedMemoryMonitor() { stopMonitoring(); }
    
    void monitoringLoop();
    MemorySnapshot captureSystemSnapshot() const;
    
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
    mutable std::mutex snapshots_mutex_;
    std::vector<MemorySnapshot> snapshots_;
    std::ofstream log_file_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace utils
} // namespace alphazero