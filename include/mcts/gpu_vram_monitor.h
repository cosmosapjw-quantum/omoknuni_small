// include/mcts/gpu_vram_monitor.h
#ifndef ALPHAZERO_GPU_VRAM_MONITOR_H
#define ALPHAZERO_GPU_VRAM_MONITOR_H

#ifdef WITH_TORCH
#include <cuda_runtime.h>
#include <torch/torch.h>
#endif
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include "core/export_macros.h"
#include "utils/logger.h"

namespace alphazero {
namespace mcts {

#ifdef WITH_TORCH
/**
 * @brief GPU VRAM Monitor for tracking and managing GPU memory usage
 * 
 * Features:
 * - Real-time VRAM usage monitoring
 * - Memory pressure detection
 * - Automatic cleanup triggers
 * - Peak usage tracking
 * - Memory leak detection
 */
class ALPHAZERO_API GPUVRAMMonitor {
public:
    struct VRAMStats {
        size_t total_memory;        // Total GPU memory
        size_t free_memory;         // Currently free memory
        size_t used_memory;         // Currently used memory
        size_t allocated_memory;    // PyTorch allocated memory
        size_t reserved_memory;     // PyTorch reserved memory
        size_t peak_memory;         // Peak usage since last reset
        float usage_percentage;     // Usage percentage
        bool under_pressure;        // Memory pressure flag
    };

    struct MonitorConfig {
        float warning_threshold;
        float critical_threshold;
        float emergency_threshold;
        int check_interval_ms;
        bool enable_auto_cleanup;
        bool enable_logging;
        size_t target_free_mb;
        
        // Constructor with defaults
        MonitorConfig() : 
            warning_threshold(0.8f),
            critical_threshold(0.9f),
            emergency_threshold(0.95f),
            check_interval_ms(100),
            enable_auto_cleanup(true),
            enable_logging(true),
            target_free_mb(1024) {}
    };

    explicit GPUVRAMMonitor(const MonitorConfig& config = MonitorConfig());
    ~GPUVRAMMonitor();

    // Start/stop monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return monitoring_.load(); }

    // Get current stats
    VRAMStats getCurrentStats() const;
    VRAMStats getPeakStats() const { return peak_stats_; }
    void resetPeakStats();

    // Memory pressure checks
    bool isUnderPressure() const;
    bool isCritical() const;
    bool isEmergency() const;

    // Manual cleanup triggers
    void triggerCleanup(bool force = false);
    void clearPyTorchCache();
    void synchronizeAndClean();

    // Callbacks for memory events
    using MemoryCallback = std::function<void(const VRAMStats&)>;
    void setWarningCallback(MemoryCallback callback) { warning_callback_ = callback; }
    void setCriticalCallback(MemoryCallback callback) { critical_callback_ = callback; }
    void setEmergencyCallback(MemoryCallback callback) { emergency_callback_ = callback; }

    // Memory allocation tracking
    void recordAllocation(size_t bytes, const std::string& tag = "");
    void recordDeallocation(size_t bytes, const std::string& tag = "");
    void printAllocationSummary() const;

private:
    MonitorConfig config_;
    std::atomic<bool> monitoring_{false};
    std::thread monitor_thread_;
    mutable std::mutex stats_mutex_;
    
    // Current and peak stats
    VRAMStats current_stats_;
    VRAMStats peak_stats_;
    
    // Memory pressure tracking
    std::chrono::steady_clock::time_point last_pressure_time_;
    std::chrono::steady_clock::time_point last_cleanup_time_;
    int consecutive_pressure_count_ = 0;
    
    // Callbacks
    MemoryCallback warning_callback_;
    MemoryCallback critical_callback_;
    MemoryCallback emergency_callback_;
    
    // Allocation tracking
    struct AllocationInfo {
        std::atomic<size_t> allocated{0};
        std::atomic<size_t> deallocated{0};
        std::atomic<size_t> peak{0};
        std::atomic<int> count{0};
    };
    std::unordered_map<std::string, AllocationInfo> allocations_;
    mutable std::mutex allocation_mutex_;
    
    // Internal methods
    void monitoringLoop();
    VRAMStats queryVRAMStats() const;
    void handleMemoryPressure(const VRAMStats& stats);
    void logMemoryEvent(const std::string& event, const VRAMStats& stats) const;
};

/**
 * @brief RAII wrapper for tracking tensor allocations
 */
class ALPHAZERO_API VRAMAllocationScope {
public:
    VRAMAllocationScope(GPUVRAMMonitor& monitor, size_t bytes, const std::string& tag)
        : monitor_(monitor), bytes_(bytes), tag_(tag) {
        monitor_.recordAllocation(bytes_, tag_);
    }
    
    ~VRAMAllocationScope() {
        monitor_.recordDeallocation(bytes_, tag_);
    }
    
private:
    GPUVRAMMonitor& monitor_;
    size_t bytes_;
    std::string tag_;
};

/**
 * @brief Global VRAM monitor instance
 */
class ALPHAZERO_API GlobalVRAMMonitor {
public:
    static GPUVRAMMonitor& getInstance() {
        static GPUVRAMMonitor instance;
        return instance;
    }
    
    static void initialize(const GPUVRAMMonitor::MonitorConfig& config) {
        static std::once_flag init_flag;
        std::call_once(init_flag, [&config]() {
            getInstance().~GPUVRAMMonitor();
            new (&getInstance()) GPUVRAMMonitor(config);
        });
    }
};

#else // !WITH_TORCH
// Dummy classes when torch is not available
class ALPHAZERO_API GPUVRAMMonitor {
public:
    struct VRAMStats {};
    struct MonitorConfig {};
    GPUVRAMMonitor(const MonitorConfig& = {}) {}
    void startMonitoring() {}
    void stopMonitoring() {}
    bool isMonitoring() const { return false; }
    VRAMStats getCurrentStats() const { return {}; }
    VRAMStats getPeakStats() const { return {}; }
    void resetPeakStats() {}
    bool isUnderPressure() const { return false; }
    bool isCritical() const { return false; }
    bool isEmergency() const { return false; }
    void triggerCleanup(bool = false) {}
    void clearPyTorchCache() {}
    void synchronizeAndClean() {}
    void recordAllocation(size_t, const std::string& = "") {}
    void recordDeallocation(size_t, const std::string& = "") {}
    void printAllocationSummary() const {}
};

class ALPHAZERO_API VRAMAllocationScope {
public:
    VRAMAllocationScope(GPUVRAMMonitor&, size_t, const std::string&) {}
};

class ALPHAZERO_API GlobalVRAMMonitor {
public:
    static GPUVRAMMonitor& getInstance() {
        static GPUVRAMMonitor instance;
        return instance;
    }
    static void initialize(const GPUVRAMMonitor::MonitorConfig&) {}
};
#endif // WITH_TORCH

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_GPU_VRAM_MONITOR_H