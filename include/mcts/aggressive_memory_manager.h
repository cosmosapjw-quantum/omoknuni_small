// include/mcts/aggressive_memory_manager.h
#ifndef ALPHAZERO_MCTS_AGGRESSIVE_MEMORY_MANAGER_H
#define ALPHAZERO_MCTS_AGGRESSIVE_MEMORY_MANAGER_H

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Aggressive memory manager with comprehensive tracking
 * 
 * This singleton class monitors memory usage, tracks allocations by category,
 * and performs aggressive cleanup when memory pressure is detected.
 */
class ALPHAZERO_API AggressiveMemoryManager {
public:
    enum class PressureLevel {
        NORMAL = 0,
        WARNING = 1,
        CRITICAL = 2,
        EMERGENCY = 3
    };
    
    struct Config {
        double warning_threshold_gb = 28.0;   // 28GB warning
        double critical_threshold_gb = 35.0;  // 35GB critical  
        double emergency_threshold_gb = 40.0; // 40GB emergency
        std::chrono::seconds cleanup_interval{30};
        bool auto_cleanup = true;
        bool track_allocations = true;
    };
    
    using CleanupCallback = std::function<void(PressureLevel)>;
    
    // Singleton access
    static AggressiveMemoryManager& getInstance();
    
    // Memory tracking
    void trackAllocation(const std::string& category, size_t bytes, 
                        const std::string& location = "");
    void trackDeallocation(const std::string& category, size_t bytes);
    
    // Memory status
    size_t getCurrentMemoryUsage();
    size_t getPeakMemoryUsage() const { return peak_memory_usage_; }
    PressureLevel getMemoryPressure();
    double getCurrentMemoryUsageGB() const;
    
    // Cleanup management
    void registerCleanupCallback(const std::string& name, CleanupCallback callback, 
                                int priority = 0);
    void forceCleanup(PressureLevel min_level = PressureLevel::WARNING);
    
    // Reporting
    std::string getMemoryReport();
    std::map<std::string, size_t> getAllocationsByCategory() const {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        return allocation_by_category_;
    }
    
    // Configuration
    void setConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }
    
    // Utility
    static std::string formatBytes(size_t bytes);
    
public:
    ~AggressiveMemoryManager();
    
private:
    AggressiveMemoryManager();
    
    // Singleton instance
    static std::unique_ptr<AggressiveMemoryManager> instance_;
    static std::mutex instance_mutex_;
    
    // Configuration
    Config config_;
    
    // Memory tracking
    std::atomic<size_t> current_memory_usage_;
    std::atomic<size_t> peak_memory_usage_;
    std::atomic<PressureLevel> current_pressure_level_{PressureLevel::NORMAL};
    
    // Allocation tracking
    mutable std::mutex allocations_mutex_;
    std::map<std::string, size_t> allocation_by_category_;
    
    struct AllocationEvent {
        std::string category;
        size_t bytes;
        std::string location;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::deque<AllocationEvent> recent_allocations_;
    
    // Allocation rate tracking
    struct AllocationRateEvent {
        size_t bytes;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<AllocationRateEvent> allocation_events_;
    
    // Cleanup callbacks
    std::mutex callbacks_mutex_;
    struct CallbackInfo {
        std::string name;
        CleanupCallback callback;
        int priority;
    };
    std::vector<CallbackInfo> cleanup_callbacks_;
    
    // Cleanup tracking
    std::chrono::steady_clock::time_point last_cleanup_time_;
    struct CleanupStats {
        std::atomic<int> total_cleanups{0};
        std::atomic<size_t> total_memory_freed{0};
    } cleanup_stats_;
    
    // Monitoring thread
    std::thread monitor_thread_;
    std::atomic<bool> shutdown_;
    
    // Helper methods
    void updateMemoryUsage();
    size_t getAvailableSystemMemory();
    void monitoringLoop();
    double calculateAllocationRate();
};

// RAII Memory tracking helper
class ALPHAZERO_API MemoryTracker {
public:
    MemoryTracker(const std::string& category, size_t bytes, const std::string& location = "")
        : category_(category), bytes_(bytes), tracked_(true) {
        AggressiveMemoryManager::getInstance().trackAllocation(category, bytes, location);
    }
    
    ~MemoryTracker() {
        if (tracked_) {
            AggressiveMemoryManager::getInstance().trackDeallocation(category_, bytes_);
        }
    }
    
    // Move constructor
    MemoryTracker(MemoryTracker&& other) noexcept
        : category_(std::move(other.category_)), 
          bytes_(other.bytes_), 
          tracked_(other.tracked_) {
        other.tracked_ = false;
    }
    
    // Disable copy
    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;
    
private:
    std::string category_;
    size_t bytes_;
    bool tracked_;
};

// Convenience macros for memory tracking
#define TRACK_MEMORY(category, bytes) \
    alphazero::mcts::MemoryTracker _memory_tracker_##__LINE__(category, bytes, __FILE__ ":" + std::to_string(__LINE__))

#define TRACK_MEMORY_ALLOC(category, bytes) \
    alphazero::mcts::AggressiveMemoryManager::getInstance().trackAllocation(category, bytes, __FILE__ ":" + std::to_string(__LINE__))

#define TRACK_MEMORY_FREE(category, bytes) \
    alphazero::mcts::AggressiveMemoryManager::getInstance().trackDeallocation(category, bytes)

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_AGGRESSIVE_MEMORY_MANAGER_H