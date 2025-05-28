// src/mcts/aggressive_memory_manager.cpp
// AGGRESSIVE MEMORY MANAGER WITH COMPREHENSIVE TRACKING

#include "mcts/aggressive_memory_manager.h"
#include "utils/debug_logger.h"
#include <sys/resource.h>
#include <unistd.h>
#include <malloc.h>  // For malloc_trim
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

// Static instance
std::unique_ptr<AggressiveMemoryManager> AggressiveMemoryManager::instance_;
std::mutex AggressiveMemoryManager::instance_mutex_;

AggressiveMemoryManager& AggressiveMemoryManager::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::unique_ptr<AggressiveMemoryManager>(new AggressiveMemoryManager());
    }
    return *instance_;
}

AggressiveMemoryManager::AggressiveMemoryManager() 
    : shutdown_(false),
      current_memory_usage_(0),
      peak_memory_usage_(0),
      last_cleanup_time_(std::chrono::steady_clock::now()) {
    
    // Initialize memory tracking
    updateMemoryUsage();
    peak_memory_usage_ = current_memory_usage_.load();
    
    // Start monitoring thread
    monitor_thread_ = std::thread(&AggressiveMemoryManager::monitoringLoop, this);
    
}

void AggressiveMemoryManager::shutdown() {
    shutdown_ = true;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

AggressiveMemoryManager::~AggressiveMemoryManager() {
    // Ensure shutdown was called
    shutdown();
}

void AggressiveMemoryManager::trackAllocation(const std::string& category, size_t bytes, 
                                            const std::string& location) {
    auto now = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        
        // Update category tracking
        allocation_by_category_[category] += bytes;
        
        // Add to recent allocations for tracking
        recent_allocations_.push_back({category, bytes, location, now});
        
        // Keep only recent allocations (last 1000)
        if (recent_allocations_.size() > 1000) {
            recent_allocations_.pop_front();
        }
        
        // Track allocation rate
        allocation_events_.push_back({bytes, now});
        
        // Remove old events (older than 10 seconds)
        auto cutoff = now - std::chrono::seconds(10);
        allocation_events_.erase(
            std::remove_if(allocation_events_.begin(), allocation_events_.end(),
                          [cutoff](const auto& event) { return event.timestamp < cutoff; }),
            allocation_events_.end()
        );
    }
    
    // Update memory usage
    updateMemoryUsage();
    
    // Log significant allocations
    if (bytes > 10 * 1024 * 1024) { // > 10MB
    }
}

void AggressiveMemoryManager::trackDeallocation(const std::string& category, size_t bytes) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    auto it = allocation_by_category_.find(category);
    if (it != allocation_by_category_.end()) {
        if (it->second >= bytes) {
            it->second -= bytes;
        } else {
            it->second = 0;
        }
        
        if (it->second == 0) {
            allocation_by_category_.erase(it);
        }
    }
}

size_t AggressiveMemoryManager::getCurrentMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss) * 1024; // Convert KB to bytes on Linux
}

size_t AggressiveMemoryManager::getAvailableSystemMemory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t available = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            iss >> label >> value >> unit;
            available = value * 1024; // Convert KB to bytes
            break;
        }
    }
    
    return available;
}

void AggressiveMemoryManager::updateMemoryUsage() {
    size_t current = getCurrentMemoryUsage();
    current_memory_usage_ = current;
    
    // Update peak
    size_t peak = peak_memory_usage_.load();
    while (current > peak && !peak_memory_usage_.compare_exchange_weak(peak, current)) {}
    
    // CRITICAL FIX: Also monitor GPU memory usage
    size_t gpu_memory_bytes = 0;
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        try {
            // Get allocated memory from PyTorch's allocator
            auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            gpu_memory_bytes = stats.allocated_bytes[0].current;
            
            // Also try to get actual device memory usage
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            size_t used_bytes = total_bytes - free_bytes;
            
            // Use the larger of the two values
            gpu_memory_bytes = std::max(gpu_memory_bytes, used_bytes);
        } catch (...) {
            // Ignore errors in GPU memory query
        }
    }
    #endif
    
    // Store GPU memory for reporting
    gpu_memory_usage_ = gpu_memory_bytes;
    
    // Calculate total memory pressure including GPU
    size_t total_memory = current + gpu_memory_bytes;
    double usage_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
    
    if (usage_gb >= config_.emergency_threshold_gb) {
        current_pressure_level_ = PressureLevel::EMERGENCY;
    } else if (usage_gb >= config_.critical_threshold_gb) {
        current_pressure_level_ = PressureLevel::CRITICAL;
    } else if (usage_gb >= config_.warning_threshold_gb) {
        current_pressure_level_ = PressureLevel::WARNING;
    } else {
        current_pressure_level_ = PressureLevel::NORMAL;
    }
}

AggressiveMemoryManager::PressureLevel AggressiveMemoryManager::getMemoryPressure() {
    updateMemoryUsage();
    return current_pressure_level_.load();
}

void AggressiveMemoryManager::registerCleanupCallback(const std::string& name, 
                                                     CleanupCallback callback,
                                                     int priority) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    cleanup_callbacks_.push_back({name, callback, priority});
    
    // Sort by priority (higher priority first)
    std::sort(cleanup_callbacks_.begin(), cleanup_callbacks_.end(),
              [](const auto& a, const auto& b) { return a.priority > b.priority; });
}

void AggressiveMemoryManager::forceCleanup(PressureLevel min_level) {
    auto level = getMemoryPressure();
    if (level < min_level) {
        return;
    }
    
    
    size_t memory_before = getCurrentMemoryUsage();
    
    // Execute cleanup callbacks
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback_info : cleanup_callbacks_) {
            try {
                callback_info.callback(level);
            } catch (const std::exception& e) {
                std::cerr << "  Cleanup error in " << callback_info.name << ": " << e.what() << std::endl;
            }
        }
    }
    
    // Force system cleanup
    #ifdef WITH_TORCH
    if (level >= PressureLevel::WARNING && torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    #endif
    
    // Aggressive malloc trim
    if (level >= PressureLevel::CRITICAL) {
        malloc_trim(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    size_t memory_after = getCurrentMemoryUsage();
    size_t freed = (memory_before > memory_after) ? (memory_before - memory_after) : 0;
    
    
    last_cleanup_time_ = std::chrono::steady_clock::now();
    cleanup_stats_.total_cleanups++;
    cleanup_stats_.total_memory_freed += freed;
}

double AggressiveMemoryManager::getCurrentMemoryUsageGB() const {
    return current_memory_usage_.load() / (1024.0 * 1024.0 * 1024.0);
}

std::string AggressiveMemoryManager::getMemoryReport() {
    std::ostringstream report;
    
    updateMemoryUsage();
    
    report << "\n=== MEMORY REPORT ===\n";
    report << "Current RAM Usage: " << formatBytes(current_memory_usage_) 
           << " (" << std::fixed << std::setprecision(1) 
           << (current_memory_usage_ / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
    report << "Current GPU Usage: " << formatBytes(gpu_memory_usage_) 
           << " (" << std::fixed << std::setprecision(1) 
           << (gpu_memory_usage_ / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
    report << "Peak RAM Usage: " << formatBytes(peak_memory_usage_) << "\n";
    report << "Pressure Level: ";
    
    switch (current_pressure_level_.load()) {
        case PressureLevel::NORMAL: report << "NORMAL ✅\n"; break;
        case PressureLevel::WARNING: report << "WARNING ⚠️\n"; break;
        case PressureLevel::CRITICAL: report << "CRITICAL 🚨\n"; break;
        case PressureLevel::EMERGENCY: report << "EMERGENCY 🔥\n"; break;
    }
    
    // Allocation by category
    report << "\nAllocations by Category:\n";
    {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        std::vector<std::pair<std::string, size_t>> sorted_categories;
        for (const auto& [category, bytes] : allocation_by_category_) {
            sorted_categories.push_back({category, bytes});
        }
        std::sort(sorted_categories.begin(), sorted_categories.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& [category, bytes] : sorted_categories) {
            if (bytes > 1024 * 1024) { // Only show > 1MB
                report << "  " << std::setw(20) << std::left << category 
                       << ": " << formatBytes(bytes) << "\n";
            }
        }
    }
    
    // Allocation rate
    double allocation_rate = calculateAllocationRate();
    report << "\nAllocation Rate: " << formatBytes(static_cast<size_t>(allocation_rate)) 
           << "/sec\n";
    
    // Cleanup statistics
    report << "\nCleanup Stats:\n";
    report << "  Total Cleanups: " << cleanup_stats_.total_cleanups << "\n";
    report << "  Total Freed: " << formatBytes(cleanup_stats_.total_memory_freed) << "\n";
    
    return report.str();
}

void AggressiveMemoryManager::monitoringLoop() {
    while (!shutdown_) {
        auto now = std::chrono::steady_clock::now();
        
        // Update memory usage
        updateMemoryUsage();
        
        // Check if cleanup is needed
        auto pressure = getMemoryPressure();
        auto time_since_cleanup = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_cleanup_time_).count();
        
        bool should_cleanup = false;
        
        switch (pressure) {
            case PressureLevel::EMERGENCY:
                should_cleanup = true; // Always cleanup in emergency
                break;
            case PressureLevel::CRITICAL:
                should_cleanup = (time_since_cleanup >= 5); // Every 5 seconds
                break;
            case PressureLevel::WARNING:
                should_cleanup = (time_since_cleanup >= 30); // Every 30 seconds
                break;
            default:
                should_cleanup = false;
        }
        
        if (should_cleanup) {
            forceCleanup(pressure);
        }
        
        // Log periodic summary
        static auto last_summary = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_summary).count() >= 10) {
            if (pressure >= PressureLevel::WARNING) {
            }
            last_summary = now;
        }
        
        // Sleep based on pressure level
        int sleep_ms = 1000; // Normal
        if (pressure == PressureLevel::EMERGENCY) sleep_ms = 100;
        else if (pressure == PressureLevel::CRITICAL) sleep_ms = 250;
        else if (pressure == PressureLevel::WARNING) sleep_ms = 500;
        
        // Use interruptible sleep
        auto sleep_duration = std::chrono::milliseconds(sleep_ms);
        auto sleep_start = std::chrono::steady_clock::now();
        
        // Check for shutdown every 100ms
        while (!shutdown_ && 
               std::chrono::steady_clock::now() - sleep_start < sleep_duration) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

double AggressiveMemoryManager::calculateAllocationRate() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    if (allocation_events_.empty()) return 0.0;
    
    auto now = std::chrono::steady_clock::now();
    auto window_start = now - std::chrono::seconds(10);
    
    size_t total_bytes = 0;
    int count = 0;
    
    for (const auto& event : allocation_events_) {
        if (event.timestamp >= window_start) {
            total_bytes += event.bytes;
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        allocation_events_.back().timestamp - allocation_events_.front().timestamp).count();
    
    if (duration == 0) return 0.0;
    
    return (total_bytes * 1000.0) / duration; // Bytes per second
}

std::string AggressiveMemoryManager::formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

} // namespace mcts
} // namespace alphazero