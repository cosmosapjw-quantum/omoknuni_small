// src/mcts/gpu_vram_monitor.cpp
#include "mcts/gpu_vram_monitor.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <sstream>
#include <iomanip>

namespace alphazero {
namespace mcts {

GPUVRAMMonitor::GPUVRAMMonitor(const MonitorConfig& config)
    : config_(config) {
    current_stats_ = queryVRAMStats();
    peak_stats_ = current_stats_;
}

GPUVRAMMonitor::~GPUVRAMMonitor() {
    stopMonitoring();
}

void GPUVRAMMonitor::startMonitoring() {
    if (monitoring_.exchange(true)) {
        return; // Already monitoring
    }
    
    monitor_thread_ = std::thread(&GPUVRAMMonitor::monitoringLoop, this);
    // LOG_SYSTEM_INFO("VRAM monitoring started with {}ms interval", config_.check_interval_ms);
}

void GPUVRAMMonitor::stopMonitoring() {
    if (!monitoring_.exchange(false)) {
        return; // Not monitoring
    }
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    // LOG_SYSTEM_INFO("VRAM monitoring stopped");
}

GPUVRAMMonitor::VRAMStats GPUVRAMMonitor::getCurrentStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_;
}

void GPUVRAMMonitor::resetPeakStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    peak_stats_ = current_stats_;
}

bool GPUVRAMMonitor::isUnderPressure() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_.under_pressure;
}

bool GPUVRAMMonitor::isCritical() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_.usage_percentage >= config_.critical_threshold;
}

bool GPUVRAMMonitor::isEmergency() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_.usage_percentage >= config_.emergency_threshold;
}

void GPUVRAMMonitor::triggerCleanup(bool force) {
    auto now = std::chrono::steady_clock::now();
    auto time_since_cleanup = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_cleanup_time_).count();
    
    // Avoid cleanup storms
    if (!force && time_since_cleanup < 1000) {
        return;
    }
    
    last_cleanup_time_ = now;
    
    // LOG_SYSTEM_INFO("Triggering VRAM cleanup (force={})", force);
    
    // Clear PyTorch caches
    clearPyTorchCache();
    
    // Synchronize and clean
    synchronizeAndClean();
    
    // Update stats after cleanup
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        current_stats_ = queryVRAMStats();
    }
}

void GPUVRAMMonitor::clearPyTorchCache() {
    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
        // LOG_SYSTEM_DEBUG("PyTorch CUDA cache cleared");
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to clear PyTorch cache: {}", e.what());
    }
}

void GPUVRAMMonitor::synchronizeAndClean() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_SYSTEM_ERROR("CUDA synchronize failed: {}", cudaGetErrorString(err));
        return;
    }
    
    // Additional cleanup can be added here
}

void GPUVRAMMonitor::recordAllocation(size_t bytes, const std::string& tag) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    auto& info = allocations_[tag];
    info.allocated.fetch_add(bytes);
    info.count.fetch_add(1);
    
    size_t current = info.allocated.load() - info.deallocated.load();
    size_t peak = info.peak.load();
    while (current > peak && !info.peak.compare_exchange_weak(peak, current)) {
        // Retry
    }
}

void GPUVRAMMonitor::recordDeallocation(size_t bytes, const std::string& tag) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    auto it = allocations_.find(tag);
    if (it != allocations_.end()) {
        it->second.deallocated.fetch_add(bytes);
    }
}

void GPUVRAMMonitor::printAllocationSummary() const {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // LOG_SYSTEM_INFO("=== VRAM Allocation Summary ===");
    for (const auto& [tag, info] : allocations_) {
        size_t allocated = info.allocated.load();
        size_t deallocated = info.deallocated.load();
        size_t current = allocated - deallocated;
        size_t peak = info.peak.load();
        
        // LOG_SYSTEM_INFO("{}: current={:.1f}MB, peak={:.1f}MB, total_allocs={}",
        //                tag, 
        //                current / (1024.0 * 1024.0),
        //                peak / (1024.0 * 1024.0),
        //                info.count.load());
    }
}

void GPUVRAMMonitor::monitoringLoop() {
    while (monitoring_.load()) {
        auto stats = queryVRAMStats();
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            current_stats_ = stats;
            
            // Update peak stats
            if (stats.used_memory > peak_stats_.used_memory) {
                peak_stats_ = stats;
            }
        }
        
        // Check for memory pressure
        if (stats.usage_percentage >= config_.warning_threshold) {
            handleMemoryPressure(stats);
        } else {
            consecutive_pressure_count_ = 0;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.check_interval_ms));
    }
}

GPUVRAMMonitor::VRAMStats GPUVRAMMonitor::queryVRAMStats() const {
    VRAMStats stats;
    
    // Get CUDA memory info
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        LOG_SYSTEM_ERROR("Failed to query CUDA memory: {}", cudaGetErrorString(err));
        return stats;
    }
    
    stats.total_memory = total_bytes;
    stats.free_memory = free_bytes;
    stats.used_memory = total_bytes - free_bytes;
    
    // Get PyTorch allocator stats
    try {
        // Try to get PyTorch memory stats
        try {
            c10::cuda::CUDACachingAllocator::DeviceStats pytorch_stats = 
                c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            
            stats.allocated_memory = pytorch_stats.allocated_bytes[0].current;
            stats.reserved_memory = pytorch_stats.reserved_bytes[0].current;
        } catch (...) {
            // Fallback if new API not available
            stats.allocated_memory = 0;
            stats.reserved_memory = 0;
        }
    } catch (const std::exception& e) {
        // LOG_SYSTEM_DEBUG("Failed to get PyTorch stats: {}", e.what());
    }
    
    stats.usage_percentage = static_cast<float>(stats.used_memory) / stats.total_memory;
    stats.under_pressure = stats.free_memory < (config_.target_free_mb * 1024 * 1024);
    
    return stats;
}

void GPUVRAMMonitor::handleMemoryPressure(const VRAMStats& stats) {
    consecutive_pressure_count_++;
    
    if (stats.usage_percentage >= config_.emergency_threshold) {
        logMemoryEvent("EMERGENCY", stats);
        if (emergency_callback_) {
            emergency_callback_(stats);
        }
        if (config_.enable_auto_cleanup) {
            triggerCleanup(true);
        }
    } else if (stats.usage_percentage >= config_.critical_threshold) {
        logMemoryEvent("CRITICAL", stats);
        if (critical_callback_) {
            critical_callback_(stats);
        }
        if (config_.enable_auto_cleanup && consecutive_pressure_count_ > 3) {
            triggerCleanup(false);
        }
    } else if (stats.usage_percentage >= config_.warning_threshold) {
        if (consecutive_pressure_count_ == 1) { // Only log first occurrence
            logMemoryEvent("WARNING", stats);
        }
        if (warning_callback_) {
            warning_callback_(stats);
        }
    }
}

void GPUVRAMMonitor::logMemoryEvent(const std::string& event, const VRAMStats& stats) const {
    if (!config_.enable_logging) {
        return;
    }
    
    LOG_SYSTEM_WARN("VRAM {} - Usage: {:.1f}% ({:.1f}/{:.1f} GB), Free: {:.1f} GB, "
                    "PyTorch allocated: {:.1f} GB, reserved: {:.1f} GB",
                    event,
                    stats.usage_percentage * 100,
                    stats.used_memory / (1024.0 * 1024.0 * 1024.0),
                    stats.total_memory / (1024.0 * 1024.0 * 1024.0),
                    stats.free_memory / (1024.0 * 1024.0 * 1024.0),
                    stats.allocated_memory / (1024.0 * 1024.0 * 1024.0),
                    stats.reserved_memory / (1024.0 * 1024.0 * 1024.0));
}

} // namespace mcts
} // namespace alphazero