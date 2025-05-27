#include "utils/advanced_memory_monitor.h"
#include "utils/progress_bar.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>

#ifdef TORCH_CUDA
#include <cuda_runtime.h>
#include <torch/torch.h>
#endif

namespace alphazero {
namespace utils {

void AdvancedMemoryMonitor::startMonitoring(const std::string& log_file) {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    
    if (monitoring_active_.load()) {
        return; // Already monitoring
    }
    
    log_file_.open(log_file);
    if (!log_file_.is_open()) {
        std::cerr << "WARNING: Failed to open memory monitor log file: " << log_file << std::endl;
    }
    
    start_time_ = std::chrono::steady_clock::now();
    monitoring_active_.store(true);
    
    monitoring_thread_ = std::thread(&AdvancedMemoryMonitor::monitoringLoop, this);
    
    if (log_file_.is_open()) {
        log_file_ << "timestamp_ms,total_memory_mb,used_memory_mb,free_memory_mb,gpu_memory_mb,gpu_free_mb,cpu_usage_percent,gpu_usage_percent,context" << std::endl;
    }
    
    auto& progress_manager = utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        std::cout << "[MEMORY_MONITOR] Started advanced memory monitoring with log: " << log_file << std::endl;
    }
}

void AdvancedMemoryMonitor::stopMonitoring() {
    // Use a separate mutex for stop to avoid deadlock
    static std::mutex stop_mutex;
    std::lock_guard<std::mutex> lock(stop_mutex);
    
    if (!monitoring_active_.load()) {
        return;
    }
    
    monitoring_active_.store(false);
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    if (log_file_.is_open()) {
        log_file_.close();
    }
    
    auto& progress_manager = utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        std::cout << "[MEMORY_MONITOR] Stopped monitoring" << std::endl;
    }
}

void AdvancedMemoryMonitor::captureSnapshot(const std::string& context) {
    MemorySnapshot snapshot = captureSystemSnapshot();
    snapshot.context = context;
    
    {
        std::lock_guard<std::mutex> lock(snapshots_mutex_);
        snapshots_.push_back(snapshot);
        
        // Keep only last 1000 snapshots to prevent memory bloat
        if (snapshots_.size() > 1000) {
            snapshots_.erase(snapshots_.begin());
        }
    }
    
    // Log to file if available
    if (log_file_.is_open()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            snapshot.timestamp - start_time_).count();
        
        log_file_ << elapsed << "," 
                  << snapshot.total_memory_mb << ","
                  << snapshot.used_memory_mb << ","
                  << snapshot.free_memory_mb << ","
                  << snapshot.gpu_memory_mb << ","
                  << snapshot.gpu_free_mb << ","
                  << std::fixed << std::setprecision(2) << snapshot.cpu_usage_percent << ","
                  << std::fixed << std::setprecision(2) << snapshot.gpu_usage_percent << ","
                  << "\"" << context << "\"" << std::endl;
        log_file_.flush();
    }
}

void AdvancedMemoryMonitor::logEvent(const std::string& event) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
    
    auto& progress_manager = utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        std::cout << "[MEMORY_MONITOR][+" << elapsed << "ms] " << event << std::endl;
    }
    
    if (log_file_.is_open()) {
        log_file_ << elapsed << ",,,,,,,\"EVENT: " << event << "\"" << std::endl;
        log_file_.flush();
    }
}

bool AdvancedMemoryMonitor::isMemoryPressureHigh() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    if (snapshots_.empty()) {
        return false;
    }
    
    const auto& latest = snapshots_.back();
    double usage_ratio = static_cast<double>(latest.used_memory_mb) / latest.total_memory_mb;
    return usage_ratio > 0.85; // High pressure if >85% used
}

bool AdvancedMemoryMonitor::isGPUMemoryPressureHigh() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    if (snapshots_.empty()) {
        return false;
    }
    
    const auto& latest = snapshots_.back();
    if (latest.gpu_memory_mb == 0) {
        return false; // No GPU
    }
    
    double gpu_used = latest.gpu_memory_mb - latest.gpu_free_mb;
    double usage_ratio = gpu_used / latest.gpu_memory_mb;
    return usage_ratio > 0.90; // High pressure if >90% used
}

double AdvancedMemoryMonitor::getAverageMemoryUsage() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    if (snapshots_.empty()) {
        return 0.0;
    }
    
    double total_usage = 0.0;
    for (const auto& snapshot : snapshots_) {
        double usage_ratio = static_cast<double>(snapshot.used_memory_mb) / snapshot.total_memory_mb;
        total_usage += usage_ratio;
    }
    
    return total_usage / snapshots_.size();
}

double AdvancedMemoryMonitor::getAverageGPUUsage() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    if (snapshots_.empty()) {
        return 0.0;
    }
    
    double total_gpu_usage = 0.0;
    int valid_samples = 0;
    
    for (const auto& snapshot : snapshots_) {
        if (snapshot.gpu_memory_mb > 0) {
            total_gpu_usage += snapshot.gpu_usage_percent;
            valid_samples++;
        }
    }
    
    return valid_samples > 0 ? total_gpu_usage / valid_samples : 0.0;
}

void AdvancedMemoryMonitor::monitoringLoop() {
    while (monitoring_active_.load()) {
        captureSnapshot("periodic");
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Sample every second
    }
}

AdvancedMemoryMonitor::MemorySnapshot AdvancedMemoryMonitor::captureSystemSnapshot() const {
    MemorySnapshot snapshot;
    snapshot.timestamp = std::chrono::steady_clock::now();
    
    // Get system memory info
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        snapshot.total_memory_mb = sys_info.totalram / (1024 * 1024);
        snapshot.free_memory_mb = sys_info.freeram / (1024 * 1024);
        snapshot.used_memory_mb = snapshot.total_memory_mb - snapshot.free_memory_mb;
    }
    
    // Get CPU usage (simplified)
    static unsigned long long prev_idle = 0, prev_total = 0;
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open()) {
        std::string cpu;
        unsigned long long user, nice, system, idle, iowait, irq, softirq;
        stat_file >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
        
        unsigned long long total = user + nice + system + idle + iowait + irq + softirq;
        unsigned long long diff_idle = idle - prev_idle;
        unsigned long long diff_total = total - prev_total;
        
        if (diff_total > 0) {
            snapshot.cpu_usage_percent = 100.0 * (diff_total - diff_idle) / diff_total;
        }
        
        prev_idle = idle;
        prev_total = total;
        stat_file.close();
    }
    
    // Get GPU memory info
#ifdef TORCH_CUDA
    if (torch::cuda::is_available()) {
        try {
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            
            snapshot.gpu_memory_mb = total_bytes / (1024 * 1024);
            snapshot.gpu_free_mb = free_bytes / (1024 * 1024);
            
            // Simple GPU usage estimation (this is crude but better than nothing)
            double gpu_used_ratio = 1.0 - (static_cast<double>(free_bytes) / total_bytes);
            snapshot.gpu_usage_percent = gpu_used_ratio * 100.0;
        } catch (...) {
            // GPU info not available
        }
    }
#endif
    
    return snapshot;
}

} // namespace utils
} // namespace alphazero