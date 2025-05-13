#ifndef ALPHAZERO_DEBUG_MONITOR_H
#define ALPHAZERO_DEBUG_MONITOR_H

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <functional>

namespace alphazero {
namespace debug {

// Statistics for a resource
struct ResourceStats {
    std::string name;
    double min_value = 0.0;
    double max_value = 0.0;
    double avg_value = 0.0;
    double current_value = 0.0;
    int64_t samples = 0;
};

// Thread status tracking
struct ThreadStatus {
    std::string thread_id;
    std::string status;
    std::string function;
    std::string details;
    std::chrono::steady_clock::time_point last_update;
};

// System resource monitor that periodically logs CPU, memory, and thread stats
class SystemMonitor {
public:
    static SystemMonitor& instance() {
        static SystemMonitor instance;
        return instance;
    }

    void start(int interval_ms = 1000) {
        if (is_running_.load()) return;
        
        is_running_ = true;
        monitor_thread_ = std::thread(&SystemMonitor::monitorLoop, this, interval_ms);
        
        std::cout << "SystemMonitor started with interval " << interval_ms << "ms" << std::endl;
    }
    
    void stop() {
        if (!is_running_.load()) return;
        
        is_running_ = false;
        cv_.notify_all();
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        
        std::cout << "SystemMonitor stopped" << std::endl;
    }
    
    // Thread tracking
    void updateThreadStatus(const std::string& thread_id, 
                           const std::string& status,
                           const std::string& function = "",
                           const std::string& details = "") {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        auto it = thread_statuses_.find(thread_id);
        if (it != thread_statuses_.end()) {
            it->second.status = status;
            it->second.function = function;
            it->second.details = details;
            it->second.last_update = std::chrono::steady_clock::now();
        } else {
            ThreadStatus ts;
            ts.thread_id = thread_id;
            ts.status = status;
            ts.function = function;
            ts.details = details;
            ts.last_update = std::chrono::steady_clock::now();
            thread_statuses_[thread_id] = ts;
        }
    }
    
    // Record a timing measurement
    void recordTiming(const std::string& name, double value_ms) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = timing_stats_.find(name);
        if (it != timing_stats_.end()) {
            auto& stats = it->second;
            stats.min_value = std::min(stats.min_value, value_ms);
            stats.max_value = std::max(stats.max_value, value_ms);
            stats.avg_value = (stats.avg_value * stats.samples + value_ms) / (stats.samples + 1);
            stats.current_value = value_ms;
            stats.samples++;
        } else {
            ResourceStats stats;
            stats.name = name;
            stats.min_value = value_ms;
            stats.max_value = value_ms;
            stats.avg_value = value_ms;
            stats.current_value = value_ms;
            stats.samples = 1;
            timing_stats_[name] = stats;
        }
    }
    
    // Record a counter increment
    void incrementCounter(const std::string& name, int64_t increment = 1) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = counter_stats_.find(name);
        if (it != counter_stats_.end()) {
            auto& stats = it->second;
            stats.current_value += increment;
            stats.samples++;
        } else {
            ResourceStats stats;
            stats.name = name;
            stats.min_value = increment;
            stats.max_value = increment;
            stats.avg_value = increment;
            stats.current_value = increment;
            stats.samples = 1;
            counter_stats_[name] = stats;
        }
    }
    
    // Record a resource usage value (like CPU/GPU percentage)
    void recordResourceUsage(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = resource_stats_.find(name);
        if (it != resource_stats_.end()) {
            auto& stats = it->second;
            stats.min_value = std::min(stats.min_value, value);
            stats.max_value = std::max(stats.max_value, value);
            stats.avg_value = (stats.avg_value * stats.samples + value) / (stats.samples + 1);
            stats.current_value = value;
            stats.samples++;
        } else {
            ResourceStats stats;
            stats.name = name;
            stats.min_value = value;
            stats.max_value = value;
            stats.avg_value = value;
            stats.current_value = value;
            stats.samples = 1;
            resource_stats_[name] = stats;
        }
    }
    
    // Get the current CPU usage
    double getCpuUsage() {
        // Simple CPU usage calculation based on /proc/stat
        static unsigned long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
        
        std::ifstream stat_file("/proc/stat");
        std::string line;
        std::getline(stat_file, line);
        stat_file.close();
        
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle, totalIrq, totalSoftIrq, totalSteal;
        totalUser = totalUserLow = totalSys = totalIdle = totalIrq = totalSoftIrq = totalSteal = 0;
        
        sscanf(line.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu", 
               &totalUser, &totalUserLow, &totalSys, &totalIdle, &totalIrq, &totalSoftIrq, &totalSteal);
        
        if (lastTotalUser == 0) {
            lastTotalUser = totalUser;
            lastTotalUserLow = totalUserLow;
            lastTotalSys = totalSys;
            lastTotalIdle = totalIdle;
            return 0.0;
        }
        
        unsigned long long totalDelta = (totalUser - lastTotalUser) + 
                                      (totalUserLow - lastTotalUserLow) + 
                                      (totalSys - lastTotalSys);
        unsigned long long idleDelta = totalIdle - lastTotalIdle;
        unsigned long long totalCpu = totalDelta + idleDelta;
        
        double percent = 0.0;
        if (totalCpu > 0) {
            percent = 100.0 * totalDelta / totalCpu;
        }
        
        lastTotalUser = totalUser;
        lastTotalUserLow = totalUserLow;
        lastTotalSys = totalSys;
        lastTotalIdle = totalIdle;
        
        return percent;
    }
    
    // Get the current GPU usage via nvidia-smi (if available)
    double getGpuUsage() {
        double usage = 0.0;
        FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                usage = atof(buffer);
            }
            pclose(pipe);
        }
        return usage;
    }
    
    // Get current memory usage
    double getMemoryUsage() {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        unsigned long total_mem = 0, free_mem = 0, buffers = 0, cached = 0;
        
        while (std::getline(meminfo, line)) {
            if (line.substr(0, 9) == "MemTotal:") {
                sscanf(line.c_str(), "MemTotal: %lu", &total_mem);
            } else if (line.substr(0, 8) == "MemFree:") {
                sscanf(line.c_str(), "MemFree: %lu", &free_mem);
            } else if (line.substr(0, 8) == "Buffers:") {
                sscanf(line.c_str(), "Buffers: %lu", &buffers);
            } else if (line.substr(0, 7) == "Cached:") {
                sscanf(line.c_str(), "Cached: %lu", &cached);
            }
        }
        
        if (total_mem > 0) {
            double used_mem = total_mem - free_mem - buffers - cached;
            return 100.0 * used_mem / total_mem;
        }
        
        return 0.0;
    }
    
    // Print current statistics
    void printStats() {
        std::lock_guard<std::mutex> lock1(stats_mutex_);
        std::lock_guard<std::mutex> lock2(thread_mutex_);
        
        std::cout << "\n====== Resource Monitor Stats ======\n";
        
        // Print resource usage
        std::cout << "Resource Usage:\n";
        for (const auto& [name, stats] : resource_stats_) {
            std::cout << "  " << std::left << std::setw(20) << name << ": "
                     << std::fixed << std::setprecision(2) << stats.current_value
                     << "% (min: " << stats.min_value
                     << "%, max: " << stats.max_value
                     << "%, avg: " << stats.avg_value << "%)\n";
        }
        
        // Print timing stats
        std::cout << "\nTiming Stats:\n";
        for (const auto& [name, stats] : timing_stats_) {
            std::cout << "  " << std::left << std::setw(30) << name << ": "
                     << std::fixed << std::setprecision(2) << stats.current_value
                     << "ms (min: " << stats.min_value
                     << "ms, max: " << stats.max_value
                     << "ms, avg: " << stats.avg_value
                     << "ms, samples: " << stats.samples << ")\n";
        }
        
        // Print counter stats
        std::cout << "\nCounter Stats:\n";
        for (const auto& [name, stats] : counter_stats_) {
            std::cout << "  " << std::left << std::setw(30) << name << ": "
                     << std::fixed << std::setprecision(2) << stats.current_value
                     << " (samples: " << stats.samples << ")\n";
        }
        
        // Print thread statuses
        std::cout << "\nThread Statuses:\n";
        auto now = std::chrono::steady_clock::now();
        for (const auto& [id, status] : thread_statuses_) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - status.last_update).count();
            
            std::cout << "  " << std::left << std::setw(20) << id << ": "
                     << std::setw(15) << status.status
                     << " " << std::setw(30) << status.function
                     << " " << status.details
                     << " (updated " << elapsed << "ms ago)\n";
        }
        
        std::cout << "====================================\n\n";
    }
    
private:
    SystemMonitor() : is_running_(false) {}
    ~SystemMonitor() {
        stop();
    }
    
    void monitorLoop(int interval_ms) {
        std::ofstream log_file("resource_monitor.log");
        log_file << "timestamp,cpu_usage,gpu_usage,memory_usage" << std::endl;
        
        while (is_running_.load()) {
            auto start_time = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(start_time);
            
            // Gather metrics
            double cpu_usage = getCpuUsage();
            double gpu_usage = getGpuUsage();
            double memory_usage = getMemoryUsage();
            
            // Record in our stats
            recordResourceUsage("CPU Usage", cpu_usage);
            recordResourceUsage("GPU Usage", gpu_usage);
            recordResourceUsage("Memory Usage", memory_usage);
            
            // Log to file
            log_file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                     << cpu_usage << "," << gpu_usage << "," << memory_usage << std::endl;
            
            // Print current stats
            printStats();
            
            // Wait for next interval
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(interval_ms), [this]() {
                return !is_running_.load();
            });
        }
        
        log_file.close();
    }
    
    std::thread monitor_thread_;
    std::atomic<bool> is_running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    
    std::mutex stats_mutex_;
    std::map<std::string, ResourceStats> resource_stats_;
    std::map<std::string, ResourceStats> timing_stats_;
    std::map<std::string, ResourceStats> counter_stats_;
    
    std::mutex thread_mutex_;
    std::map<std::string, ThreadStatus> thread_statuses_;
};

// Scope-based timer for easy function timing
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name), start_(std::chrono::steady_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
        SystemMonitor::instance().recordTiming(name_, duration);
    }
    
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

// Thread ID to string helper
inline std::string getThreadId() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

// Macros for easier usage
#define DEBUG_TIMER(name) alphazero::debug::ScopedTimer timer_##__LINE__(name)
#define DEBUG_THREAD_STATUS(status, details) alphazero::debug::SystemMonitor::instance().updateThreadStatus(alphazero::debug::getThreadId(), status, __FUNCTION__, details)
#define DEBUG_INCREMENT_COUNTER(name) alphazero::debug::SystemMonitor::instance().incrementCounter(name)

} // namespace debug
} // namespace alphazero

#endif // ALPHAZERO_DEBUG_MONITOR_H