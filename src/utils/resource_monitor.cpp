// src/utils/resource_monitor.cpp
#include "utils/resource_monitor.h"
#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace alphazero {
namespace utils {

ResourceMonitor& ResourceMonitor::getInstance() {
    static ResourceMonitor instance;
    return instance;
}

ResourceMonitor::~ResourceMonitor() {
    stopMonitoring();
}

void ResourceMonitor::startMonitoring(std::chrono::milliseconds interval) {
    if (monitoring_active_.exchange(true)) {
        return; // Already monitoring
    }
    
    monitoring_thread_ = std::make_unique<std::thread>([this, interval]() {
        while (monitoring_active_.load()) {
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                cached_stats_.cpu_usage_percent = getCPUUsage();
                cached_stats_.gpu_usage_percent = getGPUUsage();
                cached_stats_.memory_usage_mb = getMemoryUsage();
                cached_stats_.gpu_memory_usage_mb = getGPUMemoryUsage();
                cached_stats_.active_threads = std::thread::hardware_concurrency();
                
                // Calculate averages
                size_t batches = total_batches_.load();
                if (batches > 0) {
                    cached_stats_.avg_batch_size = static_cast<double>(total_requests_.load()) / batches;
                    cached_stats_.avg_inference_latency_ms = static_cast<double>(total_latency_ms_x1000_.load()) / (1000.0 * batches);
                }
                cached_stats_.total_requests_processed = total_requests_.load();
                cached_stats_.queue_depth = current_queue_depth_.load();
                
                last_update_ = std::chrono::steady_clock::now();
            }
            
            std::this_thread::sleep_for(interval);
        }
    });
}

void ResourceMonitor::stopMonitoring() {
    monitoring_active_.store(false);
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
    }
}

ResourceMonitor::ResourceStats ResourceMonitor::getCurrentStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return cached_stats_;
}

void ResourceMonitor::recordBatchProcessing(size_t batch_size, double latency_ms) {
    total_batches_.fetch_add(1);
    total_requests_.fetch_add(batch_size);
    total_latency_ms_x1000_.fetch_add(static_cast<uint64_t>(latency_ms * 1000.0)); // Convert to integer
}

void ResourceMonitor::recordQueueDepth(size_t depth) {
    current_queue_depth_.store(depth);
}

void ResourceMonitor::printSummary() const {
    auto stats = getCurrentStats();
    
    std::cout << "\n=== RESOURCE UTILIZATION SUMMARY ===" << std::endl;
    std::cout << "CPU Usage:          " << std::fixed << std::setprecision(1) << stats.cpu_usage_percent << "%" << std::endl;
    std::cout << "GPU Usage:          " << std::fixed << std::setprecision(1) << stats.gpu_usage_percent << "%" << std::endl;
    std::cout << "Memory Usage:       " << stats.memory_usage_mb << " MB" << std::endl;
    std::cout << "GPU Memory:         " << stats.gpu_memory_usage_mb << " MB" << std::endl;
    std::cout << "Active Threads:     " << stats.active_threads << std::endl;
    std::cout << "Avg Batch Size:     " << std::fixed << std::setprecision(2) << stats.avg_batch_size << std::endl;
    std::cout << "Avg Latency:        " << std::fixed << std::setprecision(2) << stats.avg_inference_latency_ms << " ms" << std::endl;
    std::cout << "Total Requests:     " << stats.total_requests_processed << std::endl;
    std::cout << "Queue Depth:        " << stats.queue_depth << std::endl;
    std::cout << "=====================================" << std::endl;
}

double ResourceMonitor::getCPUUsage() const {
#ifdef __linux__
    static long long last_total = 0, last_idle = 0;
    
    std::ifstream proc_stat("/proc/stat");
    if (!proc_stat.is_open()) return 0.0;
    
    std::string line;
    if (!std::getline(proc_stat, line)) return 0.0;
    
    std::istringstream iss(line);
    std::string cpu;
    long long user, nice, system, idle, iowait, irq, softirq, steal;
    
    if (!(iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal)) {
        return 0.0;
    }
    
    long long total = user + nice + system + idle + iowait + irq + softirq + steal;
    long long diff_total = total - last_total;
    long long diff_idle = idle - last_idle;
    
    double cpu_usage = 0.0;
    if (diff_total > 0) {
        cpu_usage = 100.0 * (diff_total - diff_idle) / diff_total;
    }
    
    last_total = total;
    last_idle = idle;
    
    return cpu_usage;
#else
    return 0.0; // Not implemented for non-Linux systems
#endif
}

double ResourceMonitor::getGPUUsage() const {
    // TODO: Implement NVIDIA GPU usage monitoring via nvidia-ml-py or similar
    return 0.0;
}

size_t ResourceMonitor::getMemoryUsage() const {
#ifdef __linux__
    std::ifstream proc_status("/proc/self/status");
    if (!proc_status.is_open()) return 0;
    
    std::string line;
    while (std::getline(proc_status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            size_t kb;
            if (iss >> label >> kb) {
                return kb / 1024; // Convert to MB
            }
        }
    }
#endif
    return 0;
}

size_t ResourceMonitor::getGPUMemoryUsage() const {
    // TODO: Implement GPU memory monitoring
    return 0;
}

} // namespace utils
} // namespace alphazero