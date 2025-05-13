#include "utils/memory_debug.h"
#include "utils/debug_monitor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdio>
#include <unistd.h>

// Define global operators for memory tracking
#if MEMORY_DEBUG_ENABLED
void* operator new(std::size_t size) {
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();

    alphazero::debug::MemoryTracker::instance().recordAllocation(ptr, size);
    return ptr;
}

void* operator new[](std::size_t size) {
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();

    alphazero::debug::MemoryTracker::instance().recordAllocation(ptr, size);
    return ptr;
}

void operator delete(void* ptr) noexcept {
    alphazero::debug::MemoryTracker::instance().recordDeallocation(ptr);
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    alphazero::debug::MemoryTracker::instance().recordDeallocation(ptr);
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    alphazero::debug::MemoryTracker::instance().recordDeallocation(ptr);
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    alphazero::debug::MemoryTracker::instance().recordDeallocation(ptr);
    std::free(ptr);
}
#endif // MEMORY_DEBUG_ENABLED

// Global functions for memory monitoring defined at the bottom of the file

namespace alphazero {
namespace debug {

// Helper function to get process memory usage from /proc/self/status
size_t getProcessMemoryUsage() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t vm_size = 0;
    
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmSize") {
            // Format is "VmSize: 12345 kB"
            std::stringstream ss(line);
            std::string key, unit;
            ss >> key >> vm_size >> unit;
            break;
        }
    }
    
    return vm_size; // In kB
}

// Helper function to get GPU memory usage via nvidia-smi
size_t getGpuMemoryUsage() {
    size_t used_memory = 0;
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            used_memory = std::stoul(buffer);
        }
        pclose(pipe);
    }
    return used_memory; // In MiB
}

// Memory snapshot helper for monitoring memory over time
class MemoryMonitor {
public:
    static MemoryMonitor& instance() {
        static MemoryMonitor instance;
        return instance;
    }
    
    void start() {
        if (is_running_) return;
        is_running_ = true;
        
        // Start monitoring thread
        monitor_thread_ = std::thread([this]() {
            monitorLoop();
        });
        
        std::cout << "Memory monitor started" << std::endl;
    }
    
    void stop() {
        if (!is_running_) return;
        is_running_ = false;
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        
        std::cout << "Memory monitor stopped" << std::endl;
    }
    
    void takeSnapshot(const std::string& label) {
        MemorySnapshot snapshot;
        snapshot.timestamp = std::chrono::system_clock::now();
        snapshot.label = label;
        snapshot.process_memory_kb = getProcessMemoryUsage();
        snapshot.gpu_memory_mb = getGpuMemoryUsage();
        snapshot.tracked_memory_bytes = MemoryTracker::instance().getCurrentUsage();
        
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots_.push_back(snapshot);
        
        // Also print the snapshot
        auto time_t = std::chrono::system_clock::to_time_t(snapshot.timestamp);
        std::cout << "[Memory] " << std::put_time(std::localtime(&time_t), "%H:%M:%S") << " "
                 << "Label: " << std::left << std::setw(30) << label << " "
                 << "Process: " << std::setw(8) << (snapshot.process_memory_kb / 1024) << " MB, "
                 << "GPU: " << std::setw(8) << snapshot.gpu_memory_mb << " MB, "
                 << "Tracked: " << std::setw(8) << (snapshot.tracked_memory_bytes / (1024 * 1024)) << " MB"
                 << std::endl;
    }
    
    void saveReport(const std::string& filename = "memory_report.csv") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        // Write header
        file << "Timestamp,Label,ProcessMemory_MB,GPUMemory_MB,TrackedMemory_MB" << std::endl;
        
        // Write snapshots
        for (const auto& snapshot : snapshots_) {
            auto time_t = std::chrono::system_clock::to_time_t(snapshot.timestamp);
            file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                 << snapshot.label << ","
                 << (snapshot.process_memory_kb / 1024) << ","
                 << snapshot.gpu_memory_mb << ","
                 << (snapshot.tracked_memory_bytes / (1024 * 1024)) << std::endl;
        }
        
        file.close();
        std::cout << "Memory report saved to " << filename << std::endl;
    }
    
private:
    MemoryMonitor() : is_running_(false) {}
    ~MemoryMonitor() {
        stop();
        saveReport();
    }
    
    struct MemorySnapshot {
        std::chrono::system_clock::time_point timestamp;
        std::string label;
        size_t process_memory_kb;
        size_t gpu_memory_mb;
        size_t tracked_memory_bytes;
    };
    
    void monitorLoop() {
        // Log to CSV file for real-time monitoring
        std::ofstream log_file("memory_monitor.csv");
        log_file << "Timestamp,ProcessMemory_MB,GPUMemory_MB,TrackedMemory_MB" << std::endl;
        
        int counter = 0;
        
        while (is_running_) {
            // Create a snapshot
            MemorySnapshot snapshot;
            snapshot.timestamp = std::chrono::system_clock::now();
            snapshot.label = "monitor_" + std::to_string(counter++);
            snapshot.process_memory_kb = getProcessMemoryUsage();
            snapshot.gpu_memory_mb = getGpuMemoryUsage();
            snapshot.tracked_memory_bytes = MemoryTracker::instance().getCurrentUsage();
            
            // Add to snapshots
            {
                std::lock_guard<std::mutex> lock(mutex_);
                snapshots_.push_back(snapshot);
            }
            
            // Log to file
            auto time_t = std::chrono::system_clock::to_time_t(snapshot.timestamp);
            log_file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                     << (snapshot.process_memory_kb / 1024) << ","
                     << snapshot.gpu_memory_mb << ","
                     << (snapshot.tracked_memory_bytes / (1024 * 1024)) << std::endl;
            
            // Update system monitor
            SystemMonitor::instance().recordResourceUsage("ProcessMemory_MB", 
                                                       snapshot.process_memory_kb / 1024.0);
            SystemMonitor::instance().recordResourceUsage("GPUMemory_MB", 
                                                       snapshot.gpu_memory_mb);
            SystemMonitor::instance().recordResourceUsage("TrackedMemory_MB", 
                                                       snapshot.tracked_memory_bytes / (1024.0 * 1024.0));
            
            // Sleep for 2 seconds
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        log_file.close();
    }
    
    std::atomic<bool> is_running_;
    std::thread monitor_thread_;
    std::mutex mutex_;
    std::vector<MemorySnapshot> snapshots_;
};

// Helper functions for debugging memory issues
void startMemoryMonitoring() {
    MemoryMonitor::instance().start();
}

void stopMemoryMonitoring() {
    MemoryMonitor::instance().stop();
}

void takeMemorySnapshot(const std::string& label) {
    MemoryMonitor::instance().takeSnapshot(label);
}

void saveMemoryReport(const std::string& filename) {
    MemoryMonitor::instance().saveReport(filename);
}

} // namespace debug
} // namespace alphazero

// Global functions outside namespace
void startMemoryMonitoring() {
    alphazero::debug::startMemoryMonitoring();
}

void stopMemoryMonitoring() {
    alphazero::debug::stopMemoryMonitoring();
}

void takeMemorySnapshot(const std::string& label) {
    alphazero::debug::takeMemorySnapshot(label);
}

void saveMemoryReport(const std::string& filename) {
    alphazero::debug::saveMemoryReport(filename);
}