#ifndef ALPHAZERO_MEMORY_DEBUG_H
#define ALPHAZERO_MEMORY_DEBUG_H

#include <cstdlib>
#include <new>
#include <iostream>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <cxxabi.h>
#include <execinfo.h>
#include <atomic>
#include <iomanip>
#include <functional>
#include <algorithm>

// Set to 1 to enable detailed memory tracking
#define MEMORY_DEBUG_ENABLED 1

// Declare global memory tracking operators (defined in memory_debug.cpp)
#if MEMORY_DEBUG_ENABLED
void* operator new(std::size_t size);
void* operator new[](std::size_t size);
void operator delete(void* ptr) noexcept;
void operator delete[](void* ptr) noexcept;
void operator delete(void* ptr, std::size_t) noexcept;
void operator delete[](void* ptr, std::size_t) noexcept;
#endif

// Declare global memory monitoring functions (defined in memory_debug.cpp)
void startMemoryMonitoring();
void stopMemoryMonitoring();
void takeMemorySnapshot(const std::string& label);
void saveMemoryReport(const std::string& filename);

namespace alphazero {
namespace debug {

// Memory allocation tracker
class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker instance;
        return instance;
    }
    
    // Record allocation
    void recordAllocation(void* ptr, size_t size, const std::string& type_name = "") {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = {size, getCurrentStackTrace(), type_name};
        allocated_bytes_ += size;
        allocation_count_++;
        
        // Check if this is a peak
        if (allocated_bytes_ > peak_bytes_) {
            peak_bytes_ = allocated_bytes_;
            peak_allocation_count_ = allocation_count_;
            peak_stack_trace_ = getCurrentStackTrace();
        }
    }
    
    // Record deallocation
    void recordDeallocation(void* ptr) {
        if (!enabled_ || !ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            allocated_bytes_ -= it->second.size;
            allocation_count_--;
            allocations_.erase(it);
        }
    }
    
    // Get current memory usage
    size_t getCurrentUsage() const {
        return allocated_bytes_;
    }
    
    // Get peak memory usage
    size_t getPeakUsage() const {
        return peak_bytes_;
    }
    
    // Enable/disable tracking
    void setEnabled(bool enabled) {
        enabled_ = enabled;
    }
    
    // Print memory report
    void printReport() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n====== Memory Allocation Report ======\n";
        std::cout << "Current allocations: " << allocation_count_ << " objects\n";
        std::cout << "Current memory: " << (allocated_bytes_ / (1024 * 1024)) << " MB\n";
        std::cout << "Peak memory: " << (peak_bytes_ / (1024 * 1024)) << " MB\n";
        std::cout << "Peak allocations: " << peak_allocation_count_ << " objects\n";
        
        // Print peak allocation stack trace
        std::cout << "\nPeak memory stack trace:\n";
        std::cout << peak_stack_trace_ << "\n";
        
        // Count by type
        std::map<std::string, size_t> type_sizes;
        std::map<std::string, size_t> type_counts;
        
        for (const auto& [ptr, info] : allocations_) {
            type_sizes[info.type_name] += info.size;
            type_counts[info.type_name]++;
        }
        
        // Print top types by size
        std::cout << "\nTop allocations by type:\n";
        std::vector<std::pair<std::string, size_t>> sorted_types(type_sizes.begin(), type_sizes.end());
        std::sort(sorted_types.begin(), sorted_types.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (size_t i = 0; i < std::min(size_t(10), sorted_types.size()); ++i) {
            const auto& [type, size] = sorted_types[i];
            std::cout << std::setw(40) << std::left << type 
                     << ": " << (size / 1024) << " KB in " 
                     << type_counts[type] << " objects\n";
        }
        
        // Find largest allocations
        std::vector<std::pair<void*, AllocationInfo>> largest_allocs;
        for (const auto& alloc : allocations_) {
            largest_allocs.push_back(alloc);
        }
        
        std::sort(largest_allocs.begin(), largest_allocs.end(), 
                 [](const auto& a, const auto& b) { return a.second.size > b.second.size; });
        
        std::cout << "\nLargest individual allocations:\n";
        for (size_t i = 0; i < std::min(size_t(5), largest_allocs.size()); ++i) {
            const auto& [ptr, info] = largest_allocs[i];
            std::cout << "Size: " << (info.size / 1024) << " KB, Type: " << info.type_name << "\n";
            std::cout << "Stack trace:\n" << info.stack_trace << "\n";
        }
        
        std::cout << "======================================\n\n";
    }
    
    // Print memory leak report on exit
    void printLeakReport() {
        if (allocations_.empty()) {
            std::cout << "No memory leaks detected!\n";
            return;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n====== Memory Leak Report ======\n";
        std::cout << "Detected " << allocations_.size() << " potential memory leaks totaling "
                 << (allocated_bytes_ / 1024) << " KB\n\n";
        
        // Count by type
        std::map<std::string, size_t> type_sizes;
        std::map<std::string, size_t> type_counts;
        
        for (const auto& [ptr, info] : allocations_) {
            type_sizes[info.type_name] += info.size;
            type_counts[info.type_name]++;
        }
        
        // Print leaks by type
        std::vector<std::pair<std::string, size_t>> sorted_types(type_sizes.begin(), type_sizes.end());
        std::sort(sorted_types.begin(), sorted_types.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& [type, size] : sorted_types) {
            std::cout << std::setw(40) << std::left << type 
                     << ": " << (size / 1024) << " KB in " 
                     << type_counts[type] << " objects\n";
        }
        
        // Print individual leak details (limit to top 10 by size)
        std::vector<std::pair<void*, AllocationInfo>> largest_leaks;
        for (const auto& leak : allocations_) {
            largest_leaks.push_back(leak);
        }
        
        std::sort(largest_leaks.begin(), largest_leaks.end(), 
                 [](const auto& a, const auto& b) { return a.second.size > b.second.size; });
        
        std::cout << "\nTop 10 largest leaks:\n";
        for (size_t i = 0; i < std::min(size_t(10), largest_leaks.size()); ++i) {
            const auto& [ptr, info] = largest_leaks[i];
            std::cout << "Address: " << ptr << ", Size: " << info.size << " bytes, Type: " << info.type_name << "\n";
            std::cout << "Allocation stack trace:\n" << info.stack_trace << "\n\n";
        }
        
        std::cout << "=================================\n\n";
    }
    
private:
    MemoryTracker() : enabled_(true), allocated_bytes_(0), peak_bytes_(0), 
                     allocation_count_(0), peak_allocation_count_(0) {}
    
    ~MemoryTracker() {
        printLeakReport();
    }
    
    struct AllocationInfo {
        size_t size;
        std::string stack_trace;
        std::string type_name;
    };
    
    std::string getCurrentStackTrace(int skip_frames = 1, int max_frames = 10) {
        void* callstack[128];
        int frames = backtrace(callstack, 128);
        char** symbols = backtrace_symbols(callstack, frames);
        
        std::ostringstream trace;
        
        // Skip the first few frames which are our own memory tracking functions
        for (int i = skip_frames; i < std::min(frames, skip_frames + max_frames); ++i) {
            std::string symbol(symbols[i]);
            
            // Try to demangle the C++ symbol name
            size_t begin = symbol.find('(');
            size_t end = symbol.find('+', begin);
            
            if (begin != std::string::npos && end != std::string::npos) {
                std::string mangled = symbol.substr(begin + 1, end - begin - 1);
                
                int status;
                char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                
                if (status == 0 && demangled) {
                    // Replace the mangled name with the demangled one
                    symbol = symbol.substr(0, begin + 1) + demangled + symbol.substr(end);
                    free(demangled);
                }
            }
            
            trace << "  " << symbol << "\n";
        }
        
        free(symbols);
        return trace.str();
    }
    
    std::mutex mutex_;
    std::atomic<bool> enabled_;
    std::map<void*, AllocationInfo> allocations_;
    size_t allocated_bytes_;
    size_t peak_bytes_;
    size_t allocation_count_;
    size_t peak_allocation_count_;
    std::string peak_stack_trace_;
};

// Helper to get demangled type name
template<typename T>
std::string getTypeName() {
    const char* mangled = typeid(T).name();
    int status;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    
    std::string result;
    if (status == 0 && demangled) {
        result = demangled;
        free(demangled);
    } else {
        result = mangled;
    }
    
    return result;
}

// Helper for installing memory allocation failure hooks
class MemoryFailureSimulator {
public:
    static MemoryFailureSimulator& instance() {
        static MemoryFailureSimulator instance;
        return instance;
    }
    
    // Set the pattern for memory allocation failures
    void setFailurePattern(const std::vector<bool>& pattern) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_pattern_ = pattern;
        current_index_ = 0;
        enabled_ = !pattern.empty();
    }
    
    // Enable/disable the simulator
    void setEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = enabled;
    }
    
    // Check if the next allocation should fail
    bool shouldFailNextAllocation() {
        if (!enabled_) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_pattern_.empty()) return false;
        
        bool should_fail = failure_pattern_[current_index_];
        current_index_ = (current_index_ + 1) % failure_pattern_.size();
        return should_fail;
    }
    
private:
    MemoryFailureSimulator() : enabled_(false), current_index_(0) {}
    
    std::mutex mutex_;
    bool enabled_;
    std::vector<bool> failure_pattern_;
    size_t current_index_;
};

// Function declarations for memory monitoring (defined in memory_debug.cpp)
void startMemoryMonitoring();
void stopMemoryMonitoring();
void takeMemorySnapshot(const std::string& label);
void saveMemoryReport(const std::string& filename);

// Macros for memory debugging
#define DEBUG_PRINT_MEMORY_USAGE() alphazero::debug::MemoryTracker::instance().printReport()
#define DEBUG_TRACK_ALLOCATION(ptr, size, type) alphazero::debug::MemoryTracker::instance().recordAllocation(ptr, size, type)
#define DEBUG_TRACK_DEALLOCATION(ptr) alphazero::debug::MemoryTracker::instance().recordDeallocation(ptr)

// Simulate allocation failure for testing robustness
#define DEBUG_SET_ALLOCATION_FAILURE_PATTERN(pattern) alphazero::debug::MemoryFailureSimulator::instance().setFailurePattern(pattern)
#define DEBUG_ENABLE_ALLOCATION_FAILURES(enabled) alphazero::debug::MemoryFailureSimulator::instance().setEnabled(enabled)

} // namespace debug
} // namespace alphazero

#endif // ALPHAZERO_MEMORY_DEBUG_H