#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>

namespace alphazero {
namespace utils {

class MemoryTracker {
public:
    static MemoryTracker& getInstance() {
        static MemoryTracker instance;
        return instance;
    }
    
    void logMemory(const std::string& label);
    
private:
    MemoryTracker() : last_rss_(0), start_time_(std::chrono::steady_clock::now()) {}
    
    size_t getCurrentRSS();
    std::string formatBytes(size_t bytes);
    
    size_t last_rss_;
    std::chrono::steady_clock::time_point start_time_;
};

// Convenience function
inline void trackMemory(const std::string& label) {
    MemoryTracker::getInstance().logMemory(label);
}

} // namespace utils
} // namespace alphazero

#endif // MEMORY_TRACKER_H