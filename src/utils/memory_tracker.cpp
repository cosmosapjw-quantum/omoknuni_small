#include "utils/memory_tracker.h"
#include <fstream>
#include <sstream>
#include <cstring>

#ifdef __linux__
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace alphazero {
namespace utils {

size_t MemoryTracker::getCurrentRSS() {
#ifdef __linux__
    long rss = 0L;
    FILE* fp = fopen("/proc/self/status", "r");
    if (fp == NULL)
        return 0;

    char line[128];
    while (fgets(line, 128, fp) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            char* start = line + 6;
            while (*start == ' ' || *start == '\t') start++;
            rss = atol(start) * 1024; // Convert from KB to bytes
            break;
        }
    }
    fclose(fp);
    return static_cast<size_t>(rss);
#else
    return 0;
#endif
}

std::string MemoryTracker::formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

void MemoryTracker::logMemory(const std::string& label) {
    size_t current_rss = getCurrentRSS();
    
    // Calculate time elapsed
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    
    // Calculate delta
    long delta = static_cast<long>(current_rss) - static_cast<long>(last_rss_);
    std::string delta_str = delta >= 0 ? "+" : "";
    delta_str += formatBytes(std::abs(delta));
    
    // Print memory status
    std::cout << "[MEMORY] " 
              << "[" << elapsed << "s] "
              << label << ": "
              << formatBytes(current_rss)
              << " (delta: " << delta_str << ")"
              << std::endl;
    
    // Update last RSS
    last_rss_ = current_rss;
}

} // namespace utils
} // namespace alphazero