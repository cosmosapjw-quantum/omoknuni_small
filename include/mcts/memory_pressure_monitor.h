#ifndef ALPHAZERO_MCTS_MEMORY_PRESSURE_MONITOR_H
#define ALPHAZERO_MCTS_MEMORY_PRESSURE_MONITOR_H

#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <sys/resource.h>

namespace alphazero {
namespace mcts {

/**
 * @brief Monitor memory pressure and trigger cleanup when needed
 * 
 * This class monitors system memory usage and triggers cleanup callbacks
 * when memory pressure is detected. It runs in a separate thread and
 * uses minimal resources.
 */
class MemoryPressureMonitor {
public:
    struct Config {
        size_t max_memory_bytes;
        double warning_threshold;
        double critical_threshold;
        std::chrono::milliseconds check_interval;
        bool auto_cleanup;
        
        Config() 
            : max_memory_bytes(48ULL * 1024 * 1024 * 1024)  // 48GB default limit
            , warning_threshold(0.8)   // Trigger warning at 80% usage
            , critical_threshold(0.9)  // Trigger critical at 90% usage
            , check_interval(1000)     // Check every second
            , auto_cleanup(true)       // Automatically trigger cleanup
        {}
    };
    
    enum class PressureLevel {
        Normal = 0,
        Warning = 1,
        Critical = 2,
        Emergency = 3
    };
    
    MemoryPressureMonitor(const Config& config = Config())
        : config_(config), running_(false), current_level_(PressureLevel::Normal) {}
    
    ~MemoryPressureMonitor() {
        stop();
    }
    
    // Start monitoring
    void start() {
        if (running_.exchange(true)) return;
        
        monitor_thread_ = std::thread([this]() {
            while (running_) {
                checkMemoryPressure();
                std::this_thread::sleep_for(config_.check_interval);
            }
        });
    }
    
    // Stop monitoring
    void stop() {
        if (!running_.exchange(false)) return;
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    
    // Get current memory usage in bytes
    static size_t getCurrentMemoryUsage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        // Convert from KB to bytes (Linux reports in KB)
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    
    // Get current pressure level
    PressureLevel getCurrentLevel() const {
        return current_level_.load();
    }
    
    // Set cleanup callback
    void setCleanupCallback(std::function<void(PressureLevel)> callback) {
        cleanup_callback_ = callback;
    }
    
    // Manual memory check
    PressureLevel checkMemoryPressure() {
        size_t current_memory = getCurrentMemoryUsage();
        double usage_ratio = static_cast<double>(current_memory) / config_.max_memory_bytes;
        
        PressureLevel new_level = PressureLevel::Normal;
        
        if (usage_ratio > 0.95) {
            new_level = PressureLevel::Emergency;
        } else if (usage_ratio > config_.critical_threshold) {
            new_level = PressureLevel::Critical;
        } else if (usage_ratio > config_.warning_threshold) {
            new_level = PressureLevel::Warning;
        }
        
        PressureLevel old_level = current_level_.exchange(new_level);
        
        // Trigger cleanup if level increased and callback is set
        if (new_level > old_level && cleanup_callback_ && config_.auto_cleanup) {
            cleanup_callback_(new_level);
        }
        
        // Update stats
        last_memory_usage_.store(current_memory);
        last_usage_ratio_.store(usage_ratio);
        
        return new_level;
    }
    
    // Get memory stats
    struct MemoryStats {
        size_t current_usage;
        size_t max_allowed;
        double usage_ratio;
        PressureLevel pressure_level;
    };
    
    MemoryStats getStats() const {
        return {
            last_memory_usage_.load(),
            config_.max_memory_bytes,
            last_usage_ratio_.load(),
            current_level_.load()
        };
    }
    
private:
    Config config_;
    std::atomic<bool> running_;
    std::atomic<PressureLevel> current_level_;
    std::atomic<size_t> last_memory_usage_{0};
    std::atomic<double> last_usage_ratio_{0.0};
    
    std::thread monitor_thread_;
    std::function<void(PressureLevel)> cleanup_callback_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_MEMORY_PRESSURE_MONITOR_H