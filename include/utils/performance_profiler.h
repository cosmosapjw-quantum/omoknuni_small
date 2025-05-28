#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace alphazero {
namespace utils {

// High-resolution timer for profiling
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Scoped timer that automatically records elapsed time
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, bool auto_print = false);
    ~ScopedTimer();
    
    double elapsed_ms() const { return timer_.elapsed_ms(); }
    
private:
    std::string name_;
    Timer timer_;
    bool auto_print_;
};

// Performance statistics for a specific operation
struct PerfStats {
    std::atomic<uint64_t> count{0};
    std::atomic<double> total_time_ms{0};
    std::atomic<double> min_time_ms{std::numeric_limits<double>::max()};
    std::atomic<double> max_time_ms{0};
    mutable std::mutex percentile_mutex;
    mutable std::vector<double> recent_times;  // For percentile calculation
    static constexpr size_t MAX_RECENT = 1000;
    
    void record(double time_ms);
    double average_ms() const;
    double p50_ms() const;
    double p90_ms() const;
    double p99_ms() const;
};

// Global performance profiler
class PerformanceProfiler {
public:
    static PerformanceProfiler& getInstance() {
        static PerformanceProfiler instance;
        return instance;
    }
    
    // Record timing for an operation
    void recordTiming(const std::string& operation, double time_ms);
    
    // Get statistics for an operation
    const PerfStats* getStats(const std::string& operation) const;
    
    // Print all statistics
    void printStats(std::ostream& os = std::cout) const;
    
    // Clear all statistics
    void reset();
    
    // Enable/disable profiling
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    // Hierarchical profiling support
    void pushContext(const std::string& context);
    void popContext();
    std::string getCurrentContext() const;
    
private:
    PerformanceProfiler() = default;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, PerfStats> stats_;
    std::atomic<bool> enabled_{true};
    
    // Context stack for hierarchical profiling
    thread_local static std::vector<std::string> context_stack_;
};

// Convenience macros for profiling
#ifndef PROFILE_SCOPE
#define PROFILE_SCOPE(name) \
    alphazero::utils::ScopedTimer _timer_##__LINE__(name)
#endif

#define PROFILE_FUNCTION() \
    alphazero::utils::ScopedTimer _timer_##__LINE__(__FUNCTION__)

#define PROFILE_START(name) \
    auto _timer_##name = alphazero::utils::Timer()

#define PROFILE_END(name) \
    alphazero::utils::PerformanceProfiler::getInstance().recordTiming(#name, _timer_##name.elapsed_ms())

// Specific profiling categories
namespace ProfileCategory {
    constexpr const char* MCTS_SELECTION = "MCTS::Selection";
    constexpr const char* MCTS_EXPANSION = "MCTS::Expansion";
    constexpr const char* MCTS_SIMULATION = "MCTS::Simulation";
    constexpr const char* MCTS_BACKPROP = "MCTS::Backpropagation";
    constexpr const char* MCTS_BATCH_COLLECT = "MCTS::BatchCollection";
    constexpr const char* NN_INFERENCE = "NN::Inference";
    constexpr const char* NN_PREPROCESSING = "NN::Preprocessing";
    constexpr const char* NN_POSTPROCESSING = "NN::Postprocessing";
    constexpr const char* GPU_ATTACK_DEFENSE = "GPU::AttackDefense";
    constexpr const char* GPU_TRANSFER = "GPU::DataTransfer";
    constexpr const char* MEMORY_ALLOC = "Memory::Allocation";
    constexpr const char* MEMORY_POOL = "Memory::PoolOperation";
    constexpr const char* QUEUE_ENQUEUE = "Queue::Enqueue";
    constexpr const char* QUEUE_DEQUEUE = "Queue::Dequeue";
    constexpr const char* TREE_TRAVERSAL = "Tree::Traversal";
    constexpr const char* STATE_CLONE = "State::Clone";
    constexpr const char* STATE_TENSOR = "State::TensorRepresentation";
}

// Thread-local profiling context - removed from header to avoid multiple definition
// Definition should be in a .cpp file

} // namespace utils
} // namespace alphazero