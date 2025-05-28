#include "utils/performance_profiler.h"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace alphazero {
namespace utils {

// Thread-local definition
thread_local std::vector<std::string> PerformanceProfiler::context_stack_;

// PerfStats implementation
void PerfStats::record(double time_ms) {
    count.fetch_add(1);
    
    // Update total
    double current_total = total_time_ms.load();
    while (!total_time_ms.compare_exchange_weak(current_total, current_total + time_ms));
    
    // Update min
    double current_min = min_time_ms.load();
    while (time_ms < current_min && !min_time_ms.compare_exchange_weak(current_min, time_ms));
    
    // Update max
    double current_max = max_time_ms.load();
    while (time_ms > current_max && !max_time_ms.compare_exchange_weak(current_max, time_ms));
    
    // Store for percentile calculation
    {
        std::lock_guard<std::mutex> lock(percentile_mutex);
        recent_times.push_back(time_ms);
        if (recent_times.size() > MAX_RECENT) {
            recent_times.erase(recent_times.begin());
        }
    }
}

double PerfStats::average_ms() const {
    uint64_t c = count.load();
    return c > 0 ? total_time_ms.load() / c : 0.0;
}

double PerfStats::p50_ms() const {
    std::lock_guard<std::mutex> lock(percentile_mutex);
    if (recent_times.empty()) return 0.0;
    
    std::vector<double> sorted = recent_times;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
}

double PerfStats::p90_ms() const {
    std::lock_guard<std::mutex> lock(percentile_mutex);
    if (recent_times.empty()) return 0.0;
    
    std::vector<double> sorted = recent_times;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() * 9 / 10];
}

double PerfStats::p99_ms() const {
    std::lock_guard<std::mutex> lock(percentile_mutex);
    if (recent_times.empty()) return 0.0;
    
    std::vector<double> sorted = recent_times;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() * 99 / 100];
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(const std::string& name, bool auto_print)
    : name_(name), auto_print_(auto_print) {
}

ScopedTimer::~ScopedTimer() {
    double elapsed = timer_.elapsed_ms();
    
    if (auto_print_) {
        std::cout << "Timer [" << name_ << "]: " << elapsed << " ms" << std::endl;
    }
    
    PerformanceProfiler::getInstance().recordTiming(name_, elapsed);
}

// PerformanceProfiler implementation
void PerformanceProfiler::recordTiming(const std::string& operation, double time_ms) {
    if (!enabled_) return;
    
    std::string full_name = getCurrentContext();
    if (!full_name.empty()) {
        full_name += "::";
    }
    full_name += operation;
    
    std::lock_guard<std::mutex> lock(mutex_);
    stats_[full_name].record(time_ms);
}

const PerfStats* PerformanceProfiler::getStats(const std::string& operation) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(operation);
    return it != stats_.end() ? &it->second : nullptr;
}

void PerformanceProfiler::printStats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Collect and sort operations by total time
    std::vector<std::pair<std::string, const PerfStats*>> sorted_stats;
    for (const auto& pair : stats_) {
        sorted_stats.emplace_back(pair.first, &pair.second);
    }
    
    std::sort(sorted_stats.begin(), sorted_stats.end(),
        [](const auto& a, const auto& b) {
            return a.second->total_time_ms.load() > b.second->total_time_ms.load();
        });
    
    // Print header
    os << "\n=== Performance Profile Report ===\n";
    os << std::left << std::setw(40) << "Operation"
       << std::right << std::setw(10) << "Count"
       << std::setw(12) << "Total(ms)"
       << std::setw(12) << "Avg(ms)"
       << std::setw(12) << "Min(ms)"
       << std::setw(12) << "Max(ms)"
       << std::setw(12) << "P50(ms)"
       << std::setw(12) << "P90(ms)"
       << std::setw(12) << "P99(ms)"
       << "\n";
    
    os << std::string(136, '-') << "\n";
    
    // Print statistics
    for (const auto& [name, stats] : sorted_stats) {
        os << std::left << std::setw(40) << name
           << std::right << std::setw(10) << stats->count.load()
           << std::setw(12) << std::fixed << std::setprecision(2) << stats->total_time_ms.load()
           << std::setw(12) << stats->average_ms()
           << std::setw(12) << stats->min_time_ms.load()
           << std::setw(12) << stats->max_time_ms.load()
           << std::setw(12) << stats->p50_ms()
           << std::setw(12) << stats->p90_ms()
           << std::setw(12) << stats->p99_ms()
           << "\n";
    }
    
    os << std::string(136, '-') << "\n\n";
}

void PerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
}

void PerformanceProfiler::pushContext(const std::string& context) {
    context_stack_.push_back(context);
}

void PerformanceProfiler::popContext() {
    if (!context_stack_.empty()) {
        context_stack_.pop_back();
    }
}

std::string PerformanceProfiler::getCurrentContext() const {
    std::stringstream ss;
    for (size_t i = 0; i < context_stack_.size(); ++i) {
        if (i > 0) ss << "::";
        ss << context_stack_[i];
    }
    return ss.str();
}

} // namespace utils
} // namespace alphazero