#include "mcts/adaptive_batch_sizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif

namespace mcts {

AdaptiveBatchSizer::AdaptiveBatchSizer(size_t initial_batch_size,
                                       size_t min_batch_size,
                                       size_t max_batch_size)
    : current_batch_size_(initial_batch_size)
    , last_adaptation_time_(std::chrono::steady_clock::now()) {
    config_.min_batch_size = min_batch_size;
    config_.max_batch_size = max_batch_size;
}

size_t AdaptiveBatchSizer::getCurrentBatchSize() const {
    return current_batch_size_.load(std::memory_order_acquire);
}

void AdaptiveBatchSizer::recordBatchPerformance(const BatchPerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    performance_history_.push_back(metrics);
    trimHistory();
}

void AdaptiveBatchSizer::adjustBatchSize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (performance_history_.size() < config_.min_samples_for_adaptation) {
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = now - last_adaptation_time_;
    
    if (time_since_last < std::chrono::seconds(5)) {
        return;
    }
    
    if (consecutive_adjustments_ >= config_.max_consecutive_adjustments) {
        consecutive_adjustments_ = 0;
        return;
    }
    
    size_t old_size = current_batch_size_.load(std::memory_order_relaxed);
    size_t new_size = old_size;
    bool should_adjust = false;
    
    if (shouldIncreaseBatchSize()) {
        new_size = std::min(config_.max_batch_size,
                           static_cast<size_t>(old_size * config_.size_increase_factor));
        should_adjust = true;
        last_adjustment_positive_ = true;
    } else if (shouldDecreaseBatchSize()) {
        new_size = std::max(config_.min_batch_size,
                           static_cast<size_t>(old_size * config_.size_decrease_factor));
        should_adjust = true;
        last_adjustment_positive_ = false;
    }
    
    if (should_adjust && new_size != old_size) {
        current_batch_size_.store(new_size, std::memory_order_release);
        last_adaptation_time_ = now;
        total_adaptations_.fetch_add(1, std::memory_order_relaxed);
        consecutive_adjustments_++;
        
        double old_score = 0.0;
        double new_score = 0.0;
        
        if (performance_history_.size() >= 2) {
            const auto& recent = performance_history_.back();
            old_score = calculatePerformanceScore(recent);
        }
        
        if (old_score > 0.0 && new_score > old_score * (1.0 + config_.performance_improvement_threshold)) {
            successful_adaptations_.fetch_add(1, std::memory_order_relaxed);
        } else if (old_score > 0.0) {
            failed_adaptations_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void AdaptiveBatchSizer::setConfig(const AdaptationConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

const AdaptiveBatchSizer::AdaptationConfig& AdaptiveBatchSizer::getConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

AdaptiveBatchSizer::AdaptationStats AdaptiveBatchSizer::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    AdaptationStats stats;
    stats.total_adaptations = total_adaptations_.load(std::memory_order_relaxed);
    stats.successful_adaptations = successful_adaptations_.load(std::memory_order_relaxed);
    stats.failed_adaptations = failed_adaptations_.load(std::memory_order_relaxed);
    stats.average_gpu_utilization = calculateAverageGpuUtilization();
    stats.average_inference_time = calculateAverageInferenceTime();
    stats.average_queue_wait = calculateAverageQueueWait();
    stats.current_batch_size = current_batch_size_.load(std::memory_order_relaxed);
    stats.last_adaptation = last_adaptation_time_;
    
    return stats;
}

void AdaptiveBatchSizer::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_adaptations_.store(0, std::memory_order_relaxed);
    successful_adaptations_.store(0, std::memory_order_relaxed);
    failed_adaptations_.store(0, std::memory_order_relaxed);
    performance_history_.clear();
    consecutive_adjustments_ = 0;
}

double AdaptiveBatchSizer::calculateAverageGpuUtilization() const {
    if (performance_history_.empty()) return 0.0;
    
    double sum = std::accumulate(performance_history_.begin(), performance_history_.end(), 0.0,
        [](double acc, const BatchPerformanceMetrics& m) {
            return acc + m.gpu_utilization_percent;
        });
    
    return sum / performance_history_.size();
}

std::chrono::microseconds AdaptiveBatchSizer::calculateAverageInferenceTime() const {
    if (performance_history_.empty()) return std::chrono::microseconds(0);
    
    auto sum = std::accumulate(performance_history_.begin(), performance_history_.end(),
        std::chrono::microseconds(0),
        [](std::chrono::microseconds acc, const BatchPerformanceMetrics& m) {
            return acc + m.inference_time;
        });
    
    return sum / performance_history_.size();
}

std::chrono::microseconds AdaptiveBatchSizer::calculateAverageQueueWait() const {
    if (performance_history_.empty()) return std::chrono::microseconds(0);
    
    auto sum = std::accumulate(performance_history_.begin(), performance_history_.end(),
        std::chrono::microseconds(0),
        [](std::chrono::microseconds acc, const BatchPerformanceMetrics& m) {
            return acc + m.queue_wait_time;
        });
    
    return sum / performance_history_.size();
}

bool AdaptiveBatchSizer::shouldIncreaseBatchSize() const {
    double avg_utilization = calculateAverageGpuUtilization();
    auto avg_queue_wait = calculateAverageQueueWait();
    
    bool low_utilization = avg_utilization < (config_.target_gpu_utilization - config_.utilization_tolerance);
    bool acceptable_latency = calculateAverageInferenceTime() < config_.max_acceptable_latency;
    bool low_queue_wait = avg_queue_wait < config_.target_queue_wait;
    
    return low_utilization && acceptable_latency && low_queue_wait;
}

bool AdaptiveBatchSizer::shouldDecreaseBatchSize() const {
    double avg_utilization = calculateAverageGpuUtilization();
    auto avg_inference_time = calculateAverageInferenceTime();
    auto avg_queue_wait = calculateAverageQueueWait();
    
    bool high_latency = avg_inference_time > config_.max_acceptable_latency;
    bool high_queue_wait = avg_queue_wait > config_.target_queue_wait * 2;
    bool excessive_utilization = avg_utilization > (config_.target_gpu_utilization + config_.utilization_tolerance);
    
    return high_latency || high_queue_wait || excessive_utilization;
}

double AdaptiveBatchSizer::calculatePerformanceScore(const BatchPerformanceMetrics& metrics) const {
    double throughput = calculateThroughput(metrics.batch_size, metrics.inference_time);
    
    double utilization_factor = std::min(1.0, metrics.gpu_utilization_percent / config_.target_gpu_utilization);
    
    double latency_penalty = 1.0;
    if (metrics.inference_time > config_.max_acceptable_latency) {
        latency_penalty = static_cast<double>(config_.max_acceptable_latency.count()) /
                         static_cast<double>(metrics.inference_time.count());
    }
    
    double queue_penalty = 1.0;
    if (metrics.queue_wait_time > config_.target_queue_wait) {
        queue_penalty = static_cast<double>(config_.target_queue_wait.count()) /
                       static_cast<double>(metrics.queue_wait_time.count());
    }
    
    return throughput * utilization_factor * latency_penalty * queue_penalty;
}

double AdaptiveBatchSizer::calculateThroughput(size_t batch_size, std::chrono::microseconds inference_time) const {
    if (inference_time.count() == 0) return 0.0;
    
    return static_cast<double>(batch_size) / (static_cast<double>(inference_time.count()) / 1000000.0);
}

void AdaptiveBatchSizer::trimHistory() {
    while (performance_history_.size() > config_.max_history_size) {
        performance_history_.pop_front();
    }
}

#ifdef WITH_CUDA
struct GPUMonitor::Impl {
    bool nvml_initialized = false;
    unsigned int device_count = 0;
    nvmlDevice_t device;
    
    Impl() {
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS) {
            nvml_initialized = true;
            result = nvmlDeviceGetCount(&device_count);
            if (result == NVML_SUCCESS && device_count > 0) {
                nvmlDeviceGetHandleByIndex(0, &device);
            }
        }
    }
    
    ~Impl() {
        if (nvml_initialized) {
            nvmlShutdown();
        }
    }
};

GPUMonitor::GPUMonitor() : pimpl_(std::make_unique<Impl>()) {}

GPUMonitor::~GPUMonitor() = default;

double GPUMonitor::getCurrentUtilization() {
    if (!isAvailable()) return 0.0;
    
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(pimpl_->device, &utilization);
    
    if (result == NVML_SUCCESS) {
        return static_cast<double>(utilization.gpu);
    }
    
    return 0.0;
}

size_t GPUMonitor::getCurrentMemoryUsage() {
    if (!isAvailable()) return 0;
    
    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(pimpl_->device, &memory);
    
    if (result == NVML_SUCCESS) {
        return memory.used / (1024 * 1024); // Convert to MB
    }
    
    return 0;
}

bool GPUMonitor::isAvailable() const {
    return pimpl_->nvml_initialized && pimpl_->device_count > 0;
}

#else

struct GPUMonitor::Impl {};

GPUMonitor::GPUMonitor() : pimpl_(std::make_unique<Impl>()) {}

GPUMonitor::~GPUMonitor() = default;

double GPUMonitor::getCurrentUtilization() { return 0.0; }
size_t GPUMonitor::getCurrentMemoryUsage() { return 0; }
bool GPUMonitor::isAvailable() const { return false; }

#endif

} // namespace mcts