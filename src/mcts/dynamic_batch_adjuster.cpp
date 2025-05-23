// src/mcts/dynamic_batch_adjuster.cpp
#include "mcts/dynamic_batch_adjuster.h"
#include <sstream>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace alphazero {
namespace mcts {

DynamicBatchAdjuster::DynamicBatchAdjuster()
    : params_(), last_arrival_time_(std::chrono::steady_clock::now()) {
    current_arrival_rate_.store(params_.initial_arrival_rate, std::memory_order_relaxed);
}

DynamicBatchAdjuster::DynamicBatchAdjuster(const AdjustmentParams& params)
    : params_(params), last_arrival_time_(std::chrono::steady_clock::now()) {
    current_arrival_rate_.store(params_.initial_arrival_rate, std::memory_order_relaxed);
}

void DynamicBatchAdjuster::recordBatch(size_t batch_size, std::chrono::milliseconds actual_wait_time) {
    auto now = std::chrono::steady_clock::now();
    double gpu_util = estimateGPUUtilization(batch_size);
    
    BatchStats stats{batch_size, actual_wait_time, now, gpu_util};
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        batch_history_.push_back(stats);
        
        // Maintain history window
        while (batch_history_.size() > params_.history_window) {
            batch_history_.pop_front();
        }
        
        // Update averages
        if (!batch_history_.empty()) {
            double sum_batch_size = 0;
            double sum_gpu_util = 0;
            for (const auto& stat : batch_history_) {
                sum_batch_size += stat.batch_size;
                sum_gpu_util += stat.gpu_utilization;
            }
            avg_batch_size_.store(sum_batch_size / batch_history_.size(), std::memory_order_relaxed);
            avg_gpu_utilization_.store(sum_gpu_util / batch_history_.size(), std::memory_order_relaxed);
        }
    }
    
    // Adjust timeout based on performance
    adjustTimeout();
}

void DynamicBatchAdjuster::recordThreadArrival() {
    arrival_count_.fetch_add(1, std::memory_order_relaxed);
    updateArrivalRate();
}

double DynamicBatchAdjuster::getAverageBatchSize() const {
    return avg_batch_size_.load(std::memory_order_relaxed);
}

double DynamicBatchAdjuster::getGPUUtilization() const {
    return avg_gpu_utilization_.load(std::memory_order_relaxed);
}

void DynamicBatchAdjuster::forceAdjustTimeout(std::chrono::milliseconds new_timeout) {
    params_.current_timeout = std::clamp(new_timeout, params_.min_timeout, params_.max_timeout);
}

std::string DynamicBatchAdjuster::getStatsSummary() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::stringstream ss;
    
    ss << "DynamicBatchAdjuster Stats:\n";
    ss << "  Current timeout: " << params_.current_timeout.count() << "ms\n";
    ss << "  Avg batch size: " << std::fixed << std::setprecision(1) << getAverageBatchSize() << "\n";
    ss << "  GPU utilization: " << std::fixed << std::setprecision(1) << (getGPUUtilization() * 100) << "%\n";
    ss << "  Arrival rate: " << std::fixed << std::setprecision(2) << getArrivalRate() << " req/ms\n";
    ss << "  History size: " << batch_history_.size() << "\n";
    
    if (!batch_history_.empty()) {
        // Recent batch sizes
        ss << "  Recent batches: ";
        size_t count = std::min(size_t(5), batch_history_.size());
        for (size_t i = batch_history_.size() - count; i < batch_history_.size(); ++i) {
            ss << batch_history_[i].batch_size << " ";
        }
        ss << "\n";
    }
    
    return ss.str();
}

void DynamicBatchAdjuster::adjustTimeout() {
    double avg_batch = getAverageBatchSize();
    double gpu_util = getGPUUtilization();
    double arrival_rate = getArrivalRate();
    
    // Calculate ideal timeout based on arrival rate and target batch size
    double ideal_timeout_ms = 0;
    if (arrival_rate > 0) {
        // Time to accumulate target batch size at current arrival rate
        ideal_timeout_ms = params_.target_batch_size / arrival_rate;
    }
    
    // Adjust based on current performance
    auto current_ms = params_.current_timeout.count();
    double new_timeout_ms = current_ms;
    
    if (avg_batch < params_.target_batch_size * 0.5) {
        // Batch size too small - increase timeout
        new_timeout_ms = current_ms * params_.timeout_increase_factor;
        
        // But also consider ideal timeout
        if (ideal_timeout_ms > 0) {
            new_timeout_ms = std::max(new_timeout_ms, ideal_timeout_ms * 0.8);
        }
    } else if (avg_batch > params_.target_batch_size * 1.2) {
        // Batch size larger than needed - can reduce timeout
        new_timeout_ms = current_ms * params_.timeout_decrease_factor;
    } else if (gpu_util < params_.target_gpu_utilization * 0.7) {
        // GPU underutilized - increase timeout for larger batches
        new_timeout_ms = current_ms * params_.timeout_increase_factor;
    }
    
    // Apply bounds
    new_timeout_ms = std::clamp(new_timeout_ms, 
                                double(params_.min_timeout.count()), 
                                double(params_.max_timeout.count()));
    
    params_.current_timeout = std::chrono::milliseconds(static_cast<long>(new_timeout_ms));
}

double DynamicBatchAdjuster::estimateGPUUtilization(size_t batch_size) const {
    // Simple model: GPU utilization increases with batch size
    // Assumes optimal batch size gives ~100% utilization
    double optimal_batch = params_.target_batch_size;
    
    if (batch_size <= 1) {
        return 0.031;  // ~3.1% for single inference (from logs)
    }
    
    // Logarithmic growth model
    double utilization = std::log(batch_size + 1) / std::log(optimal_batch + 1);
    
    // Cap at 100%
    return std::min(1.0, utilization);
}

void DynamicBatchAdjuster::updateArrivalRate() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_arrival_time_).count();
    
    if (elapsed_ms > 0) {
        // Calculate instantaneous rate
        double instant_rate = static_cast<double>(arrival_count_.load()) / elapsed_ms;
        
        // Update EMA
        double current_rate = current_arrival_rate_.load(std::memory_order_relaxed);
        double new_rate = current_rate * (1 - params_.arrival_rate_ema_alpha) + 
                         instant_rate * params_.arrival_rate_ema_alpha;
        
        current_arrival_rate_.store(new_rate, std::memory_order_relaxed);
        
        // Reset counters
        arrival_count_.store(0, std::memory_order_relaxed);
        last_arrival_time_ = now;
    }
}

} // namespace mcts
} // namespace alphazero