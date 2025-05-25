// src/mcts/dynamic_batch_manager.cpp
#include "mcts/dynamic_batch_manager.h"
#include "utils/logger.h"
#include <algorithm>
#include <numeric>

namespace alphazero {
namespace mcts {

DynamicBatchManager::DynamicBatchManager(const Config& config) 
    : config_(config) {
    metrics_.start_time = std::chrono::steady_clock::now();
    metrics_.latency_buffer.resize(PerformanceMetrics::LATENCY_BUFFER_SIZE, 0.0f);
    model_.history.reserve(PredictionModel::MAX_HISTORY);
}

int DynamicBatchManager::calculateOptimalBatchSize(
    int queue_depth,
    float gpu_utilization,
    float recent_inference_time_ms
) {
    // Update exponential averages
    updateExponentialAverage(metrics_.exponential_avg_queue_depth, 
                           static_cast<float>(queue_depth), 
                           1.0f - config_.smoothing_factor);
    updateExponentialAverage(metrics_.exponential_avg_gpu_util, 
                           gpu_utilization, 
                           1.0f - config_.smoothing_factor);

    int optimal_size;
    
    // Determine mode if adaptive
    Config::Mode mode = config_.optimization_mode;
    if (mode == Config::ADAPTIVE) {
        mode = determineOptimalMode(queue_depth, gpu_utilization);
        
        // Track mode switches
        Config::Mode current = metrics_.current_mode.load();
        if (mode != current) {
            metrics_.mode_switches.fetch_add(1);
            metrics_.current_mode.store(mode);
        }
    }

    // Calculate based on mode
    switch (mode) {
        case Config::THROUGHPUT:
            optimal_size = calculateThroughputOptimalBatch(queue_depth, gpu_utilization);
            break;
            
        case Config::LATENCY:
            optimal_size = calculateLatencyOptimalBatch(queue_depth, gpu_utilization);
            break;
            
        case Config::BALANCED:
            optimal_size = calculateBalancedBatch(queue_depth, gpu_utilization, recent_inference_time_ms);
            break;
            
        default:
            optimal_size = calculateAdaptiveBatch(queue_depth, gpu_utilization, recent_inference_time_ms);
            break;
    }

    // Apply constraints
    optimal_size = std::max(config_.min_batch_size, 
                           std::min(config_.max_batch_size, optimal_size));

    // Log if significant change
    float current_avg = metrics_.exponential_avg_batch_size.load();
    if (std::abs(optimal_size - current_avg) > current_avg * 0.5f) {
        LOG_MCTS_DEBUG("Batch size adjusted: {} -> {} (queue: {}, gpu: {:.1f}%)",
                      static_cast<int>(current_avg), optimal_size, queue_depth, gpu_utilization);
    }

    return optimal_size;
}

int DynamicBatchManager::calculateThroughputOptimalBatch(int queue_depth, float gpu_util) {
    // Maximize throughput - use larger batches when queue is deep
    if (queue_depth >= config_.critical_queue_threshold) {
        return config_.max_batch_size;
    } else if (queue_depth >= config_.high_queue_threshold) {
        // Linear interpolation between preferred and max
        float ratio = static_cast<float>(queue_depth - config_.high_queue_threshold) / 
                     (config_.critical_queue_threshold - config_.high_queue_threshold);
        return config_.preferred_batch_size + 
               static_cast<int>(ratio * (config_.max_batch_size - config_.preferred_batch_size));
    } else if (queue_depth >= config_.low_queue_threshold) {
        return config_.preferred_batch_size;
    } else {
        // Small queue - use smaller batches to maintain responsiveness
        float ratio = static_cast<float>(queue_depth) / config_.low_queue_threshold;
        return config_.min_batch_size + 
               static_cast<int>(ratio * (config_.preferred_batch_size - config_.min_batch_size));
    }
}

int DynamicBatchManager::calculateLatencyOptimalBatch(int queue_depth, float gpu_util) {
    // Minimize latency - prefer smaller batches unless queue is very deep
    if (queue_depth >= config_.critical_queue_threshold) {
        // Emergency mode - need to clear queue
        return config_.preferred_batch_size;
    } else if (queue_depth >= config_.high_queue_threshold) {
        // Moderate increase to prevent queue growth
        return (config_.min_batch_size + config_.preferred_batch_size) / 2;
    } else {
        // Low latency mode - minimal batch size
        return config_.min_batch_size;
    }
}

int DynamicBatchManager::calculateBalancedBatch(
    int queue_depth, 
    float gpu_util, 
    float inference_time
) {
    // Balance throughput and latency using a weighted approach
    
    // Base calculation on queue pressure
    float queue_pressure = static_cast<float>(queue_depth) / config_.high_queue_threshold;
    queue_pressure = std::min(2.0f, queue_pressure); // Cap at 2x
    
    // GPU efficiency factor
    float gpu_efficiency = gpu_util / config_.target_gpu_utilization_percent;
    gpu_efficiency = std::max(0.5f, std::min(1.5f, gpu_efficiency));
    
    // Time efficiency - if inference is fast, we can use larger batches
    float time_efficiency = 10.0f / (inference_time + 1.0f); // Normalize around 10ms
    time_efficiency = std::max(0.5f, std::min(2.0f, time_efficiency));
    
    // Combine factors
    float batch_multiplier = queue_pressure * gpu_efficiency * time_efficiency;
    
    // Apply to preferred size
    int target_size = static_cast<int>(config_.preferred_batch_size * batch_multiplier);
    
    // Use prediction model to refine
    float predicted = predictOptimalBatchSize(queue_depth, gpu_util, inference_time);
    
    // Blend prediction with calculation
    target_size = static_cast<int>(0.7f * target_size + 0.3f * predicted);
    
    return target_size;
}

int DynamicBatchManager::calculateAdaptiveBatch(
    int queue_depth,
    float gpu_util,
    float inference_time
) {
    // Use machine learning model to predict optimal batch size
    float predicted = predictOptimalBatchSize(queue_depth, gpu_util, inference_time);
    
    // Apply safety bounds based on queue depth
    if (queue_depth >= config_.critical_queue_threshold) {
        // Override prediction in critical situations
        predicted = std::max(predicted, static_cast<float>(config_.preferred_batch_size));
    }
    
    return static_cast<int>(predicted + 0.5f); // Round to nearest
}

float DynamicBatchManager::predictOptimalBatchSize(
    int queue_depth,
    float gpu_util,
    float inference_time
) {
    // Simple linear model with adaptive weights
    float prediction = model_.bias +
                      model_.queue_weight * std::log1p(queue_depth) +
                      model_.gpu_util_weight * gpu_util +
                      model_.time_weight * inference_time;
    
    // Ensure reasonable bounds
    prediction = std::max(static_cast<float>(config_.min_batch_size), 
                         std::min(static_cast<float>(config_.max_batch_size), prediction));
    
    return prediction;
}

void DynamicBatchManager::updateMetrics(
    int batch_size,
    float inference_time_ms,
    int queue_depth_at_start
) {
    // Update exponential averages
    updateExponentialAverage(metrics_.exponential_avg_batch_size, 
                           static_cast<float>(batch_size), 
                           config_.adaptation_rate);
    updateExponentialAverage(metrics_.exponential_avg_inference_time, 
                           inference_time_ms, 
                           config_.adaptation_rate);

    // Update counters
    metrics_.total_items_processed.fetch_add(batch_size);
    metrics_.total_batches.fetch_add(1);

    // Record latency
    size_t idx = metrics_.latency_index.fetch_add(1) % PerformanceMetrics::LATENCY_BUFFER_SIZE;
    metrics_.latency_buffer[idx] = inference_time_ms;

    // Calculate throughput
    float throughput = batch_size / (inference_time_ms / 1000.0f);
    
    // Update prediction model
    float gpu_util = metrics_.exponential_avg_gpu_util.load();
    updatePredictionModel(batch_size, throughput, queue_depth_at_start, gpu_util);
}

void DynamicBatchManager::updatePredictionModel(
    int batch_size,
    float throughput,
    int queue_depth,
    float gpu_util
) {
    // Add to history
    if (model_.history.size() >= PredictionModel::MAX_HISTORY) {
        model_.history.erase(model_.history.begin());
    }
    
    model_.history.push_back({
        queue_depth,
        gpu_util,
        metrics_.exponential_avg_inference_time.load(),
        batch_size,
        throughput
    });

    // Simple online gradient descent
    if (model_.history.size() > 100) {
        // Calculate average error over recent history
        float total_error = 0.0f;
        int count = 0;
        
        for (size_t i = model_.history.size() - 50; i < model_.history.size(); ++i) {
            const auto& sample = model_.history[i];
            float predicted = predictOptimalBatchSize(
                sample.queue_depth, 
                sample.gpu_util, 
                sample.inference_time
            );
            float error = predicted - sample.optimal_batch_size;
            total_error += error * error;
            count++;
        }
        
        float mse = total_error / count;
        
        // Update weights if error is significant
        if (mse > 1.0f) {
            const auto& recent = model_.history.back();
            float predicted = predictOptimalBatchSize(
                recent.queue_depth,
                recent.gpu_util,
                recent.inference_time
            );
            float error = predicted - recent.optimal_batch_size;
            
            // Gradient descent update
            float lr = model_.learning_rate;
            model_.bias -= lr * error;
            model_.queue_weight -= lr * error * std::log1p(recent.queue_depth);
            model_.gpu_util_weight -= lr * error * recent.gpu_util;
            model_.time_weight -= lr * error * recent.inference_time;
        }
    }
}

DynamicBatchManager::Config::Mode DynamicBatchManager::determineOptimalMode(int queue_depth, float gpu_util) {
    // Adaptive mode selection based on system state
    
    if (queue_depth >= config_.critical_queue_threshold) {
        // Critical queue depth - must prioritize throughput
        return Config::THROUGHPUT;
    }
    
    if (queue_depth < config_.low_queue_threshold && gpu_util < 50.0f) {
        // Low load - prioritize latency
        return Config::LATENCY;
    }
    
    if (gpu_util < config_.target_gpu_utilization_percent - 20) {
        // GPU underutilized - increase batch size
        return Config::THROUGHPUT;
    }
    
    if (gpu_util > config_.target_gpu_utilization_percent + 10) {
        // GPU overloaded - reduce batch size
        return Config::LATENCY;
    }
    
    // Default to balanced
    return Config::BALANCED;
}

DynamicBatchManager::BatchingStats DynamicBatchManager::getStats() const {
    BatchingStats stats;
    
    stats.avg_batch_size = metrics_.exponential_avg_batch_size.load();
    stats.avg_inference_time_ms = metrics_.exponential_avg_inference_time.load();
    stats.avg_queue_depth = metrics_.exponential_avg_queue_depth.load();
    stats.gpu_efficiency = metrics_.exponential_avg_gpu_util.load() / 
                          config_.target_gpu_utilization_percent;
    
    // Calculate throughput
    auto elapsed = std::chrono::steady_clock::now() - metrics_.start_time;
    float seconds = std::chrono::duration<float>(elapsed).count();
    stats.throughput_per_second = metrics_.total_items_processed.load() / seconds;
    
    stats.p95_latency_ms = calculateP95Latency();
    stats.total_batches = metrics_.total_batches.load();
    stats.mode_switches = metrics_.mode_switches.load();
    
    return stats;
}

float DynamicBatchManager::calculateP95Latency() const {
    // Copy latency buffer for sorting
    std::vector<float> latencies = metrics_.latency_buffer;
    
    // Remove zeros (uninitialized values)
    latencies.erase(
        std::remove(latencies.begin(), latencies.end(), 0.0f),
        latencies.end()
    );
    
    if (latencies.empty()) {
        return 0.0f;
    }
    
    // Sort and find 95th percentile
    std::sort(latencies.begin(), latencies.end());
    size_t p95_index = static_cast<size_t>(latencies.size() * 0.95);
    
    return latencies[p95_index];
}

void DynamicBatchManager::reset() {
    // Reset metrics
    metrics_.exponential_avg_batch_size.store(config_.preferred_batch_size);
    metrics_.exponential_avg_inference_time.store(10.0f);
    metrics_.exponential_avg_queue_depth.store(50.0f);
    metrics_.exponential_avg_gpu_util.store(70.0f);
    metrics_.total_items_processed.store(0);
    metrics_.total_batches.store(0);
    metrics_.start_time = std::chrono::steady_clock::now();
    metrics_.latency_index.store(0);
    metrics_.mode_switches.store(0);
    metrics_.current_mode.store(config_.optimization_mode);
    
    // Clear latency buffer
    std::fill(metrics_.latency_buffer.begin(), metrics_.latency_buffer.end(), 0.0f);
    
    // Reset model
    model_.queue_weight = 0.3f;
    model_.gpu_util_weight = -0.2f;
    model_.time_weight = -0.1f;
    model_.bias = 16.0f;
    model_.history.clear();
}

} // namespace mcts
} // namespace alphazero