// include/mcts/dynamic_batch_manager.h
#ifndef ALPHAZERO_DYNAMIC_BATCH_MANAGER_H
#define ALPHAZERO_DYNAMIC_BATCH_MANAGER_H

#include <atomic>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>
#include <functional>
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Dynamic Batch Manager for optimal GPU utilization
 * 
 * This class implements sophisticated algorithms to dynamically adjust
 * batch sizes based on queue depth, GPU utilization, and latency targets.
 * 
 * Features:
 * - Adaptive batch sizing based on queue depth
 * - GPU utilization tracking and optimization
 * - Latency-aware batching
 * - Predictive batching using exponential smoothing
 * - Multi-objective optimization (throughput vs latency)
 */
class ALPHAZERO_API DynamicBatchManager {
public:
    struct Config {
        // Batch size limits
        int min_batch_size;
        int max_batch_size;
        int preferred_batch_size;
        
        // Timing parameters
        int max_wait_time_ms;
        int target_gpu_utilization_percent;
        
        // Queue depth thresholds
        int low_queue_threshold;
        int high_queue_threshold;
        int critical_queue_threshold;
        
        // Adaptation parameters
        float adaptation_rate;
        float smoothing_factor;
        
        // Optimization mode
        enum Mode {
            THROUGHPUT,    // Maximize throughput
            LATENCY,       // Minimize latency
            BALANCED,      // Balance both
            ADAPTIVE       // Switch based on conditions
        };
        Mode optimization_mode;
        
        // Constructor with defaults
        Config() : 
            min_batch_size(1),
            max_batch_size(64),
            preferred_batch_size(16),
            max_wait_time_ms(5),
            target_gpu_utilization_percent(85),
            low_queue_threshold(10),
            high_queue_threshold(100),
            critical_queue_threshold(500),
            adaptation_rate(0.1f),
            smoothing_factor(0.8f),
            optimization_mode(BALANCED) {}
    };

    explicit DynamicBatchManager(const Config& config = Config());

    /**
     * @brief Calculate optimal batch size based on current conditions
     * 
     * @param queue_depth Current number of items waiting in queue
     * @param gpu_utilization Current GPU utilization (0-100)
     * @param recent_inference_time_ms Recent average inference time
     * @return Optimal batch size
     */
    int calculateOptimalBatchSize(
        int queue_depth,
        float gpu_utilization,
        float recent_inference_time_ms
    );

    /**
     * @brief Update performance metrics after batch completion
     * 
     * @param batch_size Size of completed batch
     * @param inference_time_ms Time taken for inference
     * @param queue_depth_at_start Queue depth when batch started
     */
    void updateMetrics(
        int batch_size,
        float inference_time_ms,
        int queue_depth_at_start
    );

    /**
     * @brief Get current batching statistics
     */
    struct BatchingStats {
        float avg_batch_size;
        float avg_inference_time_ms;
        float avg_queue_depth;
        float gpu_efficiency;
        float throughput_per_second;
        float p95_latency_ms;
        int total_batches;
        int mode_switches;
    };
    BatchingStats getStats() const;

    /**
     * @brief Reset all statistics and adaptation state
     */
    void reset();

private:
    Config config_;

    // Performance tracking
    struct PerformanceMetrics {
        std::atomic<float> exponential_avg_batch_size{16.0f};
        std::atomic<float> exponential_avg_inference_time{10.0f};
        std::atomic<float> exponential_avg_queue_depth{50.0f};
        std::atomic<float> exponential_avg_gpu_util{70.0f};
        
        // Throughput tracking
        std::atomic<int> total_items_processed{0};
        std::atomic<int> total_batches{0};
        std::chrono::steady_clock::time_point start_time;
        
        // Latency tracking (circular buffer)
        static constexpr size_t LATENCY_BUFFER_SIZE = 1000;
        std::vector<float> latency_buffer;
        std::atomic<size_t> latency_index{0};
        
        // Mode switching
        std::atomic<int> mode_switches{0};
        std::atomic<Config::Mode> current_mode{Config::BALANCED};
    };
    mutable PerformanceMetrics metrics_;

    // Batch size prediction model
    struct PredictionModel {
        // Linear regression parameters for batch size prediction
        float queue_weight = 0.3f;
        float gpu_util_weight = -0.2f;
        float time_weight = -0.1f;
        float bias = 16.0f;
        
        // Adaptive learning
        float learning_rate = 0.01f;
        
        // Performance history for adaptation
        struct Sample {
            int queue_depth;
            float gpu_util;
            float inference_time;
            int optimal_batch_size;
            float achieved_throughput;
        };
        std::vector<Sample> history;
        static constexpr size_t MAX_HISTORY = 1000;
    };
    PredictionModel model_;

    // Helper methods
    int calculateThroughputOptimalBatch(int queue_depth, float gpu_util);
    int calculateLatencyOptimalBatch(int queue_depth, float gpu_util);
    int calculateBalancedBatch(int queue_depth, float gpu_util, float inference_time);
    int calculateAdaptiveBatch(int queue_depth, float gpu_util, float inference_time);
    
    void updatePredictionModel(int batch_size, float throughput, int queue_depth, float gpu_util);
    float predictOptimalBatchSize(int queue_depth, float gpu_util, float inference_time);
    
    Config::Mode determineOptimalMode(int queue_depth, float gpu_util);
    float calculateP95Latency() const;
    
    // Exponential smoothing update
    template<typename T>
    void updateExponentialAverage(std::atomic<T>& avg, T new_value, float alpha) {
        T old_avg = avg.load();
        T new_avg = alpha * new_value + (1.0f - alpha) * old_avg;
        avg.store(new_avg);
    }
};

/**
 * @brief Advanced batch collector with dynamic sizing
 * 
 * Integrates with DynamicBatchManager to collect batches optimally
 */
template<typename T>
class ALPHAZERO_API DynamicBatchCollector {
public:
    using BatchCallback = std::function<void(std::vector<T>&&)>;

    DynamicBatchCollector(
        std::shared_ptr<DynamicBatchManager> manager,
        BatchCallback callback
    ) : manager_(manager), callback_(callback) {}

    void add(T item) {
        items_.push_back(std::move(item));
        
        // Check if we should dispatch
        int optimal_size = manager_->calculateOptimalBatchSize(
            items_.size(),
            getCurrentGPUUtilization(),
            getRecentInferenceTime()
        );
        
        if (items_.size() >= optimal_size || shouldDispatchByTime()) {
            dispatch();
        }
    }

    void flush() {
        if (!items_.empty()) {
            dispatch();
        }
    }

private:
    std::shared_ptr<DynamicBatchManager> manager_;
    BatchCallback callback_;
    std::vector<T> items_;
    std::chrono::steady_clock::time_point last_dispatch_;

    void dispatch() {
        if (items_.empty()) return;
        
        auto start = std::chrono::steady_clock::now();
        int batch_size = items_.size();
        
        // Send batch for processing
        callback_(std::move(items_));
        items_.clear();
        
        auto end = std::chrono::steady_clock::now();
        float inference_time = std::chrono::duration<float, std::milli>(end - start).count();
        
        // Update manager metrics
        manager_->updateMetrics(batch_size, inference_time, items_.size());
        
        last_dispatch_ = end;
    }

    bool shouldDispatchByTime() const {
        if (last_dispatch_.time_since_epoch().count() == 0) {
            return false;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - last_dispatch_;
        return elapsed > std::chrono::milliseconds(manager_->config_.max_wait_time_ms);
    }

    float getCurrentGPUUtilization() const {
        // This would integrate with actual GPU monitoring
        // For now, return estimate based on queue depth
        return std::min(100.0f, items_.size() * 2.0f);
    }

    float getRecentInferenceTime() const {
        // Return recent average from manager stats
        return manager_->getStats().avg_inference_time_ms;
    }
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_DYNAMIC_BATCH_MANAGER_H