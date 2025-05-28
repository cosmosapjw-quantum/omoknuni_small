#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <future>
#include "core/igamestate.h"
#include "nn/neural_network.h"
#include "mcts/evaluation_types.h"
#include "moodycamel/concurrentqueue.h"

namespace alphazero {
namespace mcts {

// Shared inference queue for batching neural network calls across multiple MCTS engines
class SharedInferenceQueue {
public:
    struct InferenceRequest {
        std::vector<std::unique_ptr<core::IGameState>> states;
        std::promise<std::vector<mcts::NetworkOutput>> promise;
        std::chrono::steady_clock::time_point timestamp;
        
        // For moodycamel::ConcurrentQueue move semantics
        InferenceRequest() = default;
        InferenceRequest(InferenceRequest&&) = default;
        InferenceRequest& operator=(InferenceRequest&&) = default;
    };
    
    explicit SharedInferenceQueue(
        std::shared_ptr<nn::NeuralNetwork> neural_net,
        int max_batch_size = 256,
        int batch_timeout_ms = 20
    );
    
    ~SharedInferenceQueue();
    
    // Submit states for inference and get future result
    std::future<std::vector<mcts::NetworkOutput>> submitBatch(
        std::vector<std::unique_ptr<core::IGameState>> states
    );
    
    // Start/stop the processing thread
    void start();
    void stop();
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> total_states{0};
        std::atomic<uint64_t> total_batches{0};
        std::atomic<uint64_t> total_inference_time_ms{0};
        std::atomic<double> average_batch_size{0};
        std::atomic<double> gpu_utilization{0};
    };
    
    const Stats& getStats() const { return stats_; }
    
private:
    void processingLoop();
    void processBatch(std::vector<InferenceRequest>& batch);
    
    std::shared_ptr<nn::NeuralNetwork> neural_net_;
    int max_batch_size_;
    int batch_timeout_ms_;
    
    // High-performance lock-free queue
    moodycamel::ConcurrentQueue<InferenceRequest> request_queue_;
    
    // Token for bulk operations
    moodycamel::ProducerToken producer_token_;
    moodycamel::ConsumerToken consumer_token_;
    
    // Notification mechanism (kept for timeout handling)
    std::mutex notify_mutex_;
    std::condition_variable notify_cv_;
    std::atomic<int> pending_requests_{0};
    
    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
    
    // Statistics
    Stats stats_;
    
    // GPU attack/defense batching support
    bool use_gpu_attack_defense_{false};
};

// Global shared queue instance
class GlobalInferenceQueue {
public:
    static SharedInferenceQueue& getInstance() {
        if (!instance_) {
            throw std::runtime_error("GlobalInferenceQueue not initialized");
        }
        return *instance_;
    }
    
    static void initialize(
        std::shared_ptr<nn::NeuralNetwork> neural_net,
        int max_batch_size = 256,
        int batch_timeout_ms = 20
    ) {
        static std::mutex init_mutex;
        std::lock_guard<std::mutex> lock(init_mutex);
        
        if (!instance_) {
            instance_ = std::make_unique<SharedInferenceQueue>(
                neural_net, max_batch_size, batch_timeout_ms
            );
            instance_->start();
        }
    }
    
    static void shutdown() {
        if (instance_) {
            instance_->stop();
            instance_.reset();
        }
    }
    
    static bool isInitialized() {
        return instance_ != nullptr;
    }
    
private:
    static std::unique_ptr<SharedInferenceQueue> instance_;
};

} // namespace mcts
} // namespace alphazero