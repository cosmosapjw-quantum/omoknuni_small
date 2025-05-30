#pragma once

#include "nn/neural_network.h"
#include "core/igamestate.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>

namespace alphazero {
namespace mcts {

/**
 * Shared evaluation server that batches requests from multiple MCTS engines
 * 
 * This allows multiple games to share GPU batches for better utilization
 */
class SharedEvaluationServer {
public:
    struct Config {
        size_t max_batch_size;
        size_t min_batch_size;
        float batch_timeout_ms;
        size_t num_worker_threads;
        bool enable_priority_queue;
        
        Config() : max_batch_size(512), min_batch_size(128), 
                   batch_timeout_ms(10.0f), num_worker_threads(2),
                   enable_priority_queue(false) {}
    };
    
    struct EvaluationRequest {
        std::unique_ptr<core::IGameState> state;
        std::promise<std::pair<std::vector<float>, float>> promise;
        int priority = 0;  // Higher priority evaluated first
        int game_id = -1;  // Track which game this is from
    };
    
    SharedEvaluationServer(std::shared_ptr<nn::NeuralNetwork> network, 
                          const Config& config = Config());
    ~SharedEvaluationServer();
    
    // Submit a state for evaluation (non-blocking)
    std::future<std::pair<std::vector<float>, float>> evaluate(
        std::unique_ptr<core::IGameState> state,
        int game_id = -1,
        int priority = 0);
    
    // Start/stop the evaluation threads
    void start();
    void stop();
    
    // Get statistics
    struct Stats {
        size_t total_evaluations;
        size_t total_batches;
        double avg_batch_size;
        double avg_wait_time_ms;
        double avg_inference_time_ms;
        size_t pending_requests;
    };
    Stats getStats() const;
    
private:
    Config config_;
    std::shared_ptr<nn::NeuralNetwork> network_;
    
    // Request queue
    std::queue<EvaluationRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // Statistics
    std::atomic<size_t> total_evaluations_{0};
    std::atomic<size_t> total_batches_{0};
    std::atomic<size_t> total_batch_size_{0};
    std::atomic<size_t> total_wait_time_ms_{0};
    std::atomic<size_t> total_inference_time_ms_{0};
    
    // Worker thread function
    void workerLoop();
    
    // Process a batch of requests
    void processBatch(std::vector<EvaluationRequest>& batch);
};

/**
 * Global evaluation server singleton
 */
class GlobalEvaluationServer {
public:
    static void initialize(std::shared_ptr<nn::NeuralNetwork> network,
                          const SharedEvaluationServer::Config& config = {});
    
    static SharedEvaluationServer* get();
    
    static void shutdown();
    
private:
    static std::unique_ptr<SharedEvaluationServer> instance_;
    static std::mutex mutex_;
};

}  // namespace mcts
}  // namespace alphazero