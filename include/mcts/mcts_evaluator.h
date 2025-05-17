// include/mcts/mcts_evaluator.h
#ifndef ALPHAZERO_MCTS_EVALUATOR_H
#define ALPHAZERO_MCTS_EVALUATOR_H

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include "mcts/evaluation_types.h"
#include "core/export_macros.h"

#include <moodycamel/concurrentqueue.h>

namespace alphazero {
namespace mcts {

class ALPHAZERO_API MCTSEvaluator {
public:
    // Signature for neural network inference function
    using InferenceFunction = std::function<std::vector<NetworkOutput>(
        const std::vector<std::unique_ptr<core::IGameState>>&)>;
    
    MCTSEvaluator(InferenceFunction inference_fn, 
                 size_t batch_size = 16, 
                 std::chrono::milliseconds timeout = std::chrono::milliseconds(5));
    
    ~MCTSEvaluator();
    
    // Start the evaluator thread
    void start();
    
    // Stop the evaluator thread
    void stop();
    
    // Submit a state for evaluation
    std::future<NetworkOutput> evaluateState(std::shared_ptr<MCTSNode> node, std::unique_ptr<core::IGameState> state);
    
    // Submit a state for evaluation with external promise
    void evaluateStateAsync(std::shared_ptr<MCTSNode> node,
                          std::unique_ptr<core::IGameState> state,
                          std::shared_ptr<std::promise<NetworkOutput>> promise);
    
    // Get metrics
    size_t getQueueSize() const;
    float getAverageBatchSize() const;
    std::chrono::milliseconds getAverageBatchLatency() const;
    size_t getTotalEvaluations() const;

    // Get direct access to the inference function (for serial mode)
    InferenceFunction getInferenceFunction() const { return inference_fn_; }
    
    // Notify that a leaf is available in the external queue
    void notifyLeafAvailable();
    
    // Set external queues for direct batch processing
    void setExternalQueues(void* leaf_queue, void* result_queue, std::function<void()> result_notify_callback = nullptr) {
        leaf_queue_ptr_ = leaf_queue;
        result_queue_ptr_ = result_queue;
        result_notify_callback_ = result_notify_callback;
        use_external_queues_ = true;
        // External queues set
    }
    
private:
    // Worker thread function
    void processBatches();
    
    // New evaluation loop with smarter batching
    void evaluationLoop();
    
    // Process a single batch
    bool processBatch();
    
    // Collect a batch of requests from the queue
    // If target_batch_size is 0, uses the default batch_size_
    std::vector<EvaluationRequest> collectBatch(size_t target_batch_size = 0);
    
    // Process a batch of requests
    void processBatch(std::vector<EvaluationRequest>& batch);
    
    // Neural network inference function
    InferenceFunction inference_fn_;
    
    // Batch processing parameters
    size_t batch_size_;
    size_t original_batch_size_;  // Store original for adaptation
    std::chrono::milliseconds timeout_;
    
    // Queue for collecting evaluation requests
    moodycamel::ConcurrentQueue<EvaluationRequest> request_queue_;
    
    // Worker thread
    std::thread worker_thread_;
    std::atomic<bool> shutdown_flag_;
    
    // Metrics
    std::atomic<size_t> total_batches_;
    std::atomic<size_t> total_evaluations_;
    std::atomic<size_t> cumulative_batch_size_;
    std::atomic<size_t> cumulative_batch_time_ms_;
    
    // Condvar for early wakeup
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    
    // Diagnostic counters
    std::atomic<size_t> timeouts_{0};
    std::atomic<size_t> full_batches_{0};
    std::atomic<size_t> partial_batches_{0};
    
    // Min batch size to attempt GPU inference (optimization)
    size_t min_batch_size_{1};
    
    // Time to wait for additional requests after reaching min_batch_size
    std::chrono::milliseconds additional_wait_time_{10};
    
    // Queue size tracking for adaptive timeouts
    std::atomic<size_t> last_queue_size_{0};
    
    // External queue integration
    void* leaf_queue_ptr_{nullptr};
    void* result_queue_ptr_{nullptr};
    std::function<void()> result_notify_callback_;
    bool use_external_queues_{false};
    
    // Queue mutex for synchronization
    std::mutex queue_mutex_;
    
    // Condition variable for batch readiness
    std::condition_variable batch_ready_cv_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_EVALUATOR_H