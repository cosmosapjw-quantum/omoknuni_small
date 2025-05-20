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
#include <future>
#include <queue>
#include "mcts/evaluation_types.h"
#include "mcts/mcts_node.h"
#include "core/export_macros.h"
#include "mcts/mcts_engine.h"

#include <moodycamel/concurrentqueue.h>

namespace alphazero {
namespace mcts {

// Structure for batched inference jobs
struct BatchForInference {
    std::vector<std::unique_ptr<core::IGameState>> states;
    std::vector<PendingEvaluation> pending_evals;
    size_t batch_id;
    std::chrono::steady_clock::time_point created_time;
};

// Structure for inference results
struct BatchInferenceResult {
    std::vector<NetworkOutput> outputs;
    std::vector<PendingEvaluation> pending_evals;
    size_t batch_id;
    std::chrono::steady_clock::time_point processed_time;
};

class ALPHAZERO_API MCTSEvaluator {
public:
    // Signature for neural network inference function
    using InferenceFunction = std::function<std::vector<NetworkOutput>(
        const std::vector<std::unique_ptr<core::IGameState>>&)>;
    
    MCTSEvaluator(InferenceFunction inference_fn, 
                 size_t batch_size = 16, 
                 std::chrono::milliseconds timeout = std::chrono::milliseconds(5),
                 size_t num_inference_threads = 1);
    
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
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // int evaluator_id = reinterpret_cast<uintptr_t>(this) & 0xFFFF;
        // std::cout << "[EVALUATOR-" << evaluator_id << "] setExternalQueues called with leaf_queue=" << leaf_queue 
        //           << ", result_queue=" << result_queue << std::endl;
        leaf_queue_ptr_ = leaf_queue;
        result_queue_ptr_ = result_queue;
        result_notify_callback_ = result_notify_callback;
        use_external_queues_ = true;
        // External queues set
    }
    
private:
    // Main batch collector thread function
    void batchCollectorLoop();
    
    // NN inference worker thread function
    void inferenceWorkerLoop();
    
    // Result distributor thread function (for internal queue mode)
    void resultDistributorLoop();
    
    // Process a single batch (old method, being deprecated)
    bool processBatch();
    
    // Collect a batch from external queue with improved batching logic
    // This optimized implementation collects larger and more efficient batches
    std::vector<PendingEvaluation> collectExternalBatch(size_t target_batch_size);
    
    // Collect a batch of requests from the internal queue with improved batching logic
    // If target_batch_size is 0, uses the default batch_size_
    std::vector<EvaluationRequest> collectInternalBatch(size_t target_batch_size = 0);
    
    // Process a batch of internal requests
    void processInternalBatch(std::vector<EvaluationRequest>& batch);
    
    // Helper for adaptive waiting with exponential backoff
    // Waits until the predicate returns true, using increasingly longer
    // wait times to reduce CPU usage when waiting
    void waitWithBackoff(std::function<bool()> predicate, std::chrono::milliseconds max_wait_time);
    
    // Missing declarations
    void evaluationLoop();
    void processBatches();
    
    // Neural network inference function
    InferenceFunction inference_fn_;
    
    // Batch processing parameters
    size_t batch_size_;
    size_t original_batch_size_;  // Store original for adaptation
    std::chrono::milliseconds timeout_;
    
    // Queue for collecting evaluation requests
    moodycamel::ConcurrentQueue<EvaluationRequest> request_queue_;
    
    // Thread pool
    std::thread batch_collector_thread_;
    std::vector<std::thread> inference_worker_threads_;
    std::thread result_distributor_thread_;
    std::atomic<bool> shutdown_flag_;
    
    // Thread pool size
    size_t num_inference_threads_;
    
    // Metrics
    std::atomic<size_t> total_batches_;
    std::atomic<size_t> total_evaluations_;
    std::atomic<size_t> cumulative_batch_size_;
    std::atomic<size_t> cumulative_batch_time_ms_;
    
    // Condvar for early wakeup
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    
    // Start/stop synchronization
    std::mutex start_mutex_;
    
    // Diagnostic counters
    std::atomic<size_t> timeouts_{0};
    std::atomic<size_t> full_batches_{0};
    std::atomic<size_t> partial_batches_{0};
    
    // Batch processing parameters
    size_t min_batch_size_{4};  // Minimum batch size to process
    size_t optimal_batch_size_{16}; // Target batch size for best performance
    
    // Time to wait for additional requests after reaching min_batch_size
    std::chrono::milliseconds additional_wait_time_{5};
    
    // Queue for inference jobs between batch collector and inference workers
    moodycamel::ConcurrentQueue<BatchForInference> inference_queue_;
    
    // Queue for results between inference workers and result processor
    moodycamel::ConcurrentQueue<BatchInferenceResult> result_queue_internal_;
    
    // Queue size tracking for adaptive timeouts
    std::atomic<size_t> last_queue_size_{0};
    
    // External queue integration
    void* leaf_queue_ptr_{nullptr};
    void* result_queue_ptr_{nullptr};
    std::function<void()> result_notify_callback_;
    bool use_external_queues_{false};
    
    // Queue mutex for synchronization
    std::mutex queue_mutex_;
    
    // Condition variables for synchronization
    std::condition_variable batch_ready_cv_; // For batch collection
    std::condition_variable inference_cv_;   // For inference workers
    std::condition_variable result_cv_;      // For result processing
    
    // Mutexes for condition variables
    std::mutex inference_mutex_;
    std::mutex result_mutex_;
    
    // Atomic counters for tracking queue sizes
    std::atomic<size_t> pending_inference_batches_{0};
    std::atomic<size_t> pending_result_batches_{0};
    
    // Batch counter for tracking
    std::atomic<size_t> batch_counter_{0};
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_EVALUATOR_H