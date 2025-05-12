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
#include "mcts/evaluation_types.h"
#include "core/export_macros.h"

#include "../third_party/concurrentqueue.h"

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
    std::future<NetworkOutput> evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state);
    
    // Get metrics
    size_t getQueueSize() const;
    float getAverageBatchSize() const;
    std::chrono::milliseconds getAverageBatchLatency() const;
    size_t getTotalEvaluations() const;

    // Get direct access to the inference function (for serial mode)
    InferenceFunction getInferenceFunction() const { return inference_fn_; }
    
private:
    // Worker thread function
    void processBatches();
    
    // Collect a batch of requests from the queue
    std::vector<EvaluationRequest> collectBatch();
    
    // Process a batch of requests
    void processBatch(std::vector<EvaluationRequest>& batch);
    
    // Neural network inference function
    InferenceFunction inference_fn_;
    
    // Batch processing parameters
    size_t batch_size_;
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
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_EVALUATOR_H