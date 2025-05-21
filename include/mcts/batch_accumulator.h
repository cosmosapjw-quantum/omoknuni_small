#ifndef ALPHAZERO_MCTS_BATCH_ACCUMULATOR_H
#define ALPHAZERO_MCTS_BATCH_ACCUMULATOR_H

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
#include "mcts/mcts_node.h"
#include "mcts/mcts_engine.h"
#include "mcts/prioritized_concurrent_queue.h"
#include "core/export_macros.h"

#include <moodycamel/concurrentqueue.h>

namespace alphazero {
namespace mcts {

/**
 * @brief Central batch accumulator for efficient neural network batch processing
 * 
 * This class collects evaluation requests from multiple threads and forms optimal
 * batches for neural network inference. It uses a dedicated thread for batch collection
 * to ensure maximum GPU utilization through larger batch sizes.
 */
class ALPHAZERO_API BatchAccumulator {
private:
    std::mutex accumulator_mutex_;
    std::condition_variable cv_;
    std::vector<PendingEvaluation> current_batch_;
    std::atomic<bool> batch_ready_{false};
    std::atomic<bool> shutdown_{false};
    size_t target_batch_size_;
    size_t min_viable_batch_size_;
    std::chrono::milliseconds max_wait_time_{50}; // Much longer than current 5-15ms
    std::thread accumulator_thread_;
    
    // Queue for completed batches - lock free
    moodycamel::ConcurrentQueue<std::vector<PendingEvaluation>> completed_batches_;
    
    // Batch metrics for dynamic adjustment
    std::atomic<size_t> total_batches_{0};
    std::atomic<size_t> total_evaluations_{0};
    std::atomic<size_t> cumulative_batch_size_{0};
    std::atomic<size_t> batch_timeouts_{0};
    std::atomic<size_t> optimal_batches_{0};
    
    // Count of consecutive empty iterations for deadlock detection
    std::atomic<int> consecutive_empty_iterations_{0};
    
    // Timestamp when batch started accumulating
    std::chrono::steady_clock::time_point batch_start_time_;
    
    // Batch prioritization using lock-free concurrent queue
    using BatchPriority = QueuePriority;
    struct PrioritizedBatch {
        std::vector<PendingEvaluation> evals;
        std::chrono::steady_clock::time_point created_time;
    };
    
    // Priority queue implementation using PrioritizedConcurrentQueue
    PrioritizedConcurrentQueue<PrioritizedBatch> priority_queue_;
    
    // Main accumulator loop function
    void accumulatorLoop();
    
    // Helper function for batch prioritization
    BatchPriority calculateBatchPriority(const std::vector<PendingEvaluation>& evals, 
                                       std::chrono::steady_clock::time_point created_time) const;
    
public:
    /**
     * @brief Construct a new Batch Accumulator
     * 
     * @param target_batch_size The optimal batch size to achieve
     * @param min_viable_batch_size Minimum acceptable batch size (default: 75% of target)
     * @param max_wait_time Maximum time to wait for batch formation (default: 50ms)
     */
    BatchAccumulator(size_t target_batch_size, 
                   size_t min_viable_batch_size = 0,
                   std::chrono::milliseconds max_wait_time = std::chrono::milliseconds(50));
    
    /**
     * @brief Destructor - ensures thread is properly shut down
     */
    ~BatchAccumulator();
    
    /**
     * @brief Add a single evaluation to the accumulator
     * 
     * @param eval The evaluation to add
     */
    void addEvaluation(PendingEvaluation&& eval);
    
    /**
     * @brief Submit a pre-formed batch directly to completed queue
     * 
     * @param batch Pre-formed batch to submit directly
     */
    void submitDirectBatch(std::vector<PendingEvaluation>&& batch);
    
    /**
     * @brief Add multiple evaluations at once
     * 
     * @tparam Iterator Iterator type for the evaluations collection
     * @param begin Iterator to the beginning of the evaluations
     * @param end Iterator to the end of the evaluations
     */
    template<typename Iterator>
    void addEvaluationBulk(Iterator begin, Iterator end) {
        std::lock_guard<std::mutex> lock(accumulator_mutex_);
        current_batch_.insert(current_batch_.end(), begin, end);
        if (current_batch_.size() >= target_batch_size_) {
            batch_ready_ = true;
            cv_.notify_one();
        }
    }
    
    /**
     * @brief Get a completed batch (non-blocking)
     * 
     * @param batch Output parameter to receive the batch
     * @return true if a batch was available, false otherwise
     */
    bool getCompletedBatch(std::vector<PendingEvaluation>& batch);
    
    /**
     * @brief Start the accumulator thread
     */
    void start();
    
    /**
     * @brief Stop the accumulator thread
     */
    void stop();
    
    /**
     * @brief Check if the accumulator is running
     * 
     * @return true if running, false otherwise
     */
    bool isRunning() const { return !shutdown_.load(std::memory_order_acquire) && accumulator_thread_.joinable(); }
    
    /**
     * @brief Get current statistics
     * 
     * @return Tuple of (avg_batch_size, total_batches, timeouts, optimal_batches)
     */
    std::tuple<float, size_t, size_t, size_t> getStats() const;
    
    /**
     * @brief Set new batch parameters
     * 
     * @param target_size New target batch size
     * @param min_viable_size New minimum viable batch size
     * @param max_wait New maximum wait time
     */
    void updateParameters(size_t target_size, 
                         size_t min_viable_size = 0,
                         std::chrono::milliseconds max_wait = std::chrono::milliseconds(50));
                         
    /**
     * @brief Reset the batch accumulator, clearing all pending evaluations
     */
    void reset();
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_BATCH_ACCUMULATOR_H