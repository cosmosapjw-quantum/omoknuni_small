// include/mcts/batch_barrier.h
#ifndef BATCH_BARRIER_H
#define BATCH_BARRIER_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <vector>
#include <functional>
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Barrier synchronization for optimal batch formation
 * 
 * This class implements a reusable barrier that allows multiple MCTS threads
 * to synchronize their leaf evaluations, ensuring optimal batch sizes for
 * GPU inference.
 */
class ALPHAZERO_API BatchBarrier {
public:
    using BatchCallback = std::function<void(const std::vector<size_t>&)>;
    
    /**
     * @brief Constructor
     * 
     * @param target_batch_size Target number of threads to wait for
     * @param timeout Maximum wait time before proceeding with partial batch
     */
    BatchBarrier(size_t target_batch_size, std::chrono::milliseconds timeout);
    
    /**
     * @brief Wait at barrier and return thread's position in batch
     * 
     * @param thread_id Unique identifier for the calling thread
     * @return Position in the batch (0 to batch_size-1)
     */
    size_t arrive_and_wait(size_t thread_id);
    
    /**
     * @brief Try to wait at barrier with custom timeout
     * 
     * @param thread_id Unique identifier for the calling thread
     * @param custom_timeout Custom timeout for this wait
     * @return Position in batch, or size_t(-1) if timed out
     */
    size_t try_arrive_and_wait(size_t thread_id, std::chrono::milliseconds custom_timeout);
    
    /**
     * @brief Register a callback to be called when batch is formed
     */
    void set_batch_callback(BatchCallback callback) {
        batch_callback_ = callback;
    }
    
    /**
     * @brief Get current number of waiting threads
     */
    size_t get_waiting_count() const {
        return waiting_count_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Reset the barrier for reuse
     */
    void reset();
    
    /**
     * @brief Force release all waiting threads
     */
    void force_release();
    
    /**
     * @brief Statistics for barrier usage
     */
    struct BarrierStats {
        std::atomic<size_t> total_batches{0};
        std::atomic<size_t> full_batches{0};
        std::atomic<size_t> partial_batches{0};
        std::atomic<size_t> timeouts{0};
        std::atomic<size_t> total_wait_time_us{0};
        
        double getFullBatchRate() const {
            size_t total = total_batches.load();
            return total > 0 ? static_cast<double>(full_batches.load()) / total : 0.0;
        }
        
        double getAverageWaitTimeMs() const {
            size_t batches = total_batches.load();
            return batches > 0 ? static_cast<double>(total_wait_time_us.load()) / (batches * 1000.0) : 0.0;
        }
    };
    
    const BarrierStats& getStats() const { return stats_; }
    
private:
    const size_t target_batch_size_;
    const std::chrono::milliseconds default_timeout_;
    
    // Synchronization primitives
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    // Barrier state
    std::atomic<size_t> waiting_count_{0};
    std::atomic<size_t> generation_{0};
    std::vector<size_t> waiting_threads_;
    bool force_release_{false};
    
    // Timing
    std::chrono::steady_clock::time_point batch_start_time_;
    
    // Callback
    BatchCallback batch_callback_;
    
    // Statistics
    mutable BarrierStats stats_;
    
    /**
     * @brief Internal wait implementation
     */
    size_t wait_internal(size_t thread_id, std::chrono::milliseconds timeout);
};

/**
 * @brief Thread-local batch coordinator for MCTS threads
 * 
 * This class manages thread-local state for batch coordination,
 * reducing contention and improving cache locality.
 */
class ALPHAZERO_API ThreadLocalBatchCoordinator {
public:
    struct ThreadState {
        size_t thread_id;
        size_t batch_position;
        std::vector<std::pair<void*, size_t>> pending_evaluations;  // (state_ptr, move_count)
        std::chrono::steady_clock::time_point arrival_time;
        
        ThreadState(size_t id) : thread_id(id), batch_position(0) {}
    };
    
    /**
     * @brief Get thread-local state
     */
    static ThreadState& getThreadState();
    
    /**
     * @brief Register evaluation request
     */
    static void registerEvaluation(void* state_ptr, size_t move_count);
    
    /**
     * @brief Clear pending evaluations
     */
    static void clearPendingEvaluations();
    
    /**
     * @brief Get total pending evaluations across all threads
     */
    static size_t getTotalPendingEvaluations();
    
private:
    static thread_local std::unique_ptr<ThreadState> thread_state_;
    static std::atomic<size_t> next_thread_id_;
};

} // namespace mcts
} // namespace alphazero

#endif // BATCH_BARRIER_H