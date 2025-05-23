#ifndef ALPHAZERO_MCTS_LOCK_FREE_BATCH_ACCUMULATOR_H
#define ALPHAZERO_MCTS_LOCK_FREE_BATCH_ACCUMULATOR_H

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "core/export_macros.h"
#include "moodycamel/concurrentqueue.h"
#include "mcts/mcts_node.h"

namespace alphazero {
namespace mcts {

// Forward declarations
struct PendingEvaluation;

struct ALPHAZERO_API LockFreeBatchConfig {
    // Target batch size for optimal performance
    size_t target_batch_size = 256;
    
    // Maximum time to wait for batch formation
    std::chrono::milliseconds max_wait_time = std::chrono::milliseconds(10);
};

/**
 * @brief Lock-free batch accumulator for efficient request batching
 * 
 * Collects and processes batch requests without any locks, providing
 * efficient batch formation with minimal contention.
 */
class ALPHAZERO_API LockFreeBatchAccumulator {
public:
    /**
     * @brief Constructor with configuration
     * 
     * @param config Batch accumulator configuration
     */
    explicit LockFreeBatchAccumulator(const LockFreeBatchConfig& config = LockFreeBatchConfig());
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~LockFreeBatchAccumulator();
    
    /**
     * @brief Submit a request to be batched
     * 
     * @param request Request to be added to a batch
     */
    void submitRequest(PendingEvaluation&& request);
    
    /**
     * @brief Collect a batch of ready requests
     * 
     * @return Vector of requests ready for processing
     */
    std::vector<PendingEvaluation> collectBatch();
    
    /**
     * @brief Get the number of pending requests
     * 
     * @return Pending request count
     */
    size_t pendingCount() const;
    
    /**
     * @brief Get the number of ready requests
     * 
     * @return Ready request count
     */
    size_t readyCount() const;
    
    /**
     * @brief Shutdown the accumulator and join threads
     */
    void shutdown();
    
private:
    // Configuration
    LockFreeBatchConfig config_;
    
    // Thread control
    std::atomic<bool> shutdown_;
    std::thread accumulation_thread_;
    
    // Request tracking
    std::atomic<size_t> pending_count_{0};
    std::atomic<size_t> ready_count_{0};
    
    // Lock-free queues
    moodycamel::ConcurrentQueue<PendingEvaluation> pending_queue_;
    moodycamel::ConcurrentQueue<PendingEvaluation> ready_queue_;
    
    // Worker methods
    void accumulationLoop();
    void flushBatch(std::vector<PendingEvaluation>&& batch);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_LOCK_FREE_BATCH_ACCUMULATOR_H