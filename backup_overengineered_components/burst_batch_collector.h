#ifndef ALPHAZERO_BURST_BATCH_COLLECTOR_H
#define ALPHAZERO_BURST_BATCH_COLLECTOR_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <moodycamel/concurrentqueue.h>

// Need full definition for template instantiation
#include "mcts/mcts_engine.h"  // For PendingEvaluation definition

/**
 * @brief High-efficiency burst-mode batch collector for MCTS evaluations
 * 
 * This collector uses aggressive burst collection strategies to rapidly
 * accumulate large batches for GPU evaluation, solving the low batch size
 * problem that plagues traditional incremental collection approaches.
 * 
 * Key features:
 * - Burst collection: Rapidly collects items in short time windows
 * - Target-oriented: Always aims for large batches (64+ items)
 * - Memory efficient: Minimal allocation overhead
 * - Thread-safe: Lock-free queues with minimal critical sections
 */
class BurstBatchCollector {
public:
    /**
     * @brief Construct a new burst batch collector
     * 
     * @param target_batch_size Target size for collected batches (typically 64-256)
     * @param burst_timeout Maximum time to wait for batch completion
     */
    BurstBatchCollector(size_t target_batch_size = 64, std::chrono::milliseconds burst_timeout = std::chrono::milliseconds(20));
    
    ~BurstBatchCollector();
    
    /**
     * @brief Start the burst collection system
     */
    void start();
    
    /**
     * @brief Shutdown the burst collection system
     */
    void shutdown();
    
    /**
     * @brief Submit a pending evaluation for batch collection
     * 
     * @param eval Pending evaluation to be batched
     */
    void submitEvaluation(PendingEvaluation eval);
    
    /**
     * @brief Collect a completed batch (blocking)
     * 
     * @return std::vector<PendingEvaluation> Batch of evaluations ready for processing
     */
    std::vector<PendingEvaluation> collectBatch();
    
    /**
     * @brief Check if a completed batch is available
     * 
     * @return true if batch available, false otherwise
     */
    bool hasPendingBatch() const;
    
    /**
     * @brief Get count of pending evaluations
     * 
     * @return size_t Number of evaluations waiting to be batched
     */
    size_t getPendingCount() const;
    
    /**
     * @brief Get average batch size achieved
     * 
     * @return float Average size of completed batches
     */
    float getAverageBatchSize() const;

private:
    // Configuration
    size_t target_batch_size_;
    std::chrono::milliseconds burst_timeout_;
    
    // Control flags
    std::atomic<bool> shutdown_;
    std::atomic<bool> collection_active_;
    
    // Statistics
    std::atomic<size_t> total_batches_collected_;
    std::atomic<size_t> total_items_collected_;
    
    // Collection thread
    std::thread collection_thread_;
    std::mutex collection_mutex_;
    std::condition_variable collection_cv_;
    
    // Batch management
    mutable std::mutex batch_mutex_;
    std::condition_variable ready_cv_;
    std::queue<std::vector<PendingEvaluation>> completed_batches_;
    
    // Working batch
    std::vector<PendingEvaluation> current_batch_;
    
    // Input queue for pending evaluations
    moodycamel::ConcurrentQueue<PendingEvaluation> pending_queue_;
    
    // Core collection logic
    void burstCollectionLoop();
    void collectBurstItems(size_t max_items);
    void validateAndCleanBatch();
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_BURST_BATCH_COLLECTOR_H