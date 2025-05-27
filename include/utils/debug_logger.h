#ifndef ALPHAZERO_DEBUG_LOGGER_H
#define ALPHAZERO_DEBUG_LOGGER_H

#include <iostream>
#include <atomic>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace alphazero {
namespace utils {

/**
 * @brief Debug logger for tracking batch and queue operations
 */
class DebugLogger {
public:
    // Singleton instance accessor
    static DebugLogger& instance() {
        static DebugLogger instance;
        return instance;
    }
    
    // Log a message with timestamp
    void log(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t_now), "%H:%M:%S") << "] " << message;
        std::cout << ss.str() << std::endl;
    }
    
    // Log engine queue operations
    void logQueueOperation(bool enqueued, int itemId, int batchId, size_t queueSize) {
        std::stringstream ss;
        ss << "ENGINE: Queue operation - Item #" << itemId 
           << " (batch " << batchId << ") - " 
           << (enqueued ? "SUCCESS" : "FAILED")
           << " - Queue size: " << queueSize;
        log(ss.str());
    }
    
    // Log evaluator operations
    void logEvaluatorOperation(const std::string& operation, size_t batchSize, size_t totalProcessed) {
        std::stringstream ss;
        ss << "EVALUATOR: " << operation << " - Batch size: " << batchSize
           << " - Total processed: " << totalProcessed;
        log(ss.str());
    }
    
    // Log batch accumulator operations
    void logBatchAccumulator(const std::string& operation, size_t batchSize, size_t targetSize) {
        std::stringstream ss;
        ss << "BATCH ACC: " << operation << " - Batch size: " << batchSize
           << " - Target size: " << targetSize;
        log(ss.str());
    }
    
    // Track items added to queue
    int trackQueueAdd() {
        return items_added_to_queue_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    
    // Track items processed
    int trackItemProcessed() {
        return items_processed_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    
    // Track batches processed
    int trackBatchProcessed() {
        return batches_processed_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    
    // Get statistics
    std::string getStats() {
        std::stringstream ss;
        ss << "QUEUE STATS: Added: " << items_added_to_queue_.load(std::memory_order_relaxed)
           << " - Processed: " << items_processed_.load(std::memory_order_relaxed)
           << " - Batches: " << batches_processed_.load(std::memory_order_relaxed);
        return ss.str();
    }
    
private:
    // Private constructor for singleton
    DebugLogger() = default;
    
    // Prevent copying
    DebugLogger(const DebugLogger&) = delete;
    DebugLogger& operator=(const DebugLogger&) = delete;
    
    // Synchronization
    std::mutex mutex_;
    
    // Statistics
    std::atomic<int> items_added_to_queue_{0};
    std::atomic<int> items_processed_{0};
    std::atomic<int> batches_processed_{0};
};

// Shorthand for logger access
inline DebugLogger& debug_logger() {
    return DebugLogger::instance();
}

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_DEBUG_LOGGER_H