#ifndef ALPHAZERO_MCTS_CONCURRENT_PIPELINE_BUFFER_H
#define ALPHAZERO_MCTS_CONCURRENT_PIPELINE_BUFFER_H

#include <moodycamel/concurrentqueue.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <chrono>

namespace alphazero {
namespace mcts {

/**
 * @brief A double-buffered concurrent pipeline implementation using lock-free queues.
 * 
 * This class implements a double-buffering scheme for pipeline parallelism using
 * moodycamel::ConcurrentQueue for lock-free operation. It maintains two buffers:
 * one for collection and one for processing, and allows them to be swapped when
 * the collection buffer reaches a target size.
 */
template<typename T>
class ConcurrentPipelineBuffer {
public:
    /**
     * @brief Constructor
     * @param target_batch_size Target size at which to swap buffers
     */
    explicit ConcurrentPipelineBuffer(size_t target_batch_size = 64) 
        : target_batch_size_(target_batch_size),
          shutdown_flag_(false),
          buffer_ready_(false) {
    }

    /**
     * @brief Adds an item to the collection buffer
     * @param item Item to add
     * @return true if added successfully, false otherwise
     */
    bool add(T&& item) {
        // Always add to collection buffer
        bool success = collection_buffer_.enqueue(std::move(item));
        
        // Check if we should swap buffers
        if (success && collection_buffer_.size_approx() >= target_batch_size_) {
            swapBuffers();
        }
        
        return success;
    }

    /**
     * @brief Gets all items from the processing buffer
     * @param items Vector to store the items
     * @param timeout Maximum time to wait for items
     * @return true if items were retrieved, false if timed out or empty
     */
    bool getItems(std::vector<T>& items, std::chrono::milliseconds timeout = std::chrono::milliseconds(50)) {
        // First check if we need to wait for a buffer swap
        bool have_items = waitForProcessingBuffer(timeout);
        if (!have_items) {
            return false;
        }
        
        // Dequeue all items from processing buffer
        T item;
        while (processing_buffer_.try_dequeue(item)) {
            items.push_back(std::move(item));
        }
        
        // Reset buffer ready flag if we emptied the processing buffer
        if (processing_buffer_.size_approx() == 0) {
            buffer_ready_.store(false, std::memory_order_release);
        }
        
        return !items.empty();
    }

    /**
     * @brief Bulk dequeue items from processing buffer
     * @param items Vector to store the items
     * @param max_items Maximum number of items to dequeue
     * @return Number of items dequeued
     */
    size_t bulkDequeue(std::vector<T>& items, size_t max_items) {
        // Resize the vector to hold the maximum number of items
        size_t original_size = items.size();
        items.resize(original_size + max_items);
        
        // Try to dequeue items in bulk
        size_t num_dequeued = processing_buffer_.try_dequeue_bulk(
            items.data() + original_size, max_items);
            
        // Resize vector to actual number of items
        items.resize(original_size + num_dequeued);
        
        // Reset buffer ready flag if we emptied the processing buffer
        if (processing_buffer_.size_approx() == 0) {
            buffer_ready_.store(false, std::memory_order_release);
        }
        
        return num_dequeued;
    }

    /**
     * @brief Swaps collection and processing buffers if collection buffer is non-empty
     * @return true if buffers were swapped, false otherwise
     */
    bool swapBuffers() {
        // Only swap if the collection buffer has items and processing buffer is empty
        if (collection_buffer_.size_approx() == 0 || 
            processing_buffer_.size_approx() > 0) {
            return false;
        }
        
        // Swap buffers using a temporary queue
        moodycamel::ConcurrentQueue<T> temp;
        std::swap(temp, processing_buffer_);
        std::swap(temp, collection_buffer_);
        
        // Set buffer ready flag and notify waiters
        buffer_ready_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
        
        return true;
    }

    /**
     * @brief Waits for the processing buffer to be ready
     * @param timeout Maximum time to wait
     * @return true if buffer is ready, false if timed out
     */
    bool waitForProcessingBuffer(std::chrono::milliseconds timeout) {
        // Check if buffer is already ready or shutdown requested
        if (buffer_ready_.load(std::memory_order_acquire) || 
            shutdown_flag_.load(std::memory_order_acquire)) {
            return buffer_ready_.load(std::memory_order_acquire);
        }
        
        // Wait with timeout for buffer to become ready
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] {
            return buffer_ready_.load(std::memory_order_acquire) || 
                   shutdown_flag_.load(std::memory_order_acquire);
        });
    }

    /**
     * @brief Forces a buffer swap regardless of buffer sizes
     * @return true if swap occurred, false otherwise
     */
    bool forceBufferSwap() {
        // Don't swap if collection buffer is empty or already shutting down
        if (collection_buffer_.size_approx() == 0 ||
            shutdown_flag_.load(std::memory_order_acquire)) {
            return false;
        }
        
        return swapBuffers();
    }

    /**
     * @brief Sets shutdown flag to terminate waiting threads
     */
    void shutdown() {
        shutdown_flag_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

    /**
     * @brief Checks if the pipeline is empty
     * @return true if both buffers are empty
     */
    bool empty() const {
        return collection_buffer_.size_approx() == 0 && 
               processing_buffer_.size_approx() == 0;
    }

    /**
     * @brief Gets approximate total size of all items in both buffers
     * @return Approximate total size
     */
    size_t size_approx() const {
        return collection_buffer_.size_approx() + processing_buffer_.size_approx();
    }

    /**
     * @brief Sets the target batch size for automatic buffer swapping
     * @param size New target batch size
     */
    void setTargetBatchSize(size_t size) {
        target_batch_size_ = size;
    }
    
    /**
     * @brief Resets the pipeline buffer, clearing all contents
     */
    void reset() {
        // Clear both buffers
        T dummy;
        while (collection_buffer_.try_dequeue(dummy)) {}
        while (processing_buffer_.try_dequeue(dummy)) {}
        
        // Reset buffer ready flag
        buffer_ready_.store(false, std::memory_order_release);
        
        // Notify any waiting threads
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

private:
    // Queue for items being collected
    moodycamel::ConcurrentQueue<T> collection_buffer_;
    
    // Queue for items being processed
    moodycamel::ConcurrentQueue<T> processing_buffer_;
    
    // Target size at which to swap buffers
    size_t target_batch_size_;
    
    // Flag to indicate when shutdown is requested
    std::atomic<bool> shutdown_flag_;
    
    // Flag to indicate when processing buffer is ready
    std::atomic<bool> buffer_ready_;
    
    // Synchronization for waiting threads
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_CONCURRENT_PIPELINE_BUFFER_H