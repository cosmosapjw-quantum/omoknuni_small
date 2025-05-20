#ifndef ALPHAZERO_PRIORITIZED_CONCURRENT_QUEUE_H
#define ALPHAZERO_PRIORITIZED_CONCURRENT_QUEUE_H

#include <moodycamel/concurrentqueue.h>
#include <vector>
#include <chrono>
#include <atomic>
#include <optional>

namespace alphazero {
namespace mcts {

/**
 * @brief Priority levels for queue items
 */
enum class QueuePriority {
    High = 0,  // Highest priority, processed first
    Normal = 1,
    Low = 2    // Lowest priority, processed last
};

/**
 * @brief A priority wrapper for items stored in the queue
 * @tparam T The type of item to store
 */
template<typename T>
struct PrioritizedItem {
    T item;
    QueuePriority priority;
    std::chrono::steady_clock::time_point created_time;
    
    // Create a prioritized item with the given priority
    static PrioritizedItem create(T&& item_value, QueuePriority prio) {
        return {
            std::move(item_value),
            prio,
            std::chrono::steady_clock::now()
        };
    }
};

/**
 * @brief A lock-free concurrent queue with priority support
 * 
 * This class implements a priority queue using separate concurrent queues
 * for each priority level. Dequeuing always checks higher priority queues first.
 * 
 * @tparam T The type of items to store in the queue
 */
template<typename T>
class PrioritizedConcurrentQueue {
public:
    /**
     * @brief Construct a new prioritized concurrent queue
     */
    PrioritizedConcurrentQueue() = default;
    
    /**
     * @brief Enqueue an item with the specified priority
     * 
     * @param item The item to enqueue
     * @param priority The priority level for this item
     * @return true if enqueued successfully, false otherwise
     */
    bool enqueue(T&& item, QueuePriority priority = QueuePriority::Normal) {
        auto prioritized = PrioritizedItem<T>::create(std::move(item), priority);
        return queues_[static_cast<int>(priority)].enqueue(std::move(prioritized));
    }
    
    /**
     * @brief Try to dequeue an item, checking high priority queues first
     * 
     * @param item Output parameter that will receive the dequeued item
     * @return true if an item was dequeued, false if all queues were empty
     */
    bool try_dequeue(T& item) {
        PrioritizedItem<T> prioritized;
        
        // Try to dequeue from high priority queue first, then normal, then low
        for (int priority = static_cast<int>(QueuePriority::High); 
             priority <= static_cast<int>(QueuePriority::Low); 
             ++priority) {
            
            if (queues_[priority].try_dequeue(prioritized)) {
                item = std::move(prioritized.item);
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief Try to dequeue multiple items at once, prioritizing high priority items
     * 
     * @param items Vector to store dequeued items
     * @param max_items Maximum number of items to dequeue
     * @return int Number of items actually dequeued
     */
    int try_dequeue_bulk(std::vector<T>& items, size_t max_items) {
        // Make sure we have capacity
        items.reserve(items.size() + max_items);
        
        size_t total_dequeued = 0;
        
        // Try to dequeue from each priority level in order
        for (int priority = static_cast<int>(QueuePriority::High); 
             priority <= static_cast<int>(QueuePriority::Low) && total_dequeued < max_items; 
             ++priority) {
            
            std::vector<PrioritizedItem<T>> prioritized_items;
            prioritized_items.resize(max_items - total_dequeued);
            
            // Try to dequeue up to remaining items
            size_t dequeued = queues_[priority].try_dequeue_bulk(
                prioritized_items.data(), max_items - total_dequeued);
                
            // Resize to actual dequeued count
            prioritized_items.resize(dequeued);
            
            // Add to output vector
            for (auto& prioritized : prioritized_items) {
                items.push_back(std::move(prioritized.item));
            }
            
            total_dequeued += dequeued;
        }
        
        return total_dequeued;
    }
    
    /**
     * @brief Get an estimate of the total queue size across all priorities
     * 
     * @return size_t Approximate size of all queues combined
     */
    size_t size_approx() const {
        size_t total = 0;
        for (int i = 0; i < 3; ++i) {
            total += queues_[i].size_approx();
        }
        return total;
    }
    
    /**
     * @brief Check if all queues are empty
     * 
     * @return true if all queues are empty
     * @return false if any queue has items
     */
    bool empty() const {
        for (int i = 0; i < 3; ++i) {
            if (queues_[i].size_approx() > 0) {
                return false;
            }
        }
        return true;
    }
    
private:
    // One queue per priority level - high, normal, low
    moodycamel::ConcurrentQueue<PrioritizedItem<T>> queues_[3];
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_PRIORITIZED_CONCURRENT_QUEUE_H