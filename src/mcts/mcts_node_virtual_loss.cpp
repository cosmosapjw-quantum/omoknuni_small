// src/mcts/mcts_node_virtual_loss.cpp
#include "mcts/mcts_node.h"
#include <thread>
#include <chrono>

namespace alphazero {
namespace mcts {

void MCTSNode::addVirtualLoss() {
    // Proper virtual loss implementation for thread safety
    // Virtual loss discourages other threads from selecting this node during evaluation
    virtual_loss_count_.fetch_add(1, std::memory_order_acq_rel);
}

void MCTSNode::addVirtualLoss(int amount) {
    // Add multiple virtual losses atomically
    if (amount > 0) {
        virtual_loss_count_.fetch_add(amount, std::memory_order_acq_rel);
    }
}

void MCTSNode::removeVirtualLoss() {
    // Remove virtual loss after evaluation completes
    // This re-enables the node for selection by other threads
    int current_value = virtual_loss_count_.load(std::memory_order_acquire);
    
    // Early exit if already zero to prevent unnecessary operations
    if (current_value <= 0) {
        return;
    }
    
    // Use bounded retry loop to prevent infinite loops
    int max_attempts = 100;
    int attempts = 0;
    
    // Ensure we don't go below zero
    while (current_value > 0 && attempts < max_attempts) {
        if (virtual_loss_count_.compare_exchange_weak(
                current_value, 
                current_value - 1, 
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            break;
        }
        attempts++;
        // Loop will retry with updated current_value from compare_exchange_weak
    }
}

void MCTSNode::removeVirtualLoss(int amount) {
    // Remove multiple virtual losses atomically
    if (amount <= 0) return;
    
    int current_value = virtual_loss_count_.load(std::memory_order_acquire);
    
    // Early exit if already zero to prevent unnecessary operations
    if (current_value <= 0) {
        return;
    }
    
    // Use bounded retry loop to prevent infinite loops
    int max_attempts = 100;
    int attempts = 0;
    
    while (current_value > 0 && attempts < max_attempts) {
        int new_value = std::max(0, current_value - amount);
        
        if (virtual_loss_count_.compare_exchange_weak(
                current_value, 
                new_value, 
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            break;
        }
        attempts++;
        // Loop will retry with updated current_value from compare_exchange_weak
    }
}

// Implementation moved to header file inline method
// void MCTSNode::applyVirtualLoss(int amount) {
//     addVirtualLoss(amount);
// }

int MCTSNode::getVirtualLoss() const {
    // Thread-safe read of virtual loss count
    return virtual_loss_count_.load(std::memory_order_acquire);
}

void MCTSNode::update(float value) {
    // Thread-safe update with timeout and backoff to prevent deadlock
    static std::atomic<int> debug_counter{0};
    // Fetch and increment debug counter for potential debugging - not used in production code
    debug_counter.fetch_add(1, std::memory_order_relaxed);
    
    // Increment visit count first
    visit_count_.fetch_add(1, std::memory_order_relaxed);
    
    // Update value sum with exponential backoff to prevent livelock
    float current_sum = value_sum_.load(std::memory_order_acquire);
    float new_sum;
    int attempts = 0;
    const int MAX_ATTEMPTS = 1000;
    
    do {
        new_sum = current_sum + value;
        attempts++;
        
        if (attempts > MAX_ATTEMPTS) {
            // Emergency fallback: use relaxed ordering to force progress
            value_sum_.store(value_sum_.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
            break;
        }
        
        // Progressive backoff to reduce contention
        if (attempts > 10) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(attempts));
        }
        
    } while (!value_sum_.compare_exchange_weak(
        current_sum, 
        new_sum,
        std::memory_order_acq_rel,
        std::memory_order_acquire
    ));
    
    
    // Only remove virtual loss if the node actually has virtual loss
    // This prevents infinite loops when update() is called on nodes without virtual loss
    int current_virtual_loss = virtual_loss_count_.load(std::memory_order_acquire);
    if (current_virtual_loss > 0) {
        removeVirtualLoss();
    }
}

} // namespace mcts
} // namespace alphazero