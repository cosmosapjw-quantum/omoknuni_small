// src/mcts/mcts_node_virtual_loss.cpp
#include "mcts/mcts_node.h"
#include <thread>
#include <chrono>

namespace alphazero {
namespace mcts {

void MCTSNode::addVirtualLoss() {
    // OPTIMIZED: Use relaxed ordering for better performance
    // Virtual loss only needs to be eventually visible to other threads
    virtual_loss_count_.fetch_add(1, std::memory_order_relaxed);
}

void MCTSNode::addVirtualLoss(int amount) {
    // OPTIMIZED: Use relaxed ordering for better performance
    if (amount > 0) {
        virtual_loss_count_.fetch_add(amount, std::memory_order_relaxed);
    }
}

void MCTSNode::removeVirtualLoss() {
    // OPTIMIZED: Use relaxed ordering for better performance
    int current = virtual_loss_count_.fetch_sub(1, std::memory_order_relaxed);
    
    // If we went negative, correct it
    if (current <= 0) {
        virtual_loss_count_.store(0, std::memory_order_relaxed);
    }
}

void MCTSNode::removeVirtualLoss(int amount) {
    // OPTIMIZED: Use relaxed ordering for better performance
    if (amount <= 0) return;
    
    int current = virtual_loss_count_.fetch_sub(amount, std::memory_order_relaxed);
    
    // If we went negative, correct it
    if (current < amount) {
        virtual_loss_count_.store(0, std::memory_order_relaxed);
    }
}

// Implementation moved to header file inline method
// void MCTSNode::applyVirtualLoss(int amount) {
//     addVirtualLoss(amount);
// }

int MCTSNode::getVirtualLoss() const {
    // OPTIMIZED: Use relaxed ordering for better performance
    return virtual_loss_count_.load(std::memory_order_relaxed);
}

void MCTSNode::update(float value) {
    // OPTIMIZED: Simple atomic operations without retry loops
    visit_count_.fetch_add(1, std::memory_order_relaxed);
    
    // Simple compare-exchange loop without backoff
    float current_sum = value_sum_.load(std::memory_order_relaxed);
    while (!value_sum_.compare_exchange_weak(
        current_sum, 
        current_sum + value,
        std::memory_order_relaxed)) {
        // current_sum is updated by compare_exchange_weak
    }
    
    // Remove virtual loss if present
    if (virtual_loss_count_.load(std::memory_order_relaxed) > 0) {
        removeVirtualLoss();
    }
}

} // namespace mcts
} // namespace alphazero