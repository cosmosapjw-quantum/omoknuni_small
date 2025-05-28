// mcts_node_fast.cpp
// Fast methods for ultra-low latency MCTS operations

#include "mcts/mcts_node.h"
#include <algorithm>
#include <cmath>

namespace alphazero {
namespace mcts {

// Fast child selection with minimal overhead
std::shared_ptr<MCTSNode> MCTSNode::selectChildFast(float exploration_constant) const {
    if (children_.empty()) {
        return nullptr;
    }
    
    // Fast path for single child
    if (children_.size() == 1) {
        return children_[0];
    }
    
    // Calculate sqrt(parent_visits) once
    const float parent_visits_sqrt = std::sqrt(static_cast<float>(visit_count_.load(std::memory_order_relaxed)));
    
    std::shared_ptr<MCTSNode> best_child = nullptr;
    float best_value = -std::numeric_limits<float>::infinity();
    
    // Unrolled loop for better performance on small child sets
    if (children_.size() <= 8) {
        // Manual unrolling for common case of few children
        for (const auto& child : children_) {
            if (!child) continue;
            
            int child_visits = child->visit_count_.load(std::memory_order_relaxed);
            if (child_visits == 0) {
                // Unexplored child has highest priority
                return child;
            }
            
            // Fast UCB calculation without virtual loss check
            float child_value = child->value_sum_.load(std::memory_order_relaxed) / child_visits;
            float exploration_term = exploration_constant * child->prior_probability_ * parent_visits_sqrt / (1 + child_visits);
            float ucb_value = child_value + exploration_term;
            
            if (ucb_value > best_value) {
                best_value = ucb_value;
                best_child = child;
            }
        }
    } else {
        // Regular loop for larger child sets
        for (const auto& child : children_) {
            if (!child) continue;
            
            int child_visits = child->visit_count_.load(std::memory_order_relaxed);
            if (child_visits == 0) {
                return child;
            }
            
            float child_value = child->value_sum_.load(std::memory_order_relaxed) / child_visits;
            float exploration_term = exploration_constant * child->prior_probability_ * parent_visits_sqrt / (1 + child_visits);
            float ucb_value = child_value + exploration_term;
            
            if (ucb_value > best_value) {
                best_value = ucb_value;
                best_child = child;
            }
        }
    }
    
    return best_child;
}

// Fast expansion without progressive widening
void MCTSNode::expandFast() {
    // Quick check if already expanded
    if (is_expanded_.load(std::memory_order_acquire)) {
        return;
    }
    
    // Try to set expanded flag atomically
    bool expected = false;
    if (!is_expanded_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        // Another thread already expanded
        return;
    }
    
    // Get legal moves
    auto legal_moves = state_->getAllLegalMoves();
    if (legal_moves.empty()) {
        return;
    }
    
    // Pre-allocate children
    children_.reserve(legal_moves.size());
    actions_.reserve(legal_moves.size());
    
    // Create children for all legal moves
    for (int move : legal_moves) {
        auto child_state = state_->clone();
        child_state->makeMove(move);
        
        auto child = MCTSNode::create(std::move(child_state), shared_from_this());
        child->setAction(move);
        
        children_.push_back(child);
        actions_.push_back(move);
    }
}

// Lock-free fast update
void MCTSNode::updateFast(float value) {
    // Update visit count and value sum atomically
    visit_count_.fetch_add(1, std::memory_order_relaxed);
    
    // For value sum, we need to use compare-exchange for floating point addition
    float current_sum = value_sum_.load(std::memory_order_relaxed);
    float new_sum;
    do {
        new_sum = current_sum + value;
    } while (!value_sum_.compare_exchange_weak(current_sum, new_sum, std::memory_order_relaxed));
}

} // namespace mcts
} // namespace alphazero