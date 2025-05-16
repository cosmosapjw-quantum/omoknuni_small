// src/mcts/mcts_node_pending_eval.cpp
#include "mcts/mcts_node.h"

namespace alphazero {
namespace mcts {

bool MCTSNode::hasPendingEvaluation() const {
    // atomic_flag doesn't have test() method until C++20
    // Use a workaround by testing if we can set it
    bool was_set = pending_evaluation_.test_and_set(std::memory_order_acquire);
    
    // If it wasn't already set, clear it immediately
    if (!was_set) {
        pending_evaluation_.clear(std::memory_order_release);
    }
    
    return was_set;
}

void MCTSNode::markEvaluationPending() {
    pending_evaluation_.test_and_set(std::memory_order_acquire);
}

void MCTSNode::clearPendingEvaluation() {
    pending_evaluation_.clear(std::memory_order_release);
}

} // namespace mcts
} // namespace alphazero