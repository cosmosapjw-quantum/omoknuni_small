// src/mcts/mcts_node_pending_eval.cpp
#include "mcts/mcts_node.h"

namespace alphazero {
namespace mcts {

bool MCTSNode::hasPendingEvaluation() const {
    // Use proper atomic load operation for thread-safe reading
    // This is a simple, race-condition-free way to check the flag
    return pending_evaluation_.load(std::memory_order_acquire);
}

void MCTSNode::markEvaluationPending() {
    // Use atomic store operation to set the flag
    pending_evaluation_.store(true, std::memory_order_release);
}

void MCTSNode::clearPendingEvaluation() {
    // Use atomic store operation to clear the flag
    pending_evaluation_.store(false, std::memory_order_release);
}

bool MCTSNode::tryMarkForEvaluation() {
    // Use compare_exchange_weak for lock-free marking
    // This atomically checks if the flag is false and sets it to true
    // Returns true if we successfully marked it (it was false before)
    bool expected = false;
    return pending_evaluation_.compare_exchange_weak(
        expected, 
        true, 
        std::memory_order_acq_rel,
        std::memory_order_acquire
    );
}

} // namespace mcts
} // namespace alphazero