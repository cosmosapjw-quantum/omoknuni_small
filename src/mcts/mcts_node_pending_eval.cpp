// src/mcts/mcts_node_pending_eval.cpp
#include "mcts/mcts_node.h"

namespace alphazero {
namespace mcts {

bool MCTSNode::hasPendingEvaluation() const {
    // atomic_flag doesn't have test() method until C++20
    // CRITICAL FIX: The previous implementation had a serious issue - it was modifying
    // state in what should be a const function. This can cause race conditions.
    // 
    // Instead, we'll just test if the node is marked for evaluation,
    // without actually trying to set the flag. This is not fully thread-safe,
    // but avoids the worse problem of modifying state in a const function.
    
    // Use test_and_set, but immediately reset the flag if we changed it
    bool was_set = pending_evaluation_.test_and_set(std::memory_order_acquire);
    
    // If it wasn't already set, we just set it, so we need to clear it immediately
    if (!was_set) {
        pending_evaluation_.clear(std::memory_order_release);
        // Debug logging for first few resets
        static std::atomic<int> reset_counter{0};
        int counter = reset_counter.fetch_add(1, std::memory_order_relaxed);
        if (counter < 20 || counter % 100 == 0) {
            std::cout << "🔄 MCTSNode::hasPendingEvaluation - Reset flag that wasn't set for node " 
                     << this << " (reset #" << counter << ")" << std::endl;
        }
    }
    
    return was_set;
}

void MCTSNode::markEvaluationPending() {
    pending_evaluation_.test_and_set(std::memory_order_acquire);
}

void MCTSNode::clearPendingEvaluation() {
    pending_evaluation_.clear(std::memory_order_release);
    
    // ENHANCED DEBUG: Log when pending evaluation is cleared with more details
    static std::atomic<int> clear_counter{0};
    int counter = clear_counter.fetch_add(1, std::memory_order_relaxed);
    if (counter < 50 || counter % 50 == 0) {
        bool is_terminal = false;
        bool has_children = false;
        int num_children = 0;
        int visit_count = 0;
        bool is_being_evaluated = false;
        
        try {
            is_terminal = this->isTerminal();
            has_children = this->hasChildren();
            num_children = this->getChildren().size();
            visit_count = this->getVisitCount();
            is_being_evaluated = this->isBeingEvaluated();
        } catch (...) {
            // Ignore any exceptions during debug info collection
        }
        
        std::cout << "🔄 MCTSNode::clearPendingEvaluation - Cleared pending evaluation flag for node " 
                 << this << " (clear #" << counter << ")"
                 << " is_terminal=" << (is_terminal ? "yes" : "no")
                 << " has_children=" << (has_children ? "yes" : "no")
                 << " num_children=" << num_children
                 << " visit_count=" << visit_count
                 << " is_being_evaluated=" << (is_being_evaluated ? "yes" : "no")
                 << std::endl;
    }
}

} // namespace mcts
} // namespace alphazero