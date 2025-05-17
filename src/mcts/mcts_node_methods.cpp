#include "mcts/mcts_node.h"

namespace alphazero {
namespace mcts {

int MCTSNode::getDepth() const {
    int depth = 0;
    auto current = parent_.lock();
    while (current) {
        depth++;
        current = current->parent_.lock();
    }
    return depth;
}

std::mutex& MCTSNode::getExpansionMutex() {
    return expansion_mutex_;
}

float MCTSNode::getTerminalValue() const {
    if (!isTerminal()) {
        return 0.0f;
    }
    
    auto result = state_->getGameResult();
    int current_player = state_->getCurrentPlayer();
    
    switch (result) {
        case core::GameResult::WIN_PLAYER1:
            return current_player == 1 ? 1.0f : -1.0f;
        case core::GameResult::WIN_PLAYER2:
            return current_player == 2 ? 1.0f : -1.0f;
        case core::GameResult::DRAW:
            return 0.0f;
        default: // ONGOING
            return 0.0f;
    }
}

std::shared_ptr<MCTSNode> MCTSNode::getMostVisitedChild() const {
    std::shared_ptr<MCTSNode> best_child = nullptr;
    int max_visits = -1;
    
    for (const auto& child : children_) {
        int visits = child->getVisitCount();
        if (visits > max_visits) {
            max_visits = visits;
            best_child = child;
        }
    }
    
    return best_child;
}

// These methods are defined in mcts_node_pending_eval.cpp

} // namespace mcts
} // namespace alphazero