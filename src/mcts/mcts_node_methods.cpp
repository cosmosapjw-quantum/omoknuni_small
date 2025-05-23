#include "mcts/mcts_node.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

namespace alphazero {
namespace mcts {

// Implementation of UCB selection for enhanced optimized search
std::shared_ptr<MCTSNode> MCTSNode::selectBestChildUCB(float exploration_constant, std::mt19937& rng) {
    if (children_.empty()) {
        return nullptr;
    }
    
    // Find child with best UCB value
    float best_ucb = -std::numeric_limits<float>::max();
    std::vector<std::shared_ptr<MCTSNode>> best_children;
    
    for (const auto& child : children_) {
        if (!child) continue;
        
        // Calculate UCB score with virtual loss consideration
        float exploit = child->getValue();
        float virtual_loss_factor = child->getVirtualLoss() > 0 ? 0.8f : 1.0f;
        
        // Parent visit count minus virtual losses for more accurate exploration term
        int parent_visits = visit_count_.load(std::memory_order_relaxed);
        int child_visits = child->getVisitCount();
        
        if (child_visits == 0) {
            // Prioritize unexplored nodes
            best_children.push_back(child);
            continue;
        }
        
        // UCB1 formula: exploitation + exploration with virtual loss discount
        float explore = exploration_constant * std::sqrt(std::log(parent_visits) / child_visits);
        float ucb = exploit + virtual_loss_factor * explore;
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_children.clear();
            best_children.push_back(child);
        } else if (ucb == best_ucb) {
            best_children.push_back(child);
        }
    }
    
    // Randomly select among best children for exploration
    if (!best_children.empty()) {
        std::uniform_int_distribution<size_t> dist(0, best_children.size() - 1);
        return best_children[dist(rng)];
    }
    
    return nullptr;
}

// Implementation of recursive update for backpropagation
void MCTSNode::updateRecursive(float value) {
    // Update this node
    update(value);
    
    // Recursively update parent with negated value (opponent's perspective)
    if (auto parent = parent_.lock()) {
        parent->updateRecursive(-value);
    }
}

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
    // Check for NULL state before accessing
    if (!state_) {
        return 0.0f; // Return neutral value for invalid nodes
    }
    
    
    try {
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
    } catch (const std::exception& e) {
        return 0.0f;
    } catch (...) {
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