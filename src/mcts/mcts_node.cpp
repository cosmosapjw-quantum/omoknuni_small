// src/mcts/mcts_node.cpp
#include "mcts/mcts_node.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

namespace alphazero {
namespace mcts {

MCTSNode::MCTSNode(std::unique_ptr<core::IGameState> state, MCTSNode* parent)
    : state_(std::move(state)), 
      parent_(parent), 
      action_(-1), 
      visit_count_(0), 
      value_sum_(0.0f), 
      virtual_loss_count_(0),
      prior_probability_(0.0f) {
}

MCTSNode::~MCTSNode() {
    for (auto child : children_) {
        delete child;
    }
}

MCTSNode* MCTSNode::selectChild(float exploration_factor) {
    float best_score = -std::numeric_limits<float>::infinity();
    MCTSNode* best_child = nullptr;
    
    float exploration_param = exploration_factor * 
        std::sqrt(static_cast<float>(visit_count_.load(std::memory_order_relaxed)));
    
    for (size_t i = 0; i < children_.size(); ++i) {
        MCTSNode* child = children_[i];
        
        // Get stats (thread-safe reads)
        int child_visits = child->visit_count_.load(std::memory_order_relaxed);
        int virtual_losses = child->virtual_loss_count_.load(std::memory_order_relaxed);
        float child_value = child->value_sum_.load(std::memory_order_relaxed);
        
        // Apply virtual loss penalty
        int effective_visits = child_visits + virtual_losses;
        float effective_value = child_value - virtual_losses;
        
        // PUCT formula (AlphaZero-style)
        float exploitation = effective_visits > 0 ? 
            effective_value / effective_visits : 0.0f;
        float exploration = child->prior_probability_ * exploration_param / 
            (1 + effective_visits);
        
        float score = exploitation + exploration;
        
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    
    return best_child;
}

void MCTSNode::expand() {
    std::lock_guard<std::mutex> lock(expansion_mutex_);
    
    if (!children_.empty()) {
        return; // Already expanded
    }
    
    if (isTerminal()) {
        return; // Terminal states cannot be expanded
    }
    
    // Get legal moves
    std::vector<int> legal_moves = state_->getLegalMoves();
    
    // Shuffle legal moves to break move order bias
    static thread_local std::mt19937 rng(std::random_device{}());
    std::shuffle(legal_moves.begin(), legal_moves.end(), rng);
    
    // Create a child for each legal move
    for (int move : legal_moves) {
        auto new_state = state_->clone();
        new_state->makeMove(move);
        
        auto child = new MCTSNode(std::move(new_state), this);
        child->setAction(move);
        
        children_.push_back(child);
        actions_.push_back(move);
    }
    
    // Set prior probabilities based on neural network output
    if (!prior_probabilities_.empty()) {
        // Check if the priors vector matches the full action space
        if (prior_probabilities_.size() == state_->getActionSpaceSize()) {
            // Use action-specific priors
            for (size_t i = 0; i < children_.size(); ++i) {
                int action = actions_[i];
                if (action >= 0 && action < static_cast<int>(prior_probabilities_.size())) {
                    children_[i]->setPriorProbability(prior_probabilities_[action]);
                }
            }
        } else if (prior_probabilities_.size() == children_.size()) {
            // Direct index matching if sizes are equal
            for (size_t i = 0; i < children_.size(); ++i) {
                children_[i]->setPriorProbability(prior_probabilities_[i]);
            }
        }
    } else if (!children_.empty()) {
        // Set uniform prior probabilities if not provided
        float uniform_prior = 1.0f / static_cast<float>(children_.size());
        for (auto child : children_) {
            child->setPriorProbability(uniform_prior);
        }
    }
}

bool MCTSNode::isFullyExpanded() const {
    return !isLeaf();
}

bool MCTSNode::isLeaf() const {
    return children_.empty();
}

bool MCTSNode::isTerminal() const {
    return state_->isTerminal();
}

void MCTSNode::addVirtualLoss() {
    virtual_loss_count_.fetch_add(1, std::memory_order_relaxed);
}

void MCTSNode::removeVirtualLoss() {
    virtual_loss_count_.fetch_sub(1, std::memory_order_relaxed);
}

void MCTSNode::update(float value) {
    visit_count_.fetch_add(1, std::memory_order_relaxed);
    
    float current = value_sum_.load(std::memory_order_relaxed);
    float desired;
    do {
        desired = current + value;
    } while (!value_sum_.compare_exchange_weak(current, desired, std::memory_order_relaxed));
}

const core::IGameState& MCTSNode::getState() const {
    return *state_;
}

core::IGameState& MCTSNode::getStateMutable() {
    return *state_;
}

std::vector<MCTSNode*>& MCTSNode::getChildren() {
    return children_;
}

std::vector<int>& MCTSNode::getActions() {
    return actions_;
}

MCTSNode* MCTSNode::getParent() {
    return parent_;
}

float MCTSNode::getValue() const {
    int visits = visit_count_.load(std::memory_order_relaxed);
    if (visits == 0) {
        return 0.0f;
    }
    return value_sum_.load(std::memory_order_relaxed) / visits;
}

int MCTSNode::getVisitCount() const {
    return visit_count_.load(std::memory_order_relaxed);
}

int MCTSNode::getAction() const {
    return action_;
}

float MCTSNode::getPriorProbability() const {
    return prior_probability_;
}

void MCTSNode::setAction(int action) {
    action_ = action;
}

void MCTSNode::setPriorProbability(float prior) {
    prior_probability_ = prior;
}

void MCTSNode::setPriorProbabilities(const std::vector<float>& policy_vector) {
    // Store the full policy vector for this node's state (optional, but can be useful for debugging/analysis)
    prior_probabilities_ = policy_vector;

    // If this node has children and a valid state, update each child's individual prior_probability.
    if (!children_.empty() && state_ && !actions_.empty()) {
        size_t num_children = children_.size();
        int action_space_size = state_->getActionSpaceSize();

        if (policy_vector.size() == static_cast<size_t>(action_space_size)) {
            // The policy_vector is for the full action space.
            // Iterate through this node's children and set their prior_probability
            // based on the policy value for the action that leads to them.
            for (size_t i = 0; i < num_children; ++i) {
                if (i < actions_.size()) { // Ensure actions_ is in sync
                    int action_to_child = actions_[i];
                    if (action_to_child >= 0 && action_to_child < action_space_size) {
                        children_[i]->setPriorProbability(policy_vector[action_to_child]);
                    } else {
                        // Handle error or set a default if action is out of bounds for policy_vector
                        // For now, let's assume valid actions from getLegalMoves
                        children_[i]->setPriorProbability(0.0f); // Or some other default
                    }
                }
            }
        } else if (policy_vector.size() == num_children) {
            // The policy_vector size matches the number of children (e.g., from Dirichlet noise or non-NN source).
            // Apply priors directly by child index.
            for (size_t i = 0; i < num_children; ++i) {
                children_[i]->setPriorProbability(policy_vector[i]);
            }
        } else {
            // Policy vector size mismatch, cannot reliably apply.
            // Children will retain priors set during expand (e.g. uniform), or previously set.
            // Optionally, log a warning here.
        }
    }
}

} // namespace mcts
} // namespace alphazero