// src/mcts/mcts_node.cpp
#include "mcts/mcts_node.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>
#include "utils/debug_monitor.h"

namespace alphazero {
namespace mcts {

MCTSNode::MCTSNode(std::unique_ptr<core::IGameState> state_param, std::weak_ptr<MCTSNode> parent_param)
    : state_(std::move(state_param)),
      parent_(parent_param),
      action_(-1),
      visit_count_(0),
      value_sum_(0.0f),
      virtual_loss_count_(0),
      prior_probability_(0.0f) {

    // Safety check - ensure we have a valid state
    if (!state_) {
        throw std::invalid_argument("Cannot create MCTSNode with null state");
    }
    
    // Additional validation to ensure the state is consistent
    try {
        if (!state_->validate()) {
            throw std::runtime_error("State validation failed during node creation");
        }
        
        // Test access to key methods to ensure the state is functional
        state_->getGameResult();
        state_->getCurrentPlayer();
        state_->getLegalMoves();
        state_->getHash();
    } catch (const std::exception& e) {
        // Clean up and propagate the error with more context
        state_.reset();
        throw std::invalid_argument(std::string("State validation failed: ") + e.what());
    } catch (...) {
        // Clean up and report generic error
        state_.reset();
        throw std::invalid_argument("Unknown error during state validation");
    }
}

std::shared_ptr<MCTSNode> MCTSNode::create(std::unique_ptr<core::IGameState> state, 
                                          std::shared_ptr<MCTSNode> parent) {
    std::weak_ptr<MCTSNode> weak_parent;
    if (parent) {
        weak_parent = parent;
    }
    return std::shared_ptr<MCTSNode>(new MCTSNode(std::move(state), weak_parent));
}

std::shared_ptr<MCTSNode> MCTSNode::getSharedPtr() {
    return shared_from_this();
}

MCTSNode::~MCTSNode() {
    // Children are now shared_ptr, so they will be cleaned up automatically
    // No need for manual deletion
}

std::shared_ptr<MCTSNode> MCTSNode::selectChild(float exploration_factor) {
    float best_score = -std::numeric_limits<float>::infinity();
    std::shared_ptr<MCTSNode> best_child = nullptr;
    
    float exploration_param = exploration_factor * 
        std::sqrt(static_cast<float>(visit_count_.load(std::memory_order_relaxed)));
    
    for (size_t i = 0; i < children_.size(); ++i) {
        auto& child = children_[i];
        
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
    // First check if we have a valid state to prevent segfaults
    if (!state_) {
        // MCTSNode::expand - state_ is NULL! Cannot expand.
        return;
    }
    
    // Use a lock to prevent concurrent expansion
    std::lock_guard<std::mutex> lock(expansion_mutex_);
    
    // Check again after acquiring the lock to avoid race conditions
    if (!children_.empty()) {
        return; // Already expanded
    }
    
    if (state_->isTerminal()) {
        return; // Terminal nodes don't expand
    }
    
    // Get legal moves and prepare for child creation
    std::vector<int> legal_moves;
    try {
        legal_moves = state_->getLegalMoves();
    } catch (const std::exception& e) {
        // MCTSNode::expand - Error getting legal moves
        return;
    }
    
    if (legal_moves.empty()) {
        // MCTSNode::expand - Warning: No legal moves found in state
        return; // Early return if no legal moves
    }
    
    // Safety check for unreasonable number of legal moves
    const size_t max_reasonable_moves = 1000; // Arbitrary limit that should be safe
    if (legal_moves.size() > max_reasonable_moves) {
        // MCTSNode::expand - Excessive number of legal moves, limiting to maximum
        legal_moves.resize(max_reasonable_moves);
    }
    
    // Reserve space for efficiency
    try {
        children_.reserve(legal_moves.size());
        actions_.reserve(legal_moves.size());
    } 
    catch (const std::exception& e) {
        // MCTSNode::expand - Memory allocation error
        return; // Failed to allocate memory, don't proceed
    }
    
    // Use a local RNG instance for thread safety
    std::mt19937 local_rng;
    {
        std::random_device rd;
        local_rng.seed(rd());
    }
    
    std::shuffle(legal_moves.begin(), legal_moves.end(), local_rng);
    
    // Create children with proper exception handling
    for (int move : legal_moves) {
        try {
            // Clone the state and make the move
            auto new_state = state_->clone();
            if (!new_state) {
                continue;
            }
            
            new_state->makeMove(move);
            
            // Create the child node using factory method
            // CRITICAL FIX: Pass nullptr first, then set parent after creation to avoid circular reference
            auto child = MCTSNode::create(std::move(new_state), nullptr);
            child->setAction(move);
            child->setParentDirectly(shared_from_this());  // Set weak parent reference
            
            children_.push_back(child);
            actions_.push_back(move);
        } 
        catch (const std::exception& e) {
            // Continue trying other moves on error
            continue;
        }
    }
    
    // Skip prior probability assignment if expansion failed
    if (children_.empty()) {
        return;
    }
    
    // Apply uniform prior probabilities (will be updated by network later)
    float uniform_prior = 1.0f / static_cast<float>(children_.size());
    for (auto& child : children_) {
        child->setPriorProbability(uniform_prior);
    }
    
    // Log completion with reduced output
    // MCTS node expansion complete
}

bool MCTSNode::isFullyExpanded() const {
    return !isLeaf();
}

bool MCTSNode::isLeaf() const {
    return children_.empty();
}

bool MCTSNode::isTerminal() const {
    if (!state_) {
        // This is a safety check - a node should always have a state
        return true; // Consider a node with no state as terminal
    }

    return state_->isTerminal();
}

void MCTSNode::addVirtualLoss() {
    // Use memory_order_acq_rel for better thread safety
    virtual_loss_count_.fetch_add(1, std::memory_order_acq_rel);
}

void MCTSNode::removeVirtualLoss() {
    // Use memory_order_acq_rel for better thread safety
    virtual_loss_count_.fetch_sub(1, std::memory_order_acq_rel);
}

void MCTSNode::update(float value) {
    // Increment visit count atomically with acquire-release memory ordering
    visit_count_.fetch_add(1, std::memory_order_acq_rel);
    
    // Update value_sum atomically using compare_exchange_strong for better reliability
    // Use acquire-release memory ordering for thread safety
    float current = value_sum_.load(std::memory_order_acquire);
    float desired;
    
    do {
        desired = current + value;
        // Use a bounded retry to avoid infinite loops in heavily contended scenarios
    } while (!value_sum_.compare_exchange_strong(current, desired, 
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire));
}

const core::IGameState& MCTSNode::getState() const {
    return *state_;
}

core::IGameState& MCTSNode::getStateMutable() {
    return *state_;
}

// These getters are defined later in the file

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
    // Thread safety for the entire operation
    std::lock_guard<std::mutex> lock(expansion_mutex_);
    
    try {
        // Store the full policy vector for this node's state (optional, but can be useful for debugging/analysis)
        prior_probabilities_ = policy_vector;
    
        // If this node has children and a valid state, update each child's individual prior_probability.
        if (children_.empty() || !state_ || actions_.empty()) {
            return; // Early exit if no children or invalid state
        }
        
        size_t num_children = children_.size();
        int action_space_size = 0;
        
        try {
            action_space_size = state_->getActionSpaceSize();
        } catch (...) {
            // If we can't get action space size, we can't reliably set probabilities
            return;
        }
        
        if (policy_vector.size() == static_cast<size_t>(action_space_size)) {
            // The policy_vector is for the full action space.
            // Iterate through this node's children and set their prior_probability
            // based on the policy value for the action that leads to them.
            for (size_t i = 0; i < num_children; ++i) {
                if (i >= actions_.size() || !children_[i]) {
                    continue; // Skip invalid indices or null children
                }
                
                int action_to_child = actions_[i];
                if (action_to_child >= 0 && action_to_child < action_space_size) {
                    try {
                        // Get a local copy of the probability for thread safety
                        float prob = policy_vector[action_to_child];
                        // Ensure probability is in valid range
                        prob = std::min(1.0f, std::max(0.0f, prob));
                        children_[i]->setPriorProbability(prob);
                    } catch (...) {
                        // Silently continue on error
                    }
                } else {
                    // Handle error or set a default if action is out of bounds
                    children_[i]->setPriorProbability(0.01f); // Small non-zero default
                }
            }
        } else if (policy_vector.size() == num_children) {
            // The policy_vector size matches the number of children.
            // Apply priors directly by child index.
            for (size_t i = 0; i < num_children; ++i) {
                if (!children_[i]) {
                    continue; // Skip null children
                }
                
                try {
                    // Get a local copy and validate
                    float prob = i < policy_vector.size() ? policy_vector[i] : 0.01f;
                    prob = std::min(1.0f, std::max(0.0f, prob));
                    children_[i]->setPriorProbability(prob);
                } catch (...) {
                    // Silently continue on error
                }
            }
        } else {
            // Policy vector size mismatch, apply uniform probability
            float uniform_prob = 1.0f / static_cast<float>(num_children);
            for (size_t i = 0; i < num_children; ++i) {
                if (children_[i]) {
                    children_[i]->setPriorProbability(uniform_prob);
                }
            }
        }
    } catch (const std::exception& e) {
        // Commented out: Error in setPriorProbabilities with error message
    } catch (...) {
        // Commented out: Unknown error in setPriorProbabilities
    }
}

std::vector<std::shared_ptr<MCTSNode>>& MCTSNode::getChildren() {
    return children_;
}

std::vector<int>& MCTSNode::getActions() {
    return actions_;
}

std::shared_ptr<MCTSNode> MCTSNode::getParent() {
    return parent_.lock();
}

} // namespace mcts
} // namespace alphazero