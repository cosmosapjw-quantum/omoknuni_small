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

MCTSNode::MCTSNode(std::unique_ptr<core::IGameState> state_param, MCTSNode* parent_param)
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
}

MCTSNode::~MCTSNode() {
    // Clean destructor with no debug prints

    // Delete children
    for (auto child : children_) {
        delete child;
    }

    // Clean up resources
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
    // First check if we have a valid state to prevent segfaults
    if (!state_) {
        std::cerr << "[MCTSNode::expand] state_ is NULL! Cannot expand." << std::endl;
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
    std::vector<int> legal_moves = state_->getLegalMoves();
    
    if (legal_moves.empty()) {
        std::cerr << "[MCTSNode::expand] Warning: No legal moves found in state." << std::endl;
        return; // Early return if no legal moves
    }
    
    // Reserve space for efficiency
    try {
        children_.reserve(legal_moves.size());
        actions_.reserve(legal_moves.size());
    } 
    catch (const std::exception& e) {
        std::cerr << "[MCTSNode::expand] Memory allocation error: " << e.what() << std::endl;
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
    bool had_expansion_error = false;
    for (int move : legal_moves) {
        try {
            // Clone the state and make the move
            auto new_state = state_->clone();
            if (!new_state) {
                std::cerr << "[MCTSNode::expand] State clone returned nullptr!" << std::endl;
                had_expansion_error = true;
                continue;
            }
            
            new_state->makeMove(move);
            
            // Create the child node
            MCTSNode* child = new MCTSNode(std::move(new_state), this);
            child->setAction(move);
            
            children_.push_back(child);
            actions_.push_back(move);
        } 
        catch (const std::bad_alloc& e) {
            std::cerr << "[MCTSNode::expand] Memory allocation failed: " << e.what() << std::endl;
            had_expansion_error = true;
            break; // Stop expanding on memory allocation failure
        } 
        catch (const std::exception& e) {
            std::cerr << "[MCTSNode::expand] Exception during child creation: " << e.what() << std::endl;
            had_expansion_error = true;
            continue; // Try other moves
        }
    }
    
    // Clean up on partial failure
    if (had_expansion_error && !children_.empty()) {
        for (auto* child : children_) {
            delete child;
        }
        children_.clear();
        actions_.clear();
        return;
    }
    
    // Skip prior probability assignment if expansion failed
    if (children_.empty()) {
        return;
    }
    
    // Apply prior probabilities in a safe manner
    if (!prior_probabilities_.empty()) {
        try {
            // Check if the priors vector matches the full action space
            if (prior_probabilities_.size() == state_->getActionSpaceSize()) {
                // Use action-specific priors
                for (size_t i = 0; i < children_.size(); ++i) {
                    int action = actions_[i];
                    if (action >= 0 && action < static_cast<int>(prior_probabilities_.size())) {
                        children_[i]->setPriorProbability(prior_probabilities_[action]);
                    }
                }
            } 
            else if (prior_probabilities_.size() == children_.size()) {
                // Direct index matching if sizes are equal
                for (size_t i = 0; i < children_.size(); ++i) {
                    children_[i]->setPriorProbability(prior_probabilities_[i]);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[MCTSNode::expand] Error applying prior probabilities: " << e.what() << std::endl;
            // Continue with default priors on error
        }
    }
    
    // Set uniform prior probabilities if not provided or on error
    if (children_.size() > 0) {
        float uniform_prior = 1.0f / static_cast<float>(children_.size());
        for (auto child : children_) {
            if (child->getPriorProbability() <= 0.0f) {
                child->setPriorProbability(uniform_prior);
            }
        }
    }
    
    // Log completion with reduced output
    std::cout << "MCTS node expansion complete: " << children_.size() << " children created" << std::endl;
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