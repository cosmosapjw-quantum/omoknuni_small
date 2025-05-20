// src/mcts/mcts_node.cpp
#include "mcts/mcts_node.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <unordered_set>
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
      rave_count_(0),  // Move before prior_probability_ to match header order
      rave_value_sum_(0.0f),
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

std::shared_ptr<MCTSNode> MCTSNode::selectChild(float exploration_factor, bool use_rave, float rave_constant) {
    static int selection_counter = 0;
    selection_counter++;
    bool debug_trace = (selection_counter <= 50 || selection_counter % 100 == 0);
    
    float best_score = -std::numeric_limits<float>::infinity();
    std::shared_ptr<MCTSNode> best_child = nullptr;
    
    if (debug_trace) {
        std::cout << "MCTSNode::selectChild - Selecting from " << children_.size() << " children, counter=" << selection_counter << std::endl;
    }
    
    float exploration_param = exploration_factor * 
        std::sqrt(static_cast<float>(visit_count_.load(std::memory_order_relaxed)));
    
    const size_t num_children = children_.size();
    
    // Count nodes with pending evaluations
    int pending_eval_count = 0;
    if (debug_trace) {
        for (const auto& child : children_) {
            if (child && (child->isBeingEvaluated() || child->hasPendingEvaluation())) {
                pending_eval_count++;
            }
        }
        std::cout << "  Total children: " << num_children << ", pending_eval: " << pending_eval_count << std::endl;
    }
    
    // CRITICAL FIX: If all children are pending evaluation and we're early in search,
    // allow selection of a pending child to break potential deadlock
    bool force_selection = false;
    if (selection_counter < 200) {
        for (const auto& child : children_) {
            if (child && !(child->isBeingEvaluated() || child->hasPendingEvaluation())) {
                // If at least one child is not being evaluated, we don't need to force
                break;
            }
            force_selection = true;
        }
    }
    
    if (force_selection && debug_trace) {
        std::cout << "⚠️ CRITICAL FIX: All children are pending evaluation, forcing selection to prevent deadlock" << std::endl;
    }
    
    // For many children, use OpenMP parallelization
    if (num_children > 32) {
        std::vector<float> scores(num_children);
        std::vector<bool> skip_child(num_children, false);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_children; ++i) {
            auto& child = children_[i];
            
            // Skip children with pending evaluations UNLESS we need to force selection
            if (!force_selection && (child->isBeingEvaluated() || child->hasPendingEvaluation())) {
                skip_child[i] = true;
                scores[i] = -std::numeric_limits<float>::infinity();
                continue;
            }
            
            // Get stats (thread-safe reads)
            int child_visits = child->visit_count_.load(std::memory_order_relaxed);
            int virtual_losses = child->virtual_loss_count_.load(std::memory_order_acquire);
            float child_value = child->value_sum_.load(std::memory_order_relaxed);
            
            // Apply virtual loss penalty
            int effective_visits = child_visits + virtual_losses;
            float effective_value = child_value - virtual_losses;
            
            // Calculate exploitation value using RAVE if enabled
            float exploitation;
            if (use_rave) {
                exploitation = child->getCombinedValue(rave_constant);
            } else {
                exploitation = effective_visits > 0 ? 
                    effective_value / effective_visits : 0.0f;
            }
            
            float exploration = child->prior_probability_ * exploration_param / 
                (1 + effective_visits);
            
            scores[i] = exploitation + exploration;
        }
        
        // Find best child sequentially
        for (size_t i = 0; i < num_children; ++i) {
            if (skip_child[i]) continue;
            
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_child = children_[i];
            }
        }
        
        // CRITICAL FIX: If no valid child found but force_selection is enabled, pick first child
        if (!best_child && force_selection && !children_.empty()) {
            std::cout << "🚨 FORCED SELECTION: All children are pending eval, choosing first available" << std::endl;
            best_child = children_[0];
            // Don't clear evaluation flag yet - allow the MCTS engine to handle this
        }
    } else {
        // For few children, use sequential processing
        for (size_t i = 0; i < num_children; ++i) {
            auto& child = children_[i];
            
            // Skip children with pending evaluations UNLESS we need to force selection
            if (!force_selection && (child->isBeingEvaluated() || child->hasPendingEvaluation())) {
                if (debug_trace) {
                    std::cout << "  Child " << i << " skipped (pending/being evaluated)" << std::endl;
                }
                continue;
            }
            
            // Get stats (thread-safe reads)
            int child_visits = child->visit_count_.load(std::memory_order_relaxed);
            int virtual_losses = child->virtual_loss_count_.load(std::memory_order_acquire);
            float child_value = child->value_sum_.load(std::memory_order_relaxed);
            
            // Apply virtual loss penalty
            int effective_visits = child_visits + virtual_losses;
            float effective_value = child_value - virtual_losses;
            
            // Calculate exploitation value using RAVE if enabled
            float exploitation;
            if (use_rave) {
                exploitation = child->getCombinedValue(rave_constant);
            } else {
                exploitation = effective_visits > 0 ? 
                    effective_value / effective_visits : 0.0f;
            }
            
            float exploration = child->prior_probability_ * exploration_param / 
                (1 + effective_visits);
            
            float score = exploitation + exploration;
            
            if (debug_trace) {
                std::cout << "  Child " << i << ": Q=" << exploitation 
                         << ", visits=" << effective_visits 
                         << ", UCB=" << score 
                         << ", pending=" << (child->hasPendingEvaluation() ? "yes" : "no")
                         << ", being_eval=" << (child->isBeingEvaluated() ? "yes" : "no")
                         << std::endl;
            }
            
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
        
        // CRITICAL FIX: If no valid child found but force_selection is enabled, pick first child
        if (!best_child && force_selection && !children_.empty()) {
            std::cout << "🚨 FORCED SELECTION: All children are pending eval, choosing first available" << std::endl;
            best_child = children_[0];
            // Don't clear evaluation flag yet - allow the MCTS engine to handle this
        }
    }
    
    if (debug_trace) {
        std::cout << "  Selected child: " << (best_child ? "found" : "nullptr") << std::endl;
    }
    
    return best_child;
}

void MCTSNode::expand(bool use_progressive_widening, float cpw, float kpw) {
    // First check if we have a valid state to prevent segfaults
    if (!state_) {
        return;
    }
    
    // Lock-free expansion check - use compare_exchange for atomicity
    bool expected = false;
    if (!is_expanded_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return; // Already expanded by another thread
    }
    
    // At this point, we have exclusive access to expand this node
    
    if (state_->isTerminal()) {
        return; // Terminal nodes don't expand
    }
    
    // Get legal moves and prepare for child creation
    std::vector<int> legal_moves;
    try {
        legal_moves = state_->getLegalMoves();
    } catch (const std::exception& e) {
        // On error, roll back the expanded flag
        is_expanded_.store(false, std::memory_order_release);
        return;
    }
    
    if (legal_moves.empty()) {
        return; // Early return if no legal moves
    }
    
    // Safety check for unreasonable number of legal moves
    const size_t max_reasonable_moves = 1000;
    if (legal_moves.size() > max_reasonable_moves) {
        // MCTSNode::expand - Excessive number of legal moves, limiting to maximum
        legal_moves.resize(max_reasonable_moves);
    }
    
    // Determine how many children to expand based on progressive widening
    size_t num_children_to_expand = legal_moves.size();
    
    // Track unexpanded moves for incremental expansion  
    if (use_progressive_widening && visit_count_ > 0) {
        // Progressive widening formula: num_children = cpw * N^kpw
        // where N is the parent's visit count
        int parent_visits = visit_count_.load();
        size_t current_children = children_.size();
        
        // Calculate total children we should have based on visit count
        size_t target_children = std::min(
            legal_moves.size(),
            static_cast<size_t>(cpw * std::pow(parent_visits, kpw))
        );
        
        // Incremental expansion: only add new children as needed
        if (current_children > 0 && is_expanded_.load(std::memory_order_acquire)) {
            // Node was previously expanded, check if we need more children
            if (target_children > current_children) {
                num_children_to_expand = target_children - current_children;
                
                // Find moves that haven't been expanded yet
                std::vector<int> unexpanded_moves;
                unexpanded_moves.reserve(legal_moves.size());
                
                // Create a set of already expanded actions for fast lookup
                std::unordered_set<int> expanded_actions(actions_.begin(), actions_.end());
                
                for (int move : legal_moves) {
                    if (expanded_actions.find(move) == expanded_actions.end()) {
                        unexpanded_moves.push_back(move);
                    }
                }
                
                // Update legal_moves to only contain unexpanded moves
                legal_moves = std::move(unexpanded_moves);
                num_children_to_expand = std::min(num_children_to_expand, legal_moves.size());
            } else {
                // Already have enough children, no need to expand more
                return;
            }
        } else {
            // First expansion: ensure we expand at least some children
            num_children_to_expand = std::max(target_children, static_cast<size_t>(1));
        }
    }
    
    // Reserve space for efficiency
    try {
        children_.reserve(num_children_to_expand);
        actions_.reserve(num_children_to_expand);
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
    
    // Shuffle legal moves to ensure random selection
    std::shuffle(legal_moves.begin(), legal_moves.end(), local_rng);
    
    // Create children with proper exception handling
    for (size_t i = 0; i < num_children_to_expand && i < legal_moves.size(); ++i) {
        int move = legal_moves[i];
        try {
            // Clone the state using memory pool
            auto new_state = utils::GameStatePoolManager::getInstance().cloneState(*state_);
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
    
    // Mark as expanded after successful expansion
    is_expanded_.store(true, std::memory_order_release);
    
    // Log completion with reduced output
    // MCTS node expansion complete
}

bool MCTSNode::isFullyExpanded() const {
    return is_expanded_.load(std::memory_order_acquire);
}

bool MCTSNode::isLeaf() const {
    return !is_expanded_.load(std::memory_order_acquire);
}

bool MCTSNode::isTerminal() const {
    if (!state_) {
        // This is a safety check - a node should always have a state
        return true; // Consider a node with no state as terminal
    }

    return state_->isTerminal();
}

int MCTSNode::getNumExpandedChildren() const {
    return children_.size();
}

void MCTSNode::addVirtualLoss() {
    // Add virtual loss with saturation to prevent overflow
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    // Cap at reasonable maximum to prevent integer overflow
    int new_value = std::min(current + 1, 1000);  
    virtual_loss_count_.store(new_value, std::memory_order_release);
}

void MCTSNode::addVirtualLoss(int amount) {
    // Add specified amount of virtual loss with saturation to prevent overflow
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    // Cap at reasonable maximum to prevent integer overflow
    int new_value = std::min(current + amount, 1000);  
    new_value = std::max(new_value, 0);  // Ensure non-negative
    virtual_loss_count_.store(new_value, std::memory_order_release);
}

void MCTSNode::applyVirtualLoss(int amount) {
    // Alias for addVirtualLoss to match header declaration
    addVirtualLoss(amount);
}

void MCTSNode::removeVirtualLoss() {
    // Remove virtual loss with floor at zero
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    int new_value = std::max(current - 1, 0);
    virtual_loss_count_.store(new_value, std::memory_order_release);
}

void MCTSNode::removeVirtualLoss(int amount) {
    // Remove specified amount of virtual loss with floor at zero
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    int new_value = std::max(current - amount, 0);
    virtual_loss_count_.store(new_value, std::memory_order_release);
}

// applyVirtualLoss method already merged with addVirtualLoss(int) above

int MCTSNode::getVirtualLoss() const {
    return virtual_loss_count_.load(std::memory_order_acquire);
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
    // Thread safety - check if already expanded
    if (!is_expanded_.load(std::memory_order_acquire)) {
        return; // Don't set priors on unexpanded nodes
    }
    
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
        // Error in setPriorProbabilities
    } catch (...) {
        // Unknown error in setPriorProbabilities
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

bool MCTSNode::tryMarkForEvaluation() {
    bool expected = false;
    // Try to set the flag from false to true atomically
    // If it's already true, this will fail and return false
    return evaluation_in_progress_.compare_exchange_strong(expected, true, 
                                                         std::memory_order_acq_rel);
}

void MCTSNode::clearEvaluationFlag() {
    evaluation_in_progress_.store(false, std::memory_order_release);
}

bool MCTSNode::isBeingEvaluated() const {
    return evaluation_in_progress_.load(std::memory_order_acquire);
}

bool MCTSNode::updateChildReference(const std::shared_ptr<MCTSNode>& old_child, const std::shared_ptr<MCTSNode>& new_child) {
    // This operation should be done atomically but is rarely used
    // For now, we rely on external synchronization when using transposition tables
    for (size_t i = 0; i < children_.size(); ++i) {
        if (children_[i] == old_child) {
            children_[i] = new_child;
            // The corresponding action in actions_[i] remains the same,
            // as new_child represents the same game state resulting from that action.
            return true;
        }
    }
    return false; // old_child not found
}

void MCTSNode::updateRAVE(float value) {
    // Atomic increment of RAVE visit count
    rave_count_.fetch_add(1, std::memory_order_relaxed);
    
    // Atomic update of RAVE value sum
    float current = rave_value_sum_.load(std::memory_order_acquire);
    float desired;
    do {
        desired = current + value;
    } while (!rave_value_sum_.compare_exchange_strong(current, desired,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire));
}

float MCTSNode::getRAVEValue() const {
    int rave_visits = rave_count_.load(std::memory_order_relaxed);
    if (rave_visits == 0) {
        return 0.0f;
    }
    return rave_value_sum_.load(std::memory_order_relaxed) / rave_visits;
}

int MCTSNode::getRAVECount() const {
    return rave_count_.load(std::memory_order_relaxed);
}

float MCTSNode::getCombinedValue(float rave_constant) const {
    int visits = visit_count_.load(std::memory_order_relaxed);
    int rave_visits = rave_count_.load(std::memory_order_relaxed);
    
    if (visits == 0 && rave_visits == 0) {
        return 0.0f;
    }
    
    // Calculate regular MCTS value
    float mcts_value = visits > 0 ? 
        value_sum_.load(std::memory_order_relaxed) / visits : 0.0f;
    
    // Calculate RAVE value
    float rave_value = rave_visits > 0 ? 
        rave_value_sum_.load(std::memory_order_relaxed) / rave_visits : 0.0f;
    
    // Calculate weight for RAVE (decreases as visit count increases)
    float beta = std::sqrt(rave_constant / (3 * visits + rave_constant));
    
    // Return weighted combination
    return (1.0f - beta) * mcts_value + beta * rave_value;
}

// Methods required by mcts_taskflow_engine - All these methods are defined in mcts_node_methods.cpp now

} // namespace mcts
} // namespace alphazero
