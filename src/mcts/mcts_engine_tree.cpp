#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include <iostream>
#include <algorithm>

namespace alphazero {
namespace mcts {

// Helper method to handle transposition table lookups
bool MCTSEngine::handleTranspositionMatch(std::shared_ptr<MCTSNode>& selected_child, std::shared_ptr<MCTSNode>& parent) {
    if (!use_transposition_table_ || !transposition_table_ || !selected_child || selected_child->isTerminal()) {
        return false;
    }
    
    try {
        uint64_t hash = selected_child->getState().getHash();
        std::shared_ptr<MCTSNode> transposition_entry = transposition_table_->get(hash);

        if (!transposition_entry || transposition_entry == selected_child) {
            return false;
        }
        
        // Validate the transposition entry
        bool valid_transposition = false;
        try {
            int visits = transposition_entry->getVisitCount();
            if (visits >= 0 && visits < 100000) { // Basic sanity check
                const core::IGameState& trans_state = transposition_entry->getState();
                valid_transposition = safeGameStateValidation(trans_state) && trans_state.getHash() == hash;
            }
        } catch (...) {
            valid_transposition = false;
        }
        
        if (!valid_transposition) {
            return false;
        }
        
        // Update parent's reference from selected_child to transposition_entry
        if (!parent->updateChildReference(selected_child, transposition_entry)) {
            return false;
        }
        
        // Fix virtual loss handling
        selected_child->removeVirtualLoss(settings_.virtual_loss);
        transposition_entry->addVirtualLoss(settings_.virtual_loss);
        
        // Clear evaluation flag from orphaned node to prevent memory leaks
        if (selected_child->tryMarkForEvaluation()) {
            selected_child->clearEvaluationFlag();
        }
        
        // Update the reference to use the transposition table entry
        selected_child = transposition_entry;
        return true;
    } catch (...) {
        // If any exception occurs during transposition table lookup or use,
        // continue with the current selected_child.
        return false;
    }
}

// Core tree traversal implementation
std::shared_ptr<MCTSNode> MCTSEngine::traverseTreeForLeaf(std::shared_ptr<MCTSNode> node, 
                                                        std::vector<std::shared_ptr<MCTSNode>>& path) {
    if (!node) {
        return nullptr;
    }
    
    path.clear();
    path.push_back(node);
    
    // Note - this method assumes that the first node (root) has already been expanded
    // and is not a leaf or terminal node
    
    std::shared_ptr<MCTSNode> current = node;
    
    // Add counters to track traversal progress and detect potential issues
    static int traversal_count = 0;
    traversal_count++;
    int depth = 0;
    
    while (current && !current->isLeaf() && !current->isTerminal()) {
        depth++;
        
        // Allow traversal through nodes with pending evaluation 
        // during early iterations to break potential deadlocks
        bool allow_pending_traversal = (traversal_count < 100) && (depth < 5);
        
        // Check if current node has pending evaluation
        if (current->hasPendingEvaluation() && !allow_pending_traversal) {
            // Remove virtual loss from path since we're not going deeper
            for (auto& node_in_path : path) {
                node_in_path->removeVirtualLoss(settings_.virtual_loss);
            }
            return nullptr; // Return null to indicate pending
        }
        
        std::shared_ptr<MCTSNode> parent_for_selection = current;
        parent_for_selection->addVirtualLoss(settings_.virtual_loss);

        std::shared_ptr<MCTSNode> selected_child = parent_for_selection->selectChild(
            settings_.exploration_constant, settings_.use_rave, settings_.rave_constant);
        
        if (!selected_child) {
            // If no child is selected, remove virtual loss from parent and return parent as leaf
            parent_for_selection->removeVirtualLoss(settings_.virtual_loss);
            break;  
        }

        // Tentatively, the traversal will proceed with selected_child
        selected_child->addVirtualLoss(settings_.virtual_loss);
        
        // Handle transposition table lookup
        if (use_transposition_table_ && transposition_table_) {
            handleTranspositionMatch(selected_child, parent_for_selection);
        }
        
        current = selected_child;
        path.push_back(current);
    }
    
    return current;
}

// Main tree traversal method with safety checks
std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
MCTSEngine::selectLeafNode(std::shared_ptr<MCTSNode> root) {
    static int leaf_selection_counter = 0;
    leaf_selection_counter++;
    
    std::vector<std::shared_ptr<MCTSNode>> path;
    
    if (!root) {
        std::cerr << "ERROR: Root node is null!" << std::endl;
        return {nullptr, path};
    }
    
    // Ensure root node is expanded before traversal
    if (root->isLeaf() && !root->isTerminal() && !root->hasPendingEvaluation() && !root->isBeingEvaluated()) {
        bool expansion_success = expandNonTerminalLeaf(root);
    }
    
    path.push_back(root);
    
    // If root is already a leaf, return it directly
    if (root->isLeaf() || root->isTerminal()) {
        return {root, path};
    }
    
    // Traverse the tree to find a leaf node
    auto leaf = traverseTreeForLeaf(root, path);
    
    // If leaf is null, it means we encountered a node with pending evaluation
    if (!leaf) {
        return {nullptr, path};
    }
    
    // If we found a leaf that is NOT terminal, try expanding it
    if (leaf->isLeaf() && !leaf->isTerminal() && !leaf->hasPendingEvaluation() && !leaf->isBeingEvaluated()) {
        // Try to expand the node
        bool expansion_success = expandNonTerminalLeaf(leaf);
        
        if (expansion_success) {
            // Evaluate the leaf node directly on first expansion
            // This ensures we get network evaluation results for this new part of the tree
            return {leaf, path};
        }
    }
    
    // More aggressive handling of stuck nodes and evaluation
    static std::unordered_map<MCTSNode*, int> stuck_nodes_counter;
    
    // If we didn't find a leaf, that's a major issue
    if (!leaf) {
        return {nullptr, path};
    }
    
    // ULTRA Always clear evaluation flags to avoid stalling, then re-evaluate
    // This is a workaround for nodes that might be stuck in evaluation state
    if (leaf->hasPendingEvaluation() || leaf->isBeingEvaluated()) {
        stuck_nodes_counter[leaf.get()]++;
        
        // If this node has been seen as stuck, clear its flags
        // Lower the threshold from 10 to 3 for faster clearing
        if (stuck_nodes_counter[leaf.get()] > 3) {
            // Force clear both flags
            leaf->clearAllEvaluationFlags();
            
            // Reset counter
            stuck_nodes_counter[leaf.get()] = 0;
        }
    } else {
        // Reset counter for nodes that aren't stuck
        stuck_nodes_counter[leaf.get()] = 0;
    }
    
    // Always try to mark non-terminal nodes for evaluation, regardless of previous status
    if (!leaf->isTerminal()) {
        bool marked = safelyMarkNodeForEvaluation(leaf);
        
        // If the node couldn't be marked, something might be wrong - clear flags and try again
        if (!marked) {
            leaf->clearAllEvaluationFlags();
            
            // Try again after clearing
            marked = safelyMarkNodeForEvaluation(leaf);
        }
    }
    
    return {leaf, path};
}

// Method to back up values through the tree
void MCTSEngine::backPropagate(std::vector<std::shared_ptr<MCTSNode>>& path, float value) {
    // Value alternates sign as we move up the tree (perspective changes)
    bool invert = false;
    
    // RAVE update preparation - collect all actions in the path
    std::vector<int> path_actions;
    if (settings_.use_rave) {
        for (const auto& node : path) {
            if (node->getAction() != -1) {
                path_actions.push_back(node->getAction());
            }
        }
    }
    
    // Process nodes in reverse order (from leaf to root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto node = *it;
        float update_value = invert ? -value : value;
        
        // Remove virtual loss and update node statistics
        node->removeVirtualLoss(settings_.virtual_loss);
        node->update(update_value);
        
        // RAVE update - update all children that match actions in the path
        if (settings_.use_rave && node->getChildren().size() > 0) {
            for (auto& child : node->getChildren()) {
                int child_action = child->getAction();
                
                // Check if this action appears later in the path (RAVE principle)
                for (int path_action : path_actions) {
                    if (child_action == path_action) {
                        // Update RAVE value for this child
                        float rave_value = invert ? -value : value;
                        child->updateRAVE(rave_value);
                        break; // Only update once per child
                    }
                }
            }
        }
        
        // Alternate perspective for next level
        invert = !invert;
    }
}

// Method to aggregate results from multiple search roots
void MCTSEngine::aggregateRootParallelResults(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty() || !root_) {
        return;
    }
    
    // First ensure root is expanded
    if (root_->isLeaf() && !root_->isTerminal()) {
        root_->expand(settings_.use_progressive_widening,
                    settings_.progressive_widening_c,
                    settings_.progressive_widening_k);
    }
    
    // Create a map of action to aggregated statistics
    std::unordered_map<int, int> action_visit_counts;
    std::unordered_map<int, double> action_value_sums;
    
    // Collect statistics from all search roots
    for (const auto& search_root : search_roots) {
        if (!search_root) {
            continue;
        }
        
        auto search_children = search_root->getChildren();
        for (const auto& child : search_children) {
            int action = child->getAction();
            int visits = child->getVisitCount();
            float value = child->getValue();
            
            action_visit_counts[action] += visits;
            action_value_sums[action] += visits * value;
        }
    }
    
    // Apply aggregated statistics to the main root's children
    auto root_children = root_->getChildren();
    for (auto& child : root_children) {
        int action = child->getAction();
        auto visits_it = action_visit_counts.find(action);
        
        if (visits_it != action_visit_counts.end() && visits_it->second > 0) {
            int total_visits = visits_it->second;
            double total_value = action_value_sums[action];
            float avg_value = total_value / total_visits;
            
            // Update the main root's child with aggregated statistics
            for (int i = 0; i < total_visits; i++) {
                child->update(avg_value);
            }
        }
    }
}

// Compute action probabilities from the search tree
std::vector<float> MCTSEngine::getActionProbabilities(std::shared_ptr<MCTSNode> root, float temperature) {
    if (!root || root->getChildren().empty()) {
        return std::vector<float>();
    }

    // Get actions and visit counts
    auto& actions = root->getActions();
    auto& children = root->getChildren();
    
    std::vector<float> counts;
    counts.reserve(children.size());

    for (auto child : children) {
        counts.push_back(static_cast<float>(child->getVisitCount()));
    }

    // Handle different temperature regimes
    std::vector<float> probabilities;
    probabilities.reserve(counts.size());

    if (temperature < 0.01f) {
        // Temperature near zero: deterministic selection - pick the move with highest visits
        auto max_it = std::max_element(counts.begin(), counts.end());
        size_t max_idx = std::distance(counts.begin(), max_it);
        
        // Set all probabilities to 0 except the highest
        probabilities.resize(counts.size(), 0.0f);
        probabilities[max_idx] = 1.0f;
    } else {
        // Apply temperature scaling: counts ^ (1/temperature)
        float sum = 0.0f;
        
        // First find the maximum count for numerical stability
        float max_count = *std::max_element(counts.begin(), counts.end());
        
        if (max_count <= 0.0f) {
            // If all counts are 0, use uniform distribution
            float uniform_prob = 1.0f / counts.size();
            probabilities.resize(counts.size(), uniform_prob);
        } else {
            // Compute the power of (count/max_count) for better numerical stability
            for (float count : counts) {
                float scaled_count = 0.0f;
                if (count > 0.0f) {
                    scaled_count = std::pow(count / max_count, 1.0f / temperature);
                }
                probabilities.push_back(scaled_count);
                sum += scaled_count;
            }
            
            // Normalize
            if (sum > 0.0f) {
                for (auto& prob : probabilities) {
                    prob /= sum;
                }
            } else {
                // Fallback to uniform if sum is zero
                float uniform_prob = 1.0f / counts.size();
                std::fill(probabilities.begin(), probabilities.end(), uniform_prob);
            }
        }
    }

    // Create full action space probabilities
    std::vector<float> action_probabilities(root->getState().getActionSpaceSize(), 0.0f);

    // Map child indices to action indices
    for (size_t i = 0; i < actions.size(); ++i) {
        int action = actions[i];
        if (action >= 0 && action < static_cast<int>(action_probabilities.size())) {
            action_probabilities[action] = probabilities[i];
        }
    }

    return action_probabilities;
}

} // namespace mcts
} // namespace alphazero