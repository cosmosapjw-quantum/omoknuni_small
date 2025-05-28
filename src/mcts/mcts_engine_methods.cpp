#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
// #include "mcts/advanced_memory_pool.h" // Removed
#include "core/igamestate.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>

namespace alphazero {
namespace mcts {

// Implementation of the missing methods that are causing undefined references

// Clone a game state using the memory pool if enabled
std::shared_ptr<core::IGameState> MCTSEngine::cloneGameState(const core::IGameState& state) {
    // CRITICAL FIX: Always prioritize pool usage to reduce memory allocations
    if (game_state_pool_enabled_) {
        auto& pool_manager = utils::GameStatePoolManager::getInstance();
        
        // Initialize pool if not exists
        if (!pool_manager.hasPool(state.getGameType())) {
            try {
                // Create large pool to handle concurrent requests
                size_t pool_size = std::max(size_t(2000), static_cast<size_t>(settings_.num_simulations * 2));
                pool_manager.initializePool(state.getGameType(), pool_size);
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize game state pool: " << e.what() << std::endl;
            }
        }
        
        // Try to use game state pool
        auto unique_state = pool_manager.cloneState(state);
        
        if (unique_state) {
            // Convert to shared_ptr (pool will handle memory internally)
            return std::shared_ptr<core::IGameState>(std::move(unique_state));
        }
    }
    
    // Advanced memory pool removed
    /*
    if (use_advanced_memory_pool_ && memory_pool_) {
        // Allocate state from our optimized memory pool
        return memory_pool_->allocateGameState(state);
    }
    */
    
    // Fallback to standard clone
    return state.clone();
}

// Expand a non-terminal leaf node
bool MCTSEngine::expandNonTerminalLeaf(std::shared_ptr<MCTSNode>& leaf) {
    // Check if the node is already expanded
    if (leaf->isFullyExpanded()) {
        return true;
    }
    
    // Get the legal moves from the state
    auto& state = leaf->getState();
    auto legal_moves = state.getLegalMoves();
    
    if (legal_moves.empty()) {
        // This is a terminal state with no legal moves
        return false;
    }
    
    // Create child nodes for each legal move using the expand method
    leaf->expand();
    
    return true;
}

// Create multiple search roots for root parallelization
std::vector<std::shared_ptr<MCTSNode>> MCTSEngine::createSearchRoots(std::shared_ptr<MCTSNode> main_root, int num_roots) {
    std::vector<std::shared_ptr<MCTSNode>> search_roots;
    
    // Always include the main root
    search_roots.push_back(main_root);
    
    // If only one root requested or root parallelization is disabled, return just the main root
    if (num_roots <= 1 || !settings_.use_root_parallelization) {
        return search_roots;
    }
    
    // Create additional roots (num_roots-1 since we already have the main root)
    for (int i = 1; i < num_roots; ++i) {
        // Clone the main root state
        auto root_state_clone = main_root->getState().clone();
        
        // Create a new root node with the cloned state
        auto new_root = MCTSNode::create(std::move(root_state_clone));
        
        // Add to roots list
        search_roots.push_back(new_root);
    }
    
    return search_roots;
}

// Safely mark a node for evaluation, handling edge cases
bool MCTSEngine::safelyMarkNodeForEvaluation(std::shared_ptr<MCTSNode> node) {
    if (!node) {
        return false;
    }
    
    // If the node is already being evaluated, skip it
    if (node->hasPendingEvaluation() || node->isBeingEvaluated()) {
        return false;
    }
    
    // Mark the node as pending evaluation
    node->markEvaluationPending();
    
    return true;
}

// Setup batch parameters
void MCTSEngine::setupBatchParameters() {
    // Synchronize batch parameters from legacy settings
    settings_.syncBatchParametersFromLegacy();
    
    // Ensure the optimal batch size is reasonable
    if (settings_.batch_params.optimal_batch_size < 4) {
        settings_.batch_params.optimal_batch_size = 4;
    }
    
    // Update minimum batch sizes based on optimal size
    settings_.batch_params.minimum_viable_batch_size = 
        std::max(static_cast<size_t>(settings_.batch_params.optimal_batch_size * 0.75), 
                static_cast<size_t>(4));
                
    settings_.batch_params.minimum_fallback_batch_size = 
        std::max(static_cast<size_t>(settings_.batch_params.optimal_batch_size * 0.3), 
                static_cast<size_t>(1));
                
    // Ensure collection batch size is reasonable
    if (settings_.batch_params.max_collection_batch_size < 1) {
        settings_.batch_params.max_collection_batch_size = 1;
    }
    
    // Update legacy settings for backward compatibility
    settings_.syncLegacyFromBatchParameters();
}

// Create a default policy (prior probability distribution) for unexplored nodes
std::vector<float> MCTSEngine::createDefaultPolicy(int action_space_size) {
    if (action_space_size <= 0) {
        return {};
    }
    
    // Uniform distribution over all possible actions
    std::vector<float> default_policy(action_space_size, 1.0f / action_space_size);
    
    return default_policy;
}

// Parallel search methods
MCTSEngine::ParallelLeafResult MCTSEngine::selectLeafNodeParallel(
    std::shared_ptr<MCTSNode> root, 
    std::vector<std::shared_ptr<MCTSNode>>& path,
    std::mt19937& rng) {
    
    ParallelLeafResult result;
    path.clear();
    
    auto current = root;
    bool virtual_loss_applied = false;
    
    // Traverse down the tree using optimized UCB selection
    while (current && !current->isLeaf()) {
        path.push_back(current);
        
        // Apply virtual loss to prevent thread collisions
        if (!virtual_loss_applied && settings_.virtual_loss > 0) {
            current->applyVirtualLoss(settings_.virtual_loss);
            virtual_loss_applied = true;
        }
        
        // Enhanced UCB selection with improved exploration
        auto next = current->selectBestChildUCB(settings_.exploration_constant, rng);
        
        if (!next) {
            // Selection failed - revert virtual loss and return
            if (virtual_loss_applied) {
                for (auto& node : path) {
                    node->revertVirtualLoss(settings_.virtual_loss);
                }
            }
            return result; // Empty result
        }
        
        current = next;
    }
    
    if (!current) {
        return result; // Empty result
    }
    
    path.push_back(current);
    result.leaf_node = current;
    result.path = path;
    result.applied_virtual_loss = virtual_loss_applied;
    
    // Check if this is a terminal node
    if (current->isTerminal()) {
        result.terminal = true;
        result.terminal_value = current->getState().getGameResult() == core::GameResult::WIN_PLAYER1 ? 1.0f : 
                               (current->getState().getGameResult() == core::GameResult::WIN_PLAYER2 ? -1.0f : 0.0f);
    }
    
    return result;
}

void MCTSEngine::backpropagateParallel(
    const std::vector<std::shared_ptr<MCTSNode>>& path, 
    float value, 
    int virtual_loss_amount) {
    
    // Backpropagate value up the path
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto node = *it;
        
        // Update node statistics
        node->updateStats(value);
        
        // Revert virtual loss
        if (virtual_loss_amount > 0) {
            node->revertVirtualLoss(virtual_loss_amount);
        }
        
        // Flip value for opponent
        value = -value;
    }
}

} // namespace mcts
} // namespace alphazero