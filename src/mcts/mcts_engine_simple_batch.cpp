#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <omp.h>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

// CRITICAL FIX: Simplified direct batching without complex infrastructure
void MCTSEngine::executeSimpleBatchedSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty() || !search_roots[0]) {
        std::cerr << "No valid search roots provided" << std::endl;
        return;
    }
    
    auto root = search_roots[0];
    const int num_simulations = settings_.num_simulations;
    const int batch_size = settings_.batch_size;
    const int num_threads = settings_.num_threads;
    
    std::cout << "🎯 SIMPLE BATCHED SEARCH: " << num_simulations << " simulations, "
              << "batch_size=" << batch_size << ", threads=" << num_threads << std::endl;
    
    // Ensure root is expanded
    if (!root->isTerminal() && !root->isExpanded()) {
        expandNonTerminalLeaf(root);
    }
    
    int simulations_completed = 0;
    
    // Main search loop - process in batches
    while (simulations_completed < num_simulations) {
        // PHASE 1: Collect leaf nodes for batch
        std::vector<PendingEvaluation> batch;
        batch.reserve(batch_size);
        
        // Collect up to batch_size leaf nodes
        int remaining_sims = num_simulations - simulations_completed;
        int batch_to_collect = std::min(batch_size, remaining_sims);
        
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < batch_to_collect; ++i) {
            // Selection: Find leaf node
            std::vector<std::shared_ptr<MCTSNode>> path;
            std::shared_ptr<MCTSNode> current = root;
            path.push_back(current);
            
            // Traverse down to leaf
            while (!current->isTerminal() && current->isExpanded() && !current->getChildren().empty()) {
                // Apply virtual loss to prevent collision
                current->applyVirtualLoss(settings_.virtual_loss);
                
                // Select child using UCB
                auto selected = current->selectChild(settings_.exploration_constant);
                if (!selected) break;
                
                current = selected;
                path.push_back(current);
            }
            
            // Expand leaf if needed
            if (!current->isTerminal() && !current->isExpanded()) {
                #pragma omp critical
                {
                    if (!current->isExpanded()) {
                        expandNonTerminalLeaf(current);
                    }
                }
                
                // Select first child after expansion
                if (!current->getChildren().empty()) {
                    current = current->getChildren()[0];
                    path.push_back(current);
                }
            }
            
            // Add to batch if not terminal
            if (!current->isTerminal()) {
                #pragma omp critical
                {
                    // Clone state efficiently using pool
                    auto state_clone = current->getState().clone();
                    batch.emplace_back(current, std::move(state_clone), path);
                }
            } else {
                // Handle terminal node immediately
                float value = 0.0f;
                auto result = current->getState().getGameResult();
                if (result == core::GameResult::WIN_PLAYER1) {
                    value = current->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
                } else if (result == core::GameResult::WIN_PLAYER2) {
                    value = current->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
                }
                
                // Backpropagate immediately
                for (auto it = path.rbegin(); it != path.rend(); ++it) {
                    (*it)->update(value);
                    (*it)->revertVirtualLoss(settings_.virtual_loss);
                    value = -value; // Flip for opponent
                }
            }
        }
        
        // PHASE 2: Process batch with neural network
        if (!batch.empty()) {
            // Extract states for inference
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.reserve(batch.size());
            
            for (auto& eval : batch) {
                if (eval.state) {
                    states.push_back(eval.state->clone());
                }
            }
            
            // DIRECT NEURAL NETWORK CALL - No intermediate layers!
            std::vector<NetworkOutput> outputs;
            if (direct_inference_fn_ && !states.empty()) {
                auto nn_start = std::chrono::steady_clock::now();
                outputs = direct_inference_fn_(states);
                auto nn_end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(nn_end - nn_start);
                
                std::cout << "✅ Batch " << (simulations_completed / batch_size + 1) 
                          << ": " << states.size() << " states in " << duration.count() << "ms" 
                          << " (" << (states.size() * 1000.0 / (duration.count() + 1)) << " states/sec)" << std::endl;
            }
            
            // PHASE 3: Apply results and backpropagate
            for (size_t i = 0; i < batch.size() && i < outputs.size(); ++i) {
                auto& eval = batch[i];
                auto& output = outputs[i];
                
                // Set prior probabilities
                if (eval.node && !output.policy.empty()) {
                    eval.node->setPriorProbabilities(output.policy);
                }
                
                // Backpropagate value
                float value = output.value;
                for (auto it = eval.path.rbegin(); it != eval.path.rend(); ++it) {
                    (*it)->update(value);
                    (*it)->revertVirtualLoss(settings_.virtual_loss);
                    value = -value; // Flip for opponent
                }
            }
            
            simulations_completed += batch.size();
            
            // Clear batch immediately to free memory
            batch.clear();
            states.clear();
            
            // Periodic memory cleanup
            if (simulations_completed % (batch_size * 10) == 0) {
                #ifdef WITH_TORCH
                if (torch::cuda::is_available()) {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
                #endif
                
                // Clear game state pools
                if (game_state_pool_enabled_) {
                    utils::GameStatePoolManager::getInstance().clearAllPools();
                }
            }
        } else {
            // No batch collected, increment to avoid infinite loop
            simulations_completed += batch_size;
        }
    }
    
    std::cout << "✅ SIMPLE BATCHED SEARCH completed: " << simulations_completed << " simulations" << std::endl;
    
    // Final cleanup
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    #endif
}

} // namespace mcts
} // namespace alphazero