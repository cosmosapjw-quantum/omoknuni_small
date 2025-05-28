// mcts_engine_batch_tree_simple.cpp
// Simplified batch tree selection focusing on the core optimization

#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/shared_inference_queue.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>

namespace alphazero {
namespace mcts {

// Simple batch tree search that processes multiple simulations in parallel
void MCTSEngine::executeBatchedTreeSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state) {
    const int num_simulations = settings_.num_simulations;
    const int BATCH_SIZE = 128;  // Process 128 simulations at once for better GPU utilization
    
    std::cout << "\n[SimpleBatchTree] Starting search with " << num_simulations 
              << " simulations, batch_size=" << BATCH_SIZE << std::endl;
    
    auto search_start = std::chrono::steady_clock::now();
    int completed_simulations = 0;
    
    // Main simulation loop - process in batches
    while (completed_simulations < num_simulations) {
        int batch_size = std::min(BATCH_SIZE, num_simulations - completed_simulations);
        
        // Phase 1: Collect leaves from multiple simulations
        std::vector<MCTSNode*> leaf_nodes;
        std::vector<std::vector<MCTSNode*>> paths;
        std::vector<std::unique_ptr<core::IGameState>> leaf_states;
        
        leaf_nodes.reserve(batch_size);
        paths.reserve(batch_size);
        leaf_states.reserve(batch_size);
        
        auto collect_start = std::chrono::steady_clock::now();
        
        // Parallel leaf collection with more aggressive threading
        #pragma omp parallel for schedule(dynamic, 1) num_threads(settings_.num_threads)
        for (int i = 0; i < batch_size; i++) {
            // Thread-local variables for better cache performance
            thread_local std::vector<MCTSNode*> path;
            path.clear();
            path.reserve(32);
            
            // Clone state for this simulation
            auto state = root_state->clone();
            
            // Traverse to leaf
            MCTSNode* current = root;
            path.push_back(current);
            
            while (!current->isLeaf()) {
                // Select best child with virtual loss
                auto best_child = current->selectChild(
                    settings_.exploration_constant,
                    settings_.use_rave,
                    settings_.rave_constant
                );
                
                if (!best_child) break;
                
                // Apply virtual loss
                best_child->applyVirtualLoss(settings_.virtual_loss);
                
                // Make move
                state->makeMove(best_child->getAction());
                
                // Update path
                current = best_child.get();
                path.push_back(current);
            }
            
            // Store leaf information with minimal critical section
            #pragma omp critical(leaf_collection)
            {
                leaf_nodes.push_back(current);
                paths.emplace_back(path);  // Copy instead of move for thread-local
                leaf_states.push_back(std::move(state));
            }
        }
        
        auto collect_end = std::chrono::steady_clock::now();
        auto collect_time = std::chrono::duration_cast<std::chrono::microseconds>(
            collect_end - collect_start).count();
        
        // Phase 2: Batch neural network evaluation
        auto eval_start = std::chrono::steady_clock::now();
        std::vector<NetworkOutput> nn_results;
        
        if (!leaf_states.empty()) {
            if (GlobalInferenceQueue::isInitialized()) {
                try {
                    // Use SharedInferenceQueue for proper batching
                    auto future = GlobalInferenceQueue::getInstance().submitBatch(
                        std::move(leaf_states));
                    nn_results = future.get();
                    
                    static int batch_count = 0;
                    if (++batch_count % 10 == 0) {
                        std::cout << "[SimpleBatchTree] Batch #" << batch_count 
                                  << ": collected " << leaf_nodes.size() 
                                  << " leaves in " << collect_time << "us, "
                                  << "evaluated in SharedQueue" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[SimpleBatchTree] SharedInferenceQueue error: " 
                              << e.what() << std::endl;
                    // Fall back to direct inference
                    nn_results = direct_inference_fn_(leaf_states);
                }
            } else {
                // Direct inference
                nn_results = direct_inference_fn_(leaf_states);
            }
        }
        
        auto eval_end = std::chrono::steady_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(
            eval_end - eval_start).count();
        
        // Phase 3: Backpropagation
        auto backprop_start = std::chrono::steady_clock::now();
        
        size_t num_results = std::min(leaf_nodes.size(), nn_results.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_results; i++) {
            auto* leaf = leaf_nodes[i];
            const auto& path = paths[i];
            const auto& nn_output = nn_results[i];
            
            // Expand leaf if needed
            if (!leaf->isExpanded() && !leaf->isTerminal()) {
                leaf->expand(
                    settings_.use_progressive_widening,
                    settings_.progressive_widening_c,
                    settings_.progressive_widening_k
                );
                
                // Set prior probabilities
                if (!nn_output.policy.empty()) {
                    leaf->setPriorProbabilities(nn_output.policy);
                }
            }
            
            // Backpropagate value
            float value = nn_output.value;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                (*it)->update(value);
                (*it)->revertVirtualLoss(settings_.virtual_loss);
                value = -value;  // Flip value for opponent
            }
        }
        
        auto backprop_end = std::chrono::steady_clock::now();
        auto backprop_time = std::chrono::duration_cast<std::chrono::microseconds>(
            backprop_end - backprop_start).count();
        
        completed_simulations += batch_size;
        
        // Log timing every 10 batches
        static int log_count = 0;
        if (++log_count % 10 == 0) {
            std::cout << "[SimpleBatchTree] Timing - "
                      << "Collect: " << collect_time << "us, "
                      << "Eval: " << eval_time << "us, "
                      << "Backprop: " << backprop_time << "us, "
                      << "Total: " << (collect_time + eval_time + backprop_time) << "us"
                      << std::endl;
        }
    }
    
    auto search_end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - search_start).count();
    
    std::cout << "[SimpleBatchTree] Completed " << num_simulations 
              << " simulations in " << total_time << "ms ("
              << (total_time / float(num_simulations)) << "ms per simulation)"
              << std::endl;
}

} // namespace mcts
} // namespace alphazero