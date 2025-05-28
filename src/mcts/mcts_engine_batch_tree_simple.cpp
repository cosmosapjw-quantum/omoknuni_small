// mcts_engine_batch_tree_simple.cpp
// Simplified batch tree selection focusing on the core optimization

#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/shared_inference_queue.h"
#include <vector>
#include <omp.h>

namespace alphazero {
namespace mcts {

// Simple batch tree search that processes multiple simulations in parallel
void MCTSEngine::executeBatchedTreeSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state) {
    const int num_simulations = settings_.num_simulations;
    const int BATCH_SIZE = 128;  // Process 128 simulations at once for better GPU utilization
    
    
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
        
        // Thread-local storage to eliminate critical sections
        const int num_threads = settings_.num_threads;
        std::vector<std::vector<MCTSNode*>> thread_leaf_nodes(num_threads);
        std::vector<std::vector<std::vector<MCTSNode*>>> thread_paths(num_threads);
        std::vector<std::vector<std::unique_ptr<core::IGameState>>> thread_leaf_states(num_threads);
        
        // Pre-reserve for each thread
        for (int t = 0; t < num_threads; t++) {
            thread_leaf_nodes[t].reserve(batch_size / num_threads + 1);
            thread_paths[t].reserve(batch_size / num_threads + 1);
            thread_leaf_states[t].reserve(batch_size / num_threads + 1);
        }
        
        // Parallel leaf collection with NO critical sections
        #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
        for (int i = 0; i < batch_size; i++) {
            int thread_id = omp_get_thread_num();
            
            // Thread-local variables for better cache performance
            thread_local std::vector<MCTSNode*> path;
            thread_local std::vector<int> move_sequence;
            path.clear();
            move_sequence.clear();
            path.reserve(32);
            move_sequence.reserve(32);
            
            // Traverse to leaf WITHOUT cloning state (major optimization)
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
                
                // Record move instead of applying it
                move_sequence.push_back(best_child->getAction());
                
                // Update path
                current = best_child.get();
                path.push_back(current);
            }
            
            // Now clone state and apply moves ONLY for leaf nodes
            auto state = root_state->clone();
            for (int move : move_sequence) {
                state->makeMove(move);
            }
            
            // Store in thread-local storage (NO critical section!)
            thread_leaf_nodes[thread_id].push_back(current);
            thread_paths[thread_id].emplace_back(path);
            thread_leaf_states[thread_id].push_back(std::move(state));
        }
        
        // Combine results from all threads (single-threaded, fast)
        for (int t = 0; t < num_threads; t++) {
            leaf_nodes.insert(leaf_nodes.end(), thread_leaf_nodes[t].begin(), thread_leaf_nodes[t].end());
            paths.insert(paths.end(), 
                        std::make_move_iterator(thread_paths[t].begin()),
                        std::make_move_iterator(thread_paths[t].end()));
            leaf_states.insert(leaf_states.end(),
                              std::make_move_iterator(thread_leaf_states[t].begin()),
                              std::make_move_iterator(thread_leaf_states[t].end()));
        }
        
        
        // Phase 2: Batch neural network evaluation
        std::vector<NetworkOutput> nn_results;
        
        if (!leaf_states.empty()) {
            if (GlobalInferenceQueue::isInitialized()) {
                try {
                    // Use SharedInferenceQueue for proper batching
                    auto future = GlobalInferenceQueue::getInstance().submitBatch(
                        std::move(leaf_states));
                    nn_results = future.get();
                    
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
        
        // Phase 3: Backpropagation
        
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
        
        completed_simulations += batch_size;
    }
    
}

} // namespace mcts
} // namespace alphazero