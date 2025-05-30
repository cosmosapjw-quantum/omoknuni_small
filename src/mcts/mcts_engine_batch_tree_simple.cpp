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
    // LOG_SYSTEM_INFO("!!! ENTERED executeBatchedTreeSearch - FIRST LINE !!!");
    
    if (!root_state) {
        LOG_SYSTEM_ERROR("executeBatchedTreeSearch: root_state is null!");
        return;
    }
    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: root_state is valid");
    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Starting with {} simulations", settings_.num_simulations);
    
    // CRITICAL FIX: Create a const reference for thread-safe access
    const core::IGameState& root_state_ref = *root_state;
    
    const int num_simulations = settings_.num_simulations;
    const int BATCH_SIZE = settings_.batch_size;  // Use configured batch size
    
    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Using batch size {} with {} threads", BATCH_SIZE, settings_.num_threads);
    
    int completed_simulations = 0;
    
    // Main simulation loop - process in batches
    while (completed_simulations < num_simulations) {
        int batch_size = std::min(BATCH_SIZE, num_simulations - completed_simulations);
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Processing batch of {} simulations (completed: {}/{})", 
        //                 batch_size, completed_simulations, num_simulations);
        
        // Phase 1: Collect leaves from multiple simulations
        std::vector<MCTSNode*> leaf_nodes;
        std::vector<std::vector<MCTSNode*>> paths;
        std::vector<std::unique_ptr<core::IGameState>> leaf_states;
        std::vector<std::shared_ptr<MCTSNode>> leaf_shared_ptrs; // Track shared_ptrs for TT storage
        
        leaf_nodes.reserve(batch_size);
        paths.reserve(batch_size);
        leaf_states.reserve(batch_size);
        leaf_shared_ptrs.reserve(batch_size);
        
        // Thread-local storage to eliminate critical sections
        const int num_threads = settings_.num_threads;
        std::vector<std::vector<MCTSNode*>> thread_leaf_nodes(num_threads);
        std::vector<std::vector<std::vector<MCTSNode*>>> thread_paths(num_threads);
        std::vector<std::vector<std::unique_ptr<core::IGameState>>> thread_leaf_states(num_threads);
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> thread_leaf_shared_ptrs(num_threads);
        
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Pre-reserving thread-local storage for {} threads", num_threads);
        
        // Pre-reserve for each thread
        for (int t = 0; t < num_threads; t++) {
            thread_leaf_nodes[t].reserve(batch_size / num_threads + 1);
            thread_paths[t].reserve(batch_size / num_threads + 1);
            thread_leaf_states[t].reserve(batch_size / num_threads + 1);
            thread_leaf_shared_ptrs[t].reserve(batch_size / num_threads + 1);
        }
        
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Starting parallel leaf collection");
        
        // Parallel leaf collection with NO critical sections
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: About to start parallel loop with {} iterations", batch_size);
        
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
            std::shared_ptr<MCTSNode> current_shared = root_;  // Start with root shared_ptr
            if (!current) {
                LOG_SYSTEM_ERROR("executeBatchedTreeSearch: Root node is null for thread {}!", thread_id);
                continue;
            }
            path.push_back(current);
            
            // Debug first iteration
            if (i == 0 && thread_id == 0) {
                // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Root node isLeaf={}, isTerminal={}, hasChildren={}", 
                //                 current->isLeaf(), current->isTerminal(), !current->getChildren().empty());
            }
            
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
                
                // Store the selected child in TT first (before checking for transpositions)
                if (use_transposition_table_ && transposition_table_ && best_child->getVisitCount() == 0) {
                    // Clone state to compute hash
                    auto temp_state = root_state_ref.clone();
                    for (int move : move_sequence) {
                        temp_state->makeMove(move);
                    }
                    uint64_t hash = temp_state->getHash();
                    
                    // Store this new node
                    static_cast<PHMapTranspositionTable*>(transposition_table_.get())->store(
                        hash, best_child, move_sequence.size());
                }
                
                // Check transposition table for existing nodes
                if (use_transposition_table_ && transposition_table_) {
                    // Clone state to compute hash (only for TT lookup)
                    auto temp_state = root_state_ref.clone();
                    for (int move : move_sequence) {
                        temp_state->makeMove(move);
                    }
                    uint64_t hash = temp_state->getHash();
                    
                    
                    // Try to find existing node in transposition table
                    auto tt_node = static_cast<PHMapTranspositionTable*>(transposition_table_.get())->lookup(hash);
                    if (tt_node && tt_node != best_child && tt_node->getVisitCount() > 0) {
                        // Found a transposition! Use the existing node
                        
                        // Remove virtual loss from the child we were going to use
                        best_child->revertVirtualLoss(settings_.virtual_loss);
                        
                        // Update parent's child reference to point to the TT node
                        current->updateChildReference(best_child, tt_node);
                        
                        // Apply virtual loss to the TT node instead
                        tt_node->applyVirtualLoss(settings_.virtual_loss);
                        
                        // Use the TT node
                        best_child = tt_node;
                        current_shared = tt_node;  // Update shared_ptr reference
                    } else {
                        current_shared = best_child;  // Update to child's shared_ptr
                    }
                } else {
                    current_shared = best_child;  // Update to child's shared_ptr
                }
                
                // Update path
                current = best_child.get();
                path.push_back(current);
            }
            
            // Now clone state and apply moves ONLY for leaf nodes
            auto state = root_state_ref.clone();
            if (!state) {
                LOG_SYSTEM_ERROR("executeBatchedTreeSearch: Failed to clone root_state for thread {}!", thread_id);
                continue;
            }
            
            for (int move : move_sequence) {
                state->makeMove(move);
            }
            
            // Store in thread-local storage (NO critical section!)
            thread_leaf_nodes[thread_id].push_back(current);
            thread_paths[thread_id].emplace_back(path);
            thread_leaf_states[thread_id].push_back(std::move(state));
            thread_leaf_shared_ptrs[thread_id].push_back(current_shared);
        }
        
        // Log total collection summary
        int total_collected = 0;
        for (int t = 0; t < num_threads; t++) {
            total_collected += thread_leaf_nodes[t].size();
        }
        
        // Combine results from all threads (single-threaded, fast)
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Combining results from {} threads", num_threads);
        for (int t = 0; t < num_threads; t++) {
            leaf_nodes.insert(leaf_nodes.end(), thread_leaf_nodes[t].begin(), thread_leaf_nodes[t].end());
            paths.insert(paths.end(), 
                        std::make_move_iterator(thread_paths[t].begin()),
                        std::make_move_iterator(thread_paths[t].end()));
            leaf_states.insert(leaf_states.end(),
                              std::make_move_iterator(thread_leaf_states[t].begin()),
                              std::make_move_iterator(thread_leaf_states[t].end()));
            leaf_shared_ptrs.insert(leaf_shared_ptrs.end(),
                                   std::make_move_iterator(thread_leaf_shared_ptrs[t].begin()),
                                   std::make_move_iterator(thread_leaf_shared_ptrs[t].end()));
        }
        
        // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Collected {} leaves for evaluation", leaf_states.size());
        
        // Phase 2: Batch neural network evaluation
        std::vector<NetworkOutput> nn_results;
        
        if (!leaf_states.empty()) {
            // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Have {} leaf states to evaluate", leaf_states.size());
            
            if (GlobalInferenceQueue::isInitialized()) {
                try {
                    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Using SharedInferenceQueue for batch inference");
                    // Use SharedInferenceQueue for proper batching
                    auto future = GlobalInferenceQueue::getInstance().submitBatch(std::move(leaf_states));
                    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Waiting for inference results...");
                    nn_results = future.get();
                    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Received {} results from SharedInferenceQueue", nn_results.size());
                } catch (const std::exception& e) {
                    LOG_SYSTEM_ERROR("executeBatchedTreeSearch: SharedInferenceQueue error: {}", e.what());
                    // Fall back to direct inference
                    // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Falling back to direct inference");
                    nn_results = direct_inference_fn_(leaf_states);
                }
            } else {
                // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Using direct inference (GlobalInferenceQueue not initialized)");
                nn_results = direct_inference_fn_(leaf_states);
            }
            // LOG_SYSTEM_INFO("executeBatchedTreeSearch: Neural network returned {} results", nn_results.size());
        } else {
            LOG_SYSTEM_WARN("executeBatchedTreeSearch: No leaf states to evaluate");
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
                
                // Store expanded node in transposition table
                if (use_transposition_table_ && transposition_table_ && i < leaf_shared_ptrs.size()) {
                    try {
                        uint64_t hash = leaf->getState().getHash();
                        
                        
                        static_cast<PHMapTranspositionTable*>(transposition_table_.get())->store(
                            hash, leaf_shared_ptrs[i], path.size());
                    } catch (...) {
                        // Ignore TT storage errors
                    }
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
        
        // Update batch statistics
        if (!leaf_states.empty()) {
            last_stats_.total_batches_processed += 1;
            size_t current_batch_size = leaf_states.size();
            last_stats_.total_evaluations += current_batch_size;
            
            // Update running average of batch size
            if (last_stats_.total_batches_processed == 1) {
                last_stats_.avg_batch_size = static_cast<float>(current_batch_size);
            } else {
                last_stats_.avg_batch_size = (last_stats_.avg_batch_size * (last_stats_.total_batches_processed - 1) + current_batch_size) / last_stats_.total_batches_processed;
            }
        }
        
        completed_simulations += batch_size;
    }
    
}

} // namespace mcts
} // namespace alphazero