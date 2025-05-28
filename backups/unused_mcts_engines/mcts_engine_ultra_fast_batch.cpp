// mcts_engine_ultra_fast_batch.cpp
// Ultra-fast batch tree selection with prefetching and pipelining

#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/shared_inference_queue.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <atomic>
#include <future>
#include <omp.h>

namespace alphazero {
namespace mcts {

// Ultra-fast batch tree search with prefetching and pipelining
void MCTSEngine::executeUltraFastBatchSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state) {
    const int num_simulations = settings_.num_simulations;
    const int BATCH_SIZE = 128;  // Optimal GPU batch size
    const int PREFETCH_BATCHES = 2;  // Number of batches to prefetch
    
    std::cout << "\n[UltraFastBatch] Starting search with " << num_simulations 
              << " simulations, batch_size=" << BATCH_SIZE << std::endl;
    
    auto search_start = std::chrono::steady_clock::now();
    std::atomic<int> completed_simulations(0);
    
    // Pipeline stages
    struct BatchData {
        std::vector<MCTSNode*> leaf_nodes;
        std::vector<std::vector<MCTSNode*>> paths;
        std::vector<std::unique_ptr<core::IGameState>> leaf_states;
        std::future<std::vector<NetworkOutput>> nn_future;
        std::chrono::steady_clock::time_point collect_start;
        std::chrono::steady_clock::time_point collect_end;
    };
    
    // Initialize pipeline with empty batches
    std::vector<BatchData> pipeline(PREFETCH_BATCHES);
    for (auto& batch : pipeline) {
        batch.leaf_nodes.reserve(BATCH_SIZE);
        batch.paths.reserve(BATCH_SIZE);
        batch.leaf_states.reserve(BATCH_SIZE);
    }
    
    int current_batch_idx = 0;
    int total_batches = 0;
    
    // Main pipeline loop
    while (completed_simulations.load() < num_simulations) {
        int remaining = num_simulations - completed_simulations.load();
        int batch_size = std::min(BATCH_SIZE, remaining);
        
        auto& current_batch = pipeline[current_batch_idx % PREFETCH_BATCHES];
        
        // Clear previous batch data
        current_batch.leaf_nodes.clear();
        current_batch.paths.clear();
        current_batch.leaf_states.clear();
        
        // STAGE 1: Ultra-fast parallel leaf collection
        current_batch.collect_start = std::chrono::steady_clock::now();
        
        // Pre-allocate thread-local storage for each thread
        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<MCTSNode*>> thread_leaves(num_threads);
        std::vector<std::vector<std::vector<MCTSNode*>>> thread_paths(num_threads);
        std::vector<std::vector<std::unique_ptr<core::IGameState>>> thread_states(num_threads);
        
        for (int t = 0; t < num_threads; t++) {
            thread_leaves[t].reserve(batch_size / num_threads + 1);
            thread_paths[t].reserve(batch_size / num_threads + 1);
            thread_states[t].reserve(batch_size / num_threads + 1);
        }
        
        // Parallel leaf collection with minimal synchronization
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            auto& my_leaves = thread_leaves[tid];
            auto& my_paths = thread_paths[tid];
            auto& my_states = thread_states[tid];
            
            // Thread-local path buffer
            std::vector<MCTSNode*> path;
            path.reserve(32);
            
            #pragma omp for schedule(static)
            for (int i = 0; i < batch_size; i++) {
                // Clone state for this simulation
                auto state = root_state->clone();
                path.clear();
                
                // Traverse to leaf with minimal overhead
                MCTSNode* current = root;
                path.push_back(current);
                
                // Unrolled selection loop for better performance
                while (!current->isLeaf() && path.size() < 30) {
                    auto best_child = current->selectChildFast(settings_.exploration_constant);
                    if (!best_child) break;
                    
                    // Apply virtual loss
                    best_child->applyVirtualLoss(settings_.virtual_loss);
                    
                    // Make move
                    state->makeMove(best_child->getAction());
                    
                    // Update path
                    current = best_child.get();
                    path.push_back(current);
                }
                
                // Store in thread-local buffer
                my_leaves.push_back(current);
                my_paths.push_back(path);
                my_states.push_back(std::move(state));
            }
        }
        
        // Merge thread-local results
        for (int t = 0; t < num_threads; t++) {
            current_batch.leaf_nodes.insert(current_batch.leaf_nodes.end(),
                thread_leaves[t].begin(), thread_leaves[t].end());
            current_batch.paths.insert(current_batch.paths.end(),
                thread_paths[t].begin(), thread_paths[t].end());
            for (auto& state : thread_states[t]) {
                current_batch.leaf_states.push_back(std::move(state));
            }
        }
        
        current_batch.collect_end = std::chrono::steady_clock::now();
        
        // STAGE 2: Submit batch for GPU inference (async)
        if (!current_batch.leaf_states.empty()) {
            if (GlobalInferenceQueue::isInitialized()) {
                // Use SharedInferenceQueue for batching
                current_batch.nn_future = GlobalInferenceQueue::getInstance().submitBatch(
                    std::move(current_batch.leaf_states));
            } else if (direct_inference_fn_) {
                // Async direct inference
                current_batch.nn_future = std::async(std::launch::async, [this, &current_batch]() {
                    return direct_inference_fn_(current_batch.leaf_states);
                });
            }
        }
        
        // STAGE 3: Process previous batch results while current batch is computing
        if (total_batches > 0) {
            int prev_batch_idx = (current_batch_idx - 1 + PREFETCH_BATCHES) % PREFETCH_BATCHES;
            auto& prev_batch = pipeline[prev_batch_idx];
            
            if (prev_batch.nn_future.valid()) {
                auto nn_results = prev_batch.nn_future.get();
                
                // Ultra-fast parallel backpropagation
                size_t num_results = std::min(prev_batch.leaf_nodes.size(), nn_results.size());
                
                #pragma omp parallel for schedule(static) num_threads(num_threads)
                for (size_t i = 0; i < num_results; i++) {
                    auto* leaf = prev_batch.leaf_nodes[i];
                    const auto& path = prev_batch.paths[i];
                    const auto& nn_output = nn_results[i];
                    
                    // Expand leaf if needed
                    if (!leaf->isExpanded() && !leaf->isTerminal()) {
                        leaf->expandFast();  // Minimal expansion
                        
                        // Set prior probabilities
                        if (!nn_output.policy.empty()) {
                            leaf->setPriorProbabilities(nn_output.policy);
                        }
                    }
                    
                    // Fast backpropagation with unrolled loop
                    float value = nn_output.value;
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        (*it)->updateFast(value);
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                        value = -value;
                    }
                }
                
                completed_simulations.fetch_add(num_results);
            }
        }
        
        // Log timing periodically
        if (++total_batches % 5 == 0) {
            auto collect_time = std::chrono::duration_cast<std::chrono::microseconds>(
                current_batch.collect_end - current_batch.collect_start).count();
            
            std::cout << "[UltraFastBatch] Batch #" << total_batches 
                      << ": collected " << current_batch.leaf_nodes.size() 
                      << " leaves in " << collect_time << "us" << std::endl;
        }
        
        current_batch_idx++;
    }
    
    // Process any remaining batches in pipeline
    for (int i = 0; i < PREFETCH_BATCHES; i++) {
        auto& batch = pipeline[i];
        if (batch.nn_future.valid()) {
            auto nn_results = batch.nn_future.get();
            
            // Process results
            size_t num_results = std::min(batch.leaf_nodes.size(), nn_results.size());
            for (size_t j = 0; j < num_results; j++) {
                auto* leaf = batch.leaf_nodes[j];
                const auto& path = batch.paths[j];
                const auto& nn_output = nn_results[j];
                
                if (!leaf->isExpanded() && !leaf->isTerminal()) {
                    leaf->expandFast();
                    if (!nn_output.policy.empty()) {
                        leaf->setPriorProbabilities(nn_output.policy);
                    }
                }
                
                float value = nn_output.value;
                for (auto it = path.rbegin(); it != path.rend(); ++it) {
                    (*it)->updateFast(value);
                    (*it)->revertVirtualLoss(settings_.virtual_loss);
                    value = -value;
                }
            }
        }
    }
    
    auto search_end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - search_start).count();
    
    std::cout << "[UltraFastBatch] Completed " << num_simulations 
              << " simulations in " << total_time << "ms ("
              << (total_time / float(num_simulations)) << "ms per simulation, "
              << "~" << (1000.0f * num_simulations / total_time) << " sims/sec)"
              << std::endl;
}

} // namespace mcts
} // namespace alphazero