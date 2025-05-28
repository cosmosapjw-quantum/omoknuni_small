// mcts_engine_batch_selection.cpp
// Batch tree selection implementation for MCTS engine

#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/shared_inference_queue.h"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <iostream>

namespace alphazero {
namespace mcts {

// Structure to hold evaluation result
struct EvalResult {
    MCTSNode* node;
    float value;
    std::vector<float> policy;
    std::vector<MCTSNode*> path;
};

// Structure to hold traversal state for one path
struct TraversalState {
    MCTSNode* current_node;
    std::vector<MCTSNode*> path;
    std::unique_ptr<core::IGameState> state;
    bool is_terminal;
    
    TraversalState(MCTSNode* root, std::unique_ptr<core::IGameState> initial_state)
        : current_node(root)
        , state(std::move(initial_state))
        , is_terminal(false) {
        path.reserve(32);  // Reserve space for typical game depth
        path.push_back(root);
    }
};

// Batch traverse multiple paths to leaves simultaneously
void MCTSEngine::batchTraverseToLeaves(
    const std::vector<MCTSNode*>& roots,
    const std::vector<std::unique_ptr<core::IGameState>>& initial_states,
    std::vector<BatchItem>& leaf_items,
    int batch_size) {
    
    // Initialize traversal states
    std::vector<TraversalState> states;
    states.reserve(batch_size);
    
    for (int i = 0; i < batch_size && i < roots.size(); i++) {
        states.emplace_back(roots[i], initial_states[i]->clone());
    }
    
    // Traverse until all paths reach leaves
    bool all_leaves = false;
    while (!all_leaves) {
        all_leaves = true;
        
        // Process each path in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < states.size(); i++) {
            auto& tstate = states[i];
            
            // Skip if already at leaf
            if (tstate.is_terminal || !tstate.current_node) {
                continue;
            }
            
            // Check if current node is a leaf
            if (tstate.current_node->isLeaf()) {
                tstate.is_terminal = true;
                continue;
            }
            
            all_leaves = false;
            
            // Select best child (with virtual loss)
            auto best_child_ptr = tstate.current_node->selectChild(
                settings_.exploration_constant,
                settings_.use_rave,
                settings_.rave_constant
            );
            MCTSNode* best_child = best_child_ptr.get();
            
            if (best_child) {
                // Apply virtual loss immediately
                best_child->applyVirtualLoss(settings_.virtual_loss);
                
                // Update state
                if (tstate.state) {
                    tstate.state->makeMove(best_child->getAction());
                }
                
                // Move to child
                tstate.current_node = best_child;
                tstate.path.push_back(best_child);
            } else {
                // No valid child, mark as terminal
                tstate.is_terminal = true;
            }
        }
    }
    
    // Collect leaf nodes
    leaf_items.clear();
    leaf_items.reserve(states.size());
    
    for (auto& tstate : states) {
        if (tstate.current_node && tstate.state) {
            BatchItem item;
            item.current_node = tstate.current_node;
            item.state = std::move(tstate.state);
            item.path = std::move(tstate.path);
            leaf_items.push_back(std::move(item));
        }
    }
}

// Optimized batch search using level-synchronous traversal
void MCTSEngine::executeBatchedTreeSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state) {
    const int num_simulations = settings_.num_simulations;
    
    std::cout << "\n[BatchSelection] Starting executeBatchedTreeSearch with " 
              << num_simulations << " simulations" << std::endl;
    
    // Configuration for batching
    const int TRAVERSAL_BATCH_SIZE = 32;  // Process 32 paths at once
    const int MIN_GPU_BATCH_SIZE = 120;   // Minimum batch for GPU
    
    // Queues for different stages
    moodycamel::ConcurrentQueue<BatchItem> traversal_queue(1024);
    moodycamel::ConcurrentQueue<BatchItem> gpu_batch_queue(256);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(1024);
    
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_pending(0);
    std::atomic<bool> shutdown(false);
    
    // Thread pool for different stages
    std::vector<std::thread> threads;
    
    // Stage 1: Batch tree traversal threads
    const int num_traversal_threads = std::min(4, settings_.num_threads / 2);
    for (int t = 0; t < num_traversal_threads; t++) {
        threads.emplace_back([&, t]() {
            // Create initial states for this thread's batch
            std::vector<MCTSNode*> roots(TRAVERSAL_BATCH_SIZE, root);
            std::vector<std::unique_ptr<core::IGameState>> states;
            
            for (int i = 0; i < TRAVERSAL_BATCH_SIZE; i++) {
                states.push_back(root_state->clone());
            }
            
            std::vector<BatchItem> leaf_items;
            
            while (simulations_completed.load() < num_simulations && !shutdown.load()) {
                // Calculate how many simulations to run
                int remaining = num_simulations - simulations_completed.load();
                int batch_size = std::min(TRAVERSAL_BATCH_SIZE, remaining);
                
                if (batch_size <= 0) break;
                
                // Batch traverse to leaves
                batchTraverseToLeaves(roots, states, leaf_items, batch_size);
                
                // Enqueue leaves
                for (auto& item : leaf_items) {
                    traversal_queue.enqueue(std::move(item));
                    leaves_pending.fetch_add(1);
                }
                
                simulations_completed.fetch_add(batch_size);
            }
        });
    }
    
    // Stage 2: Batch collection thread
    threads.emplace_back([&]() {
        std::vector<BatchItem> gpu_batch;
        gpu_batch.reserve(MIN_GPU_BATCH_SIZE);
        
        auto batch_start_time = std::chrono::steady_clock::now();
        const auto max_wait_time = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        while (!shutdown.load() || leaves_pending.load() > 0) {
            BatchItem item;
            
            // Try to collect items
            while (gpu_batch.size() < MIN_GPU_BATCH_SIZE && 
                   traversal_queue.try_dequeue(item)) {
                gpu_batch.push_back(std::move(item));
            }
            
            auto now = std::chrono::steady_clock::now();
            bool should_process = false;
            
            // Decide whether to process batch
            if (gpu_batch.size() >= MIN_GPU_BATCH_SIZE) {
                should_process = true;  // Full batch
            } else if (!gpu_batch.empty() && 
                       (now - batch_start_time) >= max_wait_time) {
                should_process = true;  // Timeout
            } else if (!gpu_batch.empty() && 
                       leaves_pending.load() == gpu_batch.size()) {
                should_process = true;  // No more items coming
            }
            
            if (should_process && !gpu_batch.empty()) {
                // Submit batch for GPU processing
                for (auto& item : gpu_batch) {
                    gpu_batch_queue.enqueue(std::move(item));
                }
                
                gpu_batch.clear();
                batch_start_time = now;
            } else if (gpu_batch.empty()) {
                // No items available, wait briefly
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    });
    
    // Stage 3: GPU inference thread
    threads.emplace_back([&]() {
        std::vector<BatchItem> batch_items;
        std::vector<std::unique_ptr<core::IGameState>> state_batch;
        
        while (!shutdown.load() || gpu_batch_queue.size_approx() > 0) {
            batch_items.clear();
            
            // Dequeue batch
            size_t dequeued = gpu_batch_queue.try_dequeue_bulk(
                std::back_inserter(batch_items), 
                MIN_GPU_BATCH_SIZE
            );
            
            if (dequeued > 0) {
                // Prepare states for NN
                state_batch.clear();
                for (auto& item : batch_items) {
                    state_batch.push_back(std::move(item.state));
                }
                
                // GPU inference via SharedInferenceQueue
                std::vector<NetworkOutput> nn_results;
                
                if (GlobalInferenceQueue::isInitialized()) {
                    try {
                        // Submit to SharedInferenceQueue for proper batching
                        auto future = GlobalInferenceQueue::getInstance().submitBatch(std::move(state_batch));
                        nn_results = future.get();
                        
                        // Debug logging
                        static std::atomic<int> batch_count(0);
                        int count = ++batch_count;
                        if (count % 10 == 0) {
                            std::cout << "[BatchSelection] Processed batch #" << count 
                                      << " with " << batch_items.size() << " items" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[BatchSelection] SharedInferenceQueue error: " << e.what() << std::endl;
                        // Fall back to direct inference
                        nn_results = direct_inference_fn_(state_batch);
                    }
                } else {
                    // Use direct inference if SharedInferenceQueue not available
                    nn_results = direct_inference_fn_(state_batch);
                }
                
                // Create results
                for (size_t i = 0; i < batch_items.size() && i < nn_results.size(); i++) {
                    EvalResult result;
                    result.node = batch_items[i].current_node;
                    result.value = nn_results[i].value;
                    result.policy = nn_results[i].policy;
                    result.path = std::move(batch_items[i].path);
                    
                    result_queue.enqueue(std::move(result));
                }
                
                leaves_pending.fetch_sub(dequeued);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    });
    
    // Stage 4: Backpropagation threads
    const int num_backprop_threads = std::min(4, settings_.num_threads / 2);
    for (int t = 0; t < num_backprop_threads; t++) {
        threads.emplace_back([&]() {
            std::vector<EvalResult> results;
            results.reserve(32);
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
                results.clear();
                
                // Bulk dequeue
                size_t dequeued = result_queue.try_dequeue_bulk(
                    std::back_inserter(results), 32
                );
                
                if (dequeued > 0) {
                    for (auto& result : results) {
                        // Expand node if needed
                        if (result.node && !result.node->isExpanded() && 
                            result.node->getVisitCount() > 0) {
                            result.node->expand(
                                settings_.use_progressive_widening,
                                settings_.progressive_widening_c,
                                settings_.progressive_widening_k
                            );
                        }
                        
                        // Set policy
                        if (!result.policy.empty() && result.node && 
                            result.node->isLeaf()) {
                            result.node->setPriorProbabilities(result.policy);
                        }
                        
                        // Backpropagate
                        float value = result.value;
                        for (auto it = result.path.rbegin(); 
                             it != result.path.rend(); ++it) {
                            if (*it) {
                                (*it)->update(value);
                                (*it)->revertVirtualLoss(settings_.virtual_loss);
                                value = -value;
                            }
                        }
                    }
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Shutdown pipeline
    shutdown.store(true);
    
    // Wait for all threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

} // namespace mcts
} // namespace alphazero