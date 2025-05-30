#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include "nn/gpu_optimizer.h"
#include "utils/profiler.h"
#include "utils/performance_profiler.h"
#include "mcts/shared_inference_queue.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <future>
#include <iterator>
#include <queue>
#include <omp.h>
#include <moodycamel/concurrentqueue.h>

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
    // Optimize thread count for Ryzen 9 5900X
    const int num_threads = std::min(settings_.num_threads, 12);
    
    // Log search parameters
    std::cout << "🎯 SIMPLE BATCHED SEARCH: " << num_simulations << " simulations, "
              << "batch_size=" << batch_size << ", threads=" << num_threads << std::endl;
    
    // DEBUG: Add timing information for each phase
    // struct DebugStats {
    //     std::atomic<int> leaves_collected{0};
    //     std::atomic<int> batches_processed{0};
    //     std::atomic<int> nodes_expanded{0};
    //     std::atomic<int> nodes_skipped{0};
    //     std::atomic<int64_t> total_collection_time_us{0};
    //     std::atomic<int64_t> total_inference_time_us{0};
    //     std::atomic<int64_t> total_backprop_time_us{0};
    //     std::atomic<int64_t> queue_wait_time_us{0};
    // } debug_stats;
    
    // Ensure root is expanded
    if (!root->isTerminal() && !root->isExpanded()) {
        expandNonTerminalLeaf(root);
    }
    
    // PRE-EXPANSION: Expand tree to depth 2-3 to reduce contention
    // This creates more starting points for parallel workers
    if (!root->isTerminal() && root->isExpanded()) {
        auto& root_children = root->getChildren();
        
        // Expand all root children (depth 1)
        #pragma omp parallel for num_threads(std::min(4, static_cast<int>(root_children.size())))
        for (size_t i = 0; i < root_children.size(); ++i) {
            auto& child = root_children[i];
            if (!child->isTerminal() && !child->isExpanded()) {
                expandNonTerminalLeaf(child);
            }
        }
        
        // Expand some grandchildren (depth 2) for high-priority moves
        for (size_t i = 0; i < std::min(size_t(4), root_children.size()); ++i) {
            auto& child = root_children[i];
            if (!child->isTerminal() && child->isExpanded()) {
                auto& grandchildren = child->getChildren();
                for (size_t j = 0; j < std::min(size_t(2), grandchildren.size()); ++j) {
                    if (!grandchildren[j]->isTerminal() && !grandchildren[j]->isExpanded()) {
                        expandNonTerminalLeaf(grandchildren[j]);
                    }
                }
            }
        }
    }
    
    int simulations_completed = 0;
    
    // Timing for batch collection
    auto batch_start_time = std::chrono::steady_clock::now();
    
    // Main search loop - process in batches
    while (simulations_completed < num_simulations) {
        // PROFILE_START(batch_collection); // Disabled for performance
        
        // PHASE 1: Collect leaf nodes for batch
        std::vector<PendingEvaluation> batch;
        batch.reserve(settings_.batch_params.optimal_batch_size);
        
        // Collect up to batch_size leaf nodes
        int remaining_sims = num_simulations - simulations_completed;
        
        // ADAPTIVE BATCH SIZING: Optimize for RTX 3060 Ti
        int target_batch_size = batch_size;  // Start with configured size (128)
        
        // Keep large batches for GPU efficiency
        if (remaining_sims < batch_size) {
            // Only reduce when we have fewer simulations than batch size
            if (remaining_sims >= 64) {
                target_batch_size = 64;  // Keep reasonable batch size
            } else if (remaining_sims >= 32) {
                target_batch_size = 32;  // Minimum efficient batch
            } else {
                target_batch_size = remaining_sims;  // Final cleanup
            }
        }
        
        // For RTX 3060 Ti, maintain minimum batch size for efficiency
        if (remaining_sims > 64 && target_batch_size < 64) {
            target_batch_size = 64;  // Optimal minimum for GPU
        }
        
        int batch_to_collect = target_batch_size;
        
        // LOCK-FREE PARALLEL BATCH COLLECTION using moodycamel::ConcurrentQueue
        auto collection_start = std::chrono::steady_clock::now();
        auto batch_start_total = std::chrono::steady_clock::now();
        
        // Node structure for traversal
        struct TraversalNode {
            std::shared_ptr<MCTSNode> node;
            std::vector<std::shared_ptr<MCTSNode>> path;
        };
        
        // Lock-free concurrent queues for work distribution
        moodycamel::ConcurrentQueue<TraversalNode> work_queue(batch_to_collect * 4);
        moodycamel::ConcurrentQueue<PendingEvaluation> batch_queue(batch_to_collect);
        moodycamel::ConcurrentQueue<TraversalNode> terminal_queue(batch_to_collect);
        
        // Initialize with root
        work_queue.enqueue({root, {root}});
        
        // Atomic counter for collected leaves
        std::atomic<int> leaves_collected(0);
        std::atomic<bool> collection_done(false);
        
        // PARALLEL LEAF COLLECTION - each thread independently processes nodes
        #pragma omp parallel num_threads(num_threads)
        {
            std::vector<TraversalNode> local_work;
            local_work.reserve(32);  // Local buffer for bulk operations
            
            while (!collection_done.load(std::memory_order_acquire)) {
                // Bulk dequeue for efficiency
                auto dequeue_start = std::chrono::high_resolution_clock::now();
                size_t dequeued = work_queue.try_dequeue_bulk(local_work.data(), 16);
                
                if (dequeued == 0) {
                    // Check if we're done
                    if (leaves_collected.load(std::memory_order_acquire) >= batch_to_collect) {
                        break;
                    }
                    // Brief yield before retry
                    std::this_thread::yield();
                    auto dequeue_end = std::chrono::high_resolution_clock::now();
                    // debug_stats.queue_wait_time_us.fetch_add(
                    //     std::chrono::duration_cast<std::chrono::microseconds>(dequeue_end - dequeue_start).count());
                    continue;
                }
                
                // Process dequeued nodes
                for (size_t i = 0; i < dequeued; ++i) {
                    auto& item = local_work[i];
                    auto current = item.node;
                    auto path = item.path;
                    
                    // Handle terminal nodes first
                    if (current->isTerminal()) {
                        terminal_queue.enqueue(std::move(item));
                        continue;
                    }
                    
                    // Process leaf nodes
                    if (!current->isExpanded()) {
                        // Try to mark for evaluation atomically (includes pending check)
                        if (current->tryMarkForEvaluation()) {
                            // Expand node
                            expandNonTerminalLeaf(current);
                            
                            // Apply virtual loss to prevent other threads from selecting
                            for (auto& node : path) {
                                node->applyVirtualLoss(settings_.virtual_loss);
                            }
                            
                            // Clone state using pool
                            auto& pool_manager = utils::GameStatePoolManager::getInstance();
                            auto state_clone = pool_manager.cloneState(current->getState());
                            
                            // Add to batch queue
                            batch_queue.enqueue({current, std::move(state_clone), path});
                            
                            // Increment counter
                            int count = leaves_collected.fetch_add(1, std::memory_order_release) + 1;
                            // debug_stats.leaves_collected.fetch_add(1);
                            // debug_stats.nodes_expanded.fetch_add(1);
                            if (count >= batch_to_collect) {
                                collection_done.store(true, std::memory_order_release);
                                break;
                            }
                        }
                    } else {
                        // Process expanded nodes - add children to work queue
                        auto& children = current->getChildren();
                        if (!children.empty()) {
                            // debug_stats.nodes_skipped.fetch_add(1);
                            // Find best children using simplified UCB (no OpenMP overhead)
                            std::vector<std::pair<float, size_t>> scores;
                            scores.reserve(children.size());
                            
                            int parent_visits = current->getVisitCount();
                            float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));
                            
                            for (size_t j = 0; j < children.size(); ++j) {
                                if (!children[j]->hasPendingEvaluation()) {
                                    // Calculate UCB score inline
                                    auto& child = children[j];
                                    int child_visits = child->getVisitCount();
                                    float child_value = child->getValue();
                                    int virtual_losses = child->getVirtualLoss();
                                    
                                    int effective_visits = child_visits + virtual_losses;
                                    float effective_value = child_value * child_visits - virtual_losses;
                                    
                                    float exploitation = effective_visits > 0 ? 
                                        effective_value / effective_visits : 0.0f;
                                    
                                    float exploration = child->getPriorProbability() * 
                                        sqrt_parent / (1.0f + effective_visits);
                                    
                                    float score = exploitation + settings_.exploration_constant * exploration;
                                    scores.emplace_back(score, j);
                                }
                            }
                            
                            // Sort by score and add top children to work queue
                            if (!scores.empty()) {
                                std::partial_sort(scores.begin(), 
                                    scores.begin() + std::min(size_t(8), scores.size()),
                                    scores.end(),
                                    std::greater<std::pair<float, size_t>>());
                                
                                // Add top children to work queue
                                size_t to_add = std::min(size_t(8), scores.size());
                                for (size_t j = 0; j < to_add; ++j) {
                                    auto child_path = path;
                                    child_path.push_back(children[scores[j].second]);
                                    work_queue.enqueue({children[scores[j].second], std::move(child_path)});
                                }
                            }
                        }
                    }
                }
                local_work.clear();
            }
        }
        
        // Collect results from batch queue
        PendingEvaluation eval;
        while (batch_queue.try_dequeue(eval)) {
            batch.push_back(std::move(eval));
        }
        
        // Process terminal nodes in parallel
        std::vector<TraversalNode> terminal_nodes;
        TraversalNode term_node;
        while (terminal_queue.try_dequeue(term_node)) {
            terminal_nodes.push_back(std::move(term_node));
        }
        
        if (!terminal_nodes.empty()) {
            #pragma omp parallel for num_threads(num_threads)
            for (size_t i = 0; i < terminal_nodes.size(); ++i) {
                auto& term = terminal_nodes[i];
                float value = 0.0f;
                auto result = term.node->getState().getGameResult();
                if (result == core::GameResult::WIN_PLAYER1) {
                    value = term.node->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
                } else if (result == core::GameResult::WIN_PLAYER2) {
                    value = term.node->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
                }
                
                // Backpropagate with virtual loss removal
                for (auto it = term.path.rbegin(); it != term.path.rend(); ++it) {
                    (*it)->update(value);
                    (*it)->revertVirtualLoss(settings_.virtual_loss);
                    value = -value;
                }
            }
        }
        
        // DIRECT BATCHING: Process what we have without waiting
        auto collection_end = std::chrono::steady_clock::now();
        auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            collection_end - collection_start);
        
        // Only try second collection if we have very few items and plenty of simulations left
        if (batch.size() < 16 && remaining_sims > 100 && collection_time.count() < 5) {
            // Calculate how many more we need
            int additional_needed = 16 - batch.size();
            int additional_to_collect = std::min(additional_needed, static_cast<int>(remaining_sims - batch.size()));
            
            if (additional_to_collect > 0) {
                // Second collection round with remaining threads
                #pragma omp parallel for num_threads(num_threads/2)
                for (int i = 0; i < additional_to_collect; ++i) {
                    // Same collection logic
                    std::vector<std::shared_ptr<MCTSNode>> path;
                    std::shared_ptr<MCTSNode> current = root;
                    path.push_back(current);
                    
                    while (!current->isTerminal() && current->isExpanded() && !current->getChildren().empty()) {
                        current->applyVirtualLoss(settings_.virtual_loss);
                        auto selected = current->selectChild(settings_.exploration_constant);
                        if (!selected) break;
                        current = selected;
                        path.push_back(current);
                    }
                    
                    if (!current->isTerminal() && !current->isExpanded()) {
                        #pragma omp critical
                        {
                            if (!current->isExpanded()) {
                                expandNonTerminalLeaf(current);
                            }
                        }
                        if (!current->getChildren().empty()) {
                            current = current->getChildren()[0];
                            path.push_back(current);
                        }
                    }
                    
                    if (!current->isTerminal()) {
                        #pragma omp critical
                        {
                            auto state_clone = current->getState().clone();
                            batch.emplace_back(current, std::move(state_clone), path);
                        }
                    }
                }
            }
        }
        
        // PROFILE_END(batch_collection); // Disabled for performance
        
        // PHASE 2: Process batch with neural network
        if (!batch.empty()) {
            // Skip verbose batch logging
            // Extract states for inference
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.reserve(batch.size());
            
            // Use GameStatePool for efficient cloning
            auto& pool_manager = utils::GameStatePoolManager::getInstance();
            for (auto& eval : batch) {
                if (eval.state) {
                    states.push_back(pool_manager.cloneState(*eval.state));
                }
            }
            
            // CRITICAL FIX: Optimized neural network inference with GPU enhancements
            std::vector<NetworkOutput> outputs;
            if (direct_inference_fn_ && !states.empty()) {
                // Start async inference
                auto nn_start = std::chrono::steady_clock::now();
                
                // OPTIMIZATION: Use GPU optimizer for efficient inference
                auto& gpu_optimizer = nn::getGlobalGPUOptimizer();
                
                // Skip batch accumulator for RTX 3060 Ti - direct inference is more efficient
                bool use_batch_accumulator = false;  // Disabled for direct batching
                
                static thread_local std::unique_ptr<nn::GPUOptimizer::DynamicBatchAccumulator> 
                    batch_accumulator = nullptr;
                
                if (use_batch_accumulator && !batch_accumulator) {
                    batch_accumulator = gpu_optimizer.createBatchAccumulator(
                        target_batch_size, target_batch_size * 2);
                }
                
                if (use_batch_accumulator && batch_accumulator) {
                    // Update accumulator target size based on queue pressure
                    batch_accumulator->updateOptimalSize(remaining_sims, 70.0f);
                }
                
                // OPTIMIZATION: Launch inference asynchronously with GPU optimizations
                auto inference_future = std::async(std::launch::async, [this, &states, &gpu_optimizer, use_batch_accumulator]() {
                    // PROFILE_SCOPE("NN_Inference"); // Disabled for performance
                    
                    // ALWAYS use SharedInferenceQueue for proper batching
                    if (GlobalInferenceQueue::isInitialized()) {
                        try {
                            auto future = GlobalInferenceQueue::getInstance().submitBatch(std::move(states));
                            return future.get();
                        } catch (const std::exception& e) {
                            std::cerr << "SharedInferenceQueue failed: " << e.what() << ", falling back to direct inference" << std::endl;
                        }
                    }
                    
                    // Fallback to direct inference
                    // Only use CUDA graphs for large batches to reduce overhead
#ifdef WITH_TORCH
                    if (use_batch_accumulator && states.size() >= 256 && 
                        neural_network_->isDeterministic() && gpu_optimizer.getConfig().enable_cuda_graphs) {
                        // Try to use CUDA graph for fixed pattern
                        std::string graph_id = "mcts_batch_" + std::to_string(states.size());
                        
                        if (!gpu_optimizer.isCudaGraphAvailable(graph_id)) {
                            // Capture graph on first use
                            auto example_tensor = gpu_optimizer.prepareStatesBatch(states, true);
                            gpu_optimizer.captureCudaGraph(graph_id, 
                                [this, &states]() -> torch::Tensor { 
                                    // Return dummy tensor for graph capture
                                    return torch::zeros({1}, torch::kFloat32);
                                }, 
                                example_tensor);
                        }
                    }
#endif
                    
                    return direct_inference_fn_(states);
                });
                
                // OPTIMIZATION: While GPU is processing, prepare next batch
                std::vector<PendingEvaluation> next_batch;
                next_batch.reserve(target_batch_size);
                
                // Collect next batch while GPU is busy
                int next_batch_size = std::min(target_batch_size, static_cast<int>(num_simulations - simulations_completed - batch.size()));
                if (next_batch_size > 0) {
                    #pragma omp parallel for num_threads(num_threads/2) // Use half threads for next batch
                    for (int i = 0; i < next_batch_size; ++i) {
                        // Same selection logic as before...
                        std::vector<std::shared_ptr<MCTSNode>> path;
                        std::shared_ptr<MCTSNode> current = root;
                        path.push_back(current);
                        
                        while (!current->isTerminal() && current->isExpanded() && !current->getChildren().empty()) {
                            current->applyVirtualLoss(settings_.virtual_loss);
                            auto selected = current->selectChild(settings_.exploration_constant);
                            if (!selected) break;
                            current = selected;
                            path.push_back(current);
                        }
                        
                        if (!current->isTerminal() && !current->isExpanded()) {
                            #pragma omp critical
                            {
                                if (!current->isExpanded()) {
                                    expandNonTerminalLeaf(current);
                                }
                            }
                            if (!current->getChildren().empty()) {
                                current = current->getChildren()[0];
                                path.push_back(current);
                            }
                        }
                        
                        if (!current->isTerminal()) {
                            #pragma omp critical
                            {
                                auto state_clone = current->getState().clone();
                                next_batch.emplace_back(current, std::move(state_clone), path);
                            }
                        }
                    }
                }
                
                // Wait for inference to complete
                outputs = inference_future.get();
                
                auto nn_end = std::chrono::steady_clock::now();
                auto nn_time_us = std::chrono::duration_cast<std::chrono::microseconds>(nn_end - nn_start).count();
                // debug_stats.total_inference_time_us.fetch_add(nn_time_us);
                // debug_stats.batches_processed.fetch_add(1);
                
                // DEBUG: Log inference time
                // if (simulations_completed % 100 == 0 || batch.size() != target_batch_size) {
                //     std::cout << "[DEBUG] NN inference completed in " << nn_time_us / 1000.0 
                //               << "ms for batch size " << states.size() 
                //               << " (target was " << target_batch_size << ")" << std::endl;
                // }
                
                // Dynamic batch manager removed
                /*
                if (dynamic_batch_manager_) {
                    dynamic_batch_manager_->updateMetrics(
                        states.size(), 
                        static_cast<float>(nn_time_us / 1000.0),
                        remaining_sims);
                }
                */
                
                // Store next batch for processing
                if (!next_batch.empty()) {
                    // Process next batch immediately after backpropagation
                    // Use move semantics since PendingEvaluation is move-only
                    batch.insert(batch.end(), std::make_move_iterator(next_batch.begin()), 
                                            std::make_move_iterator(next_batch.end()));
                }
            }
            
            // PHASE 3: Apply results and backpropagate
            auto backprop_start = std::chrono::steady_clock::now();
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
            
            auto backprop_end = std::chrono::steady_clock::now();
            auto backprop_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                backprop_end - backprop_start).count();
            // debug_stats.total_backprop_time_us.fetch_add(backprop_time_us);
            
            simulations_completed += batch.size();
            
            // DEBUG: Log complete batch timing every 10 batches
            // if (debug_stats.batches_processed.load() % 10 == 0) {
            //     auto batch_total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            //         std::chrono::steady_clock::now() - batch_start_total).count();
            //     
            //     int avg_collection_us = debug_stats.total_collection_time_us.load() / std::max(1, debug_stats.batches_processed.load());
            //     int avg_inference_us = debug_stats.total_inference_time_us.load() / std::max(1, debug_stats.batches_processed.load());
            //     int avg_backprop_us = debug_stats.total_backprop_time_us.load() / std::max(1, debug_stats.batches_processed.load());
            //     
            //     std::cout << "\n[DEBUG SUMMARY] After " << debug_stats.batches_processed.load() << " batches:\n"
            //               << "  Avg collection time: " << avg_collection_us / 1000.0 << "ms\n"
            //               << "  Avg inference time: " << avg_inference_us / 1000.0 << "ms\n"
            //               << "  Avg backprop time: " << avg_backprop_us / 1000.0 << "ms\n"
            //               << "  Avg batch size: " << (float)simulations_completed / debug_stats.batches_processed.load() << "\n"
            //               << "  Total nodes expanded: " << debug_stats.nodes_expanded.load() << "\n"
            //               << "  Total nodes skipped: " << debug_stats.nodes_skipped.load() << "\n"
            //               << std::endl;
            // }
            
            // Log GPU utilization info periodically
            static int total_batches = 0;
            static int total_batch_size = 0;
            total_batches++;
            total_batch_size += batch.size();
            
            if (total_batches % 10 == 0) {
                float avg_batch_size = static_cast<float>(total_batch_size) / total_batches;
                std::cout << "📊 Batch stats: avg size=" << avg_batch_size 
                         << ", efficiency=" << (avg_batch_size / target_batch_size * 100) << "%" << std::endl;
            }
            
            // Clear batch immediately to free memory
            batch.clear();
            states.clear();
            
            // CRITICAL FIX: Less frequent cleanup for 64GB system
            // Only clean up periodically without blocking
            if (simulations_completed % (batch_size * 50) == 0) {
                #ifdef WITH_TORCH
                if (torch::cuda::is_available()) {
                    // Non-blocking cleanup - let GPU continue working
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
    
    // Update statistics
    last_stats_.total_evaluations = simulations_completed;
    
    std::cout << "✅ SIMPLE BATCHED SEARCH completed: " << simulations_completed << " simulations" << std::endl;
    
    // Final cleanup without synchronization
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    #endif
}

} // namespace mcts
} // namespace alphazero