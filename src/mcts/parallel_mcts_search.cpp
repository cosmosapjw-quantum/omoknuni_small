#include "mcts/mcts_engine.h"
#include "mcts/advanced_memory_pool.h"
#include "mcts/mcts_object_pool.h"
#include <algorithm>
#include <chrono>
#include <omp.h>

namespace alphazero {
namespace mcts {

MCTSEngine::ParallelSearchResult MCTSEngine::executeParallelSimulations(
    const std::vector<std::shared_ptr<MCTSNode>>& search_roots, 
    int target_simulations) {
    
    ParallelSearchResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int num_threads = settings_.num_threads;
    const int simulations_per_thread = target_simulations / num_threads;
    
    std::vector<std::vector<PendingEvaluation>> thread_batches(num_threads);
    std::vector<std::vector<std::shared_ptr<MCTSNode>>> thread_expanded_nodes(num_threads);
    std::vector<int> thread_simulations_completed(num_threads, 0);
    std::vector<int> thread_terminal_nodes(num_threads, 0);
    std::vector<int> thread_virtual_loss_apps(num_threads, 0);
    
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int thread_target = (thread_id == num_threads - 1) ? 
            target_simulations - (simulations_per_thread * (num_threads - 1)) : simulations_per_thread;
        
        std::mt19937 thread_rng(random_engine_() + thread_id * 1000);
        std::vector<std::shared_ptr<MCTSNode>> thread_path;
        thread_path.reserve(100); // Pre-allocate for typical game depth
        
        auto& thread_batch = thread_batches[thread_id];
        thread_batch.reserve(settings_.batch_params.max_collection_batch_size);
        
        for (int sim = 0; sim < thread_target; ++sim) {
            // Select root node (distribute across available roots)
            auto root = search_roots[sim % search_roots.size()];
            
            // Perform optimized parallel leaf selection
            auto leaf_result = selectLeafNodeParallel(root, thread_path, thread_rng);
            
            if (leaf_result.leaf_node) {
                thread_expanded_nodes[thread_id].push_back(leaf_result.leaf_node);
                
                if (leaf_result.terminal) {
                    // Handle terminal nodes immediately
                    backpropagateParallel(leaf_result.path, leaf_result.terminal_value, settings_.virtual_loss);
                    thread_terminal_nodes[thread_id]++;
                } else if (leaf_result.needs_evaluation) {
                    // Create optimized evaluation request
                    PendingEvaluation eval;
                    eval.node = leaf_result.leaf_node;
                    eval.path = leaf_result.path;
                    // Create a clone of the state using the appropriate conversion
                    if (leaf_result.leaf_node) {
                        eval.state = leaf_result.leaf_node->getState().clone();
                    }
                    eval.batch_id = batch_counter_.fetch_add(1);
                    eval.request_id = thread_id * 100000 + sim;
                    
                    thread_batch.push_back(std::move(eval));
                }
                
                if (leaf_result.applied_virtual_loss) {
                    thread_virtual_loss_apps[thread_id] += leaf_result.path.size();
                }
            }
            
            thread_simulations_completed[thread_id]++;
            
            // Process batch when full or near end
            if (thread_batch.size() >= settings_.batch_params.max_collection_batch_size ||
                sim == thread_target - 1) {
                processParallelBatch(thread_batch, thread_id);
                thread_batch.clear();
            }
            
            thread_path.clear(); // Reuse vector memory
        }
    }
    
    // Aggregate results from all threads
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    for (int i = 0; i < num_threads; ++i) {
        result.simulations_completed += thread_simulations_completed[i];
        result.terminal_nodes_processed += thread_terminal_nodes[i];
        result.virtual_loss_applications += thread_virtual_loss_apps[i];
        
        // Collect expanded nodes
        result.expanded_nodes.insert(result.expanded_nodes.end(),
                                   thread_expanded_nodes[i].begin(),
                                   thread_expanded_nodes[i].end());
        
        // Collect remaining evaluation requests using move semantics
        for (auto& eval : thread_batches[i]) {
            result.evaluation_requests.push_back(std::move(eval));
        }
    }
    
    return result;
}

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
        result.needs_evaluation = false;
    } else {
        // Non-terminal leaf needs expansion and evaluation
        result.terminal = false;
        result.needs_evaluation = true;
        
        // Expand the node if not already expanded
        if (!current->isExpanded()) {
            expandNonTerminalLeaf(current);
        }
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

void MCTSEngine::processParallelBatch(
    std::vector<PendingEvaluation>& batch, 
    int thread_id) {
    
    if (batch.empty()) return;
    
    if (burst_coordinator_) {
        // Use burst coordinator for optimized batching
        std::vector<BurstCoordinator::BurstRequest> requests;
        requests.reserve(batch.size());
        
        for (auto& eval : batch) {
            BurstCoordinator::BurstRequest request;
            request.node = eval.node;
            request.leaf = eval.node;
            request.path = eval.path;
            if (eval.state) {
                request.state = std::move(eval.state);
            }
            request.priority = BurstCoordinator::Priority::Normal;
            request.callback = [node = eval.node, path = eval.path](const NetworkOutput& output) {
                // Set network output and trigger backpropagation
                if (node) {
                    node->updateStats(output.value);
                    // Note: Backpropagation will be handled by the burst coordinator
                }
            };
            requests.push_back(std::move(request));
        }
        
        burst_coordinator_->submitBurst(std::move(requests));
    } else {
        // Fallback to traditional queue-based processing
        for (auto& eval : batch) {
            leaf_queue_.enqueue(std::move(eval));
        }
        pending_evaluations_.fetch_add(batch.size());
    }
}

void MCTSEngine::executeOptimizedSearchV2(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty()) return;
    
    // Execute parallel simulations with enhanced memory management
    auto search_result = executeParallelSimulations(search_roots, settings_.num_simulations);
    
    // Wait for all evaluation requests to complete
    waitForParallelEvaluationCompletion(search_result.evaluation_requests);
    
    // Update search statistics
    updateParallelSearchStats(search_result);
}

void MCTSEngine::waitForParallelEvaluationCompletion(
    const std::vector<PendingEvaluation>& evaluation_requests) {
    
    if (evaluation_requests.empty()) return;
    
    const auto max_wait_time = std::chrono::seconds(60);
    auto start_time = std::chrono::steady_clock::now();
    auto deadline = start_time + max_wait_time;
    
    while (std::chrono::steady_clock::now() < deadline) {
        bool all_complete = true;
        
        if (burst_coordinator_) {
            all_complete = burst_coordinator_->allRequestsComplete();
        } else {
            all_complete = (pending_evaluations_.load() == 0);
        }
        
        if (all_complete) {
            break;
        }
        
        // Adaptive sleep - shorter sleeps when close to completion
        auto remaining_time = deadline - std::chrono::steady_clock::now();
        auto sleep_duration = std::min(
            std::chrono::microseconds(500),
            std::chrono::duration_cast<std::chrono::microseconds>(remaining_time / 100)
        );
        
        if (sleep_duration > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }
}

void MCTSEngine::updateParallelSearchStats(const ParallelSearchResult& search_result) {
    last_stats_.total_nodes = search_result.simulations_completed;
    last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(search_result.elapsed_time);
    
    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = (float)search_result.simulations_completed * 1000.0f / last_stats_.search_time.count();
    }
    
    // Update advanced statistics
    last_stats_.max_depth = calculateMaxDepth(root_);
    last_stats_.total_evaluations = search_result.evaluation_requests.size();
    
    // Memory pool statistics
    if (memory_pool_) {
        auto pool_stats = memory_pool_->getStats();
        last_stats_.pool_hit_rate = static_cast<float>(pool_stats.total_allocations) / 
                                   (static_cast<float>(pool_stats.total_allocations + pool_stats.total_deallocations) + 1.0f);
        last_stats_.pool_size = pool_stats.nodes_available + pool_stats.states_available;
        last_stats_.pool_total_allocated = pool_stats.total_allocations;
    }
}

} // namespace mcts
} // namespace alphazero
