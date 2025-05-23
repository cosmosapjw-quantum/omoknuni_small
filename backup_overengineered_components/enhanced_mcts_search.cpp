#include "mcts/mcts_engine.h"
#include "mcts/burst_coordinator.h"
#include "mcts/unified_inference_server.h"
#include <algorithm>
#include <chrono>
#include <omp.h>

namespace alphazero {
namespace mcts {

void MCTSEngine::executeEnhancedSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty()) return;
    
    // Initialize enhanced search statistics
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<int> completed_simulations{0};
    std::atomic<int> total_virtual_loss_applications{0};
    
    const int target_simulations = settings_.num_simulations;
    const int num_threads = settings_.num_threads;
    const int simulations_per_thread = target_simulations / num_threads;
    
    // Enhanced parallel search with virtual loss optimization
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int thread_simulations = (thread_id == num_threads - 1) ? 
            target_simulations - (simulations_per_thread * (num_threads - 1)) : simulations_per_thread;
        
        std::mt19937 thread_rng(random_engine_() + thread_id);
        std::vector<PendingEvaluation> thread_batch;
        thread_batch.reserve(settings_.batch_params.max_collection_batch_size);
        
        for (int sim = 0; sim < thread_simulations; ++sim) {
            // Select root for this simulation (round-robin)
            auto root = search_roots[sim % search_roots.size()];
            
            // Enhanced leaf selection with improved virtual loss handling
            auto [leaf, path] = selectLeafNodeWithVirtualLoss(root, thread_rng);
            
            if (leaf && !leaf->isTerminal()) {
                // Create optimized evaluation request
                PendingEvaluation eval;
                eval.node = leaf;
                eval.path = path;
                // Clone the state for evaluation
                eval.state = leaf->getState().clone();
                
                thread_batch.push_back(std::move(eval));
                
                // Submit batch when optimal size reached or near end of simulations
                if (thread_batch.size() >= settings_.batch_params.max_collection_batch_size ||
                    sim == thread_simulations - 1) {
                    
                    submitBatchForEvaluation(std::move(thread_batch));
                    thread_batch.clear();
                    thread_batch.reserve(settings_.batch_params.max_collection_batch_size);
                }
                
                total_virtual_loss_applications.fetch_add(path.size());
            } else if (leaf && leaf->isTerminal()) {
                // Handle terminal nodes immediately
                float terminal_value = leaf->getState().getGameResult() == core::GameResult::WIN_PLAYER1 ? 1.0f : 
                                     (leaf->getState().getGameResult() == core::GameResult::WIN_PLAYER2 ? -1.0f : 0.0f);
                backPropagate(path, terminal_value);
            }
            
            completed_simulations.fetch_add(1);
        }
        
        // Submit any remaining evaluations
        if (!thread_batch.empty()) {
            submitBatchForEvaluation(std::move(thread_batch));
        }
    }
    
    // Wait for all evaluations to complete with enhanced monitoring
    waitForEvaluationCompletion(start_time);
    
    // Update search statistics
    auto end_time = std::chrono::steady_clock::now();
    last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    last_stats_.total_nodes = completed_simulations.load();
    
    // Calculate enhanced performance metrics
    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = (float)completed_simulations.load() * 1000.0f / last_stats_.search_time.count();
    }
}

std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
MCTSEngine::selectLeafNodeWithVirtualLoss(std::shared_ptr<MCTSNode> root, std::mt19937& rng) {
    std::vector<std::shared_ptr<MCTSNode>> path;
    auto current = root;
    
    while (current && !current->isLeaf()) {
        path.push_back(current);
        
        // Apply virtual loss to prevent thread collisions
        current->applyVirtualLoss(settings_.virtual_loss);
        
        // Enhanced UCB selection with virtual loss consideration
        current = current->selectBestChildUCB(settings_.exploration_constant, rng);
        
        if (!current) {
            // Revert virtual loss if selection failed
            for (auto& node : path) {
                node->revertVirtualLoss(settings_.virtual_loss);
            }
            return {nullptr, {}};
        }
    }
    
    if (current) {
        path.push_back(current);
        // Apply virtual loss to leaf node
        current->applyVirtualLoss(settings_.virtual_loss);
    }
    
    return {current, path};
}

void MCTSEngine::submitBatchForEvaluation(std::vector<PendingEvaluation>&& batch) {
    if (batch.empty()) return;
    
    if (burst_coordinator_) {
        // Use enhanced burst coordinator for optimal batching
        std::vector<BurstCoordinator::BurstRequest> requests;
        requests.reserve(batch.size());
        
        for (auto& eval : batch) {
            BurstCoordinator::BurstRequest request;
            // Clone the state to a unique_ptr for the request
            request.state = eval.state ? eval.state->clone() : nullptr;
            request.node = eval.node;
            request.path = eval.path;
            requests.push_back(std::move(request));
        }
        
        burst_coordinator_->submitBurst(std::move(requests));
    } else {
        // Fallback to traditional queue-based evaluation
        for (auto& eval : batch) {
            leaf_queue_.enqueue(std::move(eval));
        }
    }
}

void MCTSEngine::waitForEvaluationCompletion(std::chrono::steady_clock::time_point start_time) {
    const auto max_wait_time = std::chrono::seconds(30); // Safety timeout
    auto deadline = start_time + max_wait_time;
    
    while (std::chrono::steady_clock::now() < deadline) {
        bool all_complete = true;
        
        if (burst_coordinator_) {
            // Check burst coordinator status
            all_complete = burst_coordinator_->allRequestsComplete();
        } else {
            // Check traditional queue status
            all_complete = (pending_evaluations_.load() == 0);
        }
        
        if (all_complete) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

} // namespace mcts
} // namespace alphazero
