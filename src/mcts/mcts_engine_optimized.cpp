#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/unified_inference_server.h"
#include "mcts/burst_coordinator.h"
#include "mcts/mcts_object_pool.h"
#include "core/game_export.h"
#include <chrono>
#include <thread>
#include <future>

namespace alphazero {
namespace mcts {

SearchResult MCTSEngine::executeOptimizedSearch(std::shared_ptr<core::IGameState> root_state) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Use singleton object pool manager for optimized allocations
    auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
    
    // Ensure UnifiedInferenceServer is running with optimal configuration
    if (!unified_inference_server_) {
        UnifiedInferenceServer::ServerConfig config;
        config.target_batch_size = settings_.batch_size;
        config.min_batch_size = std::max(4, settings_.batch_size / 8);
        config.max_batch_size = settings_.batch_size * 3;
        config.max_batch_wait = std::chrono::milliseconds(static_cast<int>(settings_.batch_timeout.count()));
        config.min_batch_wait = std::chrono::milliseconds(1);
        config.num_worker_threads = std::min(8, settings_.num_threads / 2);
        
        unified_inference_server_ = std::make_shared<UnifiedInferenceServer>(neural_network_, config);
        unified_inference_server_->start();
    }
    
    // Initialize BurstCoordinator for optimal batch collection
    if (!burst_coordinator_) {
        BurstCoordinator::BurstConfig burst_config;
        burst_config.target_burst_size = settings_.batch_size;
        burst_config.min_burst_size = std::max(4, settings_.batch_size / 8);
        burst_config.collection_timeout = std::chrono::milliseconds(std::max(5, static_cast<int>(settings_.batch_timeout.count()) / 4));
        burst_config.evaluation_timeout = std::chrono::milliseconds(static_cast<int>(settings_.batch_timeout.count()));
        burst_config.max_parallel_threads = settings_.num_threads;
        
        burst_coordinator_ = std::make_unique<BurstCoordinator>(
            unified_inference_server_, burst_config);
    }
    
    // Create root node using object pool manager
    auto* raw_root_node = object_pool_manager.getNodePool().acquire();
    auto root_node = std::shared_ptr<MCTSNode>(raw_root_node, [&object_pool_manager](MCTSNode* node) {
        object_pool_manager.getNodePool().release(node);
    });
    
    // Initialize root node with one visit
    root_node->updateRecursive(0.0f); // Neutral initial value
    
    // Expand root node for search initialization
    if (!root_state->isTerminal()) {
        root_node->expand(false, 
                         settings_.progressive_widening_c, 
                         settings_.progressive_widening_k);
    }
    
    // Execute optimized search with advanced parallelization
    int target_simulations = settings_.num_simulations;
    int simulations_completed = 0;
    
    // Performance tracking
    std::vector<std::chrono::microseconds> batch_times;
    std::vector<size_t> batch_sizes;
    
    // Main optimized search loop
    while (simulations_completed < target_simulations) {
        int remaining_simulations = target_simulations - simulations_completed;
        int batch_target = std::min(remaining_simulations, settings_.batch_size * 2);
        
        auto batch_start = std::chrono::steady_clock::now();
        
        // Execute enhanced burst search
        auto batch_results = executeEnhancedBurstSearch(*root_state, root_node, batch_target);
        
        auto batch_end = std::chrono::steady_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
        
        // Record performance metrics
        batch_times.push_back(batch_duration);
        batch_sizes.push_back(batch_results);
        
        simulations_completed += batch_results;
        
        // Adaptive optimization: adjust strategy based on performance
        if (batch_times.size() >= 3) {
            auto avg_time = std::accumulate(batch_times.end() - 3, batch_times.end(), 
                                          std::chrono::microseconds(0)) / 3;
            auto avg_size = std::accumulate(batch_sizes.end() - 3, batch_sizes.end(), 0UL) / 3;
            
            // If performance is degrading, adjust coordination strategy
            if (avg_time > std::chrono::milliseconds(200) && avg_size < settings_.batch_size / 2) {
                // Trigger burst coordinator optimization
                auto current_config = burst_coordinator_->getConfig();
                auto new_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(current_config.collection_timeout);
                current_config.collection_timeout = std::chrono::milliseconds(std::max(1, static_cast<int>(new_timeout.count()) - 1));
                burst_coordinator_->updateConfig(current_config);
            }
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Generate optimized search result
    SearchResult result;
    result.probabilities = getActionProbabilities(root_node, settings_.temperature);
    result.value = root_node->getValue();
    
    // Select best action based on visit counts (more robust than just probabilities)
    auto children = root_node->getChildren();
    int best_action = 0;
    int max_visits = 0;
    
    for (size_t i = 0; i < children.size(); ++i) {
        int child_visits = children[i]->getVisitCount();
        if (child_visits > max_visits) {
            max_visits = child_visits;
            best_action = children[i]->getAction();
        }
    }
    result.action = best_action;
    
    // Comprehensive optimized statistics
    result.stats.total_nodes = root_node->getVisitCount();
    result.stats.search_time = total_time;
    result.stats.total_evaluations = simulations_completed;
    result.stats.nodes_per_second = static_cast<float>(simulations_completed) / 
                                   (static_cast<float>(total_time.count()) / 1000.0f);
    
    // Object pool efficiency statistics from singleton manager
    auto pool_stats = object_pool_manager.getNodePool().getStats();
    result.stats.pool_hit_rate = static_cast<float>(pool_stats.pool_hits) / 
                                static_cast<float>(pool_stats.total_allocations);
    result.stats.pool_size = pool_stats.current_pool_size;
    result.stats.pool_total_allocated = pool_stats.total_allocations;
    
    // Unified inference server performance statistics
    if (unified_inference_server_) {
        auto server_stats = unified_inference_server_->getStats();
        result.stats.avg_batch_size = server_stats.getAverageBatchSize();
        result.stats.avg_batch_latency = std::chrono::milliseconds(static_cast<int64_t>(server_stats.getAverageBatchLatency()));
        result.stats.total_requests_processed = server_stats.total_requests;
        result.stats.total_batches_processed = server_stats.total_batches;
    }
    
    // Burst coordinator efficiency statistics  
    if (burst_coordinator_) {
        auto burst_stats = burst_coordinator_->getEfficiencyStats();
        result.stats.burst_efficiency = burst_stats.average_collection_efficiency;
        result.stats.burst_utilization = burst_stats.target_utilization_rate;
        result.stats.avg_batch_size = burst_stats.average_burst_size;
    }
    
    // Transposition table statistics
    if (transposition_table_) {
        result.stats.tt_hit_rate = transposition_table_->hitRate();
        result.stats.tt_size = transposition_table_->size();
    }
    
    // Tree expansion quality metrics
    result.stats.tree_depth = calculateMaxDepth(root_node);
    result.stats.tree_branching_factor = calculateAverageBranchingFactor(root_node);
    result.stats.exploration_efficiency = calculateExplorationEfficiency(root_node);
    
    return result;
}

size_t MCTSEngine::executeEnhancedBurstSearch(const core::IGameState& root_state, 
                                             std::shared_ptr<MCTSNode> root_node,
                                             int target_simulations) {
    std::vector<BurstCoordinator::BurstRequest> requests;
    requests.reserve(target_simulations);
    
    // Collect evaluation candidates using advanced tree traversal
    for (int sim = 0; sim < target_simulations; ++sim) {
        // Select path with optimized UCB + virtual loss
        auto selected_leaf = selectOptimizedLeafNode(root_node);
        if (!selected_leaf.first) {
            break; // Tree exhausted
        }
        
        // Create burst request
        BurstCoordinator::BurstRequest request;
        request.leaf = selected_leaf.first;
        request.node = selected_leaf.first;
        request.path = selected_leaf.second;
        
        // Create game state for node
        if (selected_leaf.first) {
            request.state = selected_leaf.first->getState().clone();
        } else {
            // Fallback to root state clone if node is not available
            request.state = root_state.clone();
        }
        
        requests.push_back(std::move(request));
    }
    
    if (requests.empty()) {
        return 0;
    }
    
    // Execute coordinated burst evaluation
    auto results = burst_coordinator_->collectAndEvaluate(requests, requests.size());
    
    // Apply results with enhanced backpropagation
    for (size_t i = 0; i < results.size() && i < requests.size(); ++i) {
        auto& leaf = requests[i].leaf;
        auto& result = results[i];
        
        // Remove virtual loss applied during selection
        leaf->removeVirtualLoss(settings_.virtual_loss);
        
        // Apply evaluation result
        leaf->updateRecursive(result.value);
        
        // Expand node if it's not terminal and not already expanded
        if (!leaf->isExpanded() && !requests[i].state->isTerminal()) {
            leaf->expand(false,
                        settings_.progressive_widening_c,
                        settings_.progressive_widening_k);
        }
    }
    
    return results.size();
}

std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
MCTSEngine::selectOptimizedLeafNode(std::shared_ptr<MCTSNode> root) {
    std::vector<std::shared_ptr<MCTSNode>> path;
    auto current = root;
    
    while (current && current->isExpanded() && !current->isTerminal()) {
        path.push_back(current);
        
        // Apply virtual loss for thread collision prevention
        current->applyVirtualLoss(settings_.virtual_loss);
        
        // Select best child using optimized UCB
        auto next_child = current->selectChild(settings_.exploration_constant, 
                                             true, // use virtual loss
                                             settings_.virtual_loss);
        
        if (!next_child) {
            // Remove virtual loss if selection failed
            current->removeVirtualLoss(settings_.virtual_loss);
            break;
        }
        
        current = next_child;
    }
    
    if (current && (!current->isExpanded() || current->isTerminal())) {
        path.push_back(current);
        return {current, path};
    }
    
    return {nullptr, {}};
}

float MCTSEngine::calculateExplorationEfficiency(std::shared_ptr<MCTSNode> root) {
    if (!root || root->getChildren().empty()) {
        return 0.0f;
    }
    
    auto children = root->getChildren();
    int total_visits = 0;
    int explored_children = 0;
    
    for (const auto& child : children) {
        total_visits += child->getVisitCount();
        if (child->getVisitCount() > 0) {
            explored_children++;
        }
    }
    
    if (children.empty()) {
        return 0.0f;
    }
    
    float exploration_ratio = static_cast<float>(explored_children) / children.size();
    float visit_distribution = total_visits > 0 ? 
        (static_cast<float>(total_visits) / (explored_children > 0 ? explored_children : 1)) / 
        static_cast<float>(total_visits) : 0.0f;
    
    return exploration_ratio * (1.0f + visit_distribution);
}


float MCTSEngine::calculateAverageBranchingFactor(std::shared_ptr<MCTSNode> root) {
    if (!root) {
        return 0.0f;
    }
    
    std::function<std::pair<float, int>(std::shared_ptr<MCTSNode>)> 
        calculateBranching = [&](std::shared_ptr<MCTSNode> node) -> std::pair<float, int> {
        if (!node || node->getChildren().empty()) {
            return {0.0f, 0};
        }
        
        auto children = node->getChildren();
        int active_children = 0;
        float total_branching = 0.0f;
        int total_nodes = 1;
        
        for (const auto& child : children) {
            if (child->getVisitCount() > 0) {
                active_children++;
                auto child_result = calculateBranching(child);
                total_branching += child_result.first;
                total_nodes += child_result.second;
            }
        }
        
        return {total_branching + active_children, total_nodes};
    };
    
    auto result = calculateBranching(root);
    return result.second > 0 ? result.first / result.second : 0.0f;
}

} // namespace mcts
} // namespace alphazero