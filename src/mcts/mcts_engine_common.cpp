#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/mcts_evaluator.h"
#include "utils/debug_monitor.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace alphazero {
namespace mcts {

// Common function to handle MCTS statistics calculation
void MCTSEngine::countTreeStatistics() {
    if (root_) {
        last_stats_.total_nodes = countTreeNodes(root_);
        last_stats_.max_depth = calculateMaxDepth(root_);
    }

    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = 1000.0f * last_stats_.total_nodes / 
                                      std::max(1, static_cast<int>(last_stats_.search_time.count()));
    }

    // Add transposition table stats if enabled
    if (use_transposition_table_ && transposition_table_) {
        last_stats_.tt_hit_rate = transposition_table_->hitRate();
        last_stats_.tt_size = transposition_table_->size();
    }
}

// Recursive function to count nodes in the tree
size_t MCTSEngine::countTreeNodes(std::shared_ptr<MCTSNode> node) {
    if (!node) return 0;
    
    size_t count = 1; // Count this node
    for (auto child : node->getChildren()) {
        if (child) {
            count += countTreeNodes(child);
        }
    }
    return count;
}

// Recursive function to calculate maximum depth of the tree
int MCTSEngine::calculateMaxDepth(std::shared_ptr<MCTSNode> node) {
    if (!node) return 0;
    if (node->getChildren().empty()) return 0;
    
    int max_depth = 0;
    for (auto child : node->getChildren()) {
        if (child) {
            max_depth = std::max(max_depth, calculateMaxDepth(child) + 1);
        }
    }
    return max_depth;
}

// Process simulations with proper error handling
void MCTSEngine::processPendingSimulations() {
    // Only process if the search is running
    if (!search_running_.load(std::memory_order_acquire)) {
        return;
    }
    
    // Process results from shared queues if configured
    if (use_shared_queues_ && shared_result_queue_) {
        std::pair<NetworkOutput, PendingEvaluation> result;
        int processed = 0;
        
        // Process all available results
        while (shared_result_queue_->try_dequeue(result)) {
            auto& output = result.first;
            auto& eval = result.second;
            
            if (eval.node) {
                try {
                    eval.node->setPriorProbabilities(output.policy);
                    backPropagate(eval.path, output.value);
                    eval.node->clearEvaluationFlag();
                    processed++;
                } catch (const std::exception& e) {
                    std::cerr << "ERROR processing evaluation result: " << e.what() << std::endl;
                    // Try to clean up even if processing fails
                    try { eval.node->clearEvaluationFlag(); } catch (...) {}
                } catch (...) {
                    std::cerr << "ERROR: Unknown exception processing evaluation result" << std::endl;
                    // Try to clean up even if processing fails
                    try { eval.node->clearEvaluationFlag(); } catch (...) {}
                }
            }
            
            pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

// Wait for any pending evaluations to complete
void MCTSEngine::waitForSimulationsToComplete(std::chrono::steady_clock::time_point start_time) {
    // Wait for pending evaluations to complete with timeout
    const auto timeout = std::chrono::seconds(5);
    
    int wait_log_count = 0;
    while (pending_evaluations_.load(std::memory_order_acquire) > 0) {
        if (wait_log_count < 10 || wait_log_count % 100 == 0) {
            std::cout << "Waiting for " << pending_evaluations_.load(std::memory_order_acquire) 
                     << " pending evaluations to complete..." << std::endl;
        }
        wait_log_count++;
        
        // Check for timeout
        if (std::chrono::steady_clock::now() - start_time > timeout) {
            std::cout << "Timeout waiting for pending evaluations, continuing..." << std::endl;
            break;
        }
        
        // Process results directly when using shared queues
        processPendingSimulations();
        
        // Notify evaluator to process remaining items
        if (use_shared_queues_ && external_queue_notify_fn_) {
            external_queue_notify_fn_();
        } else if (!use_shared_queues_ && evaluator_) {
            evaluator_->notifyLeafAvailable();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Helper method for adaptive waiting with backoff
void MCTSEngine::waitWithBackoff(std::function<bool()> predicate, std::chrono::milliseconds max_wait_time) {
    // First try with quick yields - efficient for low contention
    for (int i = 0; i < 10; ++i) {
        if (predicate()) return;
        std::this_thread::yield();
    }
    
    // Then use exponential backoff with increasingly longer sleeps
    for (int wait_us = 100; wait_us < 10000; wait_us *= 2) {
        if (predicate()) return;
        std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
    }
    
    // If still not satisfied, sleep for longer periods
    auto deadline = std::chrono::steady_clock::now() + max_wait_time;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace mcts
} // namespace alphazero