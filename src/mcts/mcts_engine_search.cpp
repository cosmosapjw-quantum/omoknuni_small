#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/mcts_evaluator.h"
#include "utils/debug_monitor.h"
#include "utils/debug_logger.h"
#include "utils/gamestate_pool.h"
#include <iostream>
#include <omp.h>
#include <thread>

namespace alphazero {
namespace mcts {

// Renamed from runSearch to better reflect its function
void MCTSEngine::runSearch(const core::IGameState& state) {
    try {
        // Sequential steps to initialize and run the search

        // Step 1: Create the root node with the current state
        root_ = createRootNode(state);
        
        // Step 2: Initialize game state pool if enabled
        initializeGameStatePool(state);
        
        // Step 3: Set up batch parameters for the evaluator
        setupBatchParameters();
        
        // Step 4: Expand the root node to prepare for search
        if (!root_->isTerminal()) {
            expandNonTerminalLeaf(root_);
        }
        
        // Step 5: Reset search statistics and prepare for new search
        resetSearchState();
        
        // Step 6: Create parallel search roots if root parallelization is enabled
        std::vector<std::shared_ptr<MCTSNode>> search_roots;
        if (settings_.use_root_parallelization && settings_.num_root_workers > 1) {
            search_roots = createSearchRoots(root_, settings_.num_root_workers);
        } else {
            search_roots.push_back(root_);
        }
        
        // Step 7: Execute the main search algorithm based on selected method
        // The main search loop - always use the serial approach with batching for better performance
        executeSerialSearch(search_roots);
        
        // Step 8: Aggregate results from different search roots if using root parallelization
        if (settings_.use_root_parallelization && settings_.num_root_workers > 1) {
            aggregateRootParallelResults(search_roots);
        }
        
        // Step 9: Update search statistics before returning
        countTreeStatistics();
        
    } catch (const std::exception& e) {
        // Log the error
        std::cerr << "Exception during MCTS search: " << e.what() << std::endl;
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow to allow caller to handle the error
        throw;
    } catch (...) {
        // Handle unknown exceptions
        std::cerr << "Unknown exception during MCTS search" << std::endl;
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow with a more descriptive message
        throw std::runtime_error("Unknown error occurred during MCTS search");
    }
}

// Initialize game state pool for better performance
void MCTSEngine::initializeGameStatePool(const core::IGameState& state) {
    if (game_state_pool_enabled_ && !utils::GameStatePoolManager::getInstance().hasPool(state.getGameType())) {
        try {
            // Initialize with reasonable defaults
            size_t pool_size = settings_.num_simulations / 4;  // Estimate based on simulations
            utils::GameStatePoolManager::getInstance().initializePool(state.getGameType(), pool_size);
        } catch (const std::exception& e) {
        }
    }
}

// Reset the search state for a new search
void MCTSEngine::resetSearchState() {
    // Initialize statistics for the new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;
    
    total_leaves_generated_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Set search running flag
    search_running_.store(true, std::memory_order_release);
    
    int sim_count = std::max(500, settings_.num_simulations);
    
    active_simulations_.store(0, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    active_simulations_.store(sim_count, std::memory_order_release);
    
    int actual_value = active_simulations_.load(std::memory_order_acquire);
    if (actual_value != sim_count) {
        active_simulations_ = sim_count;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        actual_value = active_simulations_.load(std::memory_order_acquire);
    }
    
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        active_simulations_.store(sim_count, std::memory_order_seq_cst);
    }
}

// Implementation of the serial search approach with leaf batching
void MCTSEngine::executeSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    // Track search start time
    auto search_start_time = std::chrono::steady_clock::now();
    
    // Thread-local leaf storage for batching
    std::vector<PendingEvaluation> leaf_batch;
    const size_t OPTIMAL_BATCH_SIZE = settings_.batch_size;
    leaf_batch.reserve(OPTIMAL_BATCH_SIZE);
    
    // Counters for monitoring
    int consecutive_empty_tries = 0;
    const int MAX_EMPTY_TRIES = 3;
    
    if (search_roots.empty()) {
        return;
    }
    
    if (!search_roots[0]) {
        return;
    }
    
    std::shared_ptr<MCTSNode> main_root = search_roots[0];
    
    if (!main_root->isTerminal()) {
        expandNonTerminalLeaf(main_root);
    }
    
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        active_simulations_.store(100, std::memory_order_release);
    }
    
    int main_loop_iterations = 0;
    
    while (active_simulations_.load(std::memory_order_acquire) > 0) {
        main_loop_iterations++;
        
        
        // Check if we should wait for pending evaluations to complete
        if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Aggressive batch collection - try to fill the entire batch quickly
        auto batch_start_time = std::chrono::steady_clock::now();
        const auto MAX_BATCH_COLLECTION_TIME = std::chrono::milliseconds(100); // Increased for proper batch accumulation
        
        // Force at least one iteration initially
        bool force_execution = true;
        
        while ((leaf_batch.size() < OPTIMAL_BATCH_SIZE && 
                active_simulations_.load(std::memory_order_acquire) > 0 &&
                (std::chrono::steady_clock::now() - batch_start_time) < MAX_BATCH_COLLECTION_TIME &&
                pending_evaluations_.load(std::memory_order_acquire) < settings_.batch_size * 6) || 
               force_execution) {
            
            // Reset force execution flag after first iteration
            force_execution = false;
            
            // Start a new simulation by claiming nodes from the counter
            int old_sims = active_simulations_.load(std::memory_order_acquire);
            
            if (old_sims <= 20 && total_leaves_generated_.load(std::memory_order_acquire) < 10) {
                active_simulations_.fetch_add(50, std::memory_order_release);
                old_sims = active_simulations_.load(std::memory_order_acquire);
            }
            
            // CRITICAL FIX: Claim much more simulations per iteration for proper batch accumulation
            // We need to get closer to the target batch size quickly
            int simulations_to_claim = std::min(static_cast<int>(OPTIMAL_BATCH_SIZE / 4), old_sims); // Claim 25% of target batch size
            simulations_to_claim = std::max(8, simulations_to_claim); // Minimum 8 simulations per iteration
            
            bool claimed = active_simulations_.compare_exchange_weak(
                old_sims, old_sims - simulations_to_claim, std::memory_order_acq_rel);
            
            if (!claimed) {
                if (active_simulations_.load(std::memory_order_acquire) <= 0 && 
                    total_leaves_generated_.load(std::memory_order_acquire) < 10 &&
                    std::chrono::steady_clock::now() - search_start_time < std::chrono::seconds(2)) {
                    
                    active_simulations_.store(20, std::memory_order_release);
                }
            }
            
            if (claimed) {
                int leaves_found = 0;
                
                // Execute the claimed simulations
                for (int i = 0; i < simulations_to_claim && leaf_batch.size() < OPTIMAL_BATCH_SIZE; ++i) {
                    try {
                        // Use round-robin selection of search roots
                        static int root_index = 0;
                        root_index = (root_index + 1) % search_roots.size();
                        std::shared_ptr<MCTSNode> current_root = search_roots[root_index];
                        
                        // Ensure the root is ready for traversal
                        if (!current_root->isFullyExpanded() && !current_root->isTerminal() && 
                            !current_root->hasPendingEvaluation() && !current_root->isBeingEvaluated()) {
                            expandNonTerminalLeaf(current_root);
                        }
                        
                        // Find a leaf node for evaluation
                        auto [leaf, path] = selectLeafNode(current_root);
                        
                        // Process the leaf node if valid
                        if (leaf) {
                            if (leaf->isTerminal()) {
                                // Process terminal nodes immediately
                                float value = 0.0f;
                                try {
                                    auto game_result = leaf->getState().getGameResult();
                                    int current_player = leaf->getState().getCurrentPlayer();
                                    if (game_result == core::GameResult::WIN_PLAYER1) {
                                        value = current_player == 1 ? 1.0f : -1.0f;
                                    } else if (game_result == core::GameResult::WIN_PLAYER2) {
                                        value = current_player == 2 ? 1.0f : -1.0f;
                                    }
                                } catch (...) {
                                    value = 0.0f;
                                }
                                backPropagate(path, value);
                            } else {
                                // Reset evaluation flags if needed
                                if (leaf->hasPendingEvaluation() || leaf->isBeingEvaluated()) {
                                    // Reset evaluation flag more frequently for non-expanded nodes
                                    if (leaf->hasPendingEvaluation() && !leaf->hasChildren()) {
                                        static int evaluation_reset_count = 0;
                                        evaluation_reset_count++;
                                        if (evaluation_reset_count % 10 == 0) {
                                            leaf->clearEvaluationFlag();
                                        }
                                    }
                                }
                                
                                // Request evaluation for non-terminal nodes
                                if (safelyMarkNodeForEvaluation(leaf)) {
                                    // Clone state for evaluation 
                                    auto state_clone = cloneGameState(leaf->getState());
                                    
                                    // Create pending evaluation
                                    PendingEvaluation pending;
                                    pending.node = leaf;
                                    pending.path = path;
                                    pending.state = std::move(state_clone);
                                    pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                                    pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                                    
                                    // Validate node and state
                                    if (!pending.state || !pending.node) {
                                        if (pending.node) {
                                            pending.node->clearAllEvaluationFlags();
                                        }
                                        continue;
                                    }
                                    
                                    // State validation before adding to batch
                                    bool state_valid = false;
                                    try {
                                        state_valid = pending.state->validate();
                                    } catch (...) {
                                        state_valid = false;
                                    }
                                    
                                    if (!state_valid) {
                                        pending.node->clearAllEvaluationFlags();
                                        continue;
                                    }
                                    
                                    // CRITICAL FIX: Collect evaluations in local batch instead of submitting individually
                                    // This is the core fix for batch accumulation - collect multiple evaluations 
                                    // before submitting them as a group to the shared queue
                                    leaf_batch.push_back(std::move(pending));
                                    leaves_found++;
                                    utils::debug_logger().trackItemProcessed();
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "❌ Error during leaf collection: " << e.what() << std::endl;
                        active_simulations_.fetch_add(1, std::memory_order_acq_rel); // Return the simulation
                    }
                }
                
                // Update consecutive empty tries counter
                if (leaves_found == 0) {
                    consecutive_empty_tries++;
                } else {
                    consecutive_empty_tries = 0;
                }
            }
        }
        
        if (!leaf_batch.empty()) {
            
            // Choose correct queue based on whether shared queues are configured
            auto& target_queue = use_shared_queues_ ? *shared_leaf_queue_ : leaf_queue_;
            
            const int MAX_SUBMISSION_ATTEMPTS = 5;
            size_t total_enqueued = 0;
            
            // Check queue size if using shared queues
            if (use_shared_queues_ && shared_leaf_queue_) {
                shared_leaf_queue_->size_approx();
            }
            
            // First submit in bulk if possible for better performance
            if (leaf_batch.size() > 1) {
                
                // IMPROVED: Use a temporary vector for moving items
                std::vector<PendingEvaluation> temp_batch;
                temp_batch.reserve(leaf_batch.size());
                
                // Deep copy to ensure we don't lose items if bulk enqueue fails
                for (auto& item : leaf_batch) {
                    temp_batch.push_back(std::move(item));
                }
                
                // Enqueue in bulk 
                size_t enqueued = target_queue.enqueue_bulk(
                    std::make_move_iterator(temp_batch.begin()), 
                    temp_batch.size());
                
                // Bulk enqueue completed
                
                total_enqueued = enqueued;
                
                // Check if bulk enqueue completed successfully
                if (enqueued == temp_batch.size()) {
                    // Clear the main batch since we've successfully moved all items
                    leaf_batch.clear();
                } else {
                    // Reset leaf_batch for remaining items
                    leaf_batch.clear();
                    
                    // Add back items that weren't enqueued successfully (starting from the enqueued position)
                    for (size_t i = enqueued; i < temp_batch.size(); i++) {
                        leaf_batch.push_back(std::move(temp_batch[i]));
                    }
                }
                
                // Notify after bulk enqueue
                if (use_shared_queues_ && external_queue_notify_fn_ && enqueued > 0) {
                    external_queue_notify_fn_();
                    
                    // Force duplicate notifications to ensure delivery 
                    for (int i = 0; i < 3; i++) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        external_queue_notify_fn_();
                    }
                } else if (!use_shared_queues_ && evaluator_ && enqueued > 0) {
                    evaluator_->notifyLeafAvailable();
                    
                    // Force duplicate notifications to ensure delivery
                    for (int i = 0; i < 3; i++) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        evaluator_->notifyLeafAvailable();
                    }
                }
            }
            
            // Process any remaining items individually
            // This covers both the case where bulk enqueue had leftover items 
            // and the case of single item submission
            if (!leaf_batch.empty()) {
                    
                size_t individual_enqueued = 0;
                
                for (size_t i = 0; i < leaf_batch.size(); i++) {
                    bool item_enqueued = false;
                    
                    // Try multiple times with increasing delays
                    for (int attempt = 0; attempt < MAX_SUBMISSION_ATTEMPTS; attempt++) {
                        // Using move semantics to avoid copy construction which is deleted
                        // We need to handle the case when enqueue fails and we need to try again
                        if (attempt == 0) {
                            // Only move on the first attempt
                            PendingEvaluation temp_eval = std::move(leaf_batch[i]);
                            
                            if (target_queue.enqueue(std::move(temp_eval))) {
                                individual_enqueued++;
                                total_enqueued++;
                                item_enqueued = true;
                                
                                
                                // Notify after each successful individual enqueue
                                if (use_shared_queues_ && external_queue_notify_fn_) {
                                    external_queue_notify_fn_();
                                } else if (!use_shared_queues_ && evaluator_) {
                                    evaluator_->notifyLeafAvailable();
                                }
                                
                                break; // Success, stop retry loop
                            } else {
                                // Enqueue failed, put the item back for retry
                                leaf_batch[i] = std::move(temp_eval);
                            }
                        } else {
                            // Create a new PendingEvaluation with a direct reference to the node and path
                            // This avoids the deleted copy constructor issue
                            PendingEvaluation retry_eval;
                            retry_eval.node = leaf_batch[i].node;
                            retry_eval.path = leaf_batch[i].path;
                            retry_eval.state = leaf_batch[i].state;
                            retry_eval.batch_id = leaf_batch[i].batch_id;
                            retry_eval.request_id = leaf_batch[i].request_id;
                            
                            if (target_queue.enqueue(std::move(retry_eval))) {
                                individual_enqueued++;
                                total_enqueued++;
                                item_enqueued = true;
                                
                                
                                // Notify after each successful individual enqueue
                                if (use_shared_queues_ && external_queue_notify_fn_) {
                                    external_queue_notify_fn_();
                                } else if (!use_shared_queues_ && evaluator_) {
                                    evaluator_->notifyLeafAvailable();
                                }
                                
                                // Clear the original item to avoid double processing
                                leaf_batch[i].node.reset();
                                leaf_batch[i].path.clear();
                                leaf_batch[i].state.reset();
                                
                                break; // Success, stop retry loop
                            }
                        }
                        
                        if (attempt < MAX_SUBMISSION_ATTEMPTS - 1) {
                            // IMPROVED: Exponential backoff with increasing delays
                            int delay_ms = (1 << attempt); // 1, 2, 4, 8, 16 ms
                            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                            
                        }
                    }
                    
                    if (!item_enqueued) {
                        // IMPROVED: Clear evaluation flag for failed items to avoid leaks
                        if (leaf_batch[i].node) {
                            leaf_batch[i].node->clearEvaluationFlag();
                        }
                    }
                }
                
            }
            
            // Update pending evaluations count
            pending_evaluations_.fetch_add(total_enqueued, std::memory_order_acq_rel);
            
            // ENHANCED: Extra notifications with delay to ensure evaluator processes items
            if (total_enqueued > 0) {
                // Notify external queue after submission
                if (use_shared_queues_ && shared_leaf_queue_) {
                    // ENHANCED: Sequence of notifications with delays
                    if (external_queue_notify_fn_) {
                        
                        // First notification
                        external_queue_notify_fn_();
                        
                        // Wait and send more notifications to ensure delivery
                        for (int i = 0; i < 3; i++) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            external_queue_notify_fn_();
                        }
                    }
                } else if (!use_shared_queues_ && evaluator_) {
                    // Send multiple notifications with delays
                    for (int i = 0; i < 4; i++) {
                        evaluator_->notifyLeafAvailable();
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                }
            }
            
            // Clear the batch
            leaf_batch.clear();
            consecutive_empty_tries = 0;
            
            // ENHANCED: Small delay after submission to allow evaluator to start processing
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        // Process results directly when using shared queues to prevent deadlock
        processPendingSimulations();
        
        // Adaptive wait based on pending evaluations
        if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 3) {
            // If too many pending, wait longer to prevent memory overflow
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } else if (leaf_batch.empty() && consecutive_empty_tries >= MAX_EMPTY_TRIES) {
            // If we can't find leaves, check if we're done
            if (active_simulations_.load(std::memory_order_acquire) == 0) {
                break;  // Exit the loop
            }
            std::this_thread::yield();
        }
    }
    
    // Wait for any remaining evaluations to complete
    waitForSimulationsToComplete(search_start_time);
    
    // Record total search time
    auto search_end_time = std::chrono::steady_clock::now();
    last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end_time - search_start_time);
    
    // Update evaluator stats
    if (evaluator_) {
        last_stats_.avg_batch_size = evaluator_->getAverageBatchSize();
        last_stats_.avg_batch_latency = evaluator_->getAverageBatchLatency();
        last_stats_.total_evaluations = evaluator_->getTotalEvaluations();
    }
    
    // Mark search as completed
    search_running_.store(false, std::memory_order_release);
}

} // namespace mcts
} // namespace alphazero