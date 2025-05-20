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
    std::cout << "MCTSEngine::runSearch - Starting MCTS search with num_simulations=" 
              << settings_.num_simulations << std::endl;

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
            std::cout << "Initialized GameState pool with size " << pool_size << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize GameState pool: " << e.what() << std::endl;
            // Continue without pooling
        }
    }
}

// Reset the search state for a new search
void MCTSEngine::resetSearchState() {
    // Initialize statistics for the new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;
    
    // CRITICAL FIX: Reset counters to ensure a clean starting state
    total_leaves_generated_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Set search running flag
    search_running_.store(true, std::memory_order_release);
    
    // Reset active simulations counter with a reasonable simulation count
    // Use the configured number of simulations but ensure a minimum to prevent stalling
    int sim_count = std::max(500, settings_.num_simulations);
    std::cout << "MCTSEngine::resetSearchState - Setting active_simulations_ to " << sim_count 
              << " (from config value: " << settings_.num_simulations << ")" << std::endl;
    
    // Make sure the counter is properly reset using a reliable sequence
    active_simulations_.store(0, std::memory_order_release); // First reset to 0
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Short memory barrier
    active_simulations_.store(sim_count, std::memory_order_release); // Then set to desired value
    
    // Double-check that active_simulations_ was set correctly
    int actual_value = active_simulations_.load(std::memory_order_acquire);
    if (actual_value != sim_count) {
        std::cout << "⚠️ WARNING: MCTSEngine::resetSearchState - Failed to set active_simulations_ correctly! "
                  << "Expected: " << sim_count << ", Actual: " << actual_value << std::endl;
        
        // Try with a direct assignment and memory fence
        active_simulations_ = sim_count;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        // Verify again
        actual_value = active_simulations_.load(std::memory_order_acquire);
        std::cout << "MCTSEngine::resetSearchState - After retry: active_simulations_ = " 
                  << actual_value << std::endl;
    }
    
    // Verify that we have a positive number of simulations
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        std::cout << "⚠️ CRITICAL ERROR: Active simulations count is still <= 0! Forcing to " << sim_count << std::endl;
        active_simulations_.store(sim_count, std::memory_order_seq_cst);
    }
}

// Implementation of the serial search approach with leaf batching
void MCTSEngine::executeSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    std::cout << "=============================================================" << std::endl;
    std::cout << "MCTSEngine::executeSerialSearch - Starting serial traversal with batching" << std::endl;
    std::cout << "  Root parallelization: " << (settings_.use_root_parallelization ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Number of search roots: " << search_roots.size() << std::endl;
    std::cout << "  Current/target simulations: " << active_simulations_.load() << "/" << settings_.num_simulations << std::endl;
    std::cout << "  Batch size: " << settings_.batch_size << std::endl;
    std::cout << "  Thread setup: OMP threads=" << omp_get_max_threads() << std::endl;
    std::cout << "=============================================================" << std::endl;
    
    // Track search start time
    auto search_start_time = std::chrono::steady_clock::now();
    
    // Thread-local leaf storage for batching
    std::vector<PendingEvaluation> leaf_batch;
    const size_t OPTIMAL_BATCH_SIZE = settings_.batch_size;
    // MIN_BATCH_SIZE removed as it was unused
    leaf_batch.reserve(OPTIMAL_BATCH_SIZE);
    
    // Counters for monitoring
    int consecutive_empty_tries = 0;
    const int MAX_EMPTY_TRIES = 3;
    
    // CRITICAL FIX: Ensure root node is properly initialized
    if (search_roots.empty()) {
        std::cout << "CRITICAL ERROR: MCTSEngine::executeSerialSearch - search_roots is empty! Cannot proceed." << std::endl;
        return;
    }
    
    if (!search_roots[0]) {
        std::cout << "CRITICAL ERROR: MCTSEngine::executeSerialSearch - First search root is null! Cannot proceed." << std::endl;
        return;
    }
    
    std::shared_ptr<MCTSNode> main_root = search_roots[0];
    
    // Log root state details
    std::cout << "MCTSEngine::executeSerialSearch - Root node state:" << std::endl;
    std::cout << "  - is_terminal: " << (main_root->isTerminal() ? "yes" : "no") << std::endl;
    std::cout << "  - is_fully_expanded: " << (main_root->isFullyExpanded() ? "yes" : "no") << std::endl;
    std::cout << "  - has_children: " << (main_root->hasChildren() ? "yes" : "no") << std::endl;
    
    // Always expand root node if it's not terminal to ensure we have children
    if (!main_root->isTerminal()) {
        std::cout << "MCTSEngine::executeSerialSearch - Expanding main root node" << std::endl;
        bool expansion_success = expandNonTerminalLeaf(main_root);
        std::cout << "MCTSEngine::executeSerialSearch - Root expansion " 
                 << (expansion_success ? "succeeded" : "failed") << std::endl;
    }
    
    // Main search loop
    // CRITICAL FIX: Log start of main search loop
    std::cout << "MCTSEngine::executeSerialSearch - *** BEGINNING MAIN SEARCH LOOP with active_simulations=" 
              << active_simulations_.load(std::memory_order_acquire) << " ***" << std::endl;
              
    // CRITICAL FIX: Add a check to verify we actually enter the loop by forcing at least 100 simulations
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        std::cout << "CRITICAL ERROR: MCTSEngine::executeSerialSearch - active_simulations_ is 0 or negative! Forcing to 100." << std::endl;
        active_simulations_.store(100, std::memory_order_release);
    }
    
    // Add a counter to track iterations through the main loop
    int main_loop_iterations = 0;
    
    while (active_simulations_.load(std::memory_order_acquire) > 0) {
        main_loop_iterations++;
        
        // CRITICAL FIX: Log every 20 iterations to track progress
        if (main_loop_iterations <= 5 || main_loop_iterations % 20 == 0) {
            std::cout << "MCTSEngine::executeSerialSearch - Main loop iteration #" << main_loop_iterations 
                      << ", active_simulations=" << active_simulations_.load(std::memory_order_acquire)
                      << ", pending_evaluations=" << pending_evaluations_.load(std::memory_order_acquire)
                      << ", leaves_generated=" << total_leaves_generated_.load(std::memory_order_acquire) << std::endl;
        }
        
        // Check if we should wait for pending evaluations to complete
        if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Aggressive batch collection - try to fill the entire batch quickly
        auto batch_start_time = std::chrono::steady_clock::now();
        const auto MAX_BATCH_COLLECTION_TIME = std::chrono::milliseconds(5);
        
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
            
            // Better approach: Only extend simulations if we haven't found any leaves yet
            // instead of forcing an arbitrary number that might never terminate
            if (old_sims <= 20 && total_leaves_generated_.load(std::memory_order_acquire) < 10) {
                // Only artificially extend the search if we haven't generated enough leaves
                // This ensures we give the search enough time to start finding leaves
                std::cout << "MCTSEngine::executeSerialSearch - Extending search: active_simulations_ is " << old_sims 
                         << ", adding 50 more simulations to give search time to find leaves" << std::endl;
                active_simulations_.fetch_add(50, std::memory_order_release);
                old_sims = active_simulations_.load(std::memory_order_acquire);
            }
            
            // Claim multiple simulations at once for better efficiency
            // But keep simulations_to_claim small to ensure frequent queue checking
            int simulations_to_claim = std::min(2, old_sims);
            
            // CRITICAL FIX: Ensure we always claim at least one simulation
            simulations_to_claim = std::max(1, simulations_to_claim);
            
            // Try to claim the simulations - add debug for tracking
            bool claimed = active_simulations_.compare_exchange_weak(
                old_sims, old_sims - simulations_to_claim, std::memory_order_acq_rel);
            
            if (claimed) {
                static int claim_counter = 0;
                claim_counter++;
                if (claim_counter <= 10 || claim_counter % 50 == 0) {
                    std::cout << "MCTSEngine::executeSerialSearch - Claimed " << simulations_to_claim 
                              << " simulations, " << (old_sims - simulations_to_claim) << " remaining "
                              << "(claim #" << claim_counter << ")" << std::endl;
                }
            } else {
                std::cout << "MCTSEngine::executeSerialSearch - Failed to claim simulations, will retry" << std::endl;
                
                // Improved approach: Only add more simulations if we haven't generated enough leaves
                // and we've been running for a short time
                // This prevents extending a search that should be terminating
                if (active_simulations_.load(std::memory_order_acquire) <= 0 && 
                    total_leaves_generated_.load(std::memory_order_acquire) < 10 &&
                    std::chrono::steady_clock::now() - search_start_time < std::chrono::seconds(2)) {
                    
                    active_simulations_.store(20, std::memory_order_release);
                    std::cout << "MCTSEngine::executeSerialSearch - Adding 20 more simulations after claim failure, still in early search phase" << std::endl;
                } else if (active_simulations_.load(std::memory_order_acquire) <= 0) {
                    // If we've been running for a while or have generated leaves, respect the termination condition
                    std::cout << "MCTSEngine::executeSerialSearch - Search appears to be complete (active_simulations_=" 
                             << active_simulations_.load(std::memory_order_acquire)
                             << ", leaves_generated=" << total_leaves_generated_.load(std::memory_order_acquire) << ")" << std::endl;
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
                        
                        // Enhanced debug logging for leaf selection
                        static int leaf_selection_count = 0;
                        leaf_selection_count++;
                        
                        // Log more frequently at the beginning and periodically later
                        bool should_log = (leaf_selection_count <= 50) || (leaf_selection_count % 20 == 0);
                        
                        if (should_log) {
                            std::cout << "🔍 MCTSEngine::executeSerialSearch - Leaf selection #" << leaf_selection_count
                                     << ", leaf=" << (leaf ? leaf.get() : nullptr)
                                     << ", path_length=" << path.size()
                                     << ", is_terminal=" << (leaf && leaf->isTerminal() ? "yes" : "no");
                            
                            // Add more detailed diagnostics
                            if (leaf) {
                                std::cout << ", pending_eval=" << (leaf->hasPendingEvaluation() ? "yes" : "no")
                                         << ", being_evaluated=" << (leaf->isBeingEvaluated() ? "yes" : "no")
                                         << ", has_children=" << (leaf->hasChildren() ? "yes" : "no")
                                         << ", visit_count=" << leaf->getVisitCount();
                            }
                            std::cout << std::endl;
                            
                            // Log search state
                            static int last_active = 0;
                            int current_active = active_simulations_.load(std::memory_order_acquire);
                            if (current_active != last_active || leaf_selection_count <= 5) {
                                std::cout << "🔢 Search state: active_simulations=" << current_active
                                         << ", pending_evaluations=" << pending_evaluations_.load(std::memory_order_acquire)
                                         << ", total_leaves=" << total_leaves_generated_.load(std::memory_order_acquire)
                                         << std::endl;
                                last_active = current_active;
                            }
                        }
                        
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
                                // IMPROVED: More aggressive resetting of evaluation flags
                                if (leaf->hasPendingEvaluation() || leaf->isBeingEvaluated()) {
                                    if (leaf_selection_count <= 100 || leaf_selection_count % 50 == 0) {
                                        std::cout << "⚠️ MCTSEngine::executeSerialSearch - Leaf " << leaf.get()
                                                << " is already being evaluated (pending="
                                                << leaf->hasPendingEvaluation() << ", being_evaluated="
                                                << leaf->isBeingEvaluated() << ")" << std::endl;
                                    }
                                    
                                    // IMPROVED: Reset evaluation flag more frequently for non-expanded nodes
                                    // to avoid stalled nodes that never get evaluated
                                    if (leaf->hasPendingEvaluation() && !leaf->hasChildren()) {
                                        static int evaluation_reset_count = 0;
                                        evaluation_reset_count++;
                                        // Reset every 10th node instead of every 50th
                                        if (evaluation_reset_count % 10 == 0) {
                                            std::cout << "🔄 MCTSEngine::executeSerialSearch - RESETTING evaluation flag on leaf "
                                                    << leaf.get() << " (reset #" << evaluation_reset_count << ")" << std::endl;
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
                                    
                                    // IMPROVED: Always log the first 50 leaves to ensure we're getting data flow
                                    if (pending.request_id < 50 || pending.request_id % 20 == 0) {
                                        std::cout << "✅ MCTSEngine::executeSerialSearch - Adding leaf #" << pending.request_id 
                                                << " to batch: node=" << pending.node.get()
                                                << ", state=" << pending.state.get()
                                                << ", batch_id=" << pending.batch_id << std::endl;
                                    }
                                             
                                    // Validate node and state
                                    if (!pending.state || !pending.node) {
                                        std::cout << "❌ ERROR: Null state or node in pending evaluation, skipping" << std::endl;
                                        if (pending.node) {
                                            pending.node->clearAllEvaluationFlags();
                                        }
                                        continue;
                                    }
                                    
                                    // Direct submission to batch accumulator is the preferred approach for optimal batching
                                    // It bypasses the leaf queue which can be a bottleneck
                                    if (use_shared_queues_ && evaluator_ && evaluator_->getBatchAccumulator()) {
                                        BatchAccumulator* accumulator = evaluator_->getBatchAccumulator();
                                        if (accumulator) {
                                            // Always verify accumulator is running
                                            if (!accumulator->isRunning()) {
                                                std::cout << "⚠️ MCTSEngine::executeSerialSearch - BatchAccumulator exists but is not running. Starting it now!" << std::endl;
                                                accumulator->start();
                                                
                                                // Give it a moment to initialize
                                                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                                                
                                                // Verify it started successfully
                                                if (!accumulator->isRunning()) {
                                                    std::cout << "🔴 MCTSEngine::executeSerialSearch - CRITICAL ERROR: BatchAccumulator failed to start!" << std::endl;
                                                } else {
                                                    std::cout << "✅ MCTSEngine::executeSerialSearch - Successfully started BatchAccumulator" << std::endl;
                                                }
                                            }
                                            
                                            // Log direct submission (only log initial ones or periodically)
                                            if (pending.request_id < 10 || pending.request_id % 50 == 0) {
                                                std::cout << "📥 MCTSEngine::executeSerialSearch - Adding leaf #" 
                                                        << pending.request_id 
                                                        << " to BatchAccumulator (running: " 
                                                        << (accumulator->isRunning() ? "yes" : "no") << ")" << std::endl;
                                            }
                                            
                                            // Extra validation before submission
                                            if (!pending.state) {
                                                std::cout << "❌ MCTSEngine::executeSerialSearch - Null state for leaf #" 
                                                        << pending.request_id << ", skipping submission" << std::endl;
                                                continue;
                                            }
                                            
                                            if (!pending.node) {
                                                std::cout << "❌ MCTSEngine::executeSerialSearch - Null node for leaf #" 
                                                        << pending.request_id << ", skipping submission" << std::endl;
                                                continue;
                                            }
                                            
                                            // Add to accumulator with move semantics
                                            accumulator->addEvaluation(std::move(pending));
                                            
                                            // Always notify after adding to ensure processing starts immediately
                                            if (external_queue_notify_fn_) {
                                                external_queue_notify_fn_();
                                                // Send a second notification after a tiny delay to catch any race conditions
                                                std::this_thread::sleep_for(std::chrono::microseconds(100));
                                                external_queue_notify_fn_();
                                            }
                                            
                                            // Count as found and track
                                            leaves_found++;
                                            utils::debug_logger().trackItemProcessed();
                                            
                                            // Update total leaves submitted for tracking
                                            static std::atomic<int> total_leaves_submitted{0};
                                            int leaves_so_far = total_leaves_submitted.fetch_add(1, std::memory_order_relaxed) + 1;
                                            
                                            // Log submission milestones
                                            if (leaves_so_far == 1 || leaves_so_far == 10 || 
                                                leaves_so_far == 100 || leaves_so_far % 500 == 0) {
                                                std::cout << "🎯 MCTSEngine::executeSerialSearch - MILESTONE: " 
                                                        << leaves_so_far << " total leaves submitted to BatchAccumulator" 
                                                        << std::endl;
                                            }
                                            
                                            // Skip adding to leaf_batch since we already added directly
                                            continue;
                                        }
                                    }
                                    
                                    // IMPROVED: Enhanced state validation with more details
                                    bool state_valid = false;
                                    try {
                                        state_valid = pending.state->validate();
                                        if (!state_valid) {
                                            std::cout << "❌ ERROR: State validation failed for leaf #" << pending.request_id << std::endl;
                                        }
                                    } catch (const std::exception& e) {
                                        std::cout << "❌ ERROR: Exception during state validation: " << e.what() << std::endl;
                                        state_valid = false;
                                    } catch (...) {
                                        std::cout << "❌ ERROR: Unknown exception during state validation" << std::endl;
                                        state_valid = false;
                                    }
                                    
                                    if (!state_valid) {
                                        std::cout << "❌ ERROR: Invalid state in pending evaluation, skipping leaf #" 
                                                << pending.request_id << std::endl;
                                        pending.node->clearAllEvaluationFlags();
                                        continue;
                                    }
                                    
                                    // Add to batch after validation
                                    leaf_batch.push_back(std::move(pending));
                                    leaves_found++;
                                    
                                    // IMPROVED: Track the leaves found to see progress
                                    utils::debug_logger().trackItemProcessed();
                                    if (leaves_found % 10 == 0) {
                                        std::cout << "📊 MCTSEngine::executeSerialSearch - Found " << leaves_found 
                                                << " leaves in current batch" << std::endl;
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "❌ Error during leaf collection: " << e.what() << std::endl;
                        active_simulations_.fetch_add(1, std::memory_order_acq_rel); // Return the simulation
                    }
                }
                
                // Update consecutive empty tries counter with improved logging
                if (leaves_found == 0) {
                    consecutive_empty_tries++;
                    if (consecutive_empty_tries % 5 == 0) {
                        std::cout << "⚠️ MCTSEngine::executeSerialSearch - " << consecutive_empty_tries 
                                << " consecutive tries without finding leaves" << std::endl;
                    }
                } else {
                    consecutive_empty_tries = 0;
                }
            }
        }
        
        // Submit batch if we have enough leaves
        if (!leaf_batch.empty()) {
            std::cout << "📦 MCTSEngine::executeSerialSearch - Submitting batch with " 
                     << leaf_batch.size() << " leaves" << std::endl;
            
            // Choose correct queue based on whether shared queues are configured
            auto& target_queue = use_shared_queues_ ? *shared_leaf_queue_ : leaf_queue_;
            
            // ENHANCED: Even more aggressive retry mechanism
            const int MAX_SUBMISSION_ATTEMPTS = 5; // Increased from 3 to 5
            size_t total_enqueued = 0;
            
            // IMPROVED: Log queue size before submission to diagnose issues
            size_t queue_size = 0;
            if (use_shared_queues_ && shared_leaf_queue_) {
                queue_size = shared_leaf_queue_->size_approx();
                std::cout << "📊 MCTSEngine::executeSerialSearch - Shared leaf queue size before submission: " 
                         << queue_size << std::endl;
            }
            
            // First submit in bulk if possible for better performance
            if (leaf_batch.size() > 1) {
                std::cout << "📤 MCTSEngine::executeSerialSearch - Submitting bulk batch of " 
                         << leaf_batch.size() << " leaves to "
                         << (use_shared_queues_ ? "shared" : "internal") << " queue" << std::endl;
                
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
                
                std::cout << "✅ MCTSEngine::executeSerialSearch - Successfully enqueued " 
                         << enqueued << " of " << temp_batch.size() << " leaves in bulk" << std::endl;
                
                total_enqueued = enqueued;
                
                // IMPROVED: Check if bulk enqueue completed successfully
                if (enqueued == temp_batch.size()) {
                    std::cout << "✅ MCTSEngine::executeSerialSearch - Bulk enqueue fully successful" << std::endl;
                    
                    // Clear the main batch since we've successfully moved all items
                    leaf_batch.clear();
                } else {
                    std::cout << "⚠️ MCTSEngine::executeSerialSearch - Bulk enqueue partially successful (" 
                             << enqueued << "/" << temp_batch.size() << " items)" << std::endl;
                    
                    // IMPROVED: Reset leaf_batch for remaining items
                    leaf_batch.clear();
                    
                    // Add back items that weren't enqueued successfully (starting from the enqueued position)
                    for (size_t i = enqueued; i < temp_batch.size(); i++) {
                        leaf_batch.push_back(std::move(temp_batch[i]));
                    }
                    
                    std::cout << "⚠️ MCTSEngine::executeSerialSearch - " << leaf_batch.size() 
                             << " items remaining to be enqueued individually" << std::endl;
                }
                
                // Notify after bulk enqueue
                if (use_shared_queues_ && external_queue_notify_fn_ && enqueued > 0) {
                    std::cout << "🔔 MCTSEngine::executeSerialSearch - Calling external queue notification function after bulk" << std::endl;
                    external_queue_notify_fn_();
                    
                    // ENHANCED: Force duplicate notifications to ensure delivery 
                    for (int i = 0; i < 3; i++) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        external_queue_notify_fn_();
                    }
                } else if (!use_shared_queues_ && evaluator_ && enqueued > 0) {
                    std::cout << "🔔 MCTSEngine::executeSerialSearch - Calling evaluator->notifyLeafAvailable() after bulk" << std::endl;
                    evaluator_->notifyLeafAvailable();
                    
                    // ENHANCED: Force duplicate notifications to ensure delivery
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
                std::cout << "📤 MCTSEngine::executeSerialSearch - Submitting " << leaf_batch.size() 
                         << " leaves individually" << std::endl;
                
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
                                
                                // IMPROVED: Only log every few items to reduce spam
                                if (i < 5 || i % 10 == 0) {
                                    std::cout << "✅ MCTSEngine::executeSerialSearch - Successfully enqueued item #" 
                                             << i << " (attempt " << attempt+1 << ")" << std::endl;
                                }
                                
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
                                
                                // IMPROVED: Only log every few items to reduce spam
                                if (i < 5 || i % 10 == 0) {
                                    std::cout << "✅ MCTSEngine::executeSerialSearch - Successfully enqueued item #" 
                                             << i << " (attempt " << attempt+1 << ")" << std::endl;
                                }
                                
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
                            
                            if (i < 5 || i % 10 == 0) {
                                std::cout << "⚠️ MCTSEngine::executeSerialSearch - Retrying item #" << i 
                                         << " (attempt " << attempt+1 << "/" << MAX_SUBMISSION_ATTEMPTS 
                                         << ", delay=" << delay_ms << "ms)" << std::endl;
                            }
                        }
                    }
                    
                    if (!item_enqueued) {
                        std::cout << "❌ MCTSEngine::executeSerialSearch - Failed to enqueue item #" << i 
                                 << " after " << MAX_SUBMISSION_ATTEMPTS << " attempts" << std::endl;
                        
                        // IMPROVED: Clear evaluation flag for failed items to avoid leaks
                        if (leaf_batch[i].node) {
                            leaf_batch[i].node->clearEvaluationFlag();
                        }
                    }
                }
                
                std::cout << "📊 MCTSEngine::executeSerialSearch - Individually enqueued " 
                         << individual_enqueued << " of " << leaf_batch.size() << " items" << std::endl;
            }
            
            // Update pending evaluations count
            pending_evaluations_.fetch_add(total_enqueued, std::memory_order_acq_rel);
            std::cout << "🔢 MCTSEngine::executeSerialSearch - Updated pending_evaluations_ to " 
                     << pending_evaluations_.load() << " (added " << total_enqueued << ")" << std::endl;
            
            // ENHANCED: Extra notifications with delay to ensure evaluator processes items
            if (total_enqueued > 0) {
                // Log queue size after submission for debugging
                if (use_shared_queues_ && shared_leaf_queue_) {
                    size_t after_queue_size = shared_leaf_queue_->size_approx();
                    std::cout << "📊 MCTSEngine::executeSerialSearch - Shared leaf queue size after submission: " 
                             << after_queue_size << " (added " << total_enqueued << " items)" << std::endl;
                    
                    // ENHANCED: Sequence of notifications with delays
                    if (external_queue_notify_fn_) {
                        std::cout << "🔔 MCTSEngine::executeSerialSearch - Sending multiple notifications to ensure processing" << std::endl;
                        
                        // First notification
                        external_queue_notify_fn_();
                        
                        // Wait and send more notifications to ensure delivery
                        for (int i = 0; i < 3; i++) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            external_queue_notify_fn_();
                        }
                    }
                } else if (!use_shared_queues_ && evaluator_) {
                    std::cout << "🔔 MCTSEngine::executeSerialSearch - Sending multiple notifications to evaluator" << std::endl;
                    
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