// src/mcts/mcts_engine.cpp
#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include "utils/memory_tracker.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <iomanip>
#include <queue>
#include <omp.h>
#include <cstdlib>

namespace alphazero {
namespace mcts {

// Define static members
std::mutex MCTSEngine::s_global_evaluator_mutex;
std::atomic<int> MCTSEngine::s_evaluator_init_counter{0};

} // namespace mcts
} // namespace alphazero

// Configurable debug level
#define MCTS_DEBUG 0
#define MCTS_VERBOSE 0  // Set to 1 for verbose logging, 0 for performance

// Lightweight logging macros
#if MCTS_VERBOSE
    #define MCTS_LOG_VERBOSE(msg) std::cout << msg << std::endl
#else 
    #define MCTS_LOG_VERBOSE(msg) (void)0
#endif

#if MCTS_DEBUG
    #define MCTS_LOG_DEBUG(msg) std::cout << msg << std::endl
#else
    #define MCTS_LOG_DEBUG(msg) (void)0
#endif

#define MCTS_LOG_ERROR(msg) (void)0

namespace alphazero {
namespace mcts {

MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      evaluator_started_(false),
      game_state_pool_enabled_(false) {  // Disable game state pool for debugging
    
    // Create transposition table with configurable size
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128; // Default 128MB
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Create node tracker for lock-free pending evaluation management
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Check neural network validity
    if (!neural_net) {
        MCTS_LOG_ERROR("ERROR: Null neural network passed to MCTSEngine constructor");
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Create evaluator with stronger exception handling
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            [neural_net](const std::vector<std::unique_ptr<core::IGameState>>& states) {
                return neural_net->inference(states);
            }, 
            settings.batch_size, 
            settings.batch_timeout);  // Use configured timeout for better batching
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        MCTS_LOG_ERROR("ERROR during evaluator creation: " << e.what());
        throw;
    }
}

MCTSEngine::MCTSEngine(InferenceFunction inference_fn, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      evaluator_started_(false),
      game_state_pool_enabled_(false) {  // Disable game state pool for debugging
    
    // Create transposition table with configurable size
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128; // Default 128MB
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Create node tracker for lock-free pending evaluation management
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Check inference function validity
    if (!inference_fn) {
        MCTS_LOG_ERROR("ERROR: Null inference function passed to MCTSEngine constructor");
        throw std::invalid_argument("Inference function cannot be null");
    }
    
    // Create evaluator with stronger exception handling
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            std::move(inference_fn), 
            settings.batch_size, 
            std::min(settings.batch_timeout, std::chrono::milliseconds(10)));  // Cap timeout at 10ms
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        MCTS_LOG_ERROR("ERROR during evaluator creation: " << e.what());
        throw;
    }
}

bool MCTSEngine::ensureEvaluatorStarted() {
    // Don't start local evaluator if using shared queues - the shared evaluator is managed externally
    if (use_shared_queues_) {
        return true;
    }
    
    // First check without lock for performance
    if (evaluator_started_.load(std::memory_order_acquire)) {
        return true;
    }
    
    // Acquire lock for initialization to prevent race conditions
    std::lock_guard<std::mutex> lock(evaluator_mutex_);
    
    // Double-check with lock held
    if (evaluator_started_.load(std::memory_order_relaxed)) {
        return true;
    }
    
    try {
        // Make sure evaluator exists
        if (!evaluator_) {
            return false;
        }
        
        // Start the evaluator
        try {
            evaluator_->start();
            } catch (const std::exception& e) {
            throw;
        } catch (...) {
            throw;
        }
        evaluator_started_.store(true, std::memory_order_release);
        return true;
    } catch (const std::exception& e) {
        // MCTSEngine::ensureEvaluatorStarted - Failed to start evaluator
        return false;
    } catch (...) {
        // MCTSEngine::ensureEvaluatorStarted - Unknown error starting evaluator
        return false;
    }
}

void MCTSEngine::safelyStopEvaluator() {
    // Don't stop evaluator if using shared queues - it's managed externally
    if (use_shared_queues_) {
        return;
    }
    
    if (evaluator_started_.load(std::memory_order_acquire)) {
        try {
            evaluator_->stop();
            evaluator_started_.store(false, std::memory_order_release);
        } catch (const std::exception& e) {
            // Commented out: Error stopping evaluator with error message
        } catch (...) {
            // Commented out: Unknown error stopping evaluator
        }
    }
}

void MCTSEngine::setUseTranspositionTable(bool use) {
    use_transposition_table_ = use;
}

bool MCTSEngine::isUsingTranspositionTable() const {
    return use_transposition_table_;
}

void MCTSEngine::setTranspositionTableSize(size_t size_mb) {
    // Create a new transposition table with the specified size
    // Use a reasonable number of shards based on thread count
    size_t num_shards = std::max(4u, std::thread::hardware_concurrency());
    if (settings_.num_threads > 0) {
        // Match number of shards to thread count for better performance
        num_shards = std::max(size_t(settings_.num_threads), num_shards);
    }
    
    transposition_table_ = std::make_unique<TranspositionTable>(size_mb, num_shards);
}

void MCTSEngine::clearTranspositionTable() {
    if (transposition_table_) {
        transposition_table_->clear();
    }
}

float MCTSEngine::getTranspositionTableHitRate() const {
    if (transposition_table_) {
        return transposition_table_->hitRate();
    }
    return 0.0f;
}

MCTSEngine::MCTSEngine(MCTSEngine&& other) noexcept {
    // First stop the other engine to ensure thread safety
    other.shutdown_ = true;
    
    // Stop evaluator if it was started
    if (other.evaluator_ && other.evaluator_started_) {
        try {
            other.evaluator_->stop();
        } catch (...) {
            // Ignore exceptions during move
        }
    }
    
    // Wait for threads to complete
    if (other.result_distributor_worker_.joinable()) {
        try {
            other.result_distributor_worker_.join();
        } catch (...) {
            // Ignore exceptions
        }
    }
    
    // Now safe to move resources
    settings_ = std::move(other.settings_);
    last_stats_ = std::move(other.last_stats_);
    evaluator_ = std::move(other.evaluator_);
    root_ = std::move(other.root_);
    shutdown_ = other.shutdown_.load();
    active_simulations_ = other.active_simulations_.load();
    search_running_ = other.search_running_.load();
    random_engine_ = std::move(other.random_engine_);
    transposition_table_ = std::move(other.transposition_table_);
    use_transposition_table_ = other.use_transposition_table_;
    evaluator_started_.store(other.evaluator_started_.load(), std::memory_order_release);
    pending_evaluations_ = other.pending_evaluations_.load();
    batch_counter_ = other.batch_counter_.load();
    total_leaves_generated_ = other.total_leaves_generated_.load();
    total_results_processed_ = other.total_results_processed_.load();
    leaf_queue_ = std::move(other.leaf_queue_);
    batch_queue_ = std::move(other.batch_queue_);
    result_queue_ = std::move(other.result_queue_);
    result_distributor_worker_ = std::move(other.result_distributor_worker_);
    workers_active_ = other.workers_active_.load();
    
    // Validate the moved evaluator
    if (!evaluator_) {
        // WARNING: evaluator_ is null after move constructor
    }
    other.workers_active_ = false;
    // Commented out - not used in OpenMP version
    // other.cv_.notify_all();
    // other.batch_cv_.notify_all();
    // other.result_cv_.notify_all();
    
    // Join other's threads before clearing
    // Commented out - replaced with OpenMP
    // for (auto& thread : other.tree_traversal_workers_) {
    //     if (thread.joinable()) {
    //         thread.join();
    //     }
    // }
    // Commented out - not used in OpenMP version
    // if (other.batch_accumulator_worker_.joinable()) {
    //     other.batch_accumulator_worker_.join();
    // }
    if (other.result_distributor_worker_.joinable()) {
        other.result_distributor_worker_.join();
    }
    
    // Now safe to clear
    // other.tree_traversal_workers_.clear();
    other.search_running_ = false;
    other.active_simulations_ = 0;
    other.evaluator_started_.store(false, std::memory_order_release);
}

MCTSEngine& MCTSEngine::operator=(MCTSEngine&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        shutdown_ = true;
        workers_active_ = false;
        // Commented out - not used in OpenMP version
        // cv_.notify_all();
        // batch_cv_.notify_all();
        // result_cv_.notify_all();
        
        // Join specialized workers
        // Commented out - not used in OpenMP version
        // if (batch_accumulator_worker_.joinable()) {
        //     batch_accumulator_worker_.join();
        // }
        if (result_distributor_worker_.joinable()) {
            result_distributor_worker_.join();
        }
        // Commented out - replaced with OpenMP
        // for (auto& thread : tree_traversal_workers_) {
        //     if (thread.joinable()) {
        //         thread.join();
        //     }
        // }
        
        safelyStopEvaluator();
        
        // Move resources from other
        settings_ = std::move(other.settings_);
        last_stats_ = std::move(other.last_stats_);
        evaluator_ = std::move(other.evaluator_);
        root_ = std::move(other.root_);
        shutdown_ = other.shutdown_.load();
        active_simulations_ = other.active_simulations_.load();
        search_running_ = other.search_running_.load();
        random_engine_ = std::move(other.random_engine_);
        transposition_table_ = std::move(other.transposition_table_);
        use_transposition_table_ = other.use_transposition_table_;
        evaluator_started_.store(other.evaluator_started_.load(), std::memory_order_release);
        pending_evaluations_ = other.pending_evaluations_.load();
        batch_counter_ = other.batch_counter_.load();
        total_leaves_generated_ = other.total_leaves_generated_.load();
        total_results_processed_ = other.total_results_processed_.load();
        leaf_queue_ = std::move(other.leaf_queue_);
        batch_queue_ = std::move(other.batch_queue_);
        result_queue_ = std::move(other.result_queue_);
        // batch_accumulator_worker_ = std::move(other.batch_accumulator_worker_);
        result_distributor_worker_ = std::move(other.result_distributor_worker_);
        // tree_traversal_workers_ = std::move(other.tree_traversal_workers_);
        workers_active_ = other.workers_active_.load();
        
        // Validate the moved evaluator
        if (!evaluator_) {
            // WARNING: evaluator_ is null after move assignment
        }
        
        // Properly clean up other's threads before clearing
        other.shutdown_ = true;
        other.workers_active_ = false;
        // Commented out - not used in OpenMP version
        // other.cv_.notify_all();
        // other.batch_cv_.notify_all();
        // other.result_cv_.notify_all();
        
        // Commented out - replaced with OpenMP
        // Join other's threads before clearing
        // for (auto& thread : other.tree_traversal_workers_) {
        //     if (thread.joinable()) {
        //         thread.join();
        //     }
        // }
        // Commented out - not used in OpenMP version
        // if (other.batch_accumulator_worker_.joinable()) {
        //     other.batch_accumulator_worker_.join();
        // }
        if (other.result_distributor_worker_.joinable()) {
            other.result_distributor_worker_.join();
        }
        
        // Now safe to clear
        // other.tree_traversal_workers_.clear();
        other.search_running_ = false;
        other.active_simulations_ = 0;
        other.evaluator_started_.store(false, std::memory_order_release);
    }
    
    return *this;
}

MCTSEngine::~MCTSEngine() {
    
    
    // Phase 1: Signal shutdown to all components atomically
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    active_simulations_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Commented out - not used in OpenMP version
    // Mark mutexes as destroyed to prevent threads from acquiring them
    // cv_mutex_destroyed_.store(true, std::memory_order_release);
    // batch_mutex_destroyed_.store(true, std::memory_order_release);
    // result_mutex_destroyed_.store(true, std::memory_order_release);
    
    // Phase 2: Stop the evaluator first (it's the source of new work)
    
    safelyStopEvaluator();
    
    // Phase 3: Force wake all threads immediately to check shutdown flag
    
    // Commented out - not used in OpenMP version
    // cv_.notify_all();
    // batch_cv_.notify_all();
    // result_cv_.notify_all();
    
    // Phase 4: Clear all queues to prevent stuck threads
    
    {
        PendingEvaluation temp_eval;
        while (leaf_queue_.try_dequeue(temp_eval)) {
            // Clear the evaluation flag if the node was marked for evaluation
            if (temp_eval.node) {
                try {
                    temp_eval.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
        
        BatchInfo temp_batch; // Assuming BatchInfo is light or has proper move/destructor
        while (batch_queue_.try_dequeue(temp_batch)) {
            // Clear evaluation flags for all nodes in batch
            for (auto& eval : temp_batch.evaluations) {
                if (eval.node) {
                    try {
                        eval.node->clearEvaluationFlag();
                    } catch (...) {}
                }
            }
        }
        
        std::pair<NetworkOutput, PendingEvaluation> temp_result;
        while (result_queue_.try_dequeue(temp_result)) {
            // Clear evaluation flag for result nodes
            if (temp_result.second.node) {
                try {
                    temp_result.second.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
    }
    
    // Phase 5: Join specialized worker threads (blocking join)
    
    
    // Join result distributor
    if (result_distributor_worker_.joinable()) {
        
        result_distributor_worker_.join();
        
    }
    
    // Commented out - replaced with OpenMP
    // Join tree traversal workers
    // for (size_t i = 0; i < tree_traversal_workers_.size(); ++i) {
    //     if (tree_traversal_workers_[i].joinable()) {
    //         tree_traversal_workers_[i].join();
    //     }
    // }
    
    
    // Phase 6: Final cleanup - clear transposition table and root
    
    if (transposition_table_) {
        transposition_table_->clear();
    }
    root_.reset();
    
    
}

SearchResult MCTSEngine::search(const core::IGameState& state) {
    // MCTSEngine::search - Starting search...
    // alphazero::utils::trackMemory("MCTSEngine::search started");
    auto start_time = std::chrono::steady_clock::now();

    // Validate the state before proceeding
    try {
        // MCTSEngine::search - Validating state...
        if (!state.validate()) {
            // MCTSEngine::search - Invalid game state passed to search method
            SearchResult result;
            result.action = -1;
            result.value = 0.0f;
            // Return best guess from legal moves
            // MCTSEngine::search - Getting legal moves from invalid state...
            auto legal_moves = state.getLegalMoves();
            if (!legal_moves.empty()) {
                result.action = legal_moves[0];
                // Create a uniform policy
                result.probabilities.resize(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
                // MCTSEngine::search - Using first legal move
            }
            return result;
        }
        // MCTSEngine::search - State validated successfully
    } catch (const std::exception& e) {
        // MCTSEngine::search - Exception during state validation
        SearchResult result; // Return default/error result
        result.action = -1;
        return result;
    } catch (...) {
        // MCTSEngine::search - Unknown exception during state validation
        SearchResult result; // Return default/error result
        result.action = -1;
        return result;
    }

    // Critical: Clear the transposition table BEFORE resetting the tree
    // This ensures node pointers are still valid when clearing the table
    if (use_transposition_table_ && transposition_table_) {
        // MCTSEngine::search - Clearing transposition table for new search
        transposition_table_->clear(); // Clear all entries
        transposition_table_->resetStats(); // Reset hit/miss stats
    }

    // Critical: Reset the previous search state (tree and root)
    // This will delete all MCTSNode objects from the previous search.
    // MUST be done AFTER clearing the transposition table
    root_.reset();
    
    // Don't start evaluator here - it will be configured with external queues later
    // The evaluator should only be started after external queues are configured

    // Initialize statistics for the new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;

    // Check if the game state is terminal before starting the search
    if (state.isTerminal()) {
        // MCTSEngine::search - Game state is already terminal. No search needed
        SearchResult result;
        result.action = -1; // No action to take
        try {
            core::GameResult game_res = state.getGameResult();
            int current_player = state.getCurrentPlayer();
            if (game_res == core::GameResult::WIN_PLAYER1) {
                result.value = (current_player == 1) ? 1.0f : -1.0f;
            } else if (game_res == core::GameResult::WIN_PLAYER2) {
                result.value = (current_player == 2) ? 1.0f : -1.0f;
            } else { // Draw or Ongoing (though isTerminal() should be true)
                result.value = 0.0f;
            }
        } catch (const std::exception& e) {
            // MCTSEngine::search - Error getting terminal value
            result.value = 0.0f;
        }
        result.probabilities.assign(state.getActionSpaceSize(), 0.0f);
        last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
        return result;
    }

    try {
        // MCTSEngine::search - Calling runSearch()...
        runSearch(state);
        // MCTSEngine::search - runSearch() completed successfully
    }
    catch (const std::exception& e) {
        // MCTSEngine::search - Error during search
        // Ensure proper cleanup before rethrowing
        safelyStopEvaluator();
        throw;
    }
    catch (...) {
        // MCTSEngine::search - Unknown error during search
        safelyStopEvaluator();
        throw std::runtime_error("Unknown error during search");
    }

    auto end_time = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    SearchResult result;
    result.action = -1; // Default invalid action

    try {
        // Extract action probabilities based on visit counts
        result.probabilities = getActionProbabilities(root_, settings_.temperature);

        // Select action from probabilities, safely
        if (!result.probabilities.empty()) {
            // Check if we're dealing with probabilities that sum to approximately 1
            float sum = 0.0f;
            for (float p : result.probabilities) {
                sum += p;
            }
            
            if (std::abs(sum - 1.0f) > 0.1f && sum > 0.0f) {
                // Normalize probabilities
                for (auto& p : result.probabilities) {
                    p /= sum;
                }
            }
            
            // Temperature near zero - deterministic selection (argmax)
            if (settings_.temperature < 0.01f) {
                auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
                if (max_it != result.probabilities.end()) {
                    result.action = std::distance(result.probabilities.begin(), max_it);
                }
            } else {
                // Sample according to the probability distribution
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float r = dist(random_engine_);
                float cumsum = 0.0f;
                
                for (size_t i = 0; i < result.probabilities.size(); i++) {
                    cumsum += result.probabilities[i];
                    if (r <= cumsum) {
                        result.action = static_cast<int>(i);
                        break;
                    }
                }
            }
            
            // Fallback in case of numerical issues
            if (result.action < 0 && !result.probabilities.empty()) {
                auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
                result.action = std::distance(result.probabilities.begin(), max_it);
            }
        } 
        else if (root_ && !root_->getChildren().empty()) {
            // If no probabilities, select most visited child
            int max_visits = -1;
            
            for (size_t i = 0; i < root_->getChildren().size(); i++) {
                auto child = root_->getChildren()[i];
                if (child && child->getVisitCount() > max_visits) {
                    max_visits = child->getVisitCount();
                    if (i < root_->getActions().size()) {
                        result.action = root_->getActions()[i];
                    }
                }
            }
        } 
        else {
            // Last resort - select a valid legal move
            auto legal_moves = state.getLegalMoves();
            if (!legal_moves.empty()) {
                result.action = legal_moves[0];
            }
        }

        // Get value estimate
        result.value = root_ ? root_->getValue() : 0.0f;
    }
    catch (const std::exception& e) {
        // Error extracting search results
        
        // Set fallback results
        if (result.action < 0) {
            try {
                auto legal_moves = state.getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                }
            } catch (const std::exception& e) {
                // Error getting legal moves
            } catch (...) {
                // Unknown error getting legal moves
            }
        }
    }
    catch (...) {
        // Unknown error extracting search results
        
        // Set fallback results
        if (result.action < 0) {
            try {
                auto legal_moves = state.getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                }
            } catch (...) {
                // Silently ignore errors here
            }
        }
    }

    // Update statistics
    last_stats_.search_time = search_time;
    last_stats_.avg_batch_size = evaluator_->getAverageBatchSize();
    last_stats_.avg_batch_latency = evaluator_->getAverageBatchLatency();
    last_stats_.total_evaluations = evaluator_->getTotalEvaluations();
    
    // alphazero::utils::trackMemory("MCTSEngine::search completed");

    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = 1000.0f * last_stats_.total_nodes / 
                                      std::max(1, static_cast<int>(last_stats_.search_time.count()));
    }

    // Add transposition table stats if enabled
    if (use_transposition_table_ && transposition_table_) {
        last_stats_.tt_hit_rate = transposition_table_->hitRate();
        last_stats_.tt_size = transposition_table_->size();
    }
    
    result.stats = last_stats_;
    return result;
}

const MCTSSettings& MCTSEngine::getSettings() const {
    return settings_;
}

void MCTSEngine::updateSettings(const MCTSSettings& settings) {
    settings_ = settings;
}

const MCTSStats& MCTSEngine::getLastStats() const {
    return last_stats_;
}

void MCTSEngine::runSearch(const core::IGameState& state) {
    // MCTSEngine::runSearch - Starting runSearch...
    // Reset statistics
    last_stats_ = MCTSStats();
    
    // Initialize game state pool if enabled
    if (game_state_pool_enabled_ && !utils::GameStatePoolManager::getInstance().hasPool(state.getGameType())) {
        try {
            // Initialize with reasonable defaults
            size_t pool_size = settings_.num_simulations / 4;  // Estimate based on simulations
            utils::GameStatePoolManager::getInstance().initializePool(state.getGameType(), pool_size);
            
        } catch (const std::exception& e) {
            MCTS_LOG_ERROR("Failed to initialize GameState pool: " << e.what());
            // Continue without pooling
        }
    }
    
    // Wait for all worker threads to finish processing before cleaning up
    // from any previous search iteration on this engine instance.
    {
        // MCTSEngine::runSearch - Waiting for worker threads to finish processing...
        // MCTSEngine::runSearch - Checking shutdown flag
                 
        // OpenMP version - no manual thread management needed
        // Just ensure no simulations are active from previous searches
        active_simulations_.store(0, std::memory_order_release);
        
        // Small delay to ensure any OpenMP threads have finished
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // If using the transposition table, it must be cleared BEFORE deleting the tree
    // to ensure node pointers are still valid when clearing the table
    if (use_transposition_table_ && transposition_table_) {
        // MCTSEngine::runSearch - Clearing transposition table...
        try {
            transposition_table_->clear();
            // MCTSEngine::runSearch - Transposition table cleared successfully
        } catch (const std::exception& e) {
            // MCTSEngine::runSearch - Error clearing transposition table
            // In case of any exception during clear, recreate the table entirely
            // This is safer than potentially having dangling pointers
            size_t size_mb = 128; // Default size
            size_t num_shards = std::max(4u, std::thread::hardware_concurrency());
            if (settings_.num_threads > 0) {
                num_shards = std::max(size_t(settings_.num_threads), num_shards);
            }
            // MCTSEngine::runSearch - Recreating transposition table
            transposition_table_ = std::make_unique<TranspositionTable>(size_mb, num_shards);
            // MCTSEngine::runSearch - Transposition table recreated successfully
        }
    } else {
        // MCTSEngine::runSearch - Skipping transposition table clear (not used or null)
    }

    // Clean up the old root if it exists. This invalidates all nodes in the previous tree.
    // MUST be done AFTER clearing the transposition table
    // MCTSEngine::runSearch - Cleaning up old root node...
    if (root_) {
        // MCTSEngine::runSearch - Old root exists, cleaning up...
    } else {
        // MCTSEngine::runSearch - No old root exists
    }
    root_.reset();
    // MCTSEngine::runSearch - Root node reset completed
    
    // Create the new root node.
    // If using the transposition table, it has just been cleared. We create a new root
    // from the input state and then add it to the TT.
    // We do not attempt to find the new root in the just-cleared TT, as that could lead to
    // using stale pointers if the clear operation was somehow incomplete or a hash collided.
    // MCTSEngine::runSearch - Creating new root node from state...
    try {
        // Clone the state with proper error handling
        std::shared_ptr<core::IGameState> state_clone;
        // MCTSEngine::runSearch - Cloning state...
        try {
            state_clone = cloneGameState(state);
            if (!state_clone) {
                // MCTSEngine::runSearch - ERROR: cloneGameState() returned nullptr
                throw std::runtime_error("cloneGameState() returned nullptr");
            }
            // MCTSEngine::runSearch - State cloned successfully
        } catch (const std::exception& e) {
            // MCTSEngine::runSearch - Exception during state cloning
            throw std::runtime_error(std::string("Failed to clone state: ") + e.what());
        } catch (...) {
            // MCTSEngine::runSearch - Unknown exception during state cloning
            throw std::runtime_error("Unknown error when cloning state");
        }
        
        // Additional validation of the cloned state
        // MCTSEngine::runSearch - Validating cloned state...
        try {
            if (!state_clone->validate()) {
                // MCTSEngine::runSearch - Cloned state failed validation
                throw std::runtime_error("Cloned state failed validation");
            }
            // MCTSEngine::runSearch - Cloned state validated successfully
            } catch (const std::exception& e) {
            // MCTSEngine::runSearch - Exception validating cloned state
            throw std::runtime_error(std::string("Cloned state validation error: ") + e.what());
        } catch (...) {
            // MCTSEngine::runSearch - Unknown exception validating cloned state
            throw std::runtime_error("Unknown error when validating cloned state");
        }
        
        // Create root node with proper error handling
        // MCTSEngine::runSearch - Creating root node...
        try {
            // Convert shared_ptr to unique_ptr for MCTSNode::create
            root_ = MCTSNode::create(std::unique_ptr<core::IGameState>(state_clone->clone().release()));
            // MCTSEngine::runSearch - Root node created successfully
            } catch (const std::exception& e) {
            // MCTSEngine::runSearch - Exception creating root node
            throw std::runtime_error(std::string("Failed to create root node: ") + e.what());
        } catch (...) {
            // MCTSEngine::runSearch - Unknown exception creating root node
            throw std::runtime_error("Unknown error creating root node");
        }

        // Ensure we have a valid root
        if (!root_) {
            // MCTSEngine::runSearch - Failed to create root node
            throw std::runtime_error("Failed to create root node");
        }
        // MCTSEngine::runSearch - Root node pointer is valid
        
        // Validate the root node's state
        // MCTSEngine::runSearch - Validating root node's state...
        try {
            if (!root_->getState().validate()) {
                // MCTSEngine::runSearch - Root node state invalid after creation
                throw std::runtime_error("Root node state invalid after creation");
            }
            // MCTSEngine::runSearch - Root node's state validated successfully
        } catch (const std::exception& e) {
            // MCTSEngine::runSearch - Exception validating root node state
            throw std::runtime_error(std::string("Root node state validation error: ") + e.what());
        } catch (...) {
            // MCTSEngine::runSearch - Unknown exception validating root node state
            throw std::runtime_error("Unknown error validating root node state");
        }

        // If using transposition table, store the new root.
        if (use_transposition_table_ && transposition_table_ && root_) {
            try {
                uint64_t hash = root_->getState().getHash(); // Get hash from the root's state
                transposition_table_->store(hash, std::weak_ptr<MCTSNode>(root_), 0);
            } catch (const std::exception& e) {
                // Error storing root in transposition table
                // Continue without transposition table storage
            } catch (...) {
                // Unknown error storing root in transposition table
                // Continue without transposition table storage
            }
            
            #if MCTS_DEBUG
            // Commented out: Debug printing about storing new root in transposition table with hash value
            #endif
        }

        // Initialize root node - expand it and set prior probabilities
        try {
            root_->expand();
            
            // For the initial root evaluation, we need to set prior probabilities
            // This is normally done by the neural network, but for the root we need to bootstrap
            if (root_->getNumExpandedChildren() > 0) {
                // Get neural network evaluation for root
                auto state_clone = cloneGameState(root_->getState());
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.push_back(std::unique_ptr<core::IGameState>(state_clone->clone().release()));
                
                // Get evaluation from neural network  
                if (evaluator_ && evaluator_->getInferenceFunction()) {
                    auto outputs = evaluator_->getInferenceFunction()(states);
                    if (!outputs.empty()) {
                        root_->setPriorProbabilities(outputs[0].policy);
                        } else {
                        }
                } else {
                    }
            } else {
                }
        } catch (const std::exception& e) {
            std::cerr << "[ENGINE] Error initializing root node: " << e.what() << std::endl;
            throw;
        }

        // Add Dirichlet noise to root node policy for exploration
        if (settings_.add_dirichlet_noise) {
            try {
                addDirichletNoise(root_);
            } catch (const std::exception& e) {
                // Error adding Dirichlet noise
                // Continue without noise - non-fatal
            } catch (...) {
                // Unknown error adding Dirichlet noise
                // Continue without noise - non-fatal
            }
        }

        // Set search running flag
        search_running_.store(true, std::memory_order_release);
        active_simulations_ = settings_.num_simulations;  // Initialize with total simulations to run

        // Configure evaluator to use external queues BEFORE starting it
        // Use global mutex to prevent multiple engines from initializing at once
        // Always ensure evaluator is started
        if (evaluator_ && !evaluator_started_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> global_lock(s_global_evaluator_mutex);
            
            // Increment global counter to track engine initialization order
            int init_order = s_evaluator_init_counter.fetch_add(1, std::memory_order_acq_rel);
            
            // Release global lock immediately after getting our order
            global_lock.unlock();
            
            // Now acquire local mutex for actual initialization
            std::lock_guard<std::mutex> local_lock(evaluator_mutex_);
            
            // Double-check after acquiring lock
            if (!evaluator_started_.load(std::memory_order_relaxed)) {
                // Provide a callback to notify when results are available
                auto result_notify_fn = [this]() {
                    // Notification mechanism replaced with lock-free polling in OpenMP
                };
                
                // Use shared queues if configured, otherwise use internal queues
                if (use_shared_queues_) {
                    evaluator_->setExternalQueues(shared_leaf_queue_, shared_result_queue_, result_notify_fn);
                    } else {
                    evaluator_->setExternalQueues(&leaf_queue_, &result_queue_, result_notify_fn);
                    }
                
                // Now start the evaluator with external queues configured
                // For shared queues, the evaluator is managed externally
                if (use_shared_queues_) {
                    // Mark as started but don't actually start - it's managed by SelfPlayManager
                    evaluator_started_.store(true, std::memory_order_release);
                    } else {
                    // Direct evaluator start - we already hold the lock
                    try {
                        // Make sure evaluator exists
                        if (!evaluator_) {
                            throw std::runtime_error("Evaluator is null");
                        }
                        
                        // Start the evaluator
                        try {
                            evaluator_->start();
                            } catch (const std::exception& e) {
                            throw;
                        } catch (...) {
                            throw;
                        }
                        evaluator_started_.store(true, std::memory_order_release);
                        } catch (const std::exception& e) {
                        std::cerr << "[ENGINE][" << init_order << "] Exception starting evaluator: " << e.what() << std::endl;
                        throw;
                    }
                }
            }
        } else {
            }
        
        // OpenMP implementation - use OpenMP threads instead of manual thread management
        // Note: We don't call omp_set_num_threads here since that doesn't work in nested parallel regions
        // Instead, we'll use the num_threads clause on the parallel pragma
        // Start result distributor only if not using shared queues
        // With shared queues, the SelfPlayManager handles result distribution
        if (!use_shared_queues_) {
            if (!result_distributor_worker_.joinable()) {
                workers_active_.store(true, std::memory_order_release);
                shutdown_.store(false, std::memory_order_release);
                result_distributor_worker_ = std::thread(&MCTSEngine::resultDistributorWorker, this);
            } else {
                }
        } else {
            }

        // Calculate the number of simulations to run
        int num_simulations = settings_.num_simulations;
        if (num_simulations <= 0) {
            num_simulations = 800; // Default value
        }

        // Set all simulations at once for better batching
        active_simulations_.store(num_simulations, std::memory_order_release);
        // cv_.notify_all(); // Not needed in OpenMP version

        // Create search roots based on parallelization strategy
        std::vector<std::shared_ptr<MCTSNode>> search_roots;
        
        if (settings_.use_root_parallelization && settings_.num_root_workers > 1) {
            // Root parallelization: create independent root copies for each worker
            for (int i = 0; i < settings_.num_root_workers; i++) {
                try {
                    // Create a deep copy of the root for each worker
                    auto cloned_state = root_->getState().clone();
                    if (!cloned_state) {
                        std::cerr << "[ROOT_PARALLEL] ERROR: Failed to clone root state for worker " << i << std::endl;
                        continue;
                    }
                    
                    auto root_copy = MCTSNode::create(std::move(cloned_state), nullptr);
                    if (root_copy) {
                        // Initialize the root copy same as the main root
                        root_copy->expand();
                        
                        // Copy prior probabilities from the main root
                        if (root_copy->getNumExpandedChildren() > 0 && root_->getNumExpandedChildren() > 0) {
                            // Since we cloned the state, the children should have the same actions
                            // So we can directly copy the prior probabilities
                            std::vector<float> priors(root_copy->getState().getActionSpaceSize());
                            auto& root_children = root_->getChildren();
                            auto& copy_children = root_copy->getChildren();
                            
                            // Map actions to priors from the main root
                            for (const auto& child : root_children) {
                                int action = child->getAction();
                                if (action >= 0 && action < priors.size()) {
                                    priors[action] = child->getPriorProbability();
                                    }
                            }
                            
                            // Apply priors to the copy
                            root_copy->setPriorProbabilities(priors);
                        }
                        
                        // Apply Dirichlet noise if needed
                        if (settings_.add_dirichlet_noise) {
                            try {
                                addDirichletNoise(root_copy);
                            } catch (...) {
                                // Continue without noise
                            }
                        }
                        
                        search_roots.push_back(root_copy);
                    } else {
                        std::cerr << "[MCTS] ERROR: Failed to create root copy for worker " << i << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[MCTS] Exception creating root copy for worker " << i << ": " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[MCTS] Unknown exception creating root copy for worker " << i << std::endl;
                }
            }
            
            if (search_roots.empty()) {
                std::cerr << "[MCTS] WARNING: Failed to create any root copies, falling back to single root" << std::endl;
                search_roots.push_back(root_);
            }
            
            } else {
            // Single root (default)
            search_roots.push_back(root_);
            }

        // OpenMP parallel search with aggressive leaf collection
        std::atomic<int> completed_simulations(0);
        const int BATCH_SIZE = std::max(1, settings_.batch_size / 4);  // Ensure at least 1, dynamic batch size based on settings
        
        // Check if result distributor is running
        // Thread-local leaf storage for batching
        struct ThreadLocalLeaves {
            std::vector<PendingEvaluation> leaves;
            std::vector<std::shared_ptr<MCTSNode>> visited_path;
            int thread_id;
            int batch_capacity;
            
            ThreadLocalLeaves(int batch_size) : thread_id(omp_get_thread_num()), batch_capacity(batch_size) {
                leaves.reserve(batch_capacity);
                visited_path.reserve(64);
            }
        };
        
        // Hybrid approach - OpenMP tree traversal + leaf parallelization
        // If we're already in a parallel region (from self-play), use serial tree traversal with leaf batching
        if (omp_in_parallel()) {
            // Serial leaf collection when already in parallel - optimized for batching
            std::vector<PendingEvaluation> leaf_batch;
            const size_t OPTIMAL_BATCH_SIZE = settings_.batch_size;  // Use full batch size
            const size_t MIN_BATCH_SIZE = 1;  // Always process single items to avoid deadlock
            leaf_batch.reserve(OPTIMAL_BATCH_SIZE);
            
            int consecutive_empty_tries = 0;
            const int MAX_EMPTY_TRIES = 3;  // Reduce to be more responsive
            
            while (active_simulations_.load(std::memory_order_acquire) > 0) {
                // Check if we should wait for pending evaluations to complete
                if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 4) {
                    // Too many pending - wait for some to complete
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                
                // Aggressive batch collection - try to fill the entire batch quickly
                auto batch_start_time = std::chrono::steady_clock::now();
                const auto MAX_BATCH_COLLECTION_TIME = std::chrono::milliseconds(5);
                
                while (leaf_batch.size() < OPTIMAL_BATCH_SIZE && 
                       active_simulations_.load(std::memory_order_acquire) > 0 &&
                       (std::chrono::steady_clock::now() - batch_start_time) < MAX_BATCH_COLLECTION_TIME &&
                       pending_evaluations_.load(std::memory_order_acquire) < settings_.batch_size * 6) {
                    
                    int old_sims = active_simulations_.load(std::memory_order_acquire);
                    if (old_sims <= 0) break;
                    
                    // Claim multiple simulations at once for better efficiency
                    int simulations_to_claim = std::min(4, old_sims);
                    if (active_simulations_.compare_exchange_weak(old_sims, old_sims - simulations_to_claim,
                                                                  std::memory_order_acq_rel)) {
                        int leaves_found = 0;
                        
                        // Try to collect the claimed simulations
                        for (int i = 0; i < simulations_to_claim && leaf_batch.size() < OPTIMAL_BATCH_SIZE; ++i) {
                            try {
                                auto current_root = root_;
                                
                                // Traverse and find a leaf
                                std::vector<std::shared_ptr<MCTSNode>> path;
                                auto [leaf, temp_path] = selectLeafNode(current_root);
                                path = temp_path;
                                
                                if (leaf && !leaf->isBeingEvaluated()) {
                                    if (leaf->tryMarkForEvaluation()) {
                                        // Clone state for evaluation
                                        const core::IGameState& leaf_state = leaf->getState();
                                        std::shared_ptr<core::IGameState> state_clone = cloneGameState(leaf_state);
                                        
                                        // Create pending evaluation
                                        PendingEvaluation pending;
                                        pending.node = leaf;
                                        pending.path = path;
                                        pending.state = std::move(state_clone);
                                        pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                                        pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                                        
                                        leaf_batch.push_back(std::move(pending));
                                        leaves_found++;
                                    }
                                }
                                
                                completed_simulations.fetch_add(1, std::memory_order_relaxed);
                            } catch (const std::exception& e) {
                                MCTS_LOG_ERROR("[LEAF] Error during leaf collection: " << e.what());
                            }
                        }
                        
                        if (leaves_found == 0) {
                            consecutive_empty_tries++;
                        } else {
                            consecutive_empty_tries = 0;
                        }
                    }
                }
                
                // Submit batch if we have enough leaves or timeout reached
                if (!leaf_batch.empty() && 
                    (leaf_batch.size() >= MIN_BATCH_SIZE || 
                     consecutive_empty_tries >= MAX_EMPTY_TRIES ||
                     active_simulations_.load(std::memory_order_acquire) == 0 ||
                     (std::chrono::steady_clock::now() - batch_start_time) > MAX_BATCH_COLLECTION_TIME)) {
                    
                    // Bulk enqueue for better performance
                    size_t enqueued = 0;
                    
                    // Choose correct queue based on whether shared queues are configured
                    auto& target_queue = use_shared_queues_ ? *shared_leaf_queue_ : leaf_queue_;
                    
                    // Debug: log batch details
                    if (leaf_batch.size() > 1) {
                        // Try bulk enqueue - check if it's actually enqueueing all items
                        size_t queue_size_before = target_queue.size_approx();
                        enqueued = target_queue.enqueue_bulk(std::make_move_iterator(leaf_batch.begin()), 
                                                           leaf_batch.size());
                        size_t queue_size_after = target_queue.size_approx();
                        
                        } else {
                        // Single item
                        if (target_queue.enqueue(std::move(leaf_batch[0]))) {
                            enqueued = 1;
                        }
                    }
                    
                    pending_evaluations_.fetch_add(enqueued, std::memory_order_acq_rel);
                    
                    // Always notify evaluator when we enqueue items
                    if (evaluator_ && enqueued > 0) {
                        evaluator_->notifyLeafAvailable();
                    }
                    
                    leaf_batch.clear();
                    consecutive_empty_tries = 0;
                }
                
                // Process results directly when using shared queues to prevent deadlock
                if (use_shared_queues_ && shared_result_queue_) {
                    std::pair<NetworkOutput, PendingEvaluation> result;
                    while (shared_result_queue_->try_dequeue(result)) {
                        // Process the result inline
                        auto& output = result.first;
                        auto& eval = result.second;
                        
                        if (eval.node) {
                            try {
                                eval.node->setPriorProbabilities(output.policy);
                                backPropagate(eval.path, output.value);
                                eval.node->clearEvaluationFlag();
                            } catch (...) {}
                        }
                        
                        pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
                    }
                }
                
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
        } else {
            // Use OpenMP for parallel tree traversal
            const int actual_threads = std::max(1, settings_.num_threads);  // Ensure at least 1 thread
            #pragma omp parallel num_threads(actual_threads)
            {
                // Thread-local batch for each OpenMP thread - larger for better GPU utilization
                std::vector<PendingEvaluation> thread_batch;
                const size_t OPTIMAL_THREAD_BATCH = std::max(size_t(16), size_t(settings_.batch_size / actual_threads));
                thread_batch.reserve(OPTIMAL_THREAD_BATCH);
                
                #pragma omp critical
                {
                    }
                
                int consecutive_empty = 0;
                const int MAX_EMPTY_ATTEMPTS = 3;
                
                while (active_simulations_.load(std::memory_order_acquire) > 0) {
                    // Try to claim multiple simulations at once
                    int simulations_remaining = active_simulations_.load(std::memory_order_acquire);
                    if (simulations_remaining <= 0) break;
                    
                    // Claim up to 4 simulations at once per thread
                    int to_claim = std::min(4, simulations_remaining);
                    if (active_simulations_.compare_exchange_weak(simulations_remaining, 
                                                                  simulations_remaining - to_claim,
                                                                  std::memory_order_acq_rel)) {
                        int found_leaves = 0;
                        
                        for (int sim = 0; sim < to_claim && thread_batch.size() < OPTIMAL_THREAD_BATCH; ++sim) {
                            try {
                                // Use thread ID to select root for better cache locality
                                int thread_id = omp_get_thread_num();
                                auto current_root = search_roots.empty() ? root_ : 
                                                   search_roots[thread_id % search_roots.size()];
                                
                                // Traverse and find a leaf
                                std::vector<std::shared_ptr<MCTSNode>> path;
                                auto [leaf, temp_path] = selectLeafNode(current_root);
                                path = temp_path;
                                
                                if (leaf && !leaf->isBeingEvaluated()) {
                                    if (leaf->tryMarkForEvaluation()) {
                                        // Clone state for evaluation
                                        const core::IGameState& leaf_state = leaf->getState();
                                        std::shared_ptr<core::IGameState> state_clone = cloneGameState(leaf_state);
                                        
                                        // Create pending evaluation
                                        PendingEvaluation pending;
                                        pending.node = leaf;
                                        pending.path = path;
                                        pending.state = std::move(state_clone);
                                        pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                                        pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                                        
                                        thread_batch.push_back(std::move(pending));
                                        found_leaves++;
                                    }
                                }
                                
                                completed_simulations.fetch_add(1, std::memory_order_relaxed);
                            } catch (const std::exception& e) {
                                MCTS_LOG_ERROR("[OPENMP] Error during leaf collection: " << e.what());
                            }
                        }
                        
                        consecutive_empty = (found_leaves == 0) ? consecutive_empty + 1 : 0;
                        
                        // Submit batch when we have enough or can't find more leaves
                        if (!thread_batch.empty() && 
                            (thread_batch.size() >= OPTIMAL_THREAD_BATCH || 
                             consecutive_empty >= MAX_EMPTY_ATTEMPTS ||
                             active_simulations_.load(std::memory_order_acquire) == 0)) {
                            
                            // Use lock-free bulk enqueue when possible
                            size_t to_submit = thread_batch.size();
                            auto& target_queue = use_shared_queues_ ? *shared_leaf_queue_ : leaf_queue_;
                            
                            if (to_submit > 1) {
                                // Bulk enqueue without critical section
                                size_t enqueued = target_queue.enqueue_bulk(
                                    std::make_move_iterator(thread_batch.begin()), 
                                    to_submit);
                                pending_evaluations_.fetch_add(enqueued, std::memory_order_acq_rel);
                                
                                if (enqueued > 0 && evaluator_) {
                                    evaluator_->notifyLeafAvailable();
                                }
                            } else {
                                // Single item enqueue
                                if (target_queue.enqueue(std::move(thread_batch[0]))) {
                                    pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                                    if (evaluator_) {
                                        evaluator_->notifyLeafAvailable();
                                    }
                                }
                            }
                            
                            thread_batch.clear();
                            consecutive_empty = 0;
                        }
                    }
                    
                    // Yield CPU if we can't find work
                    if (consecutive_empty >= MAX_EMPTY_ATTEMPTS) {
                        std::this_thread::yield();
                    }
                }
                
                // Submit any remaining leaves in thread-local batch
                if (!thread_batch.empty()) {
                    size_t to_submit = thread_batch.size();
                    auto& target_queue = use_shared_queues_ ? *shared_leaf_queue_ : leaf_queue_;
                    
                    // Use bulk enqueue for final submission
                    size_t enqueued = target_queue.enqueue_bulk(
                        std::make_move_iterator(thread_batch.begin()), 
                        to_submit);
                    pending_evaluations_.fetch_add(enqueued, std::memory_order_acq_rel);
                    
                    if (enqueued > 0 && evaluator_) {
                        evaluator_->notifyLeafAvailable();
                    }
                }
            }
        }
        
        // Wait for pending evaluations to complete
        // CRITICAL FIX: Notify evaluator that we're done producing leaves
        // This helps the evaluator process any remaining items in its queue
        if (evaluator_) {
            evaluator_->notifyLeafAvailable();
        }
        
        auto wait_start = std::chrono::steady_clock::now();
        int wait_log_count = 0;
        while (pending_evaluations_.load(std::memory_order_acquire) > 0) {
            if (wait_log_count < 10 || wait_log_count % 100 == 0) {
                }
            wait_log_count++;
            
            if (std::chrono::steady_clock::now() - wait_start > std::chrono::seconds(5)) {
                break;
            }
            
            // Process results directly when using shared queues - drain all available results
            if (use_shared_queues_ && shared_result_queue_) {
                std::pair<NetworkOutput, PendingEvaluation> result;
                while (shared_result_queue_->try_dequeue(result)) {
                    // Process the result inline
                    auto& output = result.first;
                    auto& eval = result.second;
                    
                    if (eval.node) {
                        try {
                            eval.node->setPriorProbabilities(output.policy);
                            backPropagate(eval.path, output.value);
                            eval.node->clearEvaluationFlag();
                        } catch (...) {}
                    }
                    
                    pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
            
            // Notify evaluator to process remaining items
            if (evaluator_) {
                evaluator_->notifyLeafAvailable();
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Aggregate results if using root parallelization
        if (settings_.use_root_parallelization && settings_.num_root_workers > 1 && root_) {
            // First ensure root is expanded
            if (root_->isLeaf() && !root_->isTerminal()) {
                root_->expand(settings_.use_progressive_widening,
                             settings_.progressive_widening_c,
                             settings_.progressive_widening_k);
            }
            
            // Create a map of action to aggregated statistics
            std::unordered_map<int, int> action_visit_counts;
            std::unordered_map<int, double> action_value_sums;
            
            // Collect statistics from all search roots
            for (size_t i = 0; i < search_roots.size(); i++) {
                const auto& search_root = search_roots[i];
                if (!search_root) {
                    continue;
                }
                
                auto search_children = search_root->getChildren();
                for (const auto& child : search_children) {
                    int action = child->getAction();
                    int visits = child->getVisitCount();
                    float value = child->getValue();
                    
                    action_visit_counts[action] += visits;
                    action_value_sums[action] += visits * value;
                    
                    }
            }
            
            // Apply aggregated statistics to the main root's children
            auto root_children = root_->getChildren();
            for (auto& child : root_children) {
                int action = child->getAction();
                auto visits_it = action_visit_counts.find(action);
                
                if (visits_it != action_visit_counts.end() && visits_it->second > 0) {
                    int total_visits = visits_it->second;
                    double total_value = action_value_sums[action];
                    float avg_value = total_value / total_visits;
                    
                    // Update the main root's child with aggregated statistics
                    for (int i = 0; i < total_visits; i++) {
                        child->update(avg_value);
                    }
                }
            }
            
            }
        
        // Log final status
        // [SEARCH] Final status after search completion
        
        // Signal workers to stop
        workers_active_.store(false, std::memory_order_release);
        // Commented out - not used in OpenMP version
        // cv_.notify_all();
        // batch_cv_.notify_all();
        
        // Record search statistics
        if (root_) {
            last_stats_.total_nodes = countTreeNodes(root_);
            last_stats_.max_depth = calculateMaxDepth(root_);
        }
        
        // Mark search as completed
        search_running_.store(false, std::memory_order_release);
        
    } catch (const std::exception& e) {
        // Log the error
        // Exception during MCTS search
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow to allow caller to handle the error
        throw;
    } catch (...) {
        // Handle unknown exceptions
        // Unknown exception during MCTS search
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow with a more descriptive message
        throw std::runtime_error("Unknown error occurred during MCTS search");
    }
}

// Commented out - replaced with OpenMP implementation
// void MCTSEngine::treeTraversalWorker(int worker_id) {
//     // Old implementation replaced with OpenMP parallel loops
// }

void MCTSEngine::traverseTree(std::shared_ptr<MCTSNode> root) {
    if (!root) return;
    
    try {
        // Selection phase
        auto [leaf, path] = selectLeafNode(root);
        if (!leaf) return;
        
        // Expansion phase - never block
        if (!leaf->isTerminal() && leaf->isLeaf()) {
            // Try to expand and mark for evaluation atomically
            bool expand_success = false;
            bool should_evaluate = false;
            
            // CRITICAL: Prevent race condition by checking and marking evaluation BEFORE expand
            if (leaf->tryMarkForEvaluation()) {
                // We got exclusive rights to evaluate this node
                should_evaluate = true;
                try {
                    leaf->expand();
                    expand_success = true;
                } catch (...) {
                    // If expansion fails, clear the evaluation flag
                    leaf->clearEvaluationFlag();
                    should_evaluate = false;
                }
            } else {
                // Another thread is already evaluating this node
                return;
            }
            
            // Only queue for evaluation if we successfully marked and expanded
            if (should_evaluate && expand_success) {
                // Create evaluation request
                auto state_clone = cloneGameState(leaf->getState());
                if (state_clone) {
                    PendingEvaluation pending;
                    pending.node = leaf;
                    pending.path = std::move(path);
                    pending.state = state_clone;
                    pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                    pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                    
                    // Submit to leaf queue with proper move semantics
                    if (leaf_queue_.enqueue(std::move(pending))) {
                        // Increment pending evaluations count (FIX: only once per leaf)
                        pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                        if (evaluator_) { // Notify evaluator that a new leaf is available
                            evaluator_->notifyLeafAvailable();
                        }
                    } else {
                        MCTS_LOG_ERROR("[TRAVERSE] Failed to enqueue evaluation request");
                        // Clear the flag since we failed to enqueue
                        leaf->clearEvaluationFlag();
                    }
                } else {
                    // Clear the flag since we failed to clone the state
                    leaf->clearEvaluationFlag();
                }
            }
            // If tryMarkForEvaluation() returned false, another thread is already evaluating this node
        } else if (leaf->isTerminal()) {
            // Handle terminal nodes immediately
            float value = 0.0f;
            auto result = leaf->getState().getGameResult();
            int current_player = leaf->getState().getCurrentPlayer();
            
            if (result == core::GameResult::WIN_PLAYER1) {
                value = current_player == 1 ? 1.0f : -1.0f;
            } else if (result == core::GameResult::WIN_PLAYER2) {
                value = current_player == 2 ? 1.0f : -1.0f;
            }
            
            backPropagate(path, value);
        }
    } catch (const std::exception& e) {
        // Ignore errors and continue
    }
}

// Removed batchAccumulatorWorker - functionality now integrated into MCTSEvaluator

void MCTSEngine::resultDistributorWorker() {
    if (use_shared_queues_ && !shared_result_queue_) {
        }
    
    pthread_setname_np(pthread_self(), "ResultDist");
    
    try {
        std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_batch;
        result_batch.reserve(64);  // Larger batch for better throughput
        
        while (!shutdown_.load(std::memory_order_acquire) || 
               (use_shared_queues_ ? shared_result_queue_->size_approx() : result_queue_.size_approx()) > 0) {
            
            // Check for shutdown more frequently
            if (shutdown_.load(std::memory_order_acquire) && 
                (use_shared_queues_ ? shared_result_queue_->size_approx() : result_queue_.size_approx()) == 0) {
                break;
            }
            
            // Aggressive bulk dequeue for maximum efficiency
            result_batch.clear();
            
            // Both shared and internal queues use the same type now
            size_t dequeued = 0;
            if (use_shared_queues_) {
                // Use shared queue directly (no cast needed - same type)
                dequeued = shared_result_queue_->try_dequeue_bulk(result_batch.data(), 64);
                
                if (dequeued == 0) {
                    std::pair<NetworkOutput, PendingEvaluation> result_pair;
                    if (shared_result_queue_->try_dequeue(result_pair)) {
                        result_batch.push_back(std::move(result_pair));
                        dequeued = 1;
                    }
                }
            } else {
                // Use internal queue
                dequeued = result_queue_.try_dequeue_bulk(result_batch.data(), 64);
                
                if (dequeued == 0) {
                    // Fall back to single dequeue
                    std::pair<NetworkOutput, PendingEvaluation> result_pair;
                    if (result_queue_.try_dequeue(result_pair)) {
                        result_batch.push_back(std::move(result_pair));
                        dequeued = 1;
                    }
                }
            }
            
            if (dequeued > 0) {
                result_batch.resize(dequeued);  // Adjust size to actual dequeued count
                
                // Report batch processing
                // Process results in parallel with OpenMP
                #pragma omp parallel for schedule(dynamic, 4)
                for (size_t i = 0; i < result_batch.size(); ++i) {
                    // Check if we should stop processing before accessing any node
                    if (shutdown_.load(std::memory_order_acquire)) {
                        continue;
                    }
                    
                    auto& [output, eval] = result_batch[i];
                    
                    // Update the node with neural network results
                    if (eval.node) {
                        // Double-check shutdown before accessing node
                        if (shutdown_.load(std::memory_order_acquire)) {
                            continue;
                        }
                        
                        // Check if the node is still valid (not destroyed)
                        try {
                            eval.node->setPriorProbabilities(output.policy);
                            
                            // Perform backpropagation
                            backPropagate(eval.path, output.value);
                            
                            // Clear the evaluation flag now that we're done
                            eval.node->clearEvaluationFlag();
                        } catch (const std::exception& e) {
                            // Node might have been destroyed, skip it
                            MCTS_LOG_ERROR("[RESULT] Error processing node: " << e.what());
                            // Try to clear flag even on error  
                            try { eval.node->clearEvaluationFlag(); } catch (...) {}
                        } catch (...) {
                            // Node might have been destroyed, skip it
                            MCTS_LOG_ERROR("[RESULT] Unknown error processing node");
                            // Try to clear flag even on error
                            try { eval.node->clearEvaluationFlag(); } catch (...) {}
                        }
                    }
                    
                    pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
                    total_results_processed_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                // Adaptive waiting - yield CPU when queue is empty
                // CRITICAL DEBUG: Log when result queue is empty
                static int empty_count = 0;
                // Count empty cycles
                if (++empty_count < 10) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    empty_count = 0;
                }
            }
        }
        
        MCTS_LOG_VERBOSE("[RESULT] Result distributor worker stopped. Total processed: " 
                  << total_results_processed_.load());
    } catch (const std::exception& e) {
        MCTS_LOG_ERROR("[RESULT] Fatal exception: " << e.what());
    } catch (...) {
        MCTS_LOG_ERROR("[RESULT] Fatal unknown exception");
    }
}

// REMOVED: Old runSimulation method with raw pointer parameter
// This entire method should be deleted as it's no longer used
#if 0
void MCTSEngine::runSimulation(MCTSNode* root) {
    if (!root) {
        // MCTSEngine::runSimulation - Null root pointer!
        return;
    }
    
    // Validate root state before proceeding
    try {
        if (!root->getState().validate()) {
            // MCTSEngine::runSimulation - Root state invalid!
            return;
        }
    } catch (const std::exception& e) {
        // MCTSEngine::runSimulation - Error validating root state
        return;
    } catch (...) {
        // MCTSEngine::runSimulation - Unknown error validating root state
        return;
    }
    
    auto sim_start_time = std::chrono::high_resolution_clock::now();
    long long selection_time_us = 0;
    long long evaluation_time_us = 0;
    long long backprop_time_us = 0;

    try {
        auto selection_start_time = std::chrono::high_resolution_clock::now();
        // Selection phase - find a leaf node
        auto [leaf, path] = selectLeafNode(root);
        auto selection_end_time = std::chrono::high_resolution_clock::now();
        selection_time_us = std::chrono::duration_cast<std::chrono::microseconds>(selection_end_time - selection_start_time).count();

        if (!leaf) {
            // Null leaf indicates a node with pending evaluation was encountered
            // Don't decrement active_simulations since the simulation is still "active"
            // waiting for the evaluation to complete
            return;
        }

        // Expansion and evaluation phase
        float value = 0.0f;
        auto evaluation_start_time = std::chrono::high_resolution_clock::now();
        try {
            // We need to handle terminal states differently
            if (leaf->isTerminal()) {
                // For terminal states, value depends on game result
                try {
                    auto game_result = leaf->getState().getGameResult();
                    int current_player = leaf->getState().getCurrentPlayer();
                    
                    // Validate values further
                    if (current_player != 1 && current_player != 2) {
                        throw std::runtime_error("Invalid current player");
                    }
                    
                    if (game_result == core::GameResult::WIN_PLAYER1) {
                        value = current_player == 1 ? 1.0f : -1.0f;
                    } else if (game_result == core::GameResult::WIN_PLAYER2) {
                        value = current_player == 2 ? 1.0f : -1.0f;
                    } else {
                        value = 0.0f; // Draw
                    }
                } catch (...) {
                    // If any exception happens, use default value
                    value = 0.0f;
                }
            } else {
                // For non-terminal states, expand and evaluate
                try {
                    value = expandAndEvaluate(leaf, path);
                } catch (...) {
                    // If any exception happens, use default value
                    value = 0.0f;
                }
            }
        } catch (const std::bad_alloc& e) {
            #if MCTS_DEBUG
            // Commented out: Debug error message about memory allocation during expansion/evaluation
            #endif
            value = 0.0f;  // Use a default value
        } catch (const std::exception& e) {
            #if MCTS_DEBUG
            // Commented out: Debug error message during expansion/evaluation
            #endif
            value = 0.0f;  // Use a default value
        }
        auto evaluation_end_time = std::chrono::high_resolution_clock::now();
        evaluation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(evaluation_end_time - evaluation_start_time).count();

        // Backpropagation phase
        auto backprop_start_time = std::chrono::high_resolution_clock::now();
        backPropagate(path, value);
        auto backprop_end_time = std::chrono::high_resolution_clock::now();
        backprop_time_us = std::chrono::duration_cast<std::chrono::microseconds>(backprop_end_time - backprop_start_time).count();

    } catch (const std::exception& e) {
        #if MCTS_DEBUG
        // Commented out: Debug error message during simulation
        #endif
        // Ensure simulation count is decremented even on exception
        // active_simulations_.fetch_sub(1, std::memory_order_release); // Already handled by the worker loop that calls this
        // cv_.notify_all();
    }
    auto sim_end_time = std::chrono::high_resolution_clock::now();
    long long total_sim_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sim_end_time - sim_start_time).count();
    // Log only if total time is significant to reduce log spam
    if (total_sim_time_us > 5000) { // e.g. > 5ms
        // [SIM_PROFILE] Performance tracking disabled
    }
}
#endif // End of disabled old runSimulation method

std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> MCTSEngine::selectLeafNode(std::shared_ptr<MCTSNode> root) {
    std::vector<std::shared_ptr<MCTSNode>> path;
    std::shared_ptr<MCTSNode> current_node_in_traversal = root;
    
    if (!current_node_in_traversal) {
        return {nullptr, path};
    }
    
    path.push_back(current_node_in_traversal);

    static int select_count = 0;
    if (++select_count < 10) {
        }

    while (current_node_in_traversal && !current_node_in_traversal->isLeaf() && !current_node_in_traversal->isTerminal()) {
        // Check if current node has pending evaluation
        if (current_node_in_traversal->hasPendingEvaluation()) {
            // Remove virtual loss from path since we're not going deeper
            for (auto& node : path) {
                node->removeVirtualLoss();
            }
            return {nullptr, path}; // Return null leaf to indicate pending
        }
        
        std::shared_ptr<MCTSNode> parent_for_selection = current_node_in_traversal;
        parent_for_selection->addVirtualLoss();

        std::shared_ptr<MCTSNode> selected_child = parent_for_selection->selectChild(
            settings_.exploration_constant, settings_.use_rave, settings_.rave_constant);
        
        if (!selected_child) {
            // If no child is selected (e.g., all children are terminal or have issues),
            // remove virtual loss from parent and break to return parent as leaf.
            parent_for_selection->removeVirtualLoss();
            break;  
        }

        // Tentatively, the traversal will proceed with selected_child.
        // Apply virtual loss to it for this traversal step.
        selected_child->addVirtualLoss();
        
        std::shared_ptr<MCTSNode> node_to_use_for_traversal = selected_child;

        if (use_transposition_table_ && transposition_table_ && selected_child && !selected_child->isTerminal()) {
            try {
                uint64_t hash = selected_child->getState().getHash();
                std::shared_ptr<MCTSNode> transposition_entry = transposition_table_->get(hash);

                if (transposition_entry && transposition_entry != selected_child) {
                    bool valid_transposition = false;
                    try {
                        int visits = transposition_entry->getVisitCount();
                        if (visits >= 0 && visits < 100000) { // Basic sanity check
                            const core::IGameState& trans_state = transposition_entry->getState();
                            valid_transposition = trans_state.validate() && trans_state.getHash() == hash;
                        }
                    } catch (...) {
                        valid_transposition = false;
                    }
                    
                    if (valid_transposition) {
                        // FIX: Prevent orphaned nodes by properly merging transposition references
                        // Update parent's reference from selected_child to transposition_entry.
                        // This ensures the tree structure points to the canonical TT node.
                        if (parent_for_selection->updateChildReference(selected_child, transposition_entry)) {
                        
                            // Virtual loss needs to be correct: selected_child's VL (from this step) removed,
                            // transposition_entry's VL (for this step) added.
                            selected_child->removeVirtualLoss(); // Remove VL from the originally selected child.
                            node_to_use_for_traversal = transposition_entry;
                            node_to_use_for_traversal->addVirtualLoss(); // Ensure VL is on the node we are actually using for traversal.
                            
                            // FIX: Clear evaluation flag from orphaned node to prevent memory leaks
                            if (selected_child->tryMarkForEvaluation()) {
                                selected_child->clearEvaluationFlag();
                            }
                        } else {
                            // Failed to update reference, continue with original child
                            // This prevents node from becoming orphaned
                        }
                    }
                }
            } catch (...) {
                // If any exception occurs during transposition table lookup or use,
                // continue with the current selected_child.
            }
        }

        current_node_in_traversal = node_to_use_for_traversal;
        path.push_back(current_node_in_traversal);
    }

    return {current_node_in_traversal, path};
}

float MCTSEngine::expandAndEvaluate(std::shared_ptr<MCTSNode> leaf, const std::vector<std::shared_ptr<MCTSNode>>& path) {
    if (!leaf) {
        return 0.0f;
    }
    
    // Handle terminal states
    if (leaf->isTerminal()) {
        try {
            auto result = leaf->getState().getGameResult();
            float value = 0.0f;
            if (result == core::GameResult::WIN_PLAYER1) {
                value = leaf->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
            } else if (result == core::GameResult::WIN_PLAYER2) {
                value = leaf->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
            }
            return value;
        } catch (const std::exception& e) {
            // Commented out: Error evaluating terminal state with error message
            return 0.0f;
        }
    }
    
    // Expand the leaf node
    try {
        leaf->expand();
    } catch (const std::exception& e) {
        // Commented out: Error expanding leaf node with error message
        return 0.0f;
    }
    
    // Store in transposition table if enabled
    if (use_transposition_table_ && transposition_table_) {
        try {
            uint64_t hash = leaf->getState().getHash();
            transposition_table_->store(hash, std::weak_ptr<MCTSNode>(leaf), path.size());
            
            #if MCTS_DEBUG
            // Commented out: Debug printing about storing expanded node in transposition table with hash and depth
            #endif
        } catch (const std::exception& e) {
            // Commented out: Error storing in transposition table with error message
            // Continue, this is not critical
        }
    }
    
    // Check children for transpositions after expansion and merge if found
    if (use_transposition_table_ && transposition_table_ && !leaf->getChildren().empty()) {
        try {
            auto& children = leaf->getChildren();
            for (size_t i = 0; i < children.size(); ++i) {
                if (!children[i]) continue;
                
                uint64_t child_hash = children[i]->getState().getHash();
                std::shared_ptr<MCTSNode> existing = transposition_table_->get(child_hash);
                
                if (existing && existing != children[i]) {
                    // Found an existing node with the same state
                    // Replace the newly created child with the existing one
                    children[i] = existing;
                    
                    // Store the existing node in the transposition table if not already there
                    transposition_table_->store(child_hash, std::weak_ptr<MCTSNode>(existing), path.size() + 1);
                } else {
                    // Store the new child in the transposition table
                    transposition_table_->store(child_hash, std::weak_ptr<MCTSNode>(children[i]), path.size() + 1);
                }
            }
        } catch (const std::exception& e) {
            // Continue if any error occurs during transposition checking
        }
    }
    
    // If leaf has no children after expansion, return a default value
    if (leaf->getChildren().empty()) {
        return 0.0f;
    }
    
    // Evaluate with the neural network
    try {
        // Special fast path for serial mode (no worker threads)
        if (settings_.num_threads == 0) {
            auto state_clone = cloneGameState(leaf->getState());
            if (!state_clone) {
                throw std::runtime_error("Failed to clone state for evaluation");
            }
            
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.push_back(std::unique_ptr<core::IGameState>(state_clone->clone().release()));
            
            auto outputs = evaluator_->getInferenceFunction()(states);
            if (!outputs.empty()) {
                leaf->setPriorProbabilities(outputs[0].policy);
                return outputs[0].value;
            } else {
                // Fallback to uniform policy on error
                int action_space_size = leaf->getState().getActionSpaceSize();
                std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                leaf->setPriorProbabilities(uniform_policy);
                return 0.0f;
            }
        } else {
            // For true leaf parallelization, queue the evaluation and return immediately
            
            // Check if this node is already being evaluated
            if (leaf->tryMarkForEvaluation()) {
                // We have exclusive rights to evaluate this node
                
                // Clone state for evaluation
                auto state_clone = cloneGameState(leaf->getState());
                if (!state_clone) {
                    leaf->clearEvaluationFlag();
                    throw std::runtime_error("Failed to clone state for evaluation");
                }
                
                // Create a pending evaluation entry
                PendingEvaluation pending;
                pending.node = leaf;
                pending.path = path;
                pending.state = state_clone;
                pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                
                // Submit to appropriate queue for batching
                bool enqueue_success = false;
                if (use_shared_queues_ && shared_leaf_queue_) {
                    enqueue_success = shared_leaf_queue_->enqueue(std::move(pending));
                    if (enqueue_success) {
                        // CRITICAL DEBUG: Log successful enqueue to shared queue
                        static int shared_enqueue_count = 0;
                        if (shared_enqueue_count < 10) {
                            }
                    }
                } else {
                    enqueue_success = leaf_queue_.enqueue(std::move(pending));
                    if (enqueue_success) {
                        // CRITICAL DEBUG: Log successful enqueue to local queue
                        static int enqueue_count = 0;
                        if (enqueue_count < 10) {
                            }
                    }
                }
                
                if (enqueue_success) {
                    // Increment pending evaluations
                    pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                    
                    // Apply virtual loss to prevent other threads from selecting this path
                    leaf->applyVirtualLoss(settings_.virtual_loss);
                    
                    // Notify evaluator
                    if (!use_shared_queues_ && evaluator_) {
                        evaluator_->notifyLeafAvailable();
                    }
                    // For shared queues, the notification is handled by the shared evaluator
                } else {
                    MCTS_LOG_ERROR("[expandAndEvaluate] Failed to enqueue evaluation request");
                    leaf->clearEvaluationFlag();
                }
            }
            
            // Return dummy value - actual value will be backpropagated when evaluation completes
            return 0.0f;
        }
    } catch (const std::exception& e) {
        // Commented out: Error during neural network evaluation with error message
        
        // Fallback to uniform policy on error
        try {
            int action_space_size = leaf->getState().getActionSpaceSize();
            std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
            leaf->setPriorProbabilities(uniform_policy);
        } catch (...) {
            // Ignore any further errors
        }
        return 0.0f;
    }
}

void MCTSEngine::backPropagate(std::vector<std::shared_ptr<MCTSNode>>& path, float value) {
    // Value alternates sign as we move up the tree (perspective changes)
    bool invert = false;
    
    // RAVE update preparation - collect all actions in the path
    std::vector<int> path_actions;
    if (settings_.use_rave) {
        for (const auto& node : path) {
            if (node->getAction() != -1) {
                path_actions.push_back(node->getAction());
            }
        }
    }
    
    // Process nodes in reverse order (from leaf to root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto node = *it;
        float update_value = invert ? -value : value;
        
        // Remove virtual loss and update node statistics
        node->removeVirtualLoss();
        node->update(update_value);
        
        // RAVE update - update all children that match actions in the path
        if (settings_.use_rave && node->getChildren().size() > 0) {
            for (auto& child : node->getChildren()) {
                int child_action = child->getAction();
                
                // Check if this action appears later in the path (RAVE principle)
                for (int path_action : path_actions) {
                    if (child_action == path_action) {
                        // Update RAVE value for this child
                        float rave_value = invert ? -value : value;
                        child->updateRAVE(rave_value);
                        break; // Only update once per child
                    }
                }
            }
        }
        
        // Alternate perspective for next level
        invert = !invert;
    }
}

void MCTSEngine::processPendingEvaluations(std::shared_ptr<MCTSNode> root) {
    if (!root || !node_tracker_) return;
    
    // Process results from node tracker
    NodeTracker::EvaluationResult result;
    while (node_tracker_->getNextResult(result)) {
        if (result.node) {
            // Apply the result
            result.node->setPriorProbabilities(result.output.policy);
            
            // Remove the virtual loss that was applied during evaluation
            result.node->removeVirtualLoss();
            
            // Clear pending evaluation flag
            result.node->clearPendingEvaluation();
            
            // Backpropagate the value using the stored path
            if (!result.path.empty()) {
                backPropagate(const_cast<std::vector<std::shared_ptr<MCTSNode>>&>(result.path), result.output.value);
            }
            
            // Decrement active simulations since this evaluation is complete
            active_simulations_.fetch_sub(1, std::memory_order_release);
            // cv_.notify_all(); // Not needed in OpenMP version
        }
    }
}

std::vector<float> MCTSEngine::getActionProbabilities(std::shared_ptr<MCTSNode> root, float temperature) {
    if (!root || root->getChildren().empty()) {
        return std::vector<float>();
    }

    // Get actions and visit counts
    auto& actions = root->getActions();
    auto& children = root->getChildren();
    
    std::vector<float> counts;
    counts.reserve(children.size());

    for (auto child : children) {
        counts.push_back(static_cast<float>(child->getVisitCount()));
    }

    // Handle different temperature regimes
    std::vector<float> probabilities;
    probabilities.reserve(counts.size());

    if (temperature < 0.01f) {
        // Temperature near zero: deterministic selection - pick the move with highest visits
        auto max_it = std::max_element(counts.begin(), counts.end());
        size_t max_idx = std::distance(counts.begin(), max_it);
        
        // Set all probabilities to 0 except the highest
        probabilities.resize(counts.size(), 0.0f);
        probabilities[max_idx] = 1.0f;
    } else {
        // Apply temperature scaling: counts ^ (1/temperature)
        float sum = 0.0f;
        
        // First find the maximum count for numerical stability
        float max_count = *std::max_element(counts.begin(), counts.end());
        
        if (max_count <= 0.0f) {
            // If all counts are 0, use uniform distribution
            float uniform_prob = 1.0f / counts.size();
            probabilities.resize(counts.size(), uniform_prob);
        } else {
            // Compute the power of (count/max_count) for better numerical stability
            for (float count : counts) {
                float scaled_count = 0.0f;
                if (count > 0.0f) {
                    scaled_count = std::pow(count / max_count, 1.0f / temperature);
                }
                probabilities.push_back(scaled_count);
                sum += scaled_count;
            }
            
            // Normalize
            if (sum > 0.0f) {
                for (auto& prob : probabilities) {
                    prob /= sum;
                }
            } else {
                // Fallback to uniform if sum is zero
                float uniform_prob = 1.0f / counts.size();
                std::fill(probabilities.begin(), probabilities.end(), uniform_prob);
            }
        }
    }

    // Create full action space probabilities
    std::vector<float> action_probabilities(root->getState().getActionSpaceSize(), 0.0f);

    // Map child indices to action indices
    for (size_t i = 0; i < actions.size(); ++i) {
        int action = actions[i];
        if (action >= 0 && action < static_cast<int>(action_probabilities.size())) {
            action_probabilities[action] = probabilities[i];
        }
    }

    #if MCTS_DEBUG
    // Commented out: Debug printing of action probabilities (top 5) with sorting and formatted output
    #endif

    return action_probabilities;
}

void MCTSEngine::addDirichletNoise(std::shared_ptr<MCTSNode> root) {
    if (!root) {
        return;
    }
    
    // Expand root node if it's not already expanded
    if (root->isLeaf() && !root->isTerminal()) {
        root->expand();
        
        if (root->getChildren().empty()) {
            return;  // No children to add noise to
        }
        
        // Get prior probabilities for the root node
        try {
            auto state_clone = cloneGameState(root->getState());
            if (settings_.num_threads == 0) {
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.push_back(std::unique_ptr<core::IGameState>(state_clone->clone().release()));
                auto outputs = evaluator_->getInferenceFunction()(states);
                if (!outputs.empty()) {
                    root->setPriorProbabilities(outputs[0].policy);
                } else {
                    int action_space_size = root->getState().getActionSpaceSize();
                    std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                    root->setPriorProbabilities(uniform_policy);
                }
            } else {
                // Convert shared_ptr to unique_ptr for evaluator
                auto unique_clone = std::unique_ptr<core::IGameState>(state_clone->clone().release());
                auto future = evaluator_->evaluateState(root, std::move(unique_clone));
                auto status = future.wait_for(std::chrono::seconds(2));
                if (status == std::future_status::ready) {
                    auto result = future.get();
                    root->setPriorProbabilities(result.policy);
                } else {
                    // Timed out, use uniform policy
                    int action_space_size = root->getState().getActionSpaceSize();
                    std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                    root->setPriorProbabilities(uniform_policy);
                }
            }
        } catch (const std::exception& e) {
            #if MCTS_DEBUG
            // Commented out: Debug error message about getting prior probabilities for root
            #endif
            
            // On error, use uniform policy
            int action_space_size = root->getState().getActionSpaceSize();
            std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
            root->setPriorProbabilities(uniform_policy);
        }
    }
    
    if (root->getChildren().empty()) {
        return;  // No children to add noise to
    }
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(settings_.dirichlet_alpha, 1.0f);
    std::vector<float> noise;
    noise.reserve(root->getChildren().size());
    
    for (size_t i = 0; i < root->getChildren().size(); ++i) {
        noise.push_back(gamma(random_engine_));
    }
    
    // Normalize noise
    float sum = std::accumulate(noise.begin(), noise.end(), 0.0f);
    if (sum > 0.0f) {
        for (auto& n : noise) {
            n /= sum;
        }
    } else {
        // If sum is zero, use uniform noise
        float uniform_noise = 1.0f / noise.size();
        std::fill(noise.begin(), noise.end(), uniform_noise);
    }
    
    #if MCTS_DEBUG
    // Commented out: Debug printing about adding Dirichlet noise to root node with epsilon value
    #endif
    
    // Apply noise to children's prior probabilities
    for (size_t i = 0; i < root->getChildren().size(); ++i) {
        std::shared_ptr<MCTSNode> child = root->getChildren()[i];
        float prior = child->getPriorProbability();
        float noisy_prior = (1.0f - settings_.dirichlet_epsilon) * prior + 
                           settings_.dirichlet_epsilon * noise[i];
        child->setPriorProbability(noisy_prior);
    }
}

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

std::shared_ptr<core::IGameState> MCTSEngine::cloneGameState(const core::IGameState& state) {
    try {
        // Add debug logging but reduce frequency
        static std::atomic<int> clone_count(0);
        static std::atomic<int> null_count(0);
        int count = clone_count.fetch_add(1, std::memory_order_relaxed);
        
        if (count < 5 || count % 1000 == 0) {
            MCTS_LOG_VERBOSE("[MCTS] cloneGameState called " << count << " times, pool_enabled=" 
                     << game_state_pool_enabled_ << ", state type=" << typeid(state).name());
        }
        
        // CRITICAL FIX: Initialize pool if not already done
        if (game_state_pool_enabled_) {
            auto game_type = state.getGameType();
            auto& pool_manager = utils::GameStatePoolManager::getInstance();
            
            if (!pool_manager.hasPool(game_type)) {
                // Initialize pool with adequate size
                size_t pool_size = std::max(size_t(2000), size_t(settings_.num_simulations * 2));
                try {
                    pool_manager.initializePool(game_type, pool_size);
                    MCTS_LOG_DEBUG("[MCTS] Initialized GameState pool for " << core::gameTypeToString(game_type) 
                                  << " with size " << pool_size);
                } catch (const std::exception& e) {
                    MCTS_LOG_ERROR("[MCTS] Failed to initialize pool: " << e.what());
                    game_state_pool_enabled_ = false; // Disable pool on failure
                }
            }
            
            // Try pool-based cloning if pool is available
            if (game_state_pool_enabled_ && pool_manager.hasPool(game_type)) {
                auto cloned = pool_manager.cloneState(state);
                if (cloned) {
                    return std::shared_ptr<core::IGameState>(cloned.release());
                }
                MCTS_LOG_ERROR("[MCTS] Pool cloning failed, falling back to regular clone");
            }
        }
        
        // Regular cloning with null check
        auto cloned = state.clone();
        if (!cloned) {
            int nc = null_count.fetch_add(1, std::memory_order_relaxed);
            MCTS_LOG_ERROR("[MCTS] ERROR: State clone returned null for state type=" 
                     << typeid(state).name() << " (null count: " << nc + 1 << ")");
            
            // CRITICAL: Throw exception instead of returning null
            throw std::runtime_error("Failed to clone game state - clone() returned null");
        }
        
        // Validate cloned state
        if (!cloned->validate()) {
            MCTS_LOG_ERROR("[MCTS] ERROR: Cloned state failed validation");
            throw std::runtime_error("Cloned state is invalid");
        }
        
        return std::shared_ptr<core::IGameState>(cloned.release());
        
    } catch (const std::exception& e) {
        MCTS_LOG_ERROR("[MCTS] ERROR: Exception in cloneGameState: " << e.what() 
                 << " for state type=" << typeid(state).name());
        throw; // Re-throw to let caller handle
    } catch (...) {
        MCTS_LOG_ERROR("[MCTS] ERROR: Unknown exception in cloneGameState for state type=" 
                 << typeid(state).name());
        throw std::runtime_error("Unknown error in cloneGameState");
    }
}

void MCTSEngine::forceCleanup() {
    MCTS_LOG_DEBUG("[MCTS] Forcing memory cleanup");
    
    // Clear transposition table
    if (use_transposition_table_) {
        transposition_table_->clear();
    }
    
    // Clear node tracker's pending evaluations
    if (node_tracker_) {
        node_tracker_->clear();
    }
    
    // Clear game state pool
    if (game_state_pool_enabled_) {
        auto& pool_manager = utils::GameStatePoolManager::getInstance();
        pool_manager.clearAllPools();
    }
    
    // Note: We cannot recreate the evaluator without the neural network
    // Just let it continue with its existing state
    
    // Force garbage collection for any remaining objects
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    MCTS_LOG_DEBUG("[MCTS] Memory cleanup completed");
}

void MCTSEngine::setSharedExternalQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* leaf_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* result_queue,
        std::function<void()> notify_fn) {
    shared_leaf_queue_ = leaf_queue;
    shared_result_queue_ = result_queue;
    use_shared_queues_ = true;
    
    // Also configure the evaluator to use these shared queues
    if (evaluator_) {
        evaluator_->setExternalQueues(shared_leaf_queue_, shared_result_queue_, notify_fn);
        
        // The engine's own evaluator should be marked as started since we're using external shared management
        evaluator_started_.store(true, std::memory_order_release);
        } else {
        }
    
    // Debug: Verify the queue pointers are stored correctly
    }



} // namespace mcts
} // namespace alphazero