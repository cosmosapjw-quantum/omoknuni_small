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

// Configurable debug level
#define MCTS_DEBUG 1
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
      game_state_pool_enabled_(true) {  // Enable game state pool
    
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
            std::min(settings.batch_timeout, std::chrono::milliseconds(10)));  // Cap timeout at 10ms
            
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
      game_state_pool_enabled_(true) {  // Enable game state pool
    
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
    // Check if already started
    if (evaluator_started_) {
        return true;
    }
    
    try {
        // Make sure evaluator exists
        if (!evaluator_) {
            // MCTSEngine::ensureEvaluatorStarted - Evaluator is null
            return false;
        }
        
        // Start the evaluator
        evaluator_->start();
        evaluator_started_ = true;
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
    if (evaluator_started_) {
        try {
            evaluator_->stop();
            evaluator_started_ = false;
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

MCTSEngine::MCTSEngine(MCTSEngine&& other) noexcept
    : settings_(std::move(other.settings_)),
      last_stats_(std::move(other.last_stats_)),
      evaluator_(std::move(other.evaluator_)),
      root_(std::move(other.root_)),
      shutdown_(other.shutdown_.load()),
      active_simulations_(other.active_simulations_.load()),
      search_running_(other.search_running_.load()),
      random_engine_(std::move(other.random_engine_)),
      transposition_table_(std::move(other.transposition_table_)),
      use_transposition_table_(other.use_transposition_table_),
      evaluator_started_(other.evaluator_started_),
      pending_evaluations_(other.pending_evaluations_.load()),
      batch_counter_(other.batch_counter_.load()),
      total_leaves_generated_(other.total_leaves_generated_.load()),
      total_results_processed_(other.total_results_processed_.load()),
      leaf_queue_(std::move(other.leaf_queue_)),
      batch_queue_(std::move(other.batch_queue_)),
      result_queue_(std::move(other.result_queue_)),
      batch_accumulator_worker_(std::move(other.batch_accumulator_worker_)),
      result_distributor_worker_(std::move(other.result_distributor_worker_)),
      tree_traversal_workers_(std::move(other.tree_traversal_workers_)),
      workers_active_(other.workers_active_.load()) {
    
    // Validate the moved evaluator
    if (!evaluator_) {
        // WARNING: evaluator_ is null after move constructor
    }
    
    // Properly clean up other's threads before clearing
    other.shutdown_ = true;
    other.workers_active_ = false;
    other.cv_.notify_all();
    other.batch_cv_.notify_all();
    other.result_cv_.notify_all();
    
    // Join other's threads before clearing
    for (auto& thread : other.tree_traversal_workers_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (other.batch_accumulator_worker_.joinable()) {
        other.batch_accumulator_worker_.join();
    }
    if (other.result_distributor_worker_.joinable()) {
        other.result_distributor_worker_.join();
    }
    
    // Now safe to clear
    other.tree_traversal_workers_.clear();
    other.search_running_ = false;
    other.active_simulations_ = 0;
    other.evaluator_started_ = false;
}

MCTSEngine& MCTSEngine::operator=(MCTSEngine&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        shutdown_ = true;
        workers_active_ = false;
        cv_.notify_all();
        batch_cv_.notify_all();
        result_cv_.notify_all();
        
        // Join specialized workers
        if (batch_accumulator_worker_.joinable()) {
            batch_accumulator_worker_.join();
        }
        if (result_distributor_worker_.joinable()) {
            result_distributor_worker_.join();
        }
        for (auto& thread : tree_traversal_workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
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
        evaluator_started_ = other.evaluator_started_;
        pending_evaluations_ = other.pending_evaluations_.load();
        batch_counter_ = other.batch_counter_.load();
        total_leaves_generated_ = other.total_leaves_generated_.load();
        total_results_processed_ = other.total_results_processed_.load();
        leaf_queue_ = std::move(other.leaf_queue_);
        batch_queue_ = std::move(other.batch_queue_);
        result_queue_ = std::move(other.result_queue_);
        batch_accumulator_worker_ = std::move(other.batch_accumulator_worker_);
        result_distributor_worker_ = std::move(other.result_distributor_worker_);
        tree_traversal_workers_ = std::move(other.tree_traversal_workers_);
        workers_active_ = other.workers_active_.load();
        
        // Validate the moved evaluator
        if (!evaluator_) {
            // WARNING: evaluator_ is null after move assignment
        }
        
        // Properly clean up other's threads before clearing
        other.shutdown_ = true;
        other.workers_active_ = false;
        other.cv_.notify_all();
        other.batch_cv_.notify_all();
        other.result_cv_.notify_all();
        
        // Join other's threads before clearing
        for (auto& thread : other.tree_traversal_workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        if (other.batch_accumulator_worker_.joinable()) {
            other.batch_accumulator_worker_.join();
        }
        if (other.result_distributor_worker_.joinable()) {
            other.result_distributor_worker_.join();
        }
        
        // Now safe to clear
        other.tree_traversal_workers_.clear();
        other.search_running_ = false;
        other.active_simulations_ = 0;
        other.evaluator_started_ = false;
    }
    
    return *this;
}

MCTSEngine::~MCTSEngine() {
    
    
    // Phase 1: Signal shutdown to all components atomically
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    active_simulations_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Mark mutexes as destroyed to prevent threads from acquiring them
    cv_mutex_destroyed_.store(true, std::memory_order_release);
    batch_mutex_destroyed_.store(true, std::memory_order_release);
    result_mutex_destroyed_.store(true, std::memory_order_release);
    
    // Phase 2: Stop the evaluator first (it's the source of new work)
    
    safelyStopEvaluator();
    
    // Phase 3: Force wake all threads immediately to check shutdown flag
    
    cv_.notify_all();
    batch_cv_.notify_all();
    result_cv_.notify_all();
    
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
    
    // Join tree traversal workers
    
    for (size_t i = 0; i < tree_traversal_workers_.size(); ++i) {
        if (tree_traversal_workers_[i].joinable()) {
            
            tree_traversal_workers_[i].join();
            
        }
    }
    
    
    // Phase 6: Final cleanup - clear transposition table and root
    
    if (transposition_table_) {
        transposition_table_->clear();
    }
    root_.reset();
    
    
}

SearchResult MCTSEngine::search(const core::IGameState& state) {
    // MCTSEngine::search - Starting search...
    alphazero::utils::trackMemory("MCTSEngine::search started");
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
    
    // Ensure evaluator is started (idempotent)
    if (!evaluator_started_) {
        if (!ensureEvaluatorStarted()) {
            // MCTSEngine::search - Evaluator could not be started. Aborting search
            SearchResult result; // Return default/error result
            result.action = -1;
            return result;
        }
    }

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
    
    alphazero::utils::trackMemory("MCTSEngine::search completed");

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
                 
        // First, set active_simulations to 0 to prevent new work from being taken
        active_simulations_.store(0, std::memory_order_release);
        cv_.notify_all();
        
        // Use multiple short waits instead of one long wait for better responsiveness
        std::unique_lock<std::mutex> lock(cv_mutex_);
        bool workers_finished = false;
        for (int attempts = 0; attempts < 10 && !workers_finished; ++attempts) {
            workers_finished = cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                // Simplified predicate: wait until workers are no longer active or shutdown is signaled.
                // The workers_active_ flag should be set to false by the end of the previous search.
                // And it's set to true at the beginning of a new search *after* this wait.
                return !workers_active_.load(std::memory_order_acquire) || 
                       shutdown_.load(std::memory_order_acquire);
            });
            
            if (!workers_finished && attempts % 3 == 2) {
                // Periodically re-signal workers
                cv_.notify_all();
            }
        }
        
        if (!workers_finished && !shutdown_.load(std::memory_order_acquire)) {
            // MCTSEngine::runSearch - WARNING: Workers still active after timeout
            // Force reset for safety
            // num_workers_actively_processing_.store(0, std::memory_order_release); // Removed
        }
        
        // MCTSEngine::runSearch - Worker threads are now inactive, can proceed
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
        std::unique_ptr<core::IGameState> state_clone;
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
            root_ = MCTSNode::create(std::move(state_clone));
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
        active_simulations_ = 0;

        // Configure evaluator to use external queues first
        if (evaluator_) {
            // [ENGINE] Setting external queues on evaluator
            
            // Provide a callback to notify when results are available
            auto result_notify_fn = [this]() {
                result_cv_.notify_one();
            };
            
            evaluator_->setExternalQueues(&leaf_queue_, &result_queue_, result_notify_fn);
        }
        
        // Create specialized worker threads if they don't exist yet
        if (tree_traversal_workers_.empty() && settings_.num_threads > 0) {
            try {
                // Start specialized workers
                workers_active_.store(true, std::memory_order_release);
                shutdown_.store(false, std::memory_order_release);
                
                // Start only the result distributor worker (evaluator handles batching now)
                result_distributor_worker_ = std::thread(&MCTSEngine::resultDistributorWorker, this);
                
                // Create tree traversal workers
                try {
                    for (int i = 0; i < settings_.num_threads; ++i) {
                        tree_traversal_workers_.emplace_back(&MCTSEngine::treeTraversalWorker, this, i);
                    }
                } catch (...) {
                    // Clean up all created threads on failure
                    workers_active_ = false;
                    cv_.notify_all();
                    
                    // Join already created traversal threads
                    for (auto& thread : tree_traversal_workers_) {
                        if (thread.joinable()) {
                            thread.join();
                        }
                    }
                    
                    // Join the result worker
                    if (result_distributor_worker_.joinable()) {
                        result_distributor_worker_.join();
                    }
                    
                    throw;
                }
                
                // [ENGINE] Created tree traversal workers
            } catch (const std::exception& e) {
                // Error creating worker threads
                throw std::runtime_error(std::string("Failed to create worker threads: ") + e.what());
            } catch (...) {
                // Unknown error creating worker threads
                throw std::runtime_error("Unknown error creating worker threads");
            }
        } else if (!tree_traversal_workers_.empty()) {
            // Reactivate existing workers
            workers_active_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // Calculate the number of simulations to run
        int num_simulations = settings_.num_simulations;
        if (num_simulations <= 0) {
            num_simulations = 800; // Default value
        }

        // Set all simulations at once for better batching
        active_simulations_.store(num_simulations, std::memory_order_release);
        cv_.notify_all(); // Wake up all workers

        // Start completion tracking for search
        std::atomic<bool> search_complete(false);
        
        // Wait for the search to complete using a more robust mechanism
        auto search_thread = std::thread([this, num_simulations, &search_complete]() {
            auto start_time = std::chrono::steady_clock::now();
            const auto max_search_time = std::chrono::seconds(10); // Fail-safe timeout
            
            while (!shutdown_.load(std::memory_order_acquire)) {
                int current_sims = active_simulations_.load(std::memory_order_acquire);
                int pending_evals = pending_evaluations_.load(std::memory_order_acquire);
                
                // Debug output every second
                static auto last_debug_time = std::chrono::steady_clock::now();
                if (std::chrono::steady_clock::now() - last_debug_time > std::chrono::seconds(1)) {
                    // [SEARCH] Status tracking
                    int pending = pending_evaluations_.load(std::memory_order_acquire);
                    int active = active_simulations_.load(std::memory_order_acquire);
                    int leaf_size = leaf_queue_.size_approx();
                    int result_size = result_queue_.size_approx();
                    
                    std::cout << "[SEARCH] Status: active_simulations=" << active
                              << ", pending_evaluations=" << pending
                              << ", leaf_queue=" << leaf_size
                              << ", result_queue=" << result_size
                              << std::endl;
                    
                    // Track memory periodically
                    static int search_memory_check = 0;
                    if (search_memory_check++ % 5 == 0) {
                        alphazero::utils::trackMemory("During search iteration");
                    }
                    
                    last_debug_time = std::chrono::steady_clock::now();
                }
                
                // Process any completed evaluations
                processPendingEvaluations(root_);
                
                // Check if search is complete
                if (current_sims <= 0 && pending_evals <= 0) {
                    // [SEARCH] Search appears complete
                    search_complete.store(true);
                    break;
                }
                
                // Fail-safe timeout
                if (std::chrono::steady_clock::now() - start_time > max_search_time) {
                    // [SEARCH] ERROR: Search timed out
                    search_complete.store(true);
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        
        // Wait for search thread to complete
        search_thread.join();
        
        // Log final status
        // [SEARCH] Final status after search completion
        
        // Signal workers to stop
        workers_active_.store(false, std::memory_order_release);
        cv_.notify_all();
        batch_cv_.notify_all();
        
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

void MCTSEngine::treeTraversalWorker(int worker_id) {
    
    
    // Set thread name for debugging
    std::string thread_name = "TreeWorker" + std::to_string(worker_id);
    pthread_setname_np(pthread_self(), thread_name.c_str());
    
    try {
        while (!shutdown_.load(std::memory_order_acquire)) {
            // Check if there's work to do
            int remaining_sims = active_simulations_.load(std::memory_order_acquire);
            if (remaining_sims <= 0 || !root_ || !workers_active_.load(std::memory_order_acquire)) {
                // Check shutdown before waiting
                if (shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                
                // Use condition variable with safer pattern
                // Use condition variable with safer pattern - wait without timeout
                if (!cv_mutex_destroyed_) {
                    try {
                        std::unique_lock<std::mutex> lock(cv_mutex_);
                        cv_.wait(lock, [this]() {
                            return shutdown_.load(std::memory_order_acquire) || 
                                   cv_mutex_destroyed_.load(std::memory_order_acquire) ||
                                   (active_simulations_.load(std::memory_order_acquire) > 0 && 
                                    root_ != nullptr && 
                                    workers_active_.load(std::memory_order_acquire));
                        });
                    } catch (...) {
                        // Ignore mutex/cv exceptions during shutdown
                        if (shutdown_.load(std::memory_order_acquire)) {
                            break;
                        }
                    }
                } else {
                    // Mutex destroyed, just check shutdown
                    if (shutdown_.load(std::memory_order_acquire)) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                if (shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                continue;
            }
            
            // Claim a batch of simulations
            int batch_size = std::min(64, std::max(16, remaining_sims / settings_.num_threads));
            int claimed = 0;
            
            while (claimed < batch_size && !shutdown_.load(std::memory_order_acquire)) {
                int old_value = active_simulations_.load(std::memory_order_acquire);
                if (old_value <= 0) break;
                
                int to_claim = std::min(batch_size - claimed, old_value);
                if (active_simulations_.compare_exchange_weak(old_value, old_value - to_claim,
                                                              std::memory_order_acq_rel, 
                                                              std::memory_order_acquire)) {
                    claimed += to_claim;
                }
            }
            
            // Process claimed simulations
            for (int i = 0; i < claimed && !shutdown_.load(std::memory_order_acquire); i++) {
                try {
                    traverseTree(root_);
                } catch (const std::exception& e) {
                    MCTS_LOG_ERROR("[WORKER " << worker_id << "] Exception during tree traversal: " << e.what());
                } catch (...) {
                    MCTS_LOG_ERROR("[WORKER " << worker_id << "] Unknown exception during tree traversal");
                }
                
                // Check shutdown more frequently
                if (i % 8 == 0) {
                    if (shutdown_.load(std::memory_order_acquire)) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        }
    } catch (const std::exception& e) {
        MCTS_LOG_ERROR("[WORKER " << worker_id << "] Fatal exception: " << e.what());
    } catch (...) {
        MCTS_LOG_ERROR("[WORKER " << worker_id << "] Fatal unknown exception");
    }
    
    
}

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
                    pending.state = std::move(state_clone);
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
    
    pthread_setname_np(pthread_self(), "ResultDist");
    
    try {
        std::vector<std::pair<NetworkOutput, PendingEvaluation>> result_batch;
        result_batch.reserve(32);  // Process results in batches
        
        while (!shutdown_.load(std::memory_order_acquire) || 
               result_queue_.size_approx() > 0) {
            
            // Check for shutdown more frequently
            if (shutdown_.load(std::memory_order_acquire) && result_queue_.size_approx() == 0) {
                break;
            }
            
            // Try to dequeue multiple results at once
        result_batch.clear();
        while (result_batch.size() < 32) {
            std::pair<NetworkOutput, PendingEvaluation> result_pair;
            if (result_queue_.try_dequeue(result_pair)) {
                result_batch.push_back(std::move(result_pair));
            } else {
                break;
            }
        }
        
        if (!result_batch.empty()) {
            // Report batch processing
            std::cout << "[RESULT DIST] Processing " << result_batch.size() 
                      << " results, queue_size=" << result_queue_.size_approx() << std::endl;
            
            // Process all results in the batch
            for (auto& [output, eval] : result_batch) {
                // Check if we should stop processing
                if (shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                
                // Update the node with neural network results
                if (eval.node) {
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
                // Wait for results with condition variable (use wait_for for responsiveness)
                if (!result_mutex_destroyed_.load(std::memory_order_acquire)) {
                    try {
                        std::unique_lock<std::mutex> lock(result_mutex_);
                        result_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                            return shutdown_.load(std::memory_order_acquire) || 
                                   result_mutex_destroyed_.load(std::memory_order_acquire) ||
                                   result_queue_.size_approx() > 0;
                        });
                    } catch (...) {
                        // Ignore exceptions during shutdown
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

        std::shared_ptr<MCTSNode> selected_child = parent_for_selection->selectChild(settings_.exploration_constant);
        
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
            states.push_back(std::move(state_clone));
            
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
                pending.state = std::move(state_clone);
                pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                
                // Submit to leaf queue for batching
                if (leaf_queue_.enqueue(std::move(pending))) {
                    // Increment pending evaluations
                    pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                    
                    // Apply virtual loss to prevent other threads from selecting this path
                    leaf->applyVirtualLoss(settings_.virtual_loss);
                    
                    // Notify evaluator
                    if (evaluator_) {
                        evaluator_->notifyLeafAvailable();
                    }
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
    
    // Process nodes in reverse order (from leaf to root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto node = *it;
        float update_value = invert ? -value : value;
        
        // Remove virtual loss and update node statistics
        node->removeVirtualLoss();
        node->update(update_value);
        
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
            cv_.notify_all();
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
                states.push_back(std::move(state_clone));
                auto outputs = evaluator_->getInferenceFunction()(states);
                if (!outputs.empty()) {
                    root->setPriorProbabilities(outputs[0].policy);
                } else {
                    int action_space_size = root->getState().getActionSpaceSize();
                    std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                    root->setPriorProbabilities(uniform_policy);
                }
            } else {
                auto future = evaluator_->evaluateState(root, std::move(state_clone));
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

std::unique_ptr<core::IGameState> MCTSEngine::cloneGameState(const core::IGameState& state) {
    // Use pool-based cloning if enabled
    if (game_state_pool_enabled_) {
        return utils::GameStatePoolManager::getInstance().cloneState(state);
    }
    
    // Fallback to regular cloning
    return state.clone();
}

} // namespace mcts
} // namespace alphazero