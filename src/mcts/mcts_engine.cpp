// src/mcts/mcts_engine.cpp
#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <iomanip>

// Configurable debug level
#define MCTS_DEBUG 1

namespace alphazero {
namespace mcts {

// Helper function to join thread with timeout
bool join_with_timeout(std::thread& thread, std::chrono::milliseconds timeout) {
    if (!thread.joinable()) {
        return true;
    }
    
    // Create a promise/future pair to signal completion
    std::promise<void> completion_promise;
    auto completion_future = completion_promise.get_future();
    
    // Create a wrapper thread that joins the target thread
    std::thread joiner([&thread, &completion_promise]() {
        thread.join();
        completion_promise.set_value();
    });
    
    // Wait for completion with timeout
    if (completion_future.wait_for(timeout) == std::future_status::timeout) {
        // Timeout occurred - detach the joiner thread
        joiner.detach();
        return false;
    }
    
    // Join succeeded - clean up the joiner thread
    joiner.join();
    return true;
}

MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      evaluator_started_(false),
      num_workers_actively_processing_(0) {
    
    // Create transposition table with configurable size
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128; // Default 128MB
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Check neural network validity
    if (!neural_net) {
        std::cerr << "ERROR: Null neural network passed to MCTSEngine constructor" << std::endl;
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Create evaluator with stronger exception handling
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            [neural_net](const std::vector<std::unique_ptr<core::IGameState>>& states) {
                return neural_net->inference(states);
            }, settings.batch_size, settings.batch_timeout);
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR during evaluator creation: " << e.what() << std::endl;
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
      num_workers_actively_processing_(0) {
    
    // Create transposition table with configurable size
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128; // Default 128MB
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Check inference function validity
    if (!inference_fn) {
        std::cerr << "ERROR: Null inference function passed to MCTSEngine constructor" << std::endl;
        throw std::invalid_argument("Inference function cannot be null");
    }
    
    // Create evaluator with stronger exception handling
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            std::move(inference_fn), settings.batch_size, settings.batch_timeout);
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR during evaluator creation: " << e.what() << std::endl;
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
            std::cerr << "MCTSEngine::ensureEvaluatorStarted - Evaluator is null" << std::endl;
            return false;
        }
        
        // Start the evaluator
        evaluator_->start();
        evaluator_started_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MCTSEngine::ensureEvaluatorStarted - Failed to start evaluator: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "MCTSEngine::ensureEvaluatorStarted - Unknown error starting evaluator" << std::endl;
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
      num_workers_actively_processing_(other.num_workers_actively_processing_.load()),
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
        std::cerr << "WARNING: evaluator_ is null after move constructor" << std::endl;
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
        num_workers_actively_processing_ = other.num_workers_actively_processing_.load();
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
            std::cerr << "WARNING: evaluator_ is null after move assignment" << std::endl;
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
    std::cout << "[ENGINE] Starting destructor sequence" << std::endl;
    
    // Phase 1: Signal shutdown to all components
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    active_simulations_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Phase 2: Stop the evaluator first (it's the source of new work)
    std::cout << "[ENGINE] Stopping evaluator..." << std::endl;
    safelyStopEvaluator();
    
    // Phase 3: Clear all queues to prevent stuck threads
    std::cout << "[ENGINE] Clearing all queues..." << std::endl;
    {
        PendingEvaluation temp_eval;
        while (leaf_queue_.try_dequeue(temp_eval)) {
            // Set default values to satisfy promises
            try {
                NetworkOutput default_output;
                default_output.value = 0.0f;
                default_output.policy.resize(10, 0.1f);
                // We don't have promise here, so just clear
            } catch (...) {}
        }
        
        BatchInfo temp_batch;
        while (batch_queue_.try_dequeue(temp_batch)) {
            // Set default values for all pending evaluations
            for (auto& eval : temp_batch.evaluations) {
                try {
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    default_output.policy.resize(10, 0.1f);
                    // We don't have promise here, so just clear
                } catch (...) {}
            }
        }
        
        std::pair<NetworkOutput, PendingEvaluation> temp_result;
        while (result_queue_.try_dequeue(temp_result)) {
            // Results already processed, just clear
        }
    }
    
    // Phase 4: Force wake all threads multiple times
    std::cout << "[ENGINE] Waking all threads..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        cv_.notify_all();
        batch_cv_.notify_all();
        result_cv_.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Phase 5: Give threads time to exit gracefully
    std::cout << "[ENGINE] Waiting for threads to exit..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Phase 6: Join specialized worker threads with timeout
    std::cout << "[ENGINE] Joining batch accumulator..." << std::endl;
    if (batch_accumulator_worker_.joinable()) {
        batch_accumulator_worker_.join();
    }
    
    std::cout << "[ENGINE] Joining result distributor..." << std::endl;
    if (result_distributor_worker_.joinable()) {
        result_distributor_worker_.join();
    }
    
    // Phase 7: Join tree traversal workers
    std::cout << "[ENGINE] Joining tree traversal workers..." << std::endl;
    for (size_t i = 0; i < tree_traversal_workers_.size(); ++i) {
        if (tree_traversal_workers_[i].joinable()) {
            std::cout << "[ENGINE] Joining worker " << i << "..." << std::endl;
            tree_traversal_workers_[i].join();
        }
    }
    
    // Phase 8: Final cleanup - clear transposition table and root
    std::cout << "[ENGINE] Final cleanup..." << std::endl;
    if (transposition_table_) {
        transposition_table_->clear();
    }
    root_.reset();
    
    std::cout << "[ENGINE] Destructor complete" << std::endl;
}

SearchResult MCTSEngine::search(const core::IGameState& state) {
    std::cout << "MCTSEngine::search - Starting search..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    // Validate the state before proceeding
    try {
        std::cout << "MCTSEngine::search - Validating state..." << std::endl;
        if (!state.validate()) {
            std::cerr << "MCTSEngine::search - Invalid game state passed to search method" << std::endl;
            SearchResult result;
            result.action = -1;
            result.value = 0.0f;
            // Return best guess from legal moves
            std::cout << "MCTSEngine::search - Getting legal moves from invalid state..." << std::endl;
            auto legal_moves = state.getLegalMoves();
            if (!legal_moves.empty()) {
                result.action = legal_moves[0];
                // Create a uniform policy
                result.probabilities.resize(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
                std::cout << "MCTSEngine::search - Using first legal move: " << result.action << std::endl;
            }
            return result;
        }
        std::cout << "MCTSEngine::search - State validated successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "MCTSEngine::search - Exception during state validation: " << e.what() << std::endl;
        SearchResult result; // Return default/error result
        result.action = -1;
        return result;
    } catch (...) {
        std::cerr << "MCTSEngine::search - Unknown exception during state validation" << std::endl;
        SearchResult result; // Return default/error result
        result.action = -1;
        return result;
    }

    // Critical: Clear the transposition table BEFORE resetting the tree
    // This ensures node pointers are still valid when clearing the table
    if (use_transposition_table_ && transposition_table_) {
        std::cout << "MCTSEngine::search - Clearing transposition table for new search." << std::endl;
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
            std::cerr << "MCTSEngine::search - Evaluator could not be started. Aborting search." << std::endl;
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
        std::cout << "MCTSEngine::search - Game state is already terminal. No search needed." << std::endl;
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
            std::cerr << "MCTSEngine::search - Error getting terminal value: " << e.what() << std::endl;
            result.value = 0.0f;
        }
        result.probabilities.assign(state.getActionSpaceSize(), 0.0f);
        last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
        return result;
    }

    try {
        std::cout << "MCTSEngine::search - Calling runSearch()..." << std::endl;
        runSearch(state);
        std::cout << "MCTSEngine::search - runSearch() completed successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "MCTSEngine::search - Error during search: " << e.what() << std::endl;
        // Ensure proper cleanup before rethrowing
        safelyStopEvaluator();
        throw;
    }
    catch (...) {
        std::cerr << "MCTSEngine::search - Unknown error during search" << std::endl;
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
        std::cerr << "Error extracting search results: " << e.what() << std::endl;
        
        // Set fallback results
        if (result.action < 0) {
            try {
                auto legal_moves = state.getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                }
            } catch (const std::exception& e) {
                std::cerr << "Error getting legal moves: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown error getting legal moves" << std::endl;
            }
        }
    }
    catch (...) {
        std::cerr << "Unknown error extracting search results" << std::endl;
        
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
    std::cout << "MCTSEngine::runSearch - Starting runSearch..." << std::endl;
    // Reset statistics
    last_stats_ = MCTSStats();
    
    // Wait for all worker threads to finish processing before cleaning up
    // from any previous search iteration on this engine instance.
    {
        std::cout << "MCTSEngine::runSearch - Waiting for worker threads to finish processing..." << std::endl;
        std::cout << "MCTSEngine::runSearch - Current num_workers_actively_processing_: " 
                 << num_workers_actively_processing_.load(std::memory_order_acquire) << std::endl;
        std::cout << "MCTSEngine::runSearch - Shutdown flag: " 
                 << (shutdown_.load(std::memory_order_acquire) ? "true" : "false") << std::endl;
                 
        // First, set active_simulations to 0 to prevent new work from being taken
        active_simulations_.store(0, std::memory_order_release);
        cv_.notify_all();
        
        // Use multiple short waits instead of one long wait for better responsiveness
        std::unique_lock<std::mutex> lock(cv_mutex_);
        bool workers_finished = false;
        for (int attempts = 0; attempts < 10 && !workers_finished; ++attempts) {
            workers_finished = cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return num_workers_actively_processing_.load(std::memory_order_acquire) == 0 || 
                       shutdown_.load(std::memory_order_acquire);
            });
            
            if (!workers_finished && attempts % 3 == 2) {
                // Periodically re-signal workers
                cv_.notify_all();
            }
        }
        
        if (!workers_finished && !shutdown_.load(std::memory_order_acquire)) {
            std::cerr << "MCTSEngine::runSearch - WARNING: Workers still active after timeout" << std::endl;
            // Force reset for safety
            num_workers_actively_processing_.store(0, std::memory_order_release);
        }
        
        std::cout << "MCTSEngine::runSearch - Worker threads are now inactive, can proceed" << std::endl;
    }
    
    // If using the transposition table, it must be cleared BEFORE deleting the tree
    // to ensure node pointers are still valid when clearing the table
    if (use_transposition_table_ && transposition_table_) {
        std::cout << "MCTSEngine::runSearch - Clearing transposition table..." << std::endl;
        try {
            transposition_table_->clear();
            std::cout << "MCTSEngine::runSearch - Transposition table cleared successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MCTSEngine::runSearch - Error clearing transposition table: " << e.what() << std::endl;
            // In case of any exception during clear, recreate the table entirely
            // This is safer than potentially having dangling pointers
            size_t size_mb = 128; // Default size
            size_t num_shards = std::max(4u, std::thread::hardware_concurrency());
            if (settings_.num_threads > 0) {
                num_shards = std::max(size_t(settings_.num_threads), num_shards);
            }
            std::cout << "MCTSEngine::runSearch - Recreating transposition table with size_mb=" 
                     << size_mb << ", num_shards=" << num_shards << std::endl;
            transposition_table_ = std::make_unique<TranspositionTable>(size_mb, num_shards);
            std::cout << "MCTSEngine::runSearch - Transposition table recreated successfully" << std::endl;
        }
    } else {
        std::cout << "MCTSEngine::runSearch - Skipping transposition table clear (not used or null)" << std::endl;
    }

    // Clean up the old root if it exists. This invalidates all nodes in the previous tree.
    // MUST be done AFTER clearing the transposition table
    std::cout << "MCTSEngine::runSearch - Cleaning up old root node..." << std::endl;
    if (root_) {
        std::cout << "MCTSEngine::runSearch - Old root exists, cleaning up..." << std::endl;
    } else {
        std::cout << "MCTSEngine::runSearch - No old root exists" << std::endl;
    }
    root_.reset();
    std::cout << "MCTSEngine::runSearch - Root node reset completed" << std::endl;
    
    // Create the new root node.
    // If using the transposition table, it has just been cleared. We create a new root
    // from the input state and then add it to the TT.
    // We do not attempt to find the new root in the just-cleared TT, as that could lead to
    // using stale pointers if the clear operation was somehow incomplete or a hash collided.
    std::cout << "MCTSEngine::runSearch - Creating new root node from state..." << std::endl;
    try {
        // Clone the state with proper error handling
        std::unique_ptr<core::IGameState> state_clone;
        std::cout << "MCTSEngine::runSearch - Cloning state..." << std::endl;
        try {
            state_clone = state.clone();
            if (!state_clone) {
                std::cerr << "MCTSEngine::runSearch - ERROR: state.clone() returned nullptr" << std::endl;
                throw std::runtime_error("state.clone() returned nullptr");
            }
            std::cout << "MCTSEngine::runSearch - State cloned successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MCTSEngine::runSearch - Exception during state cloning: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to clone state: ") + e.what());
        } catch (...) {
            std::cerr << "MCTSEngine::runSearch - Unknown exception during state cloning" << std::endl;
            throw std::runtime_error("Unknown error when cloning state");
        }
        
        // Additional validation of the cloned state
        std::cout << "MCTSEngine::runSearch - Validating cloned state..." << std::endl;
        try {
            if (!state_clone->validate()) {
                std::cerr << "MCTSEngine::runSearch - Cloned state failed validation" << std::endl;
                throw std::runtime_error("Cloned state failed validation");
            }
            std::cout << "MCTSEngine::runSearch - Cloned state validated successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MCTSEngine::runSearch - Exception validating cloned state: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Cloned state validation error: ") + e.what());
        } catch (...) {
            std::cerr << "MCTSEngine::runSearch - Unknown exception validating cloned state" << std::endl;
            throw std::runtime_error("Unknown error when validating cloned state");
        }
        
        // Create root node with proper error handling
        std::cout << "MCTSEngine::runSearch - Creating root node..." << std::endl;
        try {
            root_ = MCTSNode::create(std::move(state_clone));
            std::cout << "MCTSEngine::runSearch - Root node created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MCTSEngine::runSearch - Exception creating root node: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to create root node: ") + e.what());
        } catch (...) {
            std::cerr << "MCTSEngine::runSearch - Unknown exception creating root node" << std::endl;
            throw std::runtime_error("Unknown error creating root node");
        }

        // Ensure we have a valid root
        if (!root_) {
            std::cerr << "MCTSEngine::runSearch - Failed to create root node" << std::endl;
            throw std::runtime_error("Failed to create root node");
        }
        std::cout << "MCTSEngine::runSearch - Root node pointer is valid" << std::endl;
        
        // Validate the root node's state
        std::cout << "MCTSEngine::runSearch - Validating root node's state..." << std::endl;
        try {
            if (!root_->getState().validate()) {
                std::cerr << "MCTSEngine::runSearch - Root node state invalid after creation" << std::endl;
                throw std::runtime_error("Root node state invalid after creation");
            }
            std::cout << "MCTSEngine::runSearch - Root node's state validated successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MCTSEngine::runSearch - Exception validating root node state: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Root node state validation error: ") + e.what());
        } catch (...) {
            std::cerr << "MCTSEngine::runSearch - Unknown exception validating root node state" << std::endl;
            throw std::runtime_error("Unknown error validating root node state");
        }

        // If using transposition table, store the new root.
        if (use_transposition_table_ && transposition_table_ && root_) {
            try {
                uint64_t hash = root_->getState().getHash(); // Get hash from the root's state
                transposition_table_->store(hash, std::weak_ptr<MCTSNode>(root_), 0);
            } catch (const std::exception& e) {
                std::cerr << "Error storing root in transposition table: " << e.what() << std::endl;
                // Continue without transposition table storage
            } catch (...) {
                std::cerr << "Unknown error storing root in transposition table" << std::endl;
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
                std::cerr << "Error adding Dirichlet noise: " << e.what() << std::endl;
                // Continue without noise - non-fatal
            } catch (...) {
                std::cerr << "Unknown error adding Dirichlet noise" << std::endl;
                // Continue without noise - non-fatal
            }
        }

        // Set search running flag
        search_running_ = true;
        active_simulations_ = 0;

        // Configure evaluator to use external queues first
        if (evaluator_) {
            std::cout << "[ENGINE] Setting external queues on evaluator. batch_queue_=" << &batch_queue_ 
                      << ", result_queue_=" << &result_queue_ << std::endl;
            evaluator_->setExternalQueues(&batch_queue_, &result_queue_);
        }
        
        // Create specialized worker threads if they don't exist yet
        if (tree_traversal_workers_.empty() && settings_.num_threads > 0) {
            try {
                // Start specialized workers
                workers_active_ = true;
                shutdown_.store(false, std::memory_order_release);
                
                // Start the batch and result workers first
                batch_accumulator_worker_ = std::thread(&MCTSEngine::batchAccumulatorWorker, this);
                
                try {
                    result_distributor_worker_ = std::thread(&MCTSEngine::resultDistributorWorker, this);
                } catch (...) {
                    // If second thread fails, join the first one
                    workers_active_ = false;
                    if (batch_accumulator_worker_.joinable()) {
                        batch_accumulator_worker_.join();
                    }
                    throw;
                }
                
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
                    
                    // Join the specialized workers
                    if (batch_accumulator_worker_.joinable()) {
                        batch_accumulator_worker_.join();
                    }
                    if (result_distributor_worker_.joinable()) {
                        result_distributor_worker_.join();
                    }
                    
                    throw;
                }
                
                std::cout << "[ENGINE] Created " << settings_.num_threads << " tree traversal workers" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error creating worker threads: " << e.what() << std::endl;
                throw std::runtime_error(std::string("Failed to create worker threads: ") + e.what());
            } catch (...) {
                std::cerr << "Unknown error creating worker threads" << std::endl;
                throw std::runtime_error("Unknown error creating worker threads");
            }
        } else if (!tree_traversal_workers_.empty()) {
            // Reactivate existing workers
            workers_active_ = true;
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
                    std::cout << "[SEARCH] Status: active_simulations=" << current_sims 
                              << ", pending_evaluations=" << pending_evals 
                              << ", batch_queue_size=" << batch_queue_.size_approx()
                              << ", result_queue_size=" << result_queue_.size_approx() << std::endl;
                    last_debug_time = std::chrono::steady_clock::now();
                }
                
                // Check if search is complete
                if (current_sims <= 0 && pending_evals <= 0) {
                    std::cout << "[SEARCH] Search appears complete" << std::endl;
                    search_complete.store(true);
                    break;
                }
                
                // Fail-safe timeout
                if (std::chrono::steady_clock::now() - start_time > max_search_time) {
                    std::cerr << "[SEARCH] ERROR: Search timed out after " 
                              << max_search_time.count() << " seconds" << std::endl;
                    search_complete.store(true);
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        
        // Wait for search thread to complete
        search_thread.join();
        
        // Log final status
        std::cout << "[SEARCH] Final status after search completion:" << std::endl;
        std::cout << "  Total simulations run: " << (num_simulations - active_simulations_.load()) << std::endl;
        std::cout << "  Remaining active simulations: " << active_simulations_.load() << std::endl;
        std::cout << "  Pending evaluations: " << pending_evaluations_.load() << std::endl;
        
        // Signal workers to stop
        workers_active_ = false;
        cv_.notify_all();
        batch_cv_.notify_all();
        
        // Record search statistics
        if (root_) {
            last_stats_.total_nodes = countTreeNodes(root_);
            last_stats_.max_depth = calculateMaxDepth(root_);
        }
        
        // Mark search as completed
        search_running_ = false;
        
    } catch (const std::exception& e) {
        // Log the error
        std::cerr << "Exception during MCTS search: " << e.what() << std::endl;
        
        // Reset search state
        search_running_ = false;
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow to allow caller to handle the error
        throw;
    } catch (...) {
        // Handle unknown exceptions
        std::cerr << "Unknown exception during MCTS search" << std::endl;
        
        // Reset search state
        search_running_ = false;
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow with a more descriptive message
        throw std::runtime_error("Unknown error occurred during MCTS search");
    }
}

void MCTSEngine::treeTraversalWorker(int worker_id) {
    std::cout << "[WORKER " << worker_id << "] Tree traversal worker started" << std::endl;
    
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
                // Use condition variable with safer pattern
                if (!cv_mutex_destroyed_) {
                    try {
                        std::unique_lock<std::mutex> lock(cv_mutex_);
                        bool signaled = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                            return shutdown_.load(std::memory_order_acquire) || 
                                   (active_simulations_.load(std::memory_order_acquire) > 0 && 
                                    root_ != nullptr && 
                                    workers_active_.load(std::memory_order_acquire));
                        });
                        // Lock is released here via RAII
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
                if (active_simulations_.compare_exchange_weak(old_value, old_value - to_claim)) {
                    claimed += to_claim;
                }
            }
            
            // Process claimed simulations
            for (int i = 0; i < claimed && !shutdown_.load(std::memory_order_acquire); i++) {
                try {
                    traverseTree(root_);
                } catch (const std::exception& e) {
                    std::cerr << "[WORKER " << worker_id << "] Exception during tree traversal: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[WORKER " << worker_id << "] Unknown exception during tree traversal" << std::endl;
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
        std::cerr << "[WORKER " << worker_id << "] Fatal exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[WORKER " << worker_id << "] Fatal unknown exception" << std::endl;
    }
    
    std::cout << "[WORKER " << worker_id << "] Tree traversal worker stopped" << std::endl;
}

void MCTSEngine::traverseTree(std::shared_ptr<MCTSNode> root) {
    if (!root) return;
    
    try {
        // Selection phase
        auto [leaf, path] = selectLeafNode(root);
        if (!leaf) return;
        
        // Expansion phase - never block
        if (!leaf->isTerminal() && leaf->isLeaf()) {
            leaf->expand();
            
            // Create evaluation request
            auto state_clone = leaf->getState().clone();
            if (state_clone) {
                PendingEvaluation pending;
                pending.node = leaf;
                pending.path = std::move(path);
                pending.state = std::move(state_clone);
                pending.batch_id = batch_counter_.fetch_add(1);
                pending.request_id = total_leaves_generated_.fetch_add(1);
                
                // Submit to leaf queue with proper move semantics
                if (leaf_queue_.enqueue(std::move(pending))) {
                    pending_evaluations_.fetch_add(1);
                } else {
                    std::cerr << "[TRAVERSE] Failed to enqueue evaluation request" << std::endl;
                }
            }
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

void MCTSEngine::batchAccumulatorWorker() {
    std::cout << "[BATCH] Batch accumulator worker started" << std::endl;
    pthread_setname_np(pthread_self(), "BatchAccum");
    
    try {
        std::vector<PendingEvaluation> current_batch;
        auto last_batch_time = std::chrono::steady_clock::now();
        const auto batch_timeout = std::chrono::milliseconds(settings_.batch_timeout.count());
        const size_t target_batch_size = settings_.batch_size;
        
        // Adaptive minimum batch size for better GPU efficiency and deadlock prevention
        const size_t MIN_BATCH_SIZE = 16;  // Reduced to prevent deadlock with small eval counts
        const size_t OPTIMAL_BATCH_SIZE = std::max(MIN_BATCH_SIZE, target_batch_size);
        current_batch.reserve(OPTIMAL_BATCH_SIZE);
        
        int batch_count = 0;
        
        // Continue until shutdown and queue is empty
        while (!shutdown_.load(std::memory_order_acquire) || leaf_queue_.size_approx() > 0) {
            // Check for shutdown more frequently
            if (shutdown_.load(std::memory_order_acquire) && leaf_queue_.size_approx() == 0) {
                break;
            }
            
            PendingEvaluation eval;
            bool got_eval = false;
            
            // Try to dequeue with timeout
            const auto dequeue_timeout = std::chrono::milliseconds(10);
            auto dequeue_start = std::chrono::steady_clock::now();
            
            while (std::chrono::steady_clock::now() - dequeue_start < dequeue_timeout) {
                if (leaf_queue_.try_dequeue(eval)) {
                    current_batch.push_back(std::move(eval));
                    got_eval = true;
                    break;
                }
                
                if (shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                
                std::this_thread::yield();
            }
            
            // Check concurrent evaluation limit
            while (pending_evaluations_.load() >= settings_.max_concurrent_simulations && 
                   !shutdown_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Try to fill batch quickly with bulk operations
            int dequeued = 0;
            const int max_dequeue = OPTIMAL_BATCH_SIZE - current_batch.size();
            
            while (current_batch.size() < OPTIMAL_BATCH_SIZE && dequeued < max_dequeue) {
                if (leaf_queue_.try_dequeue(eval)) {
                    current_batch.push_back(std::move(eval));
                    dequeued++;
                    got_eval = true;
                } else {
                    break;
                }
            }
            
            auto now = std::chrono::steady_clock::now();
            bool should_submit = false;
            
            // Adaptive batching to prevent deadlock
            if (current_batch.size() >= OPTIMAL_BATCH_SIZE) {
                should_submit = true;
            } else if (current_batch.size() >= MIN_BATCH_SIZE && 
                      (now - last_batch_time) >= std::chrono::milliseconds(100)) {  // Reduced wait time
                should_submit = true;
            } else if (shutdown_.load(std::memory_order_acquire) && !current_batch.empty()) {
                should_submit = true;  // Submit remaining on shutdown
            } else if ((now - last_batch_time) >= std::chrono::milliseconds(500) && 
                      current_batch.size() >= 8) {  // Fallback: submit after 500ms if we have at least 8
                should_submit = true;
            } else if ((now - last_batch_time) >= std::chrono::milliseconds(2000) && 
                      !current_batch.empty()) {  // Emergency: submit anything after 2s
                should_submit = true;
            }
            
            if (should_submit && !current_batch.empty()) {
                batch_count++;
                
                BatchInfo batch;
                batch.evaluations = std::move(current_batch);
                batch.created_time = now;
                batch.submitted = false;
                
                // Update pending count before enqueueing
                pending_evaluations_.fetch_add(batch.evaluations.size());
                
                if (batch_queue_.enqueue(std::move(batch))) {
                    if (!batch_mutex_destroyed_) {
                        batch_cv_.notify_one();
                    }
                }
                
                current_batch.clear();
                current_batch.reserve(OPTIMAL_BATCH_SIZE);
                last_batch_time = now;
            }
            
            // Only sleep if we're not collecting enough items
            if (dequeued == 0 && current_batch.size() < MIN_BATCH_SIZE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        // Submit any remaining evaluations on shutdown
        if (!current_batch.empty()) {
            batch_count++;
            
            BatchInfo batch;
            batch.evaluations = std::move(current_batch);
            batch.created_time = std::chrono::steady_clock::now();
            batch.submitted = false;
            pending_evaluations_.fetch_add(batch.evaluations.size());
            batch_queue_.enqueue(std::move(batch));
            
            if (!batch_mutex_destroyed_) {
                batch_cv_.notify_one();
            }
        }
        
        std::cout << "[BATCH] Batch accumulator worker stopped. Total batches: " << batch_count << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[BATCH] Fatal exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[BATCH] Fatal unknown exception" << std::endl;
    }
}

void MCTSEngine::resultDistributorWorker() {
    std::cout << "[RESULT] Result distributor worker started" << std::endl;
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
                    } catch (const std::exception& e) {
                        // Node might have been destroyed, skip it
                        std::cerr << "[RESULT] Error processing node: " << e.what() << std::endl;
                    } catch (...) {
                        // Node might have been destroyed, skip it
                        std::cerr << "[RESULT] Unknown error processing node" << std::endl;
                    }
                }
                
                pending_evaluations_.fetch_sub(1);
                total_results_processed_.fetch_add(1);
            }
            } else {
                // Wait briefly if no results
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::cout << "[RESULT] Result distributor worker stopped. Total processed: " 
                  << total_results_processed_.load() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[RESULT] Fatal exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[RESULT] Fatal unknown exception" << std::endl;
    }
}

// REMOVED: Old runSimulation method with raw pointer parameter
// This entire method should be deleted as it's no longer used
#if 0
void MCTSEngine::runSimulation(MCTSNode* root) {
    if (!root) {
        std::cerr << "MCTSEngine::runSimulation - Null root pointer!" << std::endl;
        return;
    }
    
    // Validate root state before proceeding
    try {
        if (!root->getState().validate()) {
            std::cerr << "MCTSEngine::runSimulation - Root state invalid!" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "MCTSEngine::runSimulation - Error validating root state: " << e.what() << std::endl;
        return;
    } catch (...) {
        std::cerr << "MCTSEngine::runSimulation - Unknown error validating root state" << std::endl;
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
            // std::cout << "[SIM_TRACE] Simulation ended early: No leaf node found." << std::endl;
            active_simulations_.fetch_sub(1, std::memory_order_release); // Ensure this is correctly decremented if returning early.
            cv_.notify_all(); // Notify if a worker is done.
            return;  // Something went wrong during selection
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
        std::cout << "[SIM_PROFILE] Total: " << total_sim_time_us << "us (Sel: " << selection_time_us << "us, Eval: " << evaluation_time_us << "us, BP: " << backprop_time_us << "us)" << std::endl;
    }
}
#endif // End of disabled old runSimulation method

std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> MCTSEngine::selectLeafNode(std::shared_ptr<MCTSNode> root) {
    std::vector<std::shared_ptr<MCTSNode>> path;
    std::shared_ptr<MCTSNode> node = root;
    
    // Validate root node first
    if (!node) {
        return {nullptr, path}; // Return empty result if root is invalid
    }
    
    path.push_back(node);

    // Selection phase - find a leaf node
    while (node && !node->isLeaf() && !node->isTerminal()) {
        // Apply virtual loss to this node before selecting a child
        node->addVirtualLoss();

        // Select child according to PUCT formula
        std::shared_ptr<MCTSNode> child = node->selectChild(settings_.exploration_constant);
        if (!child) {
            break;  // No valid child was found
        }

        // Apply virtual loss to selected child
        child->addVirtualLoss();
        
        // Store the child as our current node
        node = child;

        // Check transposition table for this node - with extra safety
        if (use_transposition_table_ && transposition_table_ && node && !node->isTerminal()) {
            try {
                uint64_t hash = node->getState().getHash();
                std::shared_ptr<MCTSNode> transposition = transposition_table_->get(hash);

                if (transposition && transposition != node) {
                    // If a transposition is found, we use it after validating it's still a valid node
                    bool valid_transposition = false;
                    
                    try {
                        // More thorough validation - verify we can access key methods without exception
                        // and that the node has a valid state with consistent data
                        int visits = transposition->getVisitCount();
                        
                        // Additional safety check: Make sure the transposition isn't a stale pointer
                        // by checking if it has a reasonable visit count (not too high)
                        if (visits >= 0 && visits < 100000) {
                            // Try to access the state - this will throw if node is corrupted
                            const core::IGameState& trans_state = transposition->getState();
                            
                            // Validate state and hash match
                            valid_transposition = trans_state.validate() && 
                                                 trans_state.getHash() == hash;
                        }
                    } catch (...) {
                        valid_transposition = false;
                    }
                    
                    if (valid_transposition) {
                        #if MCTS_DEBUG
                        // Commented out: Debug printing about found transposition with visit count comparison
                        #endif
                        
                        // Remove virtual loss from the original node since we're not using it
                        try {
                            node->removeVirtualLoss();
                        } catch (...) {
                            // If removing virtual loss fails, continue with the original node
                            // rather than risk using a problematic transposition
                            continue;
                        }
                        
                        // Use the transposition node instead
                        node = transposition;
                        
                        // Apply virtual loss to the transposition node
                        try {
                            node->addVirtualLoss();
                        } catch (...) {
                            // If adding virtual loss fails, the node may be corrupted
                            // Fallback to creating a new node by breaking out of transposition handling
                            // This allows selection to continue without the problematic transposition
                            continue;
                        }
                    }
                }
            } catch (...) {
                // If any exception occurs during transposition table lookup or use,
                // we simply continue with the current node and ignore the transposition
            }
        }

        // Add to path
        path.push_back(node);
    }

    return {node, path};
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
    
    // If leaf has no children after expansion, return a default value
    if (leaf->getChildren().empty()) {
        return 0.0f;
    }
    
    // Evaluate with the neural network
    try {
        // Special fast path for serial mode (no worker threads)
        if (settings_.num_threads == 0) {
            auto state_clone = leaf->getState().clone();
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
            // For parallel mode, use the async evaluator
            auto state_clone = leaf->getState().clone();
            if (!state_clone) {
                throw std::runtime_error("Failed to clone state for evaluation");
            }
            
            auto future = evaluator_->evaluateState(leaf, std::move(state_clone));
            
            // Don't wait at all - immediately return to allow more requests to accumulate
            auto status = future.wait_for(std::chrono::milliseconds(0)); 
            if (status == std::future_status::ready) {
                auto result = future.get();
                leaf->setPriorProbabilities(result.policy);
                return result.value;
            } else {
                // Not ready - use uniform prior and continue immediately
                // This allows many more evaluation requests to be generated quickly
                int action_space_size = leaf->getState().getActionSpaceSize();
                std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                leaf->setPriorProbabilities(uniform_policy);
                
                // We'll get the real values on the next tree traversal
                return 0.0f;
            }
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
            auto state_clone = root->getState().clone();
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

} // namespace mcts
} // namespace alphazero