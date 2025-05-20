#include "mcts/mcts_engine.h"
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace alphazero {
namespace mcts {

// Define static members
std::mutex MCTSEngine::s_global_evaluator_mutex;
std::atomic<int> MCTSEngine::s_evaluator_init_counter{0};

// Constructor with neural network
MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      evaluator_started_(false),
      game_state_pool_enabled_(true) {
    
    // Create transposition table if enabled
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128;
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Create node tracker for pending evaluations
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Validate neural network
    if (!neural_net) {
        std::cerr << "ERROR: Null neural network passed to MCTSEngine" << std::endl;
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Create evaluator
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            [neural_net](const std::vector<std::unique_ptr<core::IGameState>>& states) {
                return neural_net->inference(states);
            }, 
            settings.batch_size, 
            settings.batch_timeout);
            
        evaluator_->setMaxCollectionBatchSize(settings.max_collection_batch_size);
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR during evaluator creation: " << e.what() << std::endl;
        throw;
    }
}

// Constructor with inference function
MCTSEngine::MCTSEngine(InferenceFunction inference_fn, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      evaluator_started_(false),
      game_state_pool_enabled_(true) {
    
    // Create transposition table if enabled
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128;
        transposition_table_ = std::make_unique<TranspositionTable>(tt_size_mb);
    }
    
    // Create node tracker for pending evaluations
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Validate inference function
    if (!inference_fn) {
        std::cerr << "ERROR: Null inference function passed to MCTSEngine" << std::endl;
        throw std::invalid_argument("Inference function cannot be null");
    }
    
    // Create evaluator
    try {
        evaluator_ = std::make_unique<MCTSEvaluator>(
            std::move(inference_fn), 
            settings.batch_size, 
            std::min(settings.batch_timeout, std::chrono::milliseconds(10)));
            
        evaluator_->setMaxCollectionBatchSize(settings.max_collection_batch_size);
            
        if (!evaluator_) {
            throw std::runtime_error("Failed to create MCTSEvaluator");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR during evaluator creation: " << e.what() << std::endl;
        throw;
    }
}

// Start the evaluator if not already running
bool MCTSEngine::ensureEvaluatorStarted() {
    // Don't start local evaluator if using shared queues
    if (use_shared_queues_) {
        return true;
    }
    
    // First check without lock for performance
    if (evaluator_started_.load(std::memory_order_acquire)) {
        return true;
    }
    
    // Acquire lock for initialization
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
        evaluator_->start();
        evaluator_started_.store(true, std::memory_order_release);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start evaluator: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error starting evaluator" << std::endl;
        return false;
    }
}

// Main search method
SearchResult MCTSEngine::search(const core::IGameState& state) {
    std::cout << "Starting new search" << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    // Validate the state
    if (!safeGameStateValidation(state)) {
        std::cout << "Invalid game state, returning default result" << std::endl;
        SearchResult result;
        result.action = -1;
        result.value = 0.0f;
        
        // Use first legal move as fallback
        auto legal_moves = state.getLegalMoves();
        if (!legal_moves.empty()) {
            result.action = legal_moves[0];
            result.probabilities.resize(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        }
        return result;
    }

    // Clear the transposition table for new search
    if (use_transposition_table_ && transposition_table_) {
        transposition_table_->clear();
        transposition_table_->resetStats();
    }

    // Reset previous search state
    root_.reset();
    
    // Initialize statistics for new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;

    // Check if state is already terminal
    if (state.isTerminal()) {
        std::cout << "Game state is already terminal. No search needed" << std::endl;
        SearchResult result;
        result.action = -1;
        
        try {
            core::GameResult game_res = state.getGameResult();
            int current_player = state.getCurrentPlayer();
            if (game_res == core::GameResult::WIN_PLAYER1) {
                result.value = (current_player == 1) ? 1.0f : -1.0f;
            } else if (game_res == core::GameResult::WIN_PLAYER2) {
                result.value = (current_player == 2) ? 1.0f : -1.0f;
            } else {
                result.value = 0.0f;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error getting terminal value: " << e.what() << std::endl;
            result.value = 0.0f;
        }
        
        result.probabilities.assign(state.getActionSpaceSize(), 0.0f);
        last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
    }

    // Ensure evaluator is started before searching
    if (!ensureEvaluatorStarted()) {
        throw std::runtime_error("Failed to start evaluator");
    }
    
    // Run the search
    try {
        runSearch(state);
    } catch (const std::exception& e) {
        safelyStopEvaluator();
        throw;
    } catch (...) {
        safelyStopEvaluator();
        throw std::runtime_error("Unknown error during search");
    }

    // Calculate search time
    auto end_time = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Prepare search result
    SearchResult result;
    result.action = -1;

    try {
        // Extract action probabilities
        result.probabilities = getActionProbabilities(root_, settings_.temperature);

        // Select action from probabilities
        if (!result.probabilities.empty()) {
            float sum = std::accumulate(result.probabilities.begin(), result.probabilities.end(), 0.0f);
            
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
    catch (...) {
        std::cerr << "Unknown error extracting search results" << std::endl;
        
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
    if (evaluator_) {
        last_stats_.avg_batch_size = evaluator_->getAverageBatchSize();
        last_stats_.avg_batch_latency = evaluator_->getAverageBatchLatency();
        last_stats_.total_evaluations = evaluator_->getTotalEvaluations();
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
    
    result.stats = last_stats_;
    return result;
}

// Safely stop the evaluator if it was started
void MCTSEngine::safelyStopEvaluator() {
    if (use_shared_queues_) {
        return;
    }
    
    if (evaluator_started_.load(std::memory_order_acquire)) {
        try {
            evaluator_->stop();
            evaluator_started_.store(false, std::memory_order_release);
        } catch (const std::exception& e) {
            std::cerr << "Error stopping evaluator: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error stopping evaluator" << std::endl;
        }
    }
}

// Configure shared external queues
void MCTSEngine::setSharedExternalQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* leaf_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* result_queue,
        std::function<void()> notify_fn) {
    shared_leaf_queue_ = leaf_queue;
    shared_result_queue_ = result_queue;
    use_shared_queues_ = true;
    external_queue_notify_fn_ = notify_fn;
    
    // Configure the evaluator to use these shared queues
    if (evaluator_) {
        evaluator_->setExternalQueues(shared_leaf_queue_, shared_result_queue_, notify_fn);
        
        // Mark as started since external management is used
        evaluator_started_.store(true, std::memory_order_release);
    }
}

// Accessor methods for transposition table
void MCTSEngine::setUseTranspositionTable(bool use) {
    use_transposition_table_ = use;
}

bool MCTSEngine::isUsingTranspositionTable() const {
    return use_transposition_table_;
}

void MCTSEngine::setTranspositionTableSize(size_t size_mb) {
    // Configure the number of shards based on thread count
    size_t num_shards = std::max(4u, std::thread::hardware_concurrency());
    if (settings_.num_threads > 0) {
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

// Settings and stats accessors
const MCTSSettings& MCTSEngine::getSettings() const {
    return settings_;
}

void MCTSEngine::updateSettings(const MCTSSettings& settings) {
    settings_ = settings;
}

const MCTSStats& MCTSEngine::getLastStats() const {
    return last_stats_;
}

// Force memory cleanup
void MCTSEngine::monitorMemoryUsage() {
    // Implement memory monitoring and cleanup logic
}

// Move constructor
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
    
    // Move resources
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
    
    // Clean up source object
    other.workers_active_ = false;
    
    if (other.result_distributor_worker_.joinable()) {
        other.result_distributor_worker_.join();
    }
    
    other.search_running_ = false;
    other.active_simulations_ = 0;
    other.evaluator_started_.store(false, std::memory_order_release);
}

// Move assignment operator
MCTSEngine& MCTSEngine::operator=(MCTSEngine&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        shutdown_ = true;
        workers_active_ = false;
        
        if (result_distributor_worker_.joinable()) {
            result_distributor_worker_.join();
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
        
        // Clean up source object
        other.shutdown_ = true;
        other.workers_active_ = false;
        
        if (other.result_distributor_worker_.joinable()) {
            other.result_distributor_worker_.join();
        }
        
        other.search_running_ = false;
        other.active_simulations_ = 0;
        other.evaluator_started_.store(false, std::memory_order_release);
    }
    
    return *this;
}

// Destructor
MCTSEngine::~MCTSEngine() {
    // Phase 1: Signal shutdown to all components
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    active_simulations_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Phase 2: Stop the evaluator
    safelyStopEvaluator();
    
    // Phase 3: Clear all queues to prevent stuck threads
    {
        PendingEvaluation temp_eval;
        while (leaf_queue_.try_dequeue(temp_eval)) {
            if (temp_eval.node) {
                try {
                    temp_eval.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
        
        BatchInfo temp_batch;
        while (batch_queue_.try_dequeue(temp_batch)) {
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
            if (temp_result.second.node) {
                try {
                    temp_result.second.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
    }
    
    // Phase 4: Join worker threads
    if (result_distributor_worker_.joinable()) {
        result_distributor_worker_.join();
    }
    
    // Phase 5: Final cleanup
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    root_.reset();
}

} // namespace mcts
} // namespace alphazero