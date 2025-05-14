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
#define MCTS_DEBUG 0

namespace alphazero {
namespace mcts {

MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings)
    : settings_(settings),
      evaluator_(std::make_unique<MCTSEvaluator>(
          [neural_net](const std::vector<std::unique_ptr<core::IGameState>>& states) {
              return neural_net->inference(states);
          }, settings.batch_size, settings.batch_timeout)),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(std::make_unique<TranspositionTable>(128)), // 128 MB default
      use_transposition_table_(true),
      evaluator_started_(false) {
    
    // Initialize, but don't start the evaluator until needed
}

MCTSEngine::MCTSEngine(InferenceFunction inference_fn, const MCTSSettings& settings)
    : settings_(settings),
      evaluator_(std::make_unique<MCTSEvaluator>(
          std::move(inference_fn), settings.batch_size, settings.batch_timeout)),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(std::make_unique<TranspositionTable>(128)), // 128 MB default
      use_transposition_table_(true),
      evaluator_started_(false) {
    
    // Initialize, but don't start the evaluator until needed
}

bool MCTSEngine::ensureEvaluatorStarted() {
    // Check if already started
    if (evaluator_started_) {
        return true;
    }
    
    try {
        // Start the evaluator
        evaluator_->start();
        evaluator_started_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start evaluator: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error starting evaluator" << std::endl;
        return false;
    }
}

void MCTSEngine::safelyStopEvaluator() {
    if (evaluator_started_) {
        try {
            evaluator_->stop();
            evaluator_started_ = false;
        } catch (const std::exception& e) {
            std::cerr << "Error stopping evaluator: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error stopping evaluator" << std::endl;
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
    transposition_table_ = std::make_unique<TranspositionTable>(size_mb);
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
      worker_threads_(std::move(other.worker_threads_)),
      shutdown_(other.shutdown_.load()),
      active_simulations_(other.active_simulations_.load()),
      search_running_(other.search_running_.load()),
      random_engine_(std::move(other.random_engine_)),
      transposition_table_(std::move(other.transposition_table_)),
      use_transposition_table_(other.use_transposition_table_),
      evaluator_started_(other.evaluator_started_) {
    
    // Clear the other's thread vector to prevent double-joining in destructor
    other.worker_threads_.clear();
    other.shutdown_ = true;
    other.search_running_ = false;
    other.active_simulations_ = 0;
    other.evaluator_started_ = false;
}

MCTSEngine& MCTSEngine::operator=(MCTSEngine&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        shutdown_ = true;
        cv_.notify_all();
        
        for (auto& thread : worker_threads_) {
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
        worker_threads_ = std::move(other.worker_threads_);
        shutdown_ = other.shutdown_.load();
        active_simulations_ = other.active_simulations_.load();
        search_running_ = other.search_running_.load();
        random_engine_ = std::move(other.random_engine_);
        transposition_table_ = std::move(other.transposition_table_);
        use_transposition_table_ = other.use_transposition_table_;
        evaluator_started_ = other.evaluator_started_;
        
        // Clear the other's thread vector to prevent double-joining in destructor
        other.worker_threads_.clear();
        other.shutdown_ = true;
        other.search_running_ = false;
        other.active_simulations_ = 0;
        other.evaluator_started_ = false;
    }
    return *this;
}

MCTSEngine::~MCTSEngine() {
    // First mark that we're shutting down
    shutdown_ = true;
    search_running_ = false;
    
    // Clear transposition table references before destroying nodes
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    // Signal shutdown to all worker threads
    cv_.notify_all();
    
    // Join all worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Stop the evaluator if it was started
    safelyStopEvaluator();
    
    // Clean up the search tree before destroying the transposition table
    // This prevents any potential use-after-free issues
    root_.reset();
    
    // Now that tree is gone, clear transposition table again
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    // Finally destroy the transposition table
    transposition_table_.reset();
}

SearchResult MCTSEngine::search(const core::IGameState& state) {
    auto start_time = std::chrono::steady_clock::now();

    // Make sure evaluator is running
    if (!ensureEvaluatorStarted()) {
        throw std::runtime_error("Failed to start neural network evaluator");
    }

    // Clear the transposition table before each search to prevent stale references
    // This is essential for thread safety between successive searches
    if (use_transposition_table_ && transposition_table_) {
        transposition_table_->clear();
    }
    
    // Reset search running flag to ensure it's in the correct state
    search_running_.store(false, std::memory_order_release);

    // Run the search with proper exception handling and ensure we're properly guarded
    try {
        // Make sure all cleanup happens even if exceptions occur
        struct SearchCleanup {
            MCTSEngine* engine;
            SearchCleanup(MCTSEngine* e) : engine(e) {}
            ~SearchCleanup() {
                // Clean up any resources allocated during the search
                if (engine->search_running_.load(std::memory_order_acquire)) {
                    engine->search_running_.store(false, std::memory_order_release);
                }
            }
        } cleanup_guard(this);
        
        // Run the actual search
        runSearch(state);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during search: " << e.what() << std::endl;
        // Ensure proper cleanup before rethrowing
        safelyStopEvaluator();
        throw;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    SearchResult result;
    result.action = -1; // Default invalid action

    try {
        // Extract action probabilities based on visit counts
        result.probabilities = getActionProbabilities(root_.get(), settings_.temperature);

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
            auto legal_moves = state.getLegalMoves();
            if (!legal_moves.empty()) {
                result.action = legal_moves[0];
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
    // Reset statistics
    last_stats_ = MCTSStats();

    // Clean up the old root if it exists
    root_.reset();

    // Safety guard: ensure we're not already running a search
    if (search_running_.exchange(true)) {
        throw std::runtime_error("Another search is already in progress");
    }

    // Set up automatic cleanup on exit (using RAII)
    struct SearchGuard {
        std::atomic<bool>& flag;
        SearchGuard(std::atomic<bool>& f) : flag(f) {}
        ~SearchGuard() { flag.store(false, std::memory_order_release); }
    } search_guard(search_running_);

    try {
        // Check if we should use the transposition table
        if (use_transposition_table_ && transposition_table_) {
            try {
                // Get hash of the current state
                uint64_t hash = state.getHash();
                
                // Use mutex for thread-safe access to the transposition table at the root level
                static std::mutex root_tt_mutex;
                std::lock_guard<std::mutex> lock(root_tt_mutex);
                
                // Look up existing position
                MCTSNode* existing_node = transposition_table_->get(hash);
                
                if (existing_node) {
                    // Found in transposition table, but we'll still create a new root
                    // to avoid ownership issues, while copying statistics if available
                    std::cout << "Found position in transposition table with " 
                              << existing_node->getVisitCount() << " visits" << std::endl;
                    
                    // Create new root with clone of the state
                    root_ = std::make_unique<MCTSNode>(state.clone());
                    
                    // We could potentially copy statistics here in a more advanced implementation
                } else {
                    // Not found, create a new root
                    root_ = std::make_unique<MCTSNode>(state.clone());
                }
                
                // Store the new root node in the transposition table
                // This is safe because we own the root_ node through the unique_ptr
                if (root_) {
                    transposition_table_->store(hash, root_.get(), 0);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error accessing transposition table: " << e.what() << std::endl;
                // If there was an error, create a new root without involving the TT
                root_ = std::make_unique<MCTSNode>(state.clone());
            }
        } else {
            // Not using transposition table, always create a new root
            root_ = std::make_unique<MCTSNode>(state.clone());
        }

        // Ensure we have a valid root
        if (!root_) {
            throw std::runtime_error("Failed to create root node");
        }

        // Add Dirichlet noise to root node policy for exploration
        if (settings_.add_dirichlet_noise) {
            addDirichletNoise(root_.get());
        }

        // Reset simulation counter
        active_simulations_.store(0, std::memory_order_release);

        // Serial mode: run in current thread
        if (settings_.num_threads == 0) {
            for (int i = 0; i < settings_.num_simulations; ++i) {
                runSimulation(root_.get());
            }
        } else {
            // Create worker threads if needed
            createWorkerThreads();

            // Distribute simulations to worker threads
            distributeSimulations();
        }

        // Count nodes and find max depth
        countTreeStatistics();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in runSearch: " << e.what() << std::endl;
        throw; // Rethrow after cleanup (SearchGuard will reset search_running_)
    }
    catch (...) {
        std::cerr << "Unknown error in runSearch" << std::endl;
        throw; // Rethrow after cleanup
    }
}

// Helper method to create worker threads if needed
void MCTSEngine::createWorkerThreads() {
    std::lock_guard<std::mutex> lock(cv_mutex_);
    
    // Only create threads if they don't exist yet
    if (worker_threads_.empty() && settings_.num_threads > 0) {
        worker_threads_.reserve(settings_.num_threads);
        
        for (int i = 0; i < settings_.num_threads; ++i) {
            worker_threads_.emplace_back([this, thread_id = i]() {
                // Set thread name on Windows (platform-specific code omitted for brevity)
                
                // Worker thread main loop
                while (!shutdown_.load(std::memory_order_acquire)) {
                    bool has_work = false;
                    {
                        std::unique_lock<std::mutex> lock(cv_mutex_);
                        // Wait with timeout to avoid missed signals
                        has_work = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                            return active_simulations_.load(std::memory_order_acquire) > 0 || 
                                   shutdown_.load(std::memory_order_acquire);
                        });
                    }
                    
                    // Check shutdown flag again after waiting
                    if (shutdown_.load(std::memory_order_acquire)) break;
                    
                    // Run simulations if work is available
                    if (has_work && search_running_.load(std::memory_order_acquire)) {
                        processPendingSimulations();
                    } else {
                        // Small sleep to avoid busy waiting
                        std::this_thread::yield();
                    }
                }
            });
        }
    }
}

// Helper method to process available simulations
void MCTSEngine::processPendingSimulations() {
    // Retrieve simulations to run (using atomic fetch_sub for thread safety)
    int sims_to_run = 0;
    int current_active = active_simulations_.load(std::memory_order_acquire);
    
    // Limit the number of simulations to process at once
    // Taking too many at once might lead to worker threads being idle while one thread does all the work
    int max_per_thread = 2;
    
    // Try to take a reasonable number of simulations at once
    do {
        if (current_active <= 0) return; // No work available
        
        sims_to_run = std::min(max_per_thread, current_active);
    } while (!active_simulations_.compare_exchange_weak(
        current_active, current_active - sims_to_run, 
        std::memory_order_acq_rel, std::memory_order_acquire));
    
    // Safety check for root validity
    if (!root_ || !search_running_.load(std::memory_order_acquire)) {
        return;
    }
    
    // Run the simulations
    for (int i = 0; i < sims_to_run; ++i) {
        try {
            runSimulation(root_.get());
        } catch (const std::exception& e) {
            std::cerr << "Error in simulation: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error in simulation" << std::endl;
        }
        
        // After each simulation, give other threads a chance 
        // This helps prevent one thread from doing all the work
        if (i < sims_to_run - 1 && settings_.num_threads > 1) {
            std::this_thread::yield();
        }
    }
}

// Helper method to distribute simulations
void MCTSEngine::distributeSimulations() {
    int remaining = settings_.num_simulations;
    const int batch_size = std::max(1, std::min(100, settings_.num_simulations / 10 + 1));
    
    auto wait_start = std::chrono::steady_clock::now();
    
    while (remaining > 0 && !shutdown_.load(std::memory_order_acquire)) {
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - wait_start);
        if (elapsed.count() > 30) {
            std::cerr << "Warning: Timeout during simulation distribution" << std::endl;
            break;
        }
        
        // Check current active simulations before adding more
        int current_active = active_simulations_.load(std::memory_order_acquire);
        if (current_active > settings_.num_threads * 2) {
            // If there are already plenty of simulations in queue for threads to process,
            // wait a bit before adding more to avoid overloading
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            cv_.notify_all(); // Make sure threads are working on existing simulations
            continue;
        }
        
        // Determine batch size for this iteration
        int batch = std::min(batch_size, remaining);
        
        // Add to active simulations counter
        active_simulations_.fetch_add(batch, std::memory_order_release);
        
        // Notify worker threads
        cv_.notify_all();
        
        remaining -= batch;
        
        // Small sleep to avoid overloading the queue
        if (remaining > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    // Wait for all simulations to complete with timeout
    waitForSimulationsToComplete(wait_start);
}

// Helper method to wait for simulations to complete
void MCTSEngine::waitForSimulationsToComplete(std::chrono::steady_clock::time_point start_time) {
    const int max_wait_seconds = 30;
    int sleep_ms = 1;
    int last_active = -1;
    int stalled_count = 0;
    
    while (active_simulations_.load(std::memory_order_acquire) > 0) {
        // Get current active count
        int current_active = active_simulations_.load(std::memory_order_acquire);
        
        // Check if we're making progress
        if (current_active == last_active) {
            stalled_count++;
            // If we've been stalled for a while, log and give up
            if (stalled_count > 20) {
                std::cerr << "Warning: Simulations appear to be stalled with " 
                         << current_active << " remaining" << std::endl;
                break;
            }
        } else {
            stalled_count = 0;
        }
        last_active = current_active;
        
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        if (elapsed.count() > max_wait_seconds) {
            std::cerr << "Warning: Timeout waiting for simulations to complete" << std::endl;
            break;
        }
        
        // Wake up any sleeping threads
        cv_.notify_all();
        
        // Exponential backoff up to 10ms
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        sleep_ms = std::min(10, sleep_ms * 2);
    }
}

// Helper method to count tree statistics
void MCTSEngine::countTreeStatistics() {
    size_t count = 0;
    int max_depth = 0;
    
    // Helper function to count nodes and find max depth
    std::function<void(MCTSNode*, int)> countNodes = [&count, &max_depth, &countNodes](MCTSNode* node, int depth) {
        if (!node) return;
        
        count++;
        max_depth = std::max(max_depth, depth);
        
        for (auto* child : node->getChildren()) {
            if (child) {
                countNodes(child, depth + 1);
            }
        }
    };
    
    if (root_) {
        countNodes(root_.get(), 0);
    }
    
    last_stats_.total_nodes = count;
    last_stats_.max_depth = max_depth;
}

void MCTSEngine::runSimulation(MCTSNode* root) {
    if (!root) {
        return;
    }

    try {
        // Selection phase - find a leaf node
        auto [leaf, path] = selectLeafNode(root);

        if (!leaf) {
            return;  // Something went wrong during selection
        }

        // Expansion and evaluation phase
        float value = 0.0f;
        try {
            // We need to handle terminal states differently
            if (leaf->isTerminal()) {
                // For terminal states, value depends on game result
                auto game_result = leaf->getState().getGameResult();
                if (game_result == core::GameResult::WIN_PLAYER1) {
                    value = leaf->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
                } else if (game_result == core::GameResult::WIN_PLAYER2) {
                    value = leaf->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
                } else {
                    value = 0.0f; // Draw
                }
            } else {
                // For non-terminal states, expand and evaluate
                value = expandAndEvaluate(leaf, path);
            }
        } catch (const std::bad_alloc& e) {
            #if MCTS_DEBUG
            std::cerr << "Memory allocation error during expansion/evaluation: " << e.what() << std::endl;
            #endif
            value = 0.0f;  // Use a default value
        } catch (const std::exception& e) {
            #if MCTS_DEBUG
            std::cerr << "Error during expansion/evaluation: " << e.what() << std::endl;
            #endif
            value = 0.0f;  // Use a default value
        }

        // Backpropagation phase
        backPropagate(path, value);
    } catch (const std::exception& e) {
        #if MCTS_DEBUG
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        #endif
    }
}

std::pair<MCTSNode*, std::vector<MCTSNode*>> MCTSEngine::selectLeafNode(MCTSNode* root) {
    std::vector<MCTSNode*> path;
    MCTSNode* node = root;
    path.push_back(node);

    // Selection phase - find a leaf node
    while (!node->isLeaf() && !node->isTerminal()) {
        // Apply virtual loss to this node before selecting a child
        node->addVirtualLoss();

        // Select child according to PUCT formula
        node = node->selectChild(settings_.exploration_constant);
        if (!node) {
            break;  // No valid child was found
        }

        // Apply virtual loss to selected child
        node->addVirtualLoss();

        // Check transposition table for this node, but only if not terminal
        // Only check transposition table after we've gone a certain depth to reduce hit rate
        if (use_transposition_table_ && !node->isTerminal() && path.size() > 2) {
            try {
                // Get the hash for this position
                uint64_t hash = node->getState().getHash();
                
                // Add a memory barrier before accessing the transposition table
                std::atomic_thread_fence(std::memory_order_acquire);
                
                // We're tracking transposition hits at the transposition table level
                
                // Look up node in transposition table
                MCTSNode* transposition = transposition_table_->get(hash);

                // Only use transposition table entries if search is still active
                if (transposition && !search_running_.load(std::memory_order_acquire)) {
                    // Search is being shut down, don't use transposition table
                    continue;
                }
                
                // Perform safety checks on the transposition node before using it
                if (transposition && transposition != node) {
                    try {
                        // Check that the node is valid by attempting to access its state
                        const auto& state = transposition->getState();
                        if (state.getHash() != hash) {
                            // Hash mismatch, don't use this node
                            continue;
                        }
                        
                        // Make sure we don't create cycles in the path
                        bool node_already_in_path = false;
                        for (auto* p : path) {
                            if (p == transposition) {
                                node_already_in_path = true;
                                break;
                            }
                        }
                        
                        if (!node_already_in_path) {
                            // Use a safer approach with a dedicated table access mutex
                            static std::mutex transposition_mutex;
                            std::lock_guard<std::mutex> lock(transposition_mutex);
                            
                            // Double-check that the node is still valid and the search is running
                            if (!search_running_.load(std::memory_order_acquire)) {
                                continue;
                            }
                            
                            try {
                                // Final validation - make sure we can still access the state
                                int visits = transposition->getVisitCount();
                                if (visits < 0) {
                                    continue;
                                }
                                
                                // We rely on the transposition table itself to track hit statistics
                                
                                // Found in transposition table - copy node value, don't replace
                                // This is safer than replacing the node in the path
                                if (visits > node->getVisitCount()) {
                                    // In a full implementation, we'd copy statistics here
                                    // For now, just record the hit and continue normally
                                }
                                
                                // For correct TT usage, we should actually use the transposition node
                                // but with extreme care to avoid memory issues
                                path.push_back(transposition);
                                node = transposition;
                                continue;
                            } catch (...) {
                                // If accessing the node fails during final validation, skip it
                                continue;
                            }
                        }
                    } catch (...) {
                        // If we can't access the state, don't use this node
                        continue;
                    }
                }
            } catch (const std::exception& e) {
                // Ignore exceptions during transposition table lookup
                std::cerr << "Error in transposition table lookup: " << e.what() << std::endl;
            }
        }

        // Add to path
        path.push_back(node);
    }

    return {node, path};
}

float MCTSEngine::expandAndEvaluate(MCTSNode* leaf, const std::vector<MCTSNode*>& path) {
    // Memory barrier to ensure consistent view of leaf pointer across threads
    std::atomic_thread_fence(std::memory_order_acquire);
    
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
            std::cerr << "Error evaluating terminal state: " << e.what() << std::endl;
            return 0.0f;
        }
    }
    
    // Expand the leaf node
    try {
        // Memory barrier before accessing leaf again
        std::atomic_thread_fence(std::memory_order_acquire);
        leaf->expand();
    } catch (const std::exception& e) {
        std::cerr << "Error expanding leaf node: " << e.what() << std::endl;
        return 0.0f;
    }
    
    // Store in transposition table if enabled
    if (use_transposition_table_ && transposition_table_ && search_running_.load(std::memory_order_acquire)) {
        try {
            // Memory barrier before accessing leaf again
            std::atomic_thread_fence(std::memory_order_acquire);
            
            // Additional validation before storing in transposition table
            if (leaf && path.size() > 0) {
                // Each node is owned by a specific thread (the one that created it)
                // Use another global mutex for thread-safe access to the transposition table
                static std::mutex tt_store_mutex;
                
                // Lock scope - minimize the critical section
                {
                    std::lock_guard<std::mutex> lock(tt_store_mutex);
                    
                    // Recheck search state after acquiring lock
                    if (!search_running_.load(std::memory_order_acquire)) {
                        return 0.0f; // Early exit if search has been stopped
                    }
                    
                    // Validate leaf node is still accessible
                    try {
                        // Accessing the state ensures the node is still valid
                        uint64_t hash = leaf->getState().getHash();
                        
                        // Check if this node is already in the transposition table to avoid duplication
                        MCTSNode* existing = transposition_table_->get(hash);
                        
                        // Only update if we have a better node (more visits)
                        if (!existing) {
                            // New position - store it
                            transposition_table_->store(hash, leaf, path.size());
                        } else if (leaf->getVisitCount() > existing->getVisitCount()) {
                            // We have a better node - update it
                            // Note: We're not transferring ownership, we're just updating a reference
                            transposition_table_->store(hash, leaf, path.size());
                        }
                    } catch (...) {
                        // If we can't access the node state, skip storing it
                    }
                } // lock scope ends here
            }
        } catch (const std::exception& e) {
            std::cerr << "Error storing in transposition table: " << e.what() << std::endl;
            // Continue, this is not critical
        }
    }
    
    // If leaf has no children after expansion, return a default value
    if (leaf->getChildren().empty()) {
        return 0.0f;
    }
    
    // Evaluate with the neural network - will set return value later
    
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
            // Add memory barrier before accessing leaf
            std::atomic_thread_fence(std::memory_order_acquire);
            
            if (!leaf) {
                throw std::runtime_error("Leaf became invalid during evaluation");
            }
            
            auto state_clone = leaf->getState().clone();
            if (!state_clone) {
                throw std::runtime_error("Failed to clone state for evaluation");
            }
            
            // Capture the leaf in a local variable to prevent it from being nullified
            // Store a local copy of the leaf node for thread safety
            MCTSNode* leaf_copy = leaf;
            
            // Create a shared_ptr to track the leaf's lifetime
            std::shared_ptr<MCTSNode*> leaf_ptr = std::make_shared<MCTSNode*>(leaf_copy);
            
            // Create a wrapper to ensure the pointer is properly validated before use
            auto safe_use_leaf = [leaf_ptr](const std::vector<float>& policy) {
                std::atomic_thread_fence(std::memory_order_acquire);
                MCTSNode* node = *leaf_ptr;
                if (node) {
                    try {
                        node->setPriorProbabilities(policy);
                    } catch (...) {
                        // Ignore exceptions during setPriorProbabilities
                    }
                }
            };
            
            // Submit evaluation request
            auto future = evaluator_->evaluateState(leaf_copy, std::move(state_clone));
            
            // Wait for the result with a reasonable timeout
            auto status = future.wait_for(std::chrono::seconds(2));
            if (status == std::future_status::ready) {
                try {
                    auto result = future.get();
                    
                    // Use the safe wrapper to update the leaf
                    safe_use_leaf(result.policy);
                    
                    return result.value;
                } catch (const std::exception& e) {
                    std::cerr << "Error processing evaluation result: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Unknown error processing evaluation result" << std::endl;
                }
            } else {
                // Timed out waiting for evaluation, use uniform prior
                std::cerr << "Warning: Timed out waiting for neural network evaluation" << std::endl;
                
                try {
                    // Create a uniform policy
                    int action_space_size = 0;
                    
                    // Try to get action space size safely
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (leaf) {
                        try {
                            action_space_size = leaf->getState().getActionSpaceSize();
                        } catch (...) {
                            // Fallback to reasonable default
                            action_space_size = 10;
                        }
                    } else {
                        action_space_size = 10;
                    }
                    
                    std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                    
                    // Use the safe wrapper to update the leaf
                    safe_use_leaf(uniform_policy);
                } catch (...) {
                    // Ignore any errors during fallback processing
                }
            }
            
            return 0.0f;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during neural network evaluation: " << e.what() << std::endl;
        
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

void MCTSEngine::backPropagate(std::vector<MCTSNode*>& path, float value) {
    // Value alternates sign as we move up the tree (perspective changes)
    bool invert = false;
    
    // Process nodes in reverse order (from leaf to root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        // Check for null pointers as a safety measure
        MCTSNode* node = *it;
        if (!node) continue;
        
        float update_value = invert ? -value : value;
        
        try {
            // Remove virtual loss and update node statistics
            node->removeVirtualLoss();
            node->update(update_value);
        } catch (const std::exception& e) {
            // Log but continue with other nodes
            std::cerr << "Error during backpropagation: " << e.what() << std::endl;
        } catch (...) {
            // Unknown error - continue with other nodes
            std::cerr << "Unknown error during backpropagation" << std::endl;
        }
        
        // Alternate perspective for next level
        invert = !invert;
    }
}

std::vector<float> MCTSEngine::getActionProbabilities(MCTSNode* root, float temperature) {
    // CRITICAL SECTION: Acquire a mutex to protect access to the tree
    // This prevents concurrent modifications to the tree structure during probability calculation
    static std::mutex action_prob_mutex;
    std::lock_guard<std::mutex> lock(action_prob_mutex);
    
    // Validate input parameters with defensive checks
    if (!root) {
        std::cerr << "getActionProbabilities called with null root" << std::endl;
        return std::vector<float>();
    }
    
    int action_space_size = 0;
    try {
        action_space_size = root->getState().getActionSpaceSize();
    } catch (const std::exception& e) {
        std::cerr << "Error getting action space size: " << e.what() << std::endl;
        return std::vector<float>();
    }
    
    // Prepare uniform empty distribution as fallback
    std::vector<float> empty_distribution(action_space_size, 1.0f / std::max(1, action_space_size));
    
    // Check for empty children early with extra defensive coding
    const auto& child_nodes = root->getChildren();
    const auto& action_nodes = root->getActions();
    
    if (child_nodes.empty() || action_nodes.empty()) {
        return empty_distribution;
    }
    
    // Create deep copies of all necessary data to prevent race conditions
    // This is crucial for thread safety - we want to operate on our own copy of data
    std::vector<int> actions(action_nodes);
    std::vector<MCTSNode*> children;
    children.reserve(child_nodes.size());
    
    // Carefully copy child pointers with null checks
    for (auto* child : child_nodes) {
        // Use nullptr for any invalid children
        children.push_back(child ? child : nullptr);
    }
    
    // Sanity check: actions and children should have the same size
    if (actions.size() != children.size()) {
        std::cerr << "Mismatch between actions and children size" << std::endl;
        return empty_distribution;
    }
    
    // Get visit counts for each child with careful null pointer handling
    std::vector<float> counts;
    counts.reserve(children.size());
    
    for (auto* child : children) {
        if (!child) {
            counts.push_back(0.0f);
        } else {
            try {
                counts.push_back(static_cast<float>(child->getVisitCount()));
            } catch (...) {
                // If any exception occurs during visit count retrieval, use 0
                counts.push_back(0.0f);
            }
        }
    }
    
    // Check if all counts are zero
    bool all_zero = true;
    for (float count : counts) {
        if (count > 0.0f) {
            all_zero = false;
            break;
        }
    }
    
    // If all counts are zero, return uniform distribution
    if (all_zero) {
        float uniform_prob = 1.0f / children.size();
        std::vector<float> child_probs(children.size(), uniform_prob);
        
        // Map to full action space
        std::vector<float> action_probs(action_space_size, 0.0f);
        for (size_t i = 0; i < actions.size(); ++i) {
            int action = actions[i];
            if (action >= 0 && action < action_space_size) {
                action_probs[action] = child_probs[i];
            }
        }
        return action_probs;
    }
    
    // Process based on temperature
    std::vector<float> probabilities;
    probabilities.reserve(counts.size());
    
    // Temperature near zero - deterministic selection
    if (temperature < 0.01f) {
        // Find the move with highest visit count
        float max_count = -1.0f;
        size_t max_idx = 0;
        
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > max_count) {
                max_count = counts[i];
                max_idx = i;
            }
        }
        
        // Set all probabilities to 0 except the highest
        probabilities.resize(counts.size(), 0.0f);
        if (max_idx < probabilities.size()) {
            probabilities[max_idx] = 1.0f;
        }
    } 
    // Normal temperature-based selection
    else {
        // Find the maximum count for numerical stability
        float max_count = *std::max_element(counts.begin(), counts.end());
        
        if (max_count <= 0.0f) {
            // Uniform distribution if no visits
            float uniform_prob = 1.0f / counts.size();
            probabilities.resize(counts.size(), uniform_prob);
        } else {
            // Apply temperature with numerical stability
            float sum = 0.0f;
            for (float count : counts) {
                // Avoid division by zero and large exponents
                float scaled_count = 0.0f;
                if (count > 0.0f) {
                    // Cap temperature to a reasonable range for numerical stability
                    float safe_temperature = std::min(5.0f, std::max(0.01f, temperature));
                    
                    if (safe_temperature >= 0.99f && safe_temperature <= 1.01f) {
                        // Special case for temperature near 1.0 - just use counts directly
                        scaled_count = count / max_count;
                    } else if (safe_temperature > 3.0f) {
                        // High temperature - approximate with a more stable calculation
                        // that approaches uniform distribution
                        float ratio = count / max_count;
                        // Boost low counts more than high counts
                        scaled_count = 0.1f + 0.9f * ratio;
                    } else {
                        // Regular temperature case with improved safety
                        try {
                            scaled_count = std::pow(count / max_count, 1.0f / safe_temperature);
                        } catch (...) {
                            // Fallback on any numerical error
                            scaled_count = count > 0 ? 1.0f : 0.0f;
                        }
                    }
                }
                probabilities.push_back(scaled_count);
                sum += scaled_count;
            }
            
            // Normalize with extra safety check
            if (sum > 0.0001f) {  // Use a small epsilon to avoid floating point issues
                for (auto& prob : probabilities) {
                    prob /= sum;
                    // Safety clamp to valid probability range
                    prob = std::min(1.0f, std::max(0.0f, prob));
                }
            } else {
                // Fallback to uniform if sum is too small
                float uniform_prob = 1.0f / counts.size();
                std::fill(probabilities.begin(), probabilities.end(), uniform_prob);
            }
            
            // Final safety - renormalize if the sum drifted off 1.0
            float final_sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
            if (std::abs(final_sum - 1.0f) > 0.01f && final_sum > 0.0f) {
                for (auto& prob : probabilities) {
                    prob /= final_sum;
                }
            }
        }
    }
    
    // Map child probabilities to action space probabilities
    std::vector<float> action_probabilities(action_space_size, 0.0f);
    
    for (size_t i = 0; i < actions.size() && i < probabilities.size(); ++i) {
        int action = actions[i];
        if (action >= 0 && action < action_space_size) {
            action_probabilities[action] = probabilities[i];
        }
    }
    
    return action_probabilities;
}

void MCTSEngine::addDirichletNoise(MCTSNode* root) {
    // Thread safety: use a static mutex for this critical section
    static std::mutex dirichlet_mutex;
    std::lock_guard<std::mutex> lock(dirichlet_mutex);
    
    if (!root) {
        return;
    }
    
    // Expand root node if it's not already expanded
    if (root->isLeaf() && !root->isTerminal()) {
        try {
            // First expand the root - this is thread-safe internally due to expansion_mutex in MCTSNode
            root->expand();
            
            if (root->getChildren().empty()) {
                return;  // No children to add noise to
            }
            
            // Create local variables to avoid potential race conditions
            int action_space_size = 0;
            std::vector<float> policy;
            
            // Get prior probabilities for the root node with proper exception handling
            try {
                // Create a clone of the state to avoid modifying the original
                auto state_clone = root->getState().clone();
                if (!state_clone) {
                    throw std::runtime_error("Failed to clone state for Dirichlet noise");
                }
                
                // Get action space size and store it locally
                action_space_size = root->getState().getActionSpaceSize();
                
                // Evaluate in a thread-safe manner
                if (settings_.num_threads == 0) {
                    // Serial mode
                    std::vector<std::unique_ptr<core::IGameState>> states;
                    states.push_back(std::move(state_clone));
                    auto outputs = evaluator_->getInferenceFunction()(states);
                    if (!outputs.empty()) {
                        policy = outputs[0].policy;
                    }
                } else {
                    // Parallel mode with timeout protection
                    auto future = evaluator_->evaluateState(root, std::move(state_clone));
                    auto status = future.wait_for(std::chrono::seconds(1)); // Reduced timeout
                    if (status == std::future_status::ready) {
                        try {
                            auto result = future.get();
                            policy = result.policy;
                        } catch (...) {
                            // Ignore get() errors
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during state evaluation for Dirichlet noise: " << e.what() << std::endl;
            }
            
            // If we failed to get a valid policy, create a uniform one
            if (policy.empty() && action_space_size > 0) {
                policy.resize(action_space_size, 1.0f / action_space_size);
            } else if (policy.empty()) {
                // Try to get action space size again if we failed earlier
                try {
                    action_space_size = root->getState().getActionSpaceSize();
                    policy.resize(action_space_size, 1.0f / action_space_size);
                } catch (...) {
                    // Last resort - arbitrary size uniform policy
                    policy.resize(10, 0.1f);
                }
            }
            
            // Set prior probabilities
            if (!policy.empty()) {
                root->setPriorProbabilities(policy);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in addDirichletNoise: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error in addDirichletNoise" << std::endl;
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
    std::cout << "Adding Dirichlet noise to root node with epsilon = " 
             << settings_.dirichlet_epsilon << std::endl;
    #endif
    
    // Apply noise to children's prior probabilities
    for (size_t i = 0; i < root->getChildren().size(); ++i) {
        MCTSNode* child = root->getChildren()[i];
        float prior = child->getPriorProbability();
        float noisy_prior = (1.0f - settings_.dirichlet_epsilon) * prior + 
                           settings_.dirichlet_epsilon * noise[i];
        child->setPriorProbability(noisy_prior);
    }
}

} // namespace mcts
} // namespace alphazero