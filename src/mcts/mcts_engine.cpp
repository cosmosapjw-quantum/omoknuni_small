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
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(std::make_unique<TranspositionTable>(128)), // 128 MB default
      use_transposition_table_(true),
      evaluator_started_(false),
      num_workers_actively_processing_(0) {
    
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
      transposition_table_(std::make_unique<TranspositionTable>(128)), // 128 MB default
      use_transposition_table_(true),
      evaluator_started_(false),
      num_workers_actively_processing_(0) {
    
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
      worker_threads_(std::move(other.worker_threads_)),
      shutdown_(other.shutdown_.load()),
      active_simulations_(other.active_simulations_.load()),
      search_running_(other.search_running_.load()),
      random_engine_(std::move(other.random_engine_)),
      transposition_table_(std::move(other.transposition_table_)),
      use_transposition_table_(other.use_transposition_table_),
      evaluator_started_(other.evaluator_started_),
      num_workers_actively_processing_(other.num_workers_actively_processing_.load()) {
    
    // Validate the moved evaluator
    if (!evaluator_) {
        std::cerr << "WARNING: evaluator_ is null after move constructor" << std::endl;
    }
    
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
        num_workers_actively_processing_ = other.num_workers_actively_processing_.load();
        
        // Validate the moved evaluator
        if (!evaluator_) {
            std::cerr << "WARNING: evaluator_ is null after move assignment" << std::endl;
        }
        
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
    // Signal shutdown to all worker threads
    shutdown_ = true;
    cv_.notify_all();
    
    // Join all worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Stop the evaluator if it was started
    safelyStopEvaluator();
}

SearchResult MCTSEngine::search(const core::IGameState& state) {
    auto start_time = std::chrono::steady_clock::now();

    // Make sure evaluator is running
    if (!ensureEvaluatorStarted()) {
        throw std::runtime_error("Failed to start neural network evaluator");
    }

    // Run the search with proper exception handling
    try {
        runSearch(state);
    }
    catch (const std::exception& e) {
        // Commented out: Error during search with error message
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
        // Commented out: Error extracting search results with error message
        
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
    
    // Wait for all worker threads to finish processing before cleaning up
    // from any previous search iteration on this engine instance.
    {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        // Wait for workers to finish their processing block, indicated by num_workers_actively_processing_ == 0.
        // Timeout after a reasonable period (e.g., 5 seconds).
        if (!cv_.wait_for(lock, std::chrono::seconds(5), [this]() {
            return num_workers_actively_processing_.load(std::memory_order_acquire) == 0 || 
                   shutdown_.load(std::memory_order_acquire);
        })) {
            // Timeout occurred and the condition is false (workers still active and not shutting down)
            if (!shutdown_.load(std::memory_order_acquire)) {
                // WARNING: Timeout waiting for worker threads, will attempt recovery
                
                // Recovery strategy: reset active_simulations_ to 0 to prevent workers from taking new work
                active_simulations_.store(0, std::memory_order_release);
                
                // Signal all worker threads to check their state
                cv_.notify_all();
                
                // Wait a short time to see if this resolves the issue
                if (!cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                    return num_workers_actively_processing_.load(std::memory_order_acquire) == 0 || 
                          shutdown_.load(std::memory_order_acquire);
                })) {
                    // If still not resolved, we'll proceed anyway but log it
                    // WARNING: Unable to fully recover stuck workers, proceeding anyway
                    // Force reset the counter - risky but prevents permanent stalling
                    num_workers_actively_processing_.store(0, std::memory_order_release);
                }
            }
        }
        // It's now safer to proceed with resetting shared resources like the root node.
    }
    
    // Clean up the old root if it exists. This invalidates all nodes in the previous tree.
    root_.reset();

    // If using the transposition table, it must be cleared now to remove any dangling pointers
    // from the tree that was just deleted by root_.reset(). This ensures that the TT
    // does not serve stale pointers from a previous, unrelated search tree context.
    if (use_transposition_table_ && transposition_table_) {
        try {
            transposition_table_->clear();
        } catch (const std::exception& e) {
            // In case of any exception during clear, recreate the table entirely
            // This is safer than potentially having dangling pointers
            size_t size_mb = 128; // Default size
            size_t num_shards = std::max(4u, std::thread::hardware_concurrency());
            if (settings_.num_threads > 0) {
                num_shards = std::max(size_t(settings_.num_threads), num_shards);
            }
            transposition_table_ = std::make_unique<TranspositionTable>(size_mb, num_shards);
        }
    }
    
    // Create the new root node.
    // If using the transposition table, it has just been cleared. We create a new root
    // from the input state and then add it to the TT.
    // We do not attempt to find the new root in the just-cleared TT, as that could lead to
    // using stale pointers if the clear operation was somehow incomplete or a hash collided.
    try {
        auto state_clone = state.clone();
        if (!state_clone) {
            throw std::runtime_error("Failed to clone state for root node");
        }
        
        // Additional validation of the cloned state
        if (!state_clone->validate()) {
            throw std::runtime_error("Cloned state failed validation");
        }
        
        root_ = std::make_unique<MCTSNode>(std::move(state_clone));

        // Ensure we have a valid root
        if (!root_) {
            throw std::runtime_error("Failed to create root node");
        }
        
        // Validate the root node's state
        if (!root_->getState().validate()) {
            throw std::runtime_error("Root node state invalid after creation");
        }

        // If using transposition table, store the new root.
        if (use_transposition_table_ && transposition_table_) {
            uint64_t hash = state.getHash(); // Get hash of the new root's state
            transposition_table_->store(hash, root_.get(), 0);
            
            #if MCTS_DEBUG
            // Commented out: Debug printing about storing new root in transposition table with hash value
            #endif
        }

        // Add Dirichlet noise to root node policy for exploration
        if (settings_.add_dirichlet_noise) {
            addDirichletNoise(root_.get());
        }

        // Set search running flag
        search_running_ = true;
        active_simulations_ = 0;

        // Create worker threads if they don't exist yet
        if (worker_threads_.empty() && settings_.num_threads > 0) {
            for (int i = 0; i < settings_.num_threads; ++i) {
                worker_threads_.emplace_back([this, thread_id = i]() {
                    // Set thread name on Windows
                    #ifdef _MSC_VER
                    const DWORD MS_VC_EXCEPTION = 0x406D1388;
                    #pragma pack(push, 8)
                    struct THREADNAME_INFO {
                        DWORD dwType;     // Must be 0x1000
                        LPCSTR szName;    // Pointer to name (in user addr space)
                        DWORD dwThreadID; // Thread ID (-1=caller thread)
                        DWORD dwFlags;    // Reserved for future use, must be zero
                    };
                    #pragma pack(pop)

                    char threadName[32];
                    sprintf_s(threadName, "MCTSWorker%d", thread_id);
                    
                    THREADNAME_INFO info;
                    info.dwType = 0x1000;
                    info.szName = threadName;
                    info.dwThreadID = GetCurrentThreadId();
                    info.dwFlags = 0;
                    
                    __try {
                        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
                    } __except (EXCEPTION_EXECUTE_HANDLER) {
                        // Just continue
                    }
                    #endif

                    // Worker thread main loop
                    while (!shutdown_) {
                        bool has_work = false;
                        {
                            std::unique_lock<std::mutex> lock(cv_mutex_);
                            has_work = cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                                return active_simulations_.load(std::memory_order_acquire) > 0 || shutdown_;
                            });

                            if (shutdown_) break;
                        }

                        // Run simulations if work is available
                        if (has_work && root_) {
                            int actual_sims_taken_this_worker = 0;
                            const int max_sims_per_worker_cycle = 5; // Max simulations a worker tries to take in one go

                            for (int k = 0; k < max_sims_per_worker_cycle; ++k) {
                                if (active_simulations_.fetch_sub(1, std::memory_order_acq_rel) > 0) {
                                    // Successfully claimed one simulation
                                    actual_sims_taken_this_worker++;
                                } else {
                                    // Failed to claim (e.g., active_simulations_ was 0 or negative)
                                    // Add back the 1 we tried to subtract.
                                    active_simulations_.fetch_add(1, std::memory_order_release);
                                    break; // Stop trying to claim more in this cycle
                                }
                            }

                            if (actual_sims_taken_this_worker > 0) {
                                num_workers_actively_processing_.fetch_add(1, std::memory_order_release);
                                for (int i = 0; i < actual_sims_taken_this_worker; ++i) {
                                    if (shutdown_.load(std::memory_order_acquire)) {
                                        // If shutdown during processing, return the remaining (unprocessed) simulations to the pool
                                        active_simulations_.fetch_add(actual_sims_taken_this_worker - i, std::memory_order_release);
                                        break; // Exit loop
                                    }
                                    try {
                                        runSimulation(root_.get());
                                    } catch (const std::exception& e) {
                                        // Error in simulation
                                    } catch (...) {
                                        // Unknown error in simulation
                                    }
                                }
                                num_workers_actively_processing_.fetch_sub(1, std::memory_order_release);
                                cv_.notify_all(); // Notify potentially waiting main thread or other workers
                            }
                        }
                    }
                });
            }
        }

        // For tests with num_threads=0, run simulations in the current thread
        if (settings_.num_threads == 0) {
            for (int i = 0; i < settings_.num_simulations; ++i) {
                runSimulation(root_.get());
            }
        } else {
            // Launch simulations on worker threads in batches for better efficiency
            int remaining = settings_.num_simulations;
            const int batch_size = std::min(100, settings_.num_simulations / 10 + 1);

            while (remaining > 0) {
                int batch = std::min(batch_size, remaining);
                active_simulations_.fetch_add(batch, std::memory_order_release);
                cv_.notify_all();
                remaining -= batch;

                // Small sleep to avoid overloading the queue
                if (remaining > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            // Wait for all active workers to finish their current batch of simulations
            auto wait_processing_start = std::chrono::steady_clock::now();
            
            // Create a lambda for the wait condition to avoid code duplication
            auto no_active_workers = [this]() {
                return (active_simulations_.load(std::memory_order_acquire) <= 0 && 
                        num_workers_actively_processing_.load(std::memory_order_acquire) <= 0);
            };
            
            // First try a condition variable wait with periodic wakeups
            bool all_finished = false;
            for (int attempt = 0; attempt < 30 && !all_finished; ++attempt) { // Try for about 3 seconds total
                // Notify all workers to check their state
                cv_.notify_all();
                
                // Wait with timeout
                std::unique_lock<std::mutex> lock(cv_mutex_);
                all_finished = cv_.wait_for(lock, std::chrono::milliseconds(100), no_active_workers);
            }
            
            // If condition variable wait didn't work, fall back to polling with a longer timeout
            if (!all_finished) {
                while (!no_active_workers()) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - wait_processing_start);

                    if (elapsed.count() > 15) { // Reduced timeout from 30 to 15 seconds
                        // Warning: Timeout waiting for worker threads
                        // Force reset the counters to prevent permanent stalls
                        active_simulations_.store(0, std::memory_order_release);
                        num_workers_actively_processing_.store(0, std::memory_order_release);
                        break;
                    }
                    cv_.notify_all();
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            
            // Double-check that we have zero workers processing
            if (num_workers_actively_processing_.load(std::memory_order_acquire) > 0) {
                // Warning: Worker threads still processing after wait timeout
                // Force reset to prevent permanent stalling
                num_workers_actively_processing_.store(0, std::memory_order_release);
            }
        }

        // Set search running flag to false
        search_running_ = false;
        
        // Count nodes and find max depth
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
    catch (const std::exception& e) {
        // Error in runSearch
        search_running_ = false;
        throw;
    }
    catch (...) {
        // Unknown error in runSearch
        search_running_ = false;
        throw;
    }
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

        // Backpropagation phase
        backPropagate(path, value);
    } catch (const std::exception& e) {
        #if MCTS_DEBUG
        // Commented out: Debug error message during simulation
        #endif
    }
}

std::pair<MCTSNode*, std::vector<MCTSNode*>> MCTSEngine::selectLeafNode(MCTSNode* root) {
    std::vector<MCTSNode*> path;
    MCTSNode* node = root;
    
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
        MCTSNode* child = node->selectChild(settings_.exploration_constant);
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
                MCTSNode* transposition = transposition_table_->get(hash);

                if (transposition && transposition != node) {
                    // If a transposition is found, we use it after validating it's still a valid node
                    bool valid_transposition = false;
                    
                    try {
                        // More thorough validation - verify we can access key methods without exception
                        // and that the node has a valid state with consistent data
                        int visits = transposition->getVisitCount();
                        valid_transposition = visits >= 0 && 
                                             transposition->getState().validate() &&
                                             transposition->getState().getHash() == hash;
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

float MCTSEngine::expandAndEvaluate(MCTSNode* leaf, const std::vector<MCTSNode*>& path) {
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
            transposition_table_->store(hash, leaf, path.size());
            
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
    float value = 0.0f;
    
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
            
            // Wait for the result with a reasonable timeout
            auto status = future.wait_for(std::chrono::seconds(2));
            if (status == std::future_status::ready) {
                auto result = future.get();
                leaf->setPriorProbabilities(result.policy);
                return result.value;
            } else {
                // Timed out waiting for evaluation, use uniform prior
                // Commented out: Warning about timeout waiting for neural network evaluation
                
                int action_space_size = leaf->getState().getActionSpaceSize();
                std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                leaf->setPriorProbabilities(uniform_policy);
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

void MCTSEngine::backPropagate(std::vector<MCTSNode*>& path, float value) {
    // Value alternates sign as we move up the tree (perspective changes)
    bool invert = false;
    
    // Process nodes in reverse order (from leaf to root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* node = *it;
        float update_value = invert ? -value : value;
        
        // Remove virtual loss and update node statistics
        node->removeVirtualLoss();
        node->update(update_value);
        
        // Alternate perspective for next level
        invert = !invert;
    }
}

std::vector<float> MCTSEngine::getActionProbabilities(MCTSNode* root, float temperature) {
    if (!root || root->getChildren().empty()) {
        return std::vector<float>();
    }

    // Get actions and visit counts
    auto& actions = root->getActions();
    auto& children = root->getChildren();
    
    std::vector<float> counts;
    counts.reserve(children.size());

    for (auto* child : children) {
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

void MCTSEngine::addDirichletNoise(MCTSNode* root) {
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
        MCTSNode* child = root->getChildren()[i];
        float prior = child->getPriorProbability();
        float noisy_prior = (1.0f - settings_.dirichlet_epsilon) * prior + 
                           settings_.dirichlet_epsilon * noise[i];
        child->setPriorProbability(noisy_prior);
    }
}

} // namespace mcts
} // namespace alphazero