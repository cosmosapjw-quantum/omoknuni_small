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
    if (!evaluator_started_) {
        try {
            evaluator_->start();
            evaluator_started_ = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start evaluator: " << e.what() << std::endl;
            return false;
        }
    }
    return true;
}

void MCTSEngine::safelyStopEvaluator() {
    if (evaluator_started_) {
        try {
            evaluator_->stop();
            evaluator_started_ = false;
        } catch (const std::exception& e) {
            std::cerr << "Error stopping evaluator: " << e.what() << std::endl;
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
    #if MCTS_DEBUG
    std::cout << "Starting MCTS search" << std::endl;
    #endif

    auto start_time = std::chrono::steady_clock::now();

    // Make sure evaluator is running
    if (!ensureEvaluatorStarted()) {
        throw std::runtime_error("Failed to start neural network evaluator");
    }

    // Run the search
    runSearch(state);

    auto end_time = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    #if MCTS_DEBUG
    std::cout << "MCTS search complete, extracting policy" << std::endl;
    #endif
    
    SearchResult result;

    // Extract action probabilities based on visit counts
    result.probabilities = getActionProbabilities(root_.get(), settings_.temperature);

    // Select action according to the computed probability distribution
    if (!result.probabilities.empty()) {
        try {
            // Check if we're dealing with probabilities that sum to approximately 1
            float sum = std::accumulate(result.probabilities.begin(), result.probabilities.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.1f) {
                // Normalize probabilities if they don't sum to approximately 1
                for (auto& p : result.probabilities) {
                    p = (sum > 0.0f) ? p / sum : 1.0f / result.probabilities.size();
                }
            }
            
            // Temperature near zero - deterministic selection (argmax)
            if (settings_.temperature < 0.01f) {
                result.action = std::distance(
                    result.probabilities.begin(),
                    std::max_element(result.probabilities.begin(), result.probabilities.end())
                );
                
                #if MCTS_DEBUG
                std::cout << "Zero temperature, selecting max probability action: " << result.action << std::endl;
                #endif
            } else {
                // Sample according to the probability distribution
                float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(random_engine_);
                float cumsum = 0.0f;
                
                for (size_t i = 0; i < result.probabilities.size(); i++) {
                    cumsum += result.probabilities[i];
                    if (r <= cumsum) {
                        result.action = static_cast<int>(i);
                        break;
                    }
                }
                
                // Fallback in case of numerical issues
                if (result.action < 0) {
                    result.action = std::distance(
                        result.probabilities.begin(),
                        std::max_element(result.probabilities.begin(), result.probabilities.end())
                    );
                    
                    #if MCTS_DEBUG
                    std::cout << "Fallback to max probability action: " << result.action << std::endl;
                    #endif
                }
            }
        } catch (const std::exception& e) {
            // Fallback in case of any errors during action selection
            if (!result.probabilities.empty()) {
                result.action = std::distance(
                    result.probabilities.begin(),
                    std::max_element(result.probabilities.begin(), result.probabilities.end())
                );
            } else {
                // If even probabilities are empty, just pick first legal move
                auto legal_moves = state.getLegalMoves();
                result.action = legal_moves.empty() ? -1 : legal_moves[0];
            }
            
            #if MCTS_DEBUG
            std::cout << "Error during action selection: " << e.what() 
                     << ", falling back to action " << result.action << std::endl;
            #endif
        }
    } else if (root_ && !root_->getChildren().empty()) {
        // If for some reason we couldn't get probabilities, select the most visited child
        result.action = -1;
        int max_visits = -1;
        
        for (size_t i = 0; i < root_->getChildren().size(); i++) {
            auto child = root_->getChildren()[i];
            if (child->getVisitCount() > max_visits) {
                max_visits = child->getVisitCount();
                result.action = root_->getActions()[i];
            }
        }
        
        #if MCTS_DEBUG
        std::cout << "No probabilities, selecting most visited child: " << result.action << std::endl;
        #endif
    } else {
        // Last resort - just pick a legal move
        auto legal_moves = state.getLegalMoves();
        result.action = legal_moves.empty() ? -1 : legal_moves[0];
        
        #if MCTS_DEBUG
        std::cout << "No root children, selecting first legal move: " << result.action << std::endl;
        #endif
    }

    // Get value estimate
    result.value = root_ ? root_->getValue() : 0.0f;

    // Update statistics
    last_stats_.search_time = search_time;
    last_stats_.avg_batch_size = evaluator_->getAverageBatchSize();
    last_stats_.avg_batch_latency = evaluator_->getAverageBatchLatency();
    last_stats_.total_evaluations = evaluator_->getTotalEvaluations();

    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = 1000.0f * last_stats_.total_nodes / last_stats_.search_time.count();
    }

    // Add transposition table stats
    if (use_transposition_table_) {
        last_stats_.tt_hit_rate = transposition_table_->hitRate();
        last_stats_.tt_size = transposition_table_->size();
    }
    
    result.stats = last_stats_;

    #if MCTS_DEBUG
    std::cout << "Search complete in " << search_time.count() << "ms with "
              << last_stats_.total_nodes << " nodes ("
              << last_stats_.nodes_per_second << " nodes/sec)" << std::endl;
    #endif

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

    // Check if we should use the transposition table
    if (use_transposition_table_) {
        // Try to find the position in the transposition table
        uint64_t hash = state.getHash();
        MCTSNode* existing_node = transposition_table_->get(hash);

        if (existing_node) {
            // Found in transposition table, use its state to create a new root
            root_.reset(); // Release old root
            // Create a new root node from the state of the existing_node
            root_ = std::make_unique<MCTSNode>(existing_node->getState().clone());
            
            #if MCTS_DEBUG
            std::cout << "Position found in transposition table" << std::endl;
            #endif
        } else {
            // Not found, create a new root
            root_ = std::make_unique<MCTSNode>(state.clone());

            // Store in transposition table
            transposition_table_->store(hash, root_.get(), 0);
            
            #if MCTS_DEBUG
            std::cout << "Creating new root node, position not in transposition table" << std::endl;
            #endif
        }
    } else {
        // Not using transposition table, always create a new root
        root_ = std::make_unique<MCTSNode>(state.clone());
        
        #if MCTS_DEBUG
        std::cout << "Creating new root node (transposition table disabled)" << std::endl;
        #endif
    }

    // Add Dirichlet noise to root node policy for exploration
    if (settings_.add_dirichlet_noise) {
        addDirichletNoise(root_.get());
    }

    // Set search running flag
    search_running_ = true;

    // Create worker threads if they don't exist yet
    if (worker_threads_.empty() && settings_.num_threads > 0) {
        #if MCTS_DEBUG
        std::cout << "Creating " << settings_.num_threads << " worker threads" << std::endl;
        #endif
        
        for (int i = 0; i < settings_.num_threads; ++i) {
            worker_threads_.emplace_back([this, thread_id = i]() {
                // Set thread name if platform supports it
                #ifdef _MSC_VER
                // MSVC-specific thread naming
                const char* threadNameTemplate = "MCTSWorker%d";
                char threadName[32];
                sprintf_s(threadName, threadNameTemplate, thread_id);
                
                typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
                SetThreadDescriptionFunc setThreadDescription = 
                    (SetThreadDescriptionFunc)GetProcAddress(
                        GetModuleHandleA("kernel32.dll"), 
                        "SetThreadDescription");
                        
                if (setThreadDescription) {
                    wchar_t wThreadName[32];
                    swprintf(wThreadName, 32, L"%S", threadName);
                    setThreadDescription(GetCurrentThread(), wThreadName);
                }
                #endif

                #if MCTS_DEBUG
                std::cout << "Worker thread " << thread_id << " started" << std::endl;
                #endif
                
                // Worker thread main loop
                while (!shutdown_) {
                    // Wait for work or shutdown signal
                    bool has_work = false;
                    {
                        std::unique_lock<std::mutex> lock(cv_mutex_);
                        has_work = cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() {
                            return active_simulations_.load() > 0 || shutdown_;
                        });

                        if (shutdown_) break;
                    }

                    // Run a simulation if there's work to do
                    if (has_work && active_simulations_.load() > 0 && root_) {
                        // Try to grab multiple simulations at once for better efficiency
                        int sims_to_run = std::min(5, active_simulations_.load());
                        if (sims_to_run > 0) {
                            active_simulations_.fetch_sub(sims_to_run, std::memory_order_relaxed);

                            // Run the simulations
                            for (int i = 0; i < sims_to_run; ++i) {
                                runSimulation(root_.get());
                            }
                        }
                    } else if (active_simulations_.load() > 0 && root_) {
                        // If we timed out waiting but there's still work, run one simulation
                        runSimulation(root_.get());
                        active_simulations_.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
                
                #if MCTS_DEBUG
                std::cout << "Worker thread " << thread_id << " exiting" << std::endl;
                #endif
            });
        }
    }

    // For tests with num_threads=0, run simulations in the current thread
    if (settings_.num_threads == 0) {
        #if MCTS_DEBUG
        std::cout << "Running " << settings_.num_simulations << " simulations in main thread" << std::endl;
        #endif
        
        for (int i = 0; i < settings_.num_simulations; ++i) {
            runSimulation(root_.get());
        }
        active_simulations_ = 0;
    } else {
        // Launch simulations on worker threads in batches for better efficiency
        int remaining = settings_.num_simulations;
        const int batch_size = std::min(100, settings_.num_simulations / 10 + 1);
        
        #if MCTS_DEBUG
        std::cout << "Scheduling " << settings_.num_simulations 
                 << " simulations in batches of up to " << batch_size << std::endl;
        #endif

        while (remaining > 0) {
            int batch = std::min(batch_size, remaining);

            // Increment active simulation count for this batch
            active_simulations_.fetch_add(batch, std::memory_order_release);

            // Signal workers that new work is available
            cv_.notify_all();

            // Decrement remaining simulations
            remaining -= batch;

            // If we still have more batches to schedule, yield a bit to avoid overloading
            if (remaining > 0) {
                std::this_thread::yield();
            }
        }

        // Wait for all simulations to complete with a more efficient loop
        int last_active = active_simulations_.load();
        auto wait_start = std::chrono::steady_clock::now();
        
        #if MCTS_DEBUG
        std::cout << "Waiting for " << last_active << " active simulations to complete" << std::endl;
        int progress_counter = 0;
        #endif

        while (active_simulations_.load() > 0) {
            // Check current active simulations
            int current_active = active_simulations_.load();

            #if MCTS_DEBUG
            // Periodically show progress
            if ((progress_counter++ % 20) == 0) {
                std::cout << "Waiting for " << current_active << " simulations to complete" << std::endl;
            }
            #endif

            // If no progress in a while, notify threads again
            if (current_active == last_active) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start);
                if (elapsed.count() > 100) {  // If stuck for 100ms
                    // Notify all threads in case some are sleeping
                    cv_.notify_all();
                    wait_start = now;
                    
                    #if MCTS_DEBUG
                    std::cout << "No progress in 100ms, notifying threads again" << std::endl;
                    #endif
                }
            } else {
                last_active = current_active;
                wait_start = std::chrono::steady_clock::now();
            }

            // Use an exponential backoff for waiting to reduce CPU usage in tight loops
            if (current_active > settings_.num_threads * 2) {
                // Many simulations left, just yield CPU time slice
                std::this_thread::yield();
            } else if (current_active > settings_.num_threads) {
                // Some simulations left, short sleep
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            } else {
                // Few simulations left, slightly longer sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    // Set search running flag to false
    search_running_ = false;
    
    // Count nodes and find max depth
    size_t count = 0;
    int max_depth = 0;
    
    std::function<void(MCTSNode*, int)> countNodes = [&count, &max_depth, &countNodes](MCTSNode* node, int depth) {
        if (!node) return;
        
        count++;
        max_depth = std::max(max_depth, depth);
        
        for (auto* child : node->getChildren()) {
            countNodes(child, depth + 1);
        }
    };
    
    countNodes(root_.get(), 0);
    
    last_stats_.total_nodes = count;
    last_stats_.max_depth = max_depth;
    
    #if MCTS_DEBUG
    std::cout << "Tree has " << count << " nodes with max depth " << max_depth << std::endl;
    #endif
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

        // Check transposition table for this node
        if (use_transposition_table_ && !node->isTerminal()) {
            uint64_t hash = node->getState().getHash();
            MCTSNode* transposition = transposition_table_->get(hash);

            if (transposition) {
                // Found in transposition table - replace current node in path
                path.push_back(transposition);
                node = transposition;
                continue;
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
    
    if (leaf->isTerminal()) {
        auto result = leaf->getState().getGameResult();
        float value = 0.0f;
        if (result == core::GameResult::WIN_PLAYER1) {
            value = leaf->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
        } else if (result == core::GameResult::WIN_PLAYER2) {
            value = leaf->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
        } else {
            value = 0.0f; // Draw
        }
        return value;
    }
    
    // If the node is not fully expanded, expand it
    leaf->expand();
    
    // Store in transposition table
    if (use_transposition_table_) {
        uint64_t hash = leaf->getState().getHash();
        transposition_table_->store(hash, leaf, path.size());
    }
    
    // If leaf has no children after expansion (e.g., all moves are illegal)
    if (leaf->getChildren().empty()) {
        return 0.0f;
    }
    
    // If this is a fresh leaf node, evaluate it with the neural network
    if (leaf->getVisitCount() == 0) {
        // Special fast path for serial mode (no worker threads)
        if (settings_.num_threads == 0) {
            auto state_clone = leaf->getState().clone();
            std::vector<std::unique_ptr<core::IGameState>> states;
            states.push_back(std::move(state_clone));
            auto outputs = evaluator_->getInferenceFunction()(states);
            if (!outputs.empty()) {
                leaf->setPriorProbabilities(outputs[0].policy);
                return outputs[0].value;
            } else {
                int action_space_size = leaf->getState().getActionSpaceSize();
                std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                leaf->setPriorProbabilities(uniform_policy);
                return 0.0f;
            }
        } else {
            // For parallel mode, use the async evaluator
            auto state_clone = leaf->getState().clone();
            auto future = evaluator_->evaluateState(leaf, std::move(state_clone));
            
            // Wait for the result with a timeout to avoid deadlocks
            auto status = future.wait_for(std::chrono::seconds(2));
            if (status == std::future_status::ready) {
                auto result = future.get();
                leaf->setPriorProbabilities(result.policy);
                return result.value;
            } else {
                // Timed out waiting for evaluation, use uniform prior
                #if MCTS_DEBUG
                std::cerr << "Warning: Timed out waiting for neural network evaluation" << std::endl;
                #endif
                
                int action_space_size = leaf->getState().getActionSpaceSize();
                std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
                leaf->setPriorProbabilities(uniform_policy);
                return 0.0f;
            }
        }
    }
    
    // If node has been visited before, select a random child
    auto& children = leaf->getChildren();
    
    // Create a shuffled index vector for random selection
    std::vector<size_t> indices(children.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), random_engine_);
    
    // Choose the first child after shuffling
    MCTSNode* child = children[indices[0]];
    
    // Store child in transposition table
    if (use_transposition_table_) {
        uint64_t hash = child->getState().getHash();
        transposition_table_->store(hash, child, path.size() + 1);
    }
    
    // Evaluate child with neural network
    if (settings_.num_threads == 0) {
        auto state_clone = child->getState().clone();
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.push_back(std::move(state_clone));
        auto outputs = evaluator_->getInferenceFunction()(states);
        if (!outputs.empty()) {
            child->setPriorProbabilities(outputs[0].policy);
            // Negate value because it's from the child's perspective
            return -outputs[0].value;
        } else {
            int action_space_size = child->getState().getActionSpaceSize();
            std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
            child->setPriorProbabilities(uniform_policy);
            return 0.0f;
        }
    } else {
        auto state_clone = child->getState().clone();
        auto future = evaluator_->evaluateState(child, std::move(state_clone));
        
        // Wait for the result with a timeout
        auto status = future.wait_for(std::chrono::seconds(2));
        if (status == std::future_status::ready) {
            auto result = future.get();
            child->setPriorProbabilities(result.policy);
            // Negate value because it's from the child's perspective
            return -result.value;
        } else {
            // Timed out, use uniform policy
            #if MCTS_DEBUG
            std::cerr << "Warning: Timed out waiting for neural network evaluation of child" << std::endl;
            #endif
            
            int action_space_size = child->getState().getActionSpaceSize();
            std::vector<float> uniform_policy(action_space_size, 1.0f / action_space_size);
            child->setPriorProbabilities(uniform_policy);
            return 0.0f;
        }
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
    // Debug output - print most likely moves
    std::cout << "Action probabilities (top 5):" << std::endl;
    std::vector<std::pair<int, float>> action_probs;
    for (size_t i = 0; i < action_probabilities.size(); ++i) {
        if (action_probabilities[i] > 0.01f) {
            action_probs.emplace_back(i, action_probabilities[i]);
        }
    }
    
    // Sort by probability (descending)
    std::sort(action_probs.begin(), action_probs.end(), 
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print top 5 (or fewer if there aren't that many)
    for (size_t i = 0; i < std::min(action_probs.size(), size_t(5)); ++i) {
        std::cout << "  Action " << action_probs[i].first << ": " 
                 << std::fixed << std::setprecision(4) << action_probs[i].second
                 << " (visits: " << (root->getChildren().size() > i ? 
                    root->getChildren()[i]->getVisitCount() : 0) << ")" << std::endl;
    }
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
            std::cerr << "Error getting prior probabilities for root: " << e.what() << std::endl;
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