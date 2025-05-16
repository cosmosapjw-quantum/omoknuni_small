// include/mcts/mcts_engine.h
#ifndef ALPHAZERO_MCTS_ENGINE_H
#define ALPHAZERO_MCTS_ENGINE_H

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <random>
#include <functional>
#include <chrono>
#include "mcts/mcts_node.h"
#include "mcts/mcts_evaluator.h"
#include "mcts/transposition_table.h"
#include "mcts/node_tracker.h"
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "nn/neural_network.h"
#include "utils/gamestate_pool.h"

namespace alphazero {
namespace mcts {

struct ALPHAZERO_API MCTSSettings {
    // Number of simulations to run
    int num_simulations = 800;
    
    // Number of worker threads
    int num_threads = 4;
    
    // Neural network batch size - optimized for GPU efficiency
    int batch_size = 128;  // Increased for better GPU utilization
    
    // Neural network batch timeout - tuned for balance between latency and throughput
    std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(20);  // Reduced for faster response
    
    // Maximum concurrent simulations to prevent memory explosion
    int max_concurrent_simulations = 512;
    
    // Exploration constant for PUCT formula
    float exploration_constant = 1.4f;
    
    // Virtual loss count to apply during selection
    int virtual_loss = 3;
    
    // Add Dirichlet noise to root node policy for exploration
    bool add_dirichlet_noise = true;
    
    // Dirichlet noise parameters
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    
    // Temperature for move selection
    float temperature = 1.0f;
    
    // Transposition table settings
    bool use_transposition_table = true;
    size_t transposition_table_size_mb = 128; // Default 128MB
};

struct ALPHAZERO_API MCTSStats {
    // Total nodes created
    size_t total_nodes = 0;
    
    // Maximum depth reached
    int max_depth = 0;
    
    // Total time spent in search
    std::chrono::milliseconds search_time{0};
    
    // Average batch size
    float avg_batch_size = 0.0f;
    
    // Average batch latency
    std::chrono::milliseconds avg_batch_latency{0};
    
    // Total evaluations performed
    size_t total_evaluations = 0;
    
    // Nodes per second
    float nodes_per_second = 0.0f;

    // Transposition table hit rate
    float tt_hit_rate = 0.0f;
    
    // Transposition table size
    size_t tt_size = 0;
    
    // Memory pool statistics
    float pool_hit_rate = 0.0f;
    size_t pool_size = 0;
    size_t pool_total_allocated = 0;
};

struct ALPHAZERO_API SearchResult {
    // Selected action
    int action = -1;
    
    // Probability distribution over actions
    std::vector<float> probabilities;
    
    // Value estimate
    float value = 0.0f;
    
    // Search statistics
    MCTSStats stats;
};

class ALPHAZERO_API MCTSEngine {
public:
    // Signature for neural network inference function
    using InferenceFunction = std::function<std::vector<NetworkOutput>(
        const std::vector<std::unique_ptr<core::IGameState>>&)>;
    
    /**
     * @brief Constructor with C++ neural network
     * 
     * @param neural_net Shared pointer to a neural network
     * @param settings MCTS settings
     */
    MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings = MCTSSettings());
    
    /**
     * @brief Constructor with custom inference function
     * 
     * @param inference_func Function to perform neural network inference
     * @param settings MCTS settings
     */
    MCTSEngine(InferenceFunction inference_func, const MCTSSettings& settings = MCTSSettings());
    
    /**
     * @brief Move constructor
     */
    MCTSEngine(MCTSEngine&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    MCTSEngine& operator=(MCTSEngine&& other) noexcept;
    
    // Delete copy constructor and copy assignment operator
    MCTSEngine(const MCTSEngine&) = delete;
    MCTSEngine& operator=(const MCTSEngine&) = delete;
    
    ~MCTSEngine();
    
    // Search from given state
    SearchResult search(const core::IGameState& state);
    
    // Get the current settings
    const MCTSSettings& getSettings() const;
    
    // Update settings
    void updateSettings(const MCTSSettings& settings);
    
    // Get the last search statistics
    const MCTSStats& getLastStats() const;

    /**
     * @brief Enable or disable the transposition table
     * 
     * @param use Whether to use the transposition table
     */
    void setUseTranspositionTable(bool use);
    
    /**
     * @brief Check if the transposition table is enabled
     * 
     * @return true if enabled, false otherwise
     */
    bool isUsingTranspositionTable() const;
    
    /**
     * @brief Set the size of the transposition table
     * 
     * @param size_mb Size in megabytes
     */
    void setTranspositionTableSize(size_t size_mb);
    
    /**
     * @brief Clear the transposition table
     */
    void clearTranspositionTable();
    
    /**
     * @brief Get the hit rate of the transposition table
     * 
     * @return Hit rate (0.0 to 1.0)
     */
    float getTranspositionTableHitRate() const;

    /**
     * @brief Start the neural network evaluator if it hasn't been started yet
     * 
     * @return true if the evaluator was started successfully or already running
     * @return false if the evaluator failed to start
     */
    bool ensureEvaluatorStarted();

public:
    // Pending evaluation tracking
    struct PendingEvaluation {
        std::shared_ptr<MCTSNode> node;
        std::vector<std::shared_ptr<MCTSNode>> path;
        std::unique_ptr<core::IGameState> state;
        int batch_id;
        int request_id;
        
        // Default constructor
        PendingEvaluation() = default;
        
        // Move constructor
        PendingEvaluation(PendingEvaluation&& other) noexcept
            : node(std::move(other.node)),
              path(std::move(other.path)),
              state(std::move(other.state)),
              batch_id(other.batch_id),
              request_id(other.request_id) {
        }
        
        // Move assignment
        PendingEvaluation& operator=(PendingEvaluation&& other) noexcept {
            if (this != &other) {
                node = std::move(other.node);
                path = std::move(other.path);
                state = std::move(other.state);
                batch_id = other.batch_id;
                request_id = other.request_id;
            }
            return *this;
        }
        
        // Delete copy operations
        PendingEvaluation(const PendingEvaluation&) = delete;
        PendingEvaluation& operator=(const PendingEvaluation&) = delete;
    };
    
    // Batch tracking
    struct BatchInfo {
        std::vector<PendingEvaluation> evaluations;
        std::chrono::steady_clock::time_point created_time;
        bool submitted;
    };
    
private:
    // Internal search method
    void runSearch(const core::IGameState& state);
    
    // Run a single simulation
    void runSimulation(std::shared_ptr<MCTSNode> root);
    
    // Select leaf node for expansion
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> selectLeafNode(std::shared_ptr<MCTSNode> root);
    
    // Expand and evaluate a leaf node
    float expandAndEvaluate(std::shared_ptr<MCTSNode> leaf, const std::vector<std::shared_ptr<MCTSNode>>& path);
    
    // Back up value through the tree
    void backPropagate(std::vector<std::shared_ptr<MCTSNode>>& path, float value);
    
    // Process pending evaluations in the tree
    void processPendingEvaluations(std::shared_ptr<MCTSNode> root);
    
    // Convert tree to action probabilities
    std::vector<float> getActionProbabilities(std::shared_ptr<MCTSNode> root, float temperature);
    
    // Add Dirichlet noise to root node policy
    void addDirichletNoise(std::shared_ptr<MCTSNode> root);
    
    // New specialized worker methods
    void treeTraversalWorker(int worker_id);
    void batchAccumulatorWorker();
    void resultDistributorWorker();
    void traverseTree(std::shared_ptr<MCTSNode> root);
    
    // Settings
    MCTSSettings settings_;
    
    // Statistics from last search
    MCTSStats last_stats_;
    
    // Neural network evaluator
    std::unique_ptr<MCTSEvaluator> evaluator_;
    
    // Tree root
    std::shared_ptr<MCTSNode> root_;
    
    // Thread pool (removed in favor of specialized workers)
    
    // Control flags
    std::atomic<bool> shutdown_;
    std::atomic<int> active_simulations_;
    std::atomic<bool> search_running_;
    
    // Synchronization
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    
    // Random generator for stochastic actions
    std::mt19937 random_engine_;

    // Transposition table
    std::unique_ptr<TranspositionTable> transposition_table_;
    
    // Whether to use the transposition table
    bool use_transposition_table_;

    // Whether the evaluator thread has been started
    bool evaluator_started_;

    // Safely stop the evaluator if it was started
    void safelyStopEvaluator();

    // Helper methods for thread management
    void createWorkerThreads();
    void processPendingSimulations();
    void distributeSimulations();
    void waitForSimulationsToComplete(std::chrono::steady_clock::time_point start_time);
    void countTreeStatistics();

    // Methods for statistics calculation
    size_t countTreeNodes(std::shared_ptr<MCTSNode> node);
    int calculateMaxDepth(std::shared_ptr<MCTSNode> node);
    
    // Helper method to clone game state using memory pool
    std::unique_ptr<core::IGameState> cloneGameState(const core::IGameState& state);

    // Producer-consumer queues and tracking
    std::atomic<int> pending_evaluations_{0};
    std::atomic<int> batch_counter_{0};
    std::atomic<int> total_leaves_generated_{0};
    std::atomic<int> total_results_processed_{0};
    
    // Producer-consumer queues
    moodycamel::ConcurrentQueue<PendingEvaluation> leaf_queue_;
    moodycamel::ConcurrentQueue<BatchInfo> batch_queue_;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>> result_queue_;
    
    // Additional synchronization for new architecture
    std::mutex batch_mutex_;
    std::condition_variable batch_cv_;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    
    // Specialized worker threads
    std::thread batch_accumulator_worker_;
    std::thread result_distributor_worker_;
    std::vector<std::thread> tree_traversal_workers_;
    
    // Control flag for specialized workers
    std::atomic<bool> workers_active_{false};
    
    // Flags to track mutex destruction
    std::atomic<bool> cv_mutex_destroyed_{false};
    std::atomic<bool> batch_mutex_destroyed_{false};
    std::atomic<bool> result_mutex_destroyed_{false};
    
    // Flag to enable memory pool for GameState cloning
    bool game_state_pool_enabled_;
    
    // Node tracker for lock-free pending evaluation management
    std::unique_ptr<NodeTracker> node_tracker_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_ENGINE_H