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
    
    // Number of worker threads - should match physical cores for best performance
    int num_threads = 12;  // Optimized for Ryzen 5900X (12 cores)
    
    // Neural network batch size - optimized for GPU efficiency
    int batch_size = 256;  // Larger batch for better GPU utilization
    
    // Neural network batch timeout - reduced for better responsiveness
    std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(5);  // Shorter timeout for leaf parallelization
    
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
    
    // Progressive widening settings
    bool use_progressive_widening = true;
    float progressive_widening_c = 1.0f;  // C parameter for progressive widening
    float progressive_widening_k = 10.0f; // K parameter for progressive widening
    
    // Root parallelization settings  
    bool use_root_parallelization = true;
    int num_root_workers = 4;  // Number of parallel MCTS trees to run
    // Threads per root worker will be num_threads / num_root_workers
    
    // RAVE (Rapid Action Value Estimation) settings
    bool use_rave = true;
    float rave_constant = 3000.0f;  // Constant for RAVE weight calculation
    
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

// Pending evaluation tracking
struct PendingEvaluation {
    std::shared_ptr<MCTSNode> node;
    std::vector<std::shared_ptr<MCTSNode>> path;
    std::shared_ptr<core::IGameState> state;  // Changed from unique_ptr to shared_ptr
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

class ALPHAZERO_API MCTSEngine {
public:
    // Static mutex for global evaluator initialization coordination
    static std::mutex s_global_evaluator_mutex;
    static std::atomic<int> s_evaluator_init_counter;
    
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
    
    /**
     * @brief Monitor memory usage and perform cleanup if needed
     * 
     * Tracks memory usage and triggers cleanup if memory consumption
     * exceeds configured thresholds. Called periodically during search.
     */
    void monitorMemoryUsage();
    
    /**
     * @brief Force aggressive memory cleanup
     * 
     * Performs immediate memory cleanup including clearing caches,
     * trimming pools, and forcing garbage collection. Use when
     * memory pressure is detected.
     */
    void forceCleanup();
    
    /**
     * @brief Set shared external queues to use instead of internal queues
     * 
     * @param leaf_queue Shared queue for pending evaluations
     * @param result_queue Shared queue for evaluation results
     * @param notify_fn Notification function when results are ready
     */
    void setSharedExternalQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* leaf_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* result_queue,
        std::function<void()> notify_fn);
        
    /**
     * @brief Get the evaluator for direct access
     * 
     * @return Pointer to the evaluator (may be nullptr)
     */
    MCTSEvaluator* getEvaluator() const { return evaluator_.get(); }
    

public:
    
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
    
    // OpenMP-based parallel search
    void runOpenMPSearch();
    
    // Root parallel search - run multiple MCTS trees in parallel
    void runRootParallelSearch();
    
    // Combine results from multiple root nodes
    void combineRootResults(const std::vector<std::shared_ptr<MCTSNode>>& root_nodes);
    
    // Single simulation execution 
    void executeSingleSimulation(std::shared_ptr<MCTSNode> root,
                                std::vector<PendingEvaluation>& thread_local_batch);
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
    
    // Removed cv_mutex_ and cv_ - using lock-free polling instead
    
    // Random generator for stochastic actions
    std::mt19937 random_engine_;

    // Transposition table
    std::unique_ptr<TranspositionTable> transposition_table_;
    
    // Whether to use the transposition table
    bool use_transposition_table_;

    // Whether the evaluator thread has been started
    std::atomic<bool> evaluator_started_{false};
    
    // Mutex for evaluator initialization
    std::mutex evaluator_mutex_;

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
    std::shared_ptr<core::IGameState> cloneGameState(const core::IGameState& state);
    

    // Producer-consumer queues and tracking
    std::atomic<int> pending_evaluations_{0};
    std::atomic<int> batch_counter_{0};
    std::atomic<int> total_leaves_generated_{0};
    std::atomic<int> total_results_processed_{0};
    
    // Lock-free batch collection
    std::atomic<int> batch_submission_counter_{0};
    static constexpr int BATCH_SUBMISSION_INTERVAL = 64;
    
    // Producer-consumer queues
    moodycamel::ConcurrentQueue<PendingEvaluation> leaf_queue_;
    moodycamel::ConcurrentQueue<BatchInfo> batch_queue_;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>> result_queue_;
    
    // Optional external shared queues  
    moodycamel::ConcurrentQueue<PendingEvaluation>* shared_leaf_queue_ = nullptr;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* shared_result_queue_ = nullptr;
    bool use_shared_queues_ = false;
    
    // Per-thread data to avoid contention
    static constexpr int MAX_THREADS = 64;
    struct ThreadData {
        std::vector<PendingEvaluation> local_batch;
        int pending_count = 0;
    };
    ThreadData thread_data_[MAX_THREADS];
    
    // Removed mutexes and condition variables - using lock-free polling instead
    
    // Result distribution thread (OpenMP handles tree traversal)
    std::thread result_distributor_worker_;
    // Commented out - replaced with OpenMP thread pool
    // std::vector<std::thread> tree_traversal_workers_;
    
    // Control flag for specialized workers
    std::atomic<bool> workers_active_{false};
    
    // Removed mutex destruction flags - no longer needed with lock-free approach
    
    // Flag to enable memory pool for GameState cloning
    bool game_state_pool_enabled_;
    
    // Node tracker for lock-free pending evaluation management
    std::unique_ptr<NodeTracker> node_tracker_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_ENGINE_H