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
#include "mcts/mcts_object_pool.h"
#include "mcts/mcts_node_pool.h"
#include "mcts/transposition_table.h"
#include "mcts/phmap_transposition_table.h"
#include "mcts/node_tracker.h"
#include "mcts/memory_pressure_monitor.h"
#include "mcts/gpu_memory_pool.h"
// Removed complex components for simplified implementation
// #include "mcts/dynamic_batch_manager.h"
// #include "mcts/unified_inference_server.h"
// #include "mcts/burst_coordinator.h"
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "nn/neural_network.h"
#include "utils/gamestate_pool.h"

namespace alphazero {
namespace mcts {

// Removed forward declarations for unused components

// Standardized batch parameters to ensure consistent batch handling across components
struct ALPHAZERO_API BatchParameters {
    // The optimal target batch size for GPU efficiency
    size_t optimal_batch_size = 256;
    
    // The minimum acceptable batch size (~75% of optimal)
    size_t minimum_viable_batch_size = 192;
    
    // The absolute minimum batch size to process after timeout (~30% of optimal)
    size_t minimum_fallback_batch_size = 64;
    
    // Limits the number of items per collection to maintain responsiveness
    size_t max_collection_batch_size = 32;
    
    // Maximum time to wait for optimal batch formation
    std::chrono::milliseconds max_wait_time = std::chrono::milliseconds(50);
    
    // Time to wait after reaching minimum viable batch before processing
    std::chrono::milliseconds additional_wait_time = std::chrono::milliseconds(10);
};

struct ALPHAZERO_API MCTSSettings {
    // Number of simulations to run
    int num_simulations = 800;
    
    // Number of worker threads - should match physical cores for best performance
    int num_threads = 12;  // Optimized for Ryzen 5900X (12 cores)
    
    // Standardized batch parameters for consistent batch handling across components
    BatchParameters batch_params;
    
    // These parameters are kept for backward compatibility
    // They will be used to initialize the batch_params if not explicitly set
    
    // Neural network batch size - optimized for GPU efficiency
    int batch_size = 256;  // Larger batch for better GPU utilization
    
    // Maximum batch size per collection - controls responsiveness vs efficiency
    int max_collection_batch_size = 32;  // Default to moderate batch size for balance
    
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
    float progressive_widening_c = 2.0f;  // C parameter for progressive widening
    float progressive_widening_k = 0.5f; // K parameter for progressive widening (typically 0.25-0.5)
    
    // Root parallelization settings  
    bool use_root_parallelization = true;
    int num_root_workers = 4;  // Number of parallel MCTS trees to run
    // Threads per root worker will be num_threads / num_root_workers
    
    // RAVE (Rapid Action Value Estimation) settings
    bool use_rave = true;
    float rave_constant = 3000.0f;  // Constant for RAVE weight calculation
    
    // Tensor MCTS configuration (loaded from config file)
    int tensor_batch_size = 64;
    int tensor_max_nodes = 2048;
    int tensor_max_actions = 512;
    int tensor_max_depth = 64;
    bool use_cuda_graphs = false;
    bool use_persistent_kernels = true;
    
    // Constructor to initialize batch parameters from legacy settings
    MCTSSettings() {
        syncBatchParametersFromLegacy();
    }
    
    // Method to synchronize batch parameters from legacy settings
    void syncBatchParametersFromLegacy() {
        // Only update if not already set
        if (batch_params.optimal_batch_size == 0) {
            batch_params.optimal_batch_size = batch_size;
        }
        if (batch_params.minimum_viable_batch_size == 0) {
            batch_params.minimum_viable_batch_size = std::max(static_cast<size_t>(batch_size * 0.5), static_cast<size_t>(512));
        }
        if (batch_params.minimum_fallback_batch_size == 0) {
            batch_params.minimum_fallback_batch_size = std::max(static_cast<size_t>(batch_size * 0.25), static_cast<size_t>(256));
        }
        if (batch_params.max_collection_batch_size == 0) {
            batch_params.max_collection_batch_size = max_collection_batch_size;
        }
        batch_params.max_wait_time = batch_timeout;
        batch_params.additional_wait_time = std::chrono::milliseconds(20);
    }
    
    // Method to update legacy settings from batch parameters for backward compatibility
    void syncLegacyFromBatchParameters() {
        batch_size = batch_params.optimal_batch_size;
        max_collection_batch_size = batch_params.max_collection_batch_size;
        batch_timeout = batch_params.max_wait_time;
    }
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
    
    // Enhanced statistics for optimized architecture
    size_t total_requests_processed = 0;
    size_t total_batches_processed = 0;
    float burst_efficiency = 0.0f;
    float burst_utilization = 0.0f;
    float parallel_efficiency = 0.0f;
    float task_scheduling_overhead = 0.0f;
    int tree_depth = 0;
    float tree_branching_factor = 0.0f;
    float exploration_efficiency = 0.0f;
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

// Using PendingEvaluation from evaluation_types.h
// Note: adapters for compatibility between the shared_ptr in PendingEvaluation and the unique_ptr in evaluation methods
inline std::shared_ptr<core::IGameState> convertToSharedPtr(const core::IGameState& state) {
    return std::shared_ptr<core::IGameState>(state.clone());
}

inline std::unique_ptr<core::IGameState> convertToUniquePtr(const std::shared_ptr<core::IGameState>& state) {
    return state ? state->clone() : nullptr;
}

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
    
    virtual ~MCTSEngine();
    
    // Search from given state
    virtual SearchResult search(const core::IGameState& state);
    
    // PERFORMANCE FIX: Search with tree reuse for sequential moves
    SearchResult searchWithTreeReuse(const core::IGameState& state, int last_action = -1);
    
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
     * @brief Enable or disable PHMap transposition table implementation
     * 
     * @param use_phmap Whether to use PHMap implementation (true) or standard (false)
     */
    void setUsePHMapTransposition(bool use_phmap);

    
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
    
        
    
    

public:
    
    // Batch tracking
    struct BatchInfo {
        std::vector<PendingEvaluation> evaluations;
        std::chrono::steady_clock::time_point created_time;
        bool submitted;
    };
    
private:
    // Core MCTS algorithm methods
    void runSearch(const core::IGameState& state);
    void initializeGameStatePool(const core::IGameState& state);
    void resetSearchState();
    void executeSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    void executeSimpleSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    void executeSimplifiedParallelSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    void runSimulation(std::shared_ptr<MCTSNode> root);
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> selectLeafNode(std::shared_ptr<MCTSNode> root);
    float expandAndEvaluate(std::shared_ptr<MCTSNode> leaf, const std::vector<std::shared_ptr<MCTSNode>>& path);
    void backPropagate(std::vector<std::shared_ptr<MCTSNode>>& path, float value);
    
    // Search tree management
    std::shared_ptr<MCTSNode> createRootNode(const core::IGameState& state);
    std::vector<std::shared_ptr<MCTSNode>> createSearchRoots(std::shared_ptr<MCTSNode> main_root, int num_roots);
    void setupBatchParameters();
    void processEvaluationResults();
    void aggregateRootParallelResults(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    void processPendingEvaluations(std::shared_ptr<MCTSNode> root);
    std::vector<float> getActionProbabilities(std::shared_ptr<MCTSNode> root, float temperature);
    
    // CRITICAL FIX: Enhanced search with virtual loss and lock-free batching
    void executeEnhancedSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    
    // Helper methods for enhanced search
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
    selectLeafNodeWithVirtualLoss(std::shared_ptr<MCTSNode> root, std::mt19937& rng);
    
    void submitBatchForEvaluation(std::vector<PendingEvaluation>&& batch);
    
    void waitForEvaluationCompletion(std::chrono::steady_clock::time_point start_time);
    
    // OPTIMIZATION 1: OpenMP parallelized search with multi-threaded simulation generation
    void executeParallelSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    void executeTrueLeafParallelSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    
    // Helper methods for parallel search
    void processParallelBatch(std::vector<PendingEvaluation>& batch, int thread_id);
    
    void waitForParallelEvaluationCompletion(const std::vector<PendingEvaluation>& evaluation_requests);
    
    // ParallelSearchResult for parallel search
    struct ParallelSearchResult {
        int simulations_completed = 0;
        int batches_evaluated = 0;
        double avg_batch_size = 0.0;
        double avg_batch_time_ms = 0.0;
        int cache_hits = 0;
        
        // Additional fields used by parallel_mcts_search.cpp
        std::chrono::microseconds elapsed_time{0};
        int terminal_nodes_processed = 0;
        int virtual_loss_applications = 0;
        std::vector<std::shared_ptr<MCTSNode>> expanded_nodes;
        std::vector<PendingEvaluation> evaluation_requests;
    };
    
    void updateParallelSearchStats(const ParallelSearchResult& search_result);
    
    // Method to execute fully optimized search with all enhancements
    SearchResult executeOptimizedSearch(std::shared_ptr<core::IGameState> root_state);
    
    // Enhanced burst search methods for optimized implementation
    size_t executeEnhancedBurstSearch(const core::IGameState& root_state, 
                                     std::shared_ptr<MCTSNode> root_node, int batch_target);
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
    selectOptimizedLeafNode(std::shared_ptr<MCTSNode> root);
    float calculateExplorationEfficiency(std::shared_ptr<MCTSNode> root);
    float calculateAverageBranchingFactor(std::shared_ptr<MCTSNode> root);
    
    // Real-time optimization worker method
    void realTimeOptimizationWorker();
    
    // NEW: Burst-mode optimized search with unified memory management
    void executeOptimizedSearchV2(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    
    
    // CRITICAL FIX: Simplified direct batching without complex layers
    void executeSimpleBatchedSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    
    
    
    // BATCH TREE SELECTION: Process multiple paths simultaneously
    struct BatchItem {
        MCTSNode* current_node;
        std::unique_ptr<core::IGameState> state;
        std::vector<MCTSNode*> path;
        int depth;
        bool is_leaf;
    };
    
    void batchTraverseToLeaves(
        const std::vector<MCTSNode*>& roots,
        const std::vector<std::unique_ptr<core::IGameState>>& initial_states,
        std::vector<BatchItem>& leaf_items,
        int batch_size);
    
    void executeBatchedTreeSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state);
    
    // ULTRA-FAST BATCH: Pipelined batch collection with prefetching for <100ms moves
    void executeUltraFastBatchSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state);
    
    // Performance mode for optimization configuration
    enum class PerformanceMode {
        MaximumAccuracy,    // Prioritize quality of search over speed
        BalancedPerformance, // Balance between quality and speed
        MaximumSpeed,       // Prioritize speed over quality
        EnergyEfficiency    // Minimize resource usage
    };
    
    // Configure performance mode
    void setPerformanceMode(PerformanceMode mode);
    
    // Enable/disable advanced optimizations
    void enableAdvancedOptimizations(bool enable = true);
    bool isAdvancedOptimizationsEnabled() const;
    
    // Disable all advanced optimizations
    void disableAdvancedOptimizations();
    
    // Real-time optimization control
    void enableRealTimeOptimization(bool enable);
    
    // Configure component optimizations
    void configureAdaptiveBatching(bool enable);
    void enableBurstPipelining(bool enable, int pipeline_depth = 2);
    void enableMemoryOptimizations(bool enable);
    void configureConcurrencyOptimizations(int max_threads);
    
    // Specific optimizations
    void optimizeInferenceServerConfiguration();
    // Removed unused optimization methods
    void enableTranspositionTableOptimizations();
    
    // Specific performance profiles
    void configureForMaximumAccuracy();
    void configureForBalancedPerformance();
    void configureForMaximumSpeed();
    void configureForEnergyEfficiency();
    
    // Metrics for optimization tuning
    struct OptimizationMetrics {
        float batch_efficiency;
        float virtual_loss_impact;
        float neural_network_utilization;
        float memory_utilization;
        float thread_utilization;
        std::chrono::milliseconds average_evaluation_latency;
        std::chrono::microseconds average_selection_time;
        std::chrono::microseconds average_expansion_time;
        
        // Additional optimization metrics for new architecture
        float average_batch_size = 0.0f;
        float batch_utilization = 0.0f;
        float burst_efficiency = 0.0f;
        float coordination_overhead = 0.0f;
        float memory_efficiency = 0.0f;
        float pool_utilization = 0.0f;
        float inference_throughput = 0.0f;
        float overall_optimization_score = 0.0f;
    };
    
    // Get optimization metrics
    OptimizationMetrics getOptimizationMetrics() const;
    
    // Forward declarations for parallel search components
    
    struct ParallelLeafResult {
        std::shared_ptr<MCTSNode> leaf_node;
        std::vector<std::shared_ptr<MCTSNode>> path;
        bool needs_evaluation = false;
        bool applied_virtual_loss = false;
        bool terminal = false;
        float terminal_value = 0.0f;
    };
    
private:
    // Parallel search helper methods
    ParallelSearchResult executeParallelSimulations(const std::vector<std::shared_ptr<MCTSNode>>& search_roots, int target_simulations);
    ParallelLeafResult selectLeafNodeParallel(std::shared_ptr<MCTSNode> root, std::vector<std::shared_ptr<MCTSNode>>& path, std::mt19937& rng);
    void backpropagateParallel(const std::vector<std::shared_ptr<MCTSNode>>& path, float value, int virtual_loss_amount);
    
public:
    
    // Tree traversal helpers
    std::shared_ptr<MCTSNode> traverseTreeForLeaf(std::shared_ptr<MCTSNode> node, std::vector<std::shared_ptr<MCTSNode>>& path);
    bool handleTranspositionMatch(std::shared_ptr<MCTSNode>& node, std::shared_ptr<MCTSNode>& parent);
    bool expandNonTerminalLeaf(std::shared_ptr<MCTSNode>& leaf);
    
    // Utility functions
    bool safeGameStateValidation(const core::IGameState& state);
    std::vector<float> createDefaultPolicy(int action_space_size);
    bool safelyMarkNodeForEvaluation(std::shared_ptr<MCTSNode> node);
    void addDirichletNoise(std::shared_ptr<MCTSNode> root);
    
    // New architecture management methods
    bool ensureEvaluatorStarted();
    void safelyStopEvaluator();
    void setSharedExternalQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* leaf_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* result_queue,
        std::function<void()> notify_fn);
    
    // Memory management methods
    void cleanupTree();
    void cleanupPendingEvaluations();
    void resetForNewSearch();
    void performAggressiveGPUCleanup();
    void monitorAndCleanupMemory();
    
    // CPU optimization methods
    void performParallelExpansion(std::shared_ptr<MCTSNode> root, 
                                 int num_expansions);
    void performCPUOptimizedSearch(const core::IGameState& root_state,
                                  int num_simulations);
    void optimizeBatchProcessing();
    
    // Parallelization strategies
    void runOpenMPSearch();
    void runRootParallelSearch();
    void combineRootResults(const std::vector<std::shared_ptr<MCTSNode>>& root_nodes);
    void executeSingleSimulation(std::shared_ptr<MCTSNode> root, std::vector<PendingEvaluation>& thread_local_batch);
    
    // Worker thread methods
    void resultDistributorWorker();
    void traverseTree(std::shared_ptr<MCTSNode> root);
    
    // Settings
    MCTSSettings settings_;
    
    // Statistics from last search
    MCTSStats last_stats_;
    
    
    
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

    // High-performance transposition table using parallel-hashmap (always used)
    std::unique_ptr<PHMapTranspositionTable> transposition_table_;
    
    // Whether to use the transposition table
    bool use_transposition_table_;


    // Helper methods for thread management
    void createWorkerThreads();
    void processPendingSimulations();
    void distributeSimulations();
    void waitForSimulationsToComplete(std::chrono::steady_clock::time_point start_time);
    void countTreeStatistics();
    
    // Helper for adaptive waiting with exponential backoff
    // Waits until the predicate returns true, using increasingly longer
    // wait times to reduce CPU usage when waiting
    void waitWithBackoff(std::function<bool()> predicate, std::chrono::milliseconds max_wait_time);

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
    std::function<void()> external_queue_notify_fn_; 
    
    // Memory pool removed - using simpler memory management
    // std::unique_ptr<AdvancedMemoryPool> memory_pool_;
    std::atomic<bool> use_advanced_memory_pool_{false};
    
    // Flag to enable memory pool for GameState cloning
    bool game_state_pool_enabled_;
    
    // Node tracker for lock-free pending evaluation management
    std::unique_ptr<NodeTracker> node_tracker_;
    
    // Node pool for efficient memory management
    std::unique_ptr<MCTSNodePool> node_pool_;
    
    
    // New optimization control variables
    std::atomic<bool> use_advanced_optimizations_{false};
    std::atomic<bool> adaptive_batching_enabled_{false};
    std::atomic<bool> burst_pipelining_enabled_{false};
    std::atomic<bool> memory_optimizations_enabled_{false};
    std::atomic<bool> real_time_optimization_enabled_{false};
    std::atomic<int> performance_mode_{0}; // 0=MaximumAccuracy, 1=BalancedPerformance, 2=MaximumSpeed, 3=EnergyEfficiency
    std::atomic<int> max_concurrent_threads_{static_cast<int>(std::thread::hardware_concurrency())};
    
    // Neural network reference for optimization files
    std::shared_ptr<nn::NeuralNetwork> neural_network_;
    
    // Real-time optimization worker thread
    std::thread optimization_thread_;
    
    // Memory cleanup callback
    void handleMemoryPressure(MemoryPressureMonitor::PressureLevel level);
    
    // Direct inference function for serial mode (bypasses complex infrastructure)
    InferenceFunction direct_inference_fn_;
    
    // Memory pressure monitoring
    std::unique_ptr<MemoryPressureMonitor> memory_pressure_monitor_;
    
    // GPU memory pool for efficient tensor management
    std::unique_ptr<GPUMemoryPool> gpu_memory_pool_;
    
    // Dynamic batch manager removed
    // std::unique_ptr<DynamicBatchManager> dynamic_batch_manager_;
    
    // Advanced components for optimized parallelization (commented out for simplification)
    // std::unique_ptr<UnifiedInferenceServer> inference_server_;
    // std::unique_ptr<BurstCoordinator> burst_coordinator_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_ENGINE_H