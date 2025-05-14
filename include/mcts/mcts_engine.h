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
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "nn/neural_network.h"

namespace alphazero {
namespace mcts {

struct ALPHAZERO_API MCTSSettings {
    // Number of simulations to run
    int num_simulations = 800;
    
    // Number of worker threads
    int num_threads = 4;
    
    // Neural network batch size
    int batch_size = 8;
    
    // Neural network batch timeout
    std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(5);
    
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

private:
    // Internal search method
    void runSearch(const core::IGameState& state);
    
    // Run a single simulation
    void runSimulation(MCTSNode* root);
    
    // Select leaf node for expansion
    std::pair<MCTSNode*, std::vector<MCTSNode*>> selectLeafNode(MCTSNode* root);
    
    // Expand and evaluate a leaf node
    float expandAndEvaluate(MCTSNode* leaf, const std::vector<MCTSNode*>& path);
    
    // Back up value through the tree
    void backPropagate(std::vector<MCTSNode*>& path, float value);
    
    // Convert tree to action probabilities
    std::vector<float> getActionProbabilities(MCTSNode* root, float temperature);
    
    // Add Dirichlet noise to root node policy
    void addDirichletNoise(MCTSNode* root);
    
    // Settings
    MCTSSettings settings_;
    
    // Statistics from last search
    MCTSStats last_stats_;
    
    // Neural network evaluator
    std::unique_ptr<MCTSEvaluator> evaluator_;
    
    // Tree root
    std::unique_ptr<MCTSNode> root_;
    
    // Thread pool
    std::vector<std::thread> worker_threads_;
    
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

    // Safely start the evaluator if it hasn't been started yet
    bool ensureEvaluatorStarted();

    // Safely stop the evaluator if it was started
    void safelyStopEvaluator();

    // Helper methods for thread management
    void createWorkerThreads();
    void processPendingSimulations();
    void distributeSimulations();
    void waitForSimulationsToComplete(std::chrono::steady_clock::time_point start_time);
    void countTreeStatistics();    
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_ENGINE_H