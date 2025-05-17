#ifndef ALPHAZERO_MCTS_TASKFLOW_ENGINE_H
#define ALPHAZERO_MCTS_TASKFLOW_ENGINE_H

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>
#include <functional>
#include <chrono>

#include "mcts/mcts_node.h"
#include "mcts/mcts_node_pool.h"
#include "mcts/mcts_evaluator.h"
#include "mcts/transposition_table.h"
#include "mcts/node_tracker.h"
#include "mcts/mcts_engine.h"  // For MCTSSettings
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "nn/neural_network.h"
#include "utils/gamestate_pool.h"

namespace alphazero {
namespace mcts {

// Forward declarations
class MCTSEngine;

/**
 * MCTS Engine using Cpp-Taskflow for improved task scheduling and CPU-GPU orchestration
 * 
 * Benefits over manual thread management:
 * - Work-stealing scheduler for better load balancing
 * - Efficient task graph construction
 * - Better CPU-GPU synchronization
 * - Dynamic task creation for irregular tree structures
 */
class ALPHAZERO_API MCTSTaskflowEngine {
private:
    // Taskflow executor with work-stealing scheduler
    tf::Executor executor_;
    tf::Taskflow taskflow_;
    
    // Core MCTS components (same as original)
    std::shared_ptr<MCTSNode> root_;
    std::unique_ptr<MCTSEvaluator> evaluator_;
    std::unique_ptr<TranspositionTable> tt_;
    std::unique_ptr<NodeTracker> node_tracker_;
    std::unique_ptr<MCTSNodePool> node_pool_;
    MCTSSettings settings_;
    
    // Synchronization
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> search_running_{false};
    std::atomic<int> active_simulations_{0};
    std::atomic<int> pending_evaluations_{0};
    std::mutex shutdown_mutex_;
    std::condition_variable shutdown_cv_;
    std::condition_variable search_complete_cv_;
    
    // Queues for batch evaluation
    std::unique_ptr<moodycamel::ConcurrentQueue<MCTSEngine::PendingEvaluation>> leaf_queue_;
    std::unique_ptr<moodycamel::ConcurrentQueue<mcts::NetworkOutput>> result_queue_;
    
    // Random number generation
    thread_local static std::mt19937 thread_local_gen_;
    
    // Private methods
    void treeTraversalTask();
    void processResultsTask();
    void evaluatorTask();
    tf::Task createSimulationTask();
    tf::Task createBatchEvaluationTask();
    
public:
    MCTSTaskflowEngine(MCTSSettings settings, 
                       std::unique_ptr<nn::NeuralNetwork> nn_model);
    ~MCTSTaskflowEngine();
    
    // Main search interface
    std::shared_ptr<MCTSNode> runSearch(std::unique_ptr<core::IGameState> root_state,
                                       int num_simulations);
    
    // Node selection and expansion (same logic, but called from tasks)
    std::shared_ptr<MCTSNode> selectBestChild(std::shared_ptr<MCTSNode> parent);
    void expandNode(std::shared_ptr<MCTSNode> node);
    
    // Getters
    std::shared_ptr<MCTSNode> getRoot() const { return root_; }
    const MCTSSettings& getSettings() const { return settings_; }
    
    // Tree management
    void reset();
    void shutdown();
    
    // Task graph construction
    void buildSearchTaskGraph();
    void submitSimulationBatch(int batch_size);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_TASKFLOW_ENGINE_H