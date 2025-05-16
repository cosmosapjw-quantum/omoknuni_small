// include/mcts/node_tracker.h
#ifndef ALPHAZERO_MCTS_NODE_TRACKER_H
#define ALPHAZERO_MCTS_NODE_TRACKER_H

#include <memory>
#include <atomic>
#include <optional>
#include <chrono>
#include "parallel_hashmap/phmap.h"
#include "third_party/concurrentqueue.h"
#include "mcts/evaluation_types.h"

namespace alphazero {
namespace mcts {

class MCTSNode;

/**
 * @brief Node tracking system using lock-free data structures
 * 
 * Aggressively uses moodycamel::ConcurrentQueue and parallel-hashmap
 * for high-performance concurrent access in MCTS leaf parallelization
 */
class NodeTracker {
public:
    using NodePtr = std::shared_ptr<MCTSNode>;
    using NetworkPromise = std::shared_ptr<std::promise<NetworkOutput>>;
    using NetworkFuture = std::shared_ptr<std::future<NetworkOutput>>;
    
    /**
     * @brief Pending evaluation info for a node
     */
    struct PendingEvaluation {
        NodePtr node;
        NetworkPromise promise;
        NetworkFuture future;
        std::vector<NodePtr> path;
        std::chrono::steady_clock::time_point submit_time;
    };
    
    /**
     * @brief Evaluation result for distribution
     */
    struct EvaluationResult {
        NodePtr node;
        NetworkOutput output;
        std::vector<NodePtr> path;
    };
    
    NodeTracker(size_t num_shards = 0);
    ~NodeTracker();
    
    /**
     * @brief Register a pending evaluation
     */
    bool registerPendingEvaluation(NodePtr node, 
                                  NetworkPromise promise,
                                  NetworkFuture future,
                                  const std::vector<NodePtr>& path);
    
    /**
     * @brief Check if a node has a pending evaluation
     */
    bool hasPendingEvaluation(const NodePtr& node) const;
    
    /**
     * @brief Get pending evaluation for a node
     */
    std::optional<PendingEvaluation> getPendingEvaluation(const NodePtr& node);
    
    /**
     * @brief Remove a pending evaluation
     */
    bool removePendingEvaluation(const NodePtr& node);
    
    /**
     * @brief Submit evaluation result
     */
    void submitResult(const NodePtr& node, const NetworkOutput& output, const std::vector<NodePtr>& path);
    
    /**
     * @brief Get next available result
     */
    bool getNextResult(EvaluationResult& result);
    
    /**
     * @brief Get number of pending evaluations
     */
    size_t getPendingCount() const { return pending_count_.load(std::memory_order_relaxed); }
    
    /**
     * @brief Clear all pending evaluations
     */
    void clear();
    
private:
    // Lock-free queue for results
    moodycamel::ConcurrentQueue<EvaluationResult> result_queue_;
    
    // Parallel hashmap for pending evaluations with node pointer as key
    using PendingMap = phmap::parallel_node_hash_map<void*, PendingEvaluation>;
    PendingMap pending_evaluations_;
    
    // Atomic counter for pending evaluations
    std::atomic<size_t> pending_count_{0};
    
    // Number of shards for the hashmap
    size_t num_shards_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_NODE_TRACKER_H