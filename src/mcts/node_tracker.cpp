// src/mcts/node_tracker.cpp
#include "mcts/node_tracker.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <thread>

namespace alphazero {
namespace mcts {

NodeTracker::NodeTracker(size_t num_shards) {
    // Use aggressive number of shards for maximum parallelism
    if (num_shards == 0) {
        num_shards_ = std::max(size_t(16), size_t(std::thread::hardware_concurrency() * 2));
    } else {
        num_shards_ = num_shards;
    }
    
    // Note: parallel_node_hash_map doesn't have reserve(size, shards) or subcnt methods
    // It automatically handles sharding internally
    pending_evaluations_.reserve(10000);
}

NodeTracker::~NodeTracker() {
    clear();
}

bool NodeTracker::registerPendingEvaluation(NodePtr node, 
                                          NetworkPromise promise,
                                          NetworkFuture future,
                                          const std::vector<NodePtr>& path) {
    if (!node) return false;
    
    // Create pending evaluation entry
    PendingEvaluation pending{
        node,
        promise,
        future,
        path,
        std::chrono::steady_clock::now()
    };
    
    // Use raw pointer as key for performance
    void* key = node.get();
    
    // Try to insert - returns false if already exists
    auto [it, inserted] = pending_evaluations_.try_emplace(key, std::move(pending));
    
    if (inserted) {
        pending_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return inserted;
}

bool NodeTracker::hasPendingEvaluation(const NodePtr& node) const {
    if (!node) return false;
    
    void* key = node.get();
    return pending_evaluations_.count(key) > 0;
}

std::optional<NodeTracker::PendingEvaluation> NodeTracker::getPendingEvaluation(const NodePtr& node) {
    if (!node) return std::nullopt;
    
    void* key = node.get();
    auto it = pending_evaluations_.find(key);
    
    if (it != pending_evaluations_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

bool NodeTracker::removePendingEvaluation(const NodePtr& node) {
    if (!node) return false;
    
    void* key = node.get();
    size_t removed = pending_evaluations_.erase(key);
    
    if (removed > 0) {
        pending_count_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

void NodeTracker::submitResult(const NodePtr& node, const NetworkOutput& output, const std::vector<NodePtr>& path) {
    if (!node) return;
    
    // Create result entry
    EvaluationResult result{
        node,
        output,
        path
    };
    
    // Submit to lock-free queue
    result_queue_.enqueue(std::move(result));
    
    // Remove from pending
    removePendingEvaluation(node);
}

bool NodeTracker::getNextResult(EvaluationResult& result) {
    return result_queue_.try_dequeue(result);
}

void NodeTracker::clear() {
    // Clear pending evaluations
    pending_evaluations_.clear();
    pending_count_.store(0, std::memory_order_relaxed);
    
    // Clear result queue
    EvaluationResult dummy;
    while (result_queue_.try_dequeue(dummy)) {}
}

} // namespace mcts
} // namespace alphazero