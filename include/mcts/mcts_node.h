// include/mcts/mcts_node.h
#ifndef ALPHAZERO_MCTS_NODE_H
#define ALPHAZERO_MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <future>
#include <random>
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "mcts/evaluation_types.h"
#include "utils/gamestate_pool.h"
#include "utils/thread_local_allocator.h"

namespace alphazero {
namespace mcts {

class ALPHAZERO_API MCTSNode : public std::enable_shared_from_this<MCTSNode> {
private:
    // Friend class for memory pool
    friend class MCTSNodePool;
    
    // Constructor accessible to pool
    MCTSNode(std::unique_ptr<core::IGameState> state, std::weak_ptr<MCTSNode> parent = {});
    
public:
    // Factory method to create nodes as shared_ptr
    static std::shared_ptr<MCTSNode> create(std::unique_ptr<core::IGameState> state, 
                                          std::shared_ptr<MCTSNode> parent = nullptr);
    
    // Get shared pointer to self (for children management)
    std::shared_ptr<MCTSNode> getSharedPtr();
    
    ~MCTSNode();

    // Node selection using PUCT formula with virtual loss
    std::shared_ptr<MCTSNode> selectChild(float exploration_factor, bool use_rave = false, float rave_constant = 3000.0f);

    // Expansion with progressive widening
    void expand(bool use_progressive_widening = false, float cpw = 1.0f, float kpw = 10.0f);
    bool isFullyExpanded() const;
    bool isLeaf() const;
    bool isTerminal() const;
    bool isExpanded() const { return is_expanded_.load(std::memory_order_acquire); }
    int getNumExpandedChildren() const;
    
    // UCB Selection for enhanced optimized search
    std::shared_ptr<MCTSNode> selectBestChildUCB(float exploration_constant, std::mt19937& rng);

    // Virtual loss
    void addVirtualLoss();
    void addVirtualLoss(int amount);
    void removeVirtualLoss();
    void removeVirtualLoss(int amount);
    
    // Aliases for compatibility with updated code
    void applyVirtualLoss(int amount) { addVirtualLoss(amount); }
    void revertVirtualLoss(int amount) { removeVirtualLoss(amount); }
    
    // Fast methods for ultra-low latency (minimal locking, inlined operations)
    std::shared_ptr<MCTSNode> selectChildFast(float exploration_constant) const;
    void expandFast();  // Minimal expansion without progressive widening
    void updateFast(float value);  // Lock-free update

    // Backpropagation
    void update(float value);
    
    // Aliases for compatibility with updated code
    void updateStats(float value) { update(value); }
    void updateRecursive(float value); // Forward declaration for optimized implementation
    
    // RAVE updates
    void updateRAVE(float value);
    float getRAVEValue() const;
    int getRAVECount() const;
    
    // Combined value with RAVE
    float getCombinedValue(float rave_constant) const;
    
    // Evaluation state - using atomic operations for thread safety
    bool tryMarkForEvaluation(); // Returns true if successfully marked (lock-free)
    void clearEvaluationFlag();  // Thread-safe clear
    bool isBeingEvaluated() const; // Thread-safe check

    // Getters
    const core::IGameState& getState() const;
    core::IGameState& getStateMutable();
    std::vector<std::shared_ptr<MCTSNode>>& getChildren();
    std::vector<int>& getActions();
    std::shared_ptr<MCTSNode> getParent();
    float getValue() const;
    int getVisitCount() const;
    int getAction() const;
    float getPriorProbability() const;

    // Setters
    void setAction(int action);
    void setPriorProbability(float prior);
    void setPriorProbabilities(const std::vector<float>& priors);
    
    // Direct parent access for expansion (to avoid circular reference)
    void setParentDirectly(std::weak_ptr<MCTSNode> parent) { parent_ = parent; }

    // Update a child reference if a transposition is found
    bool updateChildReference(const std::shared_ptr<MCTSNode>& old_child, const std::shared_ptr<MCTSNode>& new_child);
    
    // Virtual loss management
    int getVirtualLoss() const;
    
    // Pending evaluation management - simplified for lock-free operations
    bool hasPendingEvaluation() const;
    void markEvaluationPending();
    void clearPendingEvaluation();
    
    // CRITICAL FIX: Added method to clear both evaluation flags at once for stuck nodes
    void clearAllEvaluationFlags() {
        evaluation_in_progress_.store(false, std::memory_order_release);  // Use atomic store
        clearPendingEvaluation();         // Use existing method for pending flag
    }
    
    // CRITICAL FIX: Check if the node has a valid state
    bool hasValidState() const {
        return static_cast<bool>(state_);
    }
    
    // New methods required by mcts_taskflow_engine
    float getPrior() const { return prior_probability_; }
    void setPrior(float prior) { prior_probability_ = prior; }
    int getPlayer() const { 
        if (!state_) {
            std::cout << "💥 CRITICAL: getPlayer() called on NULL state!" << std::endl;
            return 1; // Default to player 1
        }
        return state_->getCurrentPlayer(); 
    }
    int getDepth() const;
    std::mutex& getExpansionMutex();
    float getTerminalValue() const;
    std::shared_ptr<MCTSNode> getMostVisitedChild() const;
    bool needsEvaluation() const { return !is_expanded_ && !isTerminal(); }
    void markEvaluationInProgress() { evaluation_in_progress_ = true; }
    bool hasChildren() const { return !children_.empty(); }

private:
    // Mutable mutex for const methods that need synchronization
    mutable std::mutex expansion_mutex_;
    // Game state
    std::unique_ptr<core::IGameState> state_;
    
    // Tree structure
    std::weak_ptr<MCTSNode> parent_;
    std::vector<std::shared_ptr<MCTSNode>> children_;
    std::vector<int> actions_;
    int action_; // Action that led to this node
    
    // MCTS statistics with cache line padding to prevent false sharing
    // Each atomic variable is placed in its own cache line to avoid thread contention
    alignas(64) std::atomic<int> visit_count_;
    char padding1[60]; // Fill the rest of the cache line (64 bytes typical)
    
    alignas(64) std::atomic<float> value_sum_;
    char padding2[60];
    
    alignas(64) std::atomic<int> virtual_loss_count_;
    char padding3[60];
    
    // RAVE statistics with padding
    alignas(64) std::atomic<int> rave_count_;
    char padding4[60];
    
    alignas(64) std::atomic<float> rave_value_sum_;
    char padding5[60];
    
    // Prior probabilities from neural network
    std::vector<float> prior_probabilities_;
    float prior_probability_;
    
    // Flag to indicate if neural network evaluation is in progress
    // Added alignment and padding to prevent false sharing
    alignas(64) std::atomic<bool> evaluation_in_progress_{false};
    char padding6[60];
    
    // Pending evaluation tracking - using atomic_bool for proper lock-free operations
    alignas(64) mutable std::atomic<bool> pending_evaluation_{false};
    char padding7[60];
    
    // Remove mutex as we use lock-free atomic operations
    // std::mutex evaluation_mutex_;
    
    // Thread safety - atomic flag for expansion status (lock-free)
    alignas(64) std::atomic<bool> is_expanded_{false};
    char padding8[60];
    
    // PERFORMANCE CACHE: Terminal status cache to avoid repeated expensive isTerminal() calls
    alignas(64) mutable std::atomic<int> cached_terminal_status_{-1}; // -1=uncached, 0=false, 1=true
    char padding9[60];
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_NODE_H