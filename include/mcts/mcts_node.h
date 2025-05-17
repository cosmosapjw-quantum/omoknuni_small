// include/mcts/mcts_node.h
#ifndef ALPHAZERO_MCTS_NODE_H
#define ALPHAZERO_MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <future>
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "mcts/evaluation_types.h"
#include "utils/gamestate_pool.h"

namespace alphazero {
namespace mcts {

class ALPHAZERO_API MCTSNode : public std::enable_shared_from_this<MCTSNode> {
private:
    // Private constructor to force using factory method
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
    int getNumExpandedChildren() const;

    // Virtual loss
    void addVirtualLoss();
    void removeVirtualLoss();

    // Backpropagation
    void update(float value);
    
    // RAVE updates
    void updateRAVE(float value);
    float getRAVEValue() const;
    int getRAVECount() const;
    
    // Combined value with RAVE
    float getCombinedValue(float rave_constant) const;
    
    // Evaluation state - using atomic_flag for lock-free checking
    bool tryMarkForEvaluation(); // Returns true if successfully marked
    void clearEvaluationFlag();
    bool isBeingEvaluated() const;

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
    void applyVirtualLoss(int amount = 1);
    int getVirtualLoss() const;
    
    // Pending evaluation management - simplified for lock-free operations
    bool hasPendingEvaluation() const;
    void markEvaluationPending();
    void clearPendingEvaluation();

private:
    // Game state
    std::unique_ptr<core::IGameState> state_;
    
    // Tree structure
    std::weak_ptr<MCTSNode> parent_;
    std::vector<std::shared_ptr<MCTSNode>> children_;
    std::vector<int> actions_;
    int action_; // Action that led to this node
    
    // MCTS statistics
    std::atomic<int> visit_count_;
    std::atomic<float> value_sum_;
    std::atomic<int> virtual_loss_count_;
    
    // RAVE statistics
    std::atomic<int> rave_count_;
    std::atomic<float> rave_value_sum_;
    
    // Prior probabilities from neural network
    std::vector<float> prior_probabilities_;
    float prior_probability_;
    
    // Flag to indicate if neural network evaluation is in progress
    std::atomic<bool> evaluation_in_progress_{false};
    
    // Pending evaluation tracking - using atomic_flag for lock-free operations
    mutable std::atomic_flag pending_evaluation_ = ATOMIC_FLAG_INIT;
    
    // Remove mutex as we use lock-free atomic operations
    // std::mutex evaluation_mutex_;
    
    // Thread safety - atomic flag for expansion status (lock-free)
    std::atomic<bool> is_expanded_{false};
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_NODE_H