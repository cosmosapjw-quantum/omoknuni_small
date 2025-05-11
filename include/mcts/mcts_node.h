// include/mcts/mcts_node.h
#ifndef ALPHAZERO_MCTS_NODE_H
#define ALPHAZERO_MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

class ALPHAZERO_API MCTSNode {
public:
    MCTSNode(std::unique_ptr<core::IGameState> state, MCTSNode* parent = nullptr);
    ~MCTSNode();

    // Node selection using PUCT formula with virtual loss
    MCTSNode* selectChild(float exploration_factor);

    // Expansion
    void expand();
    bool isFullyExpanded() const;
    bool isLeaf() const;
    bool isTerminal() const;

    // Virtual loss
    void addVirtualLoss();
    void removeVirtualLoss();

    // Backpropagation
    void update(float value);

    // Getters
    const core::IGameState& getState() const;
    core::IGameState& getStateMutable();
    std::vector<MCTSNode*>& getChildren();
    std::vector<int>& getActions();
    MCTSNode* getParent();
    float getValue() const;
    int getVisitCount() const;
    int getAction() const;
    float getPriorProbability() const;

    // Setters
    void setAction(int action);
    void setPriorProbability(float prior);
    void setPriorProbabilities(const std::vector<float>& priors);

private:
    // Game state
    std::unique_ptr<core::IGameState> state_;
    
    // Tree structure
    MCTSNode* parent_;
    std::vector<MCTSNode*> children_;
    std::vector<int> actions_;
    int action_; // Action that led to this node
    
    // MCTS statistics
    std::atomic<int> visit_count_;
    std::atomic<float> value_sum_;
    std::atomic<int> virtual_loss_count_;
    
    // Prior probabilities from neural network
    std::vector<float> prior_probabilities_;
    float prior_probability_;
    
    // Thread safety
    std::mutex expansion_mutex_;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_NODE_H