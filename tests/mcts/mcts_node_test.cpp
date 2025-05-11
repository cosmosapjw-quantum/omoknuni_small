// tests/mcts/mcts_node_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_node.h"
#include <memory>

// Simple mock game state for testing
class MockGameState : public alphazero::core::IGameState {
public:
    MockGameState() : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN), terminal_(false) {}
    
    void setTerminal(bool terminal) { terminal_ = terminal; }
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action <= 2; }
    void makeMove(int action) override {}
    bool undoMove() override { return false; }
    bool isTerminal() const override { return terminal_; }
    alphazero::core::GameResult getGameResult() const override { return alphazero::core::GameResult::DRAW; }
    int getCurrentPlayer() const override { return 1; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override { 
        return std::vector<std::vector<std::vector<float>>>
               (2, std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));
    }
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return getTensorRepresentation();
    }
    uint64_t getHash() const override { return 0; }
    std::unique_ptr<IGameState> clone() const override { 
        auto clone = std::make_unique<MockGameState>();
        clone->terminal_ = terminal_;
        return clone;
    }
    std::string actionToString(int action) const override { return std::to_string(action); }
    std::optional<int> stringToAction(const std::string& moveStr) const override { 
        try {
            return std::stoi(moveStr);
        } catch (...) {
            return std::nullopt;
        }
    }
    std::string toString() const override { return "MockGameState"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }

private:
    bool terminal_;
};

// Test fixture
class MCTSNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        game_state = std::make_unique<MockGameState>();
    }
    
    std::unique_ptr<MockGameState> game_state;
};

// Test basic initialization
TEST_F(MCTSNodeTest, Initialization) {
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    EXPECT_TRUE(node.isLeaf());
    EXPECT_FALSE(node.isTerminal());
    EXPECT_EQ(node.getVisitCount(), 0);
    EXPECT_EQ(node.getValue(), 0.0f);
    EXPECT_EQ(node.getParent(), nullptr);
    EXPECT_EQ(node.getAction(), -1);
}

// Test expansion
TEST_F(MCTSNodeTest, Expansion) {
    game_state = std::make_unique<MockGameState>();
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    node.expand();
    
    EXPECT_FALSE(node.isLeaf());
    EXPECT_EQ(node.getChildren().size(), 3); // 3 legal moves
    EXPECT_EQ(node.getActions().size(), 3);
    
    for (size_t i = 0; i < node.getChildren().size(); ++i) {
        EXPECT_EQ(node.getChildren()[i]->getParent(), &node);
        EXPECT_EQ(node.getChildren()[i]->getAction(), i);
    }
}

// Test terminal state expansion
TEST_F(MCTSNodeTest, TerminalStateExpansion) {
    game_state = std::make_unique<MockGameState>();
    game_state->setTerminal(true);
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    node.expand();
    
    EXPECT_TRUE(node.isLeaf());
    EXPECT_TRUE(node.isTerminal());
    EXPECT_EQ(node.getChildren().size(), 0);
}

// Test child selection with UCT
TEST_F(MCTSNodeTest, ChildSelection) {
    game_state = std::make_unique<MockGameState>();
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    node.expand();
    
    // Set different prior probabilities
    node.getChildren()[0]->setPriorProbability(0.7f);
    node.getChildren()[1]->setPriorProbability(0.2f);
    node.getChildren()[2]->setPriorProbability(0.1f);
    
    // First selection should be the highest prior
    alphazero::mcts::MCTSNode* selected = node.selectChild(1.0f);
    EXPECT_EQ(selected, node.getChildren()[0]);
    
    // Update the first child with a negative value
    selected->update(-1.0f);
    
    // Next selection should prefer exploration
    selected = node.selectChild(1.0f);
    EXPECT_NE(selected, node.getChildren()[0]);
}

// Test virtual loss
TEST_F(MCTSNodeTest, VirtualLoss) {
    game_state = std::make_unique<MockGameState>();
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    node.expand();
    
    // Set equal prior probabilities
    for (auto* child : node.getChildren()) {
        child->setPriorProbability(1.0f/3);
    }
    
    // First selection
    alphazero::mcts::MCTSNode* selected = node.selectChild(1.0f);
    selected->addVirtualLoss();
    
    // Next selection should be different
    alphazero::mcts::MCTSNode* second = node.selectChild(1.0f);
    EXPECT_NE(second, selected);
    
    // Remove virtual loss
    selected->removeVirtualLoss();
    
    // Should be back to original selection
    alphazero::mcts::MCTSNode* third = node.selectChild(1.0f);
    EXPECT_EQ(third, selected);
}

// Test backpropagation
TEST_F(MCTSNodeTest, Backpropagation) {
    game_state = std::make_unique<MockGameState>();
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    // Update statistics
    node.update(1.0f);
    node.update(0.5f);
    
    EXPECT_EQ(node.getVisitCount(), 2);
    EXPECT_FLOAT_EQ(node.getValue(), 0.75f); // (1.0 + 0.5) / 2
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }