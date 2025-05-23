// tests/mcts/mcts_node_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include "utils/gamestate_pool.h"
#include <memory>
#include <iostream> // Used for test results output

// MCTSMockGameState is needed by other tests, so it's restored.
class MCTSMockGameState : public alphazero::core::IGameState {
public:
    MCTSMockGameState() : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN), terminal_(false) {
    }
    MCTSMockGameState(const MCTSMockGameState& other)
        : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN),
          terminal_(other.terminal_) {
    }
    MCTSMockGameState(MCTSMockGameState&& other) noexcept
        : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN),
          terminal_(other.terminal_) {
    }
    ~MCTSMockGameState() = default;
    void setTerminal(bool terminal) { 
        terminal_ = terminal; 
    }
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { 
        return action >= 0 && action <= 2;
    }
    void makeMove(int action) override {
        if (!isLegalMove(action)) {
            throw std::runtime_error("Illegal move");
        }
    }
    bool undoMove() override { return false; }
    bool isTerminal() const override {
        return terminal_;
    }
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
    std::unique_ptr<IGameState> clone() const override final {
        return std::make_unique<MCTSMockGameState>(*this);
    }
    std::string actionToString(int action) const override { return std::to_string(action); }
    std::optional<int> stringToAction(const std::string& moveStr) const override {
        try { return std::stoi(moveStr); } catch (...) { return std::nullopt; }
    }
    std::string toString() const override { return "MCTSMockGameState"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }
    void copyFrom(const IGameState& source) override {
        const MCTSMockGameState* mock_source = dynamic_cast<const MCTSMockGameState*>(&source);
        if (!mock_source) {
            throw std::invalid_argument("Cannot copy from non-MCTSMockGameState");
        }
        terminal_ = mock_source->terminal_;
    }
private:
    bool terminal_;
};

// Test fixture restored for other tests
class MCTSNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize game_state for tests that use the fixture's state
        // If a test like Initialization doesn't use this, it will create its own.
        game_state = std::make_unique<MCTSMockGameState>();
    }
    
    void TearDown() override {
        // No actions needed, unique_ptr will handle memory cleanup
    }
    
    std::unique_ptr<MCTSMockGameState> game_state;
};

// Test basic initialization
TEST_F(MCTSNodeTest, Initialization) {
    auto local_game_state = std::make_unique<alphazero::games::gomoku::GomokuState>();
    
    auto node = alphazero::mcts::MCTSNode::create(std::move(local_game_state));
    
    bool terminal_status = node->isTerminal();
    EXPECT_FALSE(terminal_status);
    
    EXPECT_TRUE(node->isLeaf()); 
    EXPECT_EQ(node->getVisitCount(), 0);
    EXPECT_EQ(node->getValue(), 0.0f);
    EXPECT_EQ(node->getParent(), nullptr);
    EXPECT_EQ(node->getAction(), -1);
}

// Separate test for expansion to avoid any fixture interference
TEST(MCTSNodeExpansionTest, NonTerminalExpansion) {
    // Create a fresh MCTSMockGameState to ensure we have control over its state
    auto local_game_state = std::make_unique<MCTSMockGameState>();
    
    local_game_state->setTerminal(false); // Ensure it's not terminal for expansion
    
    // Verify state is actually non-terminal before creating the node
    ASSERT_FALSE(local_game_state->isTerminal()) << "MCTSMockGameState should not be terminal before creating node";
    
    // Create a node with the state
    auto node = alphazero::mcts::MCTSNode::create(std::move(local_game_state));
    
    // Check if node's state is terminal
    bool is_terminal = node->isTerminal();
    
    // Verify the node's state is also non-terminal
    ASSERT_FALSE(is_terminal) << "MCTSNode should not be terminal after construction";

    node->expand();

    EXPECT_FALSE(node->isLeaf()) << "Node should not be a leaf after expansion";
    EXPECT_EQ(node->getChildren().size(), 3) << "Node should have 3 children"; // 3 legal moves
    EXPECT_EQ(node->getActions().size(), 3) << "Node should have 3 actions";
}

// Define a NON-FIXTURE version of the test to completely replace the problematic fixture test
// This ensures the test will be isolated from any fixture state
TEST(MCTSNodeIndependentTest, NodeExpansion) {
    // Don't use the fixture's state - create a completely new state
    // This detaches this test from any global state
    MCTSMockGameState* state_for_node = new MCTSMockGameState();
    state_for_node->setTerminal(false);
    
    // Verify the state is not terminal
    ASSERT_FALSE(state_for_node->isTerminal());
    
    // Use a unique_ptr that we create explicitly rather than the state itself
    auto node_state = std::unique_ptr<MCTSMockGameState>(state_for_node);
    
    // Create the node
    auto node = alphazero::mcts::MCTSNode::create(std::move(node_state));
    
    // Check terminal status
    bool is_terminal = node->isTerminal();
    
    // Verify the node's state is not terminal
    ASSERT_FALSE(is_terminal);
    
    // Expand
    node->expand();
    
    // Verify expanded correctly
    EXPECT_FALSE(node->isLeaf());
    EXPECT_EQ(node->getChildren().size(), 3); // 3 legal moves
    EXPECT_EQ(node->getActions().size(), 3);

    // Check that parent pointers are correctly set
    for (size_t i = 0; i < node->getChildren().size(); ++i) {
        EXPECT_EQ(node->getChildren()[i]->getParent(), node);

        // Instead of checking exact action values which might be shuffled due to random shuffle in expand(),
        // just verify that each action is in the valid range
        int action = node->getChildren()[i]->getAction();
        EXPECT_GE(action, 0);
        EXPECT_LE(action, 2);
    }
}

// Test terminal state expansion using a new test class to avoid fixture interference
TEST(MCTSNodeTerminalTest, TerminalStateExpansion) {
    // Create a new state just for this test
    auto local_game_state = std::make_unique<MCTSMockGameState>();
    
    local_game_state->setTerminal(true);
    
    auto node = alphazero::mcts::MCTSNode::create(std::move(local_game_state));
    
    node->expand();
    
    EXPECT_TRUE(node->isLeaf());
    EXPECT_TRUE(node->isTerminal());
    EXPECT_EQ(node->getChildren().size(), 0);
}

// Restore the original fixture test to correctly use the fixture and assert
TEST_F(MCTSNodeTest, TerminalStateExpansion) {
    ASSERT_NE(game_state, nullptr);
    game_state->setTerminal(true);
    
    auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));
    
    // Verify the node is terminal before expansion
    ASSERT_TRUE(node->isTerminal()) << "Node should be terminal before expansion";

    node->expand();
    
    EXPECT_TRUE(node->isLeaf()) << "Terminal node should remain a leaf after calling expand";
    EXPECT_TRUE(node->isTerminal()) << "Terminal node should still be terminal after calling expand";
    EXPECT_EQ(node->getChildren().size(), 0) << "Terminal node should have no children after calling expand";
}

// This is needed since we can't delete tests in the all_tests build
TEST_F(MCTSNodeTest, Expansion) {
    // Use the fixture's game_state
    ASSERT_NE(game_state, nullptr);
    game_state->setTerminal(false);

    // Create node using the fixture's state
    auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));

    // Verify the node's state is not terminal
    ASSERT_FALSE(node->isTerminal()) << "Node should not be terminal before expansion";

    // Expand
    node->expand();

    // Verify expanded correctly
    EXPECT_FALSE(node->isLeaf()) << "Node should not be a leaf after expansion";
    EXPECT_EQ(node->getChildren().size(), 3) << "Node should have 3 children based on MCTSMockGameState legal moves";
    EXPECT_EQ(node->getActions().size(), 3) << "Node should have 3 actions";

    // Check that parent pointers are correctly set
    for (size_t i = 0; i < node->getChildren().size(); ++i) {
        EXPECT_EQ(node->getChildren()[i]->getParent(), node);
        int action = node->getChildren()[i]->getAction();
        EXPECT_GE(action, 0);
        EXPECT_LE(action, 2);
    }
}

// Test child selection with UCT
TEST_F(MCTSNodeTest, ChildSelection) {
    ASSERT_NE(game_state, nullptr);
    game_state->setTerminal(false);
    
    auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));
    
    node->expand();
    
    auto children = node->getChildren();
    EXPECT_EQ(children.size(), 3);
    
    if (children.size() > 0) {
        // Set different prior probabilities for testing selection
        node->getChildren()[0]->setPriorProbability(0.7f);
        node->getChildren()[1]->setPriorProbability(0.2f);
        node->getChildren()[2]->setPriorProbability(0.1f);
        
        // Update parent once to make visit_count_ > 0, so exploration term is non-zero
        node->update(0.0f); 

        // First selection should be the highest prior
        auto selected = node->selectChild(1.0f);
        EXPECT_EQ(selected, node->getChildren()[0]);
        
        // Update the first child with a negative value
        selected->update(-1.0f);
        
        // Next selection should prefer exploration
        selected = node->selectChild(1.0f);
        EXPECT_NE(selected, node->getChildren()[0]);
    }
}

// Test virtual loss
TEST_F(MCTSNodeTest, VirtualLoss) {
    // game_state is provided by the fixture
    ASSERT_NE(game_state, nullptr);
    game_state->setTerminal(false);
    auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));
    
    node->expand();
    
    // Set equal prior probabilities
    for (auto child : node->getChildren()) {
        child->setPriorProbability(1.0f/3);
    }
    
    // First selection
    std::shared_ptr<alphazero::mcts::MCTSNode> selected = node->selectChild(1.0f);
    selected->addVirtualLoss();
    
    // Next selection should be different
    std::shared_ptr<alphazero::mcts::MCTSNode> second = node->selectChild(1.0f);
    EXPECT_NE(second, selected);
    
    // Remove virtual loss
    selected->removeVirtualLoss();
    
    // Should be back to original selection
    std::shared_ptr<alphazero::mcts::MCTSNode> third = node->selectChild(1.0f);
    EXPECT_EQ(third, selected);
}

// Test backpropagation
TEST_F(MCTSNodeTest, Backpropagation) {
    // game_state is provided by the fixture
    ASSERT_NE(game_state, nullptr);
    game_state->setTerminal(false);
    auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));
    
    // Update statistics
    node->update(1.0f);
    node->update(0.5f);
    
    EXPECT_EQ(node->getVisitCount(), 2);
    EXPECT_FLOAT_EQ(node->getValue(), 0.75f); // (1.0 + 0.5) / 2
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }