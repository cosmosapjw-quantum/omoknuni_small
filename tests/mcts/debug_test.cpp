#include <gtest/gtest.h>
#include <iostream>
#include "mcts/mcts_node.h"

// Debug test to isolate the issue with MockGameState
class MockGameStateDebug : public alphazero::core::IGameState {
public:
    // Explicitly initialize terminal_ to false and add print statements
    MockGameStateDebug() : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN), terminal_(false) {
        std::cout << "MockGameStateDebug constructor called, terminal_=" << (terminal_ ? "true" : "false") << std::endl;
    }

    // Copy constructor with explicit terminal_ initialization
    MockGameStateDebug(const MockGameStateDebug& other)
        : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN),
          terminal_(other.terminal_) {
        std::cout << "MockGameStateDebug copy constructor called, terminal_=" << (terminal_ ? "true" : "false") << std::endl;
    }

    void setTerminal(bool terminal) { 
        terminal_ = terminal; 
        std::cout << "setTerminal(" << (terminal ? "true" : "false") << ") called" << std::endl;
    }

    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action <= 2; }
    void makeMove(int action) override {}
    bool undoMove() override { return false; }

    // Debug isTerminal
    bool isTerminal() const override {
        std::cout << "isTerminal() called, returning: " << (terminal_ ? "true" : "false") << std::endl;
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
    std::unique_ptr<IGameState> clone() const override {
        std::cout << "clone() called, current terminal_=" << (terminal_ ? "true" : "false") << std::endl;
        auto clone = std::make_unique<MockGameStateDebug>();
        clone->terminal_ = terminal_;
        std::cout << "after clone: clone->terminal_=" << (clone->terminal_ ? "true" : "false") << std::endl;
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
    std::string toString() const override { return "MockGameStateDebug"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }

private:
    bool terminal_; // Controls whether the game state is terminal
};

// Test basic initialization
TEST(DebugTest, TestTerminalState) {
    std::cout << "Creating MockGameStateDebug" << std::endl;
    auto game_state = std::make_unique<MockGameStateDebug>();
    
    std::cout << "Testing initial state (should be non-terminal)" << std::endl;
    EXPECT_FALSE(game_state->isTerminal());
    
    std::cout << "Creating MCTSNode" << std::endl;
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    std::cout << "Testing node terminal state (should be false)" << std::endl;
    EXPECT_FALSE(node.isTerminal());
}

// Test with explicitly terminal state
TEST(DebugTest, TestWithTerminalState) {
    std::cout << "Creating MockGameStateDebug with terminal=true" << std::endl;
    auto game_state = std::make_unique<MockGameStateDebug>();
    game_state->setTerminal(true);
    
    std::cout << "Testing explicit terminal state (should be terminal)" << std::endl;
    EXPECT_TRUE(game_state->isTerminal());
    
    std::cout << "Creating MCTSNode with terminal state" << std::endl;
    alphazero::mcts::MCTSNode node(std::move(game_state));
    
    std::cout << "Testing node terminal state (should be true)" << std::endl;
    EXPECT_TRUE(node.isTerminal());
}

int main(int argc, char **argv) {
    std::cout << "Starting debug tests" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}