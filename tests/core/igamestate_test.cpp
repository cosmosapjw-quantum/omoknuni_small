// tests/core/igamestate_test.cpp
#include <gtest/gtest.h>
#include "core/igamestate.h"

// Create a simple mock implementation for testing
class MockGameState : public alphazero::core::IGameState {
public:
    MockGameState() : IGameState(alphazero::core::GameType::UNKNOWN) {}
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action < 3; }
    void makeMove(int action) override { 
        if (!isLegalMove(action)) {
            throw std::runtime_error("Illegal move");
        }
        moves_.push_back(action); 
    }
    bool undoMove() override { 
        if (moves_.empty()) return false;
        moves_.pop_back();
        return true;
    }
    bool isTerminal() const override { return moves_.size() >= 3; }
    alphazero::core::GameResult getGameResult() const override { 
        return isTerminal() ? alphazero::core::GameResult::DRAW : alphazero::core::GameResult::ONGOING; 
    }
    int getCurrentPlayer() const override { return (moves_.size() % 2) + 1; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override { 
        return {{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}};
    }
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override { 
        return {{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}};
    }
    uint64_t getHash() const override { 
        // Simple hash based on move history
        uint64_t hash = 0;
        for (int move : moves_) {
            hash = hash * 31 + move;
        }
        return hash;
    }
    std::unique_ptr<alphazero::core::IGameState> clone() const override { 
        auto game = std::make_unique<MockGameState>();
        for (int move : moves_) {
            game->makeMove(move);
        }
        return game;
    }
    std::string actionToString(int action) const override { 
        return std::to_string(action);
    }
    std::optional<int> stringToAction(const std::string& moveStr) const override { 
        try {
            int action = std::stoi(moveStr);
            return isLegalMove(action) ? std::optional<int>(action) : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    std::string toString() const override { 
        std::string result = "MockGameState with moves: ";
        for (int move : moves_) {
            result += std::to_string(move) + " ";
        }
        return result;
    }
    bool equals(const alphazero::core::IGameState& other) const override { 
        if (other.getGameType() != getGameType()) return false;
        try {
            const MockGameState& otherGame = dynamic_cast<const MockGameState&>(other);
            return otherGame.moves_ == moves_;
        } catch (...) {
            return false;
        }
    }
    std::vector<int> getMoveHistory() const override { return moves_; }
    bool validate() const override { return true; }
    
private:
    std::vector<int> moves_;
};

TEST(IGameStateTest, BasicFunctionality) {
    MockGameState game;
    
    // Test game type
    EXPECT_EQ(game.getGameType(), alphazero::core::GameType::UNKNOWN);
    
    // Test basic game operations
    auto legalMoves = game.getLegalMoves();
    EXPECT_EQ(legalMoves.size(), 3);
    EXPECT_EQ(legalMoves[0], 0);
    EXPECT_EQ(legalMoves[1], 1);
    EXPECT_EQ(legalMoves[2], 2);
    
    EXPECT_TRUE(game.isLegalMove(0));
    EXPECT_TRUE(game.isLegalMove(1));
    EXPECT_TRUE(game.isLegalMove(2));
    EXPECT_FALSE(game.isLegalMove(3));
    
    // Test string conversion
    EXPECT_EQ(game.actionToString(1), "1");
    EXPECT_EQ(game.stringToAction("1"), 1);
    EXPECT_FALSE(game.stringToAction("invalid"));
    
    // Test cloning
    auto clone = game.clone();
    EXPECT_NE(clone, nullptr);
    EXPECT_TRUE(game.equals(*clone));
}

TEST(IGameStateTest, GamePlay) {
    MockGameState game;
    
    // Test initial state
    EXPECT_EQ(game.getCurrentPlayer(), 1);
    EXPECT_FALSE(game.isTerminal());
    EXPECT_EQ(game.getGameResult(), alphazero::core::GameResult::ONGOING);
    
    // Make some moves
    game.makeMove(0);
    EXPECT_EQ(game.getCurrentPlayer(), 2);
    EXPECT_FALSE(game.isTerminal());
    
    game.makeMove(1);
    EXPECT_EQ(game.getCurrentPlayer(), 1);
    EXPECT_FALSE(game.isTerminal());
    
    game.makeMove(2);
    EXPECT_EQ(game.getCurrentPlayer(), 2);
    EXPECT_TRUE(game.isTerminal());
    EXPECT_EQ(game.getGameResult(), alphazero::core::GameResult::DRAW);
    
    // Test undo
    EXPECT_TRUE(game.undoMove());
    EXPECT_EQ(game.getCurrentPlayer(), 1);
    EXPECT_FALSE(game.isTerminal());
    
    // Test hash
    MockGameState game2;
    game2.makeMove(0);
    game2.makeMove(1);
    
    EXPECT_EQ(game.getHash(), game2.getHash());
    
    // Test move history
    auto history = game.getMoveHistory();
    EXPECT_EQ(history.size(), 2);
    EXPECT_EQ(history[0], 0);
    EXPECT_EQ(history[1], 1);
}

TEST(IGameStateTest, IllegalMove) {
    MockGameState game;
    
    // Test exception for illegal move
    EXPECT_THROW(game.makeMove(3), std::runtime_error);
}

// Include test main at the end
#include "../test_main.h"