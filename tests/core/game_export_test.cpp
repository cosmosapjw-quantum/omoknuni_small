// tests/core/game_export_test.cpp
#include <gtest/gtest.h>
#include "core/game_export.h"
#include <cstdio> // for std::remove

// Define a custom game type for testing
constexpr alphazero::core::GameType TEST_GAME_TYPE = 
    static_cast<alphazero::core::GameType>(99);

// Create a simple mock game implementation for testing
class MockGame : public alphazero::core::IGameState {
public:
    MockGame() : IGameState(TEST_GAME_TYPE) {}
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action < 3; }
    void makeMove(int action) override { moves_.push_back(action); }
    bool undoMove() override { 
        if (moves_.empty()) return false;
        moves_.pop_back();
        return true;
    }
    bool isTerminal() const override { return moves_.size() >= 9; }
    alphazero::core::GameResult getGameResult() const override { 
        return isTerminal() ? alphazero::core::GameResult::DRAW : alphazero::core::GameResult::ONGOING;
    }
    int getCurrentPlayer() const override { return (moves_.size() % 2) + 1; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override { 
        return {{{0.0f}}};
    }
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override { 
        return {{{0.0f}}};
    }
    uint64_t getHash() const override { return 0; }
    std::unique_ptr<alphazero::core::IGameState> clone() const override { 
        auto game = std::make_unique<MockGame>();
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
        std::string result = "MockGame with moves: ";
        for (int move : moves_) {
            result += std::to_string(move) + " ";
        }
        return result;
    }
    bool equals(const alphazero::core::IGameState& other) const override { 
        if (other.getGameType() != getGameType()) return false;
        try {
            const MockGame& otherGame = dynamic_cast<const MockGame&>(other);
            return otherGame.moves_ == moves_;
        } catch (...) {
            return false;
        }
    }
    std::vector<int> getMoveHistory() const override { return moves_; }
    bool validate() const override { return true; }
    void copyFrom(const alphazero::core::IGameState& source) override {
        const MockGame* mock_source = dynamic_cast<const MockGame*>(&source);
        if (!mock_source) {
            throw std::invalid_argument("Cannot copy from non-MockGame");
        }
        moves_ = mock_source->moves_;
    }
    
private:
    std::vector<int> moves_;
};

// Register the mock game with the registry
static alphazero::core::GameRegistrar<MockGame> registrar(TEST_GAME_TYPE);

TEST(GameExportTest, GameRegistryWorks) {
    auto& registry = alphazero::core::GameRegistry::instance();
    
    // Check registration
    EXPECT_TRUE(registry.isRegistered(TEST_GAME_TYPE));
    EXPECT_FALSE(registry.isRegistered(static_cast<alphazero::core::GameType>(100)));
    
    // Get registered games
    auto games = registry.getRegisteredGames();
    EXPECT_FALSE(games.empty());
    
    // Check if our test game is in the list
    bool found = false;
    for (auto type : games) {
        if (type == TEST_GAME_TYPE) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
    
    // Create a game instance
    auto game = registry.createGame(TEST_GAME_TYPE);
    EXPECT_NE(game, nullptr);
    
    // Check game type
    EXPECT_EQ(game->getGameType(), TEST_GAME_TYPE);
    
    // Check basic functionality
    EXPECT_EQ(game->getBoardSize(), 3);
    EXPECT_EQ(game->getActionSpaceSize(), 9);
    
    // Test exception for unknown type
    EXPECT_THROW(
        registry.createGame(static_cast<alphazero::core::GameType>(100)),
        std::runtime_error
    );
}

TEST(GameExportTest, GameFactoryWorks) {
    // Create a basic game
    auto game = alphazero::core::GameFactory::createGame(TEST_GAME_TYPE);
    EXPECT_NE(game, nullptr);
    EXPECT_EQ(game->getGameType(), TEST_GAME_TYPE);
    
    // Create a game with moves
    auto gameWithMoves = alphazero::core::GameFactory::createGameFromMoves(TEST_GAME_TYPE, "0 1 2");
    EXPECT_NE(gameWithMoves, nullptr);
    
    // Check that moves were applied
    auto history = gameWithMoves->getMoveHistory();
    EXPECT_EQ(history.size(), 3);
    EXPECT_EQ(history[0], 0);
    EXPECT_EQ(history[1], 1);
    EXPECT_EQ(history[2], 2);
    
    // Test exception for invalid moves
    EXPECT_THROW(
        alphazero::core::GameFactory::createGameFromMoves(TEST_GAME_TYPE, "0 1 9"),
        std::runtime_error
    );
}

TEST(GameExportTest, GameSerializerWorks) {
    // Create a game with some moves
    auto game = alphazero::core::GameFactory::createGameFromMoves(TEST_GAME_TYPE, "0 1 2");
    
    // Serialize the game
    std::string serialized = alphazero::core::GameSerializer::serializeGame(*game);
    EXPECT_FALSE(serialized.empty());
    
    // Deserialize the game
    auto loadedGame = alphazero::core::GameSerializer::deserializeGame(serialized);
    EXPECT_NE(loadedGame, nullptr);
    
    // Check that the loaded game has the same moves
    auto history = loadedGame->getMoveHistory();
    EXPECT_EQ(history.size(), 3);
    EXPECT_EQ(history[0], 0);
    EXPECT_EQ(history[1], 1);
    EXPECT_EQ(history[2], 2);
    
    // Test file operations (save)
    const std::string testFile = "test_game.txt";
    EXPECT_TRUE(alphazero::core::GameSerializer::saveGame(*game, testFile));
    
    // Test file operations (load)
    auto loadedFromFile = alphazero::core::GameSerializer::loadGame(testFile);
    EXPECT_NE(loadedFromFile, nullptr);
    
    // Check that the loaded game has the same moves
    history = loadedFromFile->getMoveHistory();
    EXPECT_EQ(history.size(), 3);
    
    // Clean up
    std::remove(testFile.c_str());
}

TEST(GameExportTest, GameTypeConversionWorks) {
    // Test string to type conversion
    EXPECT_EQ(alphazero::core::stringToGameType("CHESS"), alphazero::core::GameType::CHESS);
    EXPECT_EQ(alphazero::core::stringToGameType("GO"), alphazero::core::GameType::GO);
    EXPECT_EQ(alphazero::core::stringToGameType("GOMOKU"), alphazero::core::GameType::GOMOKU);
    EXPECT_EQ(alphazero::core::stringToGameType("UNKNOWN"), alphazero::core::GameType::UNKNOWN);
    EXPECT_EQ(alphazero::core::stringToGameType("NONEXISTENT"), alphazero::core::GameType::UNKNOWN);
    
    // Test type to string conversion
    EXPECT_EQ(alphazero::core::gameTypeToString(alphazero::core::GameType::CHESS), "CHESS");
    EXPECT_EQ(alphazero::core::gameTypeToString(alphazero::core::GameType::GO), "GO");
    EXPECT_EQ(alphazero::core::gameTypeToString(alphazero::core::GameType::GOMOKU), "GOMOKU");
    EXPECT_EQ(alphazero::core::gameTypeToString(alphazero::core::GameType::UNKNOWN), "UNKNOWN");
    EXPECT_EQ(alphazero::core::gameTypeToString(static_cast<alphazero::core::GameType>(100)), "UNKNOWN");
}

// Include test main at the end
#include "../test_main.h"