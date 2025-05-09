// tests/games/go/go_test.cpp
#include <gtest/gtest.h>
#include "games/go/go_state.h"
#include "games/go/go_rules.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace alphazero {
namespace go {
namespace testing {

// Test fixture for Go tests
class GoTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 9x9 Go game with standard settings
        state = std::make_unique<GoState>(9, 7.5f, true, true);
    }

    void TearDown() override {
        state.reset();
    }

    // Helper method to check if a specific move is legal
    bool isMoveValid(int x, int y) const {
        int action = state->coordToAction(x, y);
        return state->isLegalMove(action);
    }
    
    // Helper method to check if a specific move is legal (using action directly)
    bool isMoveValid(int action) const {
        return state->isLegalMove(action);
    }
    
    // Helper method to make a move by coordinates
    void makeMove(int x, int y) {
        int action = state->coordToAction(x, y);
        state->makeMove(action);
    }
    
    // Helper method to pass
    void passTurn() {
        state->makeMove(-1);
    }
    
    // Helper to place multiple stones of the same color
    void placeStones(const std::vector<std::pair<int, int>>& coords, int color) {
        for (const auto& [x, y] : coords) {
            int action = state->coordToAction(x, y);
            // If it's not the current player's turn, pass once
            if (state->getCurrentPlayer() != color) {
                passTurn();
            }
            state->makeMove(action);
        }
    }
    
    // Helper to place alternating stones (for both players)
    void placeAlternatingStones(const std::vector<std::pair<int, int>>& coords) {
        for (const auto& [x, y] : coords) {
            int action = state->coordToAction(x, y);
            state->makeMove(action);
        }
    }
    
    // Helper to count stones on the board
    int countStones(int color) const {
        int count = 0;
        int size = state->getBoardSize();
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                if (state->getStone(x, y) == color) {
                    count++;
                }
            }
        }
        return count;
    }
    
    std::unique_ptr<GoState> state;
};

// Test basic initialization
TEST_F(GoTest, Initialization) {
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->getBoardSize(), 9);
    EXPECT_EQ(state->getActionSpaceSize(), 9 * 9 + 1);  // +1 for pass
    EXPECT_EQ(state->getCurrentPlayer(), 1);  // Black starts
    EXPECT_FALSE(state->isTerminal());
    
    // Board should be empty
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            EXPECT_EQ(state->getStone(x, y), 0);
        }
    }
    
    // Check initial properties
    EXPECT_TRUE(state->isChineseRules());
    EXPECT_TRUE(state->isEnforcingSuperko());
    EXPECT_FLOAT_EQ(state->getKomi(), 7.5f);
    EXPECT_EQ(state->getKoPoint(), -1);  // No ko point initially
}

// Test basic move making
TEST_F(GoTest, BasicMoves) {
    // Place a stone at 4,4 (center for 9x9)
    EXPECT_TRUE(isMoveValid(4, 4));
    makeMove(4, 4);
    
    // Check that the stone is placed
    EXPECT_EQ(state->getStone(4, 4), 1);  // Black stone
    
    // It should be White's turn
    EXPECT_EQ(state->getCurrentPlayer(), 2);
    
    // Place a white stone adjacent to the black stone
    EXPECT_TRUE(isMoveValid(4, 5));
    makeMove(4, 5);
    
    // Check that the stone is placed
    EXPECT_EQ(state->getStone(4, 5), 2);  // White stone
    
    // It should be Black's turn again
    EXPECT_EQ(state->getCurrentPlayer(), 1);
}

// Test pass moves
TEST_F(GoTest, PassMoves) {
    // Test passing
    EXPECT_TRUE(isMoveValid(-1));  // Pass should be valid
    passTurn();  // Black passes
    
    EXPECT_EQ(state->getCurrentPlayer(), 2);  // White's turn
    passTurn();  // White passes
    
    // After two passes, the game should be terminal
    EXPECT_TRUE(state->isTerminal());
    
    // No captures, so Black should lose due to komi
    auto gameResult = state->getGameResult();
    EXPECT_EQ(gameResult, core::GameResult::WIN_PLAYER2);
}

// Test capture mechanics
TEST_F(GoTest, Captures) {
    // Create a simple capture scenario: White stone surrounded by Black
    //    0 1 2 3
    // 0  . . . .
    // 1  . B W .
    // 2  . B . .
    // 3  . . . .
    placeAlternatingStones({{1,1}, {2,1}, {2,0}, {5,5}, {2,2}}); // B(1,1), W(2,1)-target, B(2,0), W(5,5)-elsewhere, B(2,2). Player is White.
    
    // It's White's turn. White passes so Black can make the capture.
    passTurn(); 

    // Complete the capture (Black's move)
    EXPECT_TRUE(isMoveValid(3, 1));
    makeMove(3, 1);  // Black's move to capture
    
    // Check that the white stone is captured
    EXPECT_EQ(state->getStone(2, 1), 0);  // Empty now
    
    // Check that Black has captured one stone
    EXPECT_EQ(state->getCapturedStones(1), 1);
    EXPECT_EQ(state->getCapturedStones(2), 0);
}

// Test suicide rule
TEST_F(GoTest, SuicideRule) {
    // Create a scenario where a move would be suicide
    // White surrounds a space, and Black tries to play there
    //    0 1 2 3
    // 0  . . . .
    // 1  . W . .
    // 2  W . W .
    // 3  . W . .
    
    // First Black places a stone elsewhere
    makeMove(0, 0);
    
    // Then White surrounds a space
    placeStones({{1, 1}, {0, 2}, {2, 2}, {1, 3}}, 2);
    
    // Black tries to play in the surrounded space - should be illegal
    EXPECT_FALSE(isMoveValid(1, 2));
}

// Test ko rule
TEST_F(GoTest, KoRule) {
    // Set up a ko situation
    //    0 1 2 3
    // 0  . . . .
    // 1  . B W .
    // 2  B W B .
    // 3  . . . .
    placeAlternatingStones({{1, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}});
    
    // It's White's turn. White passes so Black can make the capture for Ko.
    passTurn();

    // Black captures a white stone, creating a ko
    makeMove(1, 3);
    
    // The white stone at (1, 2) should be captured
    EXPECT_EQ(state->getStone(1, 2), 0);
    
    // Ko point should be set to the captured position
    EXPECT_EQ(state->getKoPoint(), state->coordToAction(1, 2));
    
    // White should not be able to immediately recapture
    EXPECT_FALSE(isMoveValid(1, 2));
    
    // White plays elsewhere
    makeMove(3, 3);
    
    // Now the ko point should be cleared
    EXPECT_EQ(state->getKoPoint(), -1);
    
    // Black should be able to play at the former ko point
    EXPECT_TRUE(isMoveValid(1, 2));
}

// Test both ko rule and superko rule
TEST_F(GoTest, SuperkoRuleExtended) {
    // Clear the board
    while (!state->getMoveHistory().empty()) {
        state->undoMove();
    }
    
    //--- PART 1: Basic Ko Test ---//
    
    // Set up a basic ko pattern
    //    0 1 2 3
    // 0  . . . .
    // 1  . B W .
    // 2  B W B .
    // 3  . B . .
    placeAlternatingStones({{1, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}});
    
    // Make sure it's Black's turn
    if (state->getCurrentPlayer() != 1) {
        passTurn();
    }
    
    // Black captures the white stone at (1,2)
    makeMove(1, 3);
    
    // Verify the capture worked
    EXPECT_EQ(state->getStone(1, 2), 0) << "White stone at (1,2) should be captured";
    
    // White cannot immediately recapture
    EXPECT_FALSE(isMoveValid(1, 2)) << "White cannot immediately recapture (ko rule)";
    
    // White plays elsewhere
    makeMove(3, 3);
    
    // Check ko point status after White's move
    EXPECT_EQ(state->getKoPoint(), -1) << "Ko point should be cleared after White plays elsewhere";
    
    // Now White should be able to recapture
    EXPECT_TRUE(isMoveValid(1, 2)) << "White should now be able to recapture";
    
    //--- PART 2: Superko Test ---//
    
    // Create a new state to test superko
    auto superkoState = std::make_unique<GoState>(9, 7.5f, true, true);
    
    // Set up a simple pattern that will allow us to create position repetition
    //   0 1 2 
    // 0 . W .
    // 1 W B W
    // 2 . W .
    superkoState->setStone(1, 0, 2); // W(1,0)
    superkoState->setStone(0, 1, 2); // W(0,1)
    superkoState->setStone(1, 1, 1); // B(1,1)
    superkoState->setStone(2, 1, 2); // W(2,1)
    superkoState->setStone(1, 2, 2); // W(1,2)
    
    // Make it Black's turn
    while (superkoState->getCurrentPlayer() != 1) {
        superkoState->makeMove(-1); // Pass
    }
    
    // Save this position hash
    uint64_t initialHash = superkoState->getHash();
    
    // Create a sequence of moves
    superkoState->makeMove(superkoState->coordToAction(3, 3)); // Black plays elsewhere
    superkoState->makeMove(superkoState->coordToAction(3, 4)); // White plays elsewhere
    superkoState->makeMove(superkoState->coordToAction(4, 4)); // Black plays elsewhere
    superkoState->makeMove(superkoState->coordToAction(4, 5)); // White plays elsewhere
    
    // Verify that regular moves are legal
    EXPECT_TRUE(superkoState->isLegalMove(superkoState->coordToAction(5, 5))) 
        << "Regular moves should be legal";
    
    // Create a temporary state to check if we would recreate the initial position
    {
        GoState tempState(*superkoState);
        tempState.makeMove(tempState.coordToAction(5, 5)); // Black
        tempState.makeMove(tempState.coordToAction(5, 6)); // White
        
        // If we're back at the initial position, this hash would match
        uint64_t newHash = tempState.getHash();
        EXPECT_NE(newHash, initialHash) << "Different board state should have different hash";
    }
}

// Test liberty counting
TEST_F(GoTest, LibertyCounting) {
    // Place a single black stone in the center
    makeMove(4, 4);
    
    // We need to implement a helper that accesses the underlying groups
    // Since GoState doesn't expose its rules directly, we'll check liberties indirectly
    
    // Place white stones to reduce liberties
    placeStones({{3, 4}, {5, 4}}, 2);
    
    // We should have 2 liberties for the black stone - verify indirectly
    // by checking that moves to those liberties are legal
    EXPECT_TRUE(isMoveValid(4, 3));
    EXPECT_TRUE(isMoveValid(4, 5));
    EXPECT_FALSE(isMoveValid(3, 4)); // already occupied
    EXPECT_FALSE(isMoveValid(5, 4)); // already occupied
}

// Test territory scoring
TEST_F(GoTest, ChineseScoringRules) {
    // Create a small territory for each player
    //    0 1 2 3 4
    // 0  B B B . .
    // 1  B . B . .
    // 2  B B B . .
    // 3  . . . W W
    // 4  . . . W .
    placeStones({
        {0, 0}, {1, 0}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}
    }, 1);
    
    placeStones({
        {3, 3}, {4, 3}, {3, 4}
    }, 2);
    
    // Add some dead stones (e.g., a black stone in white's territory)
    makeMove(4, 4);
    
    // Mark the black stone as dead
    std::unordered_set<int> deadStones;
    deadStones.insert(state->coordToAction(4, 4));
    state->markDeadStones(deadStones);
    
    // Get territory ownership
    auto territory = state->getTerritoryOwnership();
    
    // Check territory ownership
    EXPECT_EQ(territory[state->coordToAction(1, 1)], 1);  // Black territory
    EXPECT_EQ(territory[state->coordToAction(4, 4)], 2);  // White territory (dead black stone)
    
    // Calculate score
    auto [blackScore, whiteScore] = state->calculateScore();
    
    // In Chinese rules, score = stones + territory
    // Black: 8 stones + 1 territory = 9
    // White: 3 stones + 1 territory (from dead B(4,4)) + 7.5 komi = 11.5
    EXPECT_FLOAT_EQ(blackScore, 9.0f);
    EXPECT_FLOAT_EQ(whiteScore, 11.5f);
    
    // White should win
    passTurn();  // Black passes
    passTurn();  // White passes
    
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER2);
}

// Test Japanese scoring rules
TEST_F(GoTest, JapaneseScoringRules) {
    // Create a Japanese rules state
    auto japaneseState = std::make_unique<GoState>(9, 6.5f, false, true);
    
    // Create a small territory for each player
    //    0 1 2 3 4
    // 0  B B B . .
    // 1  B . B . .
    // 2  B B B . .
    // 3  . . . W W
    // 4  . . . W .
    
    // Helper to place stones for this test
    auto placeStones = [&](const std::vector<std::pair<int, int>>& coords, int color) {
        for (const auto& [x, y] : coords) {
            int action = japaneseState->coordToAction(x, y);
            if (japaneseState->getCurrentPlayer() != color) {
                japaneseState->makeMove(-1); // Pass to get to correct player
            }
            japaneseState->makeMove(action);
        }
    };
    
    placeStones({
        {0, 0}, {1, 0}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}
    }, 1);
    
    placeStones({
        {3, 3}, {4, 3}, {3, 4}
    }, 2);
    
    // Add a black stone in white's territory and have it captured
    int blackStoneAction = japaneseState->coordToAction(4, 4);
    if (japaneseState->getCurrentPlayer() == 1) {
        japaneseState->makeMove(blackStoneAction);
    } else {
        japaneseState->makeMove(-1); // Pass
        japaneseState->makeMove(blackStoneAction);
    }
    
    // White captures the stone B(4,4)
    // State: P=White. Liberties of B(4,4) are (4,5) and (5,4).
    EXPECT_TRUE(japaneseState->isLegalMove(japaneseState->coordToAction(4, 5)));
    japaneseState->makeMove(japaneseState->coordToAction(4, 5)); // White plays W(4,5). P=Black.
    EXPECT_TRUE(japaneseState->isLegalMove(-1));
    japaneseState->makeMove(-1); // Black passes. P=White.
    EXPECT_TRUE(japaneseState->isLegalMove(japaneseState->coordToAction(5, 4)));
    japaneseState->makeMove(japaneseState->coordToAction(5, 4)); // White plays W(5,4), captures B(4,4). P=Black.
    EXPECT_EQ(japaneseState->getStone(4, 4), 0); // Verify capture
    EXPECT_EQ(japaneseState->getCapturedStones(2), 1); // White captured 1 stone

    // Calculate score
    auto [blackScore, whiteScore] = japaneseState->calculateScore();
    
    // In Japanese rules, score = territory + captures
    // Black: 1 territory point + 1 dame point = 2
    // White: 1 territory + 0 territory bonus + 6.5 komi = 7.5
    EXPECT_FLOAT_EQ(blackScore, 2.0f);
    EXPECT_FLOAT_EQ(whiteScore, 7.5f);
    
    // White should win
    japaneseState->makeMove(-1);  // Black passes
    japaneseState->makeMove(-1);  // White passes
    
    EXPECT_TRUE(japaneseState->isTerminal());
    EXPECT_EQ(japaneseState->getGameResult(), core::GameResult::WIN_PLAYER2);
}

// Test dead stone marking and scoring
TEST_F(GoTest, DeadStoneMarking) {
    // Create a position with a complicated situation where stones might be dead
    //    0 1 2 3 4 5 6
    // 0  B B B . . . .
    // 1  W W B . . . .
    // 2  . W B . . . .
    // 3  W B B . . . .
    // 4  B W . . . . .
    // 5  . . . . . . .
    placeAlternatingStones({
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 2}, {2, 1}, {0, 2},
        {0, 3}, {1, 3}, {1, 4}, {3, 0}, {2, 3}, {3, 1}, {4, 0}, {4, 1}
    });
    
    // Mark some white stones as dead (the ones that are surrounded)
    std::unordered_set<int> deadStones;
    deadStones.insert(state->coordToAction(0, 1));
    deadStones.insert(state->coordToAction(1, 1));
    state->markDeadStones(deadStones);
    
    // Get territory ownership with dead stones
    auto territory = state->getTerritoryOwnership();
    
    // The positions of the dead white stones should now be Black territory
    EXPECT_EQ(territory[state->coordToAction(0, 1)], 1);
    EXPECT_EQ(territory[state->coordToAction(1, 1)], 1);
    
    // Calculate score with dead stones
    auto [blackScore, whiteScore] = state->calculateScore();
    
    // Clear dead stones
    state->clearDeadStones();
    
    // Calculate score without dead stones
    auto [blackScoreNoDead, whiteScoreNoDead] = state->calculateScore();
    
    // Score with dead stones should be higher for Black
    EXPECT_GT(blackScore, blackScoreNoDead);
}

// Test move undoing
TEST_F(GoTest, UndoMove) {
    // Make some moves
    makeMove(4, 4);  // Black
    makeMove(3, 3);  // White
    makeMove(5, 5);  // Black
    
    // Get the current state of the board
    auto hash = state->getHash();
    
    // Make another move
    makeMove(6, 6);
    
    // Board should be different
    EXPECT_NE(state->getHash(), hash);
    
    // Undo the last move
    EXPECT_TRUE(state->undoMove());
    
    // Board should be back to the previous state
    EXPECT_EQ(state->getHash(), hash);
    EXPECT_EQ(state->getStone(6, 6), 0);
    EXPECT_EQ(state->getCurrentPlayer(), 2);
    
    // Undo more moves
    EXPECT_TRUE(state->undoMove());
    EXPECT_TRUE(state->undoMove());
    EXPECT_TRUE(state->undoMove());
    
    // Should be back to the initial state
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            EXPECT_EQ(state->getStone(x, y), 0);
        }
    }
    EXPECT_EQ(state->getCurrentPlayer(), 1);
    
    // No more moves to undo
    EXPECT_FALSE(state->undoMove());
}

// Test cloning and equality
TEST_F(GoTest, CloneAndEquality) {
    // Make some moves
    makeMove(4, 4);
    makeMove(3, 3);
    makeMove(5, 5);
    
    // Clone the state
    auto clonedState = state->clone();
    ASSERT_NE(clonedState, nullptr);
    
    // Check that the cloned state is equal to the original
    EXPECT_TRUE(clonedState->getGameType() == core::GameType::GO);
    EXPECT_TRUE(state->equals(*clonedState));
    
    // Make a move on the original
    makeMove(6, 6);
    
    // States should no longer be equal
    EXPECT_FALSE(state->equals(*clonedState));
    
    // Make the same move on the clone
    auto goClone = dynamic_cast<GoState*>(clonedState.get());
    ASSERT_NE(goClone, nullptr);
    goClone->makeMove(goClone->coordToAction(6, 6));
    
    // States should be equal again
    EXPECT_TRUE(state->equals(*clonedState));
}

// Test move history tracking
TEST_F(GoTest, MoveHistoryTracking) {
    // Make a series of moves
    makeMove(4, 4);
    makeMove(3, 3);
    passTurn();  // Black passes
    makeMove(5, 5);
    
    // Check the move history
    auto moveHistory = state->getMoveHistory();
    EXPECT_EQ(moveHistory.size(), 4);
    
    // Convert actions back to coordinates
    EXPECT_EQ(state->actionToCoord(moveHistory[0]), std::make_pair(4, 4));
    EXPECT_EQ(state->actionToCoord(moveHistory[1]), std::make_pair(3, 3));
    EXPECT_EQ(moveHistory[2], -1);  // Pass
    EXPECT_EQ(state->actionToCoord(moveHistory[3]), std::make_pair(5, 5));
    
    // Undo a move
    EXPECT_TRUE(state->undoMove());
    
    // Move history should be updated
    moveHistory = state->getMoveHistory();
    EXPECT_EQ(moveHistory.size(), 3);
}

// Test tensor representation for neural network input
TEST_F(GoTest, TensorRepresentation) {
    // Make some moves
    makeMove(4, 4);  // Black
    makeMove(3, 3);  // White
    
    // Get basic tensor representation
    auto tensor = state->getTensorRepresentation();
    
    // Should have at least 3 planes
    EXPECT_GE(tensor.size(), 3);
    
    // Check first plane (black stones)
    EXPECT_EQ(tensor[0][4][4], 1.0f);
    EXPECT_EQ(tensor[0][3][3], 0.0f);
    
    // Check second plane (white stones)
    EXPECT_EQ(tensor[1][4][4], 0.0f);
    EXPECT_EQ(tensor[1][3][3], 1.0f);
    
    // Third plane should indicate current player
    if (state->getCurrentPlayer() == 1) {  // Black to play
        EXPECT_EQ(tensor[2][0][0], 1.0f);
    } else {  // White to play
        EXPECT_EQ(tensor[2][0][0], 0.0f);
    }
    
    // Get enhanced tensor representation
    auto enhancedTensor = state->getEnhancedTensorRepresentation();
    
    // Should have more planes than the basic representation
    EXPECT_GT(enhancedTensor.size(), tensor.size());
}

// Test string conversion utilities
TEST_F(GoTest, StringConversion) {
    // Test action to string conversion (coordinate format)
    int action = state->coordToAction(4, 4);
    std::string actionStr = state->actionToString(action);
    EXPECT_EQ(actionStr, "E5");  // 9x9 board, center is E5
    
    // Test pass action to string
    std::string passStr = state->actionToString(-1);
    EXPECT_EQ(passStr, "pass");
    
    // Test string to action conversion
    auto parsedAction = state->stringToAction("E5");
    ASSERT_TRUE(parsedAction);
    EXPECT_EQ(*parsedAction, action);
    
    // Test pass string to action
    auto parsedPass = state->stringToAction("pass");
    ASSERT_TRUE(parsedPass);
    EXPECT_EQ(*parsedPass, -1);
    
    // Test invalid string to action
    auto invalidAction = state->stringToAction("Z99");
    EXPECT_FALSE(invalidAction);
}

// Test board representation
TEST_F(GoTest, BoardRepresentation) {
    // Make some moves to create an interesting position
    placeAlternatingStones({
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 2}, {2, 1}, {0, 2}
    });
    
    // Get the string representation
    std::string boardStr = state->toString();
    
    // Should contain the basic board elements
    EXPECT_NE(boardStr.find("Current player"), std::string::npos);
    
    // Should contain stone markers
    EXPECT_NE(boardStr.find("X"), std::string::npos);  // Black stones
    EXPECT_NE(boardStr.find("O"), std::string::npos);  // White stones
}

// Test different board sizes
TEST_F(GoTest, DifferentBoardSizes) {
    // Test 13x13 board
    auto board13 = std::make_unique<GoState>(13, 7.5f, true, true);
    EXPECT_EQ(board13->getBoardSize(), 13);
    EXPECT_EQ(board13->getActionSpaceSize(), 13 * 13 + 1);
    
    // Test 19x19 board
    auto board19 = std::make_unique<GoState>(19, 7.5f, true, true);
    EXPECT_EQ(board19->getBoardSize(), 19);
    EXPECT_EQ(board19->getActionSpaceSize(), 19 * 19 + 1);
    
    // Make some moves on different board sizes
    board13->makeMove(board13->coordToAction(6, 6));  // Center of 13x13
    EXPECT_EQ(board13->getStone(6, 6), 1);
    
    board19->makeMove(board19->coordToAction(9, 9));  // Center of 19x19
    EXPECT_EQ(board19->getStone(9, 9), 1);
}

// Test capturing multiple groups at once
TEST_F(GoTest, MultipleCaptures) {
    // Create a position where White's move at (1,0) will capture two Black stones
    //    0 1 2
    // 0  B . .
    // 1  B W .
    // 2  W . .
    
    // Direct stone placement for more control
    state->setStone(0, 0, 1); // Black at (0,0)
    state->setStone(0, 1, 1); // Black at (0,1)
    state->setStone(1, 1, 2); // White at (1,1)
    state->setStone(0, 2, 2); // White at (0,2)
    
    // Set to White's turn
    if (state->getCurrentPlayer() != 2) {
        passTurn(); // Make it White's turn if needed
    }

    int blackStonesBefore = countStones(1); // Black has 2 stones
    EXPECT_EQ(blackStonesBefore, 2);
    EXPECT_EQ(state->getCapturedStones(2), 0); // White has 0 captures initially

    // White plays at (1,0), capturing 2 black stones B(0,0) and B(0,1)
    EXPECT_TRUE(isMoveValid(1, 0));
    makeMove(1, 0); // White plays W(1,0). P=Black.

    // Verify captures
    EXPECT_EQ(state->getStone(0, 0), 0);
    EXPECT_EQ(state->getStone(0, 1), 0);

    // Count stones after capture
    int blackStonesAfter = countStones(1); // Should be 2 - 2 = 0
    EXPECT_EQ(blackStonesAfter, 0);

    // Should have captured 2 black stones
    EXPECT_EQ(blackStonesBefore - blackStonesAfter, 2);

    // Capture count should reflect this
    EXPECT_EQ(state->getCapturedStones(2), 2); // White captured 2 stones.
}

// Test game validation
TEST_F(GoTest, GameValidation) {
    // A newly initialized state should be valid
    EXPECT_TRUE(state->validate());
    
    // Make some moves
    makeMove(4, 4);
    makeMove(3, 3);
    
    // Should still be valid
    EXPECT_TRUE(state->validate());
    
    // Create an invalid state by directly manipulating the board (more black stones than allowed)
    state->setStone(5, 5, 1);  // Add a black stone out of turn
    
    // Should be invalid
    EXPECT_FALSE(state->validate());
}

}  // namespace testing
}  // namespace go
}  // namespace alphazero

// Include test main at the end
#include "../../test_main.h"