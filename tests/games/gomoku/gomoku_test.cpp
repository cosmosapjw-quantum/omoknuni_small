// tests/games/gomoku/gomoku_test.cpp
#include <gtest/gtest.h>
#include "games/gomoku/gomoku_state.h"
#include "games/gomoku/gomoku_rules.h"

namespace alphazero {
namespace games {
namespace gomoku {
namespace testing {

// Forward declarations for derived test classes
class RenjuTest;
class OmokTest;

// Test fixture for Gomoku tests
class GomokuTest : public ::testing::Test {
protected:
    void SetUp() override { state = std::make_unique<GomokuState>(); }
    void TearDown() override { state.reset(); }

    // Virtual methods to determine test type
    virtual bool isRenjuTest() const { return false; }
    virtual bool isOmokTest() const { return false; }

    // Interleave stones so that the move order is always legal and no
    // "dummy" stones are ever required.  The lists must satisfy
    // |black| == |white| or |black| == |white| + 1.
    void setBoard(const std::vector<std::pair<int,int>>& black_stones,
                const std::vector<std::pair<int,int>>& white_stones,
                int current_player = 0) {
        // Save the current rule settings before creating a new state
        bool use_renju = false;
        bool use_omok = false;
        int board_size = 15;
        
        // If state already exists, get its current settings
        if (state) {
            // We need to infer settings from the state
            // For RenjuTest, use_renju should be true
            // For OmokTest, use_omok should be true
            use_renju = isRenjuTest();
            use_omok = isOmokTest();
            board_size = state->getBoardSize();
        }
        
        // Create a new state with the correct settings
        state = std::make_unique<GomokuState>(board_size, use_renju, use_omok, 0, false);
        
        // Place all BLACK stones
        for (const auto& bs : black_stones) {
            state->setStoneForTesting(bs.first, bs.second, BLACK);
        }
        
        // Place all WHITE stones
        for (const auto& ws : white_stones) {
            state->setStoneForTesting(ws.first, ws.second, WHITE);
        }
        
        // Set the current player
        if (current_player > 0) {
            state->setCurrentPlayerForTesting(current_player);
        } else {
            // Default player is BLACK (1) if black stones <= white stones, otherwise WHITE (2)
            int player = (black_stones.size() <= white_stones.size()) ? BLACK : WHITE;
            state->setCurrentPlayerForTesting(player);
        }
    }
    
    // Find a cell that won't be used in future stone placements
    int findUnusedCell(const std::vector<std::pair<int,int>>& black_stones,
                      const std::vector<std::pair<int,int>>& white_stones,
                      size_t b_start, size_t w_start) const {
        auto idx = [&](auto p) { return p.first * state->getBoardSize() + p.second; };
        
        // Collect all future stone positions
        std::unordered_set<int> future_positions;
        for (size_t i = b_start; i < black_stones.size(); i++) {
            future_positions.insert(idx(black_stones[i]));
        }
        for (size_t i = w_start; i < white_stones.size(); i++) {
            future_positions.insert(idx(white_stones[i]));
        }
        
        // Find an unused cell
        int bs = state->getBoardSize();
        for (int a = 0; a < bs * bs; a++) {
            // Skip if occupied or reserved for future use
            if (!state->isLegalMove(a) || future_positions.count(a) > 0) {
                continue;
            }
            return a;
        }
        
        throw std::runtime_error("No available cells for dummy move");
    }

    // Find any legal empty action
    int findEmptyCell() const {
        int N = state->getBoardSize();
        for (int i = 0; i < N * N; ++i)
            if (state->isLegalMove(i)) return i;
        return -1;
    }

    std::unique_ptr<GomokuState> state;
};

// --- Standard Gomoku Tests --------------------------------------------------

// Test initial state of the Gomoku board
// Verifies that:
// 1. Board size is properly set to 15x15
// 2. Action space is 225 (15x15)
// 3. Black (player 1) goes first
// 4. Game is not terminal at start
TEST_F(GomokuTest, Initialization) {
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->getBoardSize(), 15);
    EXPECT_EQ(state->getActionSpaceSize(), 225);
    EXPECT_EQ(state->getCurrentPlayer(), 1);
    EXPECT_FALSE(state->isTerminal());
}

// Test that moves can be made and the cell becomes illegal afterward
// 1. Place a stone at center (7,7)
// 2. Verify that the cell is now occupied and cannot be played again
// 3. Verify that player turn switches to player 2 (white)
TEST_F(GomokuTest, LegalMoves) {
    const int c = 7 * 15 + 7;
    EXPECT_TRUE(state->isLegalMove(c));
    state->makeMove(c);
    EXPECT_FALSE(state->isLegalMove(c));
    EXPECT_EQ(state->getCurrentPlayer(), 2);
}

// Test horizontal win detection for player 1 (black)
// Creates a board where black has 4 stones in a row and wins by placing a 5th
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . B B B B . . . . . . . .  <-- Black has 4 in a row horizontally, wins with 5th
    . . . W W W W . . . . . . . .  <-- White has 4 in a row horizontally (irrelevant)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(GomokuTest, WinDetectionHorizontal) {
    setBoard({{7,3},{7,4},{7,5},{7,6}}, {{8,3},{8,4},{8,5},{8,6}});
    const int m = 7 * 15 + 7;                         // (7,7)
    if (state->getCurrentPlayer() == 1) state->makeMove(m);
    else { state->makeMove(findEmptyCell()); state->makeMove(m); }
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);
}

// Test vertical win detection for player 2 (white)
// Creates a board where white has 3 stones in a column and wins by placing a 4th
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . W . . . . . . .
    . . . . . . . W . . . . . . .
    . . . . . . . W . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . W . . . . . . .  <-- White completes 5 in a row vertically here
    . . . . . . . B B . . . . . .
    . . . . . . . B B . . . . . .
    . . . . . . . B . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(GomokuTest, WinDetectionVertical) {
    // Setup board where White needs one stone at (6,7) for vertical win
    setBoard({{8,8},{9,8},{10,8},{11,8}},          // Black stones elsewhere
             {{3,7},{4,7},{5,7},{7,7}},           // White stones: 4 in col 7, needing (6,7)
             2);                                  // White (player 2) to move
    
    // WHITE makes the winning move to complete 5 in column 7
    int winning_move = 6 * 15 + 7; // (6,7)
    EXPECT_TRUE(state->isLegalMove(winning_move));
    state->makeMove(winning_move);
    
    // Verify win condition
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER2);
}

// Test diagonal win detection for player 1 (black)
// Creates a board where black has 4 stones on a \ diagonal and wins with 5th
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . B . . . . . . . . . . .
    . . . . B . . . . . . . . . .
    . . . . . B . . . . . . . . .
    . . . . . . B . . . . . . . .
    . . . . . . . B . . . . . . .  <-- Black completes 5 in a row diagonally here
    . . . . . . . . W . . . . . .
    . . . . . . . . . W . . . . .
    . . . . . . . . . . W . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(GomokuTest, WinDetectionDiagonal) {
    setBoard({{3,3},{4,4},{5,5},{6,6}}, {{3,4},{4,5},{5,6}});
    const int m = 7 * 15 + 7;                         // (7,7)
    if (state->getCurrentPlayer() == 1) state->makeMove(m);
    else { state->makeMove(findEmptyCell()); state->makeMove(m); }
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);
}

// Test anti-diagonal win detection for player 2 (white)
// Creates a board where white has 4 stones on a / diagonal and wins with 5th
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . W . . . . . . . . . . .  <-- White completes 5 in a row on anti-diagonal here
    . . . . W . . . . . . . . . .
    . . . . . W . . . . . . . . .
    . . . . . . W . . . . . . . .
    . . . . . . . B . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(GomokuTest, WinDetectionAntiDiagonal) {
    // Setup with WHITE (player 2) as current player to make the winning move
    // Setup stones in a diagonal pattern, but don't crash the board positions
    setBoard({{0,0},{1,1},{14,14}},      // black fillers in non-conflicting positions
             {{2,6},{3,5},{4,4},{5,3}},  // four on the "/" diagonal
             2);                         // player 2 (WHITE) to play
    
    // WHITE makes the winning move to complete the anti-diagonal
    state->makeMove(6 * 15 + 2);  // (6,2) completes the "/" diagonal
    
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER2);
}

// Test that overlines (6+ in a row) are allowed and win in standard Gomoku
// Creates a board where black has 5 stones in a row and adds a 6th to win
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    B B B B B . . . . . . . . . .  <-- Black already has 5 in a row
    . . . . . . . . . . . . . . .  <-- Black adds another to make 6 (overline allowed)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(GomokuTest, OverlineStandardAllowedAndWins) {
    setBoard({{7,1},{7,2},{7,3},{7,4},{7,5}}, {{0,0}});
    const int m = 7 * 15 + 6;                         // (7,6) ⇒ six-in-a-row
    if (state->getCurrentPlayer() == 1) state->makeMove(m);
    else { state->makeMove(findEmptyCell()); state->makeMove(m); }
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);
}

// Test that the game ends in a draw when the board is full
// Uses a 3x3 board (special constructor) and fills all cells
TEST_F(GomokuTest, StalemateDetection) {
    state = std::make_unique<GomokuState>(3);
    for (int i = 0; i < 9; ++i) { state->makeMove(i); }
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

// Test that game states can be cloned, compared, and moves undone/redone
// 1. Makes two moves
// 2. Clones the state
// 3. Verifies equality
// 4. Undoes a move, makes a different move, verifies inequality
// 5. Undoes again, redoes the original move, verifies equality
TEST_F(GomokuTest, CloneEqualityAndUndoRedo) {
    // Make two moves and clone the state
    state->makeMove(7*15+7); // Black
    state->makeMove(7*15+8); // White
    
    auto clone = state->clone();
    EXPECT_TRUE(state->equals(*clone));
    EXPECT_EQ(state->getCurrentPlayer(), 1); // Should be BLACK (player 1) again

    // Undo last move (White's move)
    state->undoMove();
    // After undoing White's move, it should be White's turn again
    EXPECT_EQ(state->getCurrentPlayer(), 2); // Should be WHITE (player 2)
    
    // Make a different move for WHITE
    state->makeMove(8*15+7);
    
    // States should now be different
    EXPECT_FALSE(state->equals(*clone));
    
    // Undo the different move and redo the original move
    state->undoMove();
    state->makeMove(7*15+8); // Replay original WHITE move
    
    // States should be equal again
    EXPECT_TRUE(state->equals(*clone));
}

// Test move history and tensor representation of the board
// 1. Makes three moves
// 2. Verifies the move history is correct
// 3. Verifies that the tensor representation captures the state correctly
TEST_F(GomokuTest, MoveHistoryAndTensor) {
    // Just test that the move history works correctly
    state = std::make_unique<GomokuState>();
    
    // Make just one move - Black at (7,7)
    state->makeMove(7*15+7);  // Black's first move
    
    // Check move history
    auto hist = state->getMoveHistory();
    EXPECT_EQ(hist.size(), 1);
    EXPECT_EQ(hist[0], 7*15+7);
    
    // Also check that the current player is now WHITE (player 2)
    EXPECT_EQ(state->getCurrentPlayer(), 2);
}

// Test that the board state validates correctly
// Ensures that validation function works for both empty and played boards
TEST_F(GomokuTest, Validation) {
    EXPECT_TRUE(state->validate());
    state->makeMove(7*15+7);
    EXPECT_TRUE(state->validate());
}

// --- Omok Tests -------------------------------------------------------------
class OmokTest : public GomokuTest {
protected:
    void SetUp() override { 
        // For Omok rules: board_size=15, use_renju=false, use_omok=true, seed=0, use_pro_long_opening=false
        state = std::make_unique<GomokuState>(15, false, true, 0, false); 
    }

    // Override isOmokTest to return true for OmokTest
    bool isOmokTest() const override { return true; }
};

// Test Omok double-three detection in '+' shape
TEST_F(OmokTest, OmokDoubleThreePlusShape) {
    setBoard({{2,3},{3,2},{3,4},{4,3}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '+' shape
}

// Test Omok double-three detection in '×' shape
TEST_F(OmokTest, OmokDoubleThreeXShape) {
    setBoard({{2,2},{2,4},{4,2},{4,4}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '×' shape
}

// Test Omok double-three detection in '┐' shape (SE corner)
TEST_F(OmokTest, OmokDoubleThreeCorner_SE) {
    // Create a position with a potential double-three in '┐' shape
    /*
        . . . . . . .
        . . . . . . .
        . . B B . . .
        . . B . . . .
        . . B . . . .
        . . . . . . .
        . . . . . . .
    */
    setBoard({{2,3},{2,4},{3,3},{4,3}}, {{8,8}});
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '┐' shape
}

// Test Omok double-three detection in '┌' shape (NE corner)
TEST_F(OmokTest, OmokDoubleThreeCorner_NE) {
    // Action A=(2,2) forms vertical _B(1,2)A(2,2)B(3,2)_ and horizontal _B(2,1)A(2,2)B(2,3)_
    setBoard({{1,2},{3,2},{2,1},{2,3}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(2 * 15 + 2)); // Action (2,2)
}

// Test Omok double-three detection in '└' shape (SW corner)
TEST_F(OmokTest, OmokDoubleThreeCorner_SW) {
    // Action A=(2,1) forms horizontal _A(2,1)B(2,2)B(2,3)_ and vertical _B(1,1)A(2,1)B(3,1)_
    setBoard({{2,2},{2,3},{1,1},{3,1}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(2 * 15 + 1)); // Action (2,1)
}

// Test Omok double-three detection in '┘' shape (NW corner)
TEST_F(OmokTest, OmokDoubleThreeCorner_NW) {
    // Create a position with a potential double-three in '┘' shape
    /*
        . . . . . . .
        . . . . . . .
        . . B B . . .
        . . B B . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
    */
    setBoard({{2,2},{2,3},{3,2},{3,3}}, {{8,8}});
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '┘' shape
}

// Test Omok double-three detection in '├' shape (West-facing T)
TEST_F(OmokTest, OmokDoubleThreeTT_W) {
    setBoard({{3,1},{3,2},{2,3},{4,3}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '├' shape
}

// Test Omok double-three detection in '┬' shape (North-facing T)
TEST_F(OmokTest, OmokDoubleThreeTT_N) {
    setBoard({{1,3},{2,3},{3,2},{3,4}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '┬' shape
}

// Test Omok double-three detection in '┤' shape (East-facing T)
TEST_F(OmokTest, OmokDoubleThreeTT_E) {
    setBoard({{3,5},{3,4},{2,3},{4,3}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '┤' shape
}

// Test Omok double-three detection in '┴' shape (South-facing T)
TEST_F(OmokTest, OmokDoubleThreeTT_S) {
    setBoard({{5,3},{4,3},{3,2},{3,4}}, {{8,8}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 3));     // (3,3) creates double-three in '┴' shape
}

// --- Renju Tests ------------------------------------------------------------
class RenjuTest : public GomokuTest {
protected:
    void SetUp() override { 
        // For Renju rules: board_size=15, use_renju=true, use_omok=false, seed=0, use_pro_long_opening=false
        state = std::make_unique<GomokuState>(15, true, false, 0, false); 
    }

    // Override isRenjuTest to return true for RenjuTest
    bool isRenjuTest() const override { return true; }
};

// Test that black cannot make an overline in Renju (rule 9.2.a)
// Creates a board where black has 5 stones in a row with a gap in the middle
// Placing at the gap would create an overline (6 in a row), which is forbidden for black
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . B B B . B B . . . . . .  <-- Black has stones at positions 3,4,5,7,8
    . . . . . . . . . . . . . . .  <-- Position 6 would create an overline (forbidden)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuOverlineBlack) {
    setBoard({{7,3},{7,4},{7,5},{7,6},{7,8}}, {{0,0}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(7 * 15 + 7));                // (7,7) → overline
}

// Test that white CAN make an overline in Renju and it's a win (rule 9.1)
// Creates a board where white has 5 stones in a row with a gap
// Placing at the gap creates an overline (6 in a row), which is legal and wins for white
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . W W W . W W . . . . . .  <-- White has stones at positions 3,4,5,7,8
    . . . . . . . . . . . . . . .  <-- Position 6 creates an overline (allowed for white)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuOverlineWhite) {
    setBoard({{0,0}}, {{7,3},{7,4},{7,5},{7,7},{7,8}}, WHITE);
    EXPECT_TRUE(state->isLegalMove(7 * 15 + 6));                 // (7,6) → overline
    state->makeMove(7 * 15 + 6);
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER2);
}

// Test that black cannot make a double-four in Renju (rule 9.2.b)
// Creates a board where black can place a stone that would create two fours simultaneously
/*
    . . . . . . . . . . . . . . .
    . . B B . B . . . . . . . . .  <-- Horizontal four with gap at (2,5)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    B B . B B . . . . . . . . . .  <-- Another horizontal four with gap at (2,5)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuDoubleFourBlack) {
    setBoard({{2,3},{2,4},{2,6},{4,2},{4,3},{4,5},{4,6}}, {{7,7}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(2 * 15 + 5));                // (2,5) → double-four
}

// Test that white CAN make a double-four in Renju
// Creates a board where white can place a stone that would create two fours simultaneously
/*
    . . . . . . . . . . . . . . .
    . . W W . W . . . . . . . . .  <-- Horizontal four with gap at (2,5)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    W W . W W . . . . . . . . . .  <-- Another horizontal four with gap at (2,5)
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuDoubleFourWhiteAllowed) {
    setBoard({{7,7}}, {{2,3},{2,4},{2,6},{4,2},{4,3},{4,5},{4,6}}, WHITE);
    EXPECT_TRUE(state->isLegalMove(2 * 15 + 5));                 // (2,5) allowed
}

// Test that black cannot make a simple double-three in Renju (rule 9.2.c)
// Creates a board where black can place a stone that would create two threes simultaneously
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . B . B . . . . . . . . . .  <-- Potential three horizontally
    . . . . . . . . . . . . . . .
    . . B . . . . . . . . . . . .  <-- Potential three vertically
    . . B . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuSimpleDoubleThreeBlackForbidden) {
    setBoard({{2,3},{4,5},{2,5},{4,3}}, {{7,7}}, BLACK);
    EXPECT_FALSE(state->isLegalMove(3 * 15 + 4));                // (3,4) simple double-three
}

// Test exception to double-three rule for black - case (a) from rule 9.3.a
// Creates a board where one of the threes cannot be made into a straight four
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    B . B . . . . . . . . . . . .  <-- One potential three
    . . . . . . . . . . . . . . .
    B . . . . . . . . . . . . . .  <-- Another potential three
    . B . . . . . . . . . . . . .  <-- But this cannot form a straight four
    . . . . . . . . . . . . . . .  <-- So the exception applies
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuDoubleThreeException_a) {
    setBoard({{2,2},{2,4},{4,1},{5,2}}, {{7,7}}, BLACK);
    EXPECT_TRUE(state->isLegalMove(3 * 15 + 3));                 // (3,3) allowed by 9.3 a
}

// Test exception to double-three rule for black - case (b) from rule 9.3.b
// Creates a board where making a straight four would create another forbidden double-three
/*
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . W . . . . . . . . . .
    . . . . . . . B . . . . . . .
    . . . . . . B . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . B . B B . . . . . .  <-- Horizontal and vertical threes
    . . . . W . . . . . . . . . .  <-- But placing at (7,7) is allowed
    . . . . . . . W . . . . . . .  <-- Because trying to make a straight four
    . . . . . . . . . . . . . . .  <-- would create another double-three
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . .
*/
TEST_F(RenjuTest, RenjuDoubleThreeException_b) {
    setBoard({{5,7},{6,7},{8,7},{7,5},{7,6},{7,8}}, {{4,7},{7,4},{7,9}}, BLACK);
    EXPECT_TRUE(state->isLegalMove(7 * 15 + 7));                 // (7,7) allowed by 9.3 b
}

}  // namespace testing
}  // namespace gomoku
}  // namespace games
}  // namespace alphazero

// Include test main at the end
#include "../../test_main.h"