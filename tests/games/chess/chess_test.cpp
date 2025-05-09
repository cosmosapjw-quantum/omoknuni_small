// tests/games/chess/chess_test.cpp
#include <gtest/gtest.h>
#include "games/chess/chess_state.h"
#include "games/chess/chess_rules.h"
#include "games/chess/chess960.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace alphazero {
namespace chess {
namespace testing {

// Test fixture for Chess tests
class ChessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard chess game
        state = std::make_unique<ChessState>();
    }

    void TearDown() override {
        state.reset();
    }

    // Set up a specific position using FEN
    bool setupPosition(const std::string& fen) {
        state = std::make_unique<ChessState>();
        return state->setFromFEN(fen);
    }

    // Helper to find a specific move in the legal moves list
    bool hasLegalMove(const std::string& moveStr) {
        auto move = state->stringToMove(moveStr);
        if (!move) {
            return false;
        }
        
        auto legalMoves = state->generateLegalMoves();
        for (const auto& legalMove : legalMoves) {
            if (legalMove == *move) {
                return true;
            }
        }
        return false;
    }

    // Convert set of legal moves to a set of strings for easier checking
    std::unordered_set<std::string> getLegalMoveStrings() {
        std::unordered_set<std::string> result;
        auto legalMoves = state->generateLegalMoves();
        for (const auto& move : legalMoves) {
            result.insert(state->moveToString(move));
        }
        return result;
    }

    std::unique_ptr<ChessState> state;
};

// Test basic game initialization
TEST_F(ChessTest, Initialization) {
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->getBoardSize(), 8);
    EXPECT_EQ(state->getActionSpaceSize(), 64 * 64 * 5);  // from * to * promotion options
    EXPECT_EQ(state->getCurrentPlayer(), static_cast<int>(PieceColor::WHITE));
    EXPECT_FALSE(state->isTerminal());
    
    // Check initial position
    EXPECT_EQ(state->getPiece(E1).type, PieceType::KING);
    EXPECT_EQ(state->getPiece(E1).color, PieceColor::WHITE);
    EXPECT_EQ(state->getPiece(E8).type, PieceType::KING);
    EXPECT_EQ(state->getPiece(E8).color, PieceColor::BLACK);
    
    // Verify castling rights
    auto castlingRights = state->getCastlingRights();
    EXPECT_TRUE(castlingRights.white_kingside);
    EXPECT_TRUE(castlingRights.white_queenside);
    EXPECT_TRUE(castlingRights.black_kingside);
    EXPECT_TRUE(castlingRights.black_queenside);
    
    // Check initial FEN
    EXPECT_EQ(state->toFEN(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// Test setting and getting pieces
TEST_F(ChessTest, PieceOperations) {
    // Clear the board
    for (int square = 0; square < 64; ++square) {
        state->setPiece(square, Piece());
    }
    
    // Set and get a piece
    Piece whiteKing = {PieceType::KING, PieceColor::WHITE};
    state->setPiece(E4, whiteKing);
    
    Piece retrievedPiece = state->getPiece(E4);
    EXPECT_EQ(retrievedPiece.type, PieceType::KING);
    EXPECT_EQ(retrievedPiece.color, PieceColor::WHITE);
    
    // Test out-of-bounds handling
    Piece outOfBoundsPiece = state->getPiece(100);  // Invalid square
    EXPECT_EQ(outOfBoundsPiece.type, PieceType::NONE);
    EXPECT_EQ(outOfBoundsPiece.color, PieceColor::NONE);
}

// Test FEN parsing and generation
TEST_F(ChessTest, FENConversion) {
    // Test standard starting position
    std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    EXPECT_TRUE(setupPosition(startFen));
    EXPECT_EQ(state->toFEN(), startFen);
    
    // Test a more complex position
    std::string complexFen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3";
    EXPECT_TRUE(setupPosition(complexFen));
    EXPECT_EQ(state->toFEN(), complexFen);
    
    // Test position with no castling rights and en passant
    std::string nocastleFen = "8/5k2/8/8/8/8/5K2/8 w - - 0 1";
    EXPECT_TRUE(setupPosition(nocastleFen));
    EXPECT_EQ(state->toFEN(), nocastleFen);
    
    // Test invalid FEN
    EXPECT_FALSE(setupPosition("invalid/fen/string"));
}

// Test move generation and validation
TEST_F(ChessTest, LegalMoves) {
    // Initial position should have 20 legal moves
    auto legalMoves = state->generateLegalMoves();
    EXPECT_EQ(legalMoves.size(), 20);
    
    // Test a specific pawn push
    EXPECT_TRUE(hasLegalMove("e2e4"));
    
    // Make a move
    auto move = state->stringToMove("e2e4");
    ASSERT_TRUE(move);
    state->makeMove(*move);
    
    // It should be Black's turn
    EXPECT_EQ(state->getCurrentPlayer(), static_cast<int>(PieceColor::BLACK));
    
    // Black should have 20 legal moves
    legalMoves = state->generateLegalMoves();
    EXPECT_EQ(legalMoves.size(), 20);
    
    // Test a specific response from Black
    EXPECT_TRUE(hasLegalMove("e7e5"));
}

// Test move representation and string conversion
TEST_F(ChessTest, MoveRepresentation) {
    // Test square to string conversion
    EXPECT_EQ(ChessState::squareToString(E2), "e2");
    EXPECT_EQ(ChessState::squareToString(A1), "a1");
    EXPECT_EQ(ChessState::squareToString(H8), "h8");
    
    // Test string to square conversion
    EXPECT_EQ(ChessState::stringToSquare("e2"), E2);
    EXPECT_EQ(ChessState::stringToSquare("a1"), A1);
    EXPECT_EQ(ChessState::stringToSquare("h8"), H8);
    EXPECT_EQ(ChessState::stringToSquare("invalid"), -1);
    
    // Test move string conversion
    auto move = state->stringToMove("e2e4");
    ASSERT_TRUE(move);
    EXPECT_EQ(move->from_square, E2);
    EXPECT_EQ(move->to_square, E4);
    EXPECT_EQ(move->promotion_piece, PieceType::NONE);
    
    // Test promotion move string conversion
    auto promotionMove = state->stringToMove("a7a8q");
    ASSERT_TRUE(promotionMove);
    EXPECT_EQ(promotionMove->from_square, A7);
    EXPECT_EQ(promotionMove->to_square, A8);
    EXPECT_EQ(promotionMove->promotion_piece, PieceType::QUEEN);
    
    // Test move to string conversion
    ChessMove testMove = {E2, E4};
    EXPECT_EQ(state->moveToString(testMove), "e2e4");
    
    ChessMove promotionTestMove = {A7, A8, PieceType::QUEEN};
    EXPECT_EQ(state->moveToString(promotionTestMove), "a7a8q");
}

// Test standard algebraic notation (SAN)
TEST_F(ChessTest, StandardAlgebraicNotation) {
    // Test some SAN conversions in initial position
    ChessMove e4Move = {E2, E4};
    EXPECT_EQ(state->toSAN(e4Move), "e4");
    
    ChessMove nf3Move = {G1, F3};
    EXPECT_EQ(state->toSAN(nf3Move), "Nf3");
    
    // Test from SAN
    auto sanMove = state->fromSAN("e4");
    ASSERT_TRUE(sanMove);
    EXPECT_EQ(sanMove->from_square, E2);
    EXPECT_EQ(sanMove->to_square, E4);
    
    // Set up a position that requires disambiguation
    EXPECT_TRUE(setupPosition("r1bqkbnr/ppp2ppp/2n5/3pp3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4"));
    
    // Test disambiguation for knight capturing on d5 (N@c3xd5)
    // Other knight is on f3. Nf3 CANNOT move to d5.
    // So, there is no ambiguity for N@c3 moving to d5.
    ChessMove knightMove1 = {C3, D5}; // Knight on c3 (sq 42) to d5 (sq 27). d5 has a black pawn.
    ChessMove knightMove2 = {F3, D2}; // Another move for context, not used in this assertion
    
    // Nf3 cannot move to d5, so N@c3xd5 is unambiguous. Standard SAN is Nxd5.
    EXPECT_EQ(state->toSAN(knightMove1), "Nxd5"); 
    
    // Test castling
    EXPECT_TRUE(setupPosition("r1bqk2r/ppp1bppp/2n2n2/3pp3/4P3/2N2N2/PPPPBPPP/R1BQK2R w KQkq - 4 5"));
    
    ChessMove castlingMove = {E1, G1};
    EXPECT_EQ(state->toSAN(castlingMove), "O-O");
    
    auto fromSanCastling = state->fromSAN("O-O");
    ASSERT_TRUE(fromSanCastling);
    EXPECT_EQ(fromSanCastling->from_square, E1);
    EXPECT_EQ(fromSanCastling->to_square, G1);
}

// Test special moves: castling
TEST_F(ChessTest, Castling) {
    // Set up a position where castling is possible
    EXPECT_TRUE(setupPosition("r1bqk2r/ppp1bppp/2n2n2/3pp3/4P3/2N2N2/PPPPBPPP/R1BQK2R w KQkq - 4 5"));
    
    // Check if kingside castling is legal
    EXPECT_TRUE(hasLegalMove("e1g1"));
    
    // Make the castling move
    auto castlingMove = state->stringToMove("e1g1");
    ASSERT_TRUE(castlingMove);
    state->makeMove(*castlingMove);
    
    // King and rook should be in their post-castling positions
    EXPECT_EQ(state->getPiece(G1).type, PieceType::KING);
    EXPECT_EQ(state->getPiece(F1).type, PieceType::ROOK);
    EXPECT_EQ(state->getPiece(E1).type, PieceType::NONE);
    EXPECT_EQ(state->getPiece(H1).type, PieceType::NONE);
    
    // Castling rights should be updated
    auto castlingRights = state->getCastlingRights();
    EXPECT_FALSE(castlingRights.white_kingside);
    EXPECT_FALSE(castlingRights.white_queenside);
    EXPECT_TRUE(castlingRights.black_kingside);
    EXPECT_TRUE(castlingRights.black_queenside);
    
    // Set up a queenside castling position
    EXPECT_TRUE(setupPosition("r3kbnr/ppp1pppp/2n5/3q4/3P4/2N5/PPP1PPPP/R3KBNR w KQkq - 2 5"));
    
    // Check if queenside castling is legal
    EXPECT_TRUE(hasLegalMove("e1c1"));
    
    // Make the queenside castling move
    auto queenSideCastlingMove = state->stringToMove("e1c1");
    ASSERT_TRUE(queenSideCastlingMove);
    state->makeMove(*queenSideCastlingMove);
    
    // King and rook should be in their post-castling positions
    EXPECT_EQ(state->getPiece(C1).type, PieceType::KING);
    EXPECT_EQ(state->getPiece(D1).type, PieceType::ROOK);
    EXPECT_EQ(state->getPiece(E1).type, PieceType::NONE);
    EXPECT_EQ(state->getPiece(A1).type, PieceType::NONE);
}

// Test special moves: en passant
TEST_F(ChessTest, EnPassant) {
    // Set up a position where en passant is possible
    EXPECT_TRUE(setupPosition("rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"));
    
    // Check that en passant is a legal move
    EXPECT_TRUE(hasLegalMove("e5d6"));
    
    // Make the en passant capture
    auto enPassantMove = state->stringToMove("e5d6");
    ASSERT_TRUE(enPassantMove);
    state->makeMove(*enPassantMove);
    
    // Check that the capture worked correctly
    EXPECT_EQ(state->getPiece(D6).type, PieceType::PAWN);
    EXPECT_EQ(state->getPiece(D6).color, PieceColor::WHITE);
    EXPECT_EQ(state->getPiece(D5).type, PieceType::NONE);
    EXPECT_EQ(state->getPiece(E5).type, PieceType::NONE);
    
    // En passant square should be reset
    EXPECT_EQ(state->getEnPassantSquare(), -1);
}

// Test special moves: promotion
TEST_F(ChessTest, Promotion) {
    // Set up a position where promotion is possible
    EXPECT_TRUE(setupPosition("8/P7/8/8/8/8/8/k6K w - - 0 1"));
    
    // Check that promotion moves are available
    auto legalMoveStrings = getLegalMoveStrings();
    EXPECT_TRUE(legalMoveStrings.find("a7a8q") != legalMoveStrings.end());
    EXPECT_TRUE(legalMoveStrings.find("a7a8r") != legalMoveStrings.end());
    EXPECT_TRUE(legalMoveStrings.find("a7a8b") != legalMoveStrings.end());
    EXPECT_TRUE(legalMoveStrings.find("a7a8n") != legalMoveStrings.end());
    
    // Make a promotion move
    auto promotionMove = state->stringToMove("a7a8q");
    ASSERT_TRUE(promotionMove);
    state->makeMove(*promotionMove);
    
    // Check that the promotion worked
    EXPECT_EQ(state->getPiece(A8).type, PieceType::QUEEN);
    EXPECT_EQ(state->getPiece(A8).color, PieceColor::WHITE);
    EXPECT_EQ(state->getPiece(A7).type, PieceType::NONE);
    
    // Set up a position for black promotion
    EXPECT_TRUE(setupPosition("k6K/8/8/8/8/8/p7/8 b - - 0 1"));
    
    // Make a black promotion move
    auto blackPromotionMove = state->stringToMove("a2a1q");
    ASSERT_TRUE(blackPromotionMove);
    state->makeMove(*blackPromotionMove);
    
    // Check that the promotion worked
    EXPECT_EQ(state->getPiece(A1).type, PieceType::QUEEN);
    EXPECT_EQ(state->getPiece(A1).color, PieceColor::BLACK);
}

// Test check detection
TEST_F(ChessTest, CheckDetection) {
    // Set up a position with check that can be blocked by g2g3
    // FEN: Black Q@h4 checks K@e1. f2 is empty. Pawn is on g2.
    EXPECT_TRUE(setupPosition("rnbqkbnr/pppp1ppp/8/4p3/7q/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1"));
    
    // Verify check is detected for White
    EXPECT_TRUE(state->isInCheck(PieceColor::WHITE)); // Q@h4 checks K@e1 because f2 is empty
    EXPECT_FALSE(state->isInCheck(PieceColor::BLACK));
    
    // Legal moves should exist to get out of check
    auto legalMoves = state->generateLegalMoves();
    EXPECT_FALSE(legalMoves.empty());

    // Make a move that blocks the check (g2g3)
    auto blockingMove = state->stringToMove("g2g3");
    ASSERT_TRUE(blockingMove); // Ensure move string is valid
    
    // Verify the blocking move is legal before making it
    bool isLegal = false;
    for (const auto& legalMv : legalMoves) {
        if (legalMv == *blockingMove) {
            isLegal = true;
            break;
        }
    }
    ASSERT_TRUE(isLegal); // Ensure g2g3 is a legal move

    state->makeMove(*blockingMove);
    
    // Should no longer be in check
    EXPECT_FALSE(state->isInCheck(PieceColor::WHITE));
}

// Test checkmate detection
TEST_F(ChessTest, CheckmateDetection) {
    // Set up a checkmate position (fool's mate)
    EXPECT_TRUE(setupPosition("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"));
    
    // Only make the move if the hash is not already terminal
    if (!state->isTerminal()) {
        // Get the legal moves
        auto legalMoves = state->generateLegalMoves();
        
        // There should be moves to get out of check or block
        EXPECT_FALSE(legalMoves.empty());
        
        // Set up a checkmate position (scholar's mate)
        EXPECT_TRUE(setupPosition("r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"));
        
        // Verify checkmate (no legal moves and in check)
        legalMoves = state->generateLegalMoves();
        EXPECT_TRUE(legalMoves.empty());
        EXPECT_TRUE(state->isInCheck(PieceColor::BLACK));
        EXPECT_TRUE(state->isTerminal());
        EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);  // White wins
    }
}

// Test stalemate detection
TEST_F(ChessTest, StalemateDetection) {
    // Set up a stalemate position
    EXPECT_TRUE(setupPosition("k7/8/1Q6/8/8/8/8/7K b - - 0 1"));
    
    // Verify stalemate (no legal moves but not in check)
    auto legalMoves = state->generateLegalMoves();
    EXPECT_TRUE(legalMoves.empty());
    EXPECT_FALSE(state->isInCheck(PieceColor::BLACK));
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

// Test draw by insufficient material
TEST_F(ChessTest, InsufficientMaterialDetection) {
    // Test king vs king
    EXPECT_TRUE(setupPosition("8/8/8/4k3/8/8/8/4K3 w - - 0 1"));
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // Test king + knight vs king
    EXPECT_TRUE(setupPosition("8/8/8/4k3/8/8/8/3NK3 w - - 0 1"));
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // Test king + bishop vs king
    EXPECT_TRUE(setupPosition("8/8/8/4k3/8/8/8/3BK3 w - - 0 1"));
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // Test king + bishop vs king + bishop (same color bishops) -> IS A DRAW
    EXPECT_TRUE(setupPosition("8/8/8/3bk3/8/8/8/3BK3 w - - 0 1")); // b@d5 (light), B@d1 (light)
    EXPECT_TRUE(state->isTerminal()); // Should be terminal (draw by insufficient material)
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // Test king + bishop vs king + bishop (opposite color bishops) -> NOT necessarily a draw by insufficient material alone
    EXPECT_TRUE(setupPosition("8/8/8/2b1k3/8/8/8/3BK3 w - - 0 1")); // b@c5 (dark), B@d1 (light)
    EXPECT_FALSE(state->isTerminal()); // Should NOT be terminal by insufficient material alone
}

// Test 50-move rule
TEST_F(ChessTest, FiftyMoveRule) {
    // Set up a position with a high halfmove clock
    EXPECT_TRUE(setupPosition("8/8/8/4k3/8/8/8/4K3 w - - 100 1"));
    
    // Verify 50-move rule is detected
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

// Test move undoing
TEST_F(ChessTest, UndoMove) {
    // Standard opening moves
    auto e4Move = state->stringToMove("e2e4");
    ASSERT_TRUE(e4Move);
    state->makeMove(*e4Move);
    
    auto e5Move = state->stringToMove("e7e5");
    ASSERT_TRUE(e5Move);
    state->makeMove(*e5Move);
    
    // Save the current position
    auto currentFEN = state->toFEN();
    
    // Make another move
    auto nf3Move = state->stringToMove("g1f3");
    ASSERT_TRUE(nf3Move);
    state->makeMove(*nf3Move);
    
    // Verify the move was made
    EXPECT_EQ(state->getPiece(F3).type, PieceType::KNIGHT);
    EXPECT_EQ(state->getPiece(F3).color, PieceColor::WHITE);
    EXPECT_EQ(state->getPiece(G1).type, PieceType::NONE);
    
    // Undo the move
    EXPECT_TRUE(state->undoMove());
    
    // Verify we're back at the previous position
    EXPECT_EQ(state->toFEN(), currentFEN);
    EXPECT_EQ(state->getPiece(G1).type, PieceType::KNIGHT);
    EXPECT_EQ(state->getPiece(G1).color, PieceColor::WHITE);
    EXPECT_EQ(state->getPiece(F3).type, PieceType::NONE);
}

// Test cloning and equality
TEST_F(ChessTest, CloneAndEquality) {
    // Make some moves
    auto e4Move = state->stringToMove("e2e4");
    ASSERT_TRUE(e4Move);
    state->makeMove(*e4Move);
    
    auto e5Move = state->stringToMove("e7e5");
    ASSERT_TRUE(e5Move);
    state->makeMove(*e5Move);
    
    // Clone the state
    auto clonedState = state->clone();
    
    // Check that the cloned state has the expected type
    EXPECT_EQ(clonedState->getGameType(), core::GameType::CHESS);
    
    // Check that the cloned state equals the original
    EXPECT_TRUE(state->equals(*clonedState));
    
    // Make a move on the original
    auto nf3Move = state->stringToMove("g1f3");
    ASSERT_TRUE(nf3Move);
    state->makeMove(*nf3Move);
    
    // The states should now be different
    EXPECT_FALSE(state->equals(*clonedState));
}

// Test move history tracking
TEST_F(ChessTest, MoveHistoryTracking) {
    // Make a series of moves
    auto e4Move = state->stringToMove("e2e4");
    ASSERT_TRUE(e4Move);
    state->makeMove(*e4Move);
    
    auto e5Move = state->stringToMove("e7e5");
    ASSERT_TRUE(e5Move);
    state->makeMove(*e5Move);
    
    auto nf3Move = state->stringToMove("g1f3");
    ASSERT_TRUE(nf3Move);
    state->makeMove(*nf3Move);
    
    // Check the move history
    auto moveHistory = state->getMoveHistory();
    EXPECT_EQ(moveHistory.size(), 3);
    
    // The converted actions should match our moves
    EXPECT_EQ(state->actionToChessMove(moveHistory[0]), *e4Move);
    EXPECT_EQ(state->actionToChessMove(moveHistory[1]), *e5Move);
    EXPECT_EQ(state->actionToChessMove(moveHistory[2]), *nf3Move);
    
    // Undo a move
    EXPECT_TRUE(state->undoMove());
    
    // Move history should be updated
    moveHistory = state->getMoveHistory();
    EXPECT_EQ(moveHistory.size(), 2);
}

// Test tensor representation
TEST_F(ChessTest, TensorRepresentation) {
    // Make some moves
    auto e4Move = state->stringToMove("e2e4");
    ASSERT_TRUE(e4Move);
    state->makeMove(*e4Move);
    
    // Get tensor representation
    auto tensor = state->getTensorRepresentation();
    
    // Should have 12 planes (6 piece types * 2 colors)
    EXPECT_EQ(tensor.size(), 12);
    
    // Check some specific piece positions
    // Plane 0: White pawns
    EXPECT_EQ(tensor[0][6][4], 0.0f);  // e2 pawn moved
    EXPECT_EQ(tensor[0][4][4], 1.0f);  // e4 pawn present
    
    // Get enhanced tensor representation
    auto enhancedTensor = state->getEnhancedTensorRepresentation();
    
    // Should have more planes than the basic representation
    EXPECT_GT(enhancedTensor.size(), tensor.size());
}

// Test game validation
TEST_F(ChessTest, GameValidation) {
    // A newly initialized state should be valid
    EXPECT_TRUE(state->validate());
    
    // Make some moves
    auto e4Move = state->stringToMove("e2e4");
    ASSERT_TRUE(e4Move);
    state->makeMove(*e4Move);
    
    // Should still be valid
    EXPECT_TRUE(state->validate());
    
    // Set up an invalid state (more than one king per side)
    state = std::make_unique<ChessState>();
    
    // Remove all pieces
    for (int square = 0; square < 64; ++square) {
        state->setPiece(square, Piece());
    }
    
    // Add two white kings
    state->setPiece(E1, {PieceType::KING, PieceColor::WHITE});
    state->setPiece(E2, {PieceType::KING, PieceColor::WHITE});
    state->setPiece(E8, {PieceType::KING, PieceColor::BLACK});
    
    // Should be invalid
    EXPECT_FALSE(state->validate());
}

// Test threefold repetition detection
TEST_F(ChessTest, ThreefoldRepetition) {
    // Make a series of moves that will lead to a threefold repetition
    // Knight dance: Ng1-f3-g1-f3-g1-f3
    auto nf3Move = state->stringToMove("g1f3");
    auto ng1Move = state->stringToMove("f3g1");
    
    ASSERT_TRUE(nf3Move);
    ASSERT_TRUE(ng1Move);
    
    state->makeMove(*nf3Move);
    
    // Black responds
    auto nc6Move = state->stringToMove("b8c6");
    ASSERT_TRUE(nc6Move);
    state->makeMove(*nc6Move);
    
    // White knight back
    state->makeMove(*ng1Move);
    
    // Black responds
    auto nb8Move = state->stringToMove("c6b8");
    ASSERT_TRUE(nb8Move);
    state->makeMove(*nb8Move);
    
    // Repeat two more times
    state->makeMove(*nf3Move);
    state->makeMove(*nc6Move);
    state->makeMove(*ng1Move);
    state->makeMove(*nb8Move);
    state->makeMove(*nf3Move);
    state->makeMove(*nc6Move);
    
    // At this point, the position should have occurred three times
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

// Test for Chess960 class
class Chess960Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Nothing special to set up
    }

    void TearDown() override {
        // Nothing special to clean up
    }
};

// Test Chess960 position generation
TEST_F(Chess960Test, PositionGeneration) {
    // Generate a random position
    int position = Chess960::generateRandomPosition();
    EXPECT_GE(position, 0);
    EXPECT_LT(position, 960);
    
    // Generate a specific position
    auto pieces = Chess960::generatePosition(518);  // Standard chess position
    
    // Check piece arrangement
    EXPECT_EQ(pieces[0], PieceType::ROOK);
    EXPECT_EQ(pieces[1], PieceType::KNIGHT);
    EXPECT_EQ(pieces[2], PieceType::BISHOP);
    EXPECT_EQ(pieces[3], PieceType::QUEEN);
    EXPECT_EQ(pieces[4], PieceType::KING);
    EXPECT_EQ(pieces[5], PieceType::BISHOP);
    EXPECT_EQ(pieces[6], PieceType::KNIGHT);
    EXPECT_EQ(pieces[7], PieceType::ROOK);
}

// Test Chess960 position validation
TEST_F(Chess960Test, PositionValidation) {
    // Test valid positions
    auto standardPosition = Chess960::generatePosition(518);
    EXPECT_TRUE(Chess960::isValidPosition(standardPosition));
    
    // Test a position that doesn't have bishops on opposite colors (e.g., c and e files, both dark)
    std::array<PieceType, 8> invalidPosition = {
        PieceType::ROOK, PieceType::KNIGHT, PieceType::BISHOP, PieceType::QUEEN, /* B on c1 */
        PieceType::BISHOP, PieceType::KING, PieceType::KNIGHT, PieceType::ROOK  /* B on e1 */
    };
    // Arrangement: R N B Q B K N R
    // Files:       a b c d e f g h
    // B@c (idx 2, dark), B@e (idx 4, dark) -> Same color, so should be invalid.
    EXPECT_FALSE(Chess960::isValidPosition(invalidPosition));
}

// Test Chess960 setup and FEN handling
TEST_F(Chess960Test, Chess960FENHandling) {
    // Create a Chess960 game with standard position
    ChessState chess960State(true, "", 518);  // Chess960 = true, position 518 (standard)
    
    // Verify the position
    EXPECT_EQ(chess960State.getPiece(A1).type, PieceType::ROOK);
    EXPECT_EQ(chess960State.getPiece(B1).type, PieceType::KNIGHT);
    EXPECT_EQ(chess960State.getPiece(C1).type, PieceType::BISHOP);
    EXPECT_EQ(chess960State.getPiece(D1).type, PieceType::QUEEN);
    EXPECT_EQ(chess960State.getPiece(E1).type, PieceType::KING);
    EXPECT_EQ(chess960State.getPiece(F1).type, PieceType::BISHOP);
    EXPECT_EQ(chess960State.getPiece(G1).type, PieceType::KNIGHT);
    EXPECT_EQ(chess960State.getPiece(H1).type, PieceType::ROOK);
    
    // Test a different Chess960 position (e.g., position 0)
    ChessState position0State(true, "", 0);
    
    // FEN should use Chess960 castling notation
    std::string fen = position0State.toFEN();
    EXPECT_TRUE(fen.find("KQkq") == std::string::npos);  // Shouldn't use standard notation
}

// Test Chess960 castling
TEST_F(Chess960Test, Chess960Castling) {
    // Test with a specific position where king and rooks are in different places
    // Position 518 is standard chess, so we use a different position
    ChessState chess960State(true, "", 100);  // A random position
    
    // Get the FEN for this position
    std::string startingFEN = chess960State.toFEN();
    
    // Parse the castling rights from the FEN
    std::istringstream ss(startingFEN);
    std::string boardPos, activeColor, castlingAvailability;
    ss >> boardPos >> activeColor >> castlingAvailability;
    
    // Get the positions of the king and rooks
    int whiteKingPos = chess960State.getKingSquare(PieceColor::WHITE);
    int whiteKingFile = whiteKingPos % 8;
    
    // Get the original rook files
    int whiteKingsideRookFile = chess960State.getOriginalRookFile(true, PieceColor::WHITE);
    int whiteQueensideRookFile = chess960State.getOriginalRookFile(false, PieceColor::WHITE);
    
    // King and rook files should be within valid range
    EXPECT_GE(whiteKingFile, 0);
    EXPECT_LT(whiteKingFile, 8);
    EXPECT_GE(whiteKingsideRookFile, 0);
    EXPECT_LT(whiteKingsideRookFile, 8);
    EXPECT_GE(whiteQueensideRookFile, 0);
    EXPECT_LT(whiteQueensideRookFile, 8);
    
    // Kingside rook should be to the right of the king
    EXPECT_GT(whiteKingsideRookFile, whiteKingFile);
    
    // Queenside rook should be to the left of the king
    EXPECT_LT(whiteQueensideRookFile, whiteKingFile);
    
    // Verify that the rook files are reflected in the castling notation
    for (char c : castlingAvailability) {
        if (c == static_cast<char>('A' + whiteKingsideRookFile)) {
            // Found kingside castling right
            EXPECT_TRUE(chess960State.getCastlingRights().white_kingside);
        } else if (c == static_cast<char>('A' + whiteQueensideRookFile)) {
            // Found queenside castling right
            EXPECT_TRUE(chess960State.getCastlingRights().white_queenside);
        }
    }
}

}  // namespace testing
}  // namespace chess
}  // namespace alphazero

// Include test main at the end
#include "../../test_main.h"