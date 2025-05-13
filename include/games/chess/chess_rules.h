// include/games/chess/chess_rules.h
#ifndef CHESS_RULES_H
#define CHESS_RULES_H

#include <vector>
#include <functional>
#include <optional>
#include <unordered_set>
#include <cstdint>
#include "games/chess/chess_types.h"

namespace alphazero {
namespace games {
namespace chess {

// Forward declarations
struct Piece;
struct CastlingRights;
struct ChessMove;
enum class PieceType;
enum class PieceColor;
class ChessState;

// Square constants using little-endian rank-file mapping (0-63)
// A1 is the bottom-left of the board (white's queenside corner)
constexpr int A1 = 56, B1 = 57, C1 = 58, D1 = 59, E1 = 60, F1 = 61, G1 = 62, H1 = 63;
constexpr int A2 = 48, B2 = 49, C2 = 50, D2 = 51, E2 = 52, F2 = 53, G2 = 54, H2 = 55;
constexpr int A3 = 40, B3 = 41, C3 = 42, D3 = 43, E3 = 44, F3 = 45, G3 = 46, H3 = 47;
constexpr int A4 = 32, B4 = 33, C4 = 34, D4 = 35, E4 = 36, F4 = 37, G4 = 38, H4 = 39;
constexpr int A5 = 24, B5 = 25, C5 = 26, D5 = 27, E5 = 28, F5 = 29, G5 = 30, H5 = 31;
constexpr int A6 = 16, B6 = 17, C6 = 18, D6 = 19, E6 = 20, F6 = 21, G6 = 22, H6 = 23;
constexpr int A7 = 8, B7 = 9, C7 = 10, D7 = 11, E7 = 12, F7 = 13, G7 = 14, H7 = 15;
constexpr int A8 = 0, B8 = 1, C8 = 2, D8 = 3, E8 = 4, F8 = 5, G8 = 6, H8 = 7;

/**
 * @brief Rules implementation for Chess
 */
class ChessRules {
public:
    /**
     * @brief Constructor
     * 
     * @param chess960 Whether to use Chess960 rules
     */
    ChessRules(bool chess960 = false);
    
    /**
     * @brief Generate all legal moves
     * 
     * @param state The chess state
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return Vector of legal ChessMove objects
     */
    std::vector<ChessMove> generateLegalMoves(
        const ChessState& state,
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Generate pseudo-legal moves (not checking for check)
     * 
     * @param state The chess state
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return Vector of pseudo-legal ChessMove objects
     */
    std::vector<ChessMove> generatePseudoLegalMoves(
        const ChessState& state,
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Check if a move is legal
     * 
     * @param state The chess state
     * @param move Move to check
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return true if legal, false otherwise
     */
    bool isLegalMove(
        const ChessState& state,
        const ChessMove& move,
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Check if a position is in check
     * 
     * @param state The chess state
     * @param color Color to check for
     * @return true if in check, false otherwise
     */
    bool isInCheck(const ChessState& state, PieceColor color) const;
    
    /**
     * @brief Check if a square is attacked by a player
     * 
     * @param state The chess state
     * @param square Square index
     * @param by_color Color of the attacker
     * @return true if attacked, false otherwise
     */
    bool isSquareAttacked(const ChessState& state, int square, PieceColor by_color) const;
    
    /**
     * @brief Check for insufficient material (draw condition)
     * 
     * @param state The chess state
     * @return true if position has insufficient material for checkmate
     */
    bool hasInsufficientMaterial(const ChessState& state) const;
    
    /**
     * @brief Check for fifty-move rule
     * 
     * @param halfmove_clock Current halfmove clock value
     * @return true if 50-move rule applies
     */
    bool isFiftyMoveRule(int halfmove_clock) const;
    
    /**
     * @brief Get updated castling rights after a move
     * 
     * @param state The chess state
     * @param move The executed move
     * @param piece The piece that moved
     * @param captured Captured piece (if any)
     * @param current_rights Current castling rights
     * @return Updated castling rights
     */
    CastlingRights getUpdatedCastlingRights(
        const ChessState& state,
        const ChessMove& move,
        const Piece& piece,
        const Piece& captured,
        const CastlingRights& current_rights) const;
        
private:
    bool chess960_;
    
    // Move generation helpers
    void addPawnMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player, int en_passant_square) const;
    void addKnightMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addBishopMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addRookMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addQueenMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addKingMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addCastlingMoves(const ChessState& state, std::vector<ChessMove>& moves, PieceColor current_player, const CastlingRights& castling_rights) const;
    
    // Check if a move is a valid castle
    bool isValidCastle(const ChessState& state, int from_square, int to_square, PieceColor current_player, const CastlingRights& castling_rights) const;
    
    // Sliding piece movement
    void addSlidingMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player, 
                         const std::vector<std::pair<int, int>>& directions) const;
    
    // Get pieces involved in castling for Chess960
    std::pair<int, int> getCastlingSquares(const ChessState& state, PieceColor color, bool kingside) const;
    
    // Move legality checking
    bool moveExposesKing(const ChessState& state, const ChessMove& move, PieceColor current_player) const;
    
    // Utility functions
    static int getRank(int square) { return square / 8; }
    static int getFile(int square) { return square % 8; }
    static int getSquare(int rank, int file) { return rank * 8 + file; }
    static PieceColor oppositeColor(PieceColor color) {
        return color == PieceColor::WHITE ? PieceColor::BLACK : PieceColor::WHITE;
    }
};

} // namespace chess
} // namespace games
} // namespace alphazero

#endif // CHESS_RULES_H