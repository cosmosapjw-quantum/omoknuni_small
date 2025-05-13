// include/alphazero/games/chess/chess_types.h
#ifndef CHESS_TYPES_H
#define CHESS_TYPES_H

namespace alphazero {
namespace games {
namespace chess {

// Piece type definitions
enum class PieceType {
    NONE = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6
};

// Piece color definitions
enum class PieceColor {
    NONE = 0,
    WHITE = 1,
    BLACK = 2
};

} // namespace chess
} // namespace games
} // namespace alphazero

#endif // CHESS_TYPES_H 