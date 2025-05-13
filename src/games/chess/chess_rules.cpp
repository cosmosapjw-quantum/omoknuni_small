// src/games/chess/chess_rules.cpp
#include "games/chess/chess_rules.h"
#include "games/chess/chess_state.h"
#include <algorithm>
#include <array>

namespace alphazero {
namespace games {
namespace chess {

// Helper constant arrays for knight and king moves
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}
};

const std::vector<std::pair<int, int>> KING_MOVES = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
};

// Sliding piece directions (bishop, rook, queen)
const std::vector<std::pair<int, int>> BISHOP_DIRECTIONS = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

const std::vector<std::pair<int, int>> ROOK_DIRECTIONS = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

const std::vector<std::pair<int, int>> QUEEN_DIRECTIONS = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
};

// Board representation constants are defined in chess_rules.h

// Modified constructor to not store a reference to the state
ChessRules::ChessRules(bool chess960) 
    : chess960_(chess960) {
}

std::vector<ChessMove> ChessRules::generateLegalMoves(
    const ChessState& state,
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    std::vector<ChessMove> pseudoLegalMoves = generatePseudoLegalMoves(
        state, current_player, castling_rights, en_passant_square);
    std::vector<ChessMove> legalMoves;
    
    // Filter out moves that leave the king in check
    for (const ChessMove& move : pseudoLegalMoves) {
        if (!moveExposesKing(state, move, current_player)) {
            legalMoves.push_back(move);
        }
    }
    
    return legalMoves;
}

std::vector<ChessMove> ChessRules::generatePseudoLegalMoves(
    const ChessState& state,
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    std::vector<ChessMove> moves;
    
    // Generate moves for each piece of the current player
    for (int square = 0; square < 64; ++square) {
        Piece piece = state.getPiece(square);
        
        if (piece.color == current_player) {
            switch (piece.type) {
                case PieceType::PAWN:
                    addPawnMoves(state, moves, square, current_player, en_passant_square);
                    break;
                case PieceType::KNIGHT:
                    addKnightMoves(state, moves, square, current_player);
                    break;
                case PieceType::BISHOP:
                    addBishopMoves(state, moves, square, current_player);
                    break;
                case PieceType::ROOK:
                    addRookMoves(state, moves, square, current_player);
                    break;
                case PieceType::QUEEN:
                    addQueenMoves(state, moves, square, current_player);
                    break;
                case PieceType::KING:
                    addKingMoves(state, moves, square, current_player);
                    break;
                default:
                    break;
            }
        }
    }
    
    // Add castling moves
    addCastlingMoves(state, moves, current_player, castling_rights);
    
    return moves;
}

bool ChessRules::isLegalMove(
    const ChessState& state,
    const ChessMove& move,
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    // Check if the move is in the list of pseudo-legal moves
    const std::vector<ChessMove>& pseudoLegalMoves = generatePseudoLegalMoves(
        state, current_player, castling_rights, en_passant_square);
    
    auto it = std::find(pseudoLegalMoves.begin(), pseudoLegalMoves.end(), move);
    if (it == pseudoLegalMoves.end()) {
        return false;
    }
    
    // Check if the move would leave the king in check
    return !moveExposesKing(state, move, current_player);
}

bool ChessRules::isInCheck(const ChessState& state, PieceColor color) const {
    // Find the king
    int kingSquare = state.getKingSquare(color);
    if (kingSquare == -1) {
        return false;  // No king found
    }
    
    // Check if the king is attacked
    return isSquareAttacked(state, kingSquare, oppositeColor(color));
}

bool ChessRules::isSquareAttacked(const ChessState& state, int square, PieceColor by_color) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    // Check for pawn attacks
    int pawnDir = (by_color == PieceColor::WHITE) ? -1 : 1;
    for (int fileOffset : {-1, 1}) {
        int attackRank = rank + pawnDir;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = state.getPiece(attackSquare);
            
            if (attacker.type == PieceType::PAWN && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for knight attacks
    for (const auto& [rankOffset, fileOffset] : KNIGHT_MOVES) {
        int attackRank = rank + rankOffset;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = state.getPiece(attackSquare);
            
            if (attacker.type == PieceType::KNIGHT && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for king attacks
    for (const auto& [rankOffset, fileOffset] : KING_MOVES) {
        int attackRank = rank + rankOffset;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = state.getPiece(attackSquare);
            
            if (attacker.type == PieceType::KING && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for sliding piece attacks (bishop, rook, queen)
    
    // Bishop/Queen: diagonal directions
    for (const auto& [rankDir, fileDir] : BISHOP_DIRECTIONS) {
        for (int step = 1; ; ++step) {
            int attackRank = rank + rankDir * step;
            int attackFile = file + fileDir * step;
            
            if (attackRank < 0 || attackRank >= 8 || attackFile < 0 || attackFile >= 8) {
                break;  // Off the board
            }
            
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = state.getPiece(attackSquare);
            
            if (!attacker.is_empty()) {
                if (attacker.color == by_color && 
                    (attacker.type == PieceType::BISHOP || attacker.type == PieceType::QUEEN)) {
                    return true;
                }
                break;  // Piece blocks further attacks in this direction
            }
        }
    }
    
    // Rook/Queen: straight directions
    for (const auto& [rankDir, fileDir] : ROOK_DIRECTIONS) {
        for (int step = 1; ; ++step) {
            int attackRank = rank + rankDir * step;
            int attackFile = file + fileDir * step;
            
            if (attackRank < 0 || attackRank >= 8 || attackFile < 0 || attackFile >= 8) {
                break;  // Off the board
            }
            
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = state.getPiece(attackSquare);
            
            if (!attacker.is_empty()) {
                if (attacker.color == by_color && 
                    (attacker.type == PieceType::ROOK || attacker.type == PieceType::QUEEN)) {
                    return true;
                }
                break;  // Piece blocks further attacks in this direction
            }
        }
    }
    
    return false;
}

bool ChessRules::hasInsufficientMaterial(const ChessState& state) const {
    // Count material on the board
    int numPieces = 0;
    int numWhitePawns = 0;
    int numBlackPawns = 0;
    int numWhiteKnights = 0;
    int numBlackKnights = 0;
    int numWhiteBishops = 0;
    int numBlackBishops = 0;
    int numWhiteRooks = 0;
    int numBlackRooks = 0;
    int numWhiteQueens = 0;
    int numBlackQueens = 0;
    
    // Track bishop square colors
    bool whiteHasLightSquareBishop = false;
    bool whiteHasDarkSquareBishop = false;
    bool blackHasLightSquareBishop = false;
    bool blackHasDarkSquareBishop = false;
    
    for (int square = 0; square < 64; ++square) {
        Piece piece = state.getPiece(square);
        
        if (piece.is_empty()) continue;
        
        numPieces++;
        
        // Determine square color (light or dark)
        int rank = getRank(square);
        int file = getFile(square);
        bool isLightSquare = ((rank + file) % 2 == 0);
        
        if (piece.color == PieceColor::WHITE) {
            switch (piece.type) {
                case PieceType::PAWN:   
                    numWhitePawns++; 
                    break;
                case PieceType::KNIGHT: 
                    numWhiteKnights++; 
                    break;
                case PieceType::BISHOP: 
                    numWhiteBishops++; 
                    if (isLightSquare) whiteHasLightSquareBishop = true;
                    else whiteHasDarkSquareBishop = true;
                    break;
                case PieceType::ROOK:   
                    numWhiteRooks++; 
                    break;
                case PieceType::QUEEN:  
                    numWhiteQueens++; 
                    break;
                default: 
                    break;
            }
        } else {
            switch (piece.type) {
                case PieceType::PAWN:   
                    numBlackPawns++; 
                    break;
                case PieceType::KNIGHT: 
                    numBlackKnights++; 
                    break;
                case PieceType::BISHOP: 
                    numBlackBishops++; 
                    if (isLightSquare) blackHasLightSquareBishop = true;
                    else blackHasDarkSquareBishop = true;
                    break;
                case PieceType::ROOK:   
                    numBlackRooks++; 
                    break;
                case PieceType::QUEEN:  
                    numBlackQueens++; 
                    break;
                default: 
                    break;
            }
        }
    }
    
    // Check for insufficient material scenarios
    
    // King vs King
    if (numPieces == 2) {
        return true;
    }
    
    // King and Knight vs King
    if ((numWhiteKnights == 1 && numBlackKnights == 0 ||
         numWhiteKnights == 0 && numBlackKnights == 1) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Bishop vs King
    if ((numWhiteBishops == 1 && numBlackBishops == 0 ||
         numWhiteBishops == 0 && numBlackBishops == 1) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteKnights == 0 && numBlackKnights == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Bishop vs King and Bishop of the same color
    if (numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteKnights == 0 && numBlackKnights == 0 &&
        numWhiteBishops == 1 && numBlackBishops == 1 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        
        // Check if bishops are on the same color
        if ((whiteHasLightSquareBishop && blackHasLightSquareBishop) ||
            (whiteHasDarkSquareBishop && blackHasDarkSquareBishop)) {
            return true;
        }
    }
    
    // King and two Knights vs King
    if ((numWhiteKnights == 2 && numBlackKnights == 0) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King vs King and two Knights
    if ((numWhiteKnights == 0 && numBlackKnights == 2) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Knight vs King and Knight
    if (numWhiteKnights == 1 && numBlackKnights == 1 &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Knight vs King and Bishop
    if ((numWhiteKnights == 1 && numBlackBishops == 1 && numBlackKnights == 0 && numWhiteBishops == 0) ||
        (numBlackKnights == 1 && numWhiteBishops == 1 && numWhiteKnights == 0 && numBlackBishops == 0)) {
        if (numWhitePawns == 0 && numBlackPawns == 0 &&
            numWhiteRooks == 0 && numBlackRooks == 0 &&
            numWhiteQueens == 0 && numBlackQueens == 0) {
            return true;
        }
    }
    
    return false;
}

bool ChessRules::isFiftyMoveRule(int halfmove_clock) const {
    return halfmove_clock >= 100;  // 50 moves = 100 half-moves
}

CastlingRights ChessRules::getUpdatedCastlingRights(
    const ChessState& state,
    const ChessMove& move,
    const Piece& piece,
    const Piece& captured,
    const CastlingRights& current_rights) const {
    
    CastlingRights updated_rights = current_rights;
    
    // Update based on king movement
    if (piece.type == PieceType::KING) {
        if (piece.color == PieceColor::WHITE) {
            updated_rights.white_kingside = false;
            updated_rights.white_queenside = false;
        } else {
            updated_rights.black_kingside = false;
            updated_rights.black_queenside = false;
        }
    }
    
    // Update based on rook movement
    if (piece.type == PieceType::ROOK) {
        if (piece.color == PieceColor::WHITE) {
            // Check if this rook is in a castling position
            int rookFile = getFile(move.from_square);
            int kingsideRookFile = state.getOriginalRookFile(true, PieceColor::WHITE);
            int queensideRookFile = state.getOriginalRookFile(false, PieceColor::WHITE);
            
            if (rookFile == kingsideRookFile && getRank(move.from_square) == 7) {
                updated_rights.white_kingside = false;
            } else if (rookFile == queensideRookFile && getRank(move.from_square) == 7) {
                updated_rights.white_queenside = false;
            }
        } else {
            // Black rook
            int rookFile = getFile(move.from_square);
            int kingsideRookFile = state.getOriginalRookFile(true, PieceColor::BLACK);
            int queensideRookFile = state.getOriginalRookFile(false, PieceColor::BLACK);
            
            if (rookFile == kingsideRookFile && getRank(move.from_square) == 0) {
                updated_rights.black_kingside = false;
            } else if (rookFile == queensideRookFile && getRank(move.from_square) == 0) {
                updated_rights.black_queenside = false;
            }
        }
    }
    
    // Update based on rook capture
    if (!captured.is_empty() && captured.type == PieceType::ROOK) {
        if (captured.color == PieceColor::WHITE) {
            int rookFile = getFile(move.to_square);
            int kingsideRookFile = state.getOriginalRookFile(true, PieceColor::WHITE);
            int queensideRookFile = state.getOriginalRookFile(false, PieceColor::WHITE);
            
            if (rookFile == kingsideRookFile && getRank(move.to_square) == 7) {
                updated_rights.white_kingside = false;
            } else if (rookFile == queensideRookFile && getRank(move.to_square) == 7) {
                updated_rights.white_queenside = false;
            }
        } else {
            // Black rook captured
            int rookFile = getFile(move.to_square);
            int kingsideRookFile = state.getOriginalRookFile(true, PieceColor::BLACK);
            int queensideRookFile = state.getOriginalRookFile(false, PieceColor::BLACK);
            
            if (rookFile == kingsideRookFile && getRank(move.to_square) == 0) {
                updated_rights.black_kingside = false;
            } else if (rookFile == queensideRookFile && getRank(move.to_square) == 0) {
                updated_rights.black_queenside = false;
            }
        }
    }
    
    return updated_rights;
}

void ChessRules::addPawnMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player, int en_passant_square) const {
    int rank = getRank(square);
    int file = getFile(square);
    int direction = (current_player == PieceColor::WHITE) ? -1 : 1;
    
    // Regular move forward
    int newRank = rank + direction;
    if (newRank >= 0 && newRank < 8) {
        int newSquare = getSquare(newRank, file);
        if (state.getPiece(newSquare).is_empty()) {
            // Check if pawn is on the last rank (promotion)
            if (newRank == 0 || newRank == 7) {
                // Add all promotion options
                moves.push_back({square, newSquare, PieceType::QUEEN});
                moves.push_back({square, newSquare, PieceType::ROOK});
                moves.push_back({square, newSquare, PieceType::BISHOP});
                moves.push_back({square, newSquare, PieceType::KNIGHT});
            } else {
                moves.push_back({square, newSquare});
            }
            
            // Initial two-square move
            if ((current_player == PieceColor::WHITE && rank == 6) ||
                (current_player == PieceColor::BLACK && rank == 1)) {
                int twoSquaresForward = newRank + direction;
                int twoSquareNewSquare = getSquare(twoSquaresForward, file);
                if (state.getPiece(twoSquareNewSquare).is_empty()) {
                    moves.push_back({square, twoSquareNewSquare});
                }
            }
        }
    }
    
    // Captures (including en passant)
    for (int fileOffset : {-1, 1}) {
        int newFile = file + fileOffset;
        if (newFile >= 0 && newFile < 8 && newRank >= 0 && newRank < 8) {
            int newSquare = getSquare(newRank, newFile);
            
            // Regular capture
            Piece targetPiece = state.getPiece(newSquare);
            if (!targetPiece.is_empty() && targetPiece.color != current_player) {
                // Check for promotion
                if (newRank == 0 || newRank == 7) {
                    moves.push_back({square, newSquare, PieceType::QUEEN});
                    moves.push_back({square, newSquare, PieceType::ROOK});
                    moves.push_back({square, newSquare, PieceType::BISHOP});
                    moves.push_back({square, newSquare, PieceType::KNIGHT});
                } else {
                    moves.push_back({square, newSquare});
                }
            }
            
            // En passant capture
            if (en_passant_square == newSquare) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addKnightMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankOffset, fileOffset] : KNIGHT_MOVES) {
        int newRank = rank + rankOffset;
        int newFile = file + fileOffset;
        
        if (newRank >= 0 && newRank < 8 && newFile >= 0 && newFile < 8) {
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = state.getPiece(newSquare);
            
            if (targetPiece.is_empty() || targetPiece.color != current_player) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addBishopMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(state, moves, square, current_player, BISHOP_DIRECTIONS);
}

void ChessRules::addRookMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(state, moves, square, current_player, ROOK_DIRECTIONS);
}

void ChessRules::addQueenMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(state, moves, square, current_player, QUEEN_DIRECTIONS);
}

void ChessRules::addSlidingMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player, 
                               const std::vector<std::pair<int, int>>& directions) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankDir, fileDir] : directions) {
        for (int step = 1; ; ++step) {
            int newRank = rank + rankDir * step;
            int newFile = file + fileDir * step;
            
            if (newRank < 0 || newRank >= 8 || newFile < 0 || newFile >= 8) {
                break;  // Off the board
            }
            
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = state.getPiece(newSquare);
            
            if (targetPiece.is_empty()) {
                // Empty square, can move here
                moves.push_back({square, newSquare});
            } else if (targetPiece.color != current_player) {
                // Capture opponent's piece
                moves.push_back({square, newSquare});
                break;  // Can't move beyond this
            } else {
                // Own piece, can't move here or beyond
                break;
            }
        }
    }
}

void ChessRules::addKingMoves(const ChessState& state, std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankOffset, fileOffset] : KING_MOVES) {
        int newRank = rank + rankOffset;
        int newFile = file + fileOffset;
        
        if (newRank >= 0 && newRank < 8 && newFile >= 0 && newFile < 8) {
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = state.getPiece(newSquare);
            
            if (targetPiece.is_empty() || targetPiece.color != current_player) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addCastlingMoves(const ChessState& state, std::vector<ChessMove>& moves, PieceColor current_player, const CastlingRights& castling_rights) const {
    // Check if king is in check
    if (isInCheck(state, current_player)) {
        return;  // Cannot castle when in check
    }
    
    // Get castling parameters based on current player
    bool canCastleKingside = (current_player == PieceColor::WHITE) ? 
                            castling_rights.white_kingside : castling_rights.black_kingside;
    bool canCastleQueenside = (current_player == PieceColor::WHITE) ? 
                             castling_rights.white_queenside : castling_rights.black_queenside;
    
    if (!canCastleKingside && !canCastleQueenside) {
        return;  // No castling rights
    }
    
    // Find the king
    int kingSquare = state.getKingSquare(current_player);
    if (kingSquare == -1) {
        return;  // No king found
    }
    
    int kingRank = getRank(kingSquare);
    int kingFile = getFile(kingSquare);
    
    // Handle kingside castling
    if (canCastleKingside) {
        std::pair<int, int> castlingSquares = getCastlingSquares(state, current_player, true);
        int kingTarget = castlingSquares.first;
        int rookTarget = castlingSquares.second;
        
        // In Chess960, the king's target is two files to the right of its starting position
        int targetFile = kingFile + 2;
        if (chess960_ && targetFile < 8) {
            kingTarget = getSquare(kingRank, targetFile);
        }
        
        // Check if the castling path is clear
        if (isValidCastle(state, kingSquare, kingTarget, current_player, castling_rights)) {
            moves.push_back({kingSquare, kingTarget});
        }
    }
    
    // Handle queenside castling
    if (canCastleQueenside) {
        std::pair<int, int> castlingSquares = getCastlingSquares(state, current_player, false);
        int kingTarget = castlingSquares.first;
        int rookTarget = castlingSquares.second;
        
        // In Chess960, the king's target is two files to the left of its starting position
        int targetFile = kingFile - 2;
        if (chess960_ && targetFile >= 0) {
            kingTarget = getSquare(kingRank, targetFile);
        }
        
        // Check if the castling path is clear
        if (isValidCastle(state, kingSquare, kingTarget, current_player, castling_rights)) {
            moves.push_back({kingSquare, kingTarget});
        }
    }
}

bool ChessRules::isValidCastle(const ChessState& state, int from_square, int to_square, PieceColor current_player, const CastlingRights& castling_rights) const {
    int fromRank = getRank(from_square);
    int fromFile = getFile(from_square);
    int toFile = getFile(to_square);
    
    // Determine castling direction
    bool isKingside = (toFile > fromFile);
    
    // Get the original rook file
    int rookFile = state.getOriginalRookFile(isKingside, current_player);
    int rookSquare = getSquare(fromRank, rookFile);
    
    // Check that rook is present
    Piece rook = state.getPiece(rookSquare);
    if (rook.type != PieceType::ROOK || rook.color != current_player) {
        return false;
    }
    
    // Check that the path between king and rook is clear
    int minFile = std::min(fromFile, rookFile);
    int maxFile = std::max(fromFile, rookFile);
    
    for (int file = minFile + 1; file < maxFile; ++file) {
        int square = getSquare(fromRank, file);
        if (!state.getPiece(square).is_empty()) {
            return false;  // Path between king and rook is not clear
        }
    }
    
    // Check that the king's path is safe
    int step = isKingside ? 1 : -1;
    for (int file = fromFile; file != toFile + step; file += step) {
        int square = getSquare(fromRank, file);
        
        // Skip the original king square check
        if (square == from_square) continue;
        
        // Check if the square is attacked
        if (isSquareAttacked(state, square, oppositeColor(current_player))) {
            return false;  // Square in the king's path is attacked
        }
        
        // For non-Chess960, all squares in the king's path must be empty
        // For Chess960, squares between king and king's destination must be empty
        if (!chess960_ || (file != rookFile)) {
            if (square != rookSquare && !state.getPiece(square).is_empty()) {
                return false;  // Square in the king's path is not empty
            }
        }
    }
    
    return true;
}

std::pair<int, int> ChessRules::getCastlingSquares(const ChessState& state, PieceColor color, bool kingside) const {
    int rank = (color == PieceColor::WHITE) ? 7 : 0;
    
    // Determine king and rook positions after castling
    int kingFile = chess960_ ? getFile(state.getKingSquare(color)) : 4;
    int rookFile = state.getOriginalRookFile(kingside, color);
    
    int kingTargetFile = kingside ? kingFile + 2 : kingFile - 2;
    int rookTargetFile = kingside ? kingTargetFile - 1 : kingTargetFile + 1;
    
    int kingTarget = getSquare(rank, kingTargetFile);
    int rookTarget = getSquare(rank, rookTargetFile);
    
    return {kingTarget, rookTarget};
}

bool ChessRules::moveExposesKing(const ChessState& state, const ChessMove& move, PieceColor current_player) const {
    // Create a temporary state to test the move
    ChessState tempState = state.cloneWithMove(move);
    
    // Check if the king is in check after the move
    return isInCheck(tempState, current_player);
}

} // namespace chess
} // namespace games
} // namespace alphazero