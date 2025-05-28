// include/alphazero/games/chess/chess_state.h
#ifndef CHESS_STATE_H
#define CHESS_STATE_H

#include <vector>
#include <string>
#include <array>
#include <optional>
#include <memory>
#include <unordered_map>
#include <map>
#include <atomic>
#include "core/igamestate.h"
#include "utils/zobrist_hash.h"
#include "games/chess/chess_types.h"
#include "core/export_macros.h"

namespace alphazero {
namespace games {
namespace chess {

// Forward declaration
class ChessRules;

// Piece type definitions
enum class PieceType;
// Piece color definitions
enum class PieceColor;

// Castling rights
struct ALPHAZERO_API CastlingRights {
    bool white_kingside = true;
    bool white_queenside = true;
    bool black_kingside = true;
    bool black_queenside = true;
    
    bool operator==(const CastlingRights& other) const {
        return white_kingside == other.white_kingside &&
               white_queenside == other.white_queenside &&
               black_kingside == other.black_kingside &&
               black_queenside == other.black_queenside;
    }
};

// Move representation
struct ALPHAZERO_API ChessMove {
    int from_square;
    int to_square;
    PieceType promotion_piece = PieceType::NONE;
    
    bool operator==(const ChessMove& other) const {
        return from_square == other.from_square &&
               to_square == other.to_square &&
               promotion_piece == other.promotion_piece;
    }
};

// Piece representation
struct ALPHAZERO_API Piece {
    PieceType type = PieceType::NONE;
    PieceColor color = PieceColor::NONE;
    
    bool operator==(const Piece& other) const {
        return type == other.type && color == other.color;
    }
    
    bool operator!=(const Piece& other) const {
        return !(*this == other);
    }
    
    bool is_empty() const {
        return type == PieceType::NONE;
    }
};

/**
 * @brief Implementation of chess game state
 */
class ALPHAZERO_API ChessState : public core::IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param chess960 Whether to use Chess960 rules
     * @param fen Optional FEN string for initial position
     * @param position_number Chess960 position number (0-959) if applicable
     */
    ChessState(bool chess960 = false, const std::string& fen = "", int position_number = -1);
    
    /**
     * @brief Copy constructor
     */
    ChessState(const ChessState& other);
    
    /**
     * @brief Assignment operator
     */
    ChessState& operator=(const ChessState& other);
    
    /**
     * @brief Destructor - returns cached tensors to pool
     */
    ~ChessState();
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    core::GameResult getGameResult() const override;
    int getCurrentPlayer() const override;
    int getBoardSize() const override { return 8; }
    int getActionSpaceSize() const override { return 64 * 64 * 5; }  // from * to * promotion options
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<core::IGameState> clone() const override;
    void copyFrom(const core::IGameState& source) override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const core::IGameState& other) const override;
    std::vector<int> getMoveHistory() const override;
    bool validate() const override;
    
    // Chess-specific methods
    /**
     * @brief Get piece at a square
     * 
     * @param square Square index (0-63)
     * @return Piece at the square
     */
    Piece getPiece(int square) const;
    
    /**
     * @brief Set piece at a square
     * 
     * @param square Square index (0-63)
     * @param piece Piece to set
     */
    void setPiece(int square, const Piece& piece);
    
    /**
     * @brief Get FEN string representation
     * 
     * @return FEN string
     */
    std::string toFEN() const;
    
    /**
     * @brief Set position from FEN string
     * 
     * @param fen FEN string
     * @return true if successful, false otherwise
     */
    bool setFromFEN(const std::string& fen);
    
    /**
     * @brief Get castling rights
     * 
     * @return Current castling rights
     */
    CastlingRights getCastlingRights() const;
    
    /**
     * @brief Convert action to ChessMove
     * 
     * @param action Action integer
     * @return ChessMove object
     */
    ChessMove actionToChessMove(int action) const;
    
    /**
     * @brief Convert ChessMove to action
     * 
     * @param move ChessMove object
     * @return Action integer
     */
    int chessMoveToAction(const ChessMove& move) const;
    
    /**
     * @brief Check if position is in check
     * 
     * @param color Color to check (defaults to current player)
     * @return true if in check, false otherwise
     */
    bool isInCheck(PieceColor color = PieceColor::NONE) const;
    
    /**
     * @brief Check if a square is attacked
     * 
     * @param square Square index
     * @param by_color Color of the attacker
     * @return true if attacked, false otherwise
     */
    bool isSquareAttacked(int square, PieceColor by_color) const;
    
    /**
     * @brief Get en passant square
     * 
     * @return En passant square index, or -1 if none
     */
    int getEnPassantSquare() const;
    
    /**
     * @brief Get halfmove clock
     * 
     * @return Halfmove clock value
     */
    int getHalfmoveClock() const;
    
    /**
     * @brief Get fullmove number
     * 
     * @return Fullmove number
     */
    int getFullmoveNumber() const;
    
    /**
     * @brief Generate all legal moves
     * 
     * @return Vector of legal ChessMove objects
     */
    std::vector<ChessMove> generateLegalMoves() const;
    
    /**
     * @brief Check if a move is legal
     * 
     * @param move ChessMove to check
     * @return true if legal, false otherwise
     */
    bool isLegalMove(const ChessMove& move) const;
    
    /**
     * @brief Make a move
     * 
     * @param move ChessMove to make
     */
    void makeMove(const ChessMove& move);
    
    /**
     * @brief Convert square index to algebraic coordinates
     * 
     * @param square Square index (0-63)
     * @return Algebraic coordinate (e.g., "e4")
     */
    static std::string squareToString(int square);
    
    /**
     * @brief Convert algebraic coordinates to square index
     * 
     * @param squareStr Algebraic coordinate (e.g., "e4")
     * @return Square index (0-63), or -1 if invalid
     */
    static int stringToSquare(const std::string& squareStr);
    
    /**
     * @brief Convert ChessMove to algebraic notation
     * 
     * @param move ChessMove to convert
     * @return Move in algebraic notation (e.g., "e2e4")
     */
    std::string moveToString(const ChessMove& move) const;
    
    /**
     * @brief Convert algebraic notation to ChessMove
     * 
     * @param moveStr Move in algebraic notation (e.g., "e2e4")
     * @return ChessMove object, or nullopt if invalid
     */
    std::optional<ChessMove> stringToMove(const std::string& moveStr) const;
    
    /**
     * @brief Convert to Standard Algebraic Notation (SAN)
     * 
     * @param move ChessMove to convert
     * @return Move in SAN (e.g., "Nf3")
     */
    std::string toSAN(const ChessMove& move) const;
    
    /**
     * @brief Parse move in Standard Algebraic Notation (SAN)
     * 
     * @param sanStr Move in SAN (e.g., "Nf3")
     * @return ChessMove object, or nullopt if invalid
     */
    std::optional<ChessMove> fromSAN(const std::string& sanStr) const;
    
    /**
     * @brief Get the king square for a color
     * 
     * @param color Color to check
     * @return Square index of the king, or -1 if not found
     */
    int getKingSquare(PieceColor color) const;
    
    /**
     * @brief Check if the current game is using Chess960 rules
     * 
     * @return true if using Chess960 rules
     */
    bool isChess960() const { return chess960_; }
    
    /**
     * @brief Get the original rook file (used for Chess960 castling)
     * 
     * @param is_kingside Whether to get kingside or queenside rook
     * @param color Which player's rook
     * @return File (0-7) of the original rook position
     */
    int getOriginalRookFile(bool is_kingside, PieceColor color) const;
    
    /**
     * @brief Create a clone of this state with a move applied
     * 
     * @param move Move to apply
     * @return New ChessState with the move applied
     */
    ChessState cloneWithMove(const ChessMove& move) const;

private:
    static const int BOARD_SIZE = 8;
    static const int NUM_SQUARES = 64;
    
    // Board representation
    std::array<Piece, NUM_SQUARES> board_;
    
    // Game state
    PieceColor current_player_;
    CastlingRights castling_rights_;
    int en_passant_square_;
    int halfmove_clock_;
    int fullmove_number_;
    bool chess960_;
    
    // Original positions for Chess960 castling
    int white_kingside_rook_file_;
    int white_queenside_rook_file_;
    int black_kingside_rook_file_;
    int black_queenside_rook_file_;
    
    // Move history for undoing moves
    struct MoveInfo {
        ChessMove move;
        Piece captured_piece;
        CastlingRights castling_rights;
        int en_passant_square;
        int halfmove_clock;
        int fullmove_number;
        bool was_castle;
        bool was_en_passant;
    };
    std::vector<MoveInfo> move_history_;
    
    // Position history for repetition detection
    std::unordered_map<uint64_t, int> position_history_;
    
    // Cache for piece equality checks and hashing
    std::map<Piece, int> piece_cache_;
    
    // Cache for legal moves
    mutable std::vector<ChessMove> cached_legal_moves_;
    mutable bool legal_moves_dirty_;
    
    // Zobrist hashing
    core::ZobristHash zobrist_;
    mutable uint64_t hash_;
    mutable bool hash_dirty_;
    
    // Terminal state cache
    mutable bool is_terminal_cached_;
    mutable core::GameResult cached_result_;
    mutable bool terminal_check_dirty_;
    
    // PERFORMANCE FIX: Cached tensor representations to avoid expensive recomputation
    mutable std::vector<std::vector<std::vector<float>>> cached_tensor_repr_;
    mutable std::vector<std::vector<std::vector<float>>> cached_enhanced_tensor_repr_;
    mutable std::atomic<bool> tensor_cache_dirty_{true};
    mutable std::atomic<bool> enhanced_tensor_cache_dirty_{true};
    
    // Rules object
    std::shared_ptr<ChessRules> rules_;
    
    // Helper methods
    void initializeStartingPosition();
    void initializeChess960Position(int position_number);
    void initializeEmpty();
    void invalidateCache();
    void clearTensorCache() const;
    
    // Attack/Defense plane computation
    void computeAttackDefensePlanes(std::vector<std::vector<std::vector<float>>>& tensor) const;
    
    // Static batch computation for multiple states (GPU-accelerated)
    static std::vector<std::vector<std::vector<std::vector<float>>>> 
    computeBatchEnhancedTensorRepresentations(const std::vector<const ChessState*>& states);
    
    // Update zobrist hash
    void updateHash() const;
    void updateHashIncrementally(int square, const Piece& old_piece, const Piece& new_piece);
    
    // Position repetition handling
    void recordPosition();
    bool isThreefoldRepetition() const;
    
    // Utility functions
    static int getRank(int square) { return square / 8; }
    static int getFile(int square) { return square % 8; }
    static int getSquare(int rank, int file) { return rank * 8 + file; }
    static bool isValidSquare(int square) { return square >= 0 && square < 64; }
    static PieceColor oppositeColor(PieceColor color) {
        return color == PieceColor::WHITE ? PieceColor::BLACK : PieceColor::WHITE;
    }
    
    // Make square computation accessible to Chess960
    friend class Chess960;
};

} // namespace chess
} // namespace games
} // namespace alphazero

#endif // CHESS_STATE_H