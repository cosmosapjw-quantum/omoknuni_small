// src/games/chess/chess_state.cpp
#include "games/chess/chess_state.h"
#include "games/chess/chess_rules.h"
#include "games/chess/chess960.h"
#include "core/tensor_pool.h"            // For GlobalTensorPool optimization
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>

namespace alphazero {
namespace games {
namespace chess {

// ChessState constructor
ChessState::ChessState(bool chess960, const std::string& fen, int position_number)
    : IGameState(core::GameType::CHESS),
      current_player_(PieceColor::WHITE),
      en_passant_square_(-1),
      halfmove_clock_(0),
      fullmove_number_(1),
      chess960_(chess960),
      white_kingside_rook_file_(7),    // Default A/H files for standard chess
      white_queenside_rook_file_(0),
      black_kingside_rook_file_(7),
      black_queenside_rook_file_(0),
      legal_moves_dirty_(true),
      zobrist_(8, 12, 2),  // boardSize=8, numPieceTypes=12, numPlayers=2
      hash_dirty_(true),
      terminal_check_dirty_(true)
{
    // Initialize with empty board first
    initializeEmpty();
    
    // Initialize rules object - now doesn't store a reference to this
    rules_ = std::make_shared<ChessRules>(chess960_);
    
    // Add named features for hash calculation
    zobrist_.addFeature("castling", 16);      // 4 bits for castling rights
    zobrist_.addFeature("en_passant", 65);    // 64 squares + 1 for no en passant
    zobrist_.addFeature("chess960", 2);       // Boolean flag for Chess960 mode
    
    // Setup the position based on inputs
    if (chess960_ && position_number >= 0 && position_number < 960) {
        // Use the specified Chess960 position number
        initializeChess960Position(position_number);
    } else if (!fen.empty()) {
        // If FEN is provided, use it
        if (!setFromFEN(fen)) {
            // If FEN parsing fails, fall back to starting position
            if (chess960_) {
                // Validate that position 518 is truly the standard chess setup
                if (!Chess960::isStandardChessPosition(518)) {
                    throw std::runtime_error("Chess960 position 518 is not the standard chess setup!");
                }
                initializeChess960Position(518); // Standard chess initial position
            } else {
                initializeStartingPosition();
            }
        }
    } else {
        // Use standard starting position by default
        if (chess960_) {
            // Validate that position 518 is truly the standard chess setup
            if (!Chess960::isStandardChessPosition(518)) {
                throw std::runtime_error("Chess960 position 518 is not the standard chess setup!");
            }
            initializeChess960Position(518); // Standard chess initial position
        } else {
            initializeStartingPosition();
        }
    }
    
    // Update rules with the current board state
    rules_ = std::make_shared<ChessRules>(chess960_);
    
    // Initialize position history map with current position
    recordPosition();
}

// Copy constructor
ChessState::ChessState(const ChessState& other)
    : IGameState(core::GameType::CHESS),
      board_(other.board_),
      current_player_(other.current_player_),
      castling_rights_(other.castling_rights_),
      en_passant_square_(other.en_passant_square_),
      halfmove_clock_(other.halfmove_clock_),
      fullmove_number_(other.fullmove_number_),
      chess960_(other.chess960_),
      white_kingside_rook_file_(other.white_kingside_rook_file_),
      white_queenside_rook_file_(other.white_queenside_rook_file_),
      black_kingside_rook_file_(other.black_kingside_rook_file_),
      black_queenside_rook_file_(other.black_queenside_rook_file_),
      move_history_(other.move_history_),
      position_history_(other.position_history_),
      cached_legal_moves_(other.cached_legal_moves_),
      legal_moves_dirty_(other.legal_moves_dirty_),
      zobrist_(other.zobrist_),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_),
      is_terminal_cached_(other.is_terminal_cached_),
      cached_result_(other.cached_result_),
      terminal_check_dirty_(other.terminal_check_dirty_),
      // Don't copy cached tensors - force recomputation to avoid memory issues
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true)
{
    // Initialize rules object after all other members are initialized
    rules_ = std::make_shared<ChessRules>(chess960_);
}

// Assignment operator
ChessState& ChessState::operator=(const ChessState& other) {
    if (this != &other) {
        board_ = other.board_;
        current_player_ = other.current_player_;
        castling_rights_ = other.castling_rights_;
        en_passant_square_ = other.en_passant_square_;
        halfmove_clock_ = other.halfmove_clock_;
        fullmove_number_ = other.fullmove_number_;
        chess960_ = other.chess960_;
        white_kingside_rook_file_ = other.white_kingside_rook_file_;
        white_queenside_rook_file_ = other.white_queenside_rook_file_;
        black_kingside_rook_file_ = other.black_kingside_rook_file_;
        black_queenside_rook_file_ = other.black_queenside_rook_file_;
        move_history_ = other.move_history_;
        position_history_ = other.position_history_;
        cached_legal_moves_ = other.cached_legal_moves_;
        legal_moves_dirty_ = other.legal_moves_dirty_;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
        is_terminal_cached_ = other.is_terminal_cached_;
        cached_result_ = other.cached_result_;
        terminal_check_dirty_ = other.terminal_check_dirty_;
        
        // Clear old tensor caches before assignment
        clearTensorCache();
        
        // Don't copy cached tensors - force recomputation
        tensor_cache_dirty_ = true;
        enhanced_tensor_cache_dirty_ = true;
        
        // Reinitialize rules object after all other members are updated
        rules_ = std::make_shared<ChessRules>(chess960_);
    }
    return *this;
}

// Initialize empty board
void ChessState::initializeEmpty() {
    // Initialize empty board
    for (int i = 0; i < NUM_SQUARES; ++i) {
        board_[i] = Piece();
    }
    
    // Reset game state
    current_player_ = PieceColor::WHITE;
    castling_rights_ = CastlingRights();
    en_passant_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    
    // Clear move history and position history
    move_history_.clear();
    position_history_.clear();
    
    // Mark caches as dirty
    invalidateCache();
}

// Initialize standard starting position
void ChessState::initializeStartingPosition() {
    initializeEmpty();
    
    // Set up pawns
    for (int i = 0; i < 8; ++i) {
        // White pawns (rank 2)
        setPiece(getSquare(6, i), {PieceType::PAWN, PieceColor::WHITE});
        
        // Black pawns (rank 7)
        setPiece(getSquare(1, i), {PieceType::PAWN, PieceColor::BLACK});
    }
    
    // Set up rooks
    setPiece(A1, {PieceType::ROOK, PieceColor::WHITE});
    setPiece(H1, {PieceType::ROOK, PieceColor::WHITE});
    setPiece(A8, {PieceType::ROOK, PieceColor::BLACK});
    setPiece(H8, {PieceType::ROOK, PieceColor::BLACK});
    
    // Set up knights
    setPiece(getSquare(7, 1), {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(getSquare(7, 6), {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(getSquare(0, 1), {PieceType::KNIGHT, PieceColor::BLACK});
    setPiece(getSquare(0, 6), {PieceType::KNIGHT, PieceColor::BLACK});
    
    // Set up bishops
    setPiece(getSquare(7, 2), {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(getSquare(7, 5), {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(getSquare(0, 2), {PieceType::BISHOP, PieceColor::BLACK});
    setPiece(getSquare(0, 5), {PieceType::BISHOP, PieceColor::BLACK});
    
    // Set up queens
    setPiece(getSquare(7, 3), {PieceType::QUEEN, PieceColor::WHITE});
    setPiece(getSquare(0, 3), {PieceType::QUEEN, PieceColor::BLACK});
    
    // Set up kings
    setPiece(E1, {PieceType::KING, PieceColor::WHITE});
    setPiece(E8, {PieceType::KING, PieceColor::BLACK});
    
    // Initialize castling rights
    castling_rights_.white_kingside = true;
    castling_rights_.white_queenside = true;
    castling_rights_.black_kingside = true;
    castling_rights_.black_queenside = true;
    
    // Initialize standard rook files
    white_kingside_rook_file_ = 7;  // H file
    white_queenside_rook_file_ = 0; // A file
    black_kingside_rook_file_ = 7;  // H file
    black_queenside_rook_file_ = 0; // A file
    
    // Update hash
    updateHash();
}

// Initialize a Chess960 position
void ChessState::initializeChess960Position(int position_number) {
    // Use Chess960 utility to set up the position
    Chess960::setupPosition(position_number, *this);
    
    // Get rook files for this position
    auto rookFiles = Chess960::getRookFiles(position_number);
    white_queenside_rook_file_ = rookFiles.first;
    white_kingside_rook_file_ = rookFiles.second;
    black_queenside_rook_file_ = rookFiles.first;
    black_kingside_rook_file_ = rookFiles.second;
    
    // Update hash
    updateHash();
}

// Board manipulation methods
Piece ChessState::getPiece(int square) const {
    if (square < 0 || square >= NUM_SQUARES) {
        return Piece();
    }
    return board_[square];
}

void ChessState::setPiece(int square, const Piece& piece) {
    if (square < 0 || square >= NUM_SQUARES) {
        return;
    }
    
    // Update hash incrementally if hash is valid
    if (!hash_dirty_) {
        updateHashIncrementally(square, board_[square], piece);
    }
    
    board_[square] = piece;
    invalidateCache();
}

// Get/set methods for game state
CastlingRights ChessState::getCastlingRights() const {
    return castling_rights_;
}

int ChessState::getEnPassantSquare() const {
    return en_passant_square_;
}

int ChessState::getHalfmoveClock() const {
    return halfmove_clock_;
}

int ChessState::getFullmoveNumber() const {
    return fullmove_number_;
}

int ChessState::getOriginalRookFile(bool is_kingside, PieceColor color) const {
    if (color == PieceColor::WHITE) {
        return is_kingside ? white_kingside_rook_file_ : white_queenside_rook_file_;
    } else {
        return is_kingside ? black_kingside_rook_file_ : black_queenside_rook_file_;
    }
}

// FEN string conversion
std::string ChessState::toFEN() const {
    std::stringstream ss;
    
    // Board position
    for (int rank = 0; rank < 8; ++rank) {
        int emptyCount = 0;
        for (int file = 0; file < 8; ++file) {
            int square = getSquare(rank, file);
            Piece piece = board_[square];
            
            if (piece.is_empty()) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    ss << emptyCount;
                    emptyCount = 0;
                }
                
                char pieceChar;
                switch (piece.type) {
                    case PieceType::PAWN:   pieceChar = 'p'; break;
                    case PieceType::KNIGHT: pieceChar = 'n'; break;
                    case PieceType::BISHOP: pieceChar = 'b'; break;
                    case PieceType::ROOK:   pieceChar = 'r'; break;
                    case PieceType::QUEEN:  pieceChar = 'q'; break;
                    case PieceType::KING:   pieceChar = 'k'; break;
                    default:                pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::WHITE) {
                    pieceChar = std::toupper(pieceChar);
                }
                
                ss << pieceChar;
            }
        }
        
        if (emptyCount > 0) {
            ss << emptyCount;
        }
        
        if (rank < 7) {
            ss << '/';
        }
    }
    
    // Active color
    ss << ' ' << (current_player_ == PieceColor::WHITE ? 'w' : 'b');
    
    // Castling availability - handle both standard and Chess960 notation
    ss << ' ';
    bool hasCastling = false;
    
    if (chess960_) {
        // Chess960 castling notation uses rook files
        if (castling_rights_.white_kingside) {
            ss << static_cast<char>('A' + white_kingside_rook_file_);
            hasCastling = true;
        }
        if (castling_rights_.white_queenside) {
            ss << static_cast<char>('A' + white_queenside_rook_file_);
            hasCastling = true;
        }
        if (castling_rights_.black_kingside) {
            ss << static_cast<char>('a' + black_kingside_rook_file_);
            hasCastling = true;
        }
        if (castling_rights_.black_queenside) {
            ss << static_cast<char>('a' + black_queenside_rook_file_);
            hasCastling = true;
        }
    } else {
        // Standard chess notation
        if (castling_rights_.white_kingside) {
            ss << 'K';
            hasCastling = true;
        }
        if (castling_rights_.white_queenside) {
            ss << 'Q';
            hasCastling = true;
        }
        if (castling_rights_.black_kingside) {
            ss << 'k';
            hasCastling = true;
        }
        if (castling_rights_.black_queenside) {
            ss << 'q';
            hasCastling = true;
        }
    }
    
    if (!hasCastling) {
        ss << '-';
    }
    
    // En passant target square
    ss << ' ';
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        ss << squareToString(en_passant_square_);
    } else {
        ss << '-';
    }
    
    // Halfmove clock
    ss << ' ' << halfmove_clock_;
    
    // Fullmove number
    ss << ' ' << fullmove_number_;
    
    return ss.str();
}

bool ChessState::setFromFEN(const std::string& fen) {
    initializeEmpty();
    
    std::istringstream ss(fen);
    std::string boardPos, activeColor, castlingAvailability, enPassantTarget, halfmoveClock, fullmoveNumber;
    
    // Parse board position
    if (!(ss >> boardPos)) return false;
    
    int rank = 0, file = 0;
    for (char c : boardPos) {
        if (c == '/') {
            rank++;
            file = 0;
        } else if (std::isdigit(c)) {
            file += c - '0';
        } else {
            if (file >= 8 || rank >= 8) return false;
            
            Piece piece;
            char lowercase = std::tolower(c);
            
            switch (lowercase) {
                case 'p': piece.type = PieceType::PAWN; break;
                case 'n': piece.type = PieceType::KNIGHT; break;
                case 'b': piece.type = PieceType::BISHOP; break;
                case 'r': piece.type = PieceType::ROOK; break;
                case 'q': piece.type = PieceType::QUEEN; break;
                case 'k': piece.type = PieceType::KING; break;
                default: return false;
            }
            
            piece.color = std::isupper(c) ? PieceColor::WHITE : PieceColor::BLACK;
            setPiece(getSquare(rank, file), piece);
            file++;
        }
    }
    
    // Parse active color
    if (!(ss >> activeColor)) return false;
    current_player_ = (activeColor == "w") ? PieceColor::WHITE : PieceColor::BLACK;
    
    // Parse castling availability
    if (!(ss >> castlingAvailability)) return false;
    
    // Reset castling rights
    castling_rights_.white_kingside = false;
    castling_rights_.white_queenside = false;
    castling_rights_.black_kingside = false;
    castling_rights_.black_queenside = false;
    
    if (castlingAvailability != "-") {
        // Handle both standard and Chess960 castling notation
        for (char c : castlingAvailability) {
            if (c == 'K') {
                castling_rights_.white_kingside = true;
            } else if (c == 'Q') {
                castling_rights_.white_queenside = true;
            } else if (c == 'k') {
                castling_rights_.black_kingside = true;
            } else if (c == 'q') {
                castling_rights_.black_queenside = true;
            } else if (c >= 'A' && c <= 'H') {
                // Chess960 white rook file
                int rookFile = c - 'A';
                if (rookFile > getFile(getKingSquare(PieceColor::WHITE))) {
                    castling_rights_.white_kingside = true;
                    white_kingside_rook_file_ = rookFile;
                } else {
                    castling_rights_.white_queenside = true;
                    white_queenside_rook_file_ = rookFile;
                }
            } else if (c >= 'a' && c <= 'h') {
                // Chess960 black rook file
                int rookFile = c - 'a';
                if (rookFile > getFile(getKingSquare(PieceColor::BLACK))) {
                    castling_rights_.black_kingside = true;
                    black_kingside_rook_file_ = rookFile;
                } else {
                    castling_rights_.black_queenside = true;
                    black_queenside_rook_file_ = rookFile;
                }
            }
        }
    }
    
    // Parse en passant target square
    if (!(ss >> enPassantTarget)) return false;
    en_passant_square_ = (enPassantTarget == "-") ? -1 : stringToSquare(enPassantTarget);
    
    // Parse halfmove clock
    if (!(ss >> halfmoveClock)) return false;
    try {
        halfmove_clock_ = std::stoi(halfmoveClock);
    } catch (...) {
        return false;
    }
    
    // Parse fullmove number
    if (!(ss >> fullmoveNumber)) return false;
    try {
        fullmove_number_ = std::stoi(fullmoveNumber);
    } catch (...) {
        return false;
    }
    
    // Update hash and invalidate caches
    updateHash();
    
    // Record initial position for repetition detection
    recordPosition();
    
    return true;
}

// IGameState interface implementation
std::vector<int> ChessState::getLegalMoves() const {
    std::vector<int> result;
    
    std::vector<ChessMove> chessMovesLegal = generateLegalMoves();
    
    // Convert to action integers
    result.reserve(chessMovesLegal.size());
    for (const auto& move : chessMovesLegal) {
        result.push_back(chessMoveToAction(move));
    }
    
    return result;
}

bool ChessState::isLegalMove(int action) const {
    ChessMove move = actionToChessMove(action);
    return isLegalMove(move);
}

void ChessState::makeMove(int action) {
    ChessMove move = actionToChessMove(action);
    makeMove(move);
}

bool ChessState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    
    // Get the last move info
    const MoveInfo& moveInfo = move_history_.back();
    
    // Piece that moved is currently at moveInfo.move.to_square
    Piece piece_that_moved = board_[moveInfo.move.to_square];

    // Restore the captured piece to its original square
    setPiece(moveInfo.move.to_square, moveInfo.captured_piece);
    
    // If it was a promotion, revert piece_that_moved to pawn type
    if (moveInfo.move.promotion_piece != PieceType::NONE) {
        piece_that_moved.type = PieceType::PAWN;
    }
    // Move piece_that_moved back to its origin
    setPiece(moveInfo.move.from_square, piece_that_moved);
    // The to_square is now correctly occupied by either the captured_piece or is empty if captured_piece was NONE.
    
    // Handle special moves undo
    if (moveInfo.was_castle) {
        // Get the starting and ending files for this castling move
        int rank = getRank(moveInfo.move.from_square);
        int kingFile = getFile(moveInfo.move.from_square);
        int newKingFile = getFile(moveInfo.move.to_square);
        bool kingside = (newKingFile > kingFile);
        
        // Get the correct rook file based on the color and side
        int rookFile = getOriginalRookFile(kingside, piece_that_moved.color);
        int rookToFile = kingside ? 5 : 3;  // Standard squares for rook after castling
        
        if (chess960_) {
            // In Chess960, we need to find where the rook ended up
            // For kingside, the rook is on the king's left; for queenside, on the king's right
            rookToFile = kingside ? newKingFile - 1 : newKingFile + 1;
        }
        
        int rookFromSquare = getSquare(rank, rookFile);
        int rookToSquare = getSquare(rank, rookToFile);
        
        Piece rook = getPiece(rookToSquare);
        setPiece(rookFromSquare, rook);
        setPiece(rookToSquare, Piece());
    } else if (moveInfo.was_en_passant) {
        // Restore the captured pawn
        int capturedPawnSquare = getSquare(getRank(moveInfo.move.from_square), getFile(moveInfo.move.to_square));
        setPiece(capturedPawnSquare, {PieceType::PAWN, oppositeColor(piece_that_moved.color)});
    }
    
    // Restore game state
    current_player_ = oppositeColor(current_player_);
    castling_rights_ = moveInfo.castling_rights;
    en_passant_square_ = moveInfo.en_passant_square;
    halfmove_clock_ = moveInfo.halfmove_clock;
    fullmove_number_ = moveInfo.fullmove_number; // Restore fullmove_number
    
    // Remove the move from history
    move_history_.pop_back();
    
    // Update position history
    if (position_history_.count(hash_) > 0 && position_history_[hash_] > 0) {
        position_history_[hash_]--;
        if (position_history_[hash_] == 0) {
            position_history_.erase(hash_);
        }
    }
    
    // Invalidate caches
    invalidateCache();
    
    return true;
}

bool ChessState::isTerminal() const {
    if (!terminal_check_dirty_) {
        return is_terminal_cached_;
    }
    
    // Check if we have legal moves
    std::vector<ChessMove> legalMoves = generateLegalMoves();
    
    // If no legal moves, the game is over
    if (legalMoves.empty()) {
        is_terminal_cached_ = true;
        
        // If in check, it's checkmate; otherwise, stalemate
        if (isInCheck(current_player_)) {
            cached_result_ = current_player_ == PieceColor::WHITE ? 
                core::GameResult::WIN_PLAYER2 : core::GameResult::WIN_PLAYER1;
        } else {
            cached_result_ = core::GameResult::DRAW;
        }
        
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Check for draw by insufficient material
    if (rules_->hasInsufficientMaterial(*this)) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Check for 50-move rule
    if (rules_->isFiftyMoveRule(halfmove_clock_)) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Check for threefold repetition
    if (isThreefoldRepetition()) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Game is not terminal
    is_terminal_cached_ = false;
    cached_result_ = core::GameResult::ONGOING;
    terminal_check_dirty_ = false;
    return false;
}

core::GameResult ChessState::getGameResult() const {
    if (terminal_check_dirty_) {
        isTerminal();  // This will update cached_result_
    }
    return cached_result_;
}

int ChessState::getCurrentPlayer() const {
    return static_cast<int>(current_player_);
}

std::vector<std::vector<std::vector<float>>> ChessState::getTensorRepresentation() const {
    // PERFORMANCE FIX: Use cached tensor if available and not dirty
    if (!tensor_cache_dirty_.load(std::memory_order_relaxed) && !cached_tensor_repr_.empty()) {
        return cached_tensor_repr_;
    }
    
    // PERFORMANCE FIX: Use global tensor pool for efficient memory reuse
    auto tensor = core::GlobalTensorPool::getInstance().getTensor(12, 8, 8);
    
    // Fill tensor with piece positions
    for (int square = 0; square < NUM_SQUARES; ++square) {
        int rank = getRank(square);
        int file = getFile(square);
        Piece piece = board_[square];
        
        if (piece.type != PieceType::NONE) {
            int planeIdx = -1;
            if (piece.color == PieceColor::WHITE) {
                switch (piece.type) {
                    case PieceType::PAWN:   planeIdx = 0; break;
                    case PieceType::KNIGHT: planeIdx = 1; break;
                    case PieceType::BISHOP: planeIdx = 2; break;
                    case PieceType::ROOK:   planeIdx = 3; break;
                    case PieceType::QUEEN:  planeIdx = 4; break;
                    case PieceType::KING:   planeIdx = 5; break;
                    default: break;
                }
            } else {
                switch (piece.type) {
                    case PieceType::PAWN:   planeIdx = 6; break;
                    case PieceType::KNIGHT: planeIdx = 7; break;
                    case PieceType::BISHOP: planeIdx = 8; break;
                    case PieceType::ROOK:   planeIdx = 9; break;
                    case PieceType::QUEEN:  planeIdx = 10; break;
                    case PieceType::KING:   planeIdx = 11; break;
                    default: break;
                }
            }
            
            if (planeIdx >= 0) {
                tensor[planeIdx][rank][file] = 1.0f;
            }
        }
    }
    
    // PERFORMANCE FIX: Cache the computed tensor
    cached_tensor_repr_ = tensor;
    tensor_cache_dirty_.store(false, std::memory_order_relaxed);
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> ChessState::getEnhancedTensorRepresentation() const {
    // PERFORMANCE FIX: Use cached enhanced tensor if available and not dirty
    if (!enhanced_tensor_cache_dirty_.load(std::memory_order_relaxed) && !cached_enhanced_tensor_repr_.empty()) {
        return cached_enhanced_tensor_repr_;
    }
    
    try {
        // Start with basic representation
        std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
        
        // Add additional planes for enhanced features
        const int boardSize = 8;
        
        // 13: Current player (1 for white, 0 for black)
        tensor.push_back(std::vector<std::vector<float>>(boardSize, 
                        std::vector<float>(boardSize, current_player_ == PieceColor::WHITE ? 1.0f : 0.0f)));
        
        // 14: Castling rights
        std::vector<std::vector<float>> castlingPlane(boardSize, std::vector<float>(boardSize, 0.0f));
        float castlingValue = 0.0f;
        if (castling_rights_.white_kingside) castlingValue += 0.25f;
        if (castling_rights_.white_queenside) castlingValue += 0.25f;
        if (castling_rights_.black_kingside) castlingValue += 0.25f;
        if (castling_rights_.black_queenside) castlingValue += 0.25f;
        
        for (int rank = 0; rank < boardSize; ++rank) {
            for (int file = 0; file < boardSize; ++file) {
                castlingPlane[rank][file] = castlingValue;
            }
        }
        tensor.push_back(castlingPlane);
        
        // 15: En passant square
        std::vector<std::vector<float>> enPassantPlane(boardSize, std::vector<float>(boardSize, 0.0f));
        if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
            int rank = getRank(en_passant_square_);
            int file = getFile(en_passant_square_);
            if (rank >= 0 && rank < boardSize && file >= 0 && file < boardSize) {
                enPassantPlane[rank][file] = 1.0f;
            }
        }
        tensor.push_back(enPassantPlane);
        
        // 16: Halfmove clock (normalized)
        float normalizedHalfmove = std::min(1.0f, static_cast<float>(halfmove_clock_) / 100.0f);
        tensor.push_back(std::vector<std::vector<float>>(boardSize, 
                        std::vector<float>(boardSize, normalizedHalfmove)));
        
        // 17: Chess960 mode
        tensor.push_back(std::vector<std::vector<float>>(boardSize, 
                        std::vector<float>(boardSize, chess960_ ? 1.0f : 0.0f)));
        
        // 18: Repetition count (normalized)
        std::vector<std::vector<float>> repetitionPlane(boardSize, std::vector<float>(boardSize, 0.0f));
        if (!hash_dirty_) {
            auto it = position_history_.find(hash_);
            if (it != position_history_.end()) {
                float repetitionCount = static_cast<float>(it->second) / 3.0f;  // Normalize to [0,1] with 3 as max
                repetitionCount = std::min(1.0f, std::max(0.0f, repetitionCount)); // Clamp to [0,1]
                for (int rank = 0; rank < boardSize; ++rank) {
                    for (int file = 0; file < boardSize; ++file) {
                        repetitionPlane[rank][file] = repetitionCount;
                    }
                }
            }
        }
        tensor.push_back(repetitionPlane);
        
        // PERFORMANCE FIX: Cache the computed enhanced tensor
        cached_enhanced_tensor_repr_ = tensor;
        enhanced_tensor_cache_dirty_.store(false, std::memory_order_relaxed);
        
        return tensor;
    } catch (const std::exception& e) {
        std::cerr << "Exception in ChessState::getEnhancedTensorRepresentation: " << e.what() << std::endl;
        
        // Return a default tensor with the correct dimensions
        // Chess enhanced tensor has 19 planes (12 piece planes + 7 additional planes)
        const int num_planes = 19;
        const int boardSize = 8;
        
        return std::vector<std::vector<std::vector<float>>>(
            num_planes,
            std::vector<std::vector<float>>(
                boardSize,
                std::vector<float>(boardSize, 0.0f)
            )
        );
    } catch (...) {
        std::cerr << "Unknown exception in ChessState::getEnhancedTensorRepresentation" << std::endl;
        
        // Return a default tensor with the correct dimensions
        const int num_planes = 19;
        const int boardSize = 8;
        
        return std::vector<std::vector<std::vector<float>>>(
            num_planes,
            std::vector<std::vector<float>>(
                boardSize,
                std::vector<float>(boardSize, 0.0f)
            )
        );
    }
}

uint64_t ChessState::getHash() const {
    if (hash_dirty_) {
        updateHash();
    }
    return hash_;
}

std::unique_ptr<core::IGameState> ChessState::clone() const {
    return std::make_unique<ChessState>(*this);
}

void ChessState::copyFrom(const core::IGameState& source) {
    // Ensure source is a ChessState
    const ChessState* chess_source = dynamic_cast<const ChessState*>(&source);
    if (!chess_source) {
        throw std::runtime_error("Cannot copy from non-ChessState: incompatible game types");
    }
    
    // Copy all member variables
    board_ = chess_source->board_;
    current_player_ = chess_source->current_player_;
    castling_rights_ = chess_source->castling_rights_;
    en_passant_square_ = chess_source->en_passant_square_;
    halfmove_clock_ = chess_source->halfmove_clock_;
    fullmove_number_ = chess_source->fullmove_number_;
    move_history_ = chess_source->move_history_;
    position_history_ = chess_source->position_history_;
    zobrist_ = chess_source->zobrist_;
    hash_ = chess_source->hash_;
    chess960_ = chess_source->chess960_;
    white_kingside_rook_file_ = chess_source->white_kingside_rook_file_;
    white_queenside_rook_file_ = chess_source->white_queenside_rook_file_;
    black_kingside_rook_file_ = chess_source->black_kingside_rook_file_;
    black_queenside_rook_file_ = chess_source->black_queenside_rook_file_;
    piece_cache_ = chess_source->piece_cache_;
    
    // Re-create rules with proper configuration
    rules_ = std::make_shared<ChessRules>(chess960_);
    
    // Mark caches as dirty
    hash_dirty_ = true;
    terminal_check_dirty_ = true;
    is_terminal_cached_ = false;
    cached_legal_moves_.clear();
    legal_moves_dirty_ = true;
}

std::string ChessState::actionToString(int action) const {
    ChessMove move = actionToChessMove(action);
    return moveToString(move);
}

std::optional<int> ChessState::stringToAction(const std::string& moveStr) const {
    std::optional<ChessMove> move = stringToMove(moveStr);
    if (!move) {
        return std::nullopt;
    }
    return chessMoveToAction(*move);
}

std::string ChessState::toString() const {
    std::stringstream ss;
    
    // Print board
    ss << "  a b c d e f g h" << std::endl;
    for (int rank = 0; rank < 8; ++rank) {
        ss << (8 - rank) << " ";
        for (int file = 0; file < 8; ++file) {
            int square = getSquare(rank, file);
            Piece piece = board_[square];
            
            if (piece.is_empty()) {
                ss << ". ";
            } else {
                char pieceChar;
                switch (piece.type) {
                    case PieceType::PAWN:   pieceChar = 'p'; break;
                    case PieceType::KNIGHT: pieceChar = 'n'; break;
                    case PieceType::BISHOP: pieceChar = 'b'; break;
                    case PieceType::ROOK:   pieceChar = 'r'; break;
                    case PieceType::QUEEN:  pieceChar = 'q'; break;
                    case PieceType::KING:   pieceChar = 'k'; break;
                    default:                pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::WHITE) {
                    pieceChar = std::toupper(pieceChar);
                }
                
                ss << pieceChar << " ";
            }
        }
        ss << (8 - rank) << std::endl;
    }
    ss << "  a b c d e f g h" << std::endl;
    
    // Print current state
    ss << "Current player: " << (current_player_ == PieceColor::WHITE ? "White" : "Black") << std::endl;
    ss << "Castling rights: ";
    if (castling_rights_.white_kingside) ss << "K";
    if (castling_rights_.white_queenside) ss << "Q";
    if (castling_rights_.black_kingside) ss << "k";
    if (castling_rights_.black_queenside) ss << "q";
    if (!castling_rights_.white_kingside && !castling_rights_.white_queenside && 
        !castling_rights_.black_kingside && !castling_rights_.black_queenside) {
        ss << "-";
    }
    ss << std::endl;
    
    ss << "En passant square: ";
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        ss << squareToString(en_passant_square_);
    } else {
        ss << "-";
    }
    ss << std::endl;
    
    ss << "Halfmove clock: " << halfmove_clock_ << std::endl;
    ss << "Fullmove number: " << fullmove_number_ << std::endl;
    
    if (chess960_) {
        ss << "Chess960 mode: Yes" << std::endl;
        ss << "Original castling rook files (a=0, h=7):" << std::endl;
        ss << "  White: queenside=" << white_queenside_rook_file_ 
           << ", kingside=" << white_kingside_rook_file_ << std::endl;
        ss << "  Black: queenside=" << black_queenside_rook_file_
           << ", kingside=" << black_kingside_rook_file_ << std::endl;
    }
    
    ss << "FEN: " << toFEN() << std::endl;
    
    return ss.str();
}

bool ChessState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::CHESS) {
        return false;
    }
    
    try {
        const ChessState& otherChess = dynamic_cast<const ChessState&>(other);
        
        // Compare board positions
        for (int square = 0; square < NUM_SQUARES; ++square) {
            if (board_[square] != otherChess.board_[square]) {
                return false;
            }
        }
        
        // Compare game state
        if (current_player_ != otherChess.current_player_ ||
            castling_rights_.white_kingside != otherChess.castling_rights_.white_kingside ||
            castling_rights_.white_queenside != otherChess.castling_rights_.white_queenside ||
            castling_rights_.black_kingside != otherChess.castling_rights_.black_kingside ||
            castling_rights_.black_queenside != otherChess.castling_rights_.black_queenside ||
            en_passant_square_ != otherChess.en_passant_square_ ||
            halfmove_clock_ != otherChess.halfmove_clock_ ||
            fullmove_number_ != otherChess.fullmove_number_ ||
            chess960_ != otherChess.chess960_) {
            return false;
        }
        
        // For Chess960, also compare rook files
        if (chess960_ && (
            white_kingside_rook_file_ != otherChess.white_kingside_rook_file_ ||
            white_queenside_rook_file_ != otherChess.white_queenside_rook_file_ ||
            black_kingside_rook_file_ != otherChess.black_kingside_rook_file_ ||
            black_queenside_rook_file_ != otherChess.black_queenside_rook_file_)) {
            return false;
        }
        
        return true;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> ChessState::getMoveHistory() const {
    std::vector<int> result;
    result.reserve(move_history_.size());
    
    for (const auto& moveInfo : move_history_) {
        result.push_back(chessMoveToAction(moveInfo.move));
    }
    
    return result;
}

bool ChessState::validate() const {
    // Check that there is exactly one king of each color
    int whiteKings = 0;
    int blackKings = 0;
    
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (piece.type == PieceType::KING) {
            if (piece.color == PieceColor::WHITE) {
                whiteKings++;
            } else if (piece.color == PieceColor::BLACK) {
                blackKings++;
            }
        }
    }
    
    if (whiteKings != 1 || blackKings != 1) {
        return false;
    }
    
    // Check that the rook files for Chess960 are valid
    if (chess960_) {
        if (white_kingside_rook_file_ < 0 || white_kingside_rook_file_ > 7 ||
            white_queenside_rook_file_ < 0 || white_queenside_rook_file_ > 7 ||
            black_kingside_rook_file_ < 0 || black_kingside_rook_file_ > 7 ||
            black_queenside_rook_file_ < 0 || black_queenside_rook_file_ > 7) {
            return false;
        }
    }
    
    // Other validation checks could be added here
    
    return true;
}

// Move generation and validation methods
std::vector<ChessMove> ChessState::generateLegalMoves() const {
    if (!legal_moves_dirty_) {
        return cached_legal_moves_;
    }
    
    // Generate legal moves using rules
    cached_legal_moves_ = rules_->generateLegalMoves(
        *this, current_player_, castling_rights_, en_passant_square_);
    legal_moves_dirty_ = false;
    
    return cached_legal_moves_;
}

bool ChessState::isLegalMove(const ChessMove& move) const {
    return rules_->isLegalMove(
        *this, move, current_player_, castling_rights_, en_passant_square_);
}

void ChessState::makeMove(const ChessMove& move) {
    // if (!isLegalMove(move)) {
    //     throw std::runtime_error("Illegal move attempted");
    // }
    
    // Store move info for undoing
    MoveInfo moveInfo;
    moveInfo.move = move;
    moveInfo.captured_piece = getPiece(move.to_square);
    moveInfo.castling_rights = castling_rights_;
    moveInfo.en_passant_square = en_passant_square_;
    moveInfo.halfmove_clock = halfmove_clock_;
    moveInfo.fullmove_number = fullmove_number_;
    moveInfo.was_castle = false;
    moveInfo.was_en_passant = false;
    
    // Get the moving piece
    Piece piece = getPiece(move.from_square);
    
    // Update halfmove clock
    if (piece.type == PieceType::PAWN || !moveInfo.captured_piece.is_empty()) {
        // Pawn move or capture resets the clock
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }
    
    // Clear en passant target
    int old_ep_square = en_passant_square_;
    en_passant_square_ = -1;
    
    // Handle special pawn moves
    if (piece.type == PieceType::PAWN) {
        int fromRank = getRank(move.from_square);
        int toRank = getRank(move.to_square);
        int fromFile = getFile(move.from_square);
        // int toFile = getFile(move.to_square); // Currently unused
        
        // Check for two-square pawn move (set en passant target)
        if (std::abs(fromRank - toRank) == 2) {
            int epRank = (fromRank + toRank) / 2;
            en_passant_square_ = getSquare(epRank, fromFile);
        }
        
        // Check for en passant capture
        if (move.to_square == old_ep_square) {
            int capturedPawnRank = getRank(move.from_square);
            int capturedPawnFile = getFile(move.to_square);
            int capturedPawnSquare = getSquare(capturedPawnRank, capturedPawnFile);
            
            // Remove the captured pawn
            setPiece(capturedPawnSquare, Piece());
            moveInfo.was_en_passant = true;
        }
        
        // Handle promotion
        if (move.promotion_piece != PieceType::NONE) {
            piece.type = move.promotion_piece;
        }
    }
    
    // Handle castling
    if (piece.type == PieceType::KING && std::abs(getFile(move.from_square) - getFile(move.to_square)) == 2) {
        int rank = getRank(move.from_square);
        bool isKingside = getFile(move.to_square) > getFile(move.from_square);
        
        // Get original rook position based on castling rights
        int rookFile = getOriginalRookFile(isKingside, piece.color);
        int rookFromSquare = getSquare(rank, rookFile);
        
        // Determine rook's target square (in Chess960, this depends on the king's final position)
        int rookToFile;
        if (chess960_) {
            // In Chess960, the rook goes to the other side of the king
            rookToFile = isKingside ? getFile(move.to_square) - 1 : getFile(move.to_square) + 1;
        } else {
            // In standard chess, rook goes to fixed position
            rookToFile = isKingside ? 5 : 3;
        }
        
        int rookToSquare = getSquare(rank, rookToFile);
        
        // Move the rook
        Piece rook = getPiece(rookFromSquare);
        setPiece(rookFromSquare, Piece());
        setPiece(rookToSquare, rook);
        
        moveInfo.was_castle = true;
    }
    
    // Update castling rights
    castling_rights_ = rules_->getUpdatedCastlingRights(
        *this, move, piece, moveInfo.captured_piece, castling_rights_);
    
    // Move the piece
    setPiece(move.from_square, Piece());
    setPiece(move.to_square, piece);
    
    // Switch players
    current_player_ = oppositeColor(current_player_);
    
    // Update fullmove number
    if (current_player_ == PieceColor::WHITE) {
        fullmove_number_++;
    }
    
    // Add to move history
    move_history_.push_back(moveInfo);
    
    // Record position for repetition detection
    recordPosition();
    
    // Invalidate caches
    invalidateCache();
}

// Position handling for repetition detection
void ChessState::recordPosition() {
    uint64_t posHash = getHash();
    position_history_[posHash]++;
}

bool ChessState::isThreefoldRepetition() const {
    if (hash_dirty_) {
        updateHash();
    }
    
    auto it = position_history_.find(hash_);
    if (it != position_history_.end() && it->second >= 3) {
        return true;
    }
    
    return false;
}

// Check and check detection
bool ChessState::isInCheck(PieceColor color) const {
    if (color == PieceColor::NONE) {
        color = current_player_;
    }
    
    return rules_->isInCheck(*this, color);
}

bool ChessState::isSquareAttacked(int square, PieceColor by_color) const {
    return rules_->isSquareAttacked(*this, square, by_color);
}

// Utility methods
int ChessState::getKingSquare(PieceColor color) const {
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (piece.type == PieceType::KING && piece.color == color) {
            return square;
        }
    }
    return -1;  // King not found
}

void ChessState::invalidateCache() {
    legal_moves_dirty_ = true;
    terminal_check_dirty_ = true;
    
    // PERFORMANCE FIX: Return old tensors before invalidating caches
    clearTensorCache();
    
    // PERFORMANCE FIX: Invalidate tensor caches when game state changes
    tensor_cache_dirty_.store(true, std::memory_order_release);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_release);
}

ChessState::~ChessState() {
    // Return cached tensors to the pool to prevent memory leaks
    clearTensorCache();
}

void ChessState::clearTensorCache() const {
    // Return cached tensors to the global pool
    if (!cached_tensor_repr_.empty()) {
        core::GlobalTensorPool::getInstance().returnTensor(
            const_cast<std::vector<std::vector<std::vector<float>>>&>(cached_tensor_repr_),
            20, 8, 8);  // Chess uses 20 channels for basic representation
        cached_tensor_repr_.clear();
    }
    
    if (!cached_enhanced_tensor_repr_.empty()) {
        core::GlobalTensorPool::getInstance().returnTensor(
            const_cast<std::vector<std::vector<std::vector<float>>>&>(cached_enhanced_tensor_repr_),
            119, 8, 8);  // Chess uses 119 channels for enhanced representation
        cached_enhanced_tensor_repr_.clear();
    }
}

void ChessState::updateHash() const {
    hash_ = 0;
    
    // Hash board position
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (!piece.is_empty()) {
            int pieceIdx = static_cast<int>(piece.type) - 1;
            if (piece.color == PieceColor::BLACK) {
                pieceIdx += 6;  // Offset for black pieces
            }
            hash_ ^= zobrist_.getPieceHash(pieceIdx, square);
        }
    }
    
    // Hash current player
    if (current_player_ == PieceColor::BLACK) {
        hash_ ^= zobrist_.getPlayerHash(1);
    }
    
    // Hash castling rights 
    int castlingValue = 0;
    if (castling_rights_.white_kingside) castlingValue |= 1;
    if (castling_rights_.white_queenside) castlingValue |= 2;
    if (castling_rights_.black_kingside) castlingValue |= 4;
    if (castling_rights_.black_queenside) castlingValue |= 8;
    hash_ ^= zobrist_.getFeatureHash("castling", castlingValue);
    
    // Hash en passant square
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        hash_ ^= zobrist_.getFeatureHash("en_passant", en_passant_square_);
    }
    
    // Hash Chess960 flag
    if (chess960_) {
        hash_ ^= zobrist_.getFeatureHash("chess960", 1);
    }
    
    hash_dirty_ = false;
}

void ChessState::updateHashIncrementally(int square, const Piece& old_piece, const Piece& new_piece) {
    if (!old_piece.is_empty()) {
        int pieceIdx = static_cast<int>(old_piece.type) - 1;
        if (old_piece.color == PieceColor::BLACK) {
            pieceIdx += 6;  // Offset for black pieces
        }
        hash_ ^= zobrist_.getPieceHash(pieceIdx, square);
    }
    
    if (!new_piece.is_empty()) {
        int pieceIdx = static_cast<int>(new_piece.type) - 1;
        if (new_piece.color == PieceColor::BLACK) {
            pieceIdx += 6;  // Offset for black pieces
        }
        hash_ ^= zobrist_.getPieceHash(pieceIdx, square);
    }
}

// Move <-> Action conversion
ChessMove ChessState::actionToChessMove(int action) const {
    int fromSquare = (action >> 6) & 0x3F;
    int toSquare = action & 0x3F;
    int promotionCode = (action >> 12) & 0x7;
    
    PieceType promotionPiece = PieceType::NONE;
    switch (promotionCode) {
        case 1: promotionPiece = PieceType::QUEEN; break;
        case 2: promotionPiece = PieceType::ROOK; break;
        case 3: promotionPiece = PieceType::BISHOP; break;
        case 4: promotionPiece = PieceType::KNIGHT; break;
        default: promotionPiece = PieceType::NONE; break;
    }
    
    return {fromSquare, toSquare, promotionPiece};
}

int ChessState::chessMoveToAction(const ChessMove& move) const {
    int fromSquare = move.from_square & 0x3F;
    int toSquare = move.to_square & 0x3F;
    
    int promotionCode = 0;
    switch (move.promotion_piece) {
        case PieceType::QUEEN:  promotionCode = 1; break;
        case PieceType::ROOK:   promotionCode = 2; break;
        case PieceType::BISHOP: promotionCode = 3; break;
        case PieceType::KNIGHT: promotionCode = 4; break;
        default: promotionCode = 0; break;
    }
    
    return (promotionCode << 12) | (fromSquare << 6) | toSquare;
}

ChessState ChessState::cloneWithMove(const ChessMove& move) const {
    ChessState newState(*this);
    newState.makeMove(move);
    return newState;
}

// String conversion utilities
std::string ChessState::squareToString(int square) {
    if (square < 0 || square >= 64) {
        return "";
    }
    
    int rank = getRank(square);
    int file = getFile(square);
    
    char fileChar = 'a' + file;
    char rankChar = '8' - rank;
    
    return std::string({fileChar, rankChar});
}

int ChessState::stringToSquare(const std::string& squareStr) {
    if (squareStr.length() != 2) {
        return -1;
    }
    
    char fileChar = squareStr[0];
    char rankChar = squareStr[1];
    
    if (fileChar < 'a' || fileChar > 'h' || rankChar < '1' || rankChar > '8') {
        return -1;
    }
    
    int file = fileChar - 'a';
    int rank = '8' - rankChar;
    
    return getSquare(rank, file);
}

std::string ChessState::moveToString(const ChessMove& move) const {
    std::string result = squareToString(move.from_square) + squareToString(move.to_square);
    
    // Add promotion piece
    if (move.promotion_piece != PieceType::NONE) {
        char promotionChar = ' ';
        switch (move.promotion_piece) {
            case PieceType::QUEEN:  promotionChar = 'q'; break;
            case PieceType::ROOK:   promotionChar = 'r'; break;
            case PieceType::BISHOP: promotionChar = 'b'; break;
            case PieceType::KNIGHT: promotionChar = 'n'; break;
            default: break;
        }
        
        if (promotionChar != ' ') {
            result += promotionChar;
        }
    }
    
    return result;
}

std::optional<ChessMove> ChessState::stringToMove(const std::string& moveStr) const {
    // Parse algebraic notation
    if (moveStr.length() < 4) {
        return std::nullopt;
    }
    
    int fromSquare = stringToSquare(moveStr.substr(0, 2));
    int toSquare = stringToSquare(moveStr.substr(2, 2));
    
    if (fromSquare == -1 || toSquare == -1) {
        return std::nullopt;
    }
    
    // Check for promotion
    PieceType promotionPiece = PieceType::NONE;
    if (moveStr.length() >= 5) {
        char promotionChar = std::tolower(moveStr[4]);
        switch (promotionChar) {
            case 'q': promotionPiece = PieceType::QUEEN; break;
            case 'r': promotionPiece = PieceType::ROOK; break;
            case 'b': promotionPiece = PieceType::BISHOP; break;
            case 'n': promotionPiece = PieceType::KNIGHT; break;
            default: break;
        }
    }
    
    return ChessMove{fromSquare, toSquare, promotionPiece};
}

std::string ChessState::toSAN(const ChessMove& move) const {
    // Implementation of Standard Algebraic Notation (SAN)
    Piece piece = getPiece(move.from_square);
    if (piece.is_empty()) {
        return "";
    }
    
    std::string san;
    
    // Castling
    if (piece.type == PieceType::KING && 
        std::abs(getFile(move.from_square) - getFile(move.to_square)) == 2) {
        if (getFile(move.to_square) > getFile(move.from_square)) {
            return "O-O";  // Kingside castling
        } else {
            return "O-O-O";  // Queenside castling
        }
    }
    
    // Piece letter (except for pawns)
    if (piece.type != PieceType::PAWN) {
        char pieceChar = ' ';
        switch (piece.type) {
            case PieceType::KNIGHT: pieceChar = 'N'; break;
            case PieceType::BISHOP: pieceChar = 'B'; break;
            case PieceType::ROOK:   pieceChar = 'R'; break;
            case PieceType::QUEEN:  pieceChar = 'Q'; break;
            case PieceType::KING:   pieceChar = 'K'; break;
            default: break;
        }
        san += pieceChar;
    }
    
    // Disambiguation
    std::vector<ChessMove> legalMoves = generateLegalMoves();
    std::vector<ChessMove> ambiguousMoves;
    
    for (const auto& m : legalMoves) {
        if (m.to_square == move.to_square && m.from_square != move.from_square) {
            Piece p = getPiece(m.from_square);
            if (p.type == piece.type && p.color == piece.color) {
                ambiguousMoves.push_back(m);
            }
        }
    }
    
    if (!ambiguousMoves.empty()) {
        bool sameFile = false;
        bool sameRank = false;
        
        for (const auto& m : ambiguousMoves) {
            if (getFile(m.from_square) == getFile(move.from_square)) {
                sameFile = true;
            }
            if (getRank(m.from_square) == getRank(move.from_square)) {
                sameRank = true;
            }
        }
        
        if (!sameFile) {
            // Disambiguate by file
            san += 'a' + getFile(move.from_square);
        } else if (!sameRank) {
            // Disambiguate by rank
            san += '8' - getRank(move.from_square);
        } else {
            // Disambiguate by both
            san += squareToString(move.from_square);
        }
    } else if (piece.type == PieceType::PAWN && getFile(move.from_square) != getFile(move.to_square)) {
        // Pawn capture: indicate file of origin
        san += 'a' + getFile(move.from_square);
    }
    
    // Capture symbol
    Piece capturedPiece = getPiece(move.to_square);
    bool isCapture = !capturedPiece.is_empty();
    
    // Special case: en passant capture
    if (piece.type == PieceType::PAWN && move.to_square == en_passant_square_) {
        isCapture = true;
    }
    
    if (isCapture) {
        san += "x";
    }
    
    // Destination square
    san += squareToString(move.to_square);
    
    // Promotion
    if (move.promotion_piece != PieceType::NONE) {
        san += "=";
        char promotionChar = ' ';
        switch (move.promotion_piece) {
            case PieceType::QUEEN:  promotionChar = 'Q'; break;
            case PieceType::ROOK:   promotionChar = 'R'; break;
            case PieceType::BISHOP: promotionChar = 'B'; break;
            case PieceType::KNIGHT: promotionChar = 'N'; break;
            default: break;
        }
        san += promotionChar;
    }
    
    // Check and checkmate
    ChessState tempState(*this);
    tempState.makeMove(move);
    
    if (tempState.isInCheck(oppositeColor(piece.color))) {
        // Generate legal moves for the opponent after this move
        tempState.legal_moves_dirty_ = true;
        std::vector<ChessMove> opponentMoves = tempState.generateLegalMoves();
        
        if (opponentMoves.empty()) {
            san += "#";  // Checkmate
        } else {
            san += "+";  // Check
        }
    }
    
    return san;
}

std::optional<ChessMove> ChessState::fromSAN(const std::string& sanStr) const {
    if (sanStr.empty()) return std::nullopt;

    // 1. Castling
    if (sanStr == "O-O" || sanStr == "0-0") {
        int kingSquare = getKingSquare(current_player_);
        if (kingSquare == -1) return std::nullopt;
        // Determine target square based on Chess960 or standard
        int targetFile = chess960_ ? (getFile(kingSquare) + 2) : 6; // g-file for standard
        int targetSquare = getSquare(getRank(kingSquare), targetFile);
        ChessMove castle = {kingSquare, targetSquare};
        return isLegalMove(castle) ? std::optional<ChessMove>(castle) : std::nullopt;
    } else if (sanStr == "O-O-O" || sanStr == "0-0-0") {
        int kingSquare = getKingSquare(current_player_);
        if (kingSquare == -1) return std::nullopt;
        int targetFile = chess960_ ? (getFile(kingSquare) - 2) : 2; // c-file for standard
        int targetSquare = getSquare(getRank(kingSquare), targetFile);
        ChessMove castle = {kingSquare, targetSquare};
        return isLegalMove(castle) ? std::optional<ChessMove>(castle) : std::nullopt;
    }

    std::string cleanSan = sanStr;
    // Remove check/checkmate symbols for easier parsing
    if (!cleanSan.empty() && (cleanSan.back() == '#' || cleanSan.back() == '+')) {
        cleanSan.pop_back();
    }
    if (cleanSan.empty()) return std::nullopt;

    // 2. Promotion
    PieceType promotionPiece = PieceType::NONE;
    size_t promotionPos = cleanSan.find('=');
    if (promotionPos != std::string::npos && promotionPos + 1 < cleanSan.length()) {
        char promChar = cleanSan[promotionPos + 1];
        switch (std::toupper(promChar)) {
            case 'Q': promotionPiece = PieceType::QUEEN; break;
            case 'R': promotionPiece = PieceType::ROOK; break;
            case 'B': promotionPiece = PieceType::BISHOP; break;
            case 'N': promotionPiece = PieceType::KNIGHT; break;
            default: return std::nullopt; // Invalid promotion char
        }
        cleanSan = cleanSan.substr(0, promotionPos);
    }
    if (cleanSan.length() < 2) return std::nullopt;

    // 3. Destination square (always last 2 chars of remaining string)
    std::string toSquareStr = cleanSan.substr(cleanSan.length() - 2);
    int toSquare = stringToSquare(toSquareStr);
    if (toSquare == -1) return std::nullopt;
    cleanSan = cleanSan.substr(0, cleanSan.length() - 2);

    // 4. Piece Type and Capture
    PieceType pieceType = PieceType::PAWN;
    bool isCapture = false;
    char pieceChar = 'P'; // Default for pawn for matching

    if (!cleanSan.empty()) {
        char firstChar = cleanSan[0];
        if (std::isupper(firstChar)) {
            pieceChar = firstChar;
            switch (firstChar) {
                case 'N': pieceType = PieceType::KNIGHT; break;
                case 'B': pieceType = PieceType::BISHOP; break;
                case 'R': pieceType = PieceType::ROOK; break;
                case 'Q': pieceType = PieceType::QUEEN; break;
                case 'K': pieceType = PieceType::KING; break;
                default: return std::nullopt; // Invalid piece char
            }
            cleanSan = cleanSan.substr(1);
        } // else it's a pawn move, pieceType remains PAWN
    }
    
    if (!cleanSan.empty() && (cleanSan.back() == 'x')) {
        isCapture = true;
        cleanSan.pop_back();
    } else if (pieceType == PieceType::PAWN && !cleanSan.empty() && cleanSan.length() == 1 && cleanSan[0] >= 'a' && cleanSan[0] <= 'h') {
        // Pawn capture like "exd5", cleanSan is now "e"
        isCapture = true; 
    }


    // 5. Disambiguation (remaining part of cleanSan)
    int fromFile = -1;
    int fromRank = -1;

    if (!cleanSan.empty()) {
        if (cleanSan.length() == 1) {
            if (cleanSan[0] >= 'a' && cleanSan[0] <= 'h') fromFile = cleanSan[0] - 'a';
            else if (cleanSan[0] >= '1' && cleanSan[0] <= '8') fromRank = '8' - cleanSan[0];
            else return std::nullopt; // Invalid disambiguation
        } else if (cleanSan.length() == 2) {
            if (cleanSan[0] >= 'a' && cleanSan[0] <= 'h') fromFile = cleanSan[0] - 'a';
            else return std::nullopt;
            if (cleanSan[1] >= '1' && cleanSan[1] <= '8') fromRank = '8' - cleanSan[1];
            else return std::nullopt;
        } else {
            return std::nullopt; // Invalid disambiguation length
        }
    }

    // 6. Find matching legal move
    auto legalMoves = generateLegalMoves();
    for (const auto& move : legalMoves) {
        Piece p = getPiece(move.from_square);
        if (p.type != pieceType) continue;
        if (move.to_square != toSquare) continue;
        if (move.promotion_piece != promotionPiece) continue;

        if (fromFile != -1 && getFile(move.from_square) != fromFile) continue;
        if (fromRank != -1 && getRank(move.from_square) != fromRank) continue;

        // Check capture flag if it was specified
        bool moveIsCapture = !getPiece(move.to_square).is_empty() || 
                             (p.type == PieceType::PAWN && move.to_square == getEnPassantSquare());
        if (isCapture && !moveIsCapture) continue; 
        // If SAN implies non-capture, but move is capture (e.g. pawn move to occupied square not marked with x)
        // This can happen if SAN is minimal e.g. "e4" when e4 is a capture. Standard SAN requires 'x' for captures.
        // However, some contexts might omit it. For strict SAN parsing, we might reject.
        // For now, if `isCapture` is false, we don't explicitly check if `moveIsCapture` is true.

        return move; // Found a match
    }

    return std::nullopt;
}

} // namespace chess
} // namespace games
} // namespace alphazero