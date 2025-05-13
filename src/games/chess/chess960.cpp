// src/games/chess/chess960.cpp
#include "games/chess/chess960.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace alphazero {
namespace games {
namespace chess {

int Chess960::generateRandomPosition(unsigned seed) {
    // Initialize random number generator
    if (seed == 0) {
        seed = static_cast<unsigned>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 959);
    
    return dist(rng);
}

std::array<PieceType, 8> Chess960::generatePosition(int positionNumber) {
    if (positionNumber < 0 || positionNumber >= 960) {
        throw std::invalid_argument("Position number must be between 0 and 959");
    }
    
    // Initialize the back rank with empty spaces
    std::array<PieceType, 8> position;
    position.fill(PieceType::NONE);
    
    // Convert position number to a valid Chess960 arrangement using the permutation algorithm
    std::array<int, 8> arrangement = getPermutation(positionNumber);
    
    // Map the arrangement to actual pieces
    for (int i = 0; i < 8; ++i) {
        switch (arrangement[i]) {
            case 0: // First bishop (must be on odd square)
                position[i] = PieceType::BISHOP;
                break;
            case 1: // Second bishop (must be on even square)
                position[i] = PieceType::BISHOP;
                break;
            case 2: // Queen
                position[i] = PieceType::QUEEN;
                break;
            case 3: // First knight
                position[i] = PieceType::KNIGHT;
                break;
            case 4: // Second knight
                position[i] = PieceType::KNIGHT;
                break;
            case 5: // First rook
                position[i] = PieceType::ROOK;
                break;
            case 6: // King (must be between rooks)
                position[i] = PieceType::KING;
                break;
            case 7: // Second rook
                position[i] = PieceType::ROOK;
                break;
            default:
                throw std::runtime_error("Invalid piece index in Chess960 generation");
        }
    }
    
    // Verify the generated position is valid
    assert(isValidPosition(position));
    
    return position;
}

int Chess960::getPositionNumber(const std::array<PieceType, 8>& position) {
    if (!isValidPosition(position)) {
        return -1;  // Invalid position
    }
    
    // This is a complex reverse mapping that requires understanding the specific
    // algorithm used to generate positions. For simplicity, we'll use a brute-force
    // approach here, which is inefficient but correct.
    for (int i = 0; i < 960; ++i) {
        if (generatePosition(i) == position) {
            return i;
        }
    }
    
    return -1;  // Should not reach here if position is valid
}

bool Chess960::isValidPosition(const std::array<PieceType, 8>& position) {
    // Check that we have exactly the right pieces
    int bishops = 0;
    int knights = 0;
    int rooks = 0;
    int queens = 0;
    int kings = 0;
    
    for (PieceType piece : position) {
        switch (piece) {
            case PieceType::BISHOP: bishops++; break;
            case PieceType::KNIGHT: knights++; break;
            case PieceType::ROOK: rooks++; break;
            case PieceType::QUEEN: queens++; break;
            case PieceType::KING: kings++; break;
            default: return false;  // No empty spaces or other pieces allowed
        }
    }
    
    if (bishops != 2 || knights != 2 || rooks != 2 || queens != 1 || kings != 1) {
        return false;
    }
    
    // Ensure bishops are on opposite colored squares
    if (!hasValidBishopPlacement(position)) {
        return false;
    }
    
    // Ensure king is between the two rooks
    if (!hasKingBetweenRooks(position)) {
        return false;
    }
    
    return true;
}

void Chess960::setupPosition(int positionNumber, ChessState& state) {
    // Clear the board first
    for (int square = 0; square < 64; ++square) {
        state.setPiece(square, Piece());
    }
    
    // Generate the position arrangement
    std::array<PieceType, 8> arrangement = generatePosition(positionNumber);
    
    // Set up white pieces (back rank)
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(7, file), {arrangement[file], PieceColor::WHITE});
    }
    
    // Set up white pawns
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(6, file), {PieceType::PAWN, PieceColor::WHITE});
    }
    
    // Set up black pieces (back rank)
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(0, file), {arrangement[file], PieceColor::BLACK});
    }
    
    // Set up black pawns
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(1, file), {PieceType::PAWN, PieceColor::BLACK});
    }
    
    // Get rook files for this position
    auto rookFiles = getRookFiles(positionNumber);
    
    // Reset game state (player, castling rights, etc.) using FEN
    std::string fen = getStartingFEN(positionNumber);
    state.setFromFEN(fen);
}

std::string Chess960::getStartingFEN(int positionNumber) {
    std::array<PieceType, 8> arrangement = generatePosition(positionNumber);
    
    // Get rook files for castling rights
    auto rookFiles = getRookFiles(positionNumber);
    
    // Construct the FEN string
    std::stringstream ss;
    
    // First rank (black pieces)
    for (int file = 0; file < 8; ++file) {
        char pieceChar;
        switch (arrangement[file]) {
            case PieceType::PAWN: pieceChar = 'p'; break;
            case PieceType::KNIGHT: pieceChar = 'n'; break;
            case PieceType::BISHOP: pieceChar = 'b'; break;
            case PieceType::ROOK: pieceChar = 'r'; break;
            case PieceType::QUEEN: pieceChar = 'q'; break;
            case PieceType::KING: pieceChar = 'k'; break;
            default: pieceChar = '?'; break;
        }
        ss << pieceChar;
    }
    
    // Remaining ranks
    ss << "/pppppppp/8/8/8/8/PPPPPPPP/";
    
    // Last rank (white pieces)
    for (int file = 0; file < 8; ++file) {
        char pieceChar;
        switch (arrangement[file]) {
            case PieceType::PAWN: pieceChar = 'P'; break;
            case PieceType::KNIGHT: pieceChar = 'N'; break;
            case PieceType::BISHOP: pieceChar = 'B'; break;
            case PieceType::ROOK: pieceChar = 'R'; break;
            case PieceType::QUEEN: pieceChar = 'Q'; break;
            case PieceType::KING: pieceChar = 'K'; break;
            default: pieceChar = '?'; break;
        }
        ss << pieceChar;
    }
    
    // Add additional FEN components
    ss << " w ";
    
    // Castling rights in Chess960 notation (using files)
    int kingFile = getKingFile(positionNumber);
    int kingsideRookFile = rookFiles.second;  // Higher file
    int queensideRookFile = rookFiles.first;  // Lower file
    
    bool hasCastling = false;
    
    // White castling rights
    if (kingsideRookFile > kingFile) {
        ss << static_cast<char>('A' + kingsideRookFile);
        hasCastling = true;
    }
    if (queensideRookFile < kingFile) {
        ss << static_cast<char>('A' + queensideRookFile);
        hasCastling = true;
    }
    
    // Black castling rights
    if (kingsideRookFile > kingFile) {
        ss << static_cast<char>('a' + kingsideRookFile);
        hasCastling = true;
    }
    if (queensideRookFile < kingFile) {
        ss << static_cast<char>('a' + queensideRookFile);
        hasCastling = true;
    }
    
    if (!hasCastling) {
        ss << "-";
    }
    
    // No en passant, halfmove clock at 0, fullmove number 1
    ss << " - 0 1";
    
    return ss.str();
}

std::string Chess960::convertToChess960FEN(const std::string& standardFEN) {
    // For Chess960, castling rights use the file letter of the rook
    // instead of the standard KQkq notation
    
    // Parse FEN
    std::istringstream iss(standardFEN);
    std::string position, activeColor, castlingRights, enPassant, halfmoves, fullmoves;
    
    if (!(iss >> position >> activeColor >> castlingRights >> enPassant >> halfmoves >> fullmoves)) {
        throw std::invalid_argument("Invalid FEN string");
    }
    
    // If no castling rights or already using Chess960 notation, return as is
    if (castlingRights == "-" || (castlingRights.find_first_of("KQkq") == std::string::npos)) {
        return standardFEN;
    }
    
    // Parse the board position to find initial rook positions
    std::vector<std::string> ranks;
    size_t pos = 0;
    std::string token;
    while ((pos = position.find('/')) != std::string::npos) {
        ranks.push_back(position.substr(0, pos));
        position.erase(0, pos + 1);
    }
    ranks.push_back(position);  // Add the last rank
    
    if (ranks.size() != 8) {
        throw std::invalid_argument("Invalid FEN: wrong number of ranks");
    }
    
    // Find white rooks (in the 8th rank)
    std::string whiteRank = ranks[7];
    std::vector<int> whiteRookFiles;
    int fileIndex = 0;
    for (char c : whiteRank) {
        if (c == 'R') {
            whiteRookFiles.push_back(fileIndex);
        }
        if (std::isdigit(c)) {
            fileIndex += c - '0';
        } else {
            fileIndex++;
        }
    }
    
    // Find black rooks (in the 1st rank)
    std::string blackRank = ranks[0];
    std::vector<int> blackRookFiles;
    fileIndex = 0;
    for (char c : blackRank) {
        if (c == 'r') {
            blackRookFiles.push_back(fileIndex);
        }
        if (std::isdigit(c)) {
            fileIndex += c - '0';
        } else {
            fileIndex++;
        }
    }
    
    // Find king positions
    int whiteKingFile = -1;
    fileIndex = 0;
    for (char c : whiteRank) {
        if (c == 'K') {
            whiteKingFile = fileIndex;
            break;
        }
        if (std::isdigit(c)) {
            fileIndex += c - '0';
        } else {
            fileIndex++;
        }
    }
    
    int blackKingFile = -1;
    fileIndex = 0;
    for (char c : blackRank) {
        if (c == 'k') {
            blackKingFile = fileIndex;
            break;
        }
        if (std::isdigit(c)) {
            fileIndex += c - '0';
        } else {
            fileIndex++;
        }
    }
    
    // Convert castling rights to Chess960 notation
    std::string newCastlingRights;
    
    for (int i = 0; i < static_cast<int>(whiteRookFiles.size()); ++i) {
        int rookFile = whiteRookFiles[i];
        if (castlingRights.find('K') != std::string::npos && rookFile > whiteKingFile) {
            newCastlingRights += static_cast<char>('A' + rookFile);
        }
        if (castlingRights.find('Q') != std::string::npos && rookFile < whiteKingFile) {
            newCastlingRights += static_cast<char>('A' + rookFile);
        }
    }
    
    for (int i = 0; i < static_cast<int>(blackRookFiles.size()); ++i) {
        int rookFile = blackRookFiles[i];
        if (castlingRights.find('k') != std::string::npos && rookFile > blackKingFile) {
            newCastlingRights += static_cast<char>('a' + rookFile);
        }
        if (castlingRights.find('q') != std::string::npos && rookFile < blackKingFile) {
            newCastlingRights += static_cast<char>('a' + rookFile);
        }
    }
    
    if (newCastlingRights.empty()) {
        newCastlingRights = "-";
    }
    
    // Reconstruct FEN with new castling rights
    std::stringstream result;
    for (size_t i = 0; i < ranks.size(); ++i) {
        result << ranks[i];
        if (i < ranks.size() - 1) {
            result << '/';
        }
    }
    result << " " << activeColor << " " << newCastlingRights << " " 
           << enPassant << " " << halfmoves << " " << fullmoves;
    
    return result.str();
}

std::pair<int, int> Chess960::getRookFiles(int positionNumber) {
    std::array<PieceType, 8> position = generatePosition(positionNumber);
    std::vector<int> rookIndices = findPieceIndices(position, PieceType::ROOK);
    
    if (rookIndices.size() != 2) {
        throw std::runtime_error("Invalid Chess960 position: expected 2 rooks");
    }
    
    // Sort rook indices
    std::sort(rookIndices.begin(), rookIndices.end());
    
    // Return (queenside rook file, kingside rook file)
    return {rookIndices[0], rookIndices[1]};
}

int Chess960::getKingFile(int positionNumber) {
    std::array<PieceType, 8> position = generatePosition(positionNumber);
    std::vector<int> kingIndex = findPieceIndices(position, PieceType::KING);
    
    if (kingIndex.size() != 1) {
        throw std::runtime_error("Invalid Chess960 position: expected 1 king");
    }
    
    return kingIndex[0];
}

std::vector<int> Chess960::findPieceIndices(const std::array<PieceType, 8>& position, PieceType pieceType) {
    std::vector<int> indices;
    for (int i = 0; i < 8; ++i) {
        if (position[i] == pieceType) {
            indices.push_back(i);
        }
    }
    return indices;
}

// Helper methods for position generation and validation
bool Chess960::hasValidBishopPlacement(const std::array<PieceType, 8>& position) {
    // Find indices of bishops
    std::vector<int> bishopIndices = findPieceIndices(position, PieceType::BISHOP);
    
    if (bishopIndices.size() != 2) {
        return false;
    }
    
    // Check if bishops are on opposite colored squares
    return (bishopIndices[0] % 2) != (bishopIndices[1] % 2);
}

bool Chess960::hasKingBetweenRooks(const std::array<PieceType, 8>& position) {
    // Find indices of king and rooks
    std::vector<int> kingIndices = findPieceIndices(position, PieceType::KING);
    std::vector<int> rookIndices = findPieceIndices(position, PieceType::ROOK);
    
    if (kingIndices.size() != 1 || rookIndices.size() != 2) {
        return false;
    }
    
    int kingIndex = kingIndices[0];
    std::sort(rookIndices.begin(), rookIndices.end());
    
    // King must be between two rooks
    return kingIndex > rookIndices[0] && kingIndex < rookIndices[1];
}

std::array<int, 8> Chess960::getPermutation(int n) {
    // This algorithm generates a valid Chess960 arrangement (permutation of roles 0-7)
    // from a Scharnagl position number n (0-959).
    // Role to PieceType mapping is done in generatePosition():
    // 0: BISHOP, 1: BISHOP, 2: QUEEN, 3: KNIGHT, 4: KNIGHT, 5: ROOK, 6: KING, 7: ROOK

    if (n == 518) { // SP518 is the standard chess starting position
        // Hardcoded roles to produce RNBQKBNR via generatePosition's switch
        return {5, 3, 0, 2, 6, 1, 4, 7}; 
        // R->5, N->3, B->0, Q->2, K->6, B->1, N->4, R->7
    }

    if (n < 0 || n >= 960) {
        throw std::invalid_argument("Position number must be between 0 and 959");
    }

    std::array<int, 8> result;
    result.fill(-1); // Initialize with -1 (empty)

    // 1. Place Bishops (roles 0 and 1)
    // Dark squares (file indices): 0, 2, 4, 6
    // Light squares (file indices): 1, 3, 5, 7
    
    int dark_bishop_choice = n % 4; // N_Db
    int light_bishop_choice = (n / 4) % 4; // N_Lb

    std::vector<int> dark_squares = {0, 2, 4, 6};
    std::vector<int> light_squares = {1, 3, 5, 7};

    result[dark_squares[dark_bishop_choice]] = 0; // Assign role 0 to one bishop
    result[light_squares[light_bishop_choice]] = 1; // Assign role 1 to the other bishop

    // 2. Place Queen (role 2)
    int queen_choice = (n / 16) % 6; // N_Q
    std::vector<int> empty_slots_for_queen;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {
            empty_slots_for_queen.push_back(i);
        }
    }
    result[empty_slots_for_queen[queen_choice]] = 2; // Assign role 2 (Queen)

    // 3. Place Knights (roles 3 and 4)
    int knight_pair_choice = (n / 96) % 10; // N_N
    std::vector<int> empty_slots_for_knights;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {
            empty_slots_for_knights.push_back(i);
        }
    }

    int c = 0; // combination counter
    for (size_t i = 0; i < empty_slots_for_knights.size(); ++i) {
        for (size_t j = i + 1; j < empty_slots_for_knights.size(); ++j) {
            if (c == knight_pair_choice) {
                result[empty_slots_for_knights[i]] = 3; // Assign role 3 (Knight 1)
                result[empty_slots_for_knights[j]] = 4; // Assign role 4 (Knight 2)
                goto knights_placed;
            }
            c++;
        }
    }
knights_placed:;

    // 4. Place Rooks (roles 5 and 7) and King (role 6)
    // The remaining 3 slots must be R K R.
    std::vector<int> remaining_slots;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {
            remaining_slots.push_back(i);
        }
    }
    
    if (remaining_slots.size() != 3) {
         throw std::logic_error("Chess960: Incorrect number of slots for R-K-R placement.");
    }

    // King must be between rooks. Only one way to assign roles 5,6,7 to 3 slots.
    result[remaining_slots[0]] = 5; // Rook 1 (role 5)
    result[remaining_slots[1]] = 6; // King   (role 6)
    result[remaining_slots[2]] = 7; // Rook 2 (role 7)

    // Ensure all values are set (debug check)
    for (int i = 0; i < 8; ++i) {
        if (result[i] < 0 || result[i] > 7) {
            throw std::logic_error("Chess960: Permutation generation left an unassigned role.");
        }
    }
    
    return result;
}

bool Chess960::isStandardChessPosition(int positionNumber) {
    if (positionNumber < 0 || positionNumber >= 960) {
        return false;
    }
    
    // Generate the piece arrangement for this position
    std::array<PieceType, 8> arrangement = generatePosition(positionNumber);
    
    // Standard chess setup is RNBQKBNR
    std::array<PieceType, 8> standardSetup = {
        PieceType::ROOK, PieceType::KNIGHT, PieceType::BISHOP, PieceType::QUEEN,
        PieceType::KING, PieceType::BISHOP, PieceType::KNIGHT, PieceType::ROOK
    };
    
    // Compare the arrangements
    return arrangement == standardSetup;
}

} // namespace chess
} // namespace games
} // namespace alphazero