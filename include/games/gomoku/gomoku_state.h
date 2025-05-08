#ifndef GOMOKU_STATE_H
#define GOMOKU_STATE_H

#include "core/igamestate.h"
#include "utils/zobrist_hash.h"
#include "core/export_macros.h"
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <random>
#include <iomanip>
#include <unordered_set>

namespace alphazero {
namespace games {
namespace gomoku {

// Constants for players
constexpr int BLACK = 1;
constexpr int WHITE = 2;

// Forward declarations
class GomokuRules;

/**
 * @brief Gomoku game state implementation
 * 
 * Implements the game state interface for Gomoku (Five in a Row),
 * where players take turns placing stones on a board and try to
 * get five in a row horizontally, vertically, or diagonally.
 */
class ALPHAZERO_API GomokuState : public core::IGameState {
public:
    /**
     * @brief Constructor with configurable board size and rule options
     * 
     * @param board_size Board size (default 15x15)
     * @param use_renju Whether to use Renju rules
     * @param use_omok Whether to use Omok rules
     * @param seed Random seed for initialization (default 0)
     * @param use_pro_long_opening Whether to use pro-long opening rules
     */
    explicit GomokuState(int board_size = 15, bool use_renju = false, bool use_omok = false, 
                        int seed = 0, bool use_pro_long_opening = false);
    
    /**
     * @brief Copy constructor
     * 
     * @param other The GomokuState to copy from
     */
    GomokuState(const GomokuState& other);
    
    /**
     * @brief Get all legal moves
     * 
     * @return Vector of legal actions
     */
    std::vector<int> getLegalMoves() const override {
        return get_valid_moves();
    }
    
    /**
     * @brief Check if a specific move is legal
     * 
     * @param action The action to check
     * @return true if legal, false otherwise
     */
    bool isLegalMove(int action) const override {
        return is_move_valid(action);
    }
    
    /**
     * @brief Execute a move
     * 
     * @param action The action to execute
     */
    void makeMove(int action) override {
        if (!is_move_valid(action)) {
            throw core::IllegalMoveException("Illegal move", action);
        }
        make_move(action, current_player);
    }
    
    /**
     * @brief Undo the last move
     * 
     * @return true if a move was undone, false otherwise
     */
    bool undoMove() override {
        if (move_history.empty()) {
            return false;
        }
        
        int last_action = move_history.back();
        undo_move(last_action);
        
        // Explicitly ensure current_player is set correctly
        if (!move_history.empty()) {
            int count = static_cast<int>(move_history.size());
            current_player = (count % 2 == 0) ? BLACK : WHITE;
        } else {
            current_player = BLACK; // Reset to starting player if all moves undone
        }
        
        return true;
    }
    
    /**
     * @brief Check if the game state is terminal
     * 
     * @return true if terminal, false otherwise
     */
    bool isTerminal() const override {
        return is_terminal();
    }
    
    /**
     * @brief Get the result of the game
     * 
     * @return Game result
     */
    core::GameResult getGameResult() const override {
        if (!is_terminal()) return core::GameResult::ONGOING;
        int winner = get_winner();
        if (winner == BLACK) return core::GameResult::WIN_PLAYER1;
        if (winner == WHITE) return core::GameResult::WIN_PLAYER2;
        return core::GameResult::DRAW;
    }
    
    /**
     * @brief Get the current player
     * 
     * @return Current player (1 or 2)
     */
    int getCurrentPlayer() const override {
        return current_player;
    }
    
    /**
     * @brief Get the board size
     * 
     * @return Board size
     */
    int getBoardSize() const override {
        return board_size;
    }
    
    /**
     * @brief Get the action space size
     * 
     * @return Board size squared
     */
    int getActionSpaceSize() const override {
        return board_size * board_size;
    }
    
    /**
     * @brief Get tensor representation for neural network
     * 
     * @return 3D tensor with basic features
     */
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override {
        std::vector<std::vector<std::vector<float>>> tensor(
            3, std::vector<std::vector<float>>(board_size, std::vector<float>(board_size, 0.0f)));
        int total = board_size * board_size;
        
        // We need BLACK stones in channel 0, WHITE stones in channel 1
        for (int a = 0; a < total; ++a) {
            int row = a / board_size;
            int col = a % board_size;
            if (board[a] == BLACK) {
                tensor[0][row][col] = 1.0f;
            } else if (board[a] == WHITE) {
                tensor[1][row][col] = 1.0f;
            }
        }
        
        // Current player plane (all 1s if BLACK to play, all 0s if WHITE to play)
        float value = (current_player == BLACK) ? 1.0f : 0.0f;
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                tensor[2][i][j] = value;
            }
        }
        
        return tensor;
    }
    
    /**
     * @brief Get enhanced tensor representation with history
     * 
     * @return 3D tensor with enhanced features
     */
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        const int num_channels = 17;
        std::vector<std::vector<std::vector<float>>> tensor(
            num_channels, std::vector<std::vector<float>>(
                board_size, std::vector<float>(board_size, 0.0f)));
        
        // Last 8 moves for each player
        int history_len = static_cast<int>(move_history.size());
        for (int i = 0; i < std::min(16, history_len); ++i) {
            int move = move_history[history_len - 1 - i];
            int row = move / board_size;
            int col = move % board_size;
            int player = ((history_len - 1 - i) % 2 == 0) ? 3 - current_player : current_player;
            int channel_idx = (player == BLACK) ? (i / 2) * 2 : (i / 2) * 2 + 1;
            if (channel_idx < 16) {
                tensor[channel_idx][row][col] = 1.0f;
            }
        }
        // Current player plane
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                tensor[16][i][j] = (current_player == BLACK) ? 1.0f : 0.0f;
            }
        }
        return tensor;
    }
    
    /**
     * @brief Get hash for transposition table
     * 
     * @return 64-bit hash
     */
    uint64_t getHash() const override {
        return compute_hash_signature();
    }
    
    /**
     * @brief Clone the current state
     * 
     * @return Unique pointer to a new copy
     */
    std::unique_ptr<core::IGameState> clone() const override {
        return std::make_unique<GomokuState>(*this);
    }
    
    /**
     * @brief Convert action to string representation
     * 
     * @param action The action to convert
     * @return String representation (e.g., "H8")
     */
    std::string actionToString(int action) const override {
        if (action < 0 || action >= board_size * board_size) {
            return "Invalid";
        }
        
        int row = action / board_size;
        int col = action % board_size;
        
        std::string colStr;
        colStr.push_back('A' + col);
        
        return colStr + std::to_string(board_size - row);
    }
    
    /**
     * @brief Convert string representation to action
     * 
     * @param moveStr String representation (e.g., "H8")
     * @return Optional action
     */
    std::optional<int> stringToAction(const std::string& moveStr) const override {
        if (moveStr.length() < 2) {
            return std::nullopt;
        }
        
        char colChar = std::toupper(moveStr[0]);
        if (colChar < 'A' || colChar >= 'A' + board_size) {
            return std::nullopt;
        }
        
        int col = colChar - 'A';
        
        std::string rowStr = moveStr.substr(1);
        int row = 0;
        try {
            row = std::stoi(rowStr);
        } catch (...) {
            return std::nullopt;
        }
        
        if (row <= 0 || row > board_size) {
            return std::nullopt;
        }
        
        row = board_size - row;
        return row * board_size + col;
    }
    
    /**
     * @brief Get string representation of the board
     * 
     * @return String representation
     */
    std::string toString() const override {
        std::stringstream ss;
        
        // Print column labels
        ss << "  ";
        for (int j = 0; j < board_size; ++j) {
            ss << static_cast<char>('A' + j) << " ";
        }
        ss << std::endl;
        
        // Print board with row labels
        for (int i = 0; i < board_size; ++i) {
            ss << std::setw(2) << (board_size - i) << " ";
            
            for (int j = 0; j < board_size; ++j) {
                int idx = i * board_size + j;
                if (board[idx] == 0) {
                    // Empty intersection
                    if (i == 0) {
                        if (j == 0) {
                            ss << "┌─";
                        } else if (j == board_size - 1) {
                            ss << "┐ ";
                        } else {
                            ss << "┬─";
                        }
                    } else if (i == board_size - 1) {
                        if (j == 0) {
                            ss << "└─";
                        } else if (j == board_size - 1) {
                            ss << "┘ ";
                        } else {
                            ss << "┴─";
                        }
                    } else {
                        if (j == 0) {
                            ss << "├─";
                        } else if (j == board_size - 1) {
                            ss << "┤ ";
                        } else {
                            ss << "┼─";
                        }
                    }
                } else if (board[idx] == 1) {
                    // Player 1 (usually black)
                    ss << "X ";
                } else {
                    // Player 2 (usually white)
                    ss << "O ";
                }
            }
            
            ss << std::endl;
        }
        
        // Add current player info
        ss << "Current player: " << (current_player == 1 ? "X" : "O") << std::endl;
        
        return ss.str();
    }
    
    /**
     * @brief Check equality with another game state
     * 
     * @param other The other game state
     * @return true if equal, false otherwise
     */
    bool equals(const core::IGameState& other) const override {
        if (other.getGameType() != core::GameType::GOMOKU) return false;
        const GomokuState* o = dynamic_cast<const GomokuState*>(&other);
        if (!o) return false;
        return board_equal(*o);
    }
    
    /**
     * @brief Get the history of moves
     * 
     * @return Vector of actions
     */
    std::vector<int> getMoveHistory() const override {
        return move_history;
    }
    
    /**
     * @brief Validate the game state for consistency
     * 
     * @return true if valid, false otherwise
     */
    bool validate() const override {
        int count_p1 = 0;
        int count_p2 = 0;
        int total_cells = board_size * board_size;
        for (int a = 0; a < total_cells; ++a) {
            if (is_bit_set(BLACK, a)) count_p1++;
            else if (is_bit_set(WHITE, a)) count_p2++;
        }
        if (current_player == BLACK) {
            return count_p1 == count_p2;
        } else {
            return count_p1 == count_p2 + 1;
        }
    }
    
    // FOR TESTING ONLY: Direct board setup methods
    void setStone(int position, int player) {
        // Ensure we're not overwriting existing stones
        if (is_occupied(position)) {
            clear_bit(BLACK, position);
            clear_bit(WHITE, position);
        }
        
        // Place the stone using the player's index
        if (player == BLACK || player == WHITE) {
            set_bit(player, position);
            hash_dirty = true;
            valid_moves_dirty = true;
            winner_check_dirty = true;
        }
    }
    
    void setCurrentPlayer(int player) {
        if (player == BLACK || player == WHITE) {
            current_player = player;
            hash_dirty = true;
        }
    }
    
private:
    /**
     * @brief Initialize Zobrist hashing
     * 
     * @return Zobrist hash object
     */
    core::ZobristHash initZobrist() const {
        core::ZobristHash zobrist(board_size * board_size, 2, 2, 42);
        return zobrist;
    }
    
    int board_size;                   // Board size (typically 15x15)
    int current_player;               // Current player (1 or 2)
    std::vector<int> board;           // Flattened board state (0=empty, 1=player1, 2=player2)
    std::vector<int> move_history;    // History of moves played
    core::ZobristHash zobrist;        // Zobrist hash generator
    uint64_t hash_key_ = 0;            // Current position hash
    
    // Additional members needed by the implementation
    bool use_renju = false;            // Whether to use Renju rules
    bool use_omok = false;             // Whether to use Omok rules
    bool use_pro_long_opening = false; // Whether to use pro-long opening
    int black_first_stone = -1;        // First stone played by black
    mutable std::unordered_set<int> cached_valid_moves; // Cache for valid moves
    mutable bool valid_moves_dirty = true;     // Whether valid moves cache is dirty
    mutable int cached_winner = 0;             // Cached winner (0=none, 1/2=player)
    mutable bool winner_check_dirty = true;    // Whether winner check is dirty
    mutable uint64_t hash_signature = 0;       // Position hash signature
    mutable bool hash_dirty = true;            // Whether hash is dirty
    int num_words;                     // Number of 64-bit words for bitboards
    std::vector<std::vector<uint64_t>> player_bitboards; // Bitboard representation
    int dirs[8];                       // Direction vectors for pattern checking
    std::shared_ptr<GomokuRules> rules; // Rules object for evaluating positions
    int action = -1;                   // Last action played
    
    // Helper functions
    bool is_bit_set(int player_index, int action) const noexcept;
    void set_bit(int player_index, int action);
    void clear_bit(int player_index, int action) noexcept;
    std::pair<int, int> action_to_coords_pair(int action) const noexcept;
    int coords_to_action(int x, int y) const;
    bool in_bounds(int x, int y) const;
    int count_total_stones() const noexcept;
    
    // Game state methods
    void refresh_winner_cache() const;
    bool is_terminal() const;
    bool is_stalemate() const;
    int get_winner() const;
    std::vector<int> get_valid_moves() const;
    void refresh_valid_moves_cache() const;
    bool is_move_valid(int action) const;
    uint64_t compute_hash_signature() const;
    bool board_equal(const GomokuState& other) const;
    void make_move(int action, int player);
    void undo_move(int action);
    void invalidate_caches();
    bool is_occupied(int action) const;
    std::vector<std::vector<int>> get_board() const;
    
    // MCTS/NN support
    GomokuState apply_action(int action) const;
    std::vector<std::vector<std::vector<float>>> to_tensor() const;
    int get_action(const GomokuState& child_state) const;
    std::vector<int> get_previous_moves(int player, int count) const;
    GomokuState copy() const;
    
    // Game logic helpers
    bool is_pro_long_move_ok(int action, int stone_count) const;
};

} // namespace gomoku
} // namespace games
} // namespace alphazero

#endif // GOMOKU_STATE_H
