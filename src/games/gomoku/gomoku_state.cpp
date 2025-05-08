// gomoku_state.cpp
#include "games/gomoku/gomoku_state.h"
#include "games/gomoku/gomoku_rules.h"
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>
#include "utils/zobrist_hash.h"

namespace alphazero {
namespace games {
namespace gomoku {

// Constructor with initialization of caching fields
GomokuState::GomokuState(int board_size, bool use_renju, bool use_omok, int seed, bool use_pro_long_opening) 
    : IGameState(core::GameType::GOMOKU),
      board_size(board_size),
      current_player(BLACK),
      action(-1),
      use_renju(use_renju),
      use_omok(use_omok),
      use_pro_long_opening(use_pro_long_opening),
      black_first_stone(-1),
      valid_moves_dirty(true),
      cached_winner(0),
      winner_check_dirty(true),
      hash_signature(0),
      hash_dirty(true),
      move_history(),
      zobrist(board_size * board_size, 2, 2) {
    
    int total_cells = board_size * board_size;
    num_words = (total_cells + 63) / 64;
    
    // Initialize bitboards with zeros
    player_bitboards.resize(2, std::vector<uint64_t>(num_words, 0));
    
    // Directions for line scanning (dx,dy pairs)
    dirs[0] = 0;   // dx=0
    dirs[1] = 1;   // dy=1   (vertical)
    dirs[2] = 1;   // dx=1
    dirs[3] = 0;   // dy=0   (horizontal)
    dirs[4] = 1;   // dx=1
    dirs[5] = 1;   // dy=1   (diag-down)
    dirs[6] = -1;  // dx=-1
    dirs[7] = 1;   // dy=1   (diag-up)
    
    // Create rules instance
    rules = std::make_shared<GomokuRules>(board_size);
    
    // Set up board access functions for rules
    rules->setBoardAccessor(
        [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); },
        [this](int x, int y) { return this->coords_to_action(x, y); },
        [this](int a) { return this->action_to_coords_pair(a); },
        [this](int x, int y) { return this->in_bounds(x, y); }
    );
    
    // Initialize features for game rules
    zobrist.addFeature("renju", 2);        // Feature for Renju rules (0=off, 1=on)
    zobrist.addFeature("omok", 2);         // Feature for Omok rules (0=off, 1=on)
    zobrist.addFeature("pro_long", 2);     // Feature for pro-long opening (0=off, 1=on)
    
    // Optional seed initialization
    if (seed != 0) {
        std::srand(seed);
    } else {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }
}

// Copy constructor with cache preservation
GomokuState::GomokuState(const GomokuState& other) 
    : IGameState(core::GameType::GOMOKU),
      board_size(other.board_size),
      current_player(other.current_player),
      player_bitboards(other.player_bitboards),
      num_words(other.num_words),
      action(other.action),
      use_renju(other.use_renju),
      use_omok(other.use_omok),
      use_pro_long_opening(other.use_pro_long_opening),
      black_first_stone(other.black_first_stone),
      cached_valid_moves(other.cached_valid_moves),
      valid_moves_dirty(other.valid_moves_dirty),
      cached_winner(other.cached_winner),
      winner_check_dirty(other.winner_check_dirty),
      hash_signature(other.hash_signature),
      hash_dirty(other.hash_dirty),
      move_history(other.move_history),
      zobrist(other.zobrist) {
    
    // Copy the directions array
    for (int i = 0; i < 8; i++) {
        dirs[i] = other.dirs[i];
    }
    
    // Create new rules instance
    rules = std::make_shared<GomokuRules>(board_size);
    
    // Set up board access functions for rules
    rules->setBoardAccessor(
        [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); },
        [this](int x, int y) { return this->coords_to_action(x, y); },
        [this](int a) { return this->action_to_coords_pair(a); },
        [this](int x, int y) { return this->in_bounds(x, y); }
    );
}

// Create a deep copy
GomokuState GomokuState::copy() const {
    return GomokuState(*this);
}

// Bitboard operations

bool GomokuState::is_bit_set(int player_index, int action) const noexcept {
    // Convert player_index from 1-based (1,2) to 0-based (0,1) if needed
    int adjusted_index = (player_index >= 1 && player_index <= 2) ? player_index - 1 : player_index;
    
    // Early bounds check to avoid out-of-bounds access
    if (adjusted_index < 0 || adjusted_index >= 2 || action < 0 || action >= board_size * board_size) {
        return false;
    }
    
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Additional bounds check for word_idx
    if (word_idx >= num_words) {
        return false;
    }
    
    // Use uint64_t mask for proper bit manipulation
    uint64_t mask = static_cast<uint64_t>(1) << bit_idx;
    return (player_bitboards[adjusted_index][word_idx] & mask) != 0;
}

void GomokuState::set_bit(int player_index, int action) {
    // Convert player_index from 1-based (1,2) to 0-based (0,1) if needed
    int adjusted_index = (player_index >= 1 && player_index <= 2) ? player_index - 1 : player_index;
    
    if (adjusted_index < 0 || adjusted_index >= 2) {
        throw std::runtime_error("Player index out of range");
    }
    
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use |= for optimal bit setting
    player_bitboards[adjusted_index][word_idx] |= (static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    invalidate_caches();
}

void GomokuState::clear_bit(int player_index, int action) noexcept {
    // Convert player_index from 1-based (1,2) to 0-based (0,1) if needed
    int adjusted_index = (player_index >= 1 && player_index <= 2) ? player_index - 1 : player_index;
    
    if (adjusted_index < 0 || adjusted_index >= 2) {
        return;
    }
    
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use &= with negated mask for optimal bit clearing
    player_bitboards[adjusted_index][word_idx] &= ~(static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    invalidate_caches();
}

std::pair<int, int> GomokuState::action_to_coords_pair(int action) const noexcept {
    return {action / board_size, action % board_size};
}

int GomokuState::coords_to_action(int x, int y) const {
    return x * board_size + y;
}

int GomokuState::count_total_stones() const noexcept {
    int total = 0;
    
    for (int p = 0; p < 2; p++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p][w];
            
            // Use __builtin_popcountll for fast bit counting if available
            #if defined(__GNUC__) || defined(__clang__)
                total += __builtin_popcountll(chunk);
            #else
                // Fallback to manual counting with Brian Kernighan's algorithm
                while (chunk != 0) {
                    chunk &= (chunk - 1);  // Clear lowest set bit
                    total++;
                }
            #endif
        }
    }
    
    return total;
}

// Game state methods

void GomokuState::refresh_winner_cache() const {
    // Check for a win by either player
    for (int p : {BLACK, WHITE}) {
        if (rules->is_five_in_a_row(-1, p)) {
            cached_winner = p;  // Store the actual player constant (BLACK or WHITE)
            winner_check_dirty = false;
            return;
        }
    }
    
    cached_winner = 0;
    winner_check_dirty = false;
}

bool GomokuState::is_terminal() const {
    // Check winner cache first
    if (winner_check_dirty) {
        refresh_winner_cache();
    }
    
    // If we have a winner, game is over
    if (cached_winner != 0) {
        return true;
    }
    
    // Check for stalemate (board full)
    return is_stalemate();
}

bool GomokuState::is_stalemate() const {
    // Use cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.empty();
    }
    
    // Simple check: if board is full, it's a stalemate
    int stones = count_total_stones();
    if (stones >= board_size * board_size) {
        return true;
    }
    
    // Otherwise, check if there are any valid moves
    refresh_valid_moves_cache();
    return cached_valid_moves.empty();
}

int GomokuState::get_winner() const {
    if (winner_check_dirty) {
        refresh_winner_cache();
    }
    
    return cached_winner;
}

std::vector<int> GomokuState::get_valid_moves() const {
    // Use cache if available
    if (!valid_moves_dirty) {
        return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
    }
    
    // Refresh cache
    refresh_valid_moves_cache();
    
    // Return cached result
    return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
}

void GomokuState::refresh_valid_moves_cache() const {
    cached_valid_moves.clear();
    int total = board_size * board_size;
    int stone_count = count_total_stones();
    
    // First identify all empty cells
    for (int a = 0; a < total; a++) {
        if (!is_occupied(a)) {
            if (use_pro_long_opening && current_player == BLACK) {
                if (!is_pro_long_move_ok(a, stone_count)) {
                    continue;
                }
            }
            
            // Check forbidden moves for Black
            if (current_player == BLACK) {
                if (use_renju) {
                    if (!rules->is_black_renju_forbidden(a)) {
                        cached_valid_moves.insert(a);
                    }
                } else if (use_omok) {
                    if (!rules->is_black_omok_forbidden(a)) {
                        cached_valid_moves.insert(a);
                    }
                } else {
                    cached_valid_moves.insert(a);
                }
            } else {
                cached_valid_moves.insert(a);
            }
        }
    }
    
    valid_moves_dirty = false;
}

bool GomokuState::is_move_valid(int action) const {
    // Quick bounds check
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        return false;
    }
    
    // Check if already occupied
    if (is_occupied(action)) {
        return false;
    }
    
    // Check cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.find(action) != cached_valid_moves.end();
    }
    
    // Special case for pro-long opening
    if (use_pro_long_opening && current_player == BLACK) {
        if (!is_pro_long_move_ok(action, count_total_stones())) {
            return false;
        }
    }
    
    // Check forbidden moves for Black
    if (current_player == BLACK) {
        if (use_renju) {
            // Verify the rules are properly set up
            bool forbidden = rules->is_black_renju_forbidden(action);
            if (forbidden) {
                return false;
            }
        } else if (use_omok) {
            // Verify the rules are properly set up
            bool forbidden = rules->is_black_omok_forbidden(action);
            if (forbidden) {
                return false;
            }
        }
    }
    
    return true;
}

uint64_t GomokuState::compute_hash_signature() const {
    if (!hash_dirty) {
        return hash_signature;
    }
    
    uint64_t hash = 0;
    const int cells = board_size * board_size;
    
    // Hash board position
    for (int action = 0; action < cells; action++) {
        if (is_bit_set(BLACK, action)) {  // BLACK = 1
            hash ^= zobrist.getPieceHash(BLACK-1, action);
        } else if (is_bit_set(WHITE, action)) {  // WHITE = 2
            hash ^= zobrist.getPieceHash(WHITE-1, action);
        }
    }
    
    // Hash current player
    hash ^= zobrist.getPlayerHash(current_player - 1);
    
    // Hash rule variants
    if (use_renju) {
        hash ^= zobrist.getFeatureHash("renju", 1);
    }
    if (use_omok) {
        hash ^= zobrist.getFeatureHash("omok", 1);
    }
    if (use_pro_long_opening) {
        hash ^= zobrist.getFeatureHash("pro_long", 1);
    }
    
    // Update cached hash
    hash_signature = hash;
    hash_dirty = false;
    
    return hash_signature;
}

bool GomokuState::board_equal(const GomokuState& other) const {
    // Quick check for different board sizes
    if (board_size != other.board_size || current_player != other.current_player) {
        return false;
    }
    
    // Compare hash signatures if available
    if (!hash_dirty && !other.hash_dirty) {
        return hash_signature == other.hash_signature;
    }
    
    // Compare individual bitboards
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < num_words; j++) {
            if (player_bitboards[i][j] != other.player_bitboards[i][j]) {
                return false;
            }
        }
    }
    
    return true;
}

void GomokuState::make_move(int action, int player) {
    // Quick validation
    if (action < 0 || action >= board_size * board_size) {
        throw std::runtime_error("Move " + std::to_string(action) + " out of range.");
    }
    if (is_occupied(action)) {
        throw std::runtime_error("Cell " + std::to_string(action) + " is already occupied.");
    }
    
    // Rule validation if needed
    if (use_pro_long_opening && player == BLACK) {
        if (!is_pro_long_move_ok(action, count_total_stones())) {
            throw std::runtime_error("Pro-Long Opening constraint violated.");
        }
    }
    
    if (player == BLACK) {
        if (use_renju && rules->is_black_renju_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Renju).");
        } else if (use_omok && rules->is_black_omok_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Omok).");
        }
    }
    
    // Place the stone with bitboard operations
    set_bit(player - 1, action);
    this->action = action;
    
    // Update black's first stone if needed
    if (player == BLACK && black_first_stone < 0) {
        black_first_stone = action;
    }
    
    // Update player turn
    current_player = 3 - player;
    
    // Add to move history
    move_history.push_back(action);
    
    // Invalidate caches
    invalidate_caches();
}

void GomokuState::undo_move(int action) {
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        throw std::runtime_error("Undo " + std::to_string(action) + " out of range.");
    }

    int prev_player = 3 - current_player;
    int p_idx = prev_player - 1;

    if (!is_bit_set(p_idx, action)) {
        throw std::runtime_error("Undo error: Stone not found for last mover.");
    }

    // Remove the stone
    clear_bit(p_idx, action);
    this->action = -1;
    
    // Update player turn
    current_player = prev_player;

    if (prev_player == BLACK && black_first_stone == action) {
        black_first_stone = -1;
    }

    // Update move history
    if (!move_history.empty()) {
        move_history.pop_back();
    }

    // Invalidate caches
    invalidate_caches();
}

void GomokuState::invalidate_caches() {
    valid_moves_dirty = true;
    winner_check_dirty = true;
    hash_dirty = true;
}

bool GomokuState::is_occupied(int action) const {
    // Quick bounds check
    if (action < 0 || action >= board_size * board_size) {
        return true; // Out of bounds is considered occupied
    }
    
    // Check both BLACK (1) and WHITE (2) using is_bit_set which handles the index conversion
    return is_bit_set(BLACK, action) || is_bit_set(WHITE, action);
}

std::vector<std::vector<int>> GomokuState::get_board() const {
    std::vector<std::vector<int>> arr(board_size, std::vector<int>(board_size, 0));
    int total = board_size * board_size;
    
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p_idx][w];
            if (chunk == 0) {
                continue;
            }
            
            for (int b = 0; b < 64; b++) {
                if ((chunk & (static_cast<uint64_t>(1) << b)) != 0) {
                    int action = w * 64 + b;
                    if (action >= total) {
                        break;
                    }
                    int x = action / board_size;
                    int y = action % board_size;
                    arr[x][y] = (p_idx + 1);
                }
            }
        }
    }
    
    return arr;
}

// MCTS/NN support functions
GomokuState GomokuState::apply_action(int action) const {
    GomokuState new_state = copy();
    new_state.make_move(action, current_player);
    return new_state;
}

std::vector<std::vector<std::vector<float>>> GomokuState::to_tensor() const {
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size, 
            std::vector<float>(board_size, 0.0f)));
    
    int p_idx = current_player - 1;
    int opp_idx = 1 - p_idx;
    int total = board_size * board_size;
    
    for (int a = 0; a < total; a++) {
        int x = a / board_size;
        int y = a % board_size;
        
        if (is_bit_set(p_idx, a)) {
            tensor[0][x][y] = 1.0f;
        } else if (is_bit_set(opp_idx, a)) {
            tensor[1][x][y] = 1.0f;
        }
    }
    
    if (current_player == BLACK) {
        for (int i = 0; i < board_size; i++) {
            for (int j = 0; j < board_size; j++) {
                tensor[2][i][j] = 1.0f;
            }
        }
    }
    
    return tensor;
}

int GomokuState::get_action(const GomokuState& child_state) const {
    int total = board_size * board_size;
    for (int a = 0; a < total; a++) {
        if (is_occupied(a) != child_state.is_occupied(a)) {
            return a;
        }
    }
    return -1;
}

std::vector<int> GomokuState::get_previous_moves(int player, int count) const {
    std::vector<int> prev_moves(count, -1);  // Initialize with -1 (no move)
    
    int found = 0;
    // Iterate backward through move history
    for (int i = static_cast<int>(move_history.size()) - 1; i >= 0 && found < count; --i) {
        int move = move_history[i];
        // Determine which player made this move based on position in history
        int move_player = (move_history.size() - i) % 2 == 1 ? current_player : 3 - current_player;
        
        if (move_player == player) {
            prev_moves[found] = move;
            found++;
        }
    }
    
    return prev_moves;
}

// Helper methods

bool GomokuState::in_bounds(int x, int y) const {
    return (0 <= x && x < board_size) && (0 <= y && y < board_size);
}

bool GomokuState::is_pro_long_move_ok(int action, int stone_count) const {
    int center = (board_size / 2) * board_size + (board_size / 2);
    
    if (stone_count == 0 || stone_count == 1) {
        return (action == center);
    } else if (stone_count == 2 || stone_count == 3) {
        if (black_first_stone < 0) {
            return false;
        }
        
        auto [x0, y0] = action_to_coords_pair(black_first_stone);
        auto [x1, y1] = action_to_coords_pair(action);
        int dist = abs(x1 - x0) + abs(y1 - y0);
        return (dist >= 4);
    }
    
    return true;
}

} // namespace gomoku
} // namespace games
} // namespace alphazero