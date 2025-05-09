// File: gomoku_state.cpp
#include "games/gomoku/gomoku_state.h"
#include "games/gomoku/gomoku_rules.h"     // For rules_engine_
#include "core/illegal_move_exception.h" // For core::IllegalMoveException
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <iostream>  // For debugging (optional, remove in production)
#include <numeric>   // For std::accumulate, std::gcd
#include <algorithm> // For std::fill, std::find


namespace alphazero {
namespace games {
namespace gomoku {

// --- Constructor ---
GomokuState::GomokuState(int board_size, bool use_renju, bool use_omok, int seed, bool use_pro_long_opening)
    : IGameState(core::GameType::GOMOKU),
      board_size_(board_size),
      current_player_(BLACK),
      move_history_(), 
      zobrist_(board_size * board_size, 2, 4), 
      use_renju_(use_renju),
      use_omok_(use_omok),
      use_pro_long_opening_(use_pro_long_opening),
      black_first_stone_(-1),
      valid_moves_dirty_(true),
      cached_winner_(NO_PLAYER),
      winner_check_dirty_(true),
      hash_signature_(0), 
      hash_dirty_(true),
      last_action_played_(-1) {

    if (board_size_ <= 0) {
        throw std::invalid_argument("Board size must be positive.");
    }
    int total_cells = board_size_ * board_size_;
    num_words_ = (total_cells + 63) / 64;

    player_bitboards_.resize(2, std::vector<uint64_t>(num_words_, 0ULL));

    rules_engine_ = std::make_shared<GomokuRules>(board_size_);
    rules_engine_->setBoardAccessor(
        [this](int p_idx, int act) { return this->is_bit_set(p_idx, act); },
        [this](int act) { return this->is_any_bit_set_for_rules(act); },
        [this](int r, int c) { return this->coords_to_action(r, c); },
        [this](int act) { return this->action_to_coords_pair(act); },
        [this](int r, int c) { return this->in_bounds(r, c); }
    );
    invalidate_caches();
}

// --- Copy Constructor ---
GomokuState::GomokuState(const GomokuState& other)
    : IGameState(core::GameType::GOMOKU), 
      board_size_(other.board_size_),
      current_player_(other.current_player_),
      move_history_(other.move_history_),
      zobrist_(other.zobrist_),
      use_renju_(other.use_renju_),
      use_omok_(other.use_omok_),
      use_pro_long_opening_(other.use_pro_long_opening_),
      black_first_stone_(other.black_first_stone_),
      cached_valid_moves_(other.cached_valid_moves_),
      valid_moves_dirty_(other.valid_moves_dirty_),
      cached_winner_(other.cached_winner_),
      winner_check_dirty_(other.winner_check_dirty_),
      hash_signature_(other.hash_signature_),
      hash_dirty_(other.hash_dirty_),
      num_words_(other.num_words_),
      player_bitboards_(other.player_bitboards_),
      rules_engine_(std::make_shared<GomokuRules>(other.board_size_)), 
      last_action_played_(other.last_action_played_) {

    rules_engine_->setBoardAccessor(
        [this](int p_idx, int act) { return this->is_bit_set(p_idx, act); },
        [this](int act) { return this->is_any_bit_set_for_rules(act); },
        [this](int r, int c) { return this->coords_to_action(r, c); },
        [this](int act) { return this->action_to_coords_pair(act); },
        [this](int r, int c) { return this->in_bounds(r, c); }
    );
}

// --- Public IGameState Interface Methods ---

std::vector<int> GomokuState::getLegalMoves() const {
    if (isTerminal()) return {};
    if (valid_moves_dirty_) {
        refresh_valid_moves_cache();
    }
    return std::vector<int>(cached_valid_moves_.begin(), cached_valid_moves_.end());
}

bool GomokuState::isLegalMove(int action) const {
    if (action < 0 || action >= getActionSpaceSize()) return false;
    if (isTerminal()) return false; 

    return is_move_valid_internal(action, true);
}

void GomokuState::makeMove(int action) {
    if (!isLegalMove(action)) {
        throw core::IllegalMoveException("Attempted illegal move: " + actionToString(action) +
                                         " for player " + std::to_string(current_player_), action);
    }
    make_move_internal(action, current_player_);
}

bool GomokuState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    int last_action_to_undo = move_history_.back();
    int player_who_made_that_move = (current_player_ == BLACK) ? WHITE : BLACK;

    undo_last_move_internal(last_action_to_undo, player_who_made_that_move);
    return true;
}

bool GomokuState::isTerminal() const {
    if (winner_check_dirty_) {
        refresh_winner_cache();
    }
    return (cached_winner_ != NO_PLAYER) || is_stalemate();
}

core::GameResult GomokuState::getGameResult() const {
    if (winner_check_dirty_) { 
        refresh_winner_cache();
    }

    if (cached_winner_ == BLACK) return core::GameResult::WIN_PLAYER1;
    if (cached_winner_ == WHITE) return core::GameResult::WIN_PLAYER2;
    
    if (is_stalemate()) return core::GameResult::DRAW;
    
    return core::GameResult::ONGOING;
}

int GomokuState::getCurrentPlayer() const {
    return current_player_;
}

int GomokuState::getBoardSize() const {
    return board_size_;
}

int GomokuState::getActionSpaceSize() const {
    return board_size_ * board_size_;
}

std::vector<std::vector<std::vector<float>>> GomokuState::getTensorRepresentation() const {
    std::vector<std::vector<std::vector<float>>> tensor(
        3, std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f)));
    
    int p_idx_current = current_player_ - 1;      
    int p_idx_opponent = 1 - p_idx_current; 

    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (is_bit_set(p_idx_current, action)) {
                tensor[0][r][c] = 1.0f; 
            } else if (is_bit_set(p_idx_opponent, action)) {
                tensor[1][r][c] = 1.0f; 
            }
            tensor[2][r][c] = (current_player_ == BLACK) ? 1.0f : 0.0f;
        }
    }
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GomokuState::getEnhancedTensorRepresentation() const {
    const int num_history_pairs = 8; 
    const int num_feature_planes = 2 * num_history_pairs + 1; 
    
    std::vector<std::vector<std::vector<float>>> tensor(
        num_feature_planes, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));

    int history_len = move_history_.size();
    
    std::vector<int> current_player_moves_in_history;
    std::vector<int> opponent_player_moves_in_history;

    for(int k=0; k < history_len; ++k) {
        int move_action = move_history_[history_len - 1 - k];
        if (k % 2 == 0) { 
            opponent_player_moves_in_history.push_back(move_action);
        } else { 
            current_player_moves_in_history.push_back(move_action);
        }
    }

    for(int i=0; i < num_history_pairs && i < current_player_moves_in_history.size(); ++i) {
        auto [r,c] = action_to_coords_pair(current_player_moves_in_history[i]);
        tensor[i*2][r][c] = 1.0f; 
    }
    for(int i=0; i < num_history_pairs && i < opponent_player_moves_in_history.size(); ++i) {
        auto [r,c] = action_to_coords_pair(opponent_player_moves_in_history[i]);
        tensor[i*2 + 1][r][c] = 1.0f; 
    }

    float color_plane_val = (current_player_ == BLACK) ? 1.0f : 0.0f;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            tensor[num_feature_planes - 1][r][c] = color_plane_val;
        }
    }
    return tensor;
}


uint64_t GomokuState::getHash() const {
    if (hash_dirty_) {
        hash_signature_ = compute_hash_signature_internal();
        hash_dirty_ = false;
    }
    return hash_signature_;
}

std::unique_ptr<core::IGameState> GomokuState::clone() const {
    return std::make_unique<GomokuState>(*this);
}

std::string GomokuState::actionToString(int action) const {
    if (action < 0 || action >= getActionSpaceSize()) return "PASS";
    auto [r, c] = action_to_coords_pair(action);
    char col_char = 'A' + c;
    if (board_size_ > 8 && col_char >= 'I') { 
        col_char++;
    }
    return std::string(1, col_char) + std::to_string(board_size_ - r);
}

std::optional<int> GomokuState::stringToAction(const std::string& moveStr) const {
    if (moveStr.empty() || moveStr == "PASS") return -1; 

    char col_char_upper = std::toupper(moveStr[0]);
    int col = col_char_upper - 'A';
    if (board_size_ > 8 && col_char_upper > 'I') { 
        col--;
    }

    if (col < 0 || col >= board_size_) return std::nullopt;

    try {
        std::string row_str = moveStr.substr(1);
        if (row_str.empty()) return std::nullopt;
        int row_num_1_based = std::stoi(row_str);
        if (row_num_1_based <= 0 || row_num_1_based > board_size_) return std::nullopt;
        
        int r_0_based = board_size_ - row_num_1_based; 
        
        if (!in_bounds(r_0_based, col)) return std::nullopt;
        return coords_to_action(r_0_based, col);
    } catch (const std::exception&) {
        return std::nullopt; 
    }
}

std::string GomokuState::toString() const {
    std::stringstream ss;
    ss << "  "; 
    for (int c = 0; c < board_size_; ++c) {
        char col_char = 'A' + c;
        if (board_size_ > 8 && col_char >= 'I') col_char++;
        ss << col_char << " ";
    }
    ss << std::endl;

    for (int r = 0; r < board_size_; ++r) {
        ss << std::setw(2) << (board_size_ - r) << " "; 
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (is_bit_set(0, action)) ss << "X ";      
            else if (is_bit_set(1, action)) ss << "O "; 
            else ss << ". ";                            
        }
        ss << std::setw(2) << (board_size_ - r); 
        ss << std::endl;
    }
    ss << "  "; 
    for (int c = 0; c < board_size_; ++c) {
        char col_char = 'A' + c;
        if (board_size_ > 8 && col_char >= 'I') col_char++;
        ss << col_char << " ";
    }
    ss << std::endl;

    ss << "Player to move: " << (current_player_ == BLACK ? "X (BLACK)" : "O (WHITE)") << std::endl;
    if (!move_history_.empty()) {
        ss << "Last move: " << actionToString(last_action_played_) << std::endl;
    }
    if (isTerminal()) { 
         ss << "Game Over. Result: ";
         core::GameResult res = getGameResult(); 
         if (res == core::GameResult::WIN_PLAYER1) ss << "BLACK (X) wins.";
         else if (res == core::GameResult::WIN_PLAYER2) ss << "WHITE (O) wins.";
         else if (res == core::GameResult::DRAW) ss << "Draw.";
         else ss << "Ongoing (Error in toString terminal check)";
         ss << std::endl;
    }
    return ss.str();
}

bool GomokuState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::GOMOKU) return false;
    const auto* o_state = dynamic_cast<const GomokuState*>(&other);
    if (!o_state) return false; 
    return board_equal_internal(*o_state);
}

std::vector<int> GomokuState::getMoveHistory() const {
    return move_history_;
}

bool GomokuState::validate() const {
    int black_stones = 0;
    int white_stones = 0;
    for(int i=0; i < getActionSpaceSize(); ++i) {
        if(is_bit_set(0, i)) black_stones++; 
        if(is_bit_set(1, i)) white_stones++; 
    }

    if (current_player_ == BLACK) {
        return black_stones == white_stones;
    } else { 
        return black_stones == white_stones + 1;
    }
}


// --- Testing Specific Methods ---
void GomokuState::setStoneForTesting(int r, int c, int player) {
    if (!in_bounds(r,c)) return;
    int action = coords_to_action(r,c);
    
    clear_bit(0, action);
    clear_bit(1, action);

    if (player == BLACK) {
        set_bit(0, action);
    } else if (player == WHITE) {
        set_bit(1, action);
    }
    invalidate_caches(); 
}

void GomokuState::setCurrentPlayerForTesting(int player) {
    if (player == BLACK || player == WHITE) {
        current_player_ = player;
        invalidate_caches();
    }
}

void GomokuState::clearBoardForTesting() {
    for(int p_idx=0; p_idx<2; ++p_idx) {
        std::fill(player_bitboards_[p_idx].begin(), player_bitboards_[p_idx].end(), 0ULL);
    }
    move_history_.clear();
    current_player_ = BLACK; 
    black_first_stone_ = -1;
    last_action_played_ = -1;
    invalidate_caches();
}


// --- Private Helper Methods ---

bool GomokuState::is_bit_set(int player_idx_0_based, int action) const noexcept {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) return false;
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) return false;
    return (player_bitboards_[player_idx_0_based][word_idx] & (1ULL << bit_idx)) != 0;
}

void GomokuState::set_bit(int player_idx_0_based, int action) {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) {
        throw std::out_of_range("set_bit: Invalid player_idx or action.");
    }
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) {
         throw std::out_of_range("set_bit: word_idx out of bounds.");
    }
    player_bitboards_[player_idx_0_based][word_idx] |= (1ULL << bit_idx);
}

void GomokuState::clear_bit(int player_idx_0_based, int action) noexcept {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) return;
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) return;
    player_bitboards_[player_idx_0_based][word_idx] &= ~(1ULL << bit_idx);
}

std::pair<int, int> GomokuState::action_to_coords_pair(int action) const noexcept {
    if (board_size_ == 0) return {-1,-1}; 
    return {action / board_size_, action % board_size_};
}

int GomokuState::coords_to_action(int r, int c) const noexcept {
    return r * board_size_ + c;
}

bool GomokuState::in_bounds(int r, int c) const noexcept {
    return r >= 0 && r < board_size_ && c >= 0 && c < board_size_;
}

int GomokuState::count_total_stones() const noexcept {
    int count = 0;
    for (int p_idx = 0; p_idx < 2; ++p_idx) {
        for (uint64_t word : player_bitboards_[p_idx]) {
            #if defined(__GNUC__) || defined(__clang__)
                count += __builtin_popcountll(word);
            #else
                uint64_t temp_word = word;
                while (temp_word > 0) {
                    temp_word &= (temp_word - 1);
                    count++;
                }
            #endif
        }
    }
    return count;
}

void GomokuState::refresh_winner_cache() const {
    cached_winner_ = NO_PLAYER; 
    if (last_action_played_ == -1 && !move_history_.empty()) { 
         const_cast<GomokuState*>(this)->last_action_played_ = move_history_.back();
    } else if (move_history_.empty() && last_action_played_ != -1) { 
        const_cast<GomokuState*>(this)->last_action_played_ = -1;
    }
    
    int player_who_made_last_move = NO_PLAYER;
    if (last_action_played_ != -1) { 
        player_who_made_last_move = (current_player_ == BLACK) ? WHITE : BLACK;
    }

    if (player_who_made_last_move != NO_PLAYER) {
        // Create a board accessor
        auto board_accessor = [this](int p_idx, int act){ 
            return this->is_bit_set(p_idx, act);
        };

        if (player_who_made_last_move == BLACK) {
            // Black must win with exactly 5 in Renju, any length >=5 in Standard/Omok
            bool allow_overline_black = !use_renju_;
            
            // Check for 5-in-a-row (or more if allowed)
            bool win_by_five = rules_engine_->is_five_in_a_row(last_action_played_, BLACK, allow_overline_black);
            
            // For Renju, we need to ensure it's exactly 5, not an overline
            if (use_renju_ && win_by_five) {
                // Get the actual line length
                const int DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
                int max_length = 0;
                for (auto& dir : DIRS) {
                    int length = rules_engine_->get_line_length_at_action(
                        last_action_played_, 0, board_accessor, dir[0], dir[1]);
                    max_length = std::max(max_length, length);
                }
                
                // Only set as winner if exactly 5 stones in a row
                if (max_length == 5) {
                    cached_winner_ = player_who_made_last_move;
                }
            } else if (win_by_five) {
                // Standard or Omok
                cached_winner_ = player_who_made_last_move;
            }
        } else { // WHITE made last move
            // White can win with 5+ in a row in any variant
            bool win_by_five = rules_engine_->is_five_in_a_row(last_action_played_, WHITE, true /*allow_overline*/);
            
            // For debugging the RenjuOverlineWhite test
            // Let's manually check the line length in each direction
            if (use_renju_) { // This is important for Renju variant specifically
                const int DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
                for (auto& dir : DIRS) {
                    int length = rules_engine_->get_line_length_at_action(
                        last_action_played_, 1, board_accessor, dir[0], dir[1]);
                    if (length >= 5) {
                        win_by_five = true;
                        break;
                    }
                }
            }
            
            if (win_by_five) {
                cached_winner_ = player_who_made_last_move;
            }
        }
    }
    
    winner_check_dirty_ = false;
}

bool GomokuState::is_stalemate() const {
    if (cached_winner_ != NO_PLAYER && !winner_check_dirty_) return false; 

    if (count_total_stones() >= getActionSpaceSize()) return true; 

    if (valid_moves_dirty_) { // Must refresh to check if any moves are possible
        const_cast<GomokuState*>(this)->refresh_valid_moves_cache();
    }
    // Stalemate if no winner and no legal moves for the current player
    return cached_valid_moves_.empty() && (cached_winner_ == NO_PLAYER);
}


void GomokuState::refresh_valid_moves_cache() const {
    cached_valid_moves_.clear();
    if (cached_winner_ != NO_PLAYER && !winner_check_dirty_) {
        valid_moves_dirty_ = false;
        return;
    }

    int total_actions = getActionSpaceSize();
    for (int action = 0; action < total_actions; ++action) {
        if (is_move_valid_internal(action, true)) { 
            cached_valid_moves_.insert(action);
        }
    }
    valid_moves_dirty_ = false;
}

bool GomokuState::is_move_valid_internal(int action, bool check_occupation) const {
    if (check_occupation) {
        if (is_occupied(action)) return false;
    }
    
    if (use_pro_long_opening_ && current_player_ == BLACK &&
        !is_pro_long_opening_move_valid(action, count_total_stones())) {
        return false;
    }

    if (current_player_ == BLACK) {
        if (use_renju_ && rules_engine_->is_black_renju_forbidden(action)) {
            return false;
        }
        if (use_omok_ && rules_engine_->is_black_omok_forbidden(action)) {
            return false;
        }
    }
    return true;
}


uint64_t GomokuState::compute_hash_signature_internal() const {
    uint64_t h = 0;
    for (int p_idx = 0; p_idx < 2; ++p_idx) {
        for (int action = 0; action < getActionSpaceSize(); ++action) {
            if (is_bit_set(p_idx, action)) {
                h ^= zobrist_.getPieceHash(p_idx, action);
            }
        }
    }
    h ^= zobrist_.getPlayerHash(current_player_ - 1); 
    return h;
}

bool GomokuState::board_equal_internal(const GomokuState& other) const {
    if (board_size_ != other.board_size_ || current_player_ != other.current_player_ ||
        use_renju_ != other.use_renju_ || use_omok_ != other.use_omok_ ||
        use_pro_long_opening_ != other.use_pro_long_opening_ ||
        player_bitboards_ != other.player_bitboards_ ) { 
        return false;
    }
    return true;
}

void GomokuState::make_move_internal(int action, int player_to_move) {
    set_bit(player_to_move - 1, action); 
    last_action_played_ = action;
    move_history_.push_back(action);

    if (player_to_move == BLACK && black_first_stone_ == -1) {
        black_first_stone_ = action;
    }
    current_player_ = (player_to_move == BLACK) ? WHITE : BLACK;
    invalidate_caches(); 
}

void GomokuState::undo_last_move_internal(int last_action_undone, int player_who_made_last_action) {
    clear_bit(player_who_made_last_action - 1, last_action_undone); 
    move_history_.pop_back();

    if (player_who_made_last_action == BLACK && black_first_stone_ == last_action_undone) {
        black_first_stone_ = -1;
        if (!move_history_.empty()) { 
            for(size_t i=0; i < move_history_.size(); ++i) {
                // Crude way to determine player of history move: assume Black starts (player 1)
                // Player of move_history_[i] is (i % 2 == 0) ? BLACK : WHITE
                if ( (i % 2) == 0 ) { 
                    black_first_stone_ = move_history_[i];
                    break;
                }
            }
        }
    }

    current_player_ = player_who_made_last_action; 
    last_action_played_ = move_history_.empty() ? -1 : move_history_.back();
    invalidate_caches();
}

void GomokuState::invalidate_caches() {
    valid_moves_dirty_ = true;
    winner_check_dirty_ = true;
    hash_dirty_ = true;
}

bool GomokuState::is_occupied(int action) const {
    return is_bit_set(0, action) || is_bit_set(1, action);
}

bool GomokuState::is_any_bit_set_for_rules(int action) const {
    return is_occupied(action);
}


bool GomokuState::is_pro_long_opening_move_valid(int action, int total_stones_on_board) const {
    if (!use_pro_long_opening_) return true; 

    int center_r = board_size_ / 2;
    int center_c = board_size_ / 2;
    int center_action = coords_to_action(center_r, center_c);

    if (current_player_ == BLACK) {
        if (total_stones_on_board == 0) { 
            return action == center_action;
        }
        if (total_stones_on_board == 2) { 
            if (black_first_stone_ == -1) {
                return false;
            }
            auto [r1, c1] = action_to_coords_pair(black_first_stone_);
            auto [r2, c2] = action_to_coords_pair(action);
            int chebyshev_dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
            return chebyshev_dist >= 2; 
        }
    }
    return true;
}


} // namespace gomoku
} // namespace games
} // namespace alphazero