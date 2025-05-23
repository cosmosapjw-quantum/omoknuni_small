// File: gomoku_state.cpp
#include "games/gomoku/gomoku_state.h"
#include "games/gomoku/gomoku_rules.h"     // For rules_engine_
#include "core/illegal_move_exception.h" // For core::IllegalMoveException
#include "core/tensor_pool.h"            // For GlobalTensorPool optimization
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
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true),
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
      valid_moves_dirty_(other.valid_moves_dirty_.load()),
      cached_winner_(other.cached_winner_.load()),
      winner_check_dirty_(other.winner_check_dirty_.load()),
      hash_signature_(other.hash_signature_.load()),
      hash_dirty_(other.hash_dirty_.load()),
      // Don't copy cached tensors - force recomputation to avoid memory issues
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true),
      cached_valid_moves_(other.cached_valid_moves_),
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
    // CRITICAL FIX: Fast path for empty board to prevent MCTS stalling
    if (move_history_.empty()) {
        return false; // Empty board cannot be terminal
    }
    
    
    // MCTS OPTIMIZATION: For very early game (≤4 moves), skip expensive winner check
    // Cannot have a winner in Gomoku with 4 or fewer stones
    int total_stones = count_total_stones();
    if (total_stones <= 4) {
        return false;
    }
    
    // LOCK-FREE: Check winner cache with atomic operations
    if (winner_check_dirty_.load(std::memory_order_acquire)) {
        refresh_winner_cache(); // Lock-free refresh
    }
    
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    bool has_winner = (current_winner != NO_PLAYER);
    
    bool is_stale = is_stalemate(); // Lock-free stalemate check
    
    return has_winner || is_stale;
}

core::GameResult GomokuState::getGameResult() const {
    if (winner_check_dirty_.load(std::memory_order_acquire)) { 
        refresh_winner_cache();
    }

    int current_winner = cached_winner_.load(std::memory_order_acquire);
    if (current_winner == BLACK) return core::GameResult::WIN_PLAYER1;
    if (current_winner == WHITE) return core::GameResult::WIN_PLAYER2;
    
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
    // PERFORMANCE FIX: Use cached tensor if available and not dirty
    if (!tensor_cache_dirty_.load(std::memory_order_relaxed) && !cached_tensor_repr_.empty()) {
        return cached_tensor_repr_;
    }
    
    // PERFORMANCE FIX: Use global tensor pool for efficient memory reuse
    auto tensor = core::GlobalTensorPool::getInstance().getTensor(3, board_size_, board_size_);
    
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
    
    // PERFORMANCE FIX: Cache the computed tensor
    cached_tensor_repr_ = tensor;
    tensor_cache_dirty_.store(false, std::memory_order_relaxed);
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GomokuState::getEnhancedTensorRepresentation() const {
    // PERFORMANCE FIX: Use cached enhanced tensor if available and not dirty
    if (!enhanced_tensor_cache_dirty_.load(std::memory_order_relaxed) && !cached_enhanced_tensor_repr_.empty()) {
        return cached_enhanced_tensor_repr_;
    }
    
    try {
        const int num_history_pairs = 8; 
        const int num_feature_planes = 2 * num_history_pairs + 1; 
        
        // PERFORMANCE FIX: Use global tensor pool for efficient memory reuse
        auto tensor = core::GlobalTensorPool::getInstance().getTensor(num_feature_planes, board_size_, board_size_);

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

        // Add bounds checking to prevent segfaults
        for(int i=0; i < num_history_pairs && i < current_player_moves_in_history.size(); ++i) {
            auto coords = action_to_coords_pair(current_player_moves_in_history[i]);
            int r = coords.first;
            int c = coords.second;
            if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_ && (i*2) < num_feature_planes) {
                tensor[i*2][r][c] = 1.0f;
            }
        }
        
        for(int i=0; i < num_history_pairs && i < opponent_player_moves_in_history.size(); ++i) {
            auto coords = action_to_coords_pair(opponent_player_moves_in_history[i]);
            int r = coords.first;
            int c = coords.second;
            if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_ && (i*2 + 1) < num_feature_planes) {
                tensor[i*2 + 1][r][c] = 1.0f;
            }
        }

        float color_plane_val = (current_player_ == BLACK) ? 1.0f : 0.0f;
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                tensor[num_feature_planes - 1][r][c] = color_plane_val;
            }
        }
        
        // PERFORMANCE FIX: Cache the computed enhanced tensor
        cached_enhanced_tensor_repr_ = tensor;
        enhanced_tensor_cache_dirty_.store(false, std::memory_order_relaxed);
        
        return tensor;
    } catch (const std::exception& e) {
        std::cerr << "Exception in getEnhancedTensorRepresentation: " << e.what() << std::endl;
        
        // Return a default tensor with the correct dimensions (17 channels)
        const int num_history_pairs = 8;
        const int num_feature_planes = 2 * num_history_pairs + 1; // 17 channels
        
        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, 
            std::vector<std::vector<float>>(
                board_size_, 
                std::vector<float>(board_size_, 0.0f)
            )
        );
    } catch (...) {
        std::cerr << "Unknown exception in getEnhancedTensorRepresentation" << std::endl;
        
        // Return a default tensor with the correct dimensions (17 channels)
        const int num_history_pairs = 8;
        const int num_feature_planes = 2 * num_history_pairs + 1; // 17 channels
        
        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, 
            std::vector<std::vector<float>>(
                board_size_, 
                std::vector<float>(board_size_, 0.0f)
            )
        );
    }
}


uint64_t GomokuState::getHash() const {
    if (hash_dirty_) {
        hash_signature_ = compute_hash_signature_internal();
        hash_dirty_ = false;
    }
    return hash_signature_;
}

std::unique_ptr<core::IGameState> GomokuState::clone() const {
    try {
        // Create a new instance with same parameters - don't do any validation yet
        auto clone_ptr = std::make_unique<GomokuState>(
            board_size_,
            use_renju_,
            use_omok_,
            0, // Using 0 as seed since we're copying existing state
            use_pro_long_opening_
        );
        
        if (!clone_ptr) {
            throw std::runtime_error("Failed to allocate memory for GomokuState clone");
        }
        
        // Copy primitive state variables
        clone_ptr->current_player_ = current_player_;
        clone_ptr->black_first_stone_ = black_first_stone_;
        clone_ptr->last_action_played_ = last_action_played_;

        // Copy move history
        clone_ptr->move_history_ = move_history_; 
        
        // Don't copy cached data to avoid race conditions. Mark caches as dirty instead.
        clone_ptr->valid_moves_dirty_.store(true, std::memory_order_release);  // Force recomputation of valid moves
        clone_ptr->cached_valid_moves_.clear(); // Clear the cache
        clone_ptr->cached_winner_.store(NO_PLAYER, std::memory_order_release);
        clone_ptr->winner_check_dirty_.store(true, std::memory_order_release);  // Force recomputation of winner
        clone_ptr->hash_signature_.store(0, std::memory_order_release);
        clone_ptr->hash_dirty_.store(true, std::memory_order_release);  // Force recomputation of hash 
        
        // Deep copy of player_bitboards_ with size checking
        // First ensure both have the correct sizes
        if (player_bitboards_.size() != 2) {
            throw std::runtime_error("Source player bitboards has invalid size: " + std::to_string(player_bitboards_.size()));
        }
        
        if (clone_ptr->player_bitboards_.size() != 2) {
            throw std::runtime_error("Clone player bitboards has invalid size: " + std::to_string(clone_ptr->player_bitboards_.size()));
        }
        
        // Ensure bitboards are properly sized for this board
        size_t expected_size = (board_size_ * board_size_ + 63) / 64;
        
        for (int p = 0; p < 2; ++p) {
            // Resize if necessary to match expected size
            if (clone_ptr->player_bitboards_[p].size() != expected_size) {
                clone_ptr->player_bitboards_[p].resize(expected_size, 0);
            }
            
            if (player_bitboards_[p].size() != expected_size) {
                throw std::runtime_error("Source bitboard[" + std::to_string(p) + "] has unexpected size: " + 
                    std::to_string(player_bitboards_[p].size()) + " (expected " + std::to_string(expected_size) + ")");
            }
            
            // Now copy with confidence that sizes match
            std::copy(
                player_bitboards_[p].begin(),
                player_bitboards_[p].end(),
                clone_ptr->player_bitboards_[p].begin()
            );
        }
        
        // Validate the clone (without extensive logging)
        if (!clone_ptr->validate()) {
            throw std::runtime_error("Cloned GomokuState failed validation");
        }
        
        return clone_ptr;
    } catch (const std::exception& e) {
        // Simple error reporting without complex logging
        std::cerr << "Error in GomokuState::clone(): " << e.what() << std::endl;
        throw;
    }
}

void GomokuState::copyFrom(const core::IGameState& source) {
    // Ensure source is a GomokuState
    const GomokuState* gomoku_source = dynamic_cast<const GomokuState*>(&source);
    if (!gomoku_source) {
        throw std::runtime_error("Cannot copy from non-GomokuState: incompatible game types");
    }
    
    // Verify board sizes match
    if (board_size_ != gomoku_source->board_size_) {
        throw std::runtime_error("Cannot copy: board sizes mismatch");
    }
    
    // Copy rule configurations
    use_renju_ = gomoku_source->use_renju_;
    use_omok_ = gomoku_source->use_omok_;
    use_pro_long_opening_ = gomoku_source->use_pro_long_opening_;
    
    // Copy game state
    current_player_ = gomoku_source->current_player_;
    black_first_stone_ = gomoku_source->black_first_stone_;
    last_action_played_ = gomoku_source->last_action_played_;
    move_history_ = gomoku_source->move_history_;
    
    // Deep copy bitboards
    size_t expected_size = (board_size_ * board_size_ + 63) / 64;
    for (int p = 0; p < 2; ++p) {
        if (player_bitboards_[p].size() != expected_size) {
            player_bitboards_[p].resize(expected_size, 0);
        }
        std::copy(
            gomoku_source->player_bitboards_[p].begin(),
            gomoku_source->player_bitboards_[p].end(),
            player_bitboards_[p].begin()
        );
    }
    
    // Mark all caches as dirty to ensure thread safety
    valid_moves_dirty_.store(true, std::memory_order_release);
    cached_valid_moves_.clear();
    cached_winner_.store(NO_PLAYER, std::memory_order_release);
    winner_check_dirty_.store(true, std::memory_order_release);
    hash_signature_.store(0, std::memory_order_release);
    hash_dirty_.store(true, std::memory_order_release);
    
    // Validate the copied state
    if (!validate()) {
        throw std::runtime_error("Copied GomokuState failed validation");
    }
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
    try {
        // First check if board_size_ is valid to prevent segfaults
        if (board_size_ <= 0) {
            std::cerr << "Invalid board size: " << board_size_ << std::endl;
            return false;
        }
        
        // Check if player_bitboards_ has the expected size
        if (player_bitboards_.size() != 2) {
            std::cerr << "Invalid player_bitboards_ size: " << player_bitboards_.size() << std::endl;
            return false;
        }
        
        // Check if current_player_ is valid
        if (current_player_ != BLACK && current_player_ != WHITE) {
            std::cerr << "Invalid current_player_: " << current_player_ << std::endl;
            return false;
        }
        
        // Check if the bitboard word vectors have the expected size
        for (int p = 0; p < 2; p++) {
            if (player_bitboards_[p].size() != num_words_) {
                std::cerr << "Invalid player_bitboards_[" << p << "] size: " 
                          << player_bitboards_[p].size() << " (expected " << num_words_ << ")" << std::endl;
                return false;
            }
        }
        
        // Count stones to verify game state is valid
        int black_stones = 0;
        int white_stones = 0;
        
        for (int i = 0; i < getActionSpaceSize(); ++i) {
            if (i < 0 || i >= board_size_ * board_size_) {
                std::cerr << "Action space index out of range: " << i << std::endl;
                return false;
            }
            
            if (is_bit_set(0, i)) black_stones++; 
            if (is_bit_set(1, i)) white_stones++; 
            
            // Check that no position has both black and white stones
            if (is_bit_set(0, i) && is_bit_set(1, i)) {
                std::cerr << "Position " << i << " has both black and white stones" << std::endl;
                return false;
            }
        }
        
        // Check the stone count is valid based on the current player
        if (current_player_ == BLACK) {
            if (black_stones != white_stones) {
                std::cerr << "Invalid stone count for BLACK to move: black=" << black_stones 
                         << ", white=" << white_stones << std::endl;
                return false;
            }
        } else { // current_player_ == WHITE
            if (black_stones != white_stones + 1) {
                std::cerr << "Invalid stone count for WHITE to move: black=" << black_stones 
                         << ", white=" << white_stones << std::endl;
                return false;
            }
        }
        
        // Check if move history is consistent with stone count
        if (move_history_.size() != black_stones + white_stones) {
            std::cerr << "Move history size (" << move_history_.size() 
                     << ") inconsistent with stone count (" << (black_stones + white_stones) << ")" << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in GomokuState::validate(): " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception in GomokuState::validate()" << std::endl;
        return false;
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
    // LOCK-FREE: Check if refresh is needed using atomic load
    if (!winner_check_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    // Compute winner without modifying cache until the end
    int new_winner = NO_PLAYER; 
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
                    new_winner = player_who_made_last_move;
                }
            } else if (win_by_five) {
                // Standard or Omok
                new_winner = player_who_made_last_move;
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
                new_winner = player_who_made_last_move;
            }
        }
    }
    
    // Atomically publish the computed winner and mark cache as clean
    cached_winner_.store(new_winner, std::memory_order_release);
    winner_check_dirty_.store(false, std::memory_order_release);
}

bool GomokuState::is_stalemate() const {
    // Fast atomic checks
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    bool winner_dirty = winner_check_dirty_.load(std::memory_order_acquire);
    
    if (current_winner != NO_PLAYER && !winner_dirty) {
        return false; 
    }

    int total_stones = count_total_stones();
    if (total_stones >= getActionSpaceSize()) {
        return true; 
    }
    
    // MCTS OPTIMIZATION: For early game (≤8 moves), compute on-demand without caching
    // This avoids the expensive mutex contention for 99% of MCTS cases
    if (total_stones <= 8) {
        // Early game positions are extremely unlikely to be stalemate in Gomoku
        // Do a quick check for any legal move instead of full enumeration
        for (int action = 0; action < getActionSpaceSize(); ++action) {
            if (!is_occupied(action)) {
                return false; 
            }
        }
        return true;
    }
    
    // For mid/late game, use thread-safe cache
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (valid_moves_dirty_.load(std::memory_order_acquire)) {
        refresh_valid_moves_cache_internal(); 
    }
    
    current_winner = cached_winner_.load(std::memory_order_acquire);
    bool result = cached_valid_moves_.empty() && (current_winner == NO_PLAYER);
    return result;
}


void GomokuState::refresh_valid_moves_cache() const {
    // LOCK-FREE: Check if refresh is needed using atomic load
    if (!valid_moves_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    refresh_valid_moves_cache_internal();
}

void GomokuState::refresh_valid_moves_cache_internal() const {
    // Double-check pattern - another thread might have refreshed it
    if (!valid_moves_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    cached_valid_moves_.clear();
    
    // Check if we have a winner (no valid moves)
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    bool winner_dirty = winner_check_dirty_.load(std::memory_order_acquire);
    
    if (current_winner != NO_PLAYER && !winner_dirty) {
        // Game has winner, no valid moves
        valid_moves_dirty_.store(false, std::memory_order_release);
        return;
    }

    // Compute all valid moves
    int total_actions = getActionSpaceSize();
    for (int action = 0; action < total_actions; ++action) {
        if (is_move_valid_internal(action, true)) { 
            cached_valid_moves_.insert(action);
        }
    }
    
    valid_moves_dirty_.store(false, std::memory_order_release);
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
    valid_moves_dirty_.store(true, std::memory_order_release);
    winner_check_dirty_.store(true, std::memory_order_release);
    hash_dirty_.store(true, std::memory_order_release);
    
    // PERFORMANCE FIX: Return old tensors before invalidating caches
    clearTensorCache();
    
    // PERFORMANCE FIX: Invalidate tensor caches when game state changes
    tensor_cache_dirty_.store(true, std::memory_order_release);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_release);
}

GomokuState::~GomokuState() {
    // Return cached tensors to the pool to prevent memory leaks
    clearTensorCache();
}

void GomokuState::clearTensorCache() const {
    // Return cached tensors to the global pool
    if (!cached_tensor_repr_.empty()) {
        core::GlobalTensorPool::getInstance().returnTensor(
            const_cast<std::vector<std::vector<std::vector<float>>>&>(cached_tensor_repr_),
            3, board_size_, board_size_);
        cached_tensor_repr_.clear();
    }
    
    if (!cached_enhanced_tensor_repr_.empty()) {
        const int num_history_pairs = 8;
        const int num_feature_planes = 2 * num_history_pairs + 1;
        core::GlobalTensorPool::getInstance().returnTensor(
            const_cast<std::vector<std::vector<std::vector<float>>>&>(cached_enhanced_tensor_repr_),
            num_feature_planes, board_size_, board_size_);
        cached_enhanced_tensor_repr_.clear();
    }
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
            bool valid = (action == center_action);
            // Debug logging (commented out for production)
            // if (!valid && action == 0) { // Log only once for the first action checked
            //     std::cerr << "Pro-long opening: Black at move 0 must play center " << center_action 
            //              << ", but tried " << action << " (board_size=" << board_size_ << ")" << std::endl;
            // }
            return valid;
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