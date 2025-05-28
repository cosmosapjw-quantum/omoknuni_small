// File: gomoku_state.h
#ifndef GOMOKU_STATE_H
#define GOMOKU_STATE_H

#include "core/igamestate.h"
#include "utils/zobrist_hash.h" // Make sure this path is correct
#include "core/export_macros.h" // Make sure this path is correct
#include <vector>
#include <string>
#include <sstream>
#include <algorithm> // For std::sort, std::find
#include <memory>    // For std::shared_ptr, std::make_unique
#include <random>    // Not strictly used here unless for a seeded Zobrist or similar
#include <iomanip>   // For std::setw in toString
#include <unordered_set> // For cached_valid_moves
#include <optional>  // For std::optional
#include <atomic>    // For lock-free thread safety
#include <mutex>     // For cache protection

// Forward declarations for GPU support
//namespace alphazero {
//    class GomokuGPUAttackDefense;
//}

namespace alphazero {
namespace games {
namespace gomoku {

// Constants for players
constexpr int NO_PLAYER = 0; // Represents an empty cell
constexpr int BLACK = 1;
constexpr int WHITE = 2;

// Forward declarations
class GomokuRules;

/**
 * @brief Gomoku game state implementation
 */
class ALPHAZERO_API GomokuState : public core::IGameState {
public:
    /**
     * @brief Constructor with configurable board size and rule options.
     * @param board_size Board size (e.g., 15 for 15x15).
     * @param use_renju Whether to use Renju rules for Black's forbidden moves.
     * @param use_omok Whether to use Omok rules for Black's forbidden moves.
     * @param seed Random seed, primarily for Zobrist hash initialization if it's randomized (though typically deterministic based on size).
     * @param use_pro_long_opening Whether to apply pro-long opening restrictions.
     */
    explicit GomokuState(int board_size = 15, bool use_renju = false, bool use_omok = false,
                        int seed = 0, bool use_pro_long_opening = false);

    /**
     * @brief Copy constructor.
     */
    GomokuState(const GomokuState& other);
    
    /**
     * @brief Destructor - returns cached tensors to pool
     */
    ~GomokuState();

    // --- IGameState Interface Implementation ---
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override; // Throws core::IllegalMoveException
    bool undoMove() override;
    bool isTerminal() const override;
    core::GameResult getGameResult() const override;
    int getCurrentPlayer() const override; // Returns BLACK (1) or WHITE (2)
    int getBoardSize() const override;
    int getActionSpaceSize() const override;
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<core::IGameState> clone() const override;
    void copyFrom(const core::IGameState& source) override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override; // For displaying the board
    bool equals(const core::IGameState& other) const override;
    std::vector<int> getMoveHistory() const override;
    bool validate() const override; // Basic validation of stone counts vs player turn

    // --- Testing Specific Methods (use with caution) ---
    /** @brief Sets a stone for testing purposes. Does not check game rules. Invalidates caches. */
    void setStoneForTesting(int r, int c, int player); // player can be NO_PLAYER, BLACK, WHITE
    /** @brief Sets the current player for testing. Invalidates caches. */
    void setCurrentPlayerForTesting(int player); // BLACK or WHITE
    /** @brief Clears the board and resets game state for testing. */
    void clearBoardForTesting();
    // Helper for tests to easily get action from coords
    int coordsToActionForTesting(int r, int c) const { return coords_to_action(r,c); }

    bool getRenjuRules() const { return use_renju_; }
    bool getOmokRules() const { return use_omok_; }
    bool getProLongOpening() const { return use_pro_long_opening_; }

private:
    int board_size_; 
    int current_player_; 
    std::vector<int> move_history_;
    core::ZobristHash zobrist_; 

    // Rule variants
    bool use_renju_;
    bool use_omok_;
    bool use_pro_long_opening_;
    int black_first_stone_; 

    // MCTS-OPTIMIZED CACHING: Fast atomic operations for thread safety
    mutable std::atomic<bool> valid_moves_dirty_;
    mutable std::atomic<int> cached_winner_; 
    mutable std::atomic<bool> winner_check_dirty_;
    mutable std::atomic<uint64_t> hash_signature_;
    mutable std::atomic<bool> hash_dirty_;
    
    // Fast early-game: compute valid moves on-demand without caching for MCTS performance
    // Only cache for mid/late game positions
    mutable std::unordered_set<int> cached_valid_moves_;
    mutable std::mutex cache_mutex_;  // Only used for late-game caching
    
    // PERFORMANCE FIX: Cached tensor representations to avoid expensive recomputation
    mutable std::vector<std::vector<std::vector<float>>> cached_tensor_repr_;
    mutable std::vector<std::vector<std::vector<float>>> cached_enhanced_tensor_repr_;
    mutable std::atomic<bool> tensor_cache_dirty_;
    mutable std::atomic<bool> enhanced_tensor_cache_dirty_;

    // Bitboard representation
    int num_words_; 
    std::vector<std::vector<uint64_t>> player_bitboards_; // [player_idx_0_based][word_idx]

    std::shared_ptr<GomokuRules> rules_engine_; 
    int last_action_played_; 

    // --- Internal Helper Methods ---
    // Bitboard operations (player_idx_0_based is 0 for BLACK, 1 for WHITE)
    bool is_bit_set(int player_idx_0_based, int action) const noexcept;
    void set_bit(int player_idx_0_based, int action);
    void clear_bit(int player_idx_0_based, int action) noexcept;

    // Coordinate and action conversion
    std::pair<int, int> action_to_coords_pair(int action) const noexcept;
    int coords_to_action(int r, int c) const noexcept;
    bool in_bounds(int r, int c) const noexcept;

    int count_total_stones() const noexcept; 

    // Cache management and game state computation
    void refresh_winner_cache() const;
    bool is_stalemate() const;         
    
    void refresh_valid_moves_cache() const;
    void refresh_valid_moves_cache_internal() const; // Internal version (assumes lock held) 
    bool is_move_valid_internal(int action, bool check_occupation = true) const; // Detailed check
    
    uint64_t compute_hash_signature_internal() const; 
    bool board_equal_internal(const GomokuState& other) const; 

    void make_move_internal(int action, int player_to_move);
    void undo_last_move_internal(int last_action_undone, int player_who_made_last_action);

    void invalidate_caches(); 
    bool is_occupied(int action) const; 
    bool is_any_bit_set_for_rules(int action) const; // Wrapper for rules_engine accessor
    
    // Tensor cache management
    void clearTensorCache() const;

    // Rule-specific helpers
    bool is_pro_long_opening_move_valid(int action, int total_stones_on_board) const;

    // GPU acceleration support
public:
    // Static methods for GPU initialization
    static void initializeGPU(int board_size);
    static void cleanupGPU();
    static void setGPUEnabled(bool enabled);
    static bool isGPUEnabled();
    
    // Batch computation support for multiple MCTS engines
    static std::vector<std::vector<std::vector<std::vector<float>>>> 
        computeEnhancedTensorBatch(const std::vector<const GomokuState*>& states);
    
private:
    // Static GPU resources shared across all instances
    //static std::unique_ptr<GomokuGPUAttackDefense> gpu_module_;
    static std::atomic<bool> gpu_enabled_;
    static std::mutex gpu_mutex_;
};

} // namespace gomoku
} // namespace games
} // namespace alphazero

#endif // GOMOKU_STATE_H