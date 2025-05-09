// File: gomoku_rules.h
#ifndef GOMOKU_RULES_H
#define GOMOKU_RULES_H

#include <vector>
#include <set>
#include <utility> // For std::pair
#include <functional> // For std::function
#include <string>
#include <algorithm> // For std::sort, std::includes, std::set_intersection, std::set_difference
#include <map>     // For internal logic if needed

// Assuming these are correctly pathed in your project
#include "utils/hash_specializations.h" // If used by this header directly
#include "core/export_macros.h"

namespace alphazero {
namespace games {
namespace gomoku {

// Forward declaration if GomokuState is used by reference/pointer in a private method signature.
// class GomokuState;

/**
 * @brief Rules implementation for Gomoku, Omok, and Renju.
 */
class ALPHAZERO_API GomokuRules {
public:
    /**
     * @brief Constructor
     * @param board_size Size of the board (e.g., 15 for 15x15)
     */
    explicit GomokuRules(int board_size);

    /**
     * @brief Sets the accessor functions that allow GomokuRules to query the board state.
     * @param is_bit_set Lambda: (player_idx_0_based, action) -> bool
     * @param is_any_bit_set Lambda: (action) -> bool (checks if cell is occupied by ANY player)
     * @param coords_to_action Lambda: (row, col) -> int_action
     * @param action_to_coords Lambda: (int_action) -> std::pair<row, col>
     * @param in_bounds Lambda: (row, col) -> bool
     */
    void setBoardAccessor(
        std::function<bool(int /*player_idx_0_based*/, int /*action*/)> is_bit_set,
        std::function<bool(int /*action*/)> is_any_bit_set,
        std::function<int(int /*r*/, int /*c*/)> coords_to_action,
        std::function<std::pair<int, int>(int /*action*/)> action_to_coords,
        std::function<bool(int /*r*/, int /*c*/)> in_bounds);

    /**
     * @brief Checks if the specified player has five (or more for White in Renju/Standard if overlines allowed) stones in a row.
     * @param action The last move made. If -1, scans the entire board.
     * @param player The player to check for (1 for BLACK, 2 for WHITE).
     * @param allow_overline If true, 6 or more in a row also counts as a win for the player being checked.
     * For Renju Black, this should effectively be false (exact 5 wins).
     * @return True if a winning line is found, false otherwise.
     */
    bool is_five_in_a_row(int action, int player, bool allow_overline) const;
    int get_line_length_at_action(int action, int player_idx, const std::function<bool(int,int)>& board_accessor, int dr, int dc) const;


    /**
     * @brief Checks if placing a stone at 'action' by Black is a forbidden move under Renju rules.
     * @param action The action (cell index) to check.
     * @return True if the move is forbidden for Black, false otherwise.
     */
    bool is_black_renju_forbidden(int action) const;

    /**
     * @brief Checks if placing a stone at 'action' by Black is a forbidden move under Omok rules.
     * (Double-threes and six-in-a-row are forbidden for Black in Omok).
     * @param action The action (cell index) to check.
     * @return True if the move is forbidden for Black, false otherwise.
     */
    bool is_black_omok_forbidden(int action) const;

    // Publicly accessible for complex win condition checks in GomokuState or AI evaluation
    bool renju_is_overline(int action, int player_idx, const std::function<bool(int, int)>& current_board_state) const;
    bool renju_makes_double_four(int action, int player_idx, const std::function<bool(int, int)>& current_board_state) const;
    bool is_renju_double_three_forbidden(int action, int player_idx, const std::function<bool(int, int)>& current_board_state) const;

    // Omok Forbidden Move Helpers - MOVED TO PUBLIC
    bool omok_is_six_in_a_row(int action, int player_idx, const std::function<bool(int, int)>& current_board_state) const;
    bool omok_makes_double_three(int action, int player_idx, const std::function<bool(int, int)>& current_board_state) const;
    // bool is_open_three(int action, int player_idx, const std::function<bool(int, int)>& board_accessor, int dr, int dc) const; // Likely unused

private:
    int board_size_;
    std::function<bool(int /*player_idx_0_based*/, int /*action*/)> is_bit_set_;
    std::function<bool(int /*action*/)> is_any_bit_set_; // Checks if cell is occupied
    std::function<int(int /*r*/, int /*c*/)> coords_to_action_;
    std::function<std::pair<int, int>(int /*action*/)> action_to_coords_;
    std::function<bool(int /*r*/, int /*c*/)> in_bounds_;

    // Helper to check if a location is empty
    bool is_empty_spot(int r, int c, const std::function<bool(int, int)>& board_accessor) const;

    // Omok helper
    bool check_omok_open_three_in_direction(int action, int player_idx, const std::function<bool(int, int)>& board_accessor, int dr, int dc) const;

    // Helper to find open threes in a specific direction - LIKELY OBSOLETE by new Omok logic
    // std::set<std::vector<int>> find_open_threes_in_direction(
    //     int action, int player_idx, const std::function<bool(int, int)>& board_accessor, int dr, int dc) const;

    // Count the number of "fours" that would be formed
    int count_renju_fours_at_action(int action, int player_idx, const std::function<bool(int, int)>& board_accessor) const;

    int count_stones_in_line(int r_start, int c_start, int dr, int dc, int player_idx,
                              const std::function<bool(int, int)>& board_accessor,
                              int& open_ends, bool count_gaps_as_part_of_line = false) const;

    // Renju Forbidden Move Helpers
    // Modified signature for is_renju_double_three_forbidden
    bool check_renju_double_three_recursive(
        int action, 
        int player_idx, 
        const std::function<bool(int,int)>& board_accessor_after_action,
        bool is_recursive_call = false, // Default for direct calls
        // Optional: For recursive checks, a checker for S4's own forbidden status
        const std::function<bool(int /*s4_action*/, const std::function<bool(int,int)>& /*board_after_s4*/)>* s4_forbidden_checker = nullptr
    ) const;

    enum class RenjuLineType { NONE, THREE, FOUR, STRAIGHT_FOUR, FIVE, OVERLINE };
    struct RenjuPatternInfo {
        std::set<int> stones;
        RenjuLineType type = RenjuLineType::NONE;
        int open_level = 0; 
        bool forms_five_or_more_with_action = false; 
        std::pair<int,int> dir = {0,0}; // Direction of the pattern

        bool operator<(const RenjuPatternInfo& other) const {
            if (stones != other.stones) return stones < other.stones;
            if (type != other.type) return type < other.type;
            return dir < other.dir;
        }
    };

    // std::vector<RenjuPatternInfo> get_renju_patterns_for_action(int action, int player_idx, const std::function<bool(int,int)>& board_accessor) const; // Likely obsolete
    
    // RenjuLineType get_renju_line_type_and_openness(const std::vector<int>& line_actions, int player_idx, // Likely obsolete
    //                                                const std::function<bool(int, int)>& board_accessor,
    //                                                std::set<int>& out_stones, int& out_open_level, bool& out_makes_five_or_more, int action_stone) const;


    // Modified signature for is_renju_three_definition
    bool is_renju_three_definition(const std::set<int>& three_candidate_stones, int action_that_formed_it, int player_idx,
                                   const std::function<bool(int,int)>& board_accessor_after_action,
                                   const std::function<bool(int /*s4_action*/, const std::function<bool(int,int)>& /*board_after_s4*/)>& is_s4_forbidden_func
                                   ) const;

    bool is_renju_straight_four_from_stones(const std::set<int>& four_stones, int player_idx, const std::function<bool(int,int)>& board_accessor) const;


    bool check_renju_double_three_exception_recursive(
        int original_action, 
        const std::vector<RenjuPatternInfo>& threes_involved, 
        int player_idx,
        const std::function<bool(int, int)>& board_after_original_action,
        int depth,
        const std::function<bool(int, int, const std::function<bool(int,int)>&)>& is_overline_func,
        const std::function<bool(int, int, const std::function<bool(int,int)>&)>& makes_double_four_func,
        const std::function<bool(int, int, const std::function<bool(int,int)>&)>& is_double_three_forbidden_external_func
    ) const;

    std::vector<int> get_empty_extensions_of_line(const std::set<int>& stones, int player_idx, const std::function<bool(int,int)>& board_accessor, int dr, int dc) const;

    bool get_line_direction_and_bounds(const std::set<int>& stones,
                                   int& dr, int& dc,
                                   int& min_r, int& min_c,
                                   int& max_r, int& max_c) const;
    
    // Special wrapper just for the call in is_black_renju_forbidden
    bool is_black_renju_d3_forbidden(int action, int player_idx, const std::function<bool(int, int)>& board_func) const {
        // Explicitly qualify the call to resolve ambiguity
        return this->is_renju_double_three_forbidden(action, player_idx, board_func);
    }
};

} // namespace gomoku
} // namespace games
} // namespace alphazero

#endif // GOMOKU_RULES_H