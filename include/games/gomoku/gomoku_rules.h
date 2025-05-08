// gomoku_rules.h
#ifndef GOMOKU_RULES_H
#define GOMOKU_RULES_H

#include <vector>
#include <set>
#include <utility>
#include <functional>
#include "utils/hash_specializations.h"
#include "core/export_macros.h"

namespace alphazero {
namespace games {
namespace gomoku {

/**
 * @brief Rules implementation for Gomoku
 */
class ALPHAZERO_API GomokuRules {
public:
    /**
     * @brief Constructor
     *
     * @param board_size Size of the board
     */
    explicit GomokuRules(int board_size) : board_size_(board_size) {}

    /**
     * @brief Set board access functions
     *
     * @param is_bit_set Function to check if a bit is set
     * @param coords_to_action Function to convert coordinates to action
     * @param action_to_coords Function to convert action to coordinates 
     * @param in_bounds Function to check if coordinates are in bounds
     */
    void setBoardAccessor(
        std::function<bool(int, int)> is_bit_set,
        std::function<int(int, int)> coords_to_action,
        std::function<std::pair<int, int>(int)> action_to_coords,
        std::function<bool(int, int)> in_bounds) {
        is_bit_set_ = is_bit_set;
        coords_to_action_ = coords_to_action;
        action_to_coords_ = action_to_coords;
        in_bounds_ = in_bounds;
    }

    /**
     * @brief Check if five in a row exists
     *
     * @param action Last action played or -1 to check entire board
     * @param player Player to check for
     * @return true if five in a row exists
     */
    bool is_five_in_a_row(int action, int player) const;

    /**
     * @brief Check if a move is forbidden in Renju rules
     *
     * @param action Action to check
     * @return true if forbidden
     */
    bool is_black_renju_forbidden(int action) const;

    /**
     * @brief Check if a move is forbidden in Omok rules
     *
     * @param action Action to check
     * @return true if forbidden
     */
    bool is_black_omok_forbidden(int action) const;

    /**
     * @brief Check if a line contains five-in-a-row
     * 
     * @param cell The cell to check from
     * @param player The player to check for
     * @param p_idx Player index (0 for black, 1 for white)
     * @return true if line contains five-in-a-row, false otherwise
     */
    bool check_line_for_five(int cell, int player, int p_idx) const;
    
    /**
     * @brief Count stones in a direction
     * 
     * @param x0 Starting X coordinate
     * @param y0 Starting Y coordinate
     * @param dx X direction
     * @param dy Y direction
     * @param p_idx Player index to count
     * @return Number of consecutive stones in the direction
     */
    int count_direction(int x0, int y0, int dx, int dy, int p_idx) const;
    
    /**
     * @brief Check if a move would create an overline (more than five in a row)
     * 
     * @param action The action to check
     * @return true if move creates overline, false otherwise
     */
    bool renju_is_overline(int action) const;
    
    /**
     * @brief Check if a move would create a double four or more
     * 
     * @param action The action to check
     * @return true if move creates double four or more, false otherwise
     */
    bool renju_double_four_or_more(int action) const;
    
    /**
     * @brief Check if a move would create a double three or more
     * 
     * @param action The action to check
     * @return true if move creates double three or more, false otherwise
     */
    bool renju_double_three_or_more(int action) const;
    
    /**
     * @brief Count the number of fours on the board
     * 
     * @return Number of fours
     */
    int renju_count_all_fours() const;
    
    /**
     * @brief Count the number of threes that would be created by a move
     * 
     * @param action The action to check
     * @return Number of threes
     */
    int renju_count_all_threes(int action) const;
    
    /**
     * @brief Check if a move would create an overline in Omok rules
     * 
     * @param action The action to check
     * @return true if move creates overline, false otherwise
     */
    bool omok_is_overline(int action) const;
    
    /**
     * @brief Check if a move would create a double three in Omok rules
     * 
     * @param action The action to check
     * @return true if move creates double three, false otherwise
     */
    bool omok_check_double_three_strict(int action) const;
    
    /**
     * @brief Count the number of open threes on the board
     * 
     * @return Number of open threes
     */
    int count_open_threes_globally() const;
    
    /**
     * @brief Get the three patterns for a specific action
     * 
     * @param action The action to check
     * @return Vector of three patterns (sets of positions)
     */
    std::vector<std::set<int>> get_three_patterns_for_action(int action) const;
    
    /**
     * @brief Get all open three patterns on the board
     * 
     * @return Vector of three patterns (sets of positions)
     */
    std::vector<std::set<int>> get_open_three_patterns_globally() const;

private:
    int board_size_;
    std::function<bool(int, int)> is_bit_set_;
    std::function<int(int, int)> coords_to_action_;
    std::function<std::pair<int, int>(int)> action_to_coords_;
    std::function<bool(int, int)> in_bounds_;
    
    // Helper function for win detection
    int check_line_from_point(int r, int c, int dr, int dc, int p_idx) const;

    // Pattern recognition helpers
    bool renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const;
    bool renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const;
    std::pair<bool, bool> ends_are_open(const std::vector<std::pair<int, int>>& segment) const;
    bool check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const;
    bool simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const;
    std::set<int> positions_of_black(const std::vector<std::pair<int, int>>& segment) const;
    bool try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                             const std::set<int>& new_fs, int size) const;
    bool try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                              const std::set<int>& new_fs, int action) const;
    std::set<int> check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const;
    bool are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const;
    
    // Enhanced double-three detection
    bool is_allowed_double_three(int action) const;
    bool can_make_straight_four(const std::set<int>& three_pattern) const;
    int count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const;
    bool is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                         int depth = 0, int max_depth = 3) const;
    bool is_straight_four(const std::set<int>& pattern) const;
    std::vector<int> find_three_to_four_placements(const std::set<int>& three_pattern) const;
    bool is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const;
    bool is_four_pattern(const std::vector<std::pair<int, int>>& segment) const;
    
    // Utility methods
    std::vector<std::pair<int, int>> build_entire_line(int x0, int y0, int dx, int dy) const;
};

} // namespace gomoku
} // namespace games
} // namespace alphazero

#endif // GOMOKU_RULES_H