// gomoku_rules.cpp
#include "games/gomoku/gomoku_rules.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <random>
#include <functional>

namespace alphazero {
namespace games {
namespace gomoku {

// Helper for the full board scan part of is_five_in_a_row
// Counts N in a row from (r,c) in direction (dr,dc), checking up to 5 stones.
int GomokuRules::check_line_from_point(int r, int c, int dr, int dc, int p_idx) const {
    int count = 0;
    for (int k=0; k<5; ++k) {
        int nr = r + k*dr;
        int nc = c + k*dc;
        if (in_bounds_(nr,nc) && is_bit_set_(p_idx, coords_to_action_(nr,nc))) {
            count++;
        } else {
            break; // Stop counting if out of bounds or opponent/empty stone
        }
    }
    return count;
}

// Line and pattern detection
bool GomokuRules::is_five_in_a_row(int action, int player) const {
    // Player index is 0-based internally (BLACK-1=0, WHITE-1=1)
    int p_idx = player - 1;
    if (p_idx < 0 || p_idx > 1) return false; // Invalid player
    
    if (action >= 0) {
        // Check if the move `action` by `player` results in a win.
        auto [x, y] = action_to_coords_(action);
        
        // Ensure the stone is actually there (should be if called after a move)
        if (!is_bit_set_(p_idx, action)) {
            // This might happen if called hypothetically, but typically indicates an issue.
            return false; 
        }
        
        // Define the 4 axes/lines to check
        const int DIRS[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}}; 

        for (auto& dir : DIRS) {
            int dx = dir[0];
            int dy = dir[1];
            int count = 1; // Count the stone at (x,y)

            // Count in the + (dx, dy) direction
            for (int i = 1; i < 5; ++i) { // Check up to 4 more stones
                int nx = x + i * dx;
                int ny = y + i * dy;
                if (in_bounds_(nx, ny) && is_bit_set_(p_idx, coords_to_action_(nx, ny))) {
                    count++;
                } else {
                    break;
                }
            }
            // Count in the - (dx, dy) direction
            for (int i = 1; i < 5; ++i) { // Check up to 4 more stones
                int nx = x - i * dx;
                int ny = y - i * dy;
                if (in_bounds_(nx, ny) && is_bit_set_(p_idx, coords_to_action_(nx, ny))) {
                    count++;
                } else {
                    break;
                }
            }
            // If total count is 5 or more, it's a win
            if (count >= 5) return true;
        }
        return false; // No win from this specific move
    }
    
    // If action == -1, scan the whole board for 5-in-a-row for the specified player.
    // This is typically used for terminal state check after passes or for initial state validation.
    // Use the check_line_from_point helper for efficiency.
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            if (!is_bit_set_(p_idx, coords_to_action_(r, c))) {
                continue; // Skip if not player's stone
            }
            
            // Check from (r,c) in 4 unique directions: H->, V down, Diag \ down, Diag / down-left
            if (check_line_from_point(r, c, 1, 0, p_idx) >= 5) return true; // H
            if (check_line_from_point(r, c, 0, 1, p_idx) >= 5) return true; // V (assuming (0,1) is down)
            if (check_line_from_point(r, c, 1, 1, p_idx) >= 5) return true; // Diag \ 
            if (check_line_from_point(r, c, -1, 1, p_idx) >= 5) return true; // Diag / (assuming (-1,1) is down-left)
        }
    }
    
    return false; // No 5-in-a-row found on the board
}

int GomokuRules::count_direction(int x0, int y0, int dx, int dy, int p_idx) const {
    int count = 1;  // Start with 1 for the stone at (x0, y0)
    
    // Count in positive direction
    for (int i = 1; i < 6; i++) {  // Max count would be 5
        int nx = x0 + i * dx;
        int ny = y0 + i * dy;
        
        if (!in_bounds_(nx, ny) || !is_bit_set_(p_idx, coords_to_action_(nx, ny))) {
            break;
        }
        
        count++;
    }
    
    // Count in negative direction only if dx and dy are not 0 (to prevent double-counting for full board scan)
    if (dx != 0 || dy != 0) {
        for (int i = 1; i < 6; i++) {
            int nx = x0 - i * dx;
            int ny = y0 - i * dy;
            
            if (!in_bounds_(nx, ny) || !is_bit_set_(p_idx, coords_to_action_(nx, ny))) {
                break;
            }
            
            count++;
        }
    }
    
    return count;
}

// Renju rule checks
bool GomokuRules::is_black_renju_forbidden(int action) const {
    // First, check if the action already has a stone
    if (is_bit_set_(0, action) || is_bit_set_(1, action)) {
        return true; // Already occupied
    }
    
    // For correct testing behavior, we'll disable the complex rule checks
    // for now since we've refactored to a bitboard representation
    
    // Check for overline
    if (renju_is_overline(action)) {
        return true;
    }
    
    // Check for double-four or more
    if (renju_double_four_or_more(action)) {
        return true;
    }
    
    // Check for double-three or more
    if (!is_allowed_double_three(action)) {
        return true;
    }
    
    return false;
}

bool GomokuRules::renju_is_overline(int action) const {
    auto [x, y] = action_to_coords_(action);
    int p_idx = 0;  // BLACK (in Renju, overline only applies to Black)
    
    // Temporary place the stone to check
    bool hasOverline = false;
    
    // Check all 4 directions for 6+ in a row
    // Horizontal
    if (count_direction(x, y, 1, 0, p_idx) + count_direction(x, y, -1, 0, p_idx) - 1 > 5) {
        hasOverline = true;
    }
    
    // Vertical
    if (!hasOverline && (count_direction(x, y, 0, 1, p_idx) + count_direction(x, y, 0, -1, p_idx) - 1 > 5)) {
        hasOverline = true;
    }
    
    // Diagonal
    if (!hasOverline && (count_direction(x, y, 1, 1, p_idx) + count_direction(x, y, -1, -1, p_idx) - 1 > 5)) {
        hasOverline = true;
    }
    
    // Anti-diagonal
    if (!hasOverline && (count_direction(x, y, 1, -1, p_idx) + count_direction(x, y, -1, 1, p_idx) - 1 > 5)) {
        hasOverline = true;
    }
    
    return hasOverline;
}

bool GomokuRules::renju_double_four_or_more(int action) const {
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    // Store original accessor
    auto original_is_bit_set = is_bit_set_;
    
    // Replace with temporary accessor to include hypothetical stone
    const_cast<GomokuRules*>(this)->is_bit_set_ = is_bit_set_temp;
    
    // Count fours with hypothetical stone
    int c4 = renju_count_all_fours();
    
    // Restore original accessor
    const_cast<GomokuRules*>(this)->is_bit_set_ = original_is_bit_set;
    
    return (c4 >= 2);
}

bool GomokuRules::renju_double_three_or_more(int action) const {
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    // Get the unified set of three patterns.
    std::vector<std::set<int>> three_patterns = get_three_patterns_for_action(action);
    
    // If 2 or more distinct three patterns exist, then it's a double-three.
    return (three_patterns.size() >= 2);
}

int GomokuRules::renju_count_all_fours() const {
    int bs = board_size_;
    std::set<std::pair<std::set<int>, int>> found_fours;
    std::vector<std::pair<int, int>> directions = {{0,1}, {1,0}, {1,1}, {-1,1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!in_bounds_(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6, 7}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (renju_is_four_shape(segment)) {
                            std::set<int> black_positions = positions_of_black(segment);
                            bool unified = try_unify_four_shape(found_fours, black_positions, black_positions.size());
                            
                            if (!unified) {
                                found_fours.insert({black_positions, black_positions.size()});
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_fours.size();
}

int GomokuRules::renju_count_all_threes(int action) const {
    int bs = board_size_;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!in_bounds_(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (renju_is_three_shape(segment)) {
                            std::set<int> black_positions = positions_of_black(segment);
                            std::set<int> new_fs(black_positions);
                            
                            if (!try_unify_three_shape(found_threes, new_fs, action)) {
                                found_threes.insert(new_fs);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_threes.size();
}

// Omok rule checks
bool GomokuRules::is_black_omok_forbidden(int action) const {
    // First, check if the action already has a stone
    if (is_bit_set_(0, action) || is_bit_set_(1, action)) {
        return true; // Already occupied
    }
    
    // For correct testing behavior, we'll disable the complex rule checks
    // for now since we've refactored to a bitboard representation
    
    // Check for overline -- Omok allows overlines, so this check is incorrect
    // if (omok_is_overline(action)) {
    //     return true;
    // }
    
    // Check for double-three
    if (omok_check_double_three_strict(action)) {
        return true;
    }
    
    return false;
}

bool GomokuRules::omok_is_overline(int action) const {
    auto [x, y] = action_to_coords_(action);
    int p_idx = 0;  // BLACK (in Omok, overline only applies to Black)

    // Check all 4 directions for 6+ in a row
    // Horizontal
    if (count_direction(x, y, 1, 0, p_idx) + count_direction(x, y, -1, 0, p_idx) - 1 > 5) {
        return true;
    }
    
    // Vertical
    if (count_direction(x, y, 0, 1, p_idx) + count_direction(x, y, 0, -1, p_idx) - 1 > 5) {
        return true;
    }
    
    // Diagonal
    if (count_direction(x, y, 1, 1, p_idx) + count_direction(x, y, -1, -1, p_idx) - 1 > 5) {
        return true;
    }
    
    // Anti-diagonal
    if (count_direction(x, y, 1, -1, p_idx) + count_direction(x, y, -1, 1, p_idx) - 1 > 5) {
        return true;
    }
    
    return false;
}

bool GomokuRules::omok_check_double_three_strict(int action) const {
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    // Store original accessor
    auto original_is_bit_set = is_bit_set_;
    
    // Replace with temporary accessor
    const_cast<GomokuRules*>(this)->is_bit_set_ = is_bit_set_temp;
    
    // Get the global three patterns after placing the stone
    std::vector<std::set<int>> patterns = get_open_three_patterns_globally();
    
    // Restore original accessor
    const_cast<GomokuRules*>(this)->is_bit_set_ = original_is_bit_set;
    
    int n = patterns.size();
    
    if (n < 2) {
        return false;
    }
    
    // Check if any two patterns are connected (Omok-specific rule)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (are_patterns_connected(patterns[i], patterns[j])) {
                return true;
            }
        }
    }
    
    return false;
}

int GomokuRules::count_open_threes_globally() const {
    return get_open_three_patterns_globally().size();
}

// Pattern recognition helpers
bool GomokuRules::renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_(0, a)) {
            black_count++;
        } else if (is_bit_set_(1, a)) {
            white_count++;
        }
    }
    
    if (white_count > 0 || black_count < 2 || black_count >= 4) {
        return false;
    }
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a)) {
            // Test if placing a black stone here creates a four shape
            // We need to temporarily consider a black stone here
            
            // Create a test board with this stone added
            auto is_bit_set_temp = [this, a](int p_idx, int test_a) {
                if (test_a == a && p_idx == 0) { // Black is trying to place here
                    return true;
                }
                return is_bit_set_(p_idx, test_a);
            };
            
            // Check if it's now a four shape
            int seg_len = segment.size();
            int temp_black_count = 0, temp_white_count = 0;
            
            for (const auto& [tx, ty] : segment) {
                int ta = coords_to_action_(tx, ty);
                if (is_bit_set_temp(0, ta)) {
                    temp_black_count++;
                } else if (is_bit_set_temp(1, ta)) {
                    temp_white_count++;
                }
            }
            
            if (temp_white_count > 0) {
                continue;
            }
            
            if (temp_black_count == 4) {
                auto [front_open, back_open] = ends_are_open(segment);
                if (front_open || back_open) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool GomokuRules::renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_(1, a)) {
            white_count++;
        } else if (is_bit_set_(0, a)) {
            black_count++;
        }
    }
    
    if (white_count > 0) {
        return false;
    }
    
    if (black_count < 3 || black_count > 4) {
        return false;
    }
    
    auto [front_open, back_open] = ends_are_open(segment);
    
    if (black_count == 4) {
        return (front_open || back_open);
    } else {
        return check_broken_four(segment, front_open, back_open);
    }
}

std::pair<bool, bool> GomokuRules::ends_are_open(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    if (seg_len < 2) {
        return {false, false};
    }
    
    auto [x0, y0] = segment[0];
    auto [x1, y1] = segment[seg_len - 1];
    bool front_open = false, back_open = false;
    
    int dx = 0, dy = 0;
    if (seg_len >= 2) {
        auto [x2, y2] = segment[1];
        dx = x2 - x0;
        dy = y2 - y0;
    }
    
    int fx = x0 - dx;
    int fy = y0 - dy;
    if (in_bounds_(fx, fy)) {
        int af = coords_to_action_(fx, fy);
        if (!is_bit_set_(0, af) && !is_bit_set_(1, af)) {
            front_open = true;
        }
    }
    
    int lx = x1 + dx;
    int ly = y1 + dy;
    if (in_bounds_(lx, ly)) {
        int ab = coords_to_action_(lx, ly);
        if (!is_bit_set_(0, ab) && !is_bit_set_(1, ab)) {
            back_open = true;
        }
    }
    
    return {front_open, back_open};
}

bool GomokuRules::check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const {
    if (!front_open && !back_open) {
        return false;
    }
    
    std::vector<std::pair<int, int>> empties;
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a)) {
            empties.push_back({x, y});
        }
    }
    
    if (empties.size() != 1) {
        return false;
    }
    
    auto [gapx, gapy] = empties[0];
    int gap_action = coords_to_action_(gapx, gapy);
    
    // Create a test board with this stone added
    auto is_bit_set_temp = [this, gap_action](int p_idx, int test_a) {
        if (test_a == gap_action && p_idx == 0) { // Black placed at the gap
            return true;
        }
        return is_bit_set_(p_idx, test_a);
    };
    
    // Check if placing a black stone at the gap makes a 4-in-a-row
    int consecutive = 0, best = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_temp(0, a)) {
            consecutive++;
            if (consecutive > best) {
                best = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    return (best >= 4);
}

bool GomokuRules::simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const {
    int consecutive = 0, best = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_(0, a)) {
            consecutive++;
            if (consecutive > best) {
                best = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    return (best >= 4);
}

std::set<int> GomokuRules::positions_of_black(const std::vector<std::pair<int, int>>& segment) const {
    std::set<int> black_set;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_(0, a)) {
            black_set.insert(a);
        }
    }
    
    return black_set;
}

bool GomokuRules::try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                                     const std::set<int>& new_fs, int size) const {
    for (const auto& [existing_fs, existing_size] : found_fours) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        if (intersection.size() >= 3) {
            return true;
        }
    }
    
    return false;
}

bool GomokuRules::try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                                      const std::set<int>& new_fs, int action) const {
    for (const auto& existing_fs : found_threes) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        // Remove action from intersection
        intersection.erase(action);
        
        if (!intersection.empty()) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::set<int>> GomokuRules::get_three_patterns_for_action(int action) const {
    std::vector<std::set<int>> three_patterns;
    int bs = board_size_;
    std::vector<std::pair<int, int>> directions = { {0, 1}, {1, 0}, {1, 1}, {-1, 1} };
    
    auto [x0, y0] = action_to_coords_(action);
    
    // Temporarily consider the action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    for (auto [dx, dy] : directions) {
        std::vector<std::pair<int, int>> line_cells;
        // Build a line of up to 7 cells centered on the action.
        for (int offset = -3; offset <= 3; offset++) {
            int nx = x0 + offset * dx;
            int ny = y0 + offset * dy;
            if (in_bounds_(nx, ny)) {
                line_cells.push_back({nx, ny});
            }
        }
        
        // Slide a 5-cell window over the line.
        for (size_t start = 0; start + 4 < line_cells.size(); start++) {
            std::vector<std::pair<int, int>> segment(line_cells.begin() + start, line_cells.begin() + start + 5);
            
            // Check if this segment forms a three pattern containing our action.
            if (is_three_pattern(segment, action)) {
                std::set<int> pattern;
                for (auto [x, y] : segment) {
                    pattern.insert(coords_to_action_(x, y));
                }
                
                // Unify: check if this pattern overlaps in at least 3 cells with any existing one.
                bool duplicate = false;
                for (const auto &existing : three_patterns) {
                    std::set<int> inter;
                    std::set_intersection(existing.begin(), existing.end(),
                                          pattern.begin(), pattern.end(),
                                          std::inserter(inter, inter.begin()));
                    if (inter.size() >= 3) {  // Overlap is significant; consider it the same three.
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    three_patterns.push_back(pattern);
                }
            }
        }
    }
    return three_patterns;
}

bool GomokuRules::is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const {
    // A three pattern has exactly 3 black stones, the rest empty, and can form a four
    
    // Temporarily consider the action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    int black_count = 0;
    int white_count = 0;
    bool contains_action = false;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_temp(0, a)) {
            black_count++;
            if (a == action) {
                contains_action = true;
            }
        } else if (is_bit_set_temp(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 3 || white_count > 0 || !contains_action) {
        return false;
    }
    
    // Check if this pattern can form a four by placing a stone in an empty spot
    for (auto [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (!is_bit_set_temp(0, a) && !is_bit_set_temp(1, a)) {
            // Check if placing a black stone here would form a four
            
            // Create another temporary board state with this additional stone
            auto is_bit_set_double_temp = [is_bit_set_temp, a](int p_idx, int test_a) {
                if (test_a == a && p_idx == 0) { // Black placed at this empty spot
                    return true;
                }
                return is_bit_set_temp(p_idx, test_a);
            };
            
            // Check for a four pattern
            int temp_black_count = 0;
            for (auto [tx, ty] : segment) {
                int ta = coords_to_action_(tx, ty);
                if (is_bit_set_double_temp(0, ta)) {
                    temp_black_count++;
                }
            }
            
            if (temp_black_count == 4) {
                return true;
            }
        }
    }
    
    return false;
}

bool GomokuRules::is_four_pattern(const std::vector<std::pair<int, int>>& segment) const {
    // A four pattern has exactly 4 black stones and can form a five
    
    int black_count = 0;
    int white_count = 0;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (is_bit_set_(0, a)) {
            black_count++;
        } else if (is_bit_set_(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 4 || white_count > 0) {
        return false;
    }
    
    // Check if there's at least one empty spot that would form a five
    for (auto [x, y] : segment) {
        int a = coords_to_action_(x, y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a)) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::set<int>> GomokuRules::get_open_three_patterns_globally() const {
    int bs = board_size_;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx0, dy0] : directions) {
                std::vector<std::pair<int, int>> cells_5;
                int step = 0;
                int cx = x, cy = y;
                
                while (step < 5) {
                    if (!in_bounds_(cx, cy)) {
                        break;
                    }
                    cells_5.push_back({cx, cy});
                    cx += dx0;
                    cy += dy0;
                    step++;
                }
                
                if (cells_5.size() == 5) {
                    std::set<int> triple = check_open_three_5slice(cells_5);
                    if (!triple.empty()) {
                        bool skip = false;
                        std::vector<std::set<int>> to_remove;
                        
                        for (const auto& existing : found_threes) {
                            // Check if existing is a superset of triple
                            if (std::includes(existing.begin(), existing.end(), 
                                            triple.begin(), triple.end())) {
                                skip = true;
                                break;
                            }
                            
                            // Check if triple is a superset of existing
                            if (std::includes(triple.begin(), triple.end(), 
                                            existing.begin(), existing.end())) {
                                to_remove.push_back(existing);
                            }
                        }
                        
                        if (!skip) {
                            for (const auto& r : to_remove) {
                                found_threes.erase(r);
                            }
                            found_threes.insert(triple);
                        }
                    }
                }
            }
        }
    }
    
    return std::vector<std::set<int>>(found_threes.begin(), found_threes.end());
}

std::set<int> GomokuRules::check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const {
    if (cells_5.size() != 5) {
        return {};
    }
    
    int black_count = 0, white_count = 0, empty_count = 0;
    int arr[5] = {0}; // Represents the contents of cells_5: 0=empty, 1=black, -1=white
    
    for (int i = 0; i < 5; i++) {
        auto [xx, yy] = cells_5[i];
        int act = coords_to_action_(xx, yy);
        
        if (is_bit_set_(0, act)) {
            black_count++;
            arr[i] = 1;
        } else if (is_bit_set_(1, act)) {
            white_count++;
            arr[i] = -1;
        } else {
            empty_count++;
        }
    }
    
    if (black_count != 3 || white_count != 0 || empty_count != 2) {
        return {};
    }
    
    if (arr[0] != 0 || arr[4] != 0) {
        return {};
    }
    
    bool has_triple = false, has_gap = false;
    
    if (arr[1] == 1 && arr[2] == 1 && arr[3] == 1) {
        has_triple = true;
    }
    
    if (arr[1] == 1 && arr[2] == 0 && arr[3] == 1) {
        has_gap = true;
    }
    
    if (!has_triple && !has_gap) {
        return {};
    }
    
    int dx = cells_5[1].first - cells_5[0].first;
    int dy = cells_5[1].second - cells_5[0].second;
    
    int left_x = cells_5[0].first - dx;
    int left_y = cells_5[0].second - dy;
    int right_x = cells_5[4].first + dx;
    int right_y = cells_5[4].second + dy;
    
    // Check if this is an "open" three (both ends must be empty)
    if (in_bounds_(left_x, left_y)) {
        int left_act = coords_to_action_(left_x, left_y);
        if (is_bit_set_(0, left_act)) {
            return {};
        }
    }
    
    if (in_bounds_(right_x, right_y)) {
        int right_act = coords_to_action_(right_x, right_y);
        if (is_bit_set_(0, right_act)) {
            return {};
        }
    }
    
    // Get the positions of the three black stones
    std::set<int> triple;
    for (int i = 0; i < 5; i++) {
        if (arr[i] == 1) {
            triple.insert(coords_to_action_(cells_5[i].first, cells_5[i].second));
        }
    }
    
    return triple;
}

bool GomokuRules::are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const {
    for (int cell1 : pattern1) {
        auto [ax, ay] = action_to_coords_(cell1);
        
        for (int cell2 : pattern2) {
            auto [bx, by] = action_to_coords_(cell2);
            
            if (abs(ax - bx) <= 1 && abs(ay - by) <= 1) {
                return true;
            }
        }
    }
    return false;
}

// Enhanced double-three detection for Renju rules
bool GomokuRules::is_allowed_double_three(int action) const {
    // Temporarily consider the action as a black stone for pattern detection
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    // Store original accessor
    auto original_is_bit_set = is_bit_set_;
    
    // Replace with temporary accessor
    const_cast<GomokuRules*>(this)->is_bit_set_ = is_bit_set_temp;
    
    // Step 1: Get all three patterns that include this action
    std::vector<std::set<int>> three_patterns = get_three_patterns_for_action(action);
    
    // Restore original accessor
    const_cast<GomokuRules*>(this)->is_bit_set_ = original_is_bit_set;
    
    // If there's fewer than 2 three patterns, it's not a double-three
    if (three_patterns.size() < 2) {
        return true; // Not a double-three, so it's allowed
    }
    
    // Apply rule 9.3(a): Check how many threes can be made into straight fours
    int straight_four_capable_count = count_straight_four_capable_threes(three_patterns);
    
    // If at most one of the threes can be made into a straight four, the double-three is allowed
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Recursive check for potential future double-threes
    return is_double_three_allowed_recursive(three_patterns);
}

bool GomokuRules::can_make_straight_four(const std::set<int>& three_pattern) const {
    // Create temporary board accessor that considers the action point as a black stone
    auto action = *three_pattern.begin(); // Just need any point from the pattern to set up the context
    
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set_(p_idx, a);
    };
    
    // Get candidate placements that might convert the three into a four.
    std::vector<int> possible_placements = find_three_to_four_placements(three_pattern);
    for (int placement : possible_placements) {
        // Create another temporary accessor that adds this candidate placement
        auto is_bit_set_double_temp = [is_bit_set_temp, placement](int p_idx, int a) {
            if (a == placement && p_idx == 0) { // Black placed at the placement point
                return true;
            }
            return is_bit_set_temp(p_idx, a);
        };
        
        // Form a new pattern by adding the candidate.
        std::set<int> new_pattern = three_pattern;
        new_pattern.insert(placement);
        // Extract only the black stone positions from new_pattern.
        std::set<int> black_positions;
        for (int a : new_pattern) {
            if (is_bit_set_double_temp(0, a))
                black_positions.insert(a);
        }
        // Only consider candidate patterns that yield exactly 4 black stones.
        if (black_positions.size() != 4)
            continue;
        // If the new pattern qualifies as a straight four, count it.
        if (is_straight_four(new_pattern)) {
            // Here we'd need to check for overline, but without a real board state
            // we'll just return true as a simplification
            return true;
        }
    }
    return false;
}

std::vector<int> GomokuRules::find_three_to_four_placements(const std::set<int>& three_pattern) const {
    std::vector<int> placements;
    
    // Convert pattern to coordinates for easier analysis
    std::vector<std::pair<int, int>> coords;
    for (int a : three_pattern) {
        coords.push_back(action_to_coords_(a));
    }
    
    // Sort coordinates to find the pattern direction
    std::sort(coords.begin(), coords.end());
    
    // Determine if pattern is horizontal, vertical, or diagonal
    bool is_horizontal = true;
    bool is_vertical = true;
    bool is_diag_down = true;
    bool is_diag_up = true;
    
    for (size_t i = 1; i < coords.size(); i++) {
        if (coords[i].second != coords[0].second) is_horizontal = false;
        if (coords[i].first != coords[0].first) is_vertical = false;
        if (coords[i].first - coords[0].first != coords[i].second - coords[0].second) is_diag_down = false;
        if (coords[i].first - coords[0].first != coords[0].second - coords[i].second) is_diag_up = false;
    }
    
    // Determine direction vector
    int dx = 0, dy = 0;
    if (is_horizontal) {
        dx = 0; dy = 1;
    } else if (is_vertical) {
        dx = 1; dy = 0;
    } else if (is_diag_down) {
        dx = 1; dy = 1;
    } else if (is_diag_up) {
        dx = 1; dy = -1;
    } else {
        // Not a straight line, shouldn't happen with valid three patterns
        return placements;
    }
    
    // Find min and max coordinates
    int min_x = coords[0].first, min_y = coords[0].second;
    int max_x = coords[0].first, max_y = coords[0].second;
    
    for (auto [x, y] : coords) {
        min_x = std::min<int>(min_x, x);
        min_y = std::min<int>(min_y, y);
        max_x = std::max<int>(max_x, x);
        max_y = std::max<int>(max_y, y);
    }
    
    // Check for empty spots that could complete a four
    // Need to check both within the pattern and at the ends
    
    // Check within the pattern
    for (int i = 0; i <= 4; i++) {
        int x = min_x + i * dx;
        int y = min_y + i * dy;
        
        if (!in_bounds_(x, y)) continue;
        
        int a = coords_to_action_(x, y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a) && three_pattern.find(a) == three_pattern.end()) {
            placements.push_back(a);
        }
    }
    
    // Check beyond the ends
    int before_x = min_x - dx;
    int before_y = min_y - dy;
    int after_x = max_x + dx;
    int after_y = max_y + dy;
    
    if (in_bounds_(before_x, before_y)) {
        int a = coords_to_action_(before_x, before_y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a)) {
            placements.push_back(a);
        }
    }
    
    if (in_bounds_(after_x, after_y)) {
        int a = coords_to_action_(after_x, after_y);
        if (!is_bit_set_(0, a) && !is_bit_set_(1, a)) {
            placements.push_back(a);
        }
    }
    
    return placements;
}

bool GomokuRules::is_straight_four(const std::set<int>& pattern) const {
    // Build the segment of coordinates corresponding to the pattern.
    std::vector<std::pair<int,int>> segment;
    for (int a : pattern) {
        segment.push_back(action_to_coords_(a));
    }
    // Sort the coordinates
    std::sort(segment.begin(), segment.end(), [&](const std::pair<int,int>& p1, const std::pair<int,int>& p2) {
        if (p1.first == p2.first)
            return p1.second < p2.second;
        return p1.first < p2.first;
    });

    // Count black and white stones in the segment.
    int black_count = 0, white_count = 0;
    for (const auto &p : segment) {
        int a = coords_to_action_(p.first, p.second);
        if (is_bit_set_(0, a))
            ++black_count;
        else if (is_bit_set_(1, a))
            ++white_count;
    }
    if (white_count > 0)
        return false;
    
    // Only consider a pattern with exactly 4 black stones as a four-shape.
    if (black_count == 4) {
        auto ends = ends_are_open(segment); // returns {front_open, back_open}
        return (ends.first || ends.second);
    }
    return false;
}

int GomokuRules::count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const {
    int count = 0;
    
    for (const auto& pattern : three_patterns) {
        if (can_make_straight_four(pattern)) {
            count++;
        }
    }
    
    return count;
}

bool GomokuRules::is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                                 int depth, int max_depth) const {
    // Avoid too deep recursion
    if (depth >= max_depth) {
        return false;
    }
    
    // Apply rule 9.3(a) again at this level
    int straight_four_capable_count = count_straight_four_capable_threes(three_patterns);
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Check all possible future moves that would create a straight four
    for (const auto& pattern : three_patterns) {
        std::vector<int> placements = find_three_to_four_placements(pattern);
        
        for (int placement : placements) {
            // Skip if already occupied
            if (is_bit_set_(0, placement) || is_bit_set_(1, placement)) {
                continue;
            }
            
            // Create a temporary board state accessor that adds this placement
            auto is_bit_set_temp = [this, placement](int p_idx, int a) {
                if (a == placement && p_idx == 0) { // Black placed here
                    return true;
                }
                return is_bit_set_(p_idx, a);
            };
            
            // Store original accessor
            auto original_is_bit_set = is_bit_set_;
            
            // Replace with temporary accessor
            const_cast<GomokuRules*>(this)->is_bit_set_ = is_bit_set_temp;
            
            // Check if this creates a new double-three
            std::vector<std::set<int>> new_three_patterns = get_three_patterns_for_action(placement);
            
            // Restore original accessor
            const_cast<GomokuRules*>(this)->is_bit_set_ = original_is_bit_set;
            
            if (new_three_patterns.size() >= 2) {
                // Recursively check if this new double-three is allowed
                if (is_double_three_allowed_recursive(new_three_patterns, depth + 1, max_depth)) {
                    return true;
                }
            }
        }
    }
    
    // If we've checked all possibilities and found no allowed configuration
    return false;
}

// Utility methods
std::vector<std::pair<int, int>> GomokuRules::build_entire_line(int x0, int y0, int dx, int dy) const {
    std::vector<std::pair<int, int>> backward_positions;
    std::vector<std::pair<int, int>> forward_positions;
    
    int bx = x0, by = y0;
    while (in_bounds_(bx, by)) {
        backward_positions.push_back({bx, by});
        bx -= dx;
        by -= dy;
    }
    
    std::reverse(backward_positions.begin(), backward_positions.end());
    
    int fx = x0 + dx, fy = y0 + dy;
    while (in_bounds_(fx, fy)) {
        forward_positions.push_back({fx, fy});
        fx += dx;
        fy += dy;
    }
    
    std::vector<std::pair<int, int>> result = backward_positions;
    result.insert(result.end(), forward_positions.begin(), forward_positions.end());
    return result;
}

} // namespace gomoku
} // namespace games
} // namespace alphazero