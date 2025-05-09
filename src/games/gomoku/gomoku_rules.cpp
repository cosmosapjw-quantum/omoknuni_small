// File: gomoku_rules.cpp
#include "games/gomoku/gomoku_rules.h"
#include <stdexcept> // For std::runtime_error
#include <algorithm> // For std::sort, std::set_intersection, etc.
#include <map>
#include <numeric> // For std::gcd
#include <set>     // For std::set

namespace alphazero {
namespace games {
namespace gomoku {

GomokuRules::GomokuRules(int board_size) : board_size_(board_size) {
    if (board_size <= 0) {
        throw std::invalid_argument("Board size must be positive.");
    }
}

void GomokuRules::setBoardAccessor(
    std::function<bool(int, int)> is_bit_set,
    std::function<bool(int)> is_any_bit_set,
    std::function<int(int, int)> coords_to_action,
    std::function<std::pair<int, int>(int)> action_to_coords,
    std::function<bool(int, int)> in_bounds) {
    is_bit_set_ = is_bit_set;
    is_any_bit_set_ = is_any_bit_set;
    coords_to_action_ = coords_to_action;
    action_to_coords_ = action_to_coords;
    in_bounds_ = in_bounds;
}

bool GomokuRules::is_empty_spot(int r, int c, const std::function<bool(int, int)>& board_accessor) const {
    if (!this->in_bounds_(r, c)) return false;
    int action = this->coords_to_action_(r, c);
    return !board_accessor(0, action) && !board_accessor(1, action);
}

int GomokuRules::get_line_length_at_action(int action, int player_idx, const std::function<bool(int,int)>& board_accessor, int dr, int dc) const {
    auto [r_start, c_start] = this->action_to_coords_(action);
    if (!board_accessor(player_idx, action)) {
        return 0;
    }

    int count = 1; // Start with the stone at action

    // Count in positive direction
    for (int i = 1; i < this->board_size_; ++i) {
        int nr = r_start + i * dr;
        int nc = c_start + i * dc;
        if (this->in_bounds_(nr, nc) && board_accessor(player_idx, this->coords_to_action_(nr, nc))) {
            count++;
        } else {
            break;
        }
    }
    
    // Count in negative direction
    for (int i = 1; i < this->board_size_; ++i) {
        int nr = r_start - i * dr;
        int nc = c_start - i * dc;
        if (this->in_bounds_(nr, nc) && board_accessor(player_idx, this->coords_to_action_(nr, nc))) {
            count++;
        } else {
            break;
        }
    }
    
    return count;
}

bool GomokuRules::is_five_in_a_row(int action, int player, bool allow_overline) const {
    if (!is_bit_set_ || !action_to_coords_ || !coords_to_action_ || !in_bounds_) {
        return false; // Essential accessors not set
    }

    int p_idx = player - 1;
    if (p_idx < 0 || p_idx > 1) return false;

    auto current_board_state_accessor = [this](int p_check_idx, int act_check){
        return this->is_bit_set_(p_check_idx, act_check);
    };

    const int DIRS[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

    auto check_from_point = [&](int r_check, int c_check) {
        int current_action_to_check = this->coords_to_action_(r_check, c_check);
        if (!current_board_state_accessor(p_idx, current_action_to_check)) {
            return false;
        }

        for (auto& dir : DIRS) {
            int length = this->get_line_length_at_action(current_action_to_check, p_idx, current_board_state_accessor, dir[0], dir[1]);
            if (allow_overline) {
                if (length >= 5) return true;
            } else {
                if (length == 5) return true;
            }
        }
        return false;
    };

    if (action >= 0) { // Check from the specific point
        auto [r, c] = this->action_to_coords_(action);
        return check_from_point(r, c);
    } else { // Scan entire board
        for (int r_scan = 0; r_scan < this->board_size_; ++r_scan) {
            for (int c_scan = 0; c_scan < this->board_size_; ++c_scan) {
                if (current_board_state_accessor(p_idx, this->coords_to_action_(r_scan, c_scan))) {
                    if (check_from_point(r_scan, c_scan)) return true;
                }
            }
        }
        return false;
    }
}

// --- Omok Forbidden Move Logic ---
bool GomokuRules::omok_is_six_in_a_row(int action, int player_idx, const std::function<bool(int, int)>& hypothetical_board_state) const {
    if (player_idx != 0) return false; // Only for Black

    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& dir : DIRS) {
        int length = this->get_line_length_at_action(action, player_idx, hypothetical_board_state, dir[0], dir[1]);
        if (length >= 6) return true;
    }
    return false;
}

bool GomokuRules::check_omok_open_three_in_direction(int action, int player_idx, const std::function<bool(int, int)>& board_accessor, int dr, int dc) const {
    auto [r_action, c_action] = this->action_to_coords_(action);

    for (int i = 0; i < 3; ++i) { 
        int r_start_of_3 = r_action - i * dr;
        int c_start_of_3 = c_action - i * dc;
        
        bool is_line_of_3 = true;
        for(int k=0; k<3; ++k) {
            int r_k = r_start_of_3 + k * dr;
            int c_k = c_start_of_3 + k * dc;
            if (!this->in_bounds_(r_k, c_k) || !board_accessor(player_idx, this->coords_to_action_(r_k, c_k))) {
                is_line_of_3 = false;
                break;
            }
        }

        if (is_line_of_3) {
            int r_before = r_start_of_3 - dr;
            int c_before = c_start_of_3 - dc;
            int r_after = r_start_of_3 + 3 * dr; 
            int c_after = c_start_of_3 + 3 * dc;

            if (this->is_empty_spot(r_before, c_before, board_accessor) && 
                this->is_empty_spot(r_after, c_after, board_accessor)) {
                return true; 
            }
        }
    }
    return false;
}

bool GomokuRules::omok_makes_double_three(int action, int player_idx, const std::function<bool(int, int)>& board_accessor) const {
    if (player_idx != 0) return false; // Only for Black

    int open_three_directions_count = 0;
    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}}; 
    
    for (auto& dir : DIRS) {
        if (this->check_omok_open_three_in_direction(action, player_idx, board_accessor, dir[0], dir[1])) {
            open_three_directions_count++;
        }
    }
    
    return open_three_directions_count >= 2;
}

bool GomokuRules::is_black_omok_forbidden(int action) const {
    if (!is_bit_set_ || !is_any_bit_set_ || !coords_to_action_ || !action_to_coords_ || !in_bounds_) {
         throw std::runtime_error("GomokuRules not properly initialized with board accessors.");
    }

    int p_idx_moving = 0; // Black
    
    auto is_bit_set_hypothetical = [this, action, p_idx_moving](int p_idx_check, int a_check) {
        if (a_check == action && p_idx_check == p_idx_moving) return true; 
        return this->is_bit_set_(p_idx_check, a_check); 
    };

    if (this->omok_is_six_in_a_row(action, p_idx_moving, is_bit_set_hypothetical)) {
        return true;
    }

    if (this->omok_makes_double_three(action, p_idx_moving, is_bit_set_hypothetical)) {
        return true;
    }

    return false;
}

// --- Renju Forbidden Move Logic ---
bool GomokuRules::is_black_renju_forbidden(int action) const {
    if (!is_bit_set_ || !is_any_bit_set_ || !coords_to_action_ || !action_to_coords_ || !in_bounds_) {
         throw std::runtime_error("GomokuRules not properly initialized with board accessors for Renju.");
    }

    int p_idx_moving = 0; // Black

    auto is_bit_set_hypothetical = [this, action, p_idx_moving](int p_idx_check, int a_check) {
        if (a_check == action && p_idx_check == p_idx_moving) return true;
        return this->is_bit_set_(p_idx_check, a_check);
    };

    bool forms_exact_five = false;
    const int DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
    for (auto& dir : DIRS) {
        int length = this->get_line_length_at_action(action, p_idx_moving, is_bit_set_hypothetical, dir[0], dir[1]);
        if (length == 5) {
            forms_exact_five = true;
        }
    }
    
    bool forms_overline_simultaneously = false;
     for (auto& dir : DIRS) {
        int length = this->get_line_length_at_action(action, p_idx_moving, is_bit_set_hypothetical, dir[0], dir[1]);
        if (length > 5) {
            forms_overline_simultaneously = true;
            break;
        }
    }

    if (forms_exact_five && !forms_overline_simultaneously) {
        return false;
    }

    if (this->renju_is_overline(action, p_idx_moving, is_bit_set_hypothetical)) {
        return true;
    }
    
    if (this->renju_makes_double_four(action, p_idx_moving, is_bit_set_hypothetical)) {
        return true;
    }
    
    // Call the public interface for double-three check
    const std::function<bool(int, int)> board_state_func = is_bit_set_hypothetical; // Assign lambda to std::function variable
    if (this->is_black_renju_d3_forbidden(action, p_idx_moving, board_state_func)) { // Use the wrapper function
        return true;
    }

    return false;
}

bool GomokuRules::renju_is_overline(int action, int player_idx, const std::function<bool(int, int)>& board_accessor) const {
    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& dir : DIRS) {
        int length = this->get_line_length_at_action(action, player_idx, board_accessor, dir[0], dir[1]);
        if (length > 5) return true; 
    }
    return false;
}

int GomokuRules::count_renju_fours_at_action(int action, int player_idx, const std::function<bool(int, int)>& board_accessor) const {
    auto [r_action, c_action] = this->action_to_coords_(action);
    int four_count = 0;
    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    
    for (auto& dir : DIRS) {
        int dr = dir[0];
        int dc = dir[1];
        bool found_four_in_this_direction = false;

        // 1. Check for Straight Fours (XXXX pattern)
        for (int offset = 0; offset < 4; ++offset) { 
            int r_start_of_line4 = r_action - offset * dr;
            int c_start_of_line4 = c_action - offset * dc;
            
            std::set<int> current_line_stones;
            bool is_solid_line_of_4 = true;
            for (int i = 0; i < 4; ++i) {
                int r_k = r_start_of_line4 + i * dr;
                int c_k = c_start_of_line4 + i * dc;
                if (!this->in_bounds_(r_k, c_k) || !board_accessor(player_idx, this->coords_to_action_(r_k, c_k))) {
                    is_solid_line_of_4 = false;
                    break;
                }
                current_line_stones.insert(this->coords_to_action_(r_k, c_k));
            }
            
            if (is_solid_line_of_4 && current_line_stones.count(action) && current_line_stones.size() == 4) {
                int r_before = r_start_of_line4 - dr;
                int c_before = c_start_of_line4 - dc;
                int r_after = r_start_of_line4 + 4 * dr;
                int c_after = c_start_of_line4 + 4 * dc;
                
                bool open_before = this->is_empty_spot(r_before, c_before, board_accessor);
                bool open_after = this->is_empty_spot(r_after, c_after, board_accessor);
                
                if (open_before || open_after) { 
                    // Check if completing to five results in exactly 5 (not overline)
                    bool makes_exactly_five = false;
                    if (open_before) {
                        auto board_if_filled_before = [&](int p, int a){ return (p == player_idx && a == coords_to_action_(r_before, c_before)) || board_accessor(p,a); };
                        if (get_line_length_at_action(coords_to_action_(r_before, c_before), player_idx, board_if_filled_before, dr, dc) == 5) makes_exactly_five = true;
                    }
                    if (!makes_exactly_five && open_after) {
                        auto board_if_filled_after = [&](int p, int a){ return (p == player_idx && a == coords_to_action_(r_after, c_after)) || board_accessor(p,a); };
                        if (get_line_length_at_action(coords_to_action_(r_after, c_after), player_idx, board_if_filled_after, dr, dc) == 5) makes_exactly_five = true;
                    }
                    if(makes_exactly_five){
                        four_count++;
                        found_four_in_this_direction = true;
                        goto next_direction_label; 
                    }
                }
            }
        }
        if (found_four_in_this_direction) goto next_direction_label;

        // 2. Check for Broken Fours (XXX_X, XX_XX, X_XXX patterns within a 5-cell segment)
        for (int offset_action_in_5 = 0; offset_action_in_5 < 5; ++offset_action_in_5) { 
            int r_start_5seg = r_action - offset_action_in_5 * dr;
            int c_start_5seg = c_action - offset_action_in_5 * dc;

            int player_stones_in_seg = 0;
            int empty_spots_in_seg = 0;
            int empty_spot_action_val = -1;
            bool action_in_segment = false;
            bool segment_valid_for_broken_four = true;

            for (int i = 0; i < 5; ++i) {
                int r_k = r_start_5seg + i * dr;
                int c_k = c_start_5seg + i * dc;
                if (!this->in_bounds_(r_k, c_k)) {
                    segment_valid_for_broken_four = false; break;
                }
                int current_spot = this->coords_to_action_(r_k, c_k);
                if (current_spot == action) action_in_segment = true;

                if (board_accessor(player_idx, current_spot)) {
                    player_stones_in_seg++;
                } else if (this->is_empty_spot(r_k, c_k, board_accessor)) {
                    empty_spots_in_seg++;
                    empty_spot_action_val = current_spot;
                } else { // Opponent stone in segment, cannot be this type of broken four
                    segment_valid_for_broken_four = false; break;
                }
            }

            if (segment_valid_for_broken_four && action_in_segment && player_stones_in_seg == 4 && empty_spots_in_seg == 1) {
                // This is a XXXX_ pattern (4 player stones, 1 empty in 5-cell window)
                // Now, check if filling that empty_spot_action_val makes exactly 5 stones in a line.
                auto board_if_gap_filled = [&](int p, int a) {
                    return (p == player_idx && a == empty_spot_action_val) || board_accessor(p,a);
                };
                
                if (get_line_length_at_action(empty_spot_action_val, player_idx, board_if_gap_filled, dr, dc) == 5) {
                    four_count++;
                    found_four_in_this_direction = true; // Mark to skip further checks for this dir
                    goto next_direction_label;
                }
            }
        }
        if (found_four_in_this_direction) goto next_direction_label;

        // 3. Check for Potential Fours (action stone + one existing stone with a gap can become a straight four)
        // Scan in both directions from action stone looking for player stones with one gap
        for (int distance = 2; distance <= 4; distance++) {
            // Forward direction
            int r_forward = r_action + distance * dr;
            int c_forward = c_action + distance * dc;
            if (this->in_bounds_(r_forward, c_forward) && board_accessor(player_idx, this->coords_to_action_(r_forward, c_forward))) {
                // Check if the gap in between is empty
                int r_gap = r_action + (distance-1) * dr;
                int c_gap = c_action + (distance-1) * dc;
                if (this->is_empty_spot(r_gap, c_gap, board_accessor)) {
                    // Check if filling that gap would create a potential straight four
                    auto gap_filled_board = [&](int p, int a) {
                        return (p == player_idx && a == this->coords_to_action_(r_gap, c_gap)) || board_accessor(p, a);
                    };
                    
                    // Test if filling the gap creates a 3-in-a-row that has potential to become a four
                    int line_length = get_line_length_at_action(this->coords_to_action_(r_gap, c_gap), player_idx, gap_filled_board, dr, dc);
                    if (line_length >= 3) {
                        // Now check if this 3-in-a-row has room to become a four
                        int r_before = r_action - dr;
                        int c_before = c_action - dc;
                        int r_after = r_forward + dr;
                        int c_after = c_forward + dc;
                        
                        bool can_extend_before = this->in_bounds_(r_before, c_before) && this->is_empty_spot(r_before, c_before, board_accessor);
                        bool can_extend_after = this->in_bounds_(r_after, c_after) && this->is_empty_spot(r_after, c_after, board_accessor);
                        
                        if (can_extend_before || can_extend_after) {
                            four_count++;
                            goto next_direction_label;
                        }
                    }
                }
            }
            
            // Backward direction
            int r_backward = r_action - distance * dr;
            int c_backward = c_action - distance * dc;
            if (this->in_bounds_(r_backward, c_backward) && board_accessor(player_idx, this->coords_to_action_(r_backward, c_backward))) {
                // Check if the gap in between is empty
                int r_gap = r_action - (distance-1) * dr;
                int c_gap = c_action - (distance-1) * dc;
                if (this->is_empty_spot(r_gap, c_gap, board_accessor)) {
                    // Check if filling that gap would create a potential straight four
                    auto gap_filled_board = [&](int p, int a) {
                        return (p == player_idx && a == this->coords_to_action_(r_gap, c_gap)) || board_accessor(p, a);
                    };
                    
                    // Test if filling the gap creates a 3-in-a-row that has potential to become a four
                    int line_length = get_line_length_at_action(this->coords_to_action_(r_gap, c_gap), player_idx, gap_filled_board, dr, dc);
                    if (line_length >= 3) {
                        // Now check if this 3-in-a-row has room to become a four
                        int r_before = r_backward - dr;
                        int c_before = c_backward - dc;
                        int r_after = r_action + dr;
                        int c_after = c_action + dc;
                        
                        bool can_extend_before = this->in_bounds_(r_before, c_before) && this->is_empty_spot(r_before, c_before, board_accessor);
                        bool can_extend_after = this->in_bounds_(r_after, c_after) && this->is_empty_spot(r_after, c_after, board_accessor);
                        
                        if (can_extend_before || can_extend_after) {
                            four_count++;
                            goto next_direction_label;
                        }
                    }
                }
            }
        }
        
        next_direction_label:;
    }
    return four_count;
}

bool GomokuRules::renju_makes_double_four(int action, int player_idx, const std::function<bool(int,int)>& board_accessor) const {
    return this->count_renju_fours_at_action(action, player_idx, board_accessor) >= 2;
}

bool GomokuRules::is_renju_double_three_forbidden(
    int action, 
    int player_idx, 
    const std::function<bool(int,int)>& board_accessor_after_action
) const {
    // Need to use std::function to allow recursive use of the lambda by pointer.
    std::function<bool(int, const std::function<bool(int,int)>&)> s4_checker_lambda_obj;
    s4_checker_lambda_obj = 
        [this, player_idx, &s4_checker_lambda_obj](int s4_action, const std::function<bool(int,int)>& board_state_for_s4_lambda) -> bool {
        if (this->renju_is_overline(s4_action, player_idx, board_state_for_s4_lambda)) return true;
        if (this->renju_makes_double_four(s4_action, player_idx, board_state_for_s4_lambda)) return true;
        // Pass pointer to the std::function object for recursive calls
        return this->check_renju_double_three_recursive(s4_action, player_idx, board_state_for_s4_lambda, true, &s4_checker_lambda_obj);
    };

    return this->check_renju_double_three_recursive(action, player_idx, board_accessor_after_action, false, &s4_checker_lambda_obj);
}

bool GomokuRules::check_renju_double_three_recursive(
    int action, 
    int player_idx, 
    const std::function<bool(int,int)>& board_accessor_after_action,
    bool is_recursive_call,
    const std::function<bool(int, const std::function<bool(int,int)>&)>* s4_forbidden_checker
) const {

    std::vector<RenjuPatternInfo> actual_renju_threes;
    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};

    auto check_s4_forbidden_primary = 
        [this, player_idx, s4_forbidden_checker, is_recursive_call](int s4_action, const std::function<bool(int,int)>& board_state_for_s4) -> bool {
        if (this->renju_is_overline(s4_action, player_idx, board_state_for_s4)) return true;
        if (this->renju_makes_double_four(s4_action, player_idx, board_state_for_s4)) return true;
        
        if (s4_forbidden_checker && *s4_forbidden_checker) {
             return (*s4_forbidden_checker)(s4_action, board_state_for_s4);
        } else {
             // Fallback path if checker is null (should not happen via public interface)
             return this->check_renju_double_three_recursive(s4_action, player_idx, board_state_for_s4, true, s4_forbidden_checker);
        }
    };

    for (auto& dir_arr : DIRS) {
        auto [r_act, c_act] = this->action_to_coords_(action);

        for (int offset_in_3 = 0; offset_in_3 < 3; ++offset_in_3) { 
            int r_start_of_3 = r_act - offset_in_3 * dir_arr[0];
            int c_start_of_3 = c_act - offset_in_3 * dir_arr[1];
            
            std::set<int> current_three_set;
            bool possible_three = true;
            for (int k=0; k<3; ++k) {
                int r_k = r_start_of_3 + k * dir_arr[0];
                int c_k = c_start_of_3 + k * dir_arr[1];
                if (!this->in_bounds_(r_k,c_k)) {possible_three = false; break;}
                int stone_action_k = this->coords_to_action_(r_k,c_k);
                if (!board_accessor_after_action(player_idx, stone_action_k)) {possible_three=false; break;}
                current_three_set.insert(stone_action_k);
            }

            if (possible_three && current_three_set.count(action)) {
                if (this->is_renju_three_definition(current_three_set, action, player_idx, 
                                              board_accessor_after_action,
                                              check_s4_forbidden_primary 
                                              )) {
                    GomokuRules::RenjuPatternInfo p_info;
                    p_info.stones = current_three_set;
                    p_info.type = GomokuRules::RenjuLineType::THREE; 
                    p_info.dir = {dir_arr[0], dir_arr[1]};
                    
                    bool found = false;
                    for(const auto& existing_three : actual_renju_threes) {
                        if (existing_three.stones == p_info.stones && existing_three.dir == p_info.dir) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                         actual_renju_threes.push_back(p_info);
                    }
                }
            }
        }
    }

    if (actual_renju_threes.size() < 2) {
        return false; 
    }

    std::function<bool(int, int, const std::function<bool(int,int)>&)> s4_d3_checker_for_exception_logic;
    s4_d3_checker_for_exception_logic = 
        [this, player_idx, &s4_forbidden_checker] 
        (int s4_action_inner, int p_idx_inner_ignored, const std::function<bool(int,int)>& board_s4_inner) -> bool {
        if (s4_forbidden_checker && *s4_forbidden_checker) {
            // s4_forbidden_checker is already a pointer to std::function, so we can define nested_s4_checker similarly if needed
            // or just pass s4_forbidden_checker directly if the recursive structure is through it.
            // The nested_s4_checker here is to ensure the context for *its* S4s uses the original checker.
            // This lambda is for the S4 of the S4.
            std::function<bool(int, const std::function<bool(int,int)>&)> nested_s4_checker_obj;
            nested_s4_checker_obj = 
                [this, player_idx, s4_forbidden_checker_outer_ptr = s4_forbidden_checker] // Capture outer checker by value (it's a ptr)
                (int nested_s4_act, const std::function<bool(int,int)>& nested_board_lambda) -> bool {
                 if (s4_forbidden_checker_outer_ptr && *s4_forbidden_checker_outer_ptr) { 
                    return (*s4_forbidden_checker_outer_ptr)(nested_s4_act, nested_board_lambda);
                 }
                 // Fallback if the outer checker was somehow null (shouldn't happen with proper setup)
                 if (this->renju_is_overline(nested_s4_act, player_idx, nested_board_lambda)) return true;
                 if (this->renju_makes_double_four(nested_s4_act, player_idx, nested_board_lambda)) return true;
                 return false; 
            };
            return this->check_renju_double_three_recursive(s4_action_inner, player_idx, board_s4_inner, true, &nested_s4_checker_obj);
        } else {
            // Fallback for robustness if s4_forbidden_checker was null.
            std::function<bool(int, const std::function<bool(int,int)>&)> fallback_s4_checker_obj;
            fallback_s4_checker_obj = 
                [this, player_idx](int s4_act, const std::function<bool(int,int)>& board_lambda) -> bool {
                if (this->renju_is_overline(s4_act, player_idx, board_lambda)) return true;
                if (this->renju_makes_double_four(s4_act, player_idx, board_lambda)) return true;
                return false; // No recursive D3 check for S4 in this fallback path
            };
            return this->check_renju_double_three_recursive(s4_action_inner, player_idx, board_s4_inner, true, &fallback_s4_checker_obj);
        }
    };

    bool exception_applies = this->check_renju_double_three_exception_recursive(
        action, actual_renju_threes, player_idx, board_accessor_after_action, 0, 
        [this](int s4_action, int p_idx, const std::function<bool(int,int)>& board_s4){ return this->renju_is_overline(s4_action, p_idx, board_s4); },
        [this](int s4_action, int p_idx, const std::function<bool(int,int)>& board_s4){ return this->renju_makes_double_four(s4_action, p_idx, board_s4); },
        s4_d3_checker_for_exception_logic
    );

    if (exception_applies) { 
        return false; 
    }
    
    return true;
}

bool GomokuRules::is_renju_three_definition(const std::set<int>& three_candidate_stones, int action_that_formed_it, 
                                          int player_idx, const std::function<bool(int,int)>& board_accessor_after_action,
                                          const std::function<bool(int /*s4_action*/, const std::function<bool(int,int)>& /*board_after_s4*/)>& is_s4_forbidden_func
                                          ) const {
    if (three_candidate_stones.size() != 3) return false;
 
    for (int s : three_candidate_stones) {
        if (!board_accessor_after_action(player_idx, s)) return false;
    }

    int dr, dc, min_r, min_c, max_r, max_c;
    if (!this->get_line_direction_and_bounds(three_candidate_stones, dr, dc, min_r, min_c, max_r, max_c)) {
        return false;
    }
    
    if (dr == 0 && dc == 0 && three_candidate_stones.size() > 1) {
        return false; 
    }

    std::vector<int> extension_spots = this->get_empty_extensions_of_line(three_candidate_stones, player_idx, 
                                                                 board_accessor_after_action, dr, dc);

    for (int spot_s4 : extension_spots) {
        auto board_after_s4 = [player_idx, spot_s4, &board_accessor_after_action](int p_check, int a_check) {
            if (a_check == spot_s4 && p_check == player_idx) return true;
            return board_accessor_after_action(p_check, a_check);
        };
        
        std::set<int> four_stones = three_candidate_stones;
        four_stones.insert(spot_s4);

        if (this->is_renju_straight_four_from_stones(four_stones, player_idx, board_after_s4)) {
                 if (is_s4_forbidden_func(spot_s4, board_after_s4)) {
                    continue; 
                 }
            
            bool s4_makes_five = false;
            const int DIRS_S4[4][2] = {{1,0},{0,1},{1,1},{-1,1}}; // Corrected {1,-1} to {-1,1} or ensure distinct
            // Actually, the standard DIRS {{1,0},{0,1},{1,1},{1,-1}} is fine. Let's revert that micro-change.
            const int S4_DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
            for (auto& dir_s4 : S4_DIRS) {
                if (this->get_line_length_at_action(spot_s4, player_idx, board_after_s4, dir_s4[0], dir_s4[1]) >= 5) {
                    s4_makes_five = true;
                    break;
                }
            }
            if (s4_makes_five) {
                continue; 
            }

            return true;
        }
    }
    
    return false;
}

bool GomokuRules::is_renju_straight_four_from_stones(const std::set<int>& four_stones_set, int player_idx, 
                                                  const std::function<bool(int,int)>& board_accessor) const {
    if (four_stones_set.size() != 4) return false;
    
    for (int s : four_stones_set) {
        if (!board_accessor(player_idx, s)) return false;
    }

    int dr, dc, min_r, min_c, max_r, max_c;
    if (!this->get_line_direction_and_bounds(four_stones_set, dr, dc, min_r, min_c, max_r, max_c)) {
        return false;
    }
    
    if (dr == 0 && dc == 0 && four_stones_set.size() > 1) {
        return false; 
    }

    std::vector<std::pair<int,int>> coords;
    for (int s : four_stones_set) {
        coords.push_back(this->action_to_coords_(s));
    }
    std::sort(coords.begin(), coords.end());

    for (size_t i = 0; i < coords.size() - 1; ++i) {
        if (coords[i+1].first != coords[i].first + dr || 
            coords[i+1].second != coords[i].second + dc) {
            return false; 
        }
    }

    int r_before = coords.front().first - dr;
    int c_before = coords.front().second - dc;
    int r_after = coords.back().first + dr;
    int c_after = coords.back().second + dc;

    bool open_before = this->is_empty_spot(r_before, c_before, board_accessor);
    bool open_after = this->is_empty_spot(r_after, c_after, board_accessor);
    
    return open_before && open_after;
}

bool GomokuRules::check_renju_double_three_exception_recursive(
    int original_action, 
    const std::vector<GomokuRules::RenjuPatternInfo>& threes_involved,
    int player_idx,
    const std::function<bool(int, int)>& board_after_original_action,
    int depth,
    const std::function<bool(int, int, const std::function<bool(int,int)>&)>& is_overline_func,
    const std::function<bool(int, int, const std::function<bool(int,int)>&)>& makes_double_four_func,
    const std::function<bool(int, int, const std::function<bool(int,int)>&)>& is_double_three_forbidden_external_func) const {
    if (depth > 3) return false; 

    int resolvable_to_valid_straight_four_count = 0;

    for (const auto& current_three_pattern_info : threes_involved) {

        std::vector<int> extension_spots = this->get_empty_extensions_of_line(
            current_three_pattern_info.stones, player_idx, board_after_original_action, 
            current_three_pattern_info.dir.first, current_three_pattern_info.dir.second);

        bool this_three_can_be_resolved_to_valid_sf = false;
        
        for (int spot_s4 : extension_spots) { 
            auto board_after_s4 = [player_idx, spot_s4, &board_after_original_action](int p_check, int a_check) {
                if (a_check == spot_s4 && p_check == player_idx) return true;
                return board_after_original_action(p_check, a_check);
            };
            
            std::set<int> four_candidate_stones = current_three_pattern_info.stones;
            four_candidate_stones.insert(spot_s4);

            if (!this->is_renju_straight_four_from_stones(four_candidate_stones, player_idx, board_after_s4)) {
                continue;
            }

            bool s4_is_overline = is_overline_func(spot_s4, player_idx, board_after_s4);
            bool s4_is_df = makes_double_four_func(spot_s4, player_idx, board_after_s4);
            bool s4_is_forbidden_d3 = false;
            if (is_double_three_forbidden_external_func) { 
                 s4_is_forbidden_d3 = is_double_three_forbidden_external_func(spot_s4, player_idx, board_after_s4);
            }

            if (s4_is_overline || s4_is_df || s4_is_forbidden_d3) {
                continue; 
            }
            
            bool s4_makes_five = false;
            const int S4_DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
             for (auto& dir_s4 : S4_DIRS) {
                if (this->get_line_length_at_action(spot_s4, player_idx, board_after_s4, dir_s4[0], dir_s4[1]) >= 5) {
                    s4_makes_five = true;
                    break;
                }
            }
            if (s4_makes_five) {
                continue;
            }

            this_three_can_be_resolved_to_valid_sf = true;
            break; 
        }

        if (this_three_can_be_resolved_to_valid_sf) {
            resolvable_to_valid_straight_four_count++;
        }
    }
    
    return resolvable_to_valid_straight_four_count <= 1;
}

std::vector<int> GomokuRules::get_empty_extensions_of_line(const std::set<int>& stones, int player_idx, 
                                                        const std::function<bool(int,int)>& board_accessor, 
                                                        int dr, int dc) const {
    std::vector<int> spots;
    
    if (stones.empty() || (dr == 0 && dc == 0 && stones.size() > 1)) {
        return spots;
    }

    std::vector<std::pair<int,int>> stone_coords;
    for (int s : stones) {
        stone_coords.push_back(this->action_to_coords_(s));
    }
    std::sort(stone_coords.begin(), stone_coords.end());

    if (stone_coords.empty()) return spots;

    auto [first_r, first_c] = stone_coords.front();
    auto [last_r, last_c] = stone_coords.back();
    
    if (this->is_empty_spot(first_r - dr, first_c - dc, board_accessor)) {
        spots.push_back(this->coords_to_action_(first_r - dr, first_c - dc));
    }
    
    if (this->is_empty_spot(last_r + dr, last_c + dc, board_accessor)) {
        spots.push_back(this->coords_to_action_(last_r + dr, last_c + dc));
    }

    if (stones.size() >= 2) {
        for (int i = 0; ; ++i) {
            int cur_r = first_r + i * dr;
            int cur_c = first_c + i * dc;
            
            bool past_last = false;
            if ((dr > 0 && cur_r > last_r) || (dr < 0 && cur_r < last_r) ||
                (dc > 0 && cur_c > last_c) || (dc < 0 && cur_c < last_c)) {
                if (!(cur_r == last_r && cur_c == last_c)) { // ensure we don't mark last stone itself as past if it's the start
                    past_last = true;
                }
            }
            if (cur_r == last_r && cur_c == last_c && i > 0 && (std::abs(dr) > 0 || std::abs(dc) >0 )) { // if we landed on the last stone (and it's not the first one due to i>0)
                 //This condition means we have processed all segments up to the last stone.
                 // If dr=dc=0, this loop is problematic. Added check for dr/dc > 0 for this specific break.
                 break; 
            }
            if (past_last || !this->in_bounds_(cur_r, cur_c)) break;

            int current_spot_action = this->coords_to_action_(cur_r, cur_c);
            
            if (stones.find(current_spot_action) == stones.end() && 
                this->is_empty_spot(cur_r, cur_c, board_accessor)) {
                spots.push_back(current_spot_action);
            }
            
            if (i > this->board_size_ + 2) break; 
             if (dr == 0 && dc == 0) break; // Avoid infinite loop for single stone case if it reaches here.
        }
    }

    std::sort(spots.begin(), spots.end());
    spots.erase(std::unique(spots.begin(), spots.end()), spots.end());
    
    return spots;
}

bool GomokuRules::get_line_direction_and_bounds(const std::set<int>& stones,
                                               int& dr, int& dc,
                                               int& min_r_out, int& min_c_out,
                                               int& max_r_out, int& max_c_out) const {
    if (stones.empty()) return false;
    
    std::vector<std::pair<int,int>> coords;
    for (int s : stones) {
        coords.push_back(this->action_to_coords_(s));
    }
    std::sort(coords.begin(), coords.end());

    min_r_out = coords.front().first; 
    min_c_out = coords.front().second;
    max_r_out = coords.back().first;  
    max_c_out = coords.back().second;

    if (stones.size() == 1) { 
        dr = 0; dc = 0; 
        return true; 
    }
    
    auto p1 = coords[0];
    auto p2 = p1;
    bool found_p2 = false;
    
    for (size_t i = 1; i < coords.size(); ++i) {
        if (coords[i] != p1) {
            p2 = coords[i];
            found_p2 = true;
            break;
        }
    }
    
    if (!found_p2) { 
        dr = 0; dc = 0;
        return true;
    }

    dr = p2.first - p1.first;
    dc = p2.second - p1.second;

    int common_divisor = std::gcd(std::abs(dr), std::abs(dc));
    if (common_divisor > 0) {
        dr /= common_divisor;
        dc /= common_divisor;
    } else {
        if (dr != 0) dr = dr / std::abs(dr);
        if (dc != 0) dc = dc / std::abs(dc);
    }

    for (const auto& coord : coords) {
        if (dr == 0) { 
            if (coord.first != p1.first) return false;
        } else if (dc == 0) { 
            if (coord.second != p1.second) return false;
        } else { 
            if ((long long)(coord.first - p1.first) * dc != (long long)(coord.second - p1.second) * dr) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace gomoku
} // namespace games
} // namespace alphazero