// src/games/go/go_rules.cpp
#include "games/go/go_rules.h"
#include <queue>

namespace alphazero {
namespace games {
namespace go {

GoRules::GoRules(int board_size, bool chinese_rules, bool enforce_superko)
    : board_size_(board_size),
      chinese_rules_(chinese_rules),
      enforce_superko_(enforce_superko) {
    
    // Default implementations (will be replaced by setBoardAccessor)
    get_stone_ = [](int) { return 0; };
    is_in_bounds_ = [](int) { return false; };
    get_adjacent_positions_ = [](int) { return std::vector<int>(); };
}

void GoRules::setBoardAccessor(
    std::function<int(int)> get_stone,
    std::function<bool(int)> is_in_bounds,
    std::function<std::vector<int>(int)> get_adjacent_positions) {
    
    get_stone_ = get_stone;
    is_in_bounds_ = is_in_bounds;
    get_adjacent_positions_ = get_adjacent_positions;
    
    // Invalidate caches since the board accessor changed
    group_cache_.clear();
    group_cache_dirty_ = true;
}

bool GoRules::isSuicidalMove(int action, int player) const {
    if (!is_in_bounds_(action) || get_stone_(action) != 0) {
        return true;  // Invalid move
    }
    
    // Create a temporary board state for simulation
    std::vector<int> temp_board(board_size_ * board_size_, 0);
    
    // Copy current board state
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (is_in_bounds_(pos)) {
            temp_board[pos] = get_stone_(pos);
        }
    }
    
    // Place the stone
    temp_board[action] = player;
    
    // Define temporary accessor function for the simulated board
    auto temp_get_stone = [&temp_board](int pos) { return temp_board[pos]; };
    
    // Check if any opponent group would be captured
    int opponent = (player == 1) ? 2 : 1;
    bool captures_opponent = false;
    
    // Check each adjacent position for opponent groups that might be captured
    for (int adj_pos : get_adjacent_positions_(action)) {
        if (temp_get_stone(adj_pos) == opponent) {
            // Check if this group would have liberties after the move
            std::unordered_set<int> group_stones;
            std::queue<int> queue;
            
            // Find connected opponent stones
            queue.push(adj_pos);
            group_stones.insert(adj_pos);
            std::vector<bool> visited(board_size_ * board_size_, false);
            visited[adj_pos] = true;
            
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                for (int next_pos : get_adjacent_positions_(current)) {
                    if (temp_get_stone(next_pos) == opponent && !visited[next_pos]) {
                        queue.push(next_pos);
                        visited[next_pos] = true;
                        group_stones.insert(next_pos);
                    }
                }
            }
            
            // Check if this group has any liberties
            bool has_liberty = false;
            for (int stone : group_stones) {
                for (int lib_pos : get_adjacent_positions_(stone)) {
                    if (temp_get_stone(lib_pos) == 0) {
                        has_liberty = true;
                        break;
                    }
                }
                if (has_liberty) break;
            }
            
            if (!has_liberty) {
                captures_opponent = true;
                break;
            }
        }
    }
    
    // If capturing opponent groups, the move is not suicidal
    if (captures_opponent) {
        return false;
    }
    
    // Check if the placed stone's group would have liberties
    std::unordered_set<int> own_stones;
    std::queue<int> queue;
    
    // Find connected stones of the same color
    queue.push(action);
    own_stones.insert(action);
    std::vector<bool> visited(board_size_ * board_size_, false);
    visited[action] = true;
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        for (int next_pos : get_adjacent_positions_(current)) {
            if (temp_get_stone(next_pos) == player && !visited[next_pos]) {
                queue.push(next_pos);
                visited[next_pos] = true;
                own_stones.insert(next_pos);
            }
        }
    }
    
    // Check if this group has any liberties
    for (int stone : own_stones) {
        for (int lib_pos : get_adjacent_positions_(stone)) {
            if (temp_get_stone(lib_pos) == 0) {
                return false;  // Has liberty, not suicide
            }
        }
    }
    
    // No liberties, move is suicidal
    return true;
}

bool GoRules::isKoViolation(int action, int ko_point) const {
    return action == ko_point;
}

std::vector<StoneGroup> GoRules::findGroups(int player) const {
    // Use cached groups if available
    if (!group_cache_dirty_ && group_cache_.find(player) != group_cache_.end()) {
        return group_cache_[player];
    }
    
    std::vector<StoneGroup> groups;
    std::vector<bool> visited(board_size_ * board_size_, false);
    
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos) || get_stone_(pos) != player || visited[pos]) {
            continue;
        }
        
        // Found an unvisited stone of the player
        StoneGroup group;
        std::queue<int> queue;
        
        queue.push(pos);
        visited[pos] = true;
        group.stones.insert(pos);
        
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            
            for (int adj : get_adjacent_positions_(current)) {
                if (get_stone_(adj) == player && !visited[adj]) {
                    queue.push(adj);
                    visited[adj] = true;
                    group.stones.insert(adj);
                }
            }
        }
        
        // Find liberties for the group
        findLiberties(group.stones, group.liberties);
        
        groups.push_back(group);
    }
    
    // Cache the result
    group_cache_[player] = groups;
    group_cache_dirty_ = false;
    
    return groups;
}

void GoRules::findLiberties(std::unordered_set<int>& stones, std::unordered_set<int>& liberties) const {
    liberties.clear();
    
    // Check adjacent positions of all stones in the group
    for (int pos : stones) {
        for (int adj : get_adjacent_positions_(pos)) {
            // Only count empty positions as liberties
            if (is_in_bounds_(adj) && get_stone_(adj) == 0) {
                liberties.insert(adj);
            }
        }
    }
}

std::vector<int> GoRules::getTerritoryOwnership(const std::unordered_set<int>& dead_stones) const {
    std::vector<int> territory(board_size_ * board_size_, 0);
    
    // Create a temporary board with dead stones removed
    std::vector<int> temp_board(board_size_ * board_size_, 0);
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (is_in_bounds_(pos)) {
            if (dead_stones.find(pos) == dead_stones.end()) {
                temp_board[pos] = get_stone_(pos);
            } else {
                // For dead stones, add them to territory of the opponent
                int stone = get_stone_(pos);
                if (stone == 1) {
                    territory[pos] = 2;  // Dead black stone is white territory
                } else if (stone == 2) {
                    territory[pos] = 1;  // Dead white stone is black territory
                }
                temp_board[pos] = 0;  // Remove dead stone from board for territory calculation
            }
        }
    }
    
    // Define temporary accessor function
    auto temp_get_stone = [&temp_board](int pos) { return temp_board[pos]; };
    
    // Find empty regions and determine ownership
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos)) continue;
        
        if (temp_get_stone(pos) == 0 && territory[pos] == 0) {
            // Unmarked empty intersection, flood fill to find territory
            int territory_color = 0;
            floodFillTerritory(territory, pos, territory_color, dead_stones);
        }
    }
    
    // Mark stones with their respective owners (for Chinese rules)
    if (chinese_rules_) {
        for (int pos = 0; pos < board_size_ * board_size_; pos++) {
            if (is_in_bounds_(pos)) {
                int stone = temp_get_stone(pos);
                if (stone != 0) {
                    territory[pos] = stone;  // Owned by the player with a stone here
                }
            }
        }
    }
    
    return territory;
}

void GoRules::floodFillTerritory(
    std::vector<int>& territory, 
    int pos, 
    int& territory_color,
    const std::unordered_set<int>& dead_stones) const {
    
    // Start with neutral territory
    territory_color = 0;
    
    // Use queue for flood fill
    std::queue<int> queue;
    std::unordered_set<int> visited_empty;
    std::unordered_set<int> boundary_stones;
    
    queue.push(pos);
    visited_empty.insert(pos);
    
    // Create a temporary board with dead stones removed for BFS pathing
    std::vector<int> temp_board(board_size_ * board_size_, 0);
    for (int p = 0; p < board_size_ * board_size_; p++) {
        if (is_in_bounds_(p)) {
            if (dead_stones.find(p) == dead_stones.end()) {
                temp_board[p] = get_stone_(p);
            } else {
                temp_board[p] = 0; // Treat dead stone locations as empty for BFS pathing
            }
        }
    }
    auto temp_get_stone = [&temp_board](int p) { return temp_board[p]; };
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        for (int adj : get_adjacent_positions_(current)) {
            if (!is_in_bounds_(adj)) continue;

            // Check original board for dead stones
            bool is_adj_dead = (dead_stones.find(adj) != dead_stones.end());
            int stone_on_temp = temp_get_stone(adj); // 0 if empty OR dead

            if (stone_on_temp == 0 && !is_adj_dead) { // Is original empty square
                // Empty intersection on temp board, not a dead stone loc
                if (visited_empty.find(adj) == visited_empty.end()) {
                    queue.push(adj);
                    visited_empty.insert(adj);
                }
            } else { // Is a live stone OR a dead stone location
                 int original_stone = get_stone_(adj); // Get stone from original board
                 if (original_stone != 0) { // If it's a live or dead stone (not originally empty)
                    boundary_stones.insert(adj); // Record adjacent stone position as boundary
                 }
                 // We don't need to do anything if original_stone was 0 but stone_on_temp wasn't
                 // because temp_board only contains 0 or original live stones.
            }
        }
    }
    
    // Determine territory color based on boundary stones (use original board)
    bool boundary_is_black = false;
    bool boundary_is_white = false;
    for (int boundary_pos : boundary_stones) {
        int original_stone = get_stone_(boundary_pos); // Check ORIGINAL board stone
        if (original_stone == 1) boundary_is_black = true;
        else if (original_stone == 2) boundary_is_white = true;
    }

    if (boundary_is_black && !boundary_is_white) {
        territory_color = 1;  // Black territory
    } else if (boundary_is_white && !boundary_is_black) {
        territory_color = 2;  // White territory
    } else {
        territory_color = 0;  // Neutral or contested
    }
    
    // Update territory with the determined color for all visited empty/dead points
    for (int visited_pos : visited_empty) {
        territory[visited_pos] = territory_color;
    }
}

std::pair<float, float> GoRules::calculateScores(
    const std::vector<int>& captured_stones, 
    float komi,
    const std::unordered_set<int>& dead_stones) const {
    
    float black_score = 0.0f;
    float white_score = 0.0f;
    
    // Count territory
    std::vector<int> territory = getTerritoryOwnership(dead_stones);
    
    // Count stones and territory according to the rules
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos)) continue;
        
        int owner = territory[pos];
        if (owner == 1) {
            black_score += 1.0f;  // Black territory
        } else if (owner == 2) {
            white_score += 1.0f;  // White territory
        }
    }
    
    if (!chinese_rules_) {
        // For Japanese/Korean rules, add prisoners (captured stones + dead stones)
        // captured_stones[1] = stones captured BY black (white prisoners)
        // captured_stones[2] = stones captured BY white (black prisoners)
        int white_prisoners = captured_stones[1];  // Stones captured by black
        int black_prisoners = captured_stones[2];  // Stones captured by white
        
        // Count dead stones as prisoners
        for (int pos : dead_stones) {
            int stone = get_stone_(pos);
            if (stone == 1) {
                white_prisoners++;  // Dead black stone is a prisoner for white
            } else if (stone == 2) {
                black_prisoners++;  // Dead white stone is a prisoner for black
            }
        }
        
        black_score += static_cast<float>(white_prisoners);
        white_score += static_cast<float>(black_prisoners);
    }
    
    // Add komi
    white_score += komi;
    
    return {black_score, white_score};
}

} // namespace go
} // namespace games
} // namespace alphazero