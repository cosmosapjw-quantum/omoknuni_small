#include "utils/attack_defense_module.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <queue>

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

#ifdef BUILD_PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

#include "utils/hash_specializations.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"

namespace alphazero {

// ========== Factory Function ==========
std::unique_ptr<AttackDefenseModule> createAttackDefenseModule(
    core::GameType game_type, int board_size) {
    switch (game_type) {
        case core::GameType::GOMOKU:
            return std::make_unique<GomokuAttackDefenseModule>(board_size);
        case core::GameType::CHESS:
            return std::make_unique<ChessAttackDefenseModule>();
        case core::GameType::GO:
            return std::make_unique<GoAttackDefenseModule>(board_size);
        default:
            throw std::runtime_error("Unsupported game type for attack/defense module");
    }
}

// ========== GomokuAttackDefenseModule Implementation ==========
GomokuAttackDefenseModule::GomokuAttackDefenseModule(int board_size) 
    : AttackDefenseModule(board_size) {
}

std::pair<std::vector<float>, std::vector<float>> GomokuAttackDefenseModule::compute_bonuses(
    const std::vector<std::vector<std::vector<int>>>& board_batch,
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    auto attack = compute_attack_bonus(board_batch, chosen_moves, player_batch);
    auto defense = compute_defense_bonus(board_batch, chosen_moves, player_batch);
    
    return {attack, defense};
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>> 
GomokuAttackDefenseModule::compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    const size_t batch_size = states.size();
    
    // Initialize attack and defense planes
    std::vector<std::vector<std::vector<float>>> attack_planes(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    std::vector<std::vector<std::vector<float>>> defense_planes(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    // For each state in the batch
    for (size_t b = 0; b < batch_size; ++b) {
        auto legal_moves = states[b]->getLegalMoves();
        std::vector<std::vector<std::vector<int>>> single_board_batch(1);
        std::vector<int> single_player_batch = {states[b]->getCurrentPlayer()};
        
        // Get current board representation
        auto board_tensor = states[b]->getTensorRepresentation();
        single_board_batch[0].resize(board_size_, std::vector<int>(board_size_, 0));
        
        // Convert tensor representation to board
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                if (board_tensor[0][i][j] > 0.5f) {
                    single_board_batch[0][i][j] = states[b]->getCurrentPlayer();
                } else if (board_tensor[1][i][j] > 0.5f) {
                    single_board_batch[0][i][j] = 3 - states[b]->getCurrentPlayer();
                }
            }
        }
        
        // Calculate attack/defense score for each legal move
        for (int move : legal_moves) {
            std::vector<int> single_move = {move};
            auto [attack_scores, defense_scores] = compute_bonuses(
                single_board_batch, single_move, single_player_batch);
            
            int row = move / board_size_;
            int col = move % board_size_;
            
            attack_planes[b][row][col] = attack_scores[0];
            defense_planes[b][row][col] = defense_scores[0];
        }
    }
    
    return {attack_planes, defense_planes};
}

std::vector<float> GomokuAttackDefenseModule::compute_attack_bonus(
    const std::vector<std::vector<std::vector<int>>>& board_batch, 
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<std::vector<std::vector<int>>> board_pre(board_batch);
    std::vector<bool> mask(B, false);
    
    // Check if cells are valid for the chosen move
    for (size_t i = 0; i < B; i++) {
        int action = chosen_moves[i];
        int row = action / board_size_;
        int col = action % board_size_;
        // Valid move if the space is empty or the current player's stone (which would be the case in test scenarios)
        mask[i] = (board_pre[i][row][col] == 0 || board_pre[i][row][col] == player_batch[i]);
    }
    
    // Clear the moves to calculate "before" state
    for (size_t i = 0; i < B; i++) {
        if (mask[i]) {
            int action = chosen_moves[i];
            int row = action / board_size_;
            int col = action % board_size_;
            board_pre[i][row][col] = 0;
        }
    }
    
    // Calculate threats after the move
    auto threats_after = count_threats_for_color(board_batch, player_batch);
    
    // Calculate threats before the move
    auto threats_before = count_threats_for_color(board_pre, player_batch);
    
    // Calculate the difference (attack score)
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = threats_after[i] - threats_before[i];
    }
    
    return result;
}

std::vector<float> GomokuAttackDefenseModule::compute_defense_bonus(
    const std::vector<std::vector<std::vector<int>>>& board_batch, 
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<std::vector<std::vector<int>>> board_pre(board_batch);
    std::vector<bool> mask(B, false);
    
    // Check if cells are valid for the chosen move
    for (size_t i = 0; i < B; i++) {
        int action = chosen_moves[i];
        int row = action / board_size_;
        int col = action % board_size_;
        // Valid move if the space is empty or the current player's stone (which would be the case in test scenarios)
        mask[i] = (board_pre[i][row][col] == 0 || board_pre[i][row][col] == player_batch[i]);
    }
    
    // Clear the moves to calculate "before" state
    for (size_t i = 0; i < B; i++) {
        if (mask[i]) {
            int action = chosen_moves[i];
            int row = action / board_size_;
            int col = action % board_size_;
            board_pre[i][row][col] = 0;
        }
    }
    
    // Calculate opponent IDs
    std::vector<int> opponent_batch(B);
    for (size_t i = 0; i < B; i++) {
        opponent_batch[i] = player_batch[i] == 1 ? 2 : 1;
    }
    
    // Calculate threats for opponent after the move
    auto threats_post = count_threats_for_color(board_batch, opponent_batch);
    
    // Calculate threats for opponent before the move
    auto threats_pre = count_threats_for_color(board_pre, opponent_batch);
    
    // Calculate the difference (defense score)
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = threats_pre[i] - threats_post[i];
    }
    
    return result;
}

std::vector<float> GomokuAttackDefenseModule::count_threats_for_color(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids) {
    
    auto open_three_hv = count_open_threats_horiz_vert(boards, opponent_ids, 5, 3);
    auto open_four_hv = count_open_threats_horiz_vert(boards, opponent_ids, 6, 4);
    auto diag_open_three = count_open_threats_diagonals(boards, opponent_ids, 5, 3);
    auto diag_open_four = count_open_threats_diagonals(boards, opponent_ids, 6, 4);
    
    const size_t B = boards.size();
    std::vector<float> result(B);
    
    for (size_t i = 0; i < B; i++) {
        result[i] = open_three_hv[i] + open_four_hv[i] + diag_open_three[i] + diag_open_four[i];
    }
    
    return result;
}

// Create a mask where 1 represents the player's stones
std::vector<std::vector<std::vector<float>>> GomokuAttackDefenseModule::create_mask(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& player_ids) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    std::vector<std::vector<std::vector<float>>> mask(B, 
        std::vector<std::vector<float>>(1, 
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (boards[b][i][j] == player_ids[b]) {
                    mask[b][0][i * W + j] = 1.0f;
                }
            }
        }
    }
    
    return mask;
}

// Create a mask where 1 represents empty cells
std::vector<std::vector<std::vector<float>>> GomokuAttackDefenseModule::create_empty_mask(
    const std::vector<std::vector<std::vector<int>>>& boards) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    std::vector<std::vector<std::vector<float>>> mask(B, 
        std::vector<std::vector<float>>(1, 
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (boards[b][i][j] == 0) {
                    mask[b][0][i * W + j] = 1.0f;
                }
            }
        }
    }
    
    return mask;
}

// Transpose a mask (swap height and width dimensions)
std::vector<std::vector<std::vector<float>>> GomokuAttackDefenseModule::transpose(
    const std::vector<std::vector<std::vector<float>>>& mask) {
    
    const size_t B = mask.size();
    const size_t C = mask[0].size();
    // Fix the sqrt conversion issue by using a temp double first
    const double sqrtResult = std::sqrt(static_cast<double>(mask[0][0].size()));
    const size_t H = static_cast<size_t>(sqrtResult);
    const size_t W = H;
    
    std::vector<std::vector<std::vector<float>>> transposed(B,
        std::vector<std::vector<float>>(C,
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t i = 0; i < H; i++) {
                for (size_t j = 0; j < W; j++) {
                    transposed[b][c][j * H + i] = mask[b][c][i * W + j];
                }
            }
        }
    }
    
    return transposed;
}

std::vector<float> GomokuAttackDefenseModule::count_open_threats_horiz_vert(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids,
    int window_length,
    int required_sum) {
    
    // Create opponent mask and empty mask
    auto opp_mask = create_mask(boards, opponent_ids);
    auto empty_mask = create_empty_mask(boards);
    
    // Count patterns in horizontal direction
    auto horiz = count_1d_patterns(opp_mask, empty_mask, window_length, required_sum);
    
    // Transpose masks for vertical direction
    auto opp_mask_vert = transpose(opp_mask);
    auto empty_mask_vert = transpose(empty_mask);
    
    // Count patterns in vertical direction
    auto vert = count_1d_patterns(opp_mask_vert, empty_mask_vert, window_length, required_sum);
    
    const size_t B = boards.size();
    std::vector<float> result(B);
    
    for (size_t i = 0; i < B; i++) {
        result[i] = horiz[i] + vert[i];
    }
    
    return result;
}

std::vector<float> GomokuAttackDefenseModule::count_1d_patterns(
    const std::vector<std::vector<std::vector<float>>>& opp_mask,
    const std::vector<std::vector<std::vector<float>>>& empty_mask,
    int window_length,
    int required_sum) {
    
    const size_t B = opp_mask.size();
    // Fix the sqrt conversion issue by using a temp double first
    const double sqrtResult = std::sqrt(static_cast<double>(opp_mask[0][0].size()));
    const size_t H = static_cast<size_t>(sqrtResult);
    const size_t W = H;
    
    std::vector<float> perfect_counts(B, 0.0f);
    std::vector<float> broken_counts(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; b++) {
        // For each row
        for (size_t i = 0; i < H; i++) {
            // For each possible window start in the row
            for (size_t j = 0; j <= W - window_length; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = i * W + (j + k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern (required opponent stones in interior, both borders empty)
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    perfect_counts[b] += 1.0f;
                }
                // Check for broken pattern (one less than required opponent stones, both borders empty)
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    broken_counts[b] += 1.0f;
                }
            }
        }
    }
    
    // Sum perfect and broken counts
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = perfect_counts[i] + broken_counts[i];
    }
    
    return result;
}

std::vector<float> GomokuAttackDefenseModule::count_open_threats_diagonals(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids,
    int window_length,
    int required_sum) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    // Create opponent and empty masks
    auto opp_mask = create_mask(boards, opponent_ids);
    auto empty_mask = create_empty_mask(boards);
    
    std::vector<float> diag_count_main(B, 0.0f);
    std::vector<float> diag_count_anti(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; b++) {
        // Main diagonals (top-left to bottom-right)
        for (size_t i = 0; i <= H - window_length; i++) {
            for (size_t j = 0; j <= W - window_length; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the diagonal window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = (i + k) * W + (j + k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    diag_count_main[b] += 1.0f;
                }
                // Check for broken pattern
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    diag_count_main[b] += 1.0f;
                }
            }
        }
        
        // Anti-diagonals (top-right to bottom-left)
        for (size_t i = 0; i <= H - window_length; i++) {
            for (size_t j = window_length - 1; j < W; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the diagonal window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = (i + k) * W + (j - k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    diag_count_anti[b] += 1.0f;
                }
                // Check for broken pattern
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    diag_count_anti[b] += 1.0f;
                }
            }
        }
    }
    
    // Sum main and anti-diagonal counts
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = diag_count_main[i] + diag_count_anti[i];
    }
    
    return result;
}

// ========== ChessAttackDefenseModule Implementation ==========
ChessAttackDefenseModule::ChessAttackDefenseModule() 
    : AttackDefenseModule(8) {  // Chess board is always 8x8
}

float ChessAttackDefenseModule::getPieceValue(int piece_type) const {
    // Assuming piece encoding: 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
    switch (std::abs(piece_type)) {
        case 1: return PAWN_VALUE;
        case 2: return KNIGHT_VALUE;
        case 3: return BISHOP_VALUE;
        case 4: return ROOK_VALUE;
        case 5: return QUEEN_VALUE;
        case 6: return KING_VALUE;
        default: return 0.0f;
    }
}

std::vector<std::pair<int,int>> ChessAttackDefenseModule::getAttackedSquares(
    const std::vector<std::vector<int>>& board, 
    int from_row, int from_col, int piece_type) const {
    
    std::vector<std::pair<int,int>> attacked_squares;
    
    // Simplified attack calculation - would need full chess move generation in practice
    // This is a placeholder that should be replaced with proper chess logic
    
    return attacked_squares;
}

std::pair<std::vector<float>, std::vector<float>> ChessAttackDefenseModule::compute_bonuses(
    const std::vector<std::vector<std::vector<int>>>& board_batch,
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<float> attack_bonuses(B, 0.0f);
    std::vector<float> defense_bonuses(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; ++b) {
        int move = chosen_moves[b];
        int to_row = move / 8;
        int to_col = move % 8;
        int player = player_batch[b];
        
        // Count newly attacked enemy pieces
        float attack_value = 0.0f;
        float defense_value = 0.0f;
        
        // This is a simplified implementation
        // In practice, you would need full chess move generation
        // to properly calculate attacks and defenses
        
        // Check all adjacent squares for simple demonstration
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                
                int r = to_row + dr;
                int c = to_col + dc;
                
                if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int piece = board_batch[b][r][c];
                    
                    // If it's an enemy piece
                    if (piece != 0 && ((piece > 0) != (player == 1))) {
                        attack_value += getPieceValue(piece);
                    }
                    // If it's a friendly piece
                    else if (piece != 0 && ((piece > 0) == (player == 1))) {
                        defense_value += getPieceValue(piece);
                    }
                }
            }
        }
        
        attack_bonuses[b] = attack_value;
        defense_bonuses[b] = defense_value;
    }
    
    return {attack_bonuses, defense_bonuses};
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>> 
ChessAttackDefenseModule::compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    const size_t batch_size = states.size();
    
    // Initialize attack and defense planes
    std::vector<std::vector<std::vector<float>>> attack_planes(
        batch_size, std::vector<std::vector<float>>(
            8, std::vector<float>(8, 0.0f)));
    
    std::vector<std::vector<std::vector<float>>> defense_planes(
        batch_size, std::vector<std::vector<float>>(
            8, std::vector<float>(8, 0.0f)));
    
    // Simplified implementation - in practice would need full chess logic
    // For now, just return zero planes
    
    return {attack_planes, defense_planes};
}

// ========== GoAttackDefenseModule Implementation ==========
GoAttackDefenseModule::GoAttackDefenseModule(int board_size) 
    : AttackDefenseModule(board_size) {
}

std::vector<GoAttackDefenseModule::Group> GoAttackDefenseModule::findGroups(
    const std::vector<std::vector<int>>& board) const {
    
    std::vector<Group> groups;
    std::vector<std::vector<bool>> visited(board_size_, std::vector<bool>(board_size_, false));
    
    // Find all groups using flood fill
    for (int i = 0; i < board_size_; ++i) {
        for (int j = 0; j < board_size_; ++j) {
            if (board[i][j] != 0 && !visited[i][j]) {
                Group group;
                group.player = board[i][j];
                
                // Flood fill to find all stones in this group
                std::vector<std::pair<int,int>> stack = {{i, j}};
                while (!stack.empty()) {
                    auto [r, c] = stack.back();
                    stack.pop_back();
                    
                    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_) continue;
                    if (visited[r][c] || board[r][c] != group.player) continue;
                    
                    visited[r][c] = true;
                    group.stones.push_back({r, c});
                    
                    // Add adjacent stones
                    stack.push_back({r-1, c});
                    stack.push_back({r+1, c});
                    stack.push_back({r, c-1});
                    stack.push_back({r, c+1});
                }
                
                // Count liberties for this group
                group.liberties = countLiberties(board, group.stones);
                groups.push_back(group);
            }
        }
    }
    
    return groups;
}

int GoAttackDefenseModule::countLiberties(
    const std::vector<std::vector<int>>& board, 
    const std::vector<std::pair<int,int>>& group) const {
    
    std::unordered_set<int> liberties;
    
    for (auto [r, c] : group) {
        // Check all four adjacent squares
        const std::pair<int,int> adj[] = {{r-1,c}, {r+1,c}, {r,c-1}, {r,c+1}};
        for (auto [ar, ac] : adj) {
            if (ar >= 0 && ar < board_size_ && ac >= 0 && ac < board_size_ && board[ar][ac] == 0) {
                liberties.insert(ar * board_size_ + ac);
            }
        }
    }
    
    return liberties.size();
}

bool GoAttackDefenseModule::wouldCapture(
    const std::vector<std::vector<int>>& board, 
    int row, int col, int player) const {
    
    // Check if placing a stone would capture any opponent groups
    const int opponent = 3 - player;
    
    // Check all four adjacent positions
    const std::pair<int,int> adj[] = {{row-1,col}, {row+1,col}, {row,col-1}, {row,col+1}};
    for (auto [r, c] : adj) {
        if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_ && board[r][c] == opponent) {
            // Find the group this stone belongs to
            std::vector<std::pair<int,int>> group;
            std::vector<std::vector<bool>> visited(board_size_, std::vector<bool>(board_size_, false));
            
            // Simple flood fill to find group
            std::vector<std::pair<int,int>> stack = {{r, c}};
            while (!stack.empty()) {
                auto [gr, gc] = stack.back();
                stack.pop_back();
                
                if (gr < 0 || gr >= board_size_ || gc < 0 || gc >= board_size_) continue;
                if (visited[gr][gc] || board[gr][gc] != opponent) continue;
                
                visited[gr][gc] = true;
                group.push_back({gr, gc});
                
                stack.push_back({gr-1, gc});
                stack.push_back({gr+1, gc});
                stack.push_back({gr, gc-1});
                stack.push_back({gr, gc+1});
            }
            
            // Check if this group would have 0 liberties after our move
            int liberties = 0;
            for (auto [gr, gc] : group) {
                const std::pair<int,int> group_adj[] = {{gr-1,gc}, {gr+1,gc}, {gr,gc-1}, {gr,gc+1}};
                for (auto [ar, ac] : group_adj) {
                    if (ar >= 0 && ar < board_size_ && ac >= 0 && ac < board_size_ && 
                        board[ar][ac] == 0 && !(ar == row && ac == col)) {
                        liberties++;
                        break;
                    }
                }
                if (liberties > 0) break;
            }
            
            if (liberties == 0) return true;
        }
    }
    
    return false;
}

bool GoAttackDefenseModule::createsAtari(
    const std::vector<std::vector<int>>& board, 
    int row, int col, int player) const {
    
    // Check if placing a stone puts any opponent group in atari (1 liberty)
    const int opponent = 3 - player;
    
    // Make a copy and place the stone
    auto board_copy = board;
    board_copy[row][col] = player;
    
    // Find all groups and check for atari
    auto groups = findGroups(board_copy);
    for (const auto& group : groups) {
        if (group.player == opponent && group.liberties == 1) {
            return true;
        }
    }
    
    return false;
}

float GoAttackDefenseModule::evaluateEyePotential(
    const std::vector<std::vector<int>>& board, 
    int row, int col, int player) const {
    
    // Simple heuristic for eye potential
    // Check if surrounded by friendly stones
    float eye_score = 0.0f;
    int friendly_count = 0;
    int total_count = 0;
    
    // Check all 8 surrounding squares
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            
            int r = row + dr;
            int c = col + dc;
            
            if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
                total_count++;
                if (board[r][c] == player) {
                    friendly_count++;
                }
            }
        }
    }
    
    if (total_count > 0) {
        eye_score = static_cast<float>(friendly_count) / total_count;
    }
    
    return eye_score;
}

std::pair<std::vector<float>, std::vector<float>> GoAttackDefenseModule::compute_bonuses(
    const std::vector<std::vector<std::vector<int>>>& board_batch,
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<float> attack_bonuses(B, 0.0f);
    std::vector<float> defense_bonuses(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; ++b) {
        int move = chosen_moves[b];
        int row = move / board_size_;
        int col = move % board_size_;
        int player = player_batch[b];
        
        const auto& board = board_batch[b];
        
        // Calculate attack score
        float attack_score = 0.0f;
        
        // Check for captures
        if (wouldCapture(board, row, col, player)) {
            attack_score += CAPTURE_WEIGHT * 5.0f; // Assume 5 stones captured on average
        }
        
        // Check for atari creation
        if (createsAtari(board, row, col, player)) {
            attack_score += ATARI_WEIGHT * 3.0f;
        }
        
        // Liberty pressure (simplified)
        attack_score += LIBERTY_WEIGHT * 2.0f;
        
        // Eye destruction potential
        float eye_potential = evaluateEyePotential(board, row, col, 3 - player);
        attack_score += EYE_WEIGHT * eye_potential * 5.0f;
        
        // Calculate defense score
        float defense_score = 0.0f;
        
        // Save from capture (simplified)
        defense_score += CAPTURE_WEIGHT * 2.0f;
        
        // Escape from atari
        defense_score += ATARI_WEIGHT * 1.0f;
        
        // Liberty gain
        defense_score += LIBERTY_WEIGHT * 3.0f;
        
        // Eye creation
        float own_eye_potential = evaluateEyePotential(board, row, col, player);
        defense_score += EYE_WEIGHT * own_eye_potential * 5.0f;
        
        attack_bonuses[b] = attack_score;
        defense_bonuses[b] = defense_score;
    }
    
    return {attack_bonuses, defense_bonuses};
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>> 
GoAttackDefenseModule::compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    const size_t batch_size = states.size();
    
    // Initialize attack and defense planes
    std::vector<std::vector<std::vector<float>>> attack_planes(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    std::vector<std::vector<std::vector<float>>> defense_planes(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    // For each state in the batch
    for (size_t b = 0; b < batch_size; ++b) {
        auto legal_moves = states[b]->getLegalMoves();
        std::vector<std::vector<std::vector<int>>> single_board_batch(1);
        std::vector<int> single_player_batch = {states[b]->getCurrentPlayer()};
        
        // Get current board representation
        auto board_tensor = states[b]->getTensorRepresentation();
        single_board_batch[0].resize(board_size_, std::vector<int>(board_size_, 0));
        
        // Convert tensor representation to board
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                if (board_tensor[0][i][j] > 0.5f) {
                    single_board_batch[0][i][j] = states[b]->getCurrentPlayer();
                } else if (board_tensor[1][i][j] > 0.5f) {
                    single_board_batch[0][i][j] = 3 - states[b]->getCurrentPlayer();
                }
            }
        }
        
        // Calculate attack/defense score for each legal move
        for (int move : legal_moves) {
            std::vector<int> single_move = {move};
            auto [attack_scores, defense_scores] = compute_bonuses(
                single_board_batch, single_move, single_player_batch);
            
            int row = move / board_size_;
            int col = move % board_size_;
            
            attack_planes[b][row][col] = attack_scores[0];
            defense_planes[b][row][col] = defense_scores[0];
        }
    }
    
    return {attack_planes, defense_planes};
}

} // namespace alphazero

namespace alphazero {
namespace utils {

#ifdef WITH_TORCH
// Check if GPU is available
bool AttackDefenseModule::isGPUAvailable() {
    return torch::cuda::is_available();
}
#endif

} // namespace utils
} // namespace alphazero

// Note: GPU implementations are in separate files:
// - gpu_attack_defense_gomoku.cpp
// - gpu_attack_defense_chess.cpp  
// - gpu_attack_defense_go.cpp

#ifdef BUILD_PYTHON_BINDINGS
PYBIND11_MODULE(attack_defense, m) {
    m.doc() = "Attack and Defense calculation module for board games";
    
    py::class_<alphazero::AttackDefenseModule>(m, "AttackDefenseModule");
    
    py::class_<alphazero::GomokuAttackDefenseModule, alphazero::AttackDefenseModule>(m, "GomokuAttackDefenseModule")
        .def(py::init<int>(), py::arg("board_size"))
        .def("__call__", [](alphazero::GomokuAttackDefenseModule& self, 
                           py::array_t<float> board_np, 
                           py::array_t<int64_t> moves_np,
                           py::array_t<int64_t> player_np) {
            auto board_buffer = board_np.request();
            auto moves_buffer = moves_np.request();
            auto player_buffer = player_np.request();
            
            // Get shape information - using appropriate types for pybind
            const auto batch_size = static_cast<size_t>(board_buffer.shape[0]);
            const auto board_height = static_cast<size_t>(board_buffer.shape[1]);
            const auto board_width = static_cast<size_t>(board_buffer.shape[2]);
            
            // Convert numpy arrays to C++ vectors
            std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
                std::vector<std::vector<int>>(board_height, 
                    std::vector<int>(board_width, 0)));
            
            std::vector<int> chosen_moves(batch_size, 0);
            std::vector<int> player_batch(batch_size, 0);
            
            // Copy data from numpy arrays
            float* board_ptr = static_cast<float*>(board_buffer.ptr);
            int64_t* moves_ptr = static_cast<int64_t*>(moves_buffer.ptr);
            int64_t* player_ptr = static_cast<int64_t*>(player_buffer.ptr);
            
            // Fill board data
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t i = 0; i < board_height; i++) {
                    for (size_t j = 0; j < board_width; j++) {
                        board_batch[b][i][j] = static_cast<int>(board_ptr[b * board_height * board_width + i * board_width + j]);
                    }
                }
                
                // Copy moves and player IDs
                chosen_moves[b] = static_cast<int>(moves_ptr[b]);
                player_batch[b] = static_cast<int>(player_ptr[b]);
            }
            
            // Call the C++ implementation
            auto [attack_bonus, defense_bonus] = self.compute_bonuses(board_batch, chosen_moves, player_batch);
            
            // Convert back to numpy arrays - fix the shape parameter to avoid narrowing
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(attack_bonus.size())};
            py::array_t<float> attack_bonus_np(shape);
            py::array_t<float> defense_bonus_np(shape);
            
            auto attack_buffer = attack_bonus_np.request();
            auto defense_buffer = defense_bonus_np.request();
            
            float* attack_ptr = static_cast<float*>(attack_buffer.ptr);
            float* defense_ptr = static_cast<float*>(defense_buffer.ptr);
            
            for (size_t i = 0; i < attack_bonus.size(); i++) {
                attack_ptr[i] = attack_bonus[i];
                defense_ptr[i] = defense_bonus[i];
            }
            
            return py::make_tuple(attack_bonus_np, defense_bonus_np);
        });
}
#endif