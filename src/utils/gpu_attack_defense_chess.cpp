#include "utils/attack_defense_module.h"
#include "games/chess/chess_state.h"
#include <torch/torch.h>
#include <bitset>

namespace alphazero {
namespace utils {

class ChessGPUAttackDefense {
private:
    // Pre-computed ray masks for sliding pieces
    static torch::Tensor compute_ray_masks(int device_type) {
        // 64x64x8 tensor for all ray directions from each square
        auto masks = torch::zeros({64, 64, 8}, torch::kBool);
        
        // Directions: N, NE, E, SE, S, SW, W, NW
        int dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
        int dc[] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (int dir = 0; dir < 8; dir++) {
                int r = from_r + dr[dir];
                int c = from_c + dc[dir];
                
                while (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int to = r * 8 + c;
                    masks[from][to][dir] = true;
                    r += dr[dir];
                    c += dc[dir];
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }
    
    static torch::Tensor compute_knight_masks(int device_type) {
        auto masks = torch::zeros({64, 64}, torch::kBool);
        
        int knight_moves[][2] = {
            {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
            {1, -2}, {1, 2}, {2, -1}, {2, 1}
        };
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (auto& move : knight_moves) {
                int r = from_r + move[0];
                int c = from_c + move[1];
                
                if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int to = r * 8 + c;
                    masks[from][to] = true;
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }
    
    static torch::Tensor compute_king_masks(int device_type) {
        auto masks = torch::zeros({64, 64}, torch::kBool);
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    
                    int r = from_r + dr;
                    int c = from_c + dc;
                    
                    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        int to = r * 8 + c;
                        masks[from][to] = true;
                    }
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }

public:
    static torch::Tensor compute_sliding_attacks_gpu_optimized(
        const torch::Tensor& flat_boards,  // [batch, 64] 
        const torch::Tensor& piece_positions,  // [batch, num_pieces, 2] (piece_type, square)
        int color) {
        
        auto device = flat_boards.device();
        auto batch_size = flat_boards.size(0);
        
        // Pre-compute masks if not cached
        static torch::Tensor ray_masks, knight_masks, king_masks;
        static bool masks_initialized = false;
        
        if (!masks_initialized) {
            ray_masks = compute_ray_masks(device.type() == torch::kCUDA ? 1 : 0);
            knight_masks = compute_knight_masks(device.type() == torch::kCUDA ? 1 : 0);
            king_masks = compute_king_masks(device.type() == torch::kCUDA ? 1 : 0);
            masks_initialized = true;
        }
        
        // Initialize attack maps
        auto attack_maps = torch::zeros({batch_size, 64}, torch::TensorOptions().dtype(torch::kBool).device(device));
        
        // Process each piece type
        // Queens and Rooks (vertical/horizontal rays)
        auto rook_queen_mask = (piece_positions.select(1, 0) == 4) | (piece_positions.select(1, 0) == 5);
        
        // Bishops and Queens (diagonal rays)  
        auto bishop_queen_mask = (piece_positions.select(1, 0) == 3) | (piece_positions.select(1, 0) == 5);
        
        // Knights
        auto knight_mask = piece_positions.select(1, 0) == 2;
        
        // Kings
        auto king_mask = piece_positions.select(1, 0) == 6;
        
        // Pawns (special handling)
        auto pawn_mask = piece_positions.select(1, 0) == 1;
        
        // GPU-parallel ray marching for sliding pieces
        for (int b = 0; b < batch_size; b++) {
            auto board = flat_boards[b];
            auto pieces = piece_positions[b];
            
            // Process sliding pieces with ray marching
            for (int p = 0; p < pieces.size(0); p++) {
                int piece_type = pieces[p][0].item<int>();
                int square = pieces[p][1].item<int>();
                
                if (square < 0 || square >= 64) continue;
                
                // Rooks and Queens - orthogonal
                if (piece_type == 4 || piece_type == 5) {
                    for (int dir : {0, 2, 4, 6}) {  // N, E, S, W
                        auto ray = ray_masks[square].select(1, dir);
                        
                        // Find first blocker along ray
                        auto blockers = board.masked_select(ray);
                        if (blockers.numel() > 0) {
                            // Mark squares up to first piece
                            auto ray_indices = torch::nonzero(ray).squeeze();
                            for (int i = 0; i < ray_indices.size(0); i++) {
                                int target = ray_indices[i].item<int>();
                                attack_maps[b][target] = true;
                                if (board[target].item<int>() != 0) break;
                            }
                        }
                    }
                }
                
                // Bishops and Queens - diagonal
                if (piece_type == 3 || piece_type == 5) {
                    for (int dir : {1, 3, 5, 7}) {  // NE, SE, SW, NW
                        auto ray = ray_masks[square].select(1, dir);
                        
                        auto blockers = board.masked_select(ray);
                        if (blockers.numel() > 0) {
                            auto ray_indices = torch::nonzero(ray).squeeze();
                            for (int i = 0; i < ray_indices.size(0); i++) {
                                int target = ray_indices[i].item<int>();
                                attack_maps[b][target] = true;
                                if (board[target].item<int>() != 0) break;
                            }
                        }
                    }
                }
                
                // Knights - precomputed moves
                if (piece_type == 2) {
                    attack_maps[b] |= knight_masks[square];
                }
                
                // Kings - precomputed moves
                if (piece_type == 6) {
                    attack_maps[b] |= king_masks[square];
                }
                
                // Pawns - special diagonal attacks
                if (piece_type == 1) {
                    int rank = square / 8;
                    int file = square % 8;
                    
                    if (color == 1) {  // White
                        if (rank > 0) {
                            if (file > 0) attack_maps[b][(rank-1)*8 + file-1] = true;
                            if (file < 7) attack_maps[b][(rank-1)*8 + file+1] = true;
                        }
                    } else {  // Black
                        if (rank < 7) {
                            if (file > 0) attack_maps[b][(rank+1)*8 + file-1] = true;
                            if (file < 7) attack_maps[b][(rank+1)*8 + file+1] = true;
                        }
                    }
                }
            }
        }
        
        return attack_maps.to(torch::kFloat32);
    }
    
    static torch::Tensor compute_mobility_features(
        const torch::Tensor& flat_boards,
        const torch::Tensor& attack_maps) {
        
        auto device = flat_boards.device();
        auto batch_size = flat_boards.size(0);
        
        // Count legal moves per square
        auto mobility = torch::zeros({batch_size, 64}, device);
        
        for (int b = 0; b < batch_size; b++) {
            auto board = flat_boards[b];
            auto attacks = attack_maps[b];
            
            // Count attacked empty squares
            auto empty_mask = (board == 0).to(torch::kFloat32);
            auto mobile_squares = attacks * empty_mask;  // Use multiplication instead of bitwise AND for float tensors
            
            mobility[b] = mobile_squares.to(torch::kFloat32);
        }
        
        return mobility;
    }
};

// Integration with AttackDefenseModule
torch::Tensor AttackDefenseModule::computeChessAttackDefenseGPU(
    const std::vector<const games::chess::ChessState*>& states) {
    
    if (states.empty()) {
        return torch::zeros({0, 0, 0});
    }
    
    int batch_size = states.size();
    
    // Convert bitboards to flat tensor representation
    auto flat_boards = torch::zeros({batch_size, 64}, torch::kInt32);
    std::vector<torch::Tensor> piece_positions_list;
    
    for (int i = 0; i < batch_size; i++) {
        const auto& state = states[i];
        std::vector<std::pair<int, int>> pieces;  // (piece_type, square)
        
        // Extract piece positions from tensor representation
        auto tensor_repr = state->getTensorRepresentation();
        
        // Chess tensor representation typically has channels for each piece type
        // This is a simplified version - real implementation would depend on exact tensor format
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int square = r * 8 + c;
                
                // Check each piece type channel
                // Assuming channels: empty, white pawn, white knight, etc.
                // This is placeholder code - need actual tensor format
                float piece_value = tensor_repr[0][r][c];
                if (piece_value != 0) {
                    flat_boards[i][square] = static_cast<int>(piece_value);
                    pieces.push_back({abs(static_cast<int>(piece_value)), square});
                }
            }
        }
        
        // Convert to tensor
        auto piece_tensor = torch::zeros({(int)pieces.size(), 2}, torch::kInt32);
        for (size_t p = 0; p < pieces.size(); p++) {
            piece_tensor[p][0] = pieces[p].first;
            piece_tensor[p][1] = pieces[p].second;
        }
        piece_positions_list.push_back(piece_tensor);
    }
    
    // Move to GPU
    if (torch::cuda::is_available()) {
        flat_boards = flat_boards.to(torch::kCUDA);
    }
    
    // Pad piece positions to same size
    int max_pieces = 0;
    for (const auto& pieces : piece_positions_list) {
        max_pieces = std::max(max_pieces, (int)pieces.size(0));
    }
    
    auto piece_positions = torch::zeros({batch_size, max_pieces, 2}, torch::kInt32);
    for (int i = 0; i < batch_size; i++) {
        auto num_pieces = piece_positions_list[i].size(0);
        piece_positions[i].slice(0, 0, num_pieces) = piece_positions_list[i];
    }
    
    if (torch::cuda::is_available()) {
        piece_positions = piece_positions.to(torch::kCUDA);
    }
    
    // Compute attack maps for both colors
    auto white_attacks = ChessGPUAttackDefense::compute_sliding_attacks_gpu_optimized(
        flat_boards, piece_positions, 1);
    auto black_attacks = ChessGPUAttackDefense::compute_sliding_attacks_gpu_optimized(
        flat_boards.neg(), piece_positions, -1);
    
    // Compute mobility features
    auto white_mobility = ChessGPUAttackDefense::compute_mobility_features(flat_boards, white_attacks);
    auto black_mobility = ChessGPUAttackDefense::compute_mobility_features(flat_boards, black_attacks);
    
    // Reshape to 8x8 boards and combine
    auto features = torch::stack({
        white_attacks.view({batch_size, 8, 8}),
        black_attacks.view({batch_size, 8, 8}),
        white_mobility.view({batch_size, 8, 8}),
        black_mobility.view({batch_size, 8, 8})
    }, 1);
    
    return features.to(torch::kCPU);
}

} // namespace utils
} // namespace alphazero