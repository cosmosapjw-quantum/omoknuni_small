#include "utils/attack_defense_module.h"
#include "games/go/go_state.h"
#include <torch/torch.h>
#include <queue>

namespace alphazero {
namespace utils {

class GoGPUAttackDefense {
private:
    // Flood fill for group detection using GPU-friendly iterative approach
    static torch::Tensor gpu_flood_fill_groups(const torch::Tensor& board, int player) {
        auto device = board.device();
        auto batch_size = board.size(0);
        auto board_size = board.size(1);
        
        
        // Initialize group IDs
        auto group_ids = torch::zeros_like(board, torch::kInt32);
        auto player_mask = (board == player);
        
        // Iterative flood fill using convolutions
        int group_id = 1;
        auto unvisited = player_mask.clone();
        
        while (unvisited.any().item<bool>()) {
            // Find first unvisited stone
            auto indices = torch::nonzero(unvisited);
            if (indices.size(0) == 0) break;
            
            auto seed_batch = indices[0][0].item<int>();
            auto seed_row = indices[0][1].item<int>();
            auto seed_col = indices[0][2].item<int>();
            
            // Create seed mask
            auto current_group = torch::zeros_like(board, torch::kBool);
            current_group[seed_batch][seed_row][seed_col] = true;
            
            // Iterative expansion using 4-connected convolution
            // Create 4D kernel: [out_channels=1, in_channels=1, height=3, width=3]
            auto kernel_3x3 = torch::tensor({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}, device).to(torch::kFloat32);
            auto kernel = kernel_3x3.unsqueeze(0).unsqueeze(0);  // Now [1, 1, 3, 3]
            
            
            bool changed = true;
            while (changed) {
                // Process each batch item separately
                auto expanded = torch::zeros_like(current_group, torch::kBool);
                for (int b = 0; b < batch_size; b++) {
                    auto group_slice = current_group[b].unsqueeze(0).unsqueeze(0).to(torch::kFloat32);
                    auto exp_slice = torch::nn::functional::conv2d(
                        group_slice,
                        kernel,
                        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
                    ).squeeze(0).squeeze(0) > 0;
                    expanded[b] = exp_slice;
                }
                
                // Mask to only player stones
                expanded = expanded & player_mask & unvisited;
                
                auto new_group = current_group | expanded;
                changed = !torch::equal(new_group, current_group);
                current_group = new_group;
            }
            
            // Assign group ID
            group_ids.masked_fill_(current_group, group_id);
            unvisited = unvisited & ~current_group;
            group_id++;
        }
        
        return group_ids;
    }
    
    // Count liberties using convolutions
    static torch::Tensor compute_liberties_gpu(const torch::Tensor& board, const torch::Tensor& group_ids) {
        auto device = board.device();
        auto batch_size = board.size(0);
        auto board_size = board.size(1);
        
        auto liberties = torch::zeros_like(board, torch::kFloat32);
        auto empty_mask = (board == 0);
        
        // 4-connected kernel
        auto kernel_2d = torch::tensor({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}, device).to(torch::kFloat32);
        auto kernel = kernel_2d.unsqueeze(0).unsqueeze(0);  // Now [1, 1, 3, 3]
        
        // Process each batch item separately
        for (int b = 0; b < batch_size; b++) {
            auto board_slice = board[b];
            auto group_ids_slice = group_ids[b];
            auto empty_mask_slice = empty_mask[b];
            
            // For each unique group in this batch
            auto unique_result = at::_unique(group_ids_slice);
            auto unique_groups = std::get<0>(unique_result);
            
            for (int i = 0; i < unique_groups.size(0); i++) {
                int group = unique_groups[i].item<int>();
                if (group == 0) continue;  // Skip empty
                
                auto group_mask = (group_ids_slice == group);
                
                // Find adjacent empty points
                auto group_mask_4d = group_mask.unsqueeze(0).unsqueeze(0).to(torch::kFloat32);
                auto adjacent_empty = torch::nn::functional::conv2d(
                    group_mask_4d,
                    kernel,
                    torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
                ).squeeze(0).squeeze(0) > 0;
                
                adjacent_empty = adjacent_empty.to(torch::kFloat32) * empty_mask_slice.to(torch::kFloat32);
                
                // Count unique liberties
                int liberty_count = adjacent_empty.sum().item<int>();
                liberties[b].masked_fill_(group_mask, liberty_count);
            }
        }
        
        return liberties;
    }
    
    // Detect eyes (surrounded empty points)
    static torch::Tensor detect_eyes_gpu(const torch::Tensor& board, int player) {
        auto device = board.device();
        auto empty_mask = (board == 0);
        auto player_mask = (board == player);
        
        // Full surround kernel (8-connected)
        auto kernel = torch::ones({1, 1, 3, 3}, device).to(torch::kFloat32);
        kernel[0][0][1][1] = 0;  // Center is empty
        
        // Check if empty points are surrounded by player stones
        auto surrounded = torch::nn::functional::conv2d(
            player_mask.unsqueeze(1).to(torch::kFloat32),
            kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
        ).squeeze(1);
        
        // Eye if empty and surrounded by at least 7 friendly stones
        auto eyes = empty_mask.to(torch::kFloat32) * (surrounded >= 7).to(torch::kFloat32);
        
        return eyes.to(torch::kFloat32);
    }
    
    // Ladder detection (simplified)
    static torch::Tensor detect_ladders_gpu(const torch::Tensor& board, const torch::Tensor& liberties) {
        // Groups with exactly 2 liberties are ladder candidates
        auto ladder_candidates = (liberties == 2);
        
        // Additional logic would check if those liberties lead to capture
        // This is a simplified version
        return ladder_candidates.to(torch::kFloat32);
    }

public:
    static torch::Tensor compute_go_features_batch(const torch::Tensor& board_batch) {
        auto device = board_batch.device();
        auto batch_size = board_batch.size(0);
        auto board_size = board_batch.size(1);
        
        
        // Initialize feature channels
        const int num_features = 12;
        auto features = torch::zeros({batch_size, num_features, board_size, board_size}, device);
        
        // Basic stone positions
        features.select(1, 0) = (board_batch == 1).to(torch::kFloat32);  // Black stones
        features.select(1, 1) = (board_batch == 2).to(torch::kFloat32);  // White stones
        
        // Group IDs for both players
        auto black_groups = gpu_flood_fill_groups(board_batch, 1);
        auto white_groups = gpu_flood_fill_groups(board_batch, 2);
        
        // Liberty counts
        auto black_liberties = compute_liberties_gpu(board_batch, black_groups);
        auto white_liberties = compute_liberties_gpu(board_batch, white_groups);
        
        features.select(1, 2) = black_liberties.to(torch::kFloat32) / 4.0;  // Normalized
        features.select(1, 3) = white_liberties.to(torch::kFloat32) / 4.0;
        
        // Atari detection (1 liberty)
        features.select(1, 4) = (black_liberties == 1).to(torch::kFloat32);
        features.select(1, 5) = (white_liberties == 1).to(torch::kFloat32);
        
        // Eye detection
        features.select(1, 6) = detect_eyes_gpu(board_batch, 1);
        features.select(1, 7) = detect_eyes_gpu(board_batch, 2);
        
        // Ladder detection
        features.select(1, 8) = detect_ladders_gpu(board_batch, black_liberties);
        features.select(1, 9) = detect_ladders_gpu(board_batch, white_liberties);
        
        // Territory estimation (simplified - distance to nearest stone)
        auto distance_kernel = torch::ones({1, 1, 5, 5}, device).to(torch::kFloat32) / 25.0;
        auto black_stones = (board_batch == 1).to(torch::kFloat32).unsqueeze(1);  // [batch, 1, h, w]
        auto white_stones = (board_batch == 2).to(torch::kFloat32).unsqueeze(1);  // [batch, 1, h, w]
        
        auto black_influence = torch::nn::functional::conv2d(
            black_stones, distance_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2)
        );
        auto white_influence = torch::nn::functional::conv2d(
            white_stones, distance_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2)
        );
        
        features.select(1, 10) = black_influence.squeeze(1);
        features.select(1, 11) = white_influence.squeeze(1);
        
        return features;
    }
    
    // CPU-GPU hybrid for complex patterns
    static torch::Tensor compute_complex_patterns_hybrid(
        const torch::Tensor& board_batch,
        const std::vector<std::pair<int, int>>& critical_points) {
        
        auto device = board_batch.device();
        auto batch_size = board_batch.size(0);
        auto board_size = board_batch.size(1);
        
        // Move to CPU for complex pattern matching
        auto cpu_boards = board_batch.to(torch::kCPU);
        
        auto pattern_features = torch::zeros({batch_size, 4, board_size, board_size});
        
        // Common Go patterns (simplified)
        // Pattern 1: Bamboo joint
        // Pattern 2: Tiger's mouth  
        // Pattern 3: Empty triangle
        // Pattern 4: Hane
        
        for (int b = 0; b < batch_size; b++) {
            auto board = cpu_boards[b];
            
            // Check each critical point
            for (const auto& point : critical_points) {
                int row = point.first;
                int col = point.second;
                
                if (row >= 2 && row < board_size - 2 && 
                    col >= 2 && col < board_size - 2) {
                    
                    // Extract 5x5 window
                    auto window = board.slice(0, row-2, row+3).slice(1, col-2, col+3);
                    
                    // Pattern matching (simplified)
                    // Would normally use pre-defined pattern templates
                    
                    // Example: detect potential ko situations
                    if (board[row][col].item<int>() == 0) {
                        int adjacent_friendly = 0;
                        int adjacent_enemy = 0;
                        
                        if (row > 0 && board[row-1][col].item<int>() == 1) adjacent_friendly++;
                        if (row < board_size-1 && board[row+1][col].item<int>() == 1) adjacent_friendly++;
                        if (col > 0 && board[row][col-1].item<int>() == 1) adjacent_friendly++;
                        if (col < board_size-1 && board[row][col+1].item<int>() == 1) adjacent_friendly++;
                        
                        if (adjacent_friendly >= 3) {
                            pattern_features[b][0][row][col] = 1.0;
                        }
                    }
                }
            }
        }
        
        // Move back to original device
        return pattern_features.to(device);
    }
};

// Integration with AttackDefenseModule
torch::Tensor AttackDefenseModule::computeGoAttackDefenseGPU(
    const std::vector<const games::go::GoState*>& states) {
    
    if (states.empty()) {
        return torch::zeros({0, 0, 0, 0});
    }
    
    int board_size = states[0]->getBoardSize();
    int batch_size = states.size();
    
    // Convert states to tensor batch
    auto board_batch = torch::zeros({batch_size, board_size, board_size}, torch::kInt32);
    std::vector<std::pair<int, int>> all_critical_points;
    
    for (int i = 0; i < batch_size; i++) {
        // Get board representation from tensor representation
        auto tensor_repr = states[i]->getTensorRepresentation();
        const auto& board = tensor_repr[0];  // First channel is current board state
        
        for (int r = 0; r < board_size; r++) {
            for (int c = 0; c < board_size; c++) {
                // In Go: 0=empty, 1=black, 2=white
                board_batch[i][r][c] = board[r][c];
                
                // Identify critical points (last moves, captures, etc.)
                // TODO: Add last move tracking if needed
                // if (states[i]->getLastMove() == r * board_size + c) {
                //     all_critical_points.push_back({r, c});
                // }
            }
        }
    }
    
    // Move to GPU if available
    if (torch::cuda::is_available()) {
        board_batch = board_batch.to(torch::kCUDA);
    }
    
    // Compute GPU features
    auto gpu_features = GoGPUAttackDefense::compute_go_features_batch(board_batch);
    
    // Compute hybrid features for complex patterns
    auto hybrid_features = GoGPUAttackDefense::compute_complex_patterns_hybrid(
        board_batch, all_critical_points);
    
    // Combine all features
    auto combined = torch::cat({gpu_features, hybrid_features}, 1);
    
    return combined.to(torch::kCPU);
}

// External function wrapper for use in other files
torch::Tensor GoGPUAttackDefense_compute_go_features_batch(const torch::Tensor& boards) {
    return GoGPUAttackDefense::compute_go_features_batch(boards);
}

} // namespace utils
} // namespace alphazero