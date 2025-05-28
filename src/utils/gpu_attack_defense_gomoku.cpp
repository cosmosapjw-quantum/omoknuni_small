#include "utils/attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include <torch/torch.h>

namespace alphazero {
namespace utils {

class GomokuGPUAttackDefense {
public:
    static torch::Tensor compute_gomoku_threats(const torch::Tensor& board_batch, int player) {
        // board_batch shape: [batch_size, board_size, board_size]
        auto device = board_batch.device();
        auto batch_size = board_batch.size(0);
        auto board_size = board_batch.size(1);
        
        // Create player mask
        auto player_mask = (board_batch == player).to(torch::kFloat32);
        auto opponent_mask = (board_batch == 3 - player).to(torch::kFloat32);
        
        // Define kernels for different patterns (5 in a row)
        // Horizontal kernel
        auto h_kernel = torch::ones({1, 1, 1, 5}, device).to(device);
        // Vertical kernel  
        auto v_kernel = torch::ones({1, 1, 5, 1}, device).to(device);
        // Diagonal kernels
        auto d1_kernel = torch::eye(5, device).unsqueeze(0).unsqueeze(0).to(device);
        auto d2_kernel = torch::flip(d1_kernel, {3});
        
        // Reshape for conv2d
        auto player_4d = player_mask.unsqueeze(1);
        auto opponent_4d = opponent_mask.unsqueeze(1);
        
        // Detect open threes (3 in a row with spaces on both sides)
        auto three_kernel_h = torch::tensor({{{0, 1, 1, 1, 0}}}, device).unsqueeze(1).to(torch::kFloat32).to(device);
        auto three_kernel_v = three_kernel_h.transpose(2, 3);
        
        auto empty_mask = (board_batch == 0).to(torch::kFloat32).unsqueeze(1);
        
        // Combine all threat patterns
        auto threats = torch::zeros({batch_size, board_size, board_size}, device).to(device);
        
        // Instead of trying to handle different sizes, use padding in convolutions
        // to maintain the same output size
        
        // Recompute with proper padding
        auto h_kernel_padded = h_kernel;
        auto v_kernel_padded = v_kernel;
        auto d1_kernel_padded = d1_kernel;
        auto d2_kernel_padded = d2_kernel;
        
        // For 1x5 kernel, we need padding of 2 on left/right to maintain width
        auto h_conv = torch::nn::functional::conv2d(player_4d, h_kernel_padded,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({0, 2}));
        
        // For 5x1 kernel, we need padding of 2 on top/bottom to maintain height
        auto v_conv = torch::nn::functional::conv2d(player_4d, v_kernel_padded,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({2, 0}));
            
        // For 5x5 diagonal kernels, we need padding of 2 all around
        auto d1_conv = torch::nn::functional::conv2d(player_4d, d1_kernel_padded,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2));
        auto d2_conv = torch::nn::functional::conv2d(player_4d, d2_kernel_padded,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2));
        
        // Now all convolutions have the same output size as input
        auto h_threat = (h_conv >= 5).squeeze(1).to(torch::kFloat32);
        auto v_threat = (v_conv >= 5).squeeze(1).to(torch::kFloat32);
        auto d1_threat = (d1_conv >= 5).squeeze(1).to(torch::kFloat32);
        auto d2_threat = (d2_conv >= 5).squeeze(1).to(torch::kFloat32);
        
        // Combine threats with different weights
        threats = h_threat * 1000 + v_threat * 1000 + d1_threat * 1000 + d2_threat * 1000;
        
        
        // Add 4-in-a-row threats (only where we don't already have 5-in-a-row)
        auto h4_threat = ((h_conv >= 4) & (h_conv < 5)).squeeze(1).to(torch::kFloat32);
        auto v4_threat = ((v_conv >= 4) & (v_conv < 5)).squeeze(1).to(torch::kFloat32);
        auto d14_threat = ((d1_conv >= 4) & (d1_conv < 5)).squeeze(1).to(torch::kFloat32);
        auto d24_threat = ((d2_conv >= 4) & (d2_conv < 5)).squeeze(1).to(torch::kFloat32);
        
        threats = threats + (h4_threat + v4_threat + d14_threat + d24_threat) * 100;
        
        // Add 3-in-a-row threats
        auto h3_conv = torch::nn::functional::conv2d(player_4d, three_kernel_h,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({0, 2}));
        auto v3_conv = torch::nn::functional::conv2d(player_4d, three_kernel_v,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({2, 0}));
            
        auto h3_threat = (h3_conv >= 3).squeeze(1).to(torch::kFloat32);
        auto v3_threat = (v3_conv >= 3).squeeze(1).to(torch::kFloat32);
        
        threats = threats + (h3_threat + v3_threat) * 10;
        
        return threats;
    }
    
    static torch::Tensor compute_pattern_features(const torch::Tensor& board_batch) {
        // Extract various pattern features for neural network
        auto device = board_batch.device();
        auto batch_size = board_batch.size(0);
        auto board_size = board_batch.size(1);
        
        // Initialize feature tensor with multiple channels
        const int num_features = 8; // Different pattern types
        auto features = torch::zeros({batch_size, num_features, board_size, board_size}, device).to(device);
        
        // Player 1 and 2 masks
        auto p1_mask = (board_batch == 1).to(torch::kFloat32).unsqueeze(1);
        auto p2_mask = (board_batch == 2).to(torch::kFloat32).unsqueeze(1);
        
        // Feature 0-1: Raw player positions
        features.slice(1, 0, 1) = p1_mask;
        features.slice(1, 1, 2) = p2_mask;
        
        // Feature 2-3: Horizontal connectivity (using conv1d-like operation)
        auto h_kernel = torch::ones({1, 1, 1, 3}, device).to(device);
        features.slice(1, 2, 3) = torch::nn::functional::conv2d(p1_mask, h_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({0, 1})) / 3.0;
        features.slice(1, 3, 4) = torch::nn::functional::conv2d(p2_mask, h_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({0, 1})) / 3.0;
        
        // Feature 4-5: Vertical connectivity
        auto v_kernel = torch::ones({1, 1, 3, 1}, device).to(device);
        features.slice(1, 4, 5) = torch::nn::functional::conv2d(p1_mask, v_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({1, 0})) / 3.0;
        features.slice(1, 5, 6) = torch::nn::functional::conv2d(p2_mask, v_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding({1, 0})) / 3.0;
        
        // Feature 6-7: Diagonal connectivity
        auto d_kernel = torch::eye(3, device).unsqueeze(0).unsqueeze(0).to(device);
        features.slice(1, 6, 7) = torch::nn::functional::conv2d(p1_mask, d_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)) / 3.0;
        features.slice(1, 7, 8) = torch::nn::functional::conv2d(p2_mask, d_kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)) / 3.0;
        
        return features;
    }
};

// Integration with existing AttackDefenseModule
torch::Tensor AttackDefenseModule::computeGomokuAttackDefenseGPU(
    const std::vector<const games::gomoku::GomokuState*>& states) {
    
    if (states.empty()) {
        return torch::zeros({0, 0, 0});
    }
    
    int board_size = states[0]->getBoardSize();
    int batch_size = states.size();
    
    // Convert states to tensor batch
    auto board_batch = torch::zeros({batch_size, board_size, board_size}, torch::kInt32);
    
    for (int i = 0; i < batch_size; i++) {
        // Get board representation from tensor representation
        auto tensor_repr = states[i]->getTensorRepresentation();
        // Combine both players' stones: 1 for player 1, 2 for player 2, 0 for empty
        for (int r = 0; r < board_size; r++) {
            for (int c = 0; c < board_size; c++) {
                if (tensor_repr[0][r][c] == 1.0f) {
                    board_batch[i][r][c] = 1;  // Player 1 (BLACK)
                } else if (tensor_repr[1][r][c] == 1.0f) {
                    board_batch[i][r][c] = 2;  // Player 2 (WHITE)
                } else {
                    board_batch[i][r][c] = 0;  // Empty
                }
            }
        }
    }
    
    // Move to GPU if available
    if (torch::cuda::is_available()) {
        board_batch = board_batch.to(torch::kCUDA);
    }
    
    // Compute threats for both players
    auto p1_threats = GomokuGPUAttackDefense::compute_gomoku_threats(board_batch, 1);
    auto p2_threats = GomokuGPUAttackDefense::compute_gomoku_threats(board_batch, 2);
    
    // Compute pattern features
    auto pattern_features = GomokuGPUAttackDefense::compute_pattern_features(board_batch);
    
    // Combine into final feature tensor
    auto combined = torch::cat({
        p1_threats.unsqueeze(1),
        p2_threats.unsqueeze(1),
        pattern_features
    }, 1);
    
    return combined.to(torch::kCPU);
}

// External function wrapper for use in other files
torch::Tensor GomokuGPUAttackDefense_compute_gomoku_threats(const torch::Tensor& boards, int player) {
    return GomokuGPUAttackDefense::compute_gomoku_threats(boards, player);
}

} // namespace utils
} // namespace alphazero