#include "utils/attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <torch/torch.h>
#include <memory>
#include <mutex>

namespace alphazero {
namespace utils {

// Forward declarations of GPU implementations
torch::Tensor computeGomokuAttackDefenseGPUBatch(const torch::Tensor& boards);
torch::Tensor computeChessAttackDefenseGPUBatch(const torch::Tensor& boards);
torch::Tensor computeGoAttackDefenseGPUBatch(const torch::Tensor& boards);

class AttackDefenseGPUManager {
private:
    static std::unique_ptr<AttackDefenseGPUManager> instance_;
    static std::mutex instance_mutex_;
    
    bool gpu_enabled_;
    torch::Device device_;
    
    // Batch processing queues
    static constexpr size_t MAX_BATCH_SIZE = 128;
    static constexpr int BATCH_TIMEOUT_MS = 10;
    
public:
    AttackDefenseGPUManager() 
        : gpu_enabled_(torch::cuda::is_available()),
          device_(gpu_enabled_ ? torch::kCUDA : torch::kCPU) {
        if (gpu_enabled_) {
            std::cout << "[AttackDefenseGPU] GPU acceleration enabled for attack/defense computation" << std::endl;
        } else {
            std::cout << "[AttackDefenseGPU] GPU not available, using CPU fallback" << std::endl;
        }
    }
    
    static AttackDefenseGPUManager& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::make_unique<AttackDefenseGPUManager>();
        }
        return *instance_;
    }
    
    bool isGPUEnabled() const { return gpu_enabled_; }
    
    // Batch computation for Gomoku
    std::vector<std::pair<torch::Tensor, torch::Tensor>> 
    computeGomokuBatch(const std::vector<const games::gomoku::GomokuState*>& states) {
        if (!gpu_enabled_ || states.empty()) {
            return {};
        }
        
        try {
            // Convert states to tensor batch
            int batch_size = states.size();
            int board_size = states[0]->getBoardSize();
            auto boards = torch::zeros({batch_size, board_size, board_size}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(device_));
            
            // Fill board tensor
            for (int b = 0; b < batch_size; ++b) {
                auto repr = states[b]->getTensorRepresentation();
                for (int r = 0; r < board_size; ++r) {
                    for (int c = 0; c < board_size; ++c) {
                        if (repr[0][r][c] > 0) {
                            boards[b][r][c] = 1;  // Player 1
                        } else if (repr[1][r][c] > 0) {
                            boards[b][r][c] = 2;  // Player 2
                        }
                    }
                }
            }
            
            // Compute on GPU
            auto result = computeGomokuAttackDefenseGPUBatch(boards);
            
            // Split result into attack/defense pairs
            std::vector<std::pair<torch::Tensor, torch::Tensor>> output;
            for (int b = 0; b < batch_size; ++b) {
                auto attack = result[b][0].to(torch::kCPU);
                auto defense = result[b][1].to(torch::kCPU);
                output.push_back({attack, defense});
            }
            
            return output;
        } catch (const std::exception& e) {
            std::cerr << "[AttackDefenseGPU] Error in Gomoku batch computation: " << e.what() << std::endl;
            return {};
        }
    }
    
    // Batch computation for Chess
    std::vector<std::pair<torch::Tensor, torch::Tensor>> 
    computeChessBatch(const std::vector<const games::chess::ChessState*>& states) {
        if (!gpu_enabled_ || states.empty()) {
            return {};
        }
        
        try {
            // Convert states to tensor batch
            int batch_size = states.size();
            auto boards = torch::zeros({batch_size, 8, 8, 12}, 
                                      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            
            // Fill board tensor with piece information
            for (int b = 0; b < batch_size; ++b) {
                // TODO: Extract piece positions from ChessState
                // This requires accessing the internal board representation
            }
            
            // Compute on GPU
            auto result = computeChessAttackDefenseGPUBatch(boards);
            
            // Split result
            std::vector<std::pair<torch::Tensor, torch::Tensor>> output;
            for (int b = 0; b < batch_size; ++b) {
                auto attack = result[b][0].to(torch::kCPU);
                auto defense = result[b][1].to(torch::kCPU);
                output.push_back({attack, defense});
            }
            
            return output;
        } catch (const std::exception& e) {
            std::cerr << "[AttackDefenseGPU] Error in Chess batch computation: " << e.what() << std::endl;
            return {};
        }
    }
    
    // Batch computation for Go
    std::vector<torch::Tensor> 
    computeGoBatch(const std::vector<const games::go::GoState*>& states) {
        if (!gpu_enabled_ || states.empty()) {
            return {};
        }
        
        try {
            // Convert states to tensor batch
            int batch_size = states.size();
            int board_size = states[0]->getBoardSize();
            auto boards = torch::zeros({batch_size, board_size, board_size}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(device_));
            
            // Fill board tensor
            for (int b = 0; b < batch_size; ++b) {
                auto repr = states[b]->getTensorRepresentation();
                for (int r = 0; r < board_size; ++r) {
                    for (int c = 0; c < board_size; ++c) {
                        if (repr[0][r][c] > 0) {
                            boards[b][r][c] = 1;  // Black
                        } else if (repr[1][r][c] > 0) {
                            boards[b][r][c] = 2;  // White
                        }
                    }
                }
            }
            
            // Compute on GPU
            auto result = computeGoAttackDefenseGPUBatch(boards);
            
            // Convert to CPU
            std::vector<torch::Tensor> output;
            for (int b = 0; b < batch_size; ++b) {
                output.push_back(result[b].to(torch::kCPU));
            }
            
            return output;
        } catch (const std::exception& e) {
            std::cerr << "[AttackDefenseGPU] Error in Go batch computation: " << e.what() << std::endl;
            return {};
        }
    }
};

// Static member definitions
std::unique_ptr<AttackDefenseGPUManager> AttackDefenseGPUManager::instance_ = nullptr;
std::mutex AttackDefenseGPUManager::instance_mutex_;

// Implementation of GPU functions with actual computations
torch::Tensor computeGomokuAttackDefenseGPUBatch(const torch::Tensor& boards) {
    // boards shape: [batch_size, board_size, board_size]
    const auto batch_size = boards.size(0);
    const auto board_size = boards.size(1);
    
    // Create output tensor for attack and defense planes
    auto output = torch::zeros({batch_size, 2, board_size, board_size}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(boards.device()));
    
    // Pattern detection kernels for Gomoku
    // Horizontal kernel for pattern detection
    auto h_kernel = torch::ones({1, 1, 1, 5}, boards.options());
    
    // Vertical kernel
    auto v_kernel = torch::ones({1, 1, 5, 1}, boards.options());
    
    // Diagonal kernels
    auto d1_kernel = torch::eye(5, boards.options()).unsqueeze(0).unsqueeze(0);
    auto d2_kernel = torch::flip(d1_kernel, {2});
    
    // For each player
    for (int player = 1; player <= 2; ++player) {
        // Create player mask
        auto player_mask = (boards == player).to(torch::kFloat32).unsqueeze(1);
        auto opponent_mask = (boards == (3 - player)).to(torch::kFloat32).unsqueeze(1);
        auto empty_mask = (boards == 0).to(torch::kFloat32).unsqueeze(1);
        
        // Detect patterns using convolutions
        auto h_patterns = torch::nn::functional::conv2d(player_mask, h_kernel, 
            torch::nn::functional::Conv2dFuncOptions().padding({0, 2}));
        auto v_patterns = torch::nn::functional::conv2d(player_mask, v_kernel,
            torch::nn::functional::Conv2dFuncOptions().padding({2, 0}));
        auto d1_patterns = torch::nn::functional::conv2d(player_mask, d1_kernel,
            torch::nn::functional::Conv2dFuncOptions().padding(2));
        auto d2_patterns = torch::nn::functional::conv2d(player_mask, d2_kernel,
            torch::nn::functional::Conv2dFuncOptions().padding(2));
        
        // Combine all patterns
        auto all_patterns = h_patterns + v_patterns + d1_patterns + d2_patterns;
        
        // Attack scores: patterns of current player
        if (player == 1) {
            output.index({torch::indexing::Slice(), 0}) += all_patterns.squeeze(1);
        } else {
            // Defense scores: patterns of opponent
            output.index({torch::indexing::Slice(), 1}) += all_patterns.squeeze(1);
        }
    }
    
    // Normalize scores
    output = torch::sigmoid(output * 0.1);
    
    return output;
}

torch::Tensor computeChessAttackDefenseGPUBatch(const torch::Tensor& boards) {
    // boards shape: [batch_size, 8, 8, 12] (12 piece types)
    const auto batch_size = boards.size(0);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, 2, 8, 8}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(boards.device()));
    
    // Simplified chess attack/defense calculation
    // In practice, this would need full chess move generation
    
    // For now, compute simple piece activity scores
    auto piece_values = torch::tensor({1.0, 3.0, 3.0, 5.0, 9.0, 100.0, // White pieces
                                      1.0, 3.0, 3.0, 5.0, 9.0, 100.0}, // Black pieces
                                     boards.options());
    
    // Sum piece values for attack potential
    for (int i = 0; i < 12; ++i) {
        auto piece_layer = boards.index({torch::indexing::Slice(), 
                                        torch::indexing::Slice(), 
                                        torch::indexing::Slice(), i});
        auto value = piece_values[i].item<float>();
        
        // White pieces contribute to attack for white, defense for black
        if (i < 6) {
            output.index({torch::indexing::Slice(), 0}) += piece_layer * value * 0.1;
        } else {
            output.index({torch::indexing::Slice(), 1}) += piece_layer * value * 0.1;
        }
    }
    
    return output;
}

torch::Tensor computeGoAttackDefenseGPUBatch(const torch::Tensor& boards) {
    // boards shape: [batch_size, board_size, board_size]
    const auto batch_size = boards.size(0);
    const auto board_size = boards.size(1);
    
    // Create output tensor for capture and liberty features
    auto output = torch::zeros({batch_size, 2, board_size, board_size}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(boards.device()));
    
    // Liberty counting kernel (4-connected)
    auto liberty_kernel = torch::tensor({{0, 1, 0}, 
                                        {1, 0, 1}, 
                                        {0, 1, 0}}, boards.options()).to(torch::kFloat32);
    liberty_kernel = liberty_kernel.unsqueeze(0).unsqueeze(0);
    
    // For each color
    for (int color = 1; color <= 2; ++color) {
        // Create masks
        auto stone_mask = (boards == color).to(torch::kFloat32).unsqueeze(1);
        auto empty_mask = (boards == 0).to(torch::kFloat32).unsqueeze(1);
        auto opponent_mask = (boards == (3 - color)).to(torch::kFloat32).unsqueeze(1);
        
        // Count liberties for each position
        auto neighbor_empty = torch::nn::functional::conv2d(empty_mask, liberty_kernel,
            torch::nn::functional::Conv2dFuncOptions().padding(1));
        
        // Positions with few liberties are vulnerable (defense)
        auto low_liberty_score = torch::relu(3.0 - neighbor_empty) * stone_mask;
        
        // Positions that reduce opponent liberties are good attacks
        auto neighbor_opponent = torch::nn::functional::conv2d(opponent_mask, liberty_kernel,
            torch::nn::functional::Conv2dFuncOptions().padding(1));
        auto attack_score = neighbor_opponent * empty_mask;
        
        // Assign to output
        if (color == 1) {
            output.index({torch::indexing::Slice(), 0}) += attack_score.squeeze(1);
            output.index({torch::indexing::Slice(), 1}) += low_liberty_score.squeeze(1);
        } else {
            output.index({torch::indexing::Slice(), 1}) += attack_score.squeeze(1);
            output.index({torch::indexing::Slice(), 0}) += low_liberty_score.squeeze(1);
        }
    }
    
    // Normalize
    output = torch::tanh(output * 0.2);
    
    return output;
}

// External functions defined in gpu_attack_defense_gomoku.cpp and gpu_attack_defense_go.cpp
extern torch::Tensor GomokuGPUAttackDefense_compute_gomoku_threats(const torch::Tensor& boards, int player);
extern torch::Tensor GoGPUAttackDefense_compute_go_features_batch(const torch::Tensor& boards);

} // namespace utils
} // namespace alphazero