#pragma once

#include "utils/attack_defense_module.h"
#include <torch/torch.h>
#include <memory>

namespace alphazero {

// GPU-accelerated attack/defense computation using PyTorch
class GPUAttackDefenseModule {
public:
    GPUAttackDefenseModule(int board_size, torch::Device device)
        : board_size_(board_size), device_(device) {}
    
    virtual ~GPUAttackDefenseModule() = default;
    
    // Main interface for batch computation
    virtual std::pair<torch::Tensor, torch::Tensor> compute_planes_gpu(
        const torch::Tensor& board_batch,  // [B, H, W] tensor
        int current_player
    ) = 0;
    
    // Compute bonuses for specific moves
    virtual std::pair<torch::Tensor, torch::Tensor> compute_bonuses_gpu(
        const torch::Tensor& board_batch,  // [B, H, W] tensor
        const torch::Tensor& chosen_moves, // [B] tensor of move indices
        int current_player
    ) = 0;

protected:
    int board_size_;
    torch::Device device_;
    
    // Pre-computed convolution kernels for pattern detection
    torch::Tensor horizontal_kernels_;
    torch::Tensor vertical_kernels_;
    torch::Tensor diagonal_kernels_;
    torch::Tensor antidiag_kernels_;
    
    // Helper functions
    torch::Tensor create_player_mask(const torch::Tensor& board, int player) {
        return (board == player).to(torch::kFloat32).to(device_);
    }
    
    torch::Tensor create_empty_mask(const torch::Tensor& board) {
        return (board == 0).to(torch::kFloat32).to(device_);
    }
    
    // To be implemented by derived classes
    virtual torch::Tensor count_threats_for_player(
        const torch::Tensor& board_batch,
        int player
    ) = 0;
};

// Gomoku-specific GPU implementation
class GomokuGPUAttackDefense : public GPUAttackDefenseModule {
public:
    GomokuGPUAttackDefense(int board_size, torch::Device device);
    
    // Override with Gomoku-specific optimizations
    torch::Tensor compute_gomoku_threats(const torch::Tensor& board_batch, int player);
    
private:
    // Gomoku-specific pattern kernels
    void initialize_gomoku_patterns();
};

// Chess-specific GPU implementation
class ChessGPUAttackDefense : public GPUAttackDefenseModule {
public:
    ChessGPUAttackDefense(torch::Device device);
    
    // Chess attack map computation using bitboards on GPU
    torch::Tensor compute_chess_attacks(const torch::Tensor& board_batch);
    
private:
    // Pre-computed attack masks for each piece type
    torch::Tensor knight_attacks_;
    torch::Tensor king_attacks_;
    torch::Tensor sliding_ray_masks_;
    
    void initialize_chess_patterns();
    torch::Tensor compute_sliding_attacks(
        const torch::Tensor& board,
        const torch::Tensor& piece_positions,
        const torch::Tensor& ray_masks
    );
};

// Go-specific GPU implementation
class GoGPUAttackDefense : public GPUAttackDefenseModule {
public:
    GoGPUAttackDefense(int board_size, torch::Device device);
    
    // Liberty counting and capture detection on GPU
    torch::Tensor compute_go_liberties(const torch::Tensor& board_batch);
    torch::Tensor detect_captures(const torch::Tensor& board_batch, int player);
    
private:
    // Convolution kernels for liberty counting
    torch::Tensor liberty_kernel_;
    torch::Tensor group_detection_kernels_;
    
    void initialize_go_patterns();
    
    // GPU-accelerated flood fill for group detection
    torch::Tensor gpu_flood_fill_groups(const torch::Tensor& board, int player);
};

// Factory function
std::unique_ptr<GPUAttackDefenseModule> createGPUAttackDefenseModule(
    core::GameType game_type, int board_size, torch::Device device);

} // namespace alphazero