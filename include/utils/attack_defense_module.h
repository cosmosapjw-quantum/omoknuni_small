#pragma once

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "core/export_macros.h"
#include "utils/hash_specializations.h"
#include "core/igamestate.h"

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

namespace alphazero {

// Base class for attack/defense calculation
class ALPHAZERO_API AttackDefenseModule {
public:
    AttackDefenseModule(int board_size) : board_size_(board_size) {}
    virtual ~AttackDefenseModule() = default;
    
    // Calculate attack and defense bonuses
    virtual std::pair<std::vector<float>, std::vector<float>> compute_bonuses(
        const std::vector<std::vector<std::vector<int>>>& board_batch,
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch) = 0;
    
    // Calculate attack and defense planes for neural network input
    virtual std::pair<std::vector<std::vector<std::vector<float>>>, 
                      std::vector<std::vector<std::vector<float>>>> 
    compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) = 0;

protected:
    int board_size_;
};

// Gomoku-specific implementation
class ALPHAZERO_API GomokuAttackDefenseModule : public AttackDefenseModule {
public:
    GomokuAttackDefenseModule(int board_size);
    
    std::pair<std::vector<float>, std::vector<float>> compute_bonuses(
        const std::vector<std::vector<std::vector<int>>>& board_batch,
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch) override;
    
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>> 
    compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) override;

private:
    // Internal implementations
    std::vector<float> compute_attack_bonus(
        const std::vector<std::vector<std::vector<int>>>& board_batch, 
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch);
    
    std::vector<float> compute_defense_bonus(
        const std::vector<std::vector<std::vector<int>>>& board_batch, 
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch);
    
    std::vector<float> count_threats_for_color(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids);
    
    std::vector<float> count_open_threats_horiz_vert(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids,
        int window_length,
        int required_sum);
    
    std::vector<float> count_open_threats_diagonals(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids,
        int window_length,
        int required_sum);
    
    std::vector<float> count_1d_patterns(
        const std::vector<std::vector<std::vector<float>>>& opp_mask,
        const std::vector<std::vector<std::vector<float>>>& empty_mask,
        int window_length,
        int required_sum);
    
    // Helper functions
    std::vector<std::vector<std::vector<float>>> create_mask(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& player_ids);
    
    std::vector<std::vector<std::vector<float>>> create_empty_mask(
        const std::vector<std::vector<std::vector<int>>>& boards);
    
    std::vector<std::vector<std::vector<float>>> transpose(
        const std::vector<std::vector<std::vector<float>>>& mask);
};

// Chess-specific implementation
class ALPHAZERO_API ChessAttackDefenseModule : public AttackDefenseModule {
public:
    ChessAttackDefenseModule();
    
    std::pair<std::vector<float>, std::vector<float>> compute_bonuses(
        const std::vector<std::vector<std::vector<int>>>& board_batch,
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch) override;
        
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>> 
    compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) override;

private:
    // Piece values for attack/defense calculation
    static constexpr float PAWN_VALUE = 1.0f;
    static constexpr float KNIGHT_VALUE = 3.0f;
    static constexpr float BISHOP_VALUE = 3.0f;
    static constexpr float ROOK_VALUE = 5.0f;
    static constexpr float QUEEN_VALUE = 9.0f;
    static constexpr float KING_VALUE = 100.0f;
    
    float getPieceValue(int piece_type) const;
    std::vector<std::pair<int,int>> getAttackedSquares(
        const std::vector<std::vector<int>>& board, 
        int from_row, int from_col, int piece_type) const;
};

// Go-specific implementation  
class ALPHAZERO_API GoAttackDefenseModule : public AttackDefenseModule {
public:
    GoAttackDefenseModule(int board_size);
    
    std::pair<std::vector<float>, std::vector<float>> compute_bonuses(
        const std::vector<std::vector<std::vector<int>>>& board_batch,
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch) override;
        
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>> 
    compute_planes(const std::vector<std::unique_ptr<core::IGameState>>& states) override;

private:
    // Weights for different components
    static constexpr float CAPTURE_WEIGHT = 1.0f;
    static constexpr float ATARI_WEIGHT = 0.7f;
    static constexpr float LIBERTY_WEIGHT = 0.3f;
    static constexpr float EYE_WEIGHT = 0.5f;
    
    struct Group {
        std::vector<std::pair<int,int>> stones;
        int liberties;
        int player;
    };
    
    std::vector<Group> findGroups(const std::vector<std::vector<int>>& board) const;
    int countLiberties(const std::vector<std::vector<int>>& board, 
                      const std::vector<std::pair<int,int>>& group) const;
    bool wouldCapture(const std::vector<std::vector<int>>& board, 
                     int row, int col, int player) const;
    bool createsAtari(const std::vector<std::vector<int>>& board, 
                     int row, int col, int player) const;
    float evaluateEyePotential(const std::vector<std::vector<int>>& board, 
                              int row, int col, int player) const;
};

// Factory function to create appropriate module based on game type
ALPHAZERO_API std::unique_ptr<AttackDefenseModule> createAttackDefenseModule(
    core::GameType game_type, int board_size);

} // namespace alphazero

// Forward declarations for GPU functions
namespace alphazero {
namespace games {
namespace gomoku { class GomokuState; }
namespace chess { class ChessState; }
namespace go { class GoState; }
}

namespace utils {

#ifdef WITH_TORCH
// GPU-accelerated attack/defense computation functions
class ALPHAZERO_API AttackDefenseModule {
public:
    // Check if GPU is available
    static bool isGPUAvailable();
    
    // GPU versions only - CPU versions use the existing compute_planes methods
    static torch::Tensor computeGomokuAttackDefenseGPU(
        const std::vector<const games::gomoku::GomokuState*>& states);
    
    static torch::Tensor computeChessAttackDefenseGPU(
        const std::vector<const games::chess::ChessState*>& states);
    
    static torch::Tensor computeGoAttackDefenseGPU(
        const std::vector<const games::go::GoState*>& states);
};
#endif

} // namespace utils
} // namespace alphazero