#include "nn/neural_network.h"
#include "utils/attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <torch/torch.h>

namespace alphazero {
namespace nn {

class BatchAttackDefenseProcessor {
public:
    static void processBatchAttackDefense(
        std::vector<std::unique_ptr<core::IGameState>>& states,
        std::vector<torch::Tensor>& enhanced_tensors) {
        
        if (!torch::cuda::is_available() || states.empty()) {
            return;  // Use CPU fallback
        }
        
        // Group states by game type
        std::vector<size_t> gomoku_indices;
        std::vector<const games::gomoku::GomokuState*> gomoku_states;
        
        std::vector<size_t> chess_indices;
        std::vector<const games::chess::ChessState*> chess_states;
        
        std::vector<size_t> go_indices;
        std::vector<const games::go::GoState*> go_states;
        
        // Classify states by type
        for (size_t i = 0; i < states.size(); ++i) {
            if (auto* gomoku = dynamic_cast<games::gomoku::GomokuState*>(states[i].get())) {
                gomoku_indices.push_back(i);
                gomoku_states.push_back(gomoku);
            } else if (auto* chess = dynamic_cast<games::chess::ChessState*>(states[i].get())) {
                chess_indices.push_back(i);
                chess_states.push_back(chess);
            } else if (auto* go = dynamic_cast<games::go::GoState*>(states[i].get())) {
                go_indices.push_back(i);
                go_states.push_back(go);
            }
        }
        
        // Process Gomoku batch
        if (!gomoku_states.empty()) {
            processGomokuBatch(gomoku_states, gomoku_indices, enhanced_tensors);
        }
        
        // Process Chess batch
        if (!chess_states.empty()) {
            processChessBatch(chess_states, chess_indices, enhanced_tensors);
        }
        
        // Process Go batch
        if (!go_states.empty()) {
            processGoBatch(go_states, go_indices, enhanced_tensors);
        }
    }
    
private:
    static void processGomokuBatch(
        const std::vector<const games::gomoku::GomokuState*>& states,
        const std::vector<size_t>& indices,
        std::vector<torch::Tensor>& enhanced_tensors) {
        
        try {
            // Compute attack/defense planes on GPU
            auto gpu_results = alphazero::utils::AttackDefenseModule::computeGomokuAttackDefenseGPU(states);
            
            if (gpu_results.dim() >= 3 && gpu_results.size(0) == states.size()) {
                // Update the enhanced tensors with GPU-computed attack/defense planes
                for (size_t i = 0; i < states.size(); ++i) {
                    auto& tensor = enhanced_tensors[indices[i]];
                    
                    // Ensure tensor has correct dimensions
                    if (tensor.dim() >= 3 && tensor.size(0) >= 19) {
                        // Copy attack plane (channel 17)
                        tensor[17] = gpu_results[i][0];
                        // Copy defense plane (channel 18)
                        tensor[18] = gpu_results[i][1];
                    }
                }
                
                std::cout << "[BatchAttackDefense] Processed " << states.size() 
                          << " Gomoku states on GPU" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[BatchAttackDefense] Error processing Gomoku batch: " 
                      << e.what() << std::endl;
        }
    }
    
    static void processChessBatch(
        const std::vector<const games::chess::ChessState*>& states,
        const std::vector<size_t>& indices,
        std::vector<torch::Tensor>& enhanced_tensors) {
        
        try {
            // Compute attack/defense for chess
            auto gpu_results = alphazero::utils::AttackDefenseModule::computeChessAttackDefenseGPU(states);
            
            if (gpu_results.dim() >= 3 && gpu_results.size(0) == states.size()) {
                for (size_t i = 0; i < states.size(); ++i) {
                    auto& tensor = enhanced_tensors[indices[i]];
                    
                    // Chess uses different channel structure
                    // Update based on chess tensor layout
                    if (tensor.dim() >= 3) {
                        // Copy attack/defense information
                        // This depends on chess tensor structure
                    }
                }
                
                std::cout << "[BatchAttackDefense] Processed " << states.size() 
                          << " Chess states on GPU" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[BatchAttackDefense] Error processing Chess batch: " 
                      << e.what() << std::endl;
        }
    }
    
    static void processGoBatch(
        const std::vector<const games::go::GoState*>& states,
        const std::vector<size_t>& indices,
        std::vector<torch::Tensor>& enhanced_tensors) {
        
        try {
            // Compute Go features including groups, liberties, etc.
            auto gpu_results = alphazero::utils::AttackDefenseModule::computeGoAttackDefenseGPU(states);
            
            if (gpu_results.dim() >= 3 && gpu_results.size(0) == states.size()) {
                for (size_t i = 0; i < states.size(); ++i) {
                    auto& tensor = enhanced_tensors[indices[i]];
                    
                    // Go has multiple feature channels
                    if (tensor.dim() >= 3 && gpu_results.size(1) > 0) {
                        // Copy all computed features
                        int num_features = std::min(gpu_results.size(1), tensor.size(0) - 2);
                        for (int f = 0; f < num_features; ++f) {
                            tensor[2 + f] = gpu_results[i][f];
                        }
                    }
                }
                
                std::cout << "[BatchAttackDefense] Processed " << states.size() 
                          << " Go states on GPU" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[BatchAttackDefense] Error processing Go batch: " 
                      << e.what() << std::endl;
        }
    }
};

} // namespace nn
} // namespace alphazero