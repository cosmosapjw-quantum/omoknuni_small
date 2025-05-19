// include/mcts/evaluation_types.h
#ifndef ALPHAZERO_MCTS_EVALUATION_TYPES_H
#define ALPHAZERO_MCTS_EVALUATION_TYPES_H

#include <vector>
#include <future>
#include <memory>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

struct ALPHAZERO_API NetworkOutput {
    std::vector<float> policy;
    float value;
};

// Evaluation result to be sent back to engines
struct ALPHAZERO_API EvaluationResult {
    NetworkOutput output;
    int batch_id;
    int request_id;
};

class MCTSNode;

struct ALPHAZERO_API EvaluationRequest {
    std::shared_ptr<MCTSNode> node;
    std::unique_ptr<core::IGameState> state;
    std::promise<NetworkOutput> promise;
    int action_space_size; // Store action space size for safe fallback
    
    EvaluationRequest() noexcept; // Default constructor
    EvaluationRequest(std::shared_ptr<MCTSNode> n, std::unique_ptr<core::IGameState> s, int action_size = 10);
    
    // Add proper move constructor
    EvaluationRequest(EvaluationRequest&& other) noexcept;
    
    // Add proper move assignment operator
    EvaluationRequest& operator=(EvaluationRequest&& other) noexcept;
    
    // Delete copy constructor and assignment
    EvaluationRequest(const EvaluationRequest&) = delete;
    EvaluationRequest& operator=(const EvaluationRequest&) = delete;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_EVALUATION_TYPES_H