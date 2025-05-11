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

class MCTSNode;

struct ALPHAZERO_API EvaluationRequest {
    MCTSNode* node;
    std::unique_ptr<core::IGameState> state;
    std::promise<NetworkOutput> promise;
    
    EvaluationRequest(MCTSNode* n, std::unique_ptr<core::IGameState> s);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_EVALUATION_TYPES_H