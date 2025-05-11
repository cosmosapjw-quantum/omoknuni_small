// src/mcts/evaluation_types.cpp
#include "mcts/evaluation_types.h"

namespace alphazero {
namespace mcts {

EvaluationRequest::EvaluationRequest(MCTSNode* n, std::unique_ptr<core::IGameState> s)
    : node(n), state(std::move(s)) {
}

} // namespace mcts
} // namespace alphazero