// src/mcts/evaluation_types.cpp
#include "mcts/evaluation_types.h"

namespace alphazero {
namespace mcts {

EvaluationRequest::EvaluationRequest(MCTSNode* n, std::unique_ptr<core::IGameState> s, int action_size)
    : node(n), state(std::move(s)), action_space_size(action_size) {
}

EvaluationRequest::EvaluationRequest() noexcept
    : node(nullptr), state(nullptr), promise(), action_space_size(10) {
    // Default constructor initializes to a safe empty state.
}

EvaluationRequest::EvaluationRequest(EvaluationRequest&& other) noexcept
    : node(other.node),
      state(std::move(other.state)),
      promise(std::move(other.promise)),
      action_space_size(other.action_space_size) {
    
    // We don't nullify other.node because:
    // 1. It doesn't represent ownership, just a reference
    // 2. Other threads may still need this pointer for the same evaluation
}

EvaluationRequest& EvaluationRequest::operator=(EvaluationRequest&& other) noexcept {
    if (this != &other) {
        node = other.node;
        state = std::move(other.state);
        promise = std::move(other.promise);
        action_space_size = other.action_space_size;
        
        // We don't nullify other.node because:
        // 1. It doesn't represent ownership, just a reference
        // 2. Other threads may still need this pointer for the same evaluation
    }
    return *this;
}

} // namespace mcts
} // namespace alphazero