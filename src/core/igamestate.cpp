// src/core/igamestate.cpp
#include "core/igamestate.h"

namespace alphazero {
namespace core {

IGameState::IGameState(GameType type) : type_(type) {
}

GameType IGameState::getGameType() const {
    return type_;
}

} // namespace core
} // namespace alphazero