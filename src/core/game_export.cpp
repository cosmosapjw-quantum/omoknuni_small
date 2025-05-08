// src/core/game_export.cpp
#include "core/game_export.h"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <limits>

namespace alphazero {
namespace core {

GameRegistry& GameRegistry::instance() {
    static GameRegistry instance;
    return instance;
}

void GameRegistry::registerGame(GameType type, GameCreator creator) {
    creators_[type] = std::move(creator);
}

std::unique_ptr<IGameState> GameRegistry::createGame(GameType type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
        throw std::runtime_error("Game type not registered: " + 
                                 gameTypeToString(type));
    }
    return it->second();
}

bool GameRegistry::isRegistered(GameType type) const {
    return creators_.find(type) != creators_.end();
}

std::vector<GameType> GameRegistry::getRegisteredGames() const {
    std::vector<GameType> result;
    result.reserve(creators_.size());
    for (const auto& pair : creators_) {
        result.push_back(pair.first);
    }
    return result;
}

std::unique_ptr<IGameState> GameFactory::createGame(GameType type) {
    return GameRegistry::instance().createGame(type);
}

std::unique_ptr<IGameState> GameFactory::createGameFromMoves(GameType type, const std::string& moves) {
    auto game = createGame(type);
    
    std::istringstream ss(moves);
    std::string moveStr;
    
    while (ss >> moveStr) {
        auto action = game->stringToAction(moveStr);
        if (!action) {
            throw std::runtime_error("Invalid move: " + moveStr);
        }
        
        game->makeMove(*action);
    }
    
    return game;
}

bool GameSerializer::saveGame(const IGameState& game, const std::string& filename) {
    try {
        std::ofstream out(filename);
        if (!out) {
            return false;
        }
        
        out << serializeGame(game);
        return !out.fail();
    } catch (...) {
        return false;
    }
}

std::unique_ptr<IGameState> GameSerializer::loadGame(const std::string& filename) {
    try {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        std::string serialized((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
        
        return deserializeGame(serialized);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load game: " + std::string(e.what()));
    }
}

std::string GameSerializer::serializeGame(const IGameState& game) {
    std::ostringstream ss;
    
    // Write the game type
    ss << gameTypeToString(game.getGameType()) << std::endl;
    
    // Write the move history
    for (int move : game.getMoveHistory()) {
        ss << game.actionToString(move) << " ";
    }
    ss << std::endl;
    
    return ss.str();
}

std::unique_ptr<IGameState> GameSerializer::deserializeGame(const std::string& serialized) {
    std::istringstream ss(serialized);
    
    // Read the game type
    std::string typeStr;
    if (!(ss >> typeStr)) {
        throw std::runtime_error("Invalid serialized game: missing type");
    }
    
    GameType type = stringToGameType(typeStr);
    if (type == GameType::UNKNOWN) {
        throw std::runtime_error("Invalid game type: " + typeStr);
    }
    
    // Skip to the next line
    ss.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    // Read the moves
    std::string movesLine;
    std::getline(ss, movesLine);
    
    // Create and initialize the game
    return GameFactory::createGameFromMoves(type, movesLine);
}

std::string gameTypeToString(GameType type) {
    switch (type) {
        case GameType::CHESS:   return "CHESS";
        case GameType::GO:      return "GO";
        case GameType::GOMOKU:  return "GOMOKU";
        case static_cast<GameType>(99): return "TEST_GAME";
        default:                return "UNKNOWN";
    }
}

GameType stringToGameType(const std::string& str) {
    if (str == "CHESS")     return GameType::CHESS;
    if (str == "GO")        return GameType::GO;
    if (str == "GOMOKU")    return GameType::GOMOKU;
    if (str == "TEST_GAME") return static_cast<GameType>(99);
    return GameType::UNKNOWN;
}

} // namespace core
} // namespace alphazero