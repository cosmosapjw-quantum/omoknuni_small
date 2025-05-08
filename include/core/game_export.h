// include/core/game_export.h
#ifndef ALPHAZERO_CORE_GAME_EXPORT_H
#define ALPHAZERO_CORE_GAME_EXPORT_H

#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace core {

// Define a game creator function type
using GameCreator = std::function<std::unique_ptr<IGameState>()>;

/**
 * @brief Registry for game implementations
 * 
 * This singleton class tracks all registered game implementations
 * and allows creating instances of them.
 */
class ALPHAZERO_API GameRegistry {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the instance
     */
    static GameRegistry& instance();
    
    /**
     * @brief Register a game implementation
     * 
     * @param type Game type
     * @param creator Function to create instances
     */
    void registerGame(GameType type, GameCreator creator);
    
    /**
     * @brief Create a game instance
     * 
     * @param type Game type
     * @return Unique pointer to the game
     */
    std::unique_ptr<IGameState> createGame(GameType type) const;
    
    /**
     * @brief Check if a game type is registered
     * 
     * @param type Game type
     * @return true if registered, false otherwise
     */
    bool isRegistered(GameType type) const;
    
    /**
     * @brief Get all registered game types
     * 
     * @return Vector of game types
     */
    std::vector<GameType> getRegisteredGames() const;
    
private:
    GameRegistry() = default;  // Private constructor for singleton
    std::unordered_map<GameType, GameCreator> creators_;
};

/**
 * @brief Factory for creating game instances
 */
class ALPHAZERO_API GameFactory {
public:
    /**
     * @brief Create a game instance
     * 
     * @param type Game type
     * @return Unique pointer to the game
     */
    static std::unique_ptr<IGameState> createGame(GameType type);
    
    /**
     * @brief Create a game from a string representation of moves
     * 
     * This will use the game's stringToAction method to reconstruct
     * the game state from a sequence of moves.
     * 
     * @param type Game type
     * @param moves String representation of moves (space-separated)
     * @return Unique pointer to the game
     */
    static std::unique_ptr<IGameState> createGameFromMoves(GameType type, const std::string& moves);
};

/**
 * @brief Serializer for game states
 */
class ALPHAZERO_API GameSerializer {
public:
    /**
     * @brief Save a game state to a file
     * 
     * @param game Game state to save
     * @param filename Filename to save to
     * @return true if successful, false otherwise
     */
    static bool saveGame(const IGameState& game, const std::string& filename);
    
    /**
     * @brief Load a game state from a file
     * 
     * @param filename Filename to load from
     * @return Unique pointer to the loaded game state
     */
    static std::unique_ptr<IGameState> loadGame(const std::string& filename);
    
    /**
     * @brief Serialize a game state to a string
     * 
     * @param game Game state to serialize
     * @return Serialized string
     */
    static std::string serializeGame(const IGameState& game);
    
    /**
     * @brief Deserialize a game state from a string
     * 
     * @param serialized Serialized string
     * @return Unique pointer to the deserialized game state
     */
    static std::unique_ptr<IGameState> deserializeGame(const std::string& serialized);
};

/**
 * @brief Automatic game registrar
 * 
 * Use this to automatically register a game implementation.
 * Example:
 *   static GameRegistrar<MyGame> registrar(GameType::MY_GAME);
 */
template<typename GameStateType>
class GameRegistrar {
public:
    GameRegistrar(core::GameType type) {
        GameRegistry::instance().registerGame(type, []() {
            return std::make_unique<GameStateType>();
        });
    }
};

/**
 * @brief Convert a game type to string
 * 
 * @param type Game type
 * @return String representation
 */
std::string gameTypeToString(GameType type);

/**
 * @brief Convert a string to game type
 * 
 * @param str String representation
 * @return GameType (GameType::UNKNOWN if not recognized)
 */
GameType stringToGameType(const std::string& str);

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_GAME_EXPORT_H