// include/core/igamestate.h
#ifndef ALPHAZERO_CORE_IGAMESTATE_H
#define ALPHAZERO_CORE_IGAMESTATE_H

#include <vector>
#include <string>
#include <optional>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include "core/export_macros.h"
#include "core/illegal_move_exception.h"

namespace alphazero {
namespace core {

// Game types
enum class GameType {
    UNKNOWN,
    CHESS,
    GO,
    GOMOKU
    // Can be extended with more game types
};

// Game results
enum class GameResult {
    ONGOING,
    WIN_PLAYER1,
    WIN_PLAYER2,
    DRAW,
    NO_RESULT  // For Japanese rules: triple ko, quadruple ko, eternal life
};

/**
 * @brief Interface for game state
 * 
 * This interface defines the operations that all game implementations
 * must provide. It's used by the MCTS algorithm to interact with
 * different games in a uniform way.
 */
class ALPHAZERO_API IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param type Game type
     */
    explicit IGameState(GameType type);

    /**
     * @brief Virtual destructor
     */
    virtual ~IGameState() = default;

    /**
     * @brief Get all legal moves in the current state
     * 
     * @return Vector of legal actions
     */
    virtual std::vector<int> getLegalMoves() const = 0;

    /**
     * @brief Check if a specific move is legal
     * 
     * @param action The action to check
     * @return true if legal, false otherwise
     */
    virtual bool isLegalMove(int action) const = 0;

    /**
     * @brief Execute a move
     * 
     * Updates the game state by applying the specified action.
     * The action is assumed to be legal.
     * 
     * @param action The action to execute
     * @throws std::runtime_error if the action is illegal
     */
    virtual void makeMove(int action) = 0;

    /**
     * @brief Undo the last move
     * 
     * Reverts the game state to what it was before the last move.
     * 
     * @return true if a move was undone, false if no moves to undo
     */
    virtual bool undoMove() = 0;

    /**
     * @brief Check if the game state is terminal
     * 
     * A terminal state is one where the game is over (win, loss, draw).
     * 
     * @return true if terminal, false otherwise
     */
    virtual bool isTerminal() const = 0;

    /**
     * @brief Get the result of the game
     * 
     * Should return ONGOING if the game is not terminal.
     * 
     * @return Game result
     */
    virtual GameResult getGameResult() const = 0;

    /**
     * @brief Get the current player
     * 
     * @return Current player (1 for player 1, 2 for player 2)
     */
    virtual int getCurrentPlayer() const = 0;

    /**
     * @brief Get the board size
     * 
     * @return Board size (typically width/height)
     */
    virtual int getBoardSize() const = 0;

    /**
     * @brief Get the action space size
     * 
     * The total number of possible actions, including illegal ones.
     * 
     * @return Size of the action space
     */
    virtual int getActionSpaceSize() const = 0;

    /**
     * @brief Get tensor representation for neural network
     * 
     * Creates a 3D tensor representation of the game state suitable
     * for input to a neural network. The format is:
     * [num_planes][height][width]
     * 
     * @return 3D tensor with basic features
     */
    virtual std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const = 0;

    /**
     * @brief Get enhanced tensor representation with additional features
     * 
     * Similar to getTensorRepresentation, but with additional planes
     * for features like move history, legal moves, game-specific data.
     * 
     * @return 3D tensor with enhanced features
     */
    virtual std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const = 0;

    /**
     * @brief Get hash for transposition table
     * 
     * Returns a Zobrist hash of the current state for efficient
     * lookups in transposition tables.
     * 
     * @return 64-bit hash
     */
    virtual uint64_t getHash() const = 0;

    /**
     * @brief Clone the current state
     * 
     * Creates a deep copy of the current game state.
     * 
     * @return Unique pointer to a new copy
     */
    virtual std::unique_ptr<IGameState> clone() const = 0;
    
    /**
     * @brief Copy the state from another game state instance
     * 
     * Copies all relevant fields from the source state to this state.
     * This allows reusing existing state objects from a pool rather than
     * allocating new ones. The source and destination must be the same game type.
     * 
     * @param source The source state to copy from
     * @throws std::runtime_error if the game types don't match
     */
    virtual void copyFrom(const IGameState& source) = 0;

    /**
     * @brief Convert action to string representation
     * 
     * Useful for human-readable move notation (e.g., "e2e4" in chess).
     * 
     * @param action The action to convert
     * @return String representation
     */
    virtual std::string actionToString(int action) const = 0;

    /**
     * @brief Convert string representation to action
     * 
     * The inverse of actionToString.
     * 
     * @param moveStr String representation
     * @return Optional action (nullopt if invalid)
     */
    virtual std::optional<int> stringToAction(const std::string& moveStr) const = 0;

    /**
     * @brief Get string representation of the state
     * 
     * Creates a human-readable representation of the entire game state.
     * 
     * @return String representation
     */
    virtual std::string toString() const = 0;

    /**
     * @brief Check equality with another game state
     * 
     * Two states are equal if they represent the same game position.
     * 
     * @param other The other game state
     * @return true if equal, false otherwise
     */
    virtual bool equals(const IGameState& other) const = 0;

    /**
     * @brief Get the history of moves
     * 
     * Returns the sequence of actions that led to the current state.
     * 
     * @return Vector of actions
     */
    virtual std::vector<int> getMoveHistory() const = 0;

    /**
     * @brief Validate the game state for consistency
     *
     * Checks if the current state is valid according to game rules.
     *
     * @return true if valid, false otherwise
     */
    virtual bool validate() const = 0;

    /**
     * @brief Estimate memory usage of this game state
     *
     * Provides a rough estimate of the memory used by this game state
     * in bytes. Used for memory tracking and debugging.
     *
     * @return Estimated memory usage in bytes
     */
    virtual size_t estimateMemoryUsage() const {
        // Default implementation - derived classes should override
        // for more accurate accounting
        return sizeof(*this) +
               getMoveHistory().capacity() * sizeof(int);
    }

    /**
     * @brief Get the game type
     *
     * @return Game type
     */
    GameType getGameType() const;

protected:
    GameType type_;
};

// Free functions
ALPHAZERO_API std::string gameTypeToString(GameType type);
ALPHAZERO_API GameType stringToGameType(const std::string& str);

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_IGAMESTATE_H