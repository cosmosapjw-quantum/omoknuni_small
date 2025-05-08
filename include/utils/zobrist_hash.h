// zobrist_hash.h
#ifndef ZOBRIST_HASH_H
#define ZOBRIST_HASH_H

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <random>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace core {

/**
 * @brief Zobrist hashing for game positions
 * 
 * Implementation of Zobrist hashing to generate unique hash values
 * for board positions, supporting efficient transposition table lookups.
 */
class ALPHAZERO_API ZobristHash {
public:
    /**
     * @brief Constructor for board games
     * 
     * @param boardSize Size of the game board
     * @param numPieceTypes Number of different piece types
     * @param numPlayers Number of players
     * @param seed Random seed for deterministic initialization
     */
    ZobristHash(int boardSize, int numPieceTypes, int numPlayers, unsigned seed = 0);
    
    /**
     * @brief Get hash value for a piece at a position
     * 
     * @param pieceType The piece type
     * @param position The board position
     * @return 64-bit hash value
     */
    uint64_t getPieceHash(int pieceType, int position) const;
    
    /**
     * @brief Get player turn hash
     * 
     * @param player Current player (0-based index)
     * @return 64-bit hash value for the player
     */
    uint64_t getPlayerHash(int player) const;
    
    /**
     * @brief Add a custom feature with specified values
     * 
     * @param featureName Name of the feature
     * @param numValues Number of possible values
     */
    void addFeature(const std::string& featureName, int numValues);
    
    /**
     * @brief Get feature hash value
     * 
     * @param featureName Name of the feature
     * @param value Value of the feature
     * @return 64-bit hash value
     */
    uint64_t getFeatureHash(const std::string& featureName, int value) const;
    
    /**
     * @brief Get feature hash value by index (legacy interface)
     * 
     * @param featureIndex Index of the feature (0-based, in order of addition)
     * @param value Value of the feature
     * @return 64-bit hash value
     */
    uint64_t getFeatureHash(int featureIndex, int value) const;
    
    /**
     * @brief Check if a feature exists
     * 
     * @param featureName Name of the feature
     * @return true if feature exists, false otherwise
     */
    bool hasFeature(const std::string& featureName) const;
    
    /**
     * @brief Get number of values for a feature
     * 
     * @param featureName Name of the feature
     * @return Number of values, or 0 if feature not found
     */
    int getFeatureValueCount(const std::string& featureName) const;
    
    /**
     * @brief Get board size
     * 
     * @return Board size
     */
    int getBoardSize() const { return boardSize_; }
    
    /**
     * @brief Get number of piece types
     * 
     * @return Number of piece types
     */
    int getNumPieceTypes() const { return numPieceTypes_; }
    
    /**
     * @brief Get number of players
     * 
     * @return Number of players
     */
    int getNumPlayers() const { return numPlayers_; }
    
private:
    int boardSize_;                                     // Board size
    int numPieceTypes_;                                 // Number of piece types
    int numPlayers_;                                    // Number of players
    
    std::vector<std::vector<uint64_t>> pieceHashes_;    // [pieceType][position]
    std::vector<uint64_t> playerHashes_;                // [player]
    
    // Named features for game-specific state
    std::unordered_map<std::string, std::vector<uint64_t>> features_;
    
    /**
     * @brief Generate random 64-bit hash value
     * 
     * @param rng Random number generator
     * @return Random 64-bit hash
     */
    static uint64_t generateRandomHash(std::mt19937_64& rng);
    
    /**
     * @brief Safe modulo operation (handles negative values)
     * 
     * @param value Value to take modulo of
     * @param modulus Modulus value
     * @return Safe modulo result
     */
    static int safeModulo(int value, int modulus);
};

} // namespace core
} // namespace alphazero

#endif // ZOBRIST_HASH_H