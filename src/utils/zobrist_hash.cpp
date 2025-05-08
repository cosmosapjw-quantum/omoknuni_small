// zobrist_hash.cpp
#include "utils/zobrist_hash.h"
#include <stdexcept>
#include <chrono>

namespace alphazero {
namespace core {

ZobristHash::ZobristHash(int boardSize, int numPieceTypes, int numPlayers, unsigned seed)
    : boardSize_(boardSize), numPieceTypes_(numPieceTypes), numPlayers_(numPlayers) {
    
    // Validate parameters
    if (boardSize <= 0) throw std::invalid_argument("Board size must be positive");
    if (numPieceTypes < 0) throw std::invalid_argument("Number of piece types must be non-negative");
    if (numPlayers <= 0) throw std::invalid_argument("Number of players must be positive");
    
    // Use provided seed or generate from time
    unsigned actualSeed = seed != 0 ? seed : 
        static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(actualSeed);
    
    // Initialize piece hashes
    pieceHashes_.resize(numPieceTypes);
    for (int p = 0; p < numPieceTypes; ++p) {
        pieceHashes_[p].resize(boardSize * boardSize);
        for (int pos = 0; pos < boardSize * boardSize; ++pos) {
            pieceHashes_[p][pos] = generateRandomHash(rng);
        }
    }
    
    // Initialize player hashes
    playerHashes_.resize(numPlayers);
    for (int p = 0; p < numPlayers; ++p) {
        playerHashes_[p] = generateRandomHash(rng);
    }
}

uint64_t ZobristHash::getPieceHash(int pieceType, int position) const {
    if (pieceType < 0 || pieceType >= numPieceTypes_) {
        throw std::out_of_range("Piece type index out of range");
    }
    if (position < 0 || position >= boardSize_ * boardSize_) {
        throw std::out_of_range("Position index out of range");
    }
    return pieceHashes_[pieceType][position];
}

uint64_t ZobristHash::getPlayerHash(int player) const {
    if (player < 0 || player >= numPlayers_) {
        throw std::out_of_range("Player index out of range");
    }
    return playerHashes_[player];
}

void ZobristHash::addFeature(const std::string& featureName, int numValues) {
    if (numValues <= 0) {
        throw std::invalid_argument("Number of feature values must be positive");
    }
    
    // Generate random hashes for each value
    std::mt19937_64 rng(std::hash<std::string>{}(featureName));
    
    std::vector<uint64_t> valueHashes(numValues);
    for (int i = 0; i < numValues; ++i) {
        valueHashes[i] = generateRandomHash(rng);
    }
    
    features_[featureName] = std::move(valueHashes);
}

uint64_t ZobristHash::getFeatureHash(const std::string& featureName, int value) const {
    auto it = features_.find(featureName);
    if (it == features_.end()) {
        throw std::out_of_range("Feature not found: " + featureName);
    }
    
    const auto& values = it->second;
    if (values.empty()) {
        throw std::logic_error("Feature has no values: " + featureName);
    }
    
    // Handle negative values safely
    int safeValue = safeModulo(value, values.size());
    return values[safeValue];
}

uint64_t ZobristHash::getFeatureHash(int featureIndex, int value) const {
    // Ensure we have features to index
    if (features_.empty()) {
        throw std::out_of_range("No features available");
    }
    
    // Get the feature name at the given index
    if (featureIndex < 0 || featureIndex >= static_cast<int>(features_.size())) {
        throw std::out_of_range("Feature index out of range: " + std::to_string(featureIndex));
    }
    
    // Use a cache of feature names for faster lookup
    static std::vector<std::string> featureNames;
    static size_t lastCacheSize = 0;
    
    // Rebuild cache if needed
    if (lastCacheSize != features_.size()) {
        featureNames.clear();
        featureNames.reserve(features_.size());
        for (const auto& pair : features_) {
            featureNames.push_back(pair.first);
        }
        lastCacheSize = features_.size();
    }
    
    // Fast lookup from cache
    if (featureIndex < static_cast<int>(featureNames.size())) {
        return getFeatureHash(featureNames[featureIndex], value);
    }
    
    throw std::out_of_range("Feature index not found in cache: " + std::to_string(featureIndex));
}

bool ZobristHash::hasFeature(const std::string& featureName) const {
    return features_.find(featureName) != features_.end();
}

int ZobristHash::getFeatureValueCount(const std::string& featureName) const {
    auto it = features_.find(featureName);
    return it != features_.end() ? static_cast<int>(it->second.size()) : 0;
}

uint64_t ZobristHash::generateRandomHash(std::mt19937_64& rng) {
    return rng();
}

int ZobristHash::safeModulo(int value, int modulus) {
    return ((value % modulus) + modulus) % modulus;
}

} // namespace core
} // namespace alphazero