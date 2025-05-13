#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <cstdint>
#include <unordered_set>

// Forward declarations for game-specific structures
namespace alphazero {
    namespace games {
        namespace chess {
            struct Piece;
            struct CastlingRights;
        }

        namespace go {
            struct StoneGroup;
        }
    }
}

namespace std {
    // Specialization of hash for vector<int>
    template<>
    struct hash<vector<int>> {
        size_t operator()(const vector<int>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // Specialization of hash for pair<int, int>
    template<>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
    
    // Specialization of hash for unordered_set<int>
    template<>
    struct hash<unordered_set<int>> {
        size_t operator()(const unordered_set<int>& s) const {
            size_t seed = s.size();
            for (const auto& i : s) {
                seed ^= hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
    
    // Specialization of hash for Chess::Piece
    template<>
    struct hash<alphazero::games::chess::Piece> {
        size_t operator()(const alphazero::games::chess::Piece& p) const;
    };

    // Specialization of hash for Chess::CastlingRights
    template<>
    struct hash<alphazero::games::chess::CastlingRights> {
        size_t operator()(const alphazero::games::chess::CastlingRights& cr) const;
    };

    // Specialization of hash for Go::StoneGroup
    template<>
    struct hash<alphazero::games::go::StoneGroup> {
        size_t operator()(const alphazero::games::go::StoneGroup& sg) const;
    };
}

// Utility function for combining hash values
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}