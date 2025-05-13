#include "utils/hash_specializations.h"
#include "games/chess/chess_state.h"
#include "games/go/go_rules.h"

namespace std {

// Implementation of hash for Chess::Piece
size_t hash<alphazero::games::chess::Piece>::operator()(const alphazero::games::chess::Piece& p) const {
    // Combine the type and color fields
    size_t seed = 0;
    hash_combine(seed, static_cast<int>(p.type));
    hash_combine(seed, static_cast<int>(p.color));
    return seed;
}

// Implementation of hash for Chess::CastlingRights
size_t hash<alphazero::games::chess::CastlingRights>::operator()(const alphazero::games::chess::CastlingRights& cr) const {
    // Create a bit field from the boolean values
    int bitfield = 0;
    if (cr.white_kingside) bitfield |= 1;
    if (cr.white_queenside) bitfield |= 2;
    if (cr.black_kingside) bitfield |= 4;
    if (cr.black_queenside) bitfield |= 8;

    return hash<int>()(bitfield);
}

// Implementation of hash for Go::StoneGroup
size_t hash<alphazero::games::go::StoneGroup>::operator()(const alphazero::games::go::StoneGroup& sg) const {
    size_t seed = 0;
    
    // Hash stones
    for (int stone : sg.stones) {
        hash_combine(seed, stone);
    }
    
    // Hash liberties
    for (int liberty : sg.liberties) {
        hash_combine(seed, liberty);
    }
    
    return seed;
}

} // namespace std