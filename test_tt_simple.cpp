#include <iostream>
#include <memory>
#include "mcts/phmap_transposition_table.h"
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"

using namespace alphazero;

int main() {
    // Create a simple transposition table
    mcts::PHMapTranspositionTable::Config config;
    config.size_mb = 1;
    config.enable_stats = true;
    
    mcts::PHMapTranspositionTable tt(config);
    
    // Create a simple game state and node
    auto game = std::make_unique<games::gomoku::GomokuState>(15);
    auto node = mcts::MCTSNode::create(std::move(game), nullptr);
    
    uint64_t hash = node->getState().getHash();
    std::cout << "Hash: " << hash << std::endl;
    
    // Store the node
    std::cout << "Storing node..." << std::endl;
    tt.store(hash, node, 0);
    
    // Try to retrieve it
    std::cout << "Looking up node..." << std::endl;
    auto retrieved = tt.lookup(hash);
    
    if (retrieved) {
        std::cout << "SUCCESS: Node retrieved!" << std::endl;
    } else {
        std::cout << "FAILED: Node not found!" << std::endl;
    }
    
    // Check stats
    auto stats = tt.getStats();
    std::cout << "Stats:" << std::endl;
    std::cout << "  Total lookups: " << stats.total_lookups << std::endl;
    std::cout << "  Successful lookups: " << stats.successful_lookups << std::endl;
    std::cout << "  Hit rate: " << stats.hit_rate << std::endl;
    
    return 0;
}