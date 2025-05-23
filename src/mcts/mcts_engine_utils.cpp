#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/gamestate_pool.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace alphazero {
namespace mcts {

// This file originally contained implementations of several methods, but they were
// already defined in other files:
// - waitWithBackoff - Already in mcts_engine_common.cpp
// - forceCleanup - Was causing multiple definition errors 
// - monitorMemoryUsage - Already in mcts_engine_main.cpp
// - countTreeNodes - Already in mcts_engine_common.cpp
// - calculateMaxDepth - Already in mcts_engine_common.cpp
// - countTreeStatistics - Already in mcts_engine_common.cpp
//
// We'll implement a simpler version of forceCleanup here, which is the only
// method we need to add.

// Force aggressive memory cleanup
void MCTSEngine::forceCleanup() {
    // UnifiedInferenceServer was removed in simplification
    
    // Clean up the transposition table
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    // Reset leaf and result queues
    PendingEvaluation dummy;
    while (leaf_queue_.try_dequeue(dummy)) {
        // Just empty the queue
    }
    
    std::pair<NetworkOutput, PendingEvaluation> dummy_result;
    while (result_queue_.try_dequeue(dummy_result)) {
        // Just empty the queue
    }
    
    // Reset evaluation counters
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Hint to the garbage collector
    std::this_thread::yield();
}

// Utility function to safely validate a game state
bool MCTSEngine::safeGameStateValidation(const core::IGameState& state) {
    try {
        return state.validate();
    } catch (const std::exception& e) {
        std::cerr << "Exception during game state validation: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception during game state validation" << std::endl;
        return false;
    }
}

// Create a root node for MCTS search
std::shared_ptr<MCTSNode> MCTSEngine::createRootNode(const core::IGameState& state) {
    // Create a clone of the game state
    auto state_clone = state.clone();
    
    // Validate the state before using it
    if (!safeGameStateValidation(*state_clone)) {
        std::cerr << "Warning: Invalid game state provided for root node!" << std::endl;
        throw std::invalid_argument("Invalid game state for root node");
    }
    
    // Create a new root node with the cloned state using the factory method
    auto root = MCTSNode::create(std::move(state_clone));
    
    std::cout << "Created root node for MCTS search with game type: " 
              << static_cast<int>(root->getState().getGameType()) << std::endl;
    
    return root;
}

} // namespace mcts
} // namespace alphazero