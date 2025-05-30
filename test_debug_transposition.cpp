#include <iostream>
#include <memory>
#include <chrono>
#include "mcts/mcts_engine.h"
#include "games/gomoku/gomoku_state.h"

using namespace alphazero;

// Simple mock neural network
std::vector<mcts::NetworkOutput> mockNN(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    // std::cout << "[DEBUG] mockNN called with " << states.size() << " states" << std::endl;
    std::vector<mcts::NetworkOutput> outputs;
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput output;
        output.policy = std::vector<float>(states[i]->getActionSpaceSize(), 1.0f / states[i]->getActionSpaceSize());
        output.value = 0.0f;
        outputs.push_back(output);
    }
    return outputs;
}

int main() {
    // std::cout << "[DEBUG] Starting test program" << std::endl;
    
    try {
        // Create minimal settings
        // std::cout << "[DEBUG] Creating MCTSSettings" << std::endl;
        mcts::MCTSSettings settings;
        settings.num_simulations = 1;
        settings.num_threads = 0;
        settings.batch_size = 1;
        settings.batch_timeout = std::chrono::milliseconds(1);
        settings.use_transposition_table = false; // Start with TT disabled
        
        // std::cout << "[DEBUG] Settings created, creating engine" << std::endl;
        
        // Create engine
        auto engine = std::make_unique<mcts::MCTSEngine>(mockNN, settings);
        
        // std::cout << "[DEBUG] Engine created successfully" << std::endl;
        
        // Create simple game state
        // std::cout << "[DEBUG] Creating Gomoku state" << std::endl;
        games::GomokuState game(9);
        
        // std::cout << "[DEBUG] Running search without TT" << std::endl;
        auto result = engine->search(game);
        // std::cout << "[DEBUG] Search completed, action: " << result.action << std::endl;
        
        // Now enable TT and try again
        // std::cout << "[DEBUG] Enabling transposition table" << std::endl;
        engine->setUseTranspositionTable(true);
        
        // std::cout << "[DEBUG] Running search with TT" << std::endl;
        result = engine->search(game);
        // std::cout << "[DEBUG] Search with TT completed, action: " << result.action << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
    
    // std::cout << "[DEBUG] Test completed successfully" << std::endl;
    return 0;
}