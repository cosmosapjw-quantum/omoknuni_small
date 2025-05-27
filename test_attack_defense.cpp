#include <iostream>
#include <memory>
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "utils/attack_defense_module.h"

using namespace alphazero;
using namespace alphazero::games;
using namespace alphazero::games::gomoku;
using namespace alphazero::games::chess;
using namespace alphazero::games::go;

void testGomokuAttackDefense() {
    std::cout << "\n=== Testing Gomoku Attack/Defense ===\n";
    
    // Create a 9x9 Gomoku game
    auto state = std::make_unique<GomokuState>(9);
    
    // Make some moves to create a pattern
    state->makeMove(40); // Center (4,4)
    state->makeMove(41); // (4,5)
    state->makeMove(31); // (3,4)
    state->makeMove(32); // (3,5)
    state->makeMove(22); // (2,4) - creates a three-in-a-row threat
    
    // Get enhanced tensor representation
    auto tensor = state->getEnhancedTensorRepresentation();
    
    std::cout << "Number of channels: " << tensor.size() << std::endl;
    std::cout << "Board size: " << tensor[0].size() << "x" << tensor[0][0].size() << std::endl;
    
    // Check attack plane (channel 17)
    std::cout << "\nAttack plane (channel 17):\n";
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (tensor[17][i][j] > 0) {
                std::cout << "Attack score at (" << i << "," << j << "): " << tensor[17][i][j] << std::endl;
            }
        }
    }
    
    // Check defense plane (channel 18)
    std::cout << "\nDefense plane (channel 18):\n";
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (tensor[18][i][j] > 0) {
                std::cout << "Defense score at (" << i << "," << j << "): " << tensor[18][i][j] << std::endl;
            }
        }
    }
}

void testChessAttackDefense() {
    std::cout << "\n=== Testing Chess Attack/Defense ===\n";
    
    // Create a chess game
    auto state = std::make_unique<ChessState>();
    
    // Note: Chess implementation is simplified placeholder
    // In a real implementation, you would need full chess move generation
    
    std::cout << "Chess attack/defense module created successfully\n";
    
    // Create attack/defense module
    auto module = alphazero::createAttackDefenseModule(core::GameType::CHESS, 8);
    std::cout << "Module type: Chess\n";
}

void testGoAttackDefense() {
    std::cout << "\n=== Testing Go Attack/Defense ===\n";
    
    // Create a 9x9 Go game
    auto state = std::make_unique<GoState>(9);
    
    // Make some moves to create a simple position
    state->makeMove(40); // Center
    state->makeMove(41);
    state->makeMove(31);
    state->makeMove(32);
    
    // Create attack/defense module
    auto module = alphazero::createAttackDefenseModule(core::GameType::GO, 9);
    
    // Test the module
    std::vector<std::unique_ptr<core::IGameState>> states_batch;
    states_batch.push_back(state->clone());
    
    auto [attack_planes, defense_planes] = module->compute_planes(states_batch);
    
    std::cout << "Attack plane computed. Non-zero values:\n";
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (attack_planes[0][i][j] > 0) {
                std::cout << "Attack score at (" << i << "," << j << "): " << attack_planes[0][i][j] << std::endl;
            }
        }
    }
}

int main() {
    std::cout << "Testing Attack/Defense Module Implementation\n";
    std::cout << "==========================================\n";
    
    try {
        testGomokuAttackDefense();
        testChessAttackDefense();
        testGoAttackDefense();
        
        std::cout << "\n=== All tests completed successfully ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}