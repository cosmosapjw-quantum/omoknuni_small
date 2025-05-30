#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include "mcts/mcts_engine.h"
#include "games/gomoku/gomoku_state.h"

using namespace alphazero;
using namespace alphazero::mcts;

// Simple mock inference
std::vector<NetworkOutput> mockInference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    std::cout << "Mock inference called with " << states.size() << " states" << std::endl;
    
    std::vector<NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); ++i) {
        NetworkOutput output;
        output.value = 0.0f;
        
        int action_space_size = states[i]->getActionSpaceSize();
        output.policy.resize(action_space_size, 1.0f / action_space_size);
        
        outputs.push_back(std::move(output));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return outputs;
}

int main() {
    std::cout << "Creating test settings..." << std::endl;
    
    MCTSSettings settings;
    settings.num_threads = 1;  // Single thread
    settings.num_simulations = 10;  // Very few simulations
    settings.batch_timeout = std::chrono::milliseconds(10);
    
    std::cout << "Creating engine 1..." << std::endl;
    auto engine1 = std::make_unique<MCTSEngine>(mockInference, settings);
    
    std::cout << "Creating engine 2..." << std::endl;
    auto engine2 = std::make_unique<MCTSEngine>(mockInference, settings);
    
    std::cout << "Creating game state..." << std::endl;
    auto state = std::make_unique<games::gomoku::GomokuState>(5);
    
    std::cout << "Starting moves..." << std::endl;
    
    // Play one move with each engine
    std::cout << "Engine 1 searching..." << std::endl;
    auto result1 = engine1->search(*state);
    std::cout << "Engine 1 selected action: " << result1.action << std::endl;
    
    state->makeMove(result1.action);
    
    if (!state->isTerminal()) {
        std::cout << "Engine 2 searching..." << std::endl;
        auto result2 = engine2->search(*state);
        std::cout << "Engine 2 selected action: " << result2.action << std::endl;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}