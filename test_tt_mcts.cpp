#include <iostream>
#include <memory>
#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"

using namespace alphazero;

// Simple neural network for testing
class TestNN : public nn::NeuralNetwork {
public:
    nn::NetworkOutput evaluate(const std::vector<std::vector<std::vector<float>>>& input) override {
        nn::NetworkOutput output;
        output.policy = std::vector<float>(225, 1.0f / 225.0f); // Uniform policy
        output.value = 0.0f;
        return output;
    }
    
    std::vector<nn::NetworkOutput> evaluateBatch(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& inputs) override {
        std::vector<nn::NetworkOutput> outputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            outputs.push_back(evaluate(inputs[i]));
        }
        return outputs;
    }
};

int main() {
    // Create MCTS settings
    mcts::MCTSSettings settings;
    settings.num_simulations = 100;
    settings.num_threads = 1;  // Single thread for deterministic behavior
    settings.batch_size = 1;
    settings.use_transposition_table = true;
    settings.transposition_table_size_mb = 1;
    
    // Create neural network and engine
    auto nn = std::make_shared<TestNN>();
    mcts::MCTSEngine engine(nn, settings);
    
    // Create a simple position
    auto game1 = std::make_unique<games::gomoku::GomokuState>(15);
    
    // Make moves: 0-0, 1-1
    game1->makeMove(7 * 15 + 7);  // Center
    game1->makeMove(7 * 15 + 8);  // Right of center
    
    uint64_t hash1 = game1->getHash();
    std::cout << "Position 1 hash: " << hash1 << std::endl;
    
    // Create the same position via different move order
    auto game2 = std::make_unique<games::gomoku::GomokuState>(15);
    
    // This should NOT create the same position (different moves)
    game2->makeMove(7 * 15 + 7);  // Center
    game2->makeMove(7 * 15 + 6);  // Left of center
    
    uint64_t hash2 = game2->getHash();
    std::cout << "Position 2 hash: " << hash2 << std::endl;
    
    if (hash1 == hash2) {
        std::cout << "ERROR: Different positions have same hash!" << std::endl;
    } else {
        std::cout << "Good: Different positions have different hashes" << std::endl;
    }
    
    // Now test with MCTS search
    std::cout << "\nRunning first search..." << std::endl;
    auto result1 = engine.search(*game1);
    
    std::cout << "\nRunning second search on different position..." << std::endl;
    auto result2 = engine.search(*game2);
    
    // Check transposition table stats
    float hit_rate = engine.getTranspositionTableHitRate();
    std::cout << "\nTransposition table hit rate: " << hit_rate << std::endl;
    
    return 0;
}