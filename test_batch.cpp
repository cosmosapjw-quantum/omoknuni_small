#include "games/gomoku/gomoku_state.h"
#include "mcts/mcts_engine.h"
#include "nn/neural_network_factory.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Starting MCTS batch test..." << std::endl;
    
    // Create a simple Gomoku game
    auto game = std::make_unique<alphazero::games::gomoku::GomokuState>();
    
    // Load neural network
    std::string model_path = "models/model.pt";
    bool use_gpu = false;
    auto neural_net = alphazero::nn::NeuralNetworkFactory::loadResNet(
        model_path, 17, 15, 10, 64, 225, use_gpu);
    
    // Configure MCTS with minimal settings
    alphazero::mcts::MCTSSettings settings;
    settings.num_simulations = 10;
    settings.num_threads = 1;
    settings.batch_size = 1;
    settings.batch_timeout = std::chrono::milliseconds(1000);
    settings.use_root_parallelization = false;
    settings.use_transposition_table = false;
    
    // Create MCTS engine
    alphazero::mcts::MCTSEngine engine(neural_net, settings);
    
    std::cout << "Running search..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    
    try {
        auto result = engine.search(*game);
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Search completed in " << duration.count() << "ms" << std::endl;
        std::cout << "Selected action: " << result.action << std::endl;
        std::cout << "Value: " << result.value << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}