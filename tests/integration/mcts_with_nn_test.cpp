// tests/integration/mcts_with_nn_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_engine.h"
#ifdef WITH_TORCH
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include <memory>
#include <chrono>
#include <functional>

using namespace alphazero;

class MCTSWithNNTest : public ::testing::Test {
protected:
    std::shared_ptr<nn::ResNetModel> model;
    mcts::MCTSSettings settings;
    std::unique_ptr<mcts::MCTSEngine> engine;
    std::unique_ptr<games::gomoku::GomokuState> game;
    int64_t input_channels = 17;
    int64_t board_size = 9;
    int64_t policy_size = board_size * board_size;

    void SetUp() override {
        // Create a small ResNet model for testing
        // Parameters: input_channels, board_size, num_res_blocks, num_filters, policy_size
        model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);
        
        // Create MCTS settings for fast testing
        settings.num_simulations = 20;
        settings.num_threads = 2;
        settings.batch_size = 4;
        settings.add_dirichlet_noise = false;  // Deterministic for tests
        
        // Create MCTS engine with the neural network
        engine = std::make_unique<mcts::MCTSEngine>(
            std::static_pointer_cast<nn::NeuralNetwork>(model), settings);
        
        // Create Gomoku game
        game = std::make_unique<games::gomoku::GomokuState>(9);
    }
};

// Test search with neural network
TEST_F(MCTSWithNNTest, Search) {
    // Run search from initial state
    auto result = engine->search(*game);
    
    // Check results
    EXPECT_NE(result.action, -1);
    EXPECT_EQ(result.probabilities.size(), 81);
    EXPECT_GT(result.stats.total_nodes, 1);
    EXPECT_GT(result.stats.search_time.count(), 0);
}

// Test playing a full game
TEST_F(MCTSWithNNTest, PlayGame) {
    // Play a full game
    int max_moves = 20;
    int move_count = 0;
    bool game_over = false;
    
    while (!game_over && move_count < max_moves) {
        // Run search
        auto result = engine->search(*game);
        
        // Make move
        game->makeMove(result.action);
        move_count++;
        
        // Check if game is over
        game_over = game->isTerminal();
    }
    
    // We should have played some moves
    EXPECT_GT(move_count, 0);
}

// Test batch processing
TEST_F(MCTSWithNNTest, BatchProcessing) {
    // Increase simulations to ensure batching occurs
    settings.num_simulations = 50;
    settings.num_threads = 4;
    
    // Create new engine with updated settings
    engine = std::make_unique<mcts::MCTSEngine>(std::static_pointer_cast<nn::NeuralNetwork>(model), settings);
    
    // Run search
    auto result = engine->search(*game);
    
    // Check batch statistics
    EXPECT_GT(result.stats.avg_batch_size, 1.0f);  // Should batch multiple requests
    EXPECT_GT(result.stats.total_evaluations, 0);
}
#else
// Dummy tests when torch is not available
TEST(MCTSWithNNTest, WithoutTorchTest) {
    SUCCEED() << "MCTS with neural network tests are skipped when WITH_TORCH is OFF";
}
#endif // WITH_TORCH

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}