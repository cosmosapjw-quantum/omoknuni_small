// tests/selfplay/self_play_manager_test.cpp
#include <gtest/gtest.h>
#include "selfplay/self_play_manager.h"
#ifdef WITH_TORCH
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include <memory>

using namespace alphazero;

class SelfPlayManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small model for testing
        model = nn::NeuralNetworkFactory::createResNet(17, 9, 2, 32);
        
        // Create manager
        manager = std::make_unique<selfplay::SelfPlayManager>(
            std::static_pointer_cast<nn::NeuralNetwork>(model));
    }
    
    std::shared_ptr<nn::ResNetModel> model;
    std::unique_ptr<selfplay::SelfPlayManager> manager;
};

// Test game generation
TEST_F(SelfPlayManagerTest, GenerateGames) {
    // Generate a small batch of games
    auto games = manager->generateGames(core::GameType::GOMOKU, 2, 9);
    
    // Check results
    EXPECT_EQ(games.size(), 2);
    for (const auto& game : games) {
        EXPECT_FALSE(game.game_id.empty());
        EXPECT_GT(game.moves.size(), 0);
        EXPECT_EQ(game.policies.size(), game.moves.size());
    }
}

// Test parallel game generation using normal generateGames
// (with settings.num_parallel_games controlling parallelism)
TEST_F(SelfPlayManagerTest, GenerateGamesParallel) {
    // Set parallel games in settings
    selfplay::SelfPlaySettings settings = manager->getSettings();
    settings.num_parallel_games = 2;  // Use 2 parallel threads
    manager->updateSettings(settings);
    
    // Generate a small batch of games
    auto games = manager->generateGames(core::GameType::GOMOKU, 2, 9);
    
    // Check results
    EXPECT_EQ(games.size(), 2);
    for (const auto& game : games) {
        EXPECT_FALSE(game.game_id.empty());
        EXPECT_GT(game.moves.size(), 0);
        EXPECT_EQ(game.policies.size(), game.moves.size());
    }
}
#else
// Dummy tests when torch is not available
TEST(SelfPlayManagerTest, WithoutTorchTest) {
    SUCCEED() << "Self play manager tests are skipped when WITH_TORCH is OFF";
}
#endif // WITH_TORCH

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
