// tests/evaluation/model_evaluator_test.cpp
#include <gtest/gtest.h>
#include "evaluation/model_evaluator.h"
#ifdef WITH_TORCH
#include "nn/neural_network_factory.h"
#include "mcts/mcts_engine.h"
#include "games/gomoku/gomoku_state.h"
#include "core/igamestate.h"
#include <memory>

using namespace alphazero;

class ModelEvaluatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create small models for testing
        model1 = nn::NeuralNetworkFactory::createResNet(17, 9, 2, 32);
        model2 = nn::NeuralNetworkFactory::createResNet(17, 9, 2, 32);
        
        // Initialize evaluator
        evaluator = std::make_unique<evaluation::ModelEvaluator>(
            std::static_pointer_cast<nn::NeuralNetwork>(model1),
            std::static_pointer_cast<nn::NeuralNetwork>(model2),
            evaluation::EvaluationSettings{});
    }
    
    std::shared_ptr<nn::ResNetModel> model1;
    std::shared_ptr<nn::ResNetModel> model2;
    std::unique_ptr<evaluation::ModelEvaluator> evaluator;
};

// Test match function
TEST_F(ModelEvaluatorTest, MatchTest) {
    // Create a match
    auto result = evaluator->playMatch(core::GameType::GOMOKU, 9, 0, true);
    result.match_id = "test_match";
    
    // Basic checks
    EXPECT_EQ(result.match_id, "test_match");
    EXPECT_GE(result.moves.size(), 0);
    EXPECT_TRUE(result.result == 1.0f || result.result == 0.0f || result.result == -1.0f);
}

// Test tournament function
TEST_F(ModelEvaluatorTest, TournamentTest) {
    // Run a small tournament
    // Update settings to run only 2 games
    evaluation::EvaluationSettings settings = evaluator->getSettings();
    settings.num_games = 2;
    evaluator->updateSettings(settings);
    
    auto result = evaluator->runTournament(core::GameType::GOMOKU, 9);
    
    // Basic checks
    EXPECT_EQ(result.matches.size(), 2);
    EXPECT_GE(result.wins_first, 0);
    EXPECT_GE(result.wins_second, 0);
    EXPECT_GE(result.draws, 0);
    EXPECT_EQ(result.wins_first + result.wins_second + result.draws, 2);
}

// Test tournament with parallel games function
TEST_F(ModelEvaluatorTest, TournamentParallelTest) {
    // Update settings to run 2 games with 2 parallel games
    evaluation::EvaluationSettings settings = evaluator->getSettings();
    settings.num_games = 2;
    settings.num_parallel_games = 2;
    evaluator->updateSettings(settings);
    
    // Run a tournament with parallel games
    auto result = evaluator->runTournament(core::GameType::GOMOKU, 9);
    
    // Basic checks
    EXPECT_EQ(result.matches.size(), 2);
    EXPECT_GE(result.wins_first, 0);
    EXPECT_GE(result.wins_second, 0);
    EXPECT_GE(result.draws, 0);
    EXPECT_EQ(result.wins_first + result.wins_second + result.draws, 2);
}
#else
// Dummy tests when torch is not available
TEST(ModelEvaluatorTest, WithoutTorchTest) {
    SUCCEED() << "Model evaluator tests are skipped when WITH_TORCH is OFF";
}
#endif // WITH_TORCH

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}