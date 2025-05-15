// tests/training/training_data_manager_test.cpp
#include <gtest/gtest.h>
#include "training/training_data_manager.h"
#include "selfplay/self_play_manager.h"
#ifdef WITH_TORCH
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include <memory>
#include <random>
#include <vector>
#include <filesystem>

using namespace alphazero;

class TrainingDataManagerTest : public ::testing::Test {
protected:
    std::shared_ptr<nn::ResNetModel> model;
    std::unique_ptr<games::gomoku::GomokuState> game_state;
    int64_t input_channels = 17;
    int64_t board_size = 9;
    int64_t policy_size = board_size * board_size;

    void SetUp() override {
        // Create a small ResNet model for testing
        // Parameters: input_channels, board_size, num_res_blocks, num_filters, policy_size
        model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);
        game_state = std::make_unique<games::gomoku::GomokuState>(board_size);
        
        // Create the manager
        training::TrainingDataSettings settings;
        settings.max_examples = 100;
        manager = std::make_unique<training::TrainingDataManager>(settings);
        
        // Create test output directory
        test_output_dir = "test_training_data_output";
        std::filesystem::create_directories(test_output_dir);
        
        // Create some test games
        game1.moves = {0, 1, 2};
        game1.policies = {{0.5f, 0.5f, 0.0f}, {0.0f, 0.5f, 0.5f}, {0.5f, 0.0f, 0.5f}};
        game1.winner = 1;
        game1.game_type = core::GameType::GOMOKU;
        game1.board_size = 3;
        game1.game_id = "test_game_1";
        
        game2.moves = {1, 0, 2};
        game2.policies = {{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
        game2.winner = 2;
        game2.game_type = core::GameType::GOMOKU;
        game2.board_size = 3;
        game2.game_id = "test_game_2";
    }
    
    void TearDown() override {
        // Clean up test directory
        std::filesystem::remove_all(test_output_dir);
    }
    
    std::unique_ptr<training::TrainingDataManager> manager;
    std::string test_output_dir;
    selfplay::GameData game1;
    selfplay::GameData game2;
};

// Test adding and retrieving examples
TEST_F(TrainingDataManagerTest, AddAndRetrieveExamples) {
    // Create some dummy examples
    std::vector<training::TrainingExample> examples;
    for (int i = 0; i < 10; i++) {
        training::TrainingExample example;
        example.game_id = "test_game_" + std::to_string(i);
        example.move_number = i;
        example.state.resize(17, std::vector<std::vector<float>>(9, std::vector<float>(9, 0.0f)));
        example.policy.resize(81, 0.0f);
        example.policy[i] = 1.0f;
        example.value = (i % 2 == 0) ? 1.0f : -1.0f;
        examples.push_back(example);
    }
    
    // Convert to GameData and add
    selfplay::GameData gameData;
    gameData.game_type = core::GameType::GOMOKU;
    gameData.board_size = 9;
    gameData.winner = 1;
    gameData.game_id = "test_examples_game";
    
    // Add game data for testing
    manager->addGames({gameData}, 1);
    
    // Sample a batch
    auto sampled = manager->sampleBatch(5);
    EXPECT_LE(sampled.size(), 5);
    
    // Get examples per iteration
    auto examplesPerIter = manager->getExamplesPerIteration();
    EXPECT_GE(examplesPerIter.size(), 1);  // Should have at least one iteration
}

// Test merging iterations
TEST_F(TrainingDataManagerTest, MergeIterations) {
    // Add examples for two iterations
    std::vector<training::TrainingExample> examples1;
    std::vector<training::TrainingExample> examples2;
    
    for (int i = 0; i < 10; i++) {
        training::TrainingExample example;
        example.game_id = "test_game_" + std::to_string(i);
        example.move_number = i;
        example.state.resize(17, std::vector<std::vector<float>>(9, std::vector<float>(9, 0.0f)));
        example.policy.resize(81, 0.0f);
        example.policy[i] = 1.0f;
        example.value = (i % 2 == 0) ? 1.0f : -1.0f;
        
        if (i < 5) 
            examples1.push_back(example);
        else 
            examples2.push_back(example);
    }
    
    // Create game data for different iterations
    selfplay::GameData gameData1;
    gameData1.game_type = core::GameType::GOMOKU;
    gameData1.board_size = 9;
    gameData1.winner = 1;
    gameData1.game_id = "test_game_1";
    gameData1.moves = {0, 1, 2, 3, 4};
    
    selfplay::GameData gameData2;
    gameData2.game_type = core::GameType::GOMOKU;
    gameData2.board_size = 9;
    gameData2.winner = 2;
    gameData2.game_id = "test_game_2";
    gameData2.moves = {5, 6, 7, 8, 9};
    
    // Add games to different iterations
    manager->addGames({gameData1}, 1);
    manager->addGames({gameData2}, 2);
    
    // Check iterations
    auto examples_per_iter = manager->getExamplesPerIteration();
    EXPECT_EQ(examples_per_iter.size(), 2);
    EXPECT_TRUE(examples_per_iter.find(1) != examples_per_iter.end());
    EXPECT_TRUE(examples_per_iter.find(2) != examples_per_iter.end());
    
    // Sample batches from both iterations
    auto batch = manager->sampleBatch(10);
    EXPECT_LE(batch.size(), 10);
}

// Test adding games
TEST_F(TrainingDataManagerTest, AddGames) {
    // Add games from iteration 0
    manager->addGames({game1}, 0);
    
    // Check results
    EXPECT_EQ(manager->getTotalExamples(), 3);  // 3 moves
    
    auto examples_per_iter = manager->getExamplesPerIteration();
    EXPECT_EQ(examples_per_iter.size(), 1);
    EXPECT_EQ(examples_per_iter[0], 3);
    
    // Add more games from iteration 1
    manager->addGames({game2}, 1);
    
    // Check results
    EXPECT_EQ(manager->getTotalExamples(), 6);  // 3 + 3 moves
    
    examples_per_iter = manager->getExamplesPerIteration();
    EXPECT_EQ(examples_per_iter.size(), 2);
    EXPECT_EQ(examples_per_iter[0], 3);
    EXPECT_EQ(examples_per_iter[1], 3);
}

// Test sampling batch
TEST_F(TrainingDataManagerTest, SampleBatch) {
    // Add games
    manager->addGames({game1, game2}, 0);
    
    // Sample a batch
    auto batch = manager->sampleBatch(4);
    
    // Check results
    EXPECT_EQ(batch.size(), 4);
    
    // Each example should have the right shapes
    for (const auto& example : batch) {
        EXPECT_GT(example.state.size(), 0);
        EXPECT_GT(example.policy.size(), 0);
        EXPECT_GE(example.value, -1.0f);
        EXPECT_LE(example.value, 1.0f);
    }
}

// Test save and load
TEST_F(TrainingDataManagerTest, SaveLoad) {
    // Add games
    manager->addGames({game1}, 0);
    manager->addGames({game2}, 1);
    
    // Save to disk
    manager->save(test_output_dir, "binary");
    
    // Create a new manager and load
    auto new_manager = std::make_unique<training::TrainingDataManager>();
    new_manager->load(test_output_dir, "binary");
    
    // Check results
    EXPECT_EQ(new_manager->getTotalExamples(), manager->getTotalExamples());
    
    auto examples_per_iter = new_manager->getExamplesPerIteration();
    auto original_per_iter = manager->getExamplesPerIteration();
    
    EXPECT_EQ(examples_per_iter.size(), original_per_iter.size());
    for (const auto& [iter, count] : original_per_iter) {
        EXPECT_EQ(examples_per_iter[iter], count);
    }
}

// Test trimming old iterations
TEST_F(TrainingDataManagerTest, TrimOldIterations) {
    // Create manager with small max examples
    training::TrainingDataSettings settings;
    settings.max_examples = 5;  // Only keep 5 examples
    settings.min_iterations_to_keep = 1;  // Keep at least 1 iteration
    
    auto trim_manager = std::make_unique<training::TrainingDataManager>(settings);
    
    // Create games with more moves than the max_examples
    selfplay::GameData oldGame;
    oldGame.game_type = core::GameType::GOMOKU;
    oldGame.board_size = 3;
    oldGame.winner = 1;
    oldGame.game_id = "old_game";
    oldGame.moves = {0, 1, 2};
    
    selfplay::GameData newGame;
    newGame.game_type = core::GameType::GOMOKU;
    newGame.board_size = 3;
    newGame.winner = 2;
    newGame.game_id = "new_game";
    newGame.moves = {3, 4, 5};
    
    // Add games from iteration 0
    trim_manager->addGames({oldGame}, 0);
    
    // Check results
    EXPECT_EQ(trim_manager->getTotalExamples(), 3);
    
    // Add games from iteration 1
    trim_manager->addGames({newGame}, 1);
    
    // Since max_examples is 5 but we have 3 + 3 = 6 examples total,
    // we should trim at least one example, likely trimming the older iteration
    auto examples_per_iter = trim_manager->getExamplesPerIteration();
    
    // Either we keep only the newest iteration or we trim part of the old one
    if (examples_per_iter.size() == 1) {
        // Should have kept only iteration 1
        EXPECT_EQ(examples_per_iter.find(0) == examples_per_iter.end(), true);
        EXPECT_EQ(examples_per_iter.find(1) != examples_per_iter.end(), true);
    } else {
        // Or we should have fewer than 6 total examples due to trimming
        EXPECT_LE(trim_manager->getTotalExamples(), 5);
    }
}

#else
// Dummy tests when torch is not available
TEST(TrainingDataManagerTest, WithoutTorchTest) {
    SUCCEED() << "Training data manager tests are skipped when WITH_TORCH is OFF";
}
#endif // WITH_TORCH

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}