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

using namespace alphazero;

class TrainingDataManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small model for testing
        model = nn::NeuralNetworkFactory::createResNet(17, 9, 2, 32);
        
        // Create the manager
        manager = std::make_unique<training::TrainingDataManager>(100);
        
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
    
    std::shared_ptr<nn::ResNetModel> model;
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
        example.move_idx = i;
        example.iteration = 1;
        example.state.resize(17, std::vector<std::vector<float>>(9, std::vector<float>(9, 0.0f)));
        example.policy.resize(81, 0.0f);
        example.policy[i] = 1.0f;
        example.value = (i % 2 == 0) ? 1.0f : -1.0f;
        examples.push_back(example);
    }
    
    // Add examples
    manager->addExamples(examples, 1);
    
    // Get examples
    auto retrieved = manager->getExamples(1, 5);
    EXPECT_EQ(retrieved.size(), 5);
    
    // Check mixing of newer examples
    auto mixed = manager->getExamples({1}, 20, 0.5f);
    EXPECT_EQ(mixed.size(), 10);  // Only 10 examples total in iteration 1
}

// Test merging iterations
TEST_F(TrainingDataManagerTest, MergeIterations) {
    // Add examples for two iterations
    std::vector<training::TrainingExample> examples1;
    std::vector<training::TrainingExample> examples2;
    
    for (int i = 0; i < 10; i++) {
        training::TrainingExample example;
        example.game_id = "test_game_" + std::to_string(i);
        example.move_idx = i;
        example.iteration = i < 5 ? 1 : 2;
        example.state.resize(17, std::vector<std::vector<float>>(9, std::vector<float>(9, 0.0f)));
        example.policy.resize(81, 0.0f);
        example.policy[i] = 1.0f;
        example.value = (i % 2 == 0) ? 1.0f : -1.0f;
        
        if (i < 5) 
            examples1.push_back(example);
        else 
            examples2.push_back(example);
    }
    
    // Add examples to different iterations
    manager->addExamples(examples1, 1);
    manager->addExamples(examples2, 2);
    
    // Check available iterations
    auto iterations = manager->getAvailableIterations();
    EXPECT_EQ(iterations.size(), 2);
    EXPECT_TRUE(std::find(iterations.begin(), iterations.end(), 1) != iterations.end());
    EXPECT_TRUE(std::find(iterations.begin(), iterations.end(), 2) != iterations.end());
    
    // Merge iterations
    manager->mergeIterations({1, 2}, 3);
    
    // Check new iteration
    iterations = manager->getAvailableIterations();
    EXPECT_TRUE(std::find(iterations.begin(), iterations.end(), 3) != iterations.end());
    
    // Check merged examples
    auto merged = manager->getExamples(3, 20);
    EXPECT_EQ(merged.size(), 10);
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
    
    // Add games from iteration 0
    trim_manager->addGames({game1}, 0);
    
    // Check results
    EXPECT_EQ(trim_manager->getTotalExamples(), 3);
    
    // Add games from iteration 1
    trim_manager->addGames({game2}, 1);
    
    // Should have trimmed iteration 0
    EXPECT_EQ(trim_manager->getTotalExamples(), 3);
    
    auto examples_per_iter = trim_manager->getExamplesPerIteration();
    EXPECT_EQ(examples_per_iter.size(), 1);
    EXPECT_EQ(examples_per_iter[1], 3);
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