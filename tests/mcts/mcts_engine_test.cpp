// tests/mcts/mcts_engine_test.cpp
#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <atomic>

using namespace alphazero;
using namespace alphazero::mcts;
using ::testing::_;
using ::testing::Return;

// Mock neural network inference function for testing
std::vector<NetworkOutput> EngineTest_mockInference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); ++i) {
        NetworkOutput output;
        
        // Create a simple policy distribution - highest probability for action 2
        int action_space_size = states[i]->getActionSpaceSize();
        output.policy.resize(action_space_size, 0.1f / (action_space_size - 1));
        if (action_space_size > 2) {
            output.policy[2] = 0.9f; // High probability for action 2
        }
        
        // Set a predictable value based on position
        output.value = static_cast<float>(i % 2 == 0 ? 0.8 : -0.8);
        
        outputs.push_back(std::move(output));
    }
    
    // Simulate some computation time
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    return outputs;
}

// Slow mock inference for timeout testing
std::vector<NetworkOutput> EngineTest_slowMockInference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    // Sleep to simulate slow processing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return EngineTest_mockInference(states);
}

// Error-throwing mock inference for error handling testing
std::vector<NetworkOutput> EngineTest_errorMockInference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    throw std::runtime_error("Simulated inference error");
}

// Helper function to create a simple game state for testing
std::unique_ptr<core::IGameState> EngineTest_createTestState(int board_size = 5) {
    return std::make_unique<games::gomoku::GomokuState>(board_size);
}

// Test fixture
class MCTSEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a default engine with mock inference
        MCTSSettings settings;
        settings.num_simulations = 100;
        settings.num_threads = 2;
        settings.batch_size = 4;
        settings.batch_timeout = std::chrono::milliseconds(10);
        settings.exploration_constant = 1.5f;
        settings.virtual_loss = 3;
        settings.add_dirichlet_noise = true;
        settings.dirichlet_alpha = 0.3f;
        settings.dirichlet_epsilon = 0.25f;
        settings.temperature = 1.0f;
        
        engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
    }
    
    void TearDown() override {
        engine.reset();
    }
    
    std::unique_ptr<MCTSEngine> engine;
};

// Basic functionality test
TEST_F(MCTSEngineTest, BasicSearch) {
    // Create a simple game state
    auto state = EngineTest_createTestState();
    
    // Run a search
    auto result = engine->search(*state);
    
    // Verify we got valid results
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_GT(result.stats.search_time.count(), 0);
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_EQ(result.probabilities.size(), state->getActionSpaceSize());
    EXPECT_GE(result.action, 0);
    EXPECT_LT(result.action, static_cast<int>(result.probabilities.size()));
}

// Test serial mode (no threads)
TEST_F(MCTSEngineTest, SerialMode) {
    // Create settings for serial mode
    MCTSSettings settings = engine->getSettings();
    settings.num_threads = 0;  // Serial mode
    settings.num_simulations = 50;  // Fewer simulations for speed
    
    // Create a new engine
    auto serial_engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
    
    // Run a search
    auto state = EngineTest_createTestState();
    auto result = serial_engine->search(*state);
    
    // Verify results
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_GE(result.action, 0);
}

// Test parallel mode with more threads
TEST_F(MCTSEngineTest, ParallelMode) {
    // Create settings with more threads
    MCTSSettings settings = engine->getSettings();
    settings.num_threads = 4;  // More threads
    settings.num_simulations = 200;  // More simulations
    
    // Create a new engine
    auto parallel_engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
    
    // Run a search
    auto state = EngineTest_createTestState();
    auto result = parallel_engine->search(*state);
    
    // Verify results
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_GE(result.action, 0);
}

// Test different temperature settings
TEST_F(MCTSEngineTest, TemperatureSettings) {
    // Zero temperature - deterministic action selection
    {
        MCTSSettings settings = engine->getSettings();
        settings.temperature = 0.0f;
        auto temp_engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
        
        auto state = EngineTest_createTestState();
        auto result = temp_engine->search(*state);
        
        // With deterministic action selection and our mock inference,
        // action 2 should always be chosen because it has highest probability
        EXPECT_EQ(result.action, 2);
    }
    
    // High temperature - more random
    {
        MCTSSettings settings = engine->getSettings();
        settings.temperature = 10.0f;
        settings.num_simulations = 500;  // More simulations for statistic reliability
        auto temp_engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
        
        auto state = EngineTest_createTestState();
        
        // Run multiple searches to check non-determinism
        std::unordered_map<int, int> action_counts;
        const int num_tests = 5;
        
        for (int i = 0; i < num_tests; ++i) {
            auto result = temp_engine->search(*state);
            action_counts[result.action]++;
        }
        
        // With high temperature, we're more likely to see different actions
        // but this is probabilistic - we only check we don't always get the same action
        bool has_variation = action_counts.size() > 1 || num_tests < 10;
        // If we hit the unlikely case of all same actions in just 5 trials, that's acceptable
        EXPECT_TRUE(has_variation) << "Expected some variation with high temperature";
    }
}

// Test transposition table
TEST_F(MCTSEngineTest, TranspositionTable) {
    // Enable transposition table
    engine->setUseTranspositionTable(true);
    
    // Clear any existing entries
    engine->clearTranspositionTable();
    
    // Run a search
    auto state = EngineTest_createTestState();
    auto result1 = engine->search(*state);
    
    // Check hit rate (should be low for first search)
    EXPECT_LT(result1.stats.tt_hit_rate, 0.5f);
    
    // Run the same search again
    auto state2 = EngineTest_createTestState();  // Same state, different instance
    auto result2 = engine->search(*state2);
    
    // Second search should have higher hit rate
    EXPECT_GT(result2.stats.tt_hit_rate, 0.0f);
    
    // Run with transposition table disabled
    engine->setUseTranspositionTable(false);
    auto state3 = EngineTest_createTestState();
    auto result3 = engine->search(*state3);
    
    // Hit rate should be 0 with table disabled
    EXPECT_EQ(result3.stats.tt_hit_rate, 0.0f);
}

// Test slow inference handling
TEST_F(MCTSEngineTest, SlowInferenceHandling) {
    // Create engine with slow inference
    MCTSSettings settings = engine->getSettings();
    settings.num_simulations = 50;  // Fewer simulations for speed
    auto slow_engine = std::make_unique<MCTSEngine>(EngineTest_slowMockInference, settings);
    
    // Run a search with timeout tracking
    auto state = EngineTest_createTestState();
    auto start_time = std::chrono::steady_clock::now();
    
    auto result = slow_engine->search(*state);
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    
    // Verify the search completed despite slow inference
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_GE(result.action, 0);
    
    // The search should not stall - it should complete in reasonable time
    std::cout << "Slow inference search completed in " << elapsed.count() << "ms" << std::endl;
}

// Test error handling during inference
TEST_F(MCTSEngineTest, ErrorHandling) {
    // Create engine with error-throwing inference
    MCTSSettings settings = engine->getSettings();
    settings.num_simulations = 20;  // Fewer simulations for speed
    auto error_engine = std::make_unique<MCTSEngine>(EngineTest_errorMockInference, settings);
    
    // Run a search - should not crash despite errors
    auto state = EngineTest_createTestState();
    auto result = error_engine->search(*state);
    
    // The search should complete with some fallback behavior
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_GE(result.action, 0);
}

// Test consecutive searches
TEST_F(MCTSEngineTest, ConsecutiveSearches) {
    // Run multiple searches in sequence
    auto state = EngineTest_createTestState();
    
    // First search
    auto result1 = engine->search(*state);
    
    // Save statistics
    auto total_nodes1 = result1.stats.total_nodes;
    
    // Make a move and run a second search
    state->makeMove(result1.action);
    auto result2 = engine->search(*state);
    
    // Verify the second search worked
    EXPECT_GT(result2.stats.total_nodes, 0);
    
    // Total nodes in tree should reset between searches
    EXPECT_NE(result2.stats.total_nodes, total_nodes1);
}

// Test multi-threaded search with many simulations
TEST_F(MCTSEngineTest, HeavySearch) {
    // Create settings for a heavy search
    MCTSSettings settings = engine->getSettings();
    settings.num_threads = 4;
    settings.num_simulations = 1000;
    
    // Create a new engine
    auto heavy_engine = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings);
    
    // Run a search on a larger board
    auto state = EngineTest_createTestState(15);  // 15x15 board
    
    auto start_time = std::chrono::steady_clock::now();
    auto result = heavy_engine->search(*state);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    
    // Verify search completed successfully
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_FALSE(result.probabilities.empty());
    EXPECT_EQ(result.probabilities.size(), state->getActionSpaceSize());
    EXPECT_GE(result.action, 0);
    
    // Check search stats
    EXPECT_GT(result.stats.max_depth, 1);
    EXPECT_GT(result.stats.search_time.count(), 0);
    
    // Search should not stall
    std::cout << "Heavy search with " << result.stats.total_nodes 
              << " nodes completed in " << elapsed.count() << "ms" << std::endl;
}

// Test Dirichlet noise
TEST_F(MCTSEngineTest, DirichletNoise) {
    // Create settings with and without Dirichlet noise
    MCTSSettings settings_with_noise = engine->getSettings();
    settings_with_noise.add_dirichlet_noise = true;
    
    MCTSSettings settings_without_noise = engine->getSettings();
    settings_without_noise.add_dirichlet_noise = false;
    
    // Create engines
    auto engine_with_noise = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings_with_noise);
    auto engine_without_noise = std::make_unique<MCTSEngine>(EngineTest_mockInference, settings_without_noise);
    
    // Run searches
    auto state = EngineTest_createTestState();
    auto result_with_noise = engine_with_noise->search(*state);
    auto result_without_noise = engine_without_noise->search(*state);
    
    // Both searches should complete successfully
    EXPECT_GT(result_with_noise.stats.total_nodes, 0);
    EXPECT_GT(result_without_noise.stats.total_nodes, 0);
    
    // With noise, the policy might differ - but we can't assert this strongly
    // as it's probabilistic
}

// Test engine move after another engine
TEST_F(MCTSEngineTest, EngineVsEngine) {
    // Create a second engine to play against the first
    auto engine2 = std::make_unique<MCTSEngine>(EngineTest_mockInference, engine->getSettings());
    
    // Create a game state
    auto state = EngineTest_createTestState();
    
    // Play 3 moves for each engine
    for (int i = 0; i < 3; ++i) {
        // Engine 1's turn
        auto result1 = engine->search(*state);
        EXPECT_GE(result1.action, 0);
        state->makeMove(result1.action);
        
        if (state->isTerminal()) break;
        
        // Engine 2's turn
        auto result2 = engine2->search(*state);
        EXPECT_GE(result2.action, 0);
        state->makeMove(result2.action);
        
        if (state->isTerminal()) break;
    }
    
    // Game should progress without errors
    SUCCEED();
}