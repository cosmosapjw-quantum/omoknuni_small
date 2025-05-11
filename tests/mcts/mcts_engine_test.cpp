// tests/mcts/mcts_engine_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_engine.h"
#include <thread>
#include <chrono>

// Mock game state for testing
class MockGameState : public alphazero::core::IGameState {
public:
    MockGameState(bool terminal = false, alphazero::core::GameResult result = alphazero::core::GameResult::ONGOING)
        : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN),
          terminal_(terminal),
          result_(result) {}
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2, 3, 4}; }
    bool isLegalMove(int action) const override { return action >= 0 && action <= 4; }
    
    void makeMove(int action) override {
        move_history_.push_back(action);
        current_player_ = (current_player_ == 1) ? 2 : 1;
    }
    
    bool undoMove() override {
        if (move_history_.empty()) return false;
        move_history_.pop_back();
        current_player_ = (current_player_ == 1) ? 2 : 1;
        return true;
    }
    
    bool isTerminal() const override { return terminal_; }
    alphazero::core::GameResult getGameResult() const override { return result_; }
    int getCurrentPlayer() const override { return current_player_; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override { 
        return std::vector<std::vector<std::vector<float>>>
               (2, std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));
    }
    
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return getTensorRepresentation();
    }
    
    uint64_t getHash() const override { return 0; }
    
    std::unique_ptr<IGameState> clone() const override { 
        auto clone = std::make_unique<MockGameState>(terminal_, result_);
        clone->current_player_ = current_player_;
        clone->move_history_ = move_history_;
        return clone;
    }
    
    std::string actionToString(int action) const override { return std::to_string(action); }
    
    std::optional<int> stringToAction(const std::string& moveStr) const override { 
        try {
            return std::stoi(moveStr);
        } catch (...) {
            return std::nullopt;
        }
    }
    
    std::string toString() const override { return "MockGameState"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return move_history_; }
    bool validate() const override { return true; }

private:
    bool terminal_;
    alphazero::core::GameResult result_;
    int current_player_ = 1;
    std::vector<int> move_history_;
};

// Mock inference function
inline std::vector<alphazero::mcts::NetworkOutput> mockInference(
    const std::vector<std::unique_ptr<alphazero::core::IGameState>>& states) {
    
    std::vector<alphazero::mcts::NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (const auto& state : states) {
        alphazero::mcts::NetworkOutput output;
        output.value = 0.0f;  // Neutral value
        
        // Create policy with high probability for action 2, low for others
        // Making sure it's sized for the full action space
        int action_space_size = state->getActionSpaceSize();
        output.policy = std::vector<float>(action_space_size, 0.05f);
        
        // Set action 2 to have much higher probability (10x others)
        output.policy[2] = 0.6f;  // Prefer action 2
        
        // Normalize policy to ensure it sums to 1.0
        float sum = 0.0f;
        for (const auto& p : output.policy) {
            sum += p;
        }
        if (sum > 0.0f) {
            for (auto& p : output.policy) {
                p /= sum;
            }
        }
        
        outputs.push_back(output);
    }
    
    // Simulate network delay
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    
    return outputs;
}

// Test fixture
class MCTSEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        alphazero::mcts::MCTSSettings settings;
        settings.num_simulations = 100;  // Reduced for tests
        settings.num_threads = 2;
        settings.batch_size = 4;
        settings.exploration_constant = 1.5f;
        settings.add_dirichlet_noise = false;  // Deterministic for tests
        
        engine = std::make_unique<alphazero::mcts::MCTSEngine>(mockInference, settings);
    }
    
    std::unique_ptr<alphazero::mcts::MCTSEngine> engine;
};

// Test basic search functionality
TEST_F(MCTSEngineTest, BasicSearch) {
    MockGameState state;
    
    auto result = engine->search(state);
    
    // Should select action 2 (highest policy)
    EXPECT_EQ(result.action, 2);
    EXPECT_GT(result.stats.total_nodes, 1);
    EXPECT_GT(result.stats.search_time.count(), 0);
}

// Test temperature parameter
TEST_F(MCTSEngineTest, TemperatureEffect) {
    MockGameState state;
    
    // Set temperature to 0 (deterministic)
    auto settings = engine->getSettings();
    settings.temperature = 0.0f;
    engine->updateSettings(settings);
    
    auto result1 = engine->search(state);
    
    // Should consistently select action 2
    EXPECT_EQ(result1.action, 2);
    
    // Set very high temperature (more uniform)
    settings.temperature = 10.0f;
    engine->updateSettings(settings);
    
    // Run multiple searches and check if we get different actions
    bool different_action = false;
    for (int i = 0; i < 5; i++) {
        auto result2 = engine->search(state);
        if (result2.action != 2) {
            different_action = true;
            break;
        }
    }
    
    // With high temperature, should sometimes select different actions
    EXPECT_TRUE(different_action);
}

// Test terminal state evaluation
TEST_F(MCTSEngineTest, TerminalStateEvaluation) {
    // Terminal state with win for player 1
    MockGameState state(true, alphazero::core::GameResult::WIN_PLAYER1);
    
    auto result = engine->search(state);
    
    // Terminal state should get quick evaluation
    EXPECT_GT(result.stats.total_nodes, 0);
    EXPECT_LT(result.stats.search_time.count(), 100);  // Fast because terminal
}

// Test parallel processing
TEST_F(MCTSEngineTest, ParallelProcessing) {
    MockGameState state;
    
    // Set up for more parallel work
    auto settings = engine->getSettings();
    settings.num_simulations = 200;
    settings.num_threads = 4;
    engine->updateSettings(settings);
    
    auto result = engine->search(state);
    
    // Should have processed batches in parallel
    EXPECT_GT(result.stats.avg_batch_size, 1.0f);
    EXPECT_GT(result.stats.total_evaluations, 0);
}

// Remove the main function when building as part of a test suite
#ifndef BUILDING_TEST_SUITE
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif