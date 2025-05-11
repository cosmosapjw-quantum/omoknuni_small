// tests/mcts/mcts_evaluator_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <thread>
#include <chrono>
#include <vector>
#include <memory>

// Mock game state for testing
class MockGameState : public alphazero::core::IGameState {
public:
    MockGameState() : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN) {}
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action <= 2; }
    void makeMove(int action) override {}
    bool undoMove() override { return false; }
    bool isTerminal() const override { return false; }
    alphazero::core::GameResult getGameResult() const override { return alphazero::core::GameResult::ONGOING; }
    int getCurrentPlayer() const override { return 1; }
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
        return std::make_unique<MockGameState>();
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
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }
};

// Mock inference function
inline std::vector<alphazero::mcts::NetworkOutput> mockInference(
    const std::vector<std::unique_ptr<alphazero::core::IGameState>>& states) {
    
    std::vector<alphazero::mcts::NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); ++i) {
        alphazero::mcts::NetworkOutput output;
        output.value = 0.5f;  // Fixed value for testing
        output.policy = std::vector<float>(9, 1.0f / 9);  // Uniform policy
        outputs.push_back(output);
    }
    
    // Simulate network delay
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    
    return outputs;
}

// Test fixture
class MCTSEvaluatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        evaluator = std::make_unique<alphazero::mcts::MCTSEvaluator>(
            mockInference, 4, std::chrono::milliseconds(10));
        evaluator->start();
    }
    
    void TearDown() override {
        evaluator->stop();
    }
    
    std::unique_ptr<alphazero::mcts::MCTSEvaluator> evaluator;
};

// Test single evaluation
TEST_F(MCTSEvaluatorTest, SingleEvaluation) {
    auto game_state = std::make_unique<MockGameState>();
    auto node = std::make_unique<alphazero::mcts::MCTSNode>(game_state->clone());
    
    auto future = evaluator->evaluateState(node.get(), std::move(game_state));
    auto result = future.get();  // Wait for evaluation
    
    EXPECT_FLOAT_EQ(result.value, 0.5f);
    EXPECT_EQ(result.policy.size(), 9);
    for (const auto& p : result.policy) {
        EXPECT_FLOAT_EQ(p, 1.0f / 9);
    }
    
    // Wait a bit for metrics to update
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    
    EXPECT_EQ(evaluator->getTotalEvaluations(), 1);
    EXPECT_FLOAT_EQ(evaluator->getAverageBatchSize(), 1.0f);
}

// Test batch evaluation
TEST_F(MCTSEvaluatorTest, BatchEvaluation) {
    constexpr int num_requests = 10;
    std::vector<std::unique_ptr<alphazero::mcts::MCTSNode>> nodes;
    std::vector<std::future<alphazero::mcts::NetworkOutput>> futures;
    
    // Submit multiple evaluation requests quickly
    for (int i = 0; i < num_requests; ++i) {
        auto game_state = std::make_unique<MockGameState>();
        auto node = std::make_unique<alphazero::mcts::MCTSNode>(game_state->clone());
        nodes.push_back(std::move(node));
        
        futures.push_back(evaluator->evaluateState(
            nodes.back().get(), std::make_unique<MockGameState>()));
    }
    
    // Collect all results
    for (auto& future : futures) {
        auto result = future.get();
        EXPECT_FLOAT_EQ(result.value, 0.5f);
    }
    
    // Wait a bit for metrics to update
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    
    EXPECT_EQ(evaluator->getTotalEvaluations(), num_requests);
    EXPECT_GT(evaluator->getAverageBatchSize(), 1.0f);  // Should be batched
}

// Test timeout behavior
TEST_F(MCTSEvaluatorTest, TimeoutBehavior) {
    // Submit a single request
    auto game_state = std::make_unique<MockGameState>();
    auto node = std::make_unique<alphazero::mcts::MCTSNode>(game_state->clone());
    
    auto future = evaluator->evaluateState(node.get(), std::move(game_state));
    
    // Wait for timeout to occur
    std::this_thread::sleep_for(std::chrono::milliseconds(15)); // > timeout (10ms)
    
    // Should have processed the single request despite not filling the batch
    auto result = future.get();
    EXPECT_FLOAT_EQ(result.value, 0.5f);
    
    // Wait a bit more for metrics to update
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    EXPECT_EQ(evaluator->getTotalEvaluations(), 1);
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }