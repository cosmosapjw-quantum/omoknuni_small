// tests/mcts/mcts_evaluator_test.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>

using namespace alphazero;
using namespace alphazero::mcts;

// Mock neural network inference function for testing
std::vector<NetworkOutput> mockInference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); ++i) {
        NetworkOutput output;
        
        // Create a simple policy distribution - highest probability for action 2
        int action_space_size = states[i]->getActionSpaceSize();
        output.policy.resize(action_space_size, 0.1f / (action_space_size - 2));
        output.policy[2] = 0.9f; // High probability for action 2
        
        // Set a predictable value for testing
        output.value = static_cast<float>(i % 2 == 0 ? 0.8 : -0.8);
        
        outputs.push_back(std::move(output));
    }
    
    // Simulate some computation time to test batching behavior
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    return outputs;
}

// Helper function to create a test game state
std::unique_ptr<core::IGameState> createTestState() {
    return std::make_unique<games::gomoku::GomokuState>(5); // Small 5x5 board for speed
}

// Test fixture
class MCTSEvaluatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mock evaluator with a small batch size and short timeout
        evaluator = std::make_unique<MCTSEvaluator>(mockInference, 4, std::chrono::milliseconds(10));
        evaluator->start();
    }
    
    void TearDown() override {
        if (evaluator) {
            evaluator->stop();
        }
    }
    
    std::unique_ptr<MCTSEvaluator> evaluator;
};

// Test single evaluation
TEST_F(MCTSEvaluatorTest, SingleEvaluation) {
    auto state = createTestState();
    auto node = std::make_unique<MCTSNode>(state->clone());
    
    auto future = evaluator->evaluateState(node.get(), std::move(state));
    
    // Wait for result with a generous timeout
    auto status = future.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(status, std::future_status::ready) << "Evaluation timed out";
    
    auto result = future.get();
    
    // Verify the result matches our mock inference function
    ASSERT_FALSE(result.policy.empty());
    EXPECT_FLOAT_EQ(result.policy[2], 0.9f);
    EXPECT_FLOAT_EQ(result.value, 0.8f);
}

// Test batch processing
TEST_F(MCTSEvaluatorTest, BatchProcessing) {
    const int NUM_REQUESTS = 10;
    std::vector<std::unique_ptr<MCTSNode>> nodes;
    std::vector<std::future<NetworkOutput>> futures;
    
    // Create and submit multiple evaluation requests
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto state = createTestState();
        nodes.push_back(std::make_unique<MCTSNode>(state->clone()));
        futures.push_back(evaluator->evaluateState(nodes.back().get(), std::move(state)));
    }
    
    // Wait for all results with timeout
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto status = futures[i].wait_for(std::chrono::seconds(1));
        ASSERT_EQ(status, std::future_status::ready) << "Evaluation " << i << " timed out";
        
        auto result = futures[i].get();
        ASSERT_FALSE(result.policy.empty());
        EXPECT_FLOAT_EQ(result.policy[2], 0.9f);
        EXPECT_FLOAT_EQ(result.value, (i % 2 == 0 ? 0.8f : -0.8f));
    }
    
    // Verify batch statistics
    EXPECT_GT(evaluator->getAverageBatchSize(), 1.0f) << "Batching isn't occurring";
    EXPECT_GE(evaluator->getTotalEvaluations(), NUM_REQUESTS);
}

// Test handling of slow inference
TEST_F(MCTSEvaluatorTest, SlowInference) {
    // Create a slow mock inference function
    auto slowInference = [](const std::vector<std::unique_ptr<core::IGameState>>& states) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate slow processing
        return mockInference(states);
    };
    
    // Create a new evaluator with the slow inference function
    auto slowEvaluator = std::make_unique<MCTSEvaluator>(slowInference, 4, std::chrono::milliseconds(10));
    slowEvaluator->start();
    
    const int NUM_REQUESTS = 6;
    std::vector<std::unique_ptr<MCTSNode>> nodes;
    std::vector<std::future<NetworkOutput>> futures;
    
    // Create and submit multiple evaluation requests
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto state = createTestState();
        nodes.push_back(std::make_unique<MCTSNode>(state->clone()));
        futures.push_back(slowEvaluator->evaluateState(nodes.back().get(), std::move(state)));
    }
    
    // Wait for all results with a longer timeout
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto status = futures[i].wait_for(std::chrono::seconds(2));
        ASSERT_EQ(status, std::future_status::ready) << "Evaluation " << i << " timed out";
        
        auto result = futures[i].get();
        ASSERT_FALSE(result.policy.empty());
    }
    
    slowEvaluator->stop();
}

// Test error handling
TEST_F(MCTSEvaluatorTest, ErrorHandling) {
    // Create a mock inference function that throws exceptions
    auto errorInference = [](const std::vector<std::unique_ptr<core::IGameState>>& states) {
        if (!states.empty()) {
            throw std::runtime_error("Simulated inference error");
        }
        return std::vector<NetworkOutput>();
    };
    
    // Create a new evaluator with the error-throwing inference function
    auto errorEvaluator = std::make_unique<MCTSEvaluator>(errorInference, 4, std::chrono::milliseconds(10));
    errorEvaluator->start();
    
    auto state = createTestState();
    auto node = std::make_unique<MCTSNode>(state->clone());
    
    auto future = errorEvaluator->evaluateState(node.get(), std::move(state));
    
    // Wait for result with timeout
    auto status = future.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(status, std::future_status::ready) << "Evaluation timed out";
    
    // Even with an error, we should get a default result
    auto result = future.get();
    ASSERT_FALSE(result.policy.empty());
    
    errorEvaluator->stop();
}

// Test empty queue behavior
TEST_F(MCTSEvaluatorTest, EmptyQueueTimeout) {
    // Let the evaluator run for a bit with no requests
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Submit a request after timeout
    auto state = createTestState();
    auto node = std::make_unique<MCTSNode>(state->clone());
    
    auto future = evaluator->evaluateState(node.get(), std::move(state));
    
    // Should still get processed quickly
    auto status = future.wait_for(std::chrono::milliseconds(100));
    ASSERT_EQ(status, std::future_status::ready) << "Evaluation timed out";
    
    auto result = future.get();
    ASSERT_FALSE(result.policy.empty());
}

// Test concurrent submit and shutdown
TEST_F(MCTSEvaluatorTest, ConcurrentSubmitAndShutdown) {
    std::atomic<bool> running{true};
    std::atomic<int> submitted{0};
    std::vector<std::unique_ptr<MCTSNode>> nodes;
    std::vector<std::future<NetworkOutput>> futures;
    std::mutex mutex;
    
    // Thread that submits requests
    std::thread submitter([&]() {
        while (running.load()) {
            std::lock_guard<std::mutex> lock(mutex);
            auto state = createTestState();
            nodes.push_back(std::make_unique<MCTSNode>(state->clone()));
            futures.push_back(evaluator->evaluateState(nodes.back().get(), std::move(state)));
            submitted++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Let it run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Signal to stop submitting and wait
    running = false;
    submitter.join();
    
    // Stop the evaluator while some requests might still be in progress
    evaluator->stop();
    
    // Now restart and ensure it works
    evaluator = std::make_unique<MCTSEvaluator>(mockInference, 4, std::chrono::milliseconds(10));
    evaluator->start();
    
    // Try a new request to make sure it's working
    auto state = createTestState();
    auto node = std::make_unique<MCTSNode>(state->clone());
    auto future = evaluator->evaluateState(node.get(), std::move(state));
    
    auto status = future.wait_for(std::chrono::milliseconds(100));
    ASSERT_EQ(status, std::future_status::ready) << "Evaluation after restart timed out";
}