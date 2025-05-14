// tests/integration/self_play_integration_test.cpp
#include "mcts/mcts_engine.h"
#include "mcts/mcts_evaluator.h"
#include "games/gomoku/gomoku_state.h"
#include "nn/neural_network.h"
#include "selfplay/self_play_manager.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

using namespace alphazero;
using namespace alphazero::mcts;
using namespace alphazero::games;
using ::testing::_;

// Mock neural network for testing
class MockNeuralNetwork : public nn::NeuralNetwork {
public:
    // Constructor with configurable board size and possible delay
    MockNeuralNetwork(int board_size = 5, int delay_ms = 0)
        : board_size_(board_size), delay_ms_(delay_ms) {}
    
    std::vector<NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        
        // Optional delay to simulate network latency
        if (delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        }
        
        std::vector<NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); ++i) {
            NetworkOutput output;
            
            // Create a reasonable policy
            int action_space_size = states[i]->getActionSpaceSize();
            output.policy.resize(action_space_size, 0.0f);
            
            // Favor center and near-center positions
            for (int pos = 0; pos < action_space_size; ++pos) {
                int row = pos / board_size_;
                int col = pos % board_size_;
                
                // Distance from center (normalized to 0-1)
                float center_row = board_size_ / 2.0f;
                float center_col = board_size_ / 2.0f;
                float dist = std::sqrt(std::pow(row - center_row, 2) + std::pow(col - center_col, 2));
                float max_dist = std::sqrt(2 * std::pow(board_size_ / 2.0f, 2));
                float normalized_dist = dist / max_dist;
                
                // Closer to center = higher probability
                output.policy[pos] = 1.0f - normalized_dist;
                
                // Add some legal move constraints for realism
                if (states[i]->isLegalMove(pos)) {
                    output.policy[pos] *= 2.0f;  // Boost legal moves
                } else {
                    output.policy[pos] = 0.0f;   // Zero out illegal moves
                }
            }
            
            // Normalize policy
            float sum = 0.0f;
            for (float p : output.policy) {
                sum += p;
            }
            
            if (sum > 0.0f) {
                for (float& p : output.policy) {
                    p /= sum;
                }
            } else {
                // Fallback to uniform over legal moves
                auto legal_moves = states[i]->getLegalMoves();
                if (!legal_moves.empty()) {
                    float uniform_prob = 1.0f / legal_moves.size();
                    for (int move : legal_moves) {
                        output.policy[move] = uniform_prob;
                    }
                }
            }
            
            // Set a reasonable value
            output.value = 0.0f;  // Neutral evaluation for test
            
            outputs.push_back(std::move(output));
        }
        
        // Track metrics for test validation
        total_calls_++;
        total_states_ += states.size();
        batch_sizes_.push_back(states.size());
        
        return outputs;
    }
    
    void save(const std::string& path) override {
        // Mock implementation
    }
    
    void load(const std::string& path) override {
        // Mock implementation
    }
    
    std::vector<int64_t> getInputShape() const override {
        return {17, board_size_, board_size_};  // Gomoku enhanced representation
    }
    
    int64_t getPolicySize() const override {
        return board_size_ * board_size_;  // Gomoku policy space
    }
    
    // Helper methods for testing
    int getTotalCalls() const { return total_calls_.load(); }
    int getTotalStates() const { return total_states_.load(); }
    float getAverageBatchSize() const {
        if (batch_sizes_.empty()) return 0.0f;
        float sum = 0.0f;
        for (size_t size : batch_sizes_) {
            sum += static_cast<float>(size);
        }
        return sum / batch_sizes_.size();
    }
    
private:
    int board_size_;
    int delay_ms_;
    std::atomic<int> total_calls_{0};
    std::atomic<int> total_states_{0};
    std::vector<size_t> batch_sizes_;
};

// Test fixture for self-play integration tests
class SelfPlayIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a shared neural network for both players
        neural_net_ = std::make_shared<MockNeuralNetwork>(board_size_);
    }
    
    void TearDown() override {
        // Clean up
    }
    
    // Helper to run a complete self-play game manually
    void runManualSelfPlayGame(bool with_timeout = false) {
        // Configure MCTS settings
        mcts::MCTSSettings mcts_settings;
        mcts_settings.num_simulations = 50;  // Small enough for tests
        mcts_settings.num_threads = 2;
        mcts_settings.batch_size = 8;
        mcts_settings.batch_timeout = std::chrono::milliseconds(with_timeout ? 200 : 50);
        mcts_settings.add_dirichlet_noise = false;  // Disable for deterministic tests
        
        // Create two MCTS engines
        auto engine1 = std::make_unique<mcts::MCTSEngine>(neural_net_, mcts_settings);
        auto engine2 = std::make_unique<mcts::MCTSEngine>(neural_net_, mcts_settings);
        
        // Create game state
        auto state = std::make_unique<gomoku::GomokuState>(board_size_);
        
        // Play until terminal or max moves
        const int max_moves = board_size_ * board_size_;
        int move_count = 0;
        
        // Use a future with timeout to detect stalling
        auto game_future = std::async(std::launch::async, [&]() {
            while (!state->isTerminal() && move_count < max_moves) {
                // Select engine based on current player
                auto& engine = (state->getCurrentPlayer() == 1) ? *engine1 : *engine2;
                
                // Run search with timeout detection
                auto search_future = std::async(std::launch::async, [&]() {
                    return engine.search(*state);
                });
                
                // Wait for search with timeout
                auto search_status = search_future.wait_for(std::chrono::seconds(10));
                ASSERT_EQ(search_status, std::future_status::ready) 
                    << "MCTS search timed out after 10 seconds";
                
                auto result = search_future.get();
                
                // Make move
                ASSERT_GE(result.action, 0) << "Invalid action returned from search";
                state->makeMove(result.action);
                move_count++;
                
                // Print progress
                if (move_count % 3 == 0) {
                    std::cout << "Completed " << move_count << " moves" << std::endl;
                }
            }
            
            return move_count;
        });
        
        // Wait for game to complete with a reasonable timeout
        auto status = game_future.wait_for(std::chrono::seconds(30));
        ASSERT_EQ(status, std::future_status::ready) << "Self-play game timed out after 30 seconds";
        
        int final_move_count = game_future.get();
        std::cout << "Game completed with " << final_move_count << " moves" << std::endl;
    }
    
    // Helper to run a self-play game using SelfPlayManager
    selfplay::GameData runSelfPlayManagerGame() {
        // Configure MCTS settings
        mcts::MCTSSettings mcts_settings;
        mcts_settings.num_simulations = 50;  // Small enough for tests
        mcts_settings.num_threads = 2;
        mcts_settings.batch_size = 8;
        mcts_settings.batch_timeout = std::chrono::milliseconds(50);
        mcts_settings.add_dirichlet_noise = false;  // Disable for deterministic tests
        
        // Configure self-play settings
        selfplay::SelfPlaySettings self_play_settings;
        self_play_settings.mcts_settings = mcts_settings;
        self_play_settings.num_parallel_games = 1; // Just one game for testing
        self_play_settings.max_moves = board_size_ * board_size_; // Limit to board size
        
        // Create self-play manager
        selfplay::SelfPlayManager manager(neural_net_, self_play_settings);
        
        // Generate a single game with timeout detection
        auto game_future = std::async(std::launch::async, [&]() {
            return manager.generateGames(core::GameType::GOMOKU, 1, board_size_);
        });
        
        // Wait for game to complete with a reasonable timeout
        auto status = game_future.wait_for(std::chrono::seconds(30));
        ASSERT_EQ(status, std::future_status::ready) 
            << "SelfPlayManager game generation timed out after 30 seconds";
        
        auto games = game_future.get();
        EXPECT_EQ(games.size(), 1) << "Expected exactly one game";
        
        return games.empty() ? selfplay::GameData() : games[0];
    }
    
    std::shared_ptr<MockNeuralNetwork> neural_net_;
    const int board_size_ = 9;  // Medium size for tests
};

// Test manual self-play with normal settings
TEST_F(SelfPlayIntegrationTest, ManualSelfPlay) {
    runManualSelfPlayGame(false);
    
    // Verify neural network metrics
    EXPECT_GT(neural_net_->getTotalCalls(), 0);
    EXPECT_GT(neural_net_->getTotalStates(), 0);
    EXPECT_GT(neural_net_->getAverageBatchSize(), 1.0f) 
        << "Average batch size too low, batching may not be working";
}

// Test manual self-play with larger timeouts
TEST_F(SelfPlayIntegrationTest, ManualSelfPlayWithTimeouts) {
    runManualSelfPlayGame(true);
    
    // Similar checks as before
    EXPECT_GT(neural_net_->getTotalCalls(), 0);
    EXPECT_GT(neural_net_->getTotalStates(), 0);
}

// Test self-play using SelfPlayManager
TEST_F(SelfPlayIntegrationTest, SelfPlayManager) {
    auto game_data = runSelfPlayManagerGame();
    
    // Verify game data
    EXPECT_GT(game_data.moves.size(), 0) << "Game should have some moves";
    EXPECT_EQ(game_data.policies.size(), game_data.moves.size()) 
        << "Each move should have a corresponding policy";
    EXPECT_EQ(game_data.game_type, core::GameType::GOMOKU);
    EXPECT_EQ(game_data.board_size, board_size_);
    
    // Verify neural network metrics
    EXPECT_GT(neural_net_->getTotalCalls(), 0);
    EXPECT_GT(neural_net_->getTotalStates(), 0);
    EXPECT_GT(neural_net_->getAverageBatchSize(), 1.0f) 
        << "Average batch size too low, batching may not be working";
}

// Test with slow neural network responses
TEST_F(SelfPlayIntegrationTest, SlowNeuralNetwork) {
    // Create a neural network with artificial delay
    auto slow_net = std::make_shared<MockNeuralNetwork>(board_size_, 20); // 20ms delay
    neural_net_ = slow_net;
    
    auto game_data = runSelfPlayManagerGame();
    
    // Verify game completed successfully despite delays
    EXPECT_GT(game_data.moves.size(), 0) << "Game should have moves even with slow network";
    EXPECT_GT(slow_net->getTotalCalls(), 0);
    EXPECT_GT(slow_net->getAverageBatchSize(), 1.0f) 
        << "Batching should still work with slow network";
}

// Test concurrent self-play games 
TEST_F(SelfPlayIntegrationTest, ConcurrentSelfPlay) {
    // Configure MCTS settings for faster play
    mcts::MCTSSettings mcts_settings;
    mcts_settings.num_simulations = 30;  // Even smaller for multiple games
    mcts_settings.num_threads = 2;
    mcts_settings.batch_size = 8;
    mcts_settings.batch_timeout = std::chrono::milliseconds(50);
    
    // Configure self-play settings with multiple parallel games
    selfplay::SelfPlaySettings self_play_settings;
    self_play_settings.mcts_settings = mcts_settings;
    self_play_settings.num_parallel_games = 3; // Run 3 games in parallel
    self_play_settings.max_moves = board_size_ * board_size_;
    
    // Create self-play manager
    selfplay::SelfPlayManager manager(neural_net_, self_play_settings);
    
    // Generate games with timeout detection
    auto game_future = std::async(std::launch::async, [&]() {
        return manager.generateGames(core::GameType::GOMOKU, 3, board_size_);
    });
    
    // Wait for games to complete with a reasonable timeout
    auto status = game_future.wait_for(std::chrono::seconds(60));
    ASSERT_EQ(status, std::future_status::ready) 
        << "Concurrent self-play games timed out after 60 seconds";
    
    auto games = game_future.get();
    EXPECT_EQ(games.size(), 3) << "Expected exactly three games";
    
    // Verify all games have moves
    for (const auto& game : games) {
        EXPECT_GT(game.moves.size(), 0) << "Each game should have moves";
    }
    
    // Verify neural network was used effectively
    EXPECT_GT(neural_net_->getTotalCalls(), 0);
    EXPECT_GT(neural_net_->getAverageBatchSize(), 1.0f) 
        << "Batching should be working in concurrent games";
}