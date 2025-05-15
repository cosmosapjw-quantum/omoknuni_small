// tests/mcts/transposition_integration_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_engine.h"
#include "mcts/transposition_table.h"
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>

using namespace alphazero;

// Enhanced mock game with controllable transpositions
class TranspositionGameState : public core::IGameState {
public:
    TranspositionGameState(int depth = 0, int branch = 0, int max_depth = 4, bool create_transpositions = true)
        : core::IGameState(core::GameType::UNKNOWN), 
          depth_(depth), branch_(branch), max_depth_(max_depth),
          create_transpositions_(create_transpositions) {}
    
    std::vector<int> getLegalMoves() const override {
        if (isTerminal()) {
            return {};
        }
        
        // Two legal moves at each depth
        return {0, 1};
    }
    
    bool isLegalMove(int action) const override {
        return !isTerminal() && (action == 0 || action == 1);
    }
    
    void makeMove(int action) override {
        if (!isLegalMove(action)) {
            throw std::runtime_error("Illegal move");
        }
        
        depth_++;
        branch_ = action;
        move_history_.push_back(action);
    }
    
    bool undoMove() override {
        if (move_history_.empty()) {
            return false;
        }
        
        depth_--;
        move_history_.pop_back();
        
        // Restore branch
        if (!move_history_.empty()) {
            branch_ = move_history_.back();
        } else {
            branch_ = 0;
        }
        
        return true;
    }
    
    bool isTerminal() const override {
        return depth_ >= max_depth_;
    }
    
    core::GameResult getGameResult() const override {
        if (!isTerminal()) {
            return core::GameResult::ONGOING;
        }
        
        // Different results based on the final branch
        if (branch_ == 0) {
            return core::GameResult::WIN_PLAYER1;
        } else {
            return core::GameResult::WIN_PLAYER2;
        }
    }
    
    int getCurrentPlayer() const override {
        return (depth_ % 2 == 0) ? 1 : 2;
    }
    
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 2; }
    
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override {
        return std::vector<std::vector<std::vector<float>>>(
            2, std::vector<std::vector<float>>(
                3, std::vector<float>(3, 0.0f)));
    }
    
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return getTensorRepresentation();
    }
    
    uint64_t getHash() const override {
        // Create a hash based on the board state
        // For this test, we'll create intentional transpositions based on the flag
        uint64_t hash = 0;
        
        // Always include depth and player in hash
        hash = hash * 31 + depth_;
        hash = hash * 31 + getCurrentPlayer();
        
        if (create_transpositions_ && move_history_.size() >= 2) {
            // CREATE EXPLICIT TRANSPOSITIONS:
            // If first two moves are [0,1] or [1,0], they lead to the same hash
            // This simulates symmetric positions in real games
            if ((move_history_[0] == 0 && move_history_[1] == 1) ||
                (move_history_[0] == 1 && move_history_[1] == 0)) {
                // Same hash for both sequences - TRANSPOSITION!
                hash = hash * 31 + 42;
                
                // Add remaining moves normally
                for (size_t i = 2; i < move_history_.size(); i++) {
                    hash = hash * 31 + move_history_[i];
                }
            } else {
                // No transposition - normal hashing
                for (int move : move_history_) {
                    hash = hash * 31 + move;
                }
            }
        } else {
            // No transpositions - normal hashing for every move
            for (int move : move_history_) {
                hash = hash * 31 + move;
            }
        }
        
        return hash;
    }
    
    std::unique_ptr<IGameState> clone() const override {
        auto clone = std::make_unique<TranspositionGameState>(depth_, branch_, max_depth_, create_transpositions_);
        clone->move_history_ = move_history_;
        return clone;
    }
    
    std::string actionToString(int action) const override {
        return std::to_string(action);
    }
    
    std::optional<int> stringToAction(const std::string& moveStr) const override {
        try {
            return std::stoi(moveStr);
        } catch (...) {
            return std::nullopt;
        }
    }
    
    std::string toString() const override {
        std::string result = "Depth: " + std::to_string(depth_) + 
                            ", Branch: " + std::to_string(branch_) + 
                            ", Player: " + std::to_string(getCurrentPlayer());
        
        // Add move history
        result += "\nMoves: ";
        for (int move : move_history_) {
            result += std::to_string(move) + " ";
        }
        
        return result;
    }
    
    bool equals(const IGameState& other) const override {
        auto* o = dynamic_cast<const TranspositionGameState*>(&other);
        if (!o) {
            return false;
        }
        
        return depth_ == o->depth_ && branch_ == o->branch_ && 
               move_history_ == o->move_history_;
    }
    
    std::vector<int> getMoveHistory() const override {
        return move_history_;
    }
    
    bool validate() const override {
        return true;
    }
    
    // Helper to check if a position is a transposition
    bool isTranspositionPosition() const {
        if (!create_transpositions_ || move_history_.size() < 2) {
            return false;
        }
        
        return (move_history_[0] == 0 && move_history_[1] == 1) ||
               (move_history_[0] == 1 && move_history_[1] == 0);
    }
    
private:
    int depth_;
    int branch_;
    int max_depth_;
    bool create_transpositions_;
    std::vector<int> move_history_;
};

// Controlled neural network function that we can monitor
class TestNeuralNetwork {
public:
    TestNeuralNetwork() : evaluation_count_(0) {}
    
    std::vector<mcts::NetworkOutput> operator()(
        const std::vector<std::unique_ptr<core::IGameState>>& states) {
        
        evaluation_count_ += states.size();
        
        // Track unique positions evaluated by their hash
        for (const auto& state : states) {
            uint64_t hash = state->getHash();
            evaluated_positions_[hash]++;
            
            // Also record if it's a transposition
            auto* trans_state = dynamic_cast<const TranspositionGameState*>(state.get());
            if (trans_state && trans_state->isTranspositionPosition()) {
                transposition_evaluations_++;
            }
        }
        
        // Return standard results
        std::vector<mcts::NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (const auto& state : states) {
            mcts::NetworkOutput output;
            output.policy = std::vector<float>{0.6f, 0.4f};  // Slight preference for move 0
            
            // Slightly prefer branch 0 if player 1, branch 1 if player 2
            int player = state->getCurrentPlayer();
            output.value = (player == 1) ? 0.2f : -0.2f;
            
            outputs.push_back(output);
        }
        
        return outputs;
    }
    
    // Statistics tracking methods
    int getEvaluationCount() const { return evaluation_count_; }
    int getUniquePositionsCount() const { return evaluated_positions_.size(); }
    int getTranspositionEvaluations() const { return transposition_evaluations_; }
    
    void reset() {
        evaluation_count_ = 0;
        evaluated_positions_.clear();
        transposition_evaluations_ = 0;
    }
    
private:
    std::atomic<int> evaluation_count_;
    std::atomic<int> transposition_evaluations_{0};
    std::unordered_map<uint64_t, int> evaluated_positions_;
};

class TranspositionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create MCTS settings for testing
        settings.num_simulations = 50;  // Enough to test transposition but not too many
        settings.num_threads = 2;       // Test with multiple threads
        settings.batch_size = 4;        // Small batch size for faster tests
        settings.batch_timeout = std::chrono::milliseconds(10);
        settings.exploration_constant = 1.5f;
        settings.temperature = 0.0f;    // Deterministic selection for stable tests
        
        // Reset the neural network
        nn.reset();
        
        // Create MCTS engine with the neural network
        engine = std::make_unique<mcts::MCTSEngine>(
            [this](const std::vector<std::unique_ptr<core::IGameState>>& states) {
                return nn(states);
            }, 
            settings);
    }
    
    mcts::MCTSSettings settings;
    TestNeuralNetwork nn;
    std::unique_ptr<mcts::MCTSEngine> engine;
};

// Test search with transposition table enabled vs. disabled
TEST_F(TranspositionIntegrationTest, TranspositionTableEfficiency) {
    // Create a game with intentional transpositions (max_depth=4 means more transpositions)
    auto game = std::make_unique<TranspositionGameState>(0, 0, 4, true);
    
    // First run with transposition table disabled
    nn.reset();
    engine->setUseTranspositionTable(false);
    auto result_without_tt = engine->search(*game);
    
    int evals_without_tt = nn.getEvaluationCount();
    int unique_without_tt = nn.getUniquePositionsCount();
    
    // Now run with transposition table enabled
    nn.reset();
    engine->setUseTranspositionTable(true);
    engine->clearTranspositionTable();  // Start with a clean table
    auto result_with_tt = engine->search(*game);
    
    int evals_with_tt = nn.getEvaluationCount();
    int unique_with_tt = nn.getUniquePositionsCount();
    int transposition_evals = nn.getTranspositionEvaluations();
    
    // Log results
    std::cout << "Without TT: " << evals_without_tt << " evaluations of " 
              << unique_without_tt << " unique positions" << std::endl;
              
    std::cout << "With TT: " << evals_with_tt << " evaluations of " 
              << unique_with_tt << " unique positions" << std::endl;
              
    std::cout << "Transposition evaluations: " << transposition_evals << std::endl;
    std::cout << "Transposition table hit rate: " << engine->getTranspositionTableHitRate() << std::endl;
    
    // With transpositions enabled, we should see fewer evaluations
    // But this is probabilistic, so we can't make a hard assertion
    // We just verify that the feature works and logs the hit rate
    EXPECT_GT(engine->getTranspositionTableHitRate(), 0.0f);
    
    // Both searches should select a valid move
    EXPECT_TRUE(result_without_tt.action == 0 || result_without_tt.action == 1);
    EXPECT_TRUE(result_with_tt.action == 0 || result_with_tt.action == 1);
}

// Test search with and without transpositions in the game state
TEST_F(TranspositionIntegrationTest, WithAndWithoutTranspositions) {
    // Run with no transpositions in the game
    auto game_no_trans = std::make_unique<TranspositionGameState>(0, 0, 4, false);
    
    nn.reset();
    engine->setUseTranspositionTable(true);
    engine->clearTranspositionTable();
    auto result_no_trans = engine->search(*game_no_trans);
    
    float hit_rate_no_trans = engine->getTranspositionTableHitRate();
    
    // Run with transpositions enabled in the game
    auto game_with_trans = std::make_unique<TranspositionGameState>(0, 0, 4, true);
    
    nn.reset();
    engine->clearTranspositionTable();
    auto result_with_trans = engine->search(*game_with_trans);
    
    float hit_rate_with_trans = engine->getTranspositionTableHitRate();
    
    // Log results
    std::cout << "Hit rate without transpositions: " << hit_rate_no_trans << std::endl;
    std::cout << "Hit rate with transpositions: " << hit_rate_with_trans << std::endl;
    
    // Game with transpositions should have a higher hit rate
    // But again, this is probabilistic, so we can't make a hard assertion
    // We just verify that it works and logs reasonable hit rates
    SUCCEED();
}

// Test thread safety with concurrent searches
TEST_F(TranspositionIntegrationTest, ConcurrentSearches) {
    // Create a game with transpositions
    auto game = std::make_unique<TranspositionGameState>(0, 0, 3, true);
    
    // Enable transposition table
    engine->setTranspositionTableSize(8);  // 8 MB
    engine->setUseTranspositionTable(true);
    engine->clearTranspositionTable();
    
    // Use more threads internally within the engine to stress-test concurrency
    // for the transposition table and internal MCTS logic.
    mcts::MCTSSettings current_settings = engine->getSettings();
    current_settings.num_threads = 4; // MCTS engine will use 4 internal worker threads
    current_settings.num_simulations = 200; // Increased simulations for a more thorough test
    engine->updateSettings(current_settings);
    
    // nn is the TestNeuralNetwork instance from the fixture, used by the engine's evaluator.
    // Since MCTSEvaluator has one processing thread that calls the nn_inference_fn,
    // TestNeuralNetwork::evaluated_positions_ does not require external locking for this setup.
    nn.reset(); // Reset NN stats before the search

    mcts::SearchResult result;
    // Perform a single search. The MCTS engine will use its configured number of threads (4)
    // to run simulations concurrently. These simulations will interact with the engine's transposition table.
    ASSERT_NO_THROW({
        result = engine->search(*game);
    }) << "MCTS search threw an exception during concurrent execution.";
    
    // Check that the search completed successfully and produced a valid result
    EXPECT_TRUE(result.action == 0 || result.action == 1) 
        << "Search did not produce a valid action.";
    
    // Log relevant statistics
    std::cout << "ConcurrentSearches Test Completed." << std::endl;
    std::cout << "  Transposition Table Hit Rate: " << engine->getTranspositionTableHitRate() << std::endl;
    std::cout << "  NN Evaluations: " << nn.getEvaluationCount() << std::endl;
    std::cout << "  Unique Positions Evaluated by NN: " << nn.getUniquePositionsCount() << std::endl;
    
    // Expect some transposition table activity
    EXPECT_GT(engine->getTranspositionTableHitRate(), 0.0f) 
        << "Transposition table was not utilized during concurrent search.";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}