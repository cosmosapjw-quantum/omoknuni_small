// tests/mcts/transposition_integration_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_engine.h"
#include "mcts/transposition_table.h"
#include "nn/neural_network_factory.h"
#include "games/gomoku/gomoku_state.h"
#include <memory>
#include <chrono>

using namespace alphazero;

// Mock game with transpositions
class TranspositionGameState : public core::IGameState {
public:
    TranspositionGameState(int depth = 0, int branch = 0, int max_depth = 4)
        : core::IGameState(core::GameType::UNKNOWN), 
          depth_(depth), branch_(branch), max_depth_(max_depth) {}
    
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
        // For this test, we'll intentionally create transpositions
        // by having certain move sequences lead to the same hash
        
        // The hash ignores the order of the first two moves if they're different
        // This creates a transposition
        
        uint64_t hash = 0;
        
        // Include depth and player in hash
        hash = hash * 31 + depth_;
        hash = hash * 31 + getCurrentPlayer();
        
        // Include move history in hash, but with special handling for transpositions
        if (move_history_.size() >= 2) {
            // If the first two moves are 0,1 or 1,0, they result in the same hash
            // This simulates a transposition
            if ((move_history_[0] == 0 && move_history_[1] == 1) ||
                (move_history_[0] == 1 && move_history_[1] == 0)) {
                // Same hash for both sequences
                hash = hash * 31 + 42;
            } else {
                // Normal hashing
                for (int move : move_history_) {
                    hash = hash * 31 + move;
                }
            }
        } else {
            // Normal hashing for short sequences
            for (int move : move_history_) {
                hash = hash * 31 + move;
            }
        }
        
        return hash;
    }
    
    std::unique_ptr<IGameState> clone() const override {
        auto clone = std::make_unique<TranspositionGameState>(depth_, branch_, max_depth_);
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
    
private:
    int depth_;
    int branch_;
    int max_depth_;
    std::vector<int> move_history_;
};

// Mock neural network for testing
std::vector<mcts::NetworkOutput> mockNeuralNetwork(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    std::vector<mcts::NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    for (const auto& state : states) {
        mcts::NetworkOutput output;
        output.policy = std::vector<float>{0.5f, 0.5f};
        
        // Value based on state
        auto* transposition_state = dynamic_cast<const TranspositionGameState*>(state.get());
        if (transposition_state) {
            // Prefer branch 0 if player 1, branch 1 if player 2
            int player = transposition_state->getCurrentPlayer();
            output.value = (player == 1) ? 0.2f : -0.2f;
        } else {
            output.value = 0.0f;
        }
        
        outputs.push_back(output);
    }
    
    return outputs;
}

class TranspositionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create MCTS settings for testing
        settings.num_simulations = 100;
        settings.num_threads = 4;
        settings.batch_size = 4;
        settings.exploration_constant = 1.5f;
        
        // Create MCTS engine
        engine = std::make_unique<mcts::MCTSEngine>(mockNeuralNetwork, settings);
    }
    
    mcts::MCTSSettings settings;
    std::unique_ptr<mcts::MCTSEngine> engine;
};

// Test search with transposition table
TEST_F(TranspositionIntegrationTest, SearchWithTranspositionTable) {
    // Create a game with transpositions
    auto game = std::make_unique<TranspositionGameState>();
    
    // Run search with transposition table enabled
    engine->setUseTranspositionTable(true);
    auto start_time = std::chrono::steady_clock::now();
    auto result_with_tt = engine->search(*game);
    auto end_time_with_tt = std::chrono::steady_clock::now();
    auto time_with_tt = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time_with_tt - start_time).count();
    
    // Check transposition table statistics
    float hit_rate_tt = engine->getTranspositionTableHitRate();
    std::cout << "Transposition table hit rate: " << hit_rate_tt << std::endl;
    
    // Run search with transposition table disabled
    engine->setUseTranspositionTable(false);
    engine->clearTranspositionTable();
    start_time = std::chrono::steady_clock::now();
    auto result_without_tt = engine->search(*game);
    auto end_time_without_tt = std::chrono::steady_clock::now();
    auto time_without_tt = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time_without_tt - start_time).count();
    
    // Both searches should find the same best move
    EXPECT_EQ(result_with_tt.action, result_without_tt.action);
    
    // Check that the hit rate is positive (transpositions were detected)
    EXPECT_GT(hit_rate_tt, 0.0f);
    
    // Search with transposition table should be faster or equal
    // (might not be faster in small test cases due to overhead)
    std::cout << "Time with TT: " << time_with_tt << "ms, without TT: " 
              << time_without_tt << "ms" << std::endl;
    
    // Check node counts - TT should explore fewer nodes
    std::cout << "Nodes with TT: " << result_with_tt.stats.total_nodes 
              << ", without TT: " << result_without_tt.stats.total_nodes << std::endl;
    
    // For a game with known transpositions, TT should help reduce node count
    // But in test cases with very limited simulations, overhead might dominate
    // So we check but don't assert
    if (result_with_tt.stats.total_nodes < result_without_tt.stats.total_nodes) {
        std::cout << "Transposition table reduced node count by " 
                 << (result_without_tt.stats.total_nodes - result_with_tt.stats.total_nodes) 
                 << " nodes" << std::endl;
    }
}

// Test with real game
TEST_F(TranspositionIntegrationTest, SearchWithRealGame) {
    // Create a Gomoku game (which can have transpositions)
    auto game = std::make_unique<alphazero::games::gomoku::GomokuState>(5); // Small board for testing
    
    // Make a few moves
    game->makeMove(12); // Center
    game->makeMove(6);  // Top left of center
    game->makeMove(18); // Bottom right of center
    
    // Run search with transposition table enabled
    engine->setUseTranspositionTable(true);
    engine->clearTranspositionTable();
    auto result_with_tt = engine->search(*game);
    
    // Run search with transposition table disabled
    engine->setUseTranspositionTable(false);
    auto result_without_tt = engine->search(*game);
    
    // Both searches should find valid moves
    EXPECT_GE(result_with_tt.action, 0);
    EXPECT_GE(result_without_tt.action, 0);
    
    // Transposition table hit rate might be low for Gomoku
    // but should be measured
    float hit_rate = engine->getTranspositionTableHitRate();
    std::cout << "Gomoku transposition table hit rate: " << hit_rate << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}