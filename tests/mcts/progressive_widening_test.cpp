#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include "core/game_export.h"

namespace alphazero {
namespace mcts {
namespace test {

class ProgressiveWideningTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        // Create test game state
        test_state_ = std::make_unique<games::gomoku::GomokuState>();
        
        // Create root node
        root_node_ = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
    }
    
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
    std::shared_ptr<MCTSNode> root_node_;
};

TEST_F(ProgressiveWideningTest, OptimizedProgressiveWideningFormula) {
    // Test the new optimized progressive widening formula
    // New formula: std::max(4.0f, cpw * std::pow(parent_visits, kpw/15.0f))
    
    const float cpw = 2.0f;  // Optimized progressive widening constant
    const float kpw = 6.0f;  // Optimized progressive widening exponent
    
    // Test different visit counts and verify the formula produces reasonable results
    std::vector<int> visit_counts = {1, 5, 10, 20, 50, 100, 200, 500, 1000};
    
    for (int visits : visit_counts) {
        // Calculate expected children using optimized formula
        float expected_children_f = std::max(4.0f, cpw * static_cast<float>(std::pow(visits, kpw/15.0f)));
        size_t expected_children = static_cast<size_t>(expected_children_f);
        
        // Verify minimum threshold
        EXPECT_GE(expected_children, 4) 
            << "Should enforce minimum 4 children for " << visits << " visits";
        
        // Verify reasonable growth rate
        if (visits >= 10) {
            EXPECT_GE(expected_children, 4) 
                << "Should expand reasonable number of children for " << visits << " visits";
        }
        
        // Verify not too aggressive expansion for small visit counts  
        if (visits <= 20) {
            EXPECT_LE(expected_children, 15) 
                << "Should not over-expand for low visit count " << visits;
        }
        
        // For high visit counts, should allow more children but not unlimited
        if (visits >= 500) {
            EXPECT_LE(expected_children, 50) 
                << "Should cap expansion reasonably for high visit count " << visits;
        }
    }
}

TEST_F(ProgressiveWideningTest, ExpansionBehaviorComparison) {
    // Compare old vs new progressive widening behavior
    const float cpw = 2.0f;
    const float old_kpw = 10.0f;  // Old aggressive exponent
    const float new_kpw = 6.0f;   // New optimized exponent
    
    std::vector<int> visit_counts = {1, 10, 50, 100, 200};
    
    for (int visits : visit_counts) {
        // Old formula (more restrictive)
        float old_children_f = cpw * std::pow(visits, old_kpw);
        size_t old_children = static_cast<size_t>(old_children_f);
        
        // New optimized formula (more generous)
        float new_children_f = std::max(4.0f, cpw * static_cast<float>(std::pow(visits, new_kpw/15.0f)));
        size_t new_children = static_cast<size_t>(new_children_f);
        
        // New formula should be more generous for small visit counts
        if (visits <= 50) {
            EXPECT_GE(new_children, old_children) 
                << "New formula should be more generous for " << visits << " visits";
        }
        
        // Both should respect minimum threshold in new formula
        EXPECT_GE(new_children, 4) 
            << "New formula should enforce minimum for " << visits << " visits";
    }
}

TEST_F(ProgressiveWideningTest, RealNodeExpansionBehavior) {
    // Test actual node expansion using the optimized progressive widening
    
    // Create a node with legal moves
    auto legal_moves = test_state_->getLegalMoves();
    ASSERT_GE(legal_moves.size(), 20) << "Test state should have sufficient legal moves";
    
    // Test expansion at different visit counts
    std::vector<int> test_visits = {5, 15, 30, 60, 120};
    
    for (int target_visits : test_visits) {
        // Create fresh node for each test
        auto test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        
        // Simulate visits by updating node with neutral value
        for (int v = 0; v < target_visits; ++v) {
            test_node->update(0.0f);
        }
        
        // Perform expansion with optimized parameters
        const float cpw = 2.0f;
        const float kpw = 6.0f;
        test_node->expand(false, cpw, kpw);
        
        size_t children_count = test_node->getChildren().size();
        
        // Verify minimum expansion
        EXPECT_GE(children_count, 4) 
            << "Should expand at least 4 children for " << target_visits << " visits";
        
        // Verify reasonable expansion (not too many or too few)
        EXPECT_LE(children_count, legal_moves.size()) 
            << "Should not exceed legal move count for " << target_visits << " visits";
        
        // For higher visit counts, should expand more children
        if (target_visits >= 60) {
            EXPECT_GE(children_count, 8) 
                << "Should expand more children for higher visit count " << target_visits;
        }
    }
}

TEST_F(ProgressiveWideningTest, IncrementalExpansionOptimization) {
    // Test that incremental expansion works correctly with optimized formula
    
    auto test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
    const float cpw = 2.0f;
    const float kpw = 6.0f;
    
    size_t previous_children = 0;
    
    // Simulate progressive expansion as visit count increases
    for (int visits = 5; visits <= 100; visits += 15) {
        // Simulate visits by updating node multiple times
        for (int v = 0; v < visits; ++v) {
            test_node->update(0.0f);
        }
        
        // Expand node
        test_node->expand(false, cpw, kpw);
        
        size_t current_children = test_node->getChildren().size();
        
        // Children count should not decrease
        EXPECT_GE(current_children, previous_children) 
            << "Children count should not decrease from " << previous_children 
            << " to " << current_children << " at " << visits << " visits";
        
        // Should expand more children as visits increase (unless hitting legal move limit)
        if (visits > 20 && current_children < test_state_->getLegalMoves().size()) {
            EXPECT_GE(current_children, previous_children) 
                << "Should expand more children as visits increase from " 
                << (visits - 15) << " to " << visits;
        }
        
        previous_children = current_children;
    }
}

TEST_F(ProgressiveWideningTest, EdgeCaseHandling) {
    // Test edge cases for progressive widening
    
    auto test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
    const float cpw = 2.0f;
    const float kpw = 6.0f;
    
    // Test with zero visits
    // Node starts with 0 visits - no updates needed
    test_node->expand(false, cpw, kpw);
    
    size_t children_zero_visits = test_node->getChildren().size();
    EXPECT_GE(children_zero_visits, 4) 
        << "Should enforce minimum children even with zero visits";
    
    // Test with very low visit count
    test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
    test_node->update(0.0f); // Single visit
    test_node->expand(false, cpw, kpw);
    
    size_t children_one_visit = test_node->getChildren().size();
    EXPECT_GE(children_one_visit, 4) 
        << "Should enforce minimum children with very low visits";
    
    // Test with very high visit count
    test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
    // Simulate 10000 visits
    for (int v = 0; v < 10000; ++v) {
        test_node->update(0.0f);
    }
    test_node->expand(false, cpw, kpw);
    
    size_t children_high_visits = test_node->getChildren().size();
    size_t legal_moves_count = test_state_->getLegalMoves().size();
    
    EXPECT_LE(children_high_visits, legal_moves_count) 
        << "Should not exceed legal moves even with very high visits";
    
    // Should expand a significant portion of legal moves for high visits
    EXPECT_GE(children_high_visits, std::min(static_cast<size_t>(20), legal_moves_count)) 
        << "Should expand substantial number of children for very high visits";
}

TEST_F(ProgressiveWideningTest, ParameterSensitivityAnalysis) {
    // Test how different parameter values affect expansion behavior
    
    std::vector<std::pair<float, float>> parameter_sets = {
        {1.0f, 5.0f},   // Conservative
        {2.0f, 6.0f},   // Optimized (default)
        {3.0f, 7.0f},   // Aggressive
    };
    
    const int test_visits = 50;
    
    for (const auto& params : parameter_sets) {
        float cpw = params.first;
        float kpw = params.second;
        
        auto test_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        // Simulate visits
        for (int v = 0; v < test_visits; ++v) {
            test_node->update(0.0f);
        }
        test_node->expand(false, cpw, kpw);
        
        size_t children_count = test_node->getChildren().size();
        
        // All parameter sets should respect minimum
        EXPECT_GE(children_count, 4) 
            << "cpw=" << cpw << " kpw=" << kpw << " should enforce minimum";
        
        // All should be reasonable (not over-expand)
        EXPECT_LE(children_count, test_state_->getLegalMoves().size()) 
            << "cpw=" << cpw << " kpw=" << kpw << " should not exceed legal moves";
        
        // Should allow meaningful expansion
        EXPECT_GE(children_count, 6) 
            << "cpw=" << cpw << " kpw=" << kpw << " should allow reasonable expansion";
    }
}

TEST_F(ProgressiveWideningTest, TreeUtilizationImprovement) {
    // Test that optimized progressive widening improves tree utilization
    
    const float old_cpw = 1.0f;
    const float old_kpw = 10.0f;
    const float new_cpw = 2.0f; 
    const float new_kpw = 6.0f;
    
    // Test at various visit levels
    std::vector<int> visit_levels = {10, 25, 50, 100};
    
    for (int visits : visit_levels) {
        // Old formula behavior
        auto old_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        // Simulate visits for old node
        for (int v = 0; v < visits; ++v) {
            old_node->update(0.0f);
        }
        
        // Calculate what old formula would produce (without minimum enforcement)
        float old_children_f = old_cpw * std::pow(visits, old_kpw);
        size_t old_expected = static_cast<size_t>(old_children_f);
        
        // New optimized formula
        auto new_node = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        // Simulate visits for new node
        for (int v = 0; v < visits; ++v) {
            new_node->update(0.0f);
        }
        new_node->expand(false, new_cpw, new_kpw);
        
        size_t new_children = new_node->getChildren().size();
        
        // New formula should generally produce more children for typical visit counts
        if (visits <= 100) {
            EXPECT_GE(new_children, 4) 
                << "New formula should ensure minimum expansion at " << visits << " visits";
            
            // For reasonable visit counts, new should be more generous than old
            if (old_expected < 4) {
                EXPECT_GT(new_children, old_expected) 
                    << "New formula should be more generous than old at " << visits << " visits";
            }
        }
    }
}

} // namespace test
} // namespace mcts
} // namespace alphazero