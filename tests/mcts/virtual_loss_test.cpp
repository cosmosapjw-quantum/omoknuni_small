#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include <chrono>

#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include "core/game_export.h"

namespace alphazero {
namespace mcts {
namespace test {

class VirtualLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        // Create test game state
        test_state_ = std::make_unique<games::gomoku::GomokuState>();
        
        // Create root node with expanded children for testing
        root_node_ = MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        root_node_->expand(false, 2.0f, 6.0f);
        
        // Ensure we have children to test with
        auto children = root_node_->getChildren();
        ASSERT_GE(children.size(), 4) << "Root should have expanded children for testing";
    }
    
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
    std::shared_ptr<MCTSNode> root_node_;
};

TEST_F(VirtualLossTest, OptimizedVirtualLossApplication) {
    // Test that optimized virtual loss (value=1) allows better tree utilization
    
    auto children = root_node_->getChildren();
    auto first_child = children[0];
    
    // Initial state - no virtual loss
    float initial_q = first_child->getValue();
    int initial_visits = first_child->getVisitCount();
    
    // Apply optimized virtual loss (value = 1)
    const float optimized_virtual_loss = 1.0f;
    first_child->applyVirtualLoss(optimized_virtual_loss);
    
    // Check that virtual loss was applied
    float q_after_virtual_loss = first_child->getValue();
    int visits_after_virtual_loss = first_child->getVisitCount();
    
    EXPECT_LT(q_after_virtual_loss, initial_q) 
        << "Q-value should decrease after applying virtual loss";
    EXPECT_GT(visits_after_virtual_loss, initial_visits) 
        << "Visit count should increase to account for virtual loss";
    
    // Remove virtual loss
    first_child->removeVirtualLoss(optimized_virtual_loss);
    
    // Should return to original state
    float q_after_removal = first_child->getValue();
    int visits_after_removal = first_child->getVisitCount();
    
    EXPECT_FLOAT_EQ(q_after_removal, initial_q) 
        << "Q-value should return to original after removing virtual loss";
    EXPECT_EQ(visits_after_removal, initial_visits) 
        << "Visit count should return to original after removing virtual loss";
}

TEST_F(VirtualLossTest, VirtualLossComparisonOldVsNew) {
    // Compare old virtual loss (3.0) vs new optimized (1.0)
    
    auto children = root_node_->getChildren();
    ASSERT_GE(children.size(), 2) << "Need at least 2 children for comparison";
    
    auto child_old = children[0];
    auto child_new = children[1];
    
    // Give both children some initial visits and value
    child_old->updateRecursive(0.5f);
    child_old->updateRecursive(0.3f);
    child_new->updateRecursive(0.5f);
    child_new->updateRecursive(0.3f);
    
    float initial_q = child_old->getValue();
    EXPECT_FLOAT_EQ(child_new->getValue(), initial_q) 
        << "Both children should start with same Q-value";
    
    // Apply different virtual loss values
    const float old_virtual_loss = 3.0f;    // Old aggressive value
    const float new_virtual_loss = 1.0f;    // New optimized value
    
    child_old->applyVirtualLoss(old_virtual_loss);
    child_new->applyVirtualLoss(new_virtual_loss);
    
    float q_old = child_old->getValue();
    float q_new = child_new->getValue();
    
    // New virtual loss should be less aggressive (higher Q-value)
    EXPECT_GT(q_new, q_old) 
        << "Optimized virtual loss should be less aggressive than old value";
    
    // Both should still be less than initial (virtual loss applied)
    EXPECT_LT(q_old, initial_q) << "Old virtual loss should decrease Q-value";
    EXPECT_LT(q_new, initial_q) << "New virtual loss should decrease Q-value";
    
    // Clean up
    child_old->removeVirtualLoss(old_virtual_loss);
    child_new->removeVirtualLoss(new_virtual_loss);
}

TEST_F(VirtualLossTest, ThreadCollisionPrevention) {
    // Test that virtual loss prevents thread collisions in tree expansion
    
    const int THREAD_COUNT = 8;
    const int SELECTIONS_PER_THREAD = 20;
    const float virtual_loss = 1.0f;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<MCTSNode>>> thread_selections(THREAD_COUNT);
    std::atomic<int> completed_threads{0};
    
    // Launch threads that simulate concurrent tree traversal
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            thread_selections[t].reserve(SELECTIONS_PER_THREAD);
            
            for (int s = 0; s < SELECTIONS_PER_THREAD; ++s) {
                // Select child using UCB with virtual loss protection
                auto selected_child = root_node_->selectChild(1.4f, true, virtual_loss);
                
                if (selected_child) {
                    thread_selections[t].push_back(selected_child);
                    
                    // Simulate some work on the selected node
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            completed_threads++;
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(completed_threads.load(), THREAD_COUNT) 
        << "All threads should complete successfully";
    
    // Analyze selection distribution
    std::map<std::shared_ptr<MCTSNode>, int> selection_counts;
    int total_selections = 0;
    
    for (int t = 0; t < THREAD_COUNT; ++t) {
        for (auto& selected : thread_selections[t]) {
            selection_counts[selected]++;
            total_selections++;
        }
    }
    
    EXPECT_GT(total_selections, 0) << "Should have made selections";
    
    // Virtual loss should promote selection diversity
    EXPECT_GE(selection_counts.size(), 2) 
        << "Should select from multiple children (virtual loss promoting diversity)";
    
    // No single child should dominate all selections
    for (const auto& pair : selection_counts) {
        float selection_ratio = static_cast<float>(pair.second) / total_selections;
        EXPECT_LT(selection_ratio, 0.8f) 
            << "No single child should be selected more than 80% of the time";
    }
}

TEST_F(VirtualLossTest, VirtualLossStackingBehavior) {
    // Test behavior when multiple virtual losses are applied to same node
    
    auto children = root_node_->getChildren();
    auto test_child = children[0];
    
    // Give child some initial value
    test_child->updateRecursive(0.6f);
    test_child->updateRecursive(0.4f);
    
    float initial_q = test_child->getValue();
    int initial_visits = test_child->getVisitCount();
    
    const float virtual_loss = 1.0f;
    
    // Apply multiple virtual losses (simulating multiple threads)
    test_child->applyVirtualLoss(virtual_loss);
    float q_after_first = test_child->getValue();
    
    test_child->applyVirtualLoss(virtual_loss);
    float q_after_second = test_child->getValue();
    
    test_child->applyVirtualLoss(virtual_loss);
    float q_after_third = test_child->getValue();
    
    // Each virtual loss should further decrease Q-value
    EXPECT_LT(q_after_first, initial_q) << "First virtual loss should decrease Q";
    EXPECT_LT(q_after_second, q_after_first) << "Second virtual loss should further decrease Q";
    EXPECT_LT(q_after_third, q_after_second) << "Third virtual loss should further decrease Q";
    
    // Remove virtual losses in reverse order
    test_child->removeVirtualLoss(virtual_loss);
    float q_after_remove_one = test_child->getValue();
    EXPECT_FLOAT_EQ(q_after_remove_one, q_after_second) 
        << "Removing one virtual loss should restore to previous state";
    
    test_child->removeVirtualLoss(virtual_loss);
    float q_after_remove_two = test_child->getValue();
    EXPECT_FLOAT_EQ(q_after_remove_two, q_after_first) 
        << "Removing second virtual loss should restore to previous state";
    
    test_child->removeVirtualLoss(virtual_loss);
    float q_after_remove_all = test_child->getValue();
    EXPECT_FLOAT_EQ(q_after_remove_all, initial_q) 
        << "Removing all virtual losses should restore original Q-value";
    
    int final_visits = test_child->getVisitCount();
    EXPECT_EQ(final_visits, initial_visits) 
        << "Visit count should be restored after removing all virtual losses";
}

TEST_F(VirtualLossTest, VirtualLossUCBImpact) {
    // Test how virtual loss affects UCB calculation and selection
    
    auto children = root_node_->getChildren();
    ASSERT_GE(children.size(), 3) << "Need at least 3 children for UCB testing";
    
    // Give children different initial values
    children[0]->updateRecursive(0.8f);  // High value
    children[1]->updateRecursive(0.5f);  // Medium value  
    children[2]->updateRecursive(0.2f);  // Low value
    
    // Without virtual loss, highest value child should be selected
    auto selected_without_virtual_loss = root_node_->selectChild(1.4f, false, 0.0f);
    EXPECT_EQ(selected_without_virtual_loss, children[0]) 
        << "Highest value child should be selected without virtual loss";
    
    // Apply virtual loss to highest value child
    const float virtual_loss = 1.0f;
    children[0]->applyVirtualLoss(virtual_loss);
    
    // Now selection should prefer other children
    auto selected_with_virtual_loss = root_node_->selectChild(1.4f, false, 0.0f);
    EXPECT_NE(selected_with_virtual_loss, children[0]) 
        << "Virtual loss should make other children more attractive";
    
    // Clean up
    children[0]->removeVirtualLoss(virtual_loss);
}

TEST_F(VirtualLossTest, VirtualLossPendingEvaluationIntegration) {
    // Test virtual loss integration with pending evaluation system
    
    auto children = root_node_->getChildren();
    auto test_child = children[0];
    
    // Mark child as pending evaluation
    test_child->tryMarkForEvaluation();
    
    float q_before = test_child->getValue();
    
    // Apply virtual loss while pending
    const float virtual_loss = 1.0f;
    test_child->applyVirtualLoss(virtual_loss);
    
    float q_during_pending = test_child->getValue();
    EXPECT_LT(q_during_pending, q_before) 
        << "Virtual loss should apply even when pending evaluation";
    
    // Complete evaluation
    test_child->clearEvaluationFlag();
    test_child->updateRecursive(0.7f);
    
    // Remove virtual loss after evaluation
    test_child->removeVirtualLoss(virtual_loss);
    
    float q_after = test_child->getValue();
    
    // Should incorporate both the evaluation result and virtual loss removal
    EXPECT_GT(q_after, q_during_pending) 
        << "Q-value should improve after removing virtual loss and evaluation";
}

TEST_F(VirtualLossTest, VirtualLossPerformanceImpact) {
    // Test that optimized virtual loss (1.0) provides better performance than old (3.0)
    
    const int SIMULATION_RUNS = 100;
    const float old_virtual_loss = 3.0f;
    const float new_virtual_loss = 1.0f;
    
    // Simulate selection behavior with old virtual loss
    std::map<std::shared_ptr<MCTSNode>, int> old_selections;
    for (int run = 0; run < SIMULATION_RUNS; ++run) {
        auto children = root_node_->getChildren();
        
        // Apply old virtual loss to most-visited child
        auto most_visited = *std::max_element(children.begin(), children.end(),
            [](const auto& a, const auto& b) {
                return a->getVisitCount() < b->getVisitCount();
            });
        
        most_visited->applyVirtualLoss(old_virtual_loss);
        auto selected = root_node_->selectChild(1.4f, true, old_virtual_loss);
        if (selected) {
            old_selections[selected]++;
        }
        most_visited->removeVirtualLoss(old_virtual_loss);
    }
    
    // Simulate selection behavior with new virtual loss
    std::map<std::shared_ptr<MCTSNode>, int> new_selections;
    for (int run = 0; run < SIMULATION_RUNS; ++run) {
        auto children = root_node_->getChildren();
        
        // Apply new virtual loss to most-visited child
        auto most_visited = *std::max_element(children.begin(), children.end(),
            [](const auto& a, const auto& b) {
                return a->getVisitCount() < b->getVisitCount();
            });
        
        most_visited->applyVirtualLoss(new_virtual_loss);
        auto selected = root_node_->selectChild(1.4f, true, new_virtual_loss);
        if (selected) {
            new_selections[selected]++;
        }
        most_visited->removeVirtualLoss(new_virtual_loss);
    }
    
    // Analyze diversity - new virtual loss should allow better tree utilization
    size_t old_diversity = old_selections.size();
    size_t new_diversity = new_selections.size();
    
    EXPECT_GE(new_diversity, old_diversity) 
        << "New virtual loss should maintain or improve selection diversity";
    
    // New virtual loss should be less aggressive in blocking selections
    int old_total_selections = 0;
    for (const auto& pair : old_selections) {
        old_total_selections += pair.second;
    }
    
    int new_total_selections = 0;
    for (const auto& pair : new_selections) {
        new_total_selections += pair.second;
    }
    
    EXPECT_GE(new_total_selections, old_total_selections) 
        << "New virtual loss should allow more successful selections";
}

TEST_F(VirtualLossTest, EdgeCaseHandling) {
    // Test edge cases for virtual loss handling
    
    auto test_child = root_node_->getChildren()[0];
    
    // Test zero virtual loss
    float initial_q = test_child->getValue();
    test_child->applyVirtualLoss(0.0f);
    EXPECT_FLOAT_EQ(test_child->getValue(), initial_q) 
        << "Zero virtual loss should not change Q-value";
    test_child->removeVirtualLoss(0.0f);
    
    // Test negative virtual loss (should be handled gracefully)
    test_child->applyVirtualLoss(-1.0f);
    // Should not crash and should handle gracefully
    test_child->removeVirtualLoss(-1.0f);
    
    // Test large virtual loss values
    test_child->applyVirtualLoss(100.0f);
    float q_after_large = test_child->getValue();
    EXPECT_LT(q_after_large, initial_q) 
        << "Large virtual loss should decrease Q-value";
    test_child->removeVirtualLoss(100.0f);
    
    // Should return to approximately original value
    float q_after_restore = test_child->getValue();
    EXPECT_NEAR(q_after_restore, initial_q, 0.01f) 
        << "Should restore approximately to original Q-value";
}

} // namespace test
} // namespace mcts
} // namespace alphazero