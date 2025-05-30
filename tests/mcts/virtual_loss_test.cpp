// tests/mcts/virtual_loss_test.cpp
#include <gtest/gtest.h>
#include "mcts/mcts_node.h"
#include "games/gomoku/gomoku_state.h"
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>

namespace alphazero {
namespace mcts {
namespace test {

// Test fixture for Virtual Loss optimization tests
class VirtualLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple game state (5x5 Gomoku for fast testing)
        test_state_ = std::make_unique<games::gomoku::GomokuState>(5);
        
        // Create root node
        root_node_ = MCTSNode::create(test_state_->clone(), nullptr);
        
        // Expand root with some children
        std::vector<int> actions = {12, 13, 14, 15, 16};  // Center positions
        for (int action : actions) {
            auto child_state = test_state_->clone();
            child_state->makeMove(action);
            auto child = MCTSNode::create(std::move(child_state), root_node_);
            child->setAction(action);
            child->setPriorProbability(0.2f);  // Equal priors
            root_node_->getChildren().push_back(child);
        }
    }
    
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
    std::shared_ptr<MCTSNode> root_node_;
};

TEST_F(VirtualLossTest, OptimizedVirtualLossApplication) {
    // Test that optimized virtual loss (value=1) allows better tree utilization
    
    auto children = root_node_->getChildren();
    auto first_child = children[0];
    
    // Initial state - no virtual loss
    int initial_virtual_loss = first_child->getVirtualLoss();
    
    // Apply optimized virtual loss (count = 1)
    const int optimized_virtual_loss = 1;
    first_child->addVirtualLoss(optimized_virtual_loss);
    
    // Check that virtual loss was applied
    int virtual_loss_after = first_child->getVirtualLoss();
    
    EXPECT_EQ(virtual_loss_after, initial_virtual_loss + optimized_virtual_loss) 
        << "Virtual loss count should increase";
    
    // Remove virtual loss
    first_child->removeVirtualLoss(optimized_virtual_loss);
    
    // Should return to original state
    int virtual_loss_after_removal = first_child->getVirtualLoss();
    
    EXPECT_EQ(virtual_loss_after_removal, initial_virtual_loss) 
        << "Virtual loss should return to initial after removal";
}

TEST_F(VirtualLossTest, VirtualLossComparisonOldVsNew) {
    // Compare old aggressive virtual loss (count=3) vs new optimized (count=1)
    
    auto children = root_node_->getChildren();
    auto child1 = children[0];
    auto child2 = children[1];
    
    // Give them identical starting stats
    for (int i = 0; i < 10; ++i) {
        child1->update(0.4f);  // Average value
        child2->update(0.4f);
    }
    
    // Apply different virtual loss values
    const int old_virtual_loss = 3;  // Old aggressive value
    const int new_virtual_loss = 1;  // New optimized value
    
    child1->addVirtualLoss(old_virtual_loss);
    child2->addVirtualLoss(new_virtual_loss);
    
    int vl_old = child1->getVirtualLoss();
    int vl_new = child2->getVirtualLoss();
    
    // Check virtual loss counts
    EXPECT_EQ(vl_old, old_virtual_loss) 
        << "Old virtual loss count should be applied";
    EXPECT_EQ(vl_new, new_virtual_loss) 
        << "New virtual loss count should be applied";
    EXPECT_LT(vl_new, vl_old) 
        << "Optimized virtual loss should be less than old value";
    
    // Clean up
    child1->removeVirtualLoss(old_virtual_loss);
    child2->removeVirtualLoss(new_virtual_loss);
}

TEST_F(VirtualLossTest, ThreadCollisionPrevention) {
    // Test that virtual loss helps prevent thread collisions
    
    // Track which children get selected
    std::map<int, int> selection_counts;
    const int num_selections = 100;
    
    // Simulate thread selection with virtual loss
    for (int i = 0; i < num_selections; ++i) {
        // Select best child using the node's actual selection method
        auto selected = root_node_->selectChild(1.5f);  // exploration constant
        if (!selected) continue;
        
        // Find which child was selected
        int selected_idx = -1;
        for (size_t j = 0; j < root_node_->getChildren().size(); ++j) {
            if (root_node_->getChildren()[j] == selected) {
                selected_idx = j;
                break;
            }
        }
        
        if (selected_idx >= 0) {
            // Apply virtual loss to selected child (simulating thread selection)
            selected->addVirtualLoss();
            selection_counts[selected_idx]++;
            
            // Remove virtual loss after "evaluation" (every 5 selections)
            if (i % 5 == 4) {
                for (auto& child : root_node_->getChildren()) {
                    if (child->getVirtualLoss() > 0) {
                        child->removeVirtualLoss();
                    }
                    child->update(0.5f);  // Simulate evaluation result
                }
            }
        }
    }
    
    // Check that selections were distributed (not all to one child)
    EXPECT_GE(selection_counts.size(), 2) 
        << "Should select from multiple children (virtual loss promoting diversity)";
    
    // No single child should dominate
    int max_selections = 0;
    for (const auto& pair : selection_counts) {
        max_selections = std::max(max_selections, pair.second);
    }
    float selection_ratio = static_cast<float>(max_selections) / num_selections;
    EXPECT_LT(selection_ratio, 0.8f) 
        << "No single child should be selected more than 80% of the time";
}

TEST_F(VirtualLossTest, VirtualLossStackingBehavior) {
    // Test multiple virtual losses stacking correctly
    
    auto child = root_node_->getChildren()[0];
    
    // Initial state
    int initial_vl = child->getVirtualLoss();
    EXPECT_EQ(initial_vl, 0) << "Should start with no virtual loss";
    
    // Apply multiple virtual losses (simulating multiple threads)
    child->addVirtualLoss();
    int vl_after_first = child->getVirtualLoss();
    
    child->addVirtualLoss();
    int vl_after_second = child->getVirtualLoss();
    
    child->addVirtualLoss();
    int vl_after_third = child->getVirtualLoss();
    
    // Each virtual loss should increase the count
    EXPECT_EQ(vl_after_first, 1) 
        << "First virtual loss should set count to 1";
    EXPECT_EQ(vl_after_second, 2) 
        << "Second virtual loss should set count to 2";
    EXPECT_EQ(vl_after_third, 3) 
        << "Third virtual loss should set count to 3";
    
    // Remove all virtual losses
    child->removeVirtualLoss(3);
    int vl_after_removal = child->getVirtualLoss();
    
    EXPECT_EQ(vl_after_removal, 0) 
        << "Virtual loss count should return to 0 after removing all";
}

TEST_F(VirtualLossTest, VirtualLossUCBImpact) {
    // Test that virtual loss affects UCB selection
    
    // Give all children identical stats
    for (auto& child : root_node_->getChildren()) {
        for (int i = 0; i < 10; ++i) {
            child->update(0.5f);
        }
    }
    
    // Apply virtual loss to first child
    auto first_child = root_node_->getChildren()[0];
    first_child->addVirtualLoss(3);  // Apply significant virtual loss
    
    // Update root visit count for UCB calculation
    for (int i = 0; i < 50; ++i) {
        root_node_->update(0.5f);
    }
    
    // Select best child - should avoid the one with virtual loss
    auto selected = root_node_->selectChild(1.5f);
    
    // The selected child should not be the one with virtual loss
    // (unless all have virtual loss or other factors override)
    EXPECT_NE(selected, first_child) 
        << "Selection should avoid child with high virtual loss";
    
    // Clean up
    first_child->removeVirtualLoss(3);
}

TEST_F(VirtualLossTest, VirtualLossPendingEvaluationIntegration) {
    // Test interaction between virtual loss and pending evaluation
    
    auto child = root_node_->getChildren()[0];
    
    // Mark as pending evaluation
    child->markEvaluationPending();
    EXPECT_TRUE(child->hasPendingEvaluation());
    
    int vl_before = child->getVirtualLoss();
    
    // Apply virtual loss while pending
    child->addVirtualLoss();
    
    int vl_during_pending = child->getVirtualLoss();
    
    EXPECT_GT(vl_during_pending, vl_before) 
        << "Virtual loss should apply even when pending evaluation";
    
    // Clear pending and remove virtual loss
    child->clearPendingEvaluation();
    child->removeVirtualLoss();
    
    int vl_after = child->getVirtualLoss();
    
    EXPECT_EQ(vl_after, vl_before) 
        << "Virtual loss should return to original after clearing";
}

TEST_F(VirtualLossTest, VirtualLossPerformanceImpact) {
    // Test performance characteristics of virtual loss operations
    
    auto child = root_node_->getChildren()[0];
    
    // Measure time for adding/removing virtual loss
    const int num_operations = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        child->addVirtualLoss();
        child->removeVirtualLoss();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time_us = static_cast<float>(duration.count()) / (2 * num_operations);
    
    // Virtual loss operations should be very fast (sub-microsecond)
    EXPECT_LT(avg_time_us, 1.0f) 
        << "Virtual loss operations should be sub-microsecond";
    
    std::cout << "Average time per virtual loss operation: " << avg_time_us << " µs" << std::endl;
}

TEST_F(VirtualLossTest, EdgeCaseHandling) {
    // Test edge cases and boundary conditions
    
    auto child = root_node_->getChildren()[0];
    
    // Test negative virtual loss (should be ignored)
    int initial_vl = child->getVirtualLoss();
    child->addVirtualLoss(-1);
    EXPECT_EQ(child->getVirtualLoss(), initial_vl) 
        << "Negative virtual loss should be ignored";
    
    // Test zero virtual loss
    child->addVirtualLoss(0);
    EXPECT_EQ(child->getVirtualLoss(), initial_vl) 
        << "Zero virtual loss should have no effect";
    
    // Test very large virtual loss
    child->addVirtualLoss(1000);
    int vl_after_large = child->getVirtualLoss();
    EXPECT_EQ(vl_after_large, initial_vl + 1000) 
        << "Large virtual loss should be tracked correctly";
    
    // Test removal of more virtual loss than applied
    child->removeVirtualLoss(2000);
    EXPECT_EQ(child->getVirtualLoss(), 0) 
        << "Virtual loss count should never go negative";
}

} // namespace test
} // namespace mcts
} // namespace alphazero