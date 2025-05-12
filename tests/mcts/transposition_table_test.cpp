// tests/mcts/transposition_table_test.cpp
#include <gtest/gtest.h>
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <memory>
#include <thread>
#include <vector>

using namespace alphazero;

// Simple mock game state for testing
class MockGameState : public core::IGameState {
public:
    MockGameState(uint64_t hash_value = 0)
        : core::IGameState(core::GameType::UNKNOWN), hash_value_(hash_value) {}
    
    std::vector<int> getLegalMoves() const override { return {0, 1, 2}; }
    bool isLegalMove(int action) const override { return action >= 0 && action <= 2; }
    void makeMove(int action) override {}
    bool undoMove() override { return false; }
    bool isTerminal() const override { return false; }
    core::GameResult getGameResult() const override { return core::GameResult::ONGOING; }
    int getCurrentPlayer() const override { return 1; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override { 
        return {}; 
    }
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return {};
    }
    uint64_t getHash() const override { return hash_value_; }
    std::unique_ptr<IGameState> clone() const override { 
        return std::make_unique<MockGameState>(hash_value_); 
    }
    std::string actionToString(int action) const override { return std::to_string(action); }
    std::optional<int> stringToAction(const std::string& moveStr) const override { 
        return std::nullopt; 
    }
    std::string toString() const override { return "MockGameState"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }

private:
    uint64_t hash_value_;
};

class TranspositionTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small transposition table for testing
        table = std::make_unique<mcts::TranspositionTable>(1); // 1 MB
    }
    
    std::unique_ptr<mcts::TranspositionTable> table;
};

// Basic storage and retrieval tests
TEST_F(TranspositionTableTest, BasicOperations) {
    auto state = std::make_unique<MockGameState>(123);
    auto node = std::make_unique<mcts::MCTSNode>(std::move(state));
    
    // Store a node
    table->store(123, node.get(), 0);
    
    // Retrieve it
    auto retrieved = table->get(123);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved, node.get());
    
    // Try to get a non-existent node
    auto nonexistent = table->get(456);
    EXPECT_EQ(nonexistent, nullptr);
    
    // Check hit rate
    EXPECT_FLOAT_EQ(table->hitRate(), 0.5f); // 1 hit, 1 miss
    
    // Clear the table
    table->clear();
    EXPECT_EQ(table->size(), 0);
    EXPECT_EQ(table->get(123), nullptr);
}

// Test replacement policy
TEST_F(TranspositionTableTest, ReplacementPolicy) {
    // Fill the table with nodes
    const int NUM_NODES = 1000; // More than the table's capacity
    std::vector<std::unique_ptr<mcts::MCTSNode>> nodes;
    
    for (int i = 0; i < NUM_NODES; ++i) {
        auto state = std::make_unique<MockGameState>(i);
        auto node = std::make_unique<mcts::MCTSNode>(std::move(state));
        
        // Set different visit counts to test replacement
        for (int j = 0; j < i % 10; ++j) {
            node->update(0.0f);
        }
        
        // Store in table
        table->store(i, node.get(), 0);
        
        // Keep node alive
        nodes.push_back(std::move(node));
    }
    
    // Table should be full but not crash
    EXPECT_GT(table->size(), 0);
    EXPECT_LE(table->size(), table->capacity());
    
    // Entries with higher visit counts should be retained
    int hits = 0;
    for (int i = NUM_NODES - 100; i < NUM_NODES; ++i) {
        if (table->get(i) != nullptr) {
            hits++;
        }
    }
    
    // More recent entries should have higher hit rate
    EXPECT_GT(hits, 50); // At least half of recent entries should be present
}

// Test thread safety
TEST_F(TranspositionTableTest, ThreadSafety) {
    const int NUM_THREADS = 8;
    const int NUM_OPERATIONS = 1000;
    
    std::vector<std::thread> threads;
    std::atomic<int> ready_count(0);
    std::atomic<bool> start_flag(false);
    
    // Create nodes
    std::vector<std::unique_ptr<mcts::MCTSNode>> nodes;
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        auto state = std::make_unique<MockGameState>(i);
        auto node = std::make_unique<mcts::MCTSNode>(std::move(state));
        nodes.push_back(std::move(node));
    }
    
    // Launch threads
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, t, NUM_OPERATIONS, &nodes, &ready_count, &start_flag]() {
            // Signal ready
            ready_count.fetch_add(1);
            
            // Wait for start signal
            while (!start_flag.load()) {
                std::this_thread::yield();
            }
            
            // Perform operations
            for (int i = t; i < NUM_OPERATIONS; i += NUM_THREADS) {
                // Store node
                table->store(i, nodes[i].get(), 0);
                
                // Get some nodes (including some that other threads are working on)
                for (int j = 0; j < 3; ++j) {
                    int idx = (i + j * NUM_THREADS / 4) % NUM_OPERATIONS;
                    table->get(idx);
                }
            }
        });
    }
    
    // Wait for all threads to be ready
    while (ready_count.load() < NUM_THREADS) {
        std::this_thread::yield();
    }
    
    // Start threads
    start_flag.store(true);
    
    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Table should not crash and should contain entries
    EXPECT_GT(table->size(), 0);
}

// Integration test with MCTS engine
TEST_F(TranspositionTableTest, MCTSIntegration) {
    // This test will be implementation-specific and depends on how 
    // the transposition table is integrated with the MCTS engine
    
    // We'll test this in the next phase
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}