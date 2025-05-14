// tests/mcts/transposition_table_test.cpp
#include <gtest/gtest.h>
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <iostream>
#include <optional>

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
    const int NUM_THREADS = 8;  // Increased to really stress the table
    const int NUM_OPERATIONS = 1000;  // Increased to really stress the table
    
    std::vector<std::thread> threads;
    std::atomic<int> ready_count(0);
    std::atomic<bool> start_flag(false);
    std::atomic<int> completed_threads(0);
    std::atomic<int> error_count(0);
    
    // Create a larger table for this test
    table = std::make_unique<mcts::TranspositionTable>(16); // 16 MB
    
    // Create nodes
    std::vector<std::unique_ptr<mcts::MCTSNode>> nodes;
    nodes.reserve(NUM_OPERATIONS);
    
    // Create nodes with unique hash values
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        try {
            auto state = std::make_unique<MockGameState>(i * 10 + 1);  // Ensure hash values don't collide
            if (state) {
                auto node = std::make_unique<mcts::MCTSNode>(std::move(state));
                if (node) {
                    nodes.push_back(std::move(node));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception creating node: " << e.what() << std::endl;
        }
    }
    
    // Check if we have nodes
    if (nodes.empty()) {
        FAIL() << "Failed to create test nodes";
        return;
    }
    
    // Set actual operations to match available nodes
    const int ACTUAL_OPS = static_cast<int>(nodes.size());
    
    // Stress test options
    enum class OperationType { STORE, GET, CLEAR };
    
    // Launch threads that perform different types of operations
    for (int t = 0; t < NUM_THREADS; ++t) {
        // Assign different operation types to different threads
        OperationType opType;
        if (t % 4 == 0) {
            opType = OperationType::CLEAR;  // 25% threads do clear operations
        } else if (t % 2 == 0) {
            opType = OperationType::GET;    // 25% threads do get operations
        } else {
            opType = OperationType::STORE;  // 50% threads do store operations
        }
        
        threads.emplace_back([this, t, opType, ACTUAL_OPS, &nodes, &ready_count, &start_flag, &completed_threads, &error_count]() {
            try {
                // Signal ready
                ready_count.fetch_add(1);
                
                // Wait for start signal with timeout
                auto start_time = std::chrono::steady_clock::now();
                while (!start_flag.load() && 
                      std::chrono::steady_clock::now() - start_time < std::chrono::seconds(5)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                // Perform operations
                int operations_done = 0;
                auto op_start_time = std::chrono::steady_clock::now();
                
                // Run for a set duration rather than a set number of operations
                // This ensures even threads that do slower operations get enough time
                while (std::chrono::steady_clock::now() - op_start_time < std::chrono::milliseconds(500)) {
                    try {
                        // Use a different pattern for each thread type
                        switch (opType) {
                            case OperationType::STORE: {
                                // Store nodes - use thread ID to pick different nodes
                                int idx = (operations_done * NUM_THREADS + t) % ACTUAL_OPS;
                                if (idx >= 0 && idx < static_cast<int>(nodes.size()) && nodes[idx]) {
                                    table->store(idx * 10 + 1, nodes[idx].get(), 0);
                                }
                                break;
                            }
                            
                            case OperationType::GET: {
                                // Get nodes - use a different pattern to hit different nodes
                                for (int j = 0; j < 5; ++j) {
                                    int idx = ((operations_done * 7 + j * 11 + t * 13) % ACTUAL_OPS) * 10 + 1;
                                    table->get(idx);
                                }
                                break;
                            }
                            
                            case OperationType::CLEAR: {
                                // Clear operations are less frequent
                                if (operations_done % 50 == 0) {
                                    table->clear();
                                }
                                
                                // Also do some gets to make this thread busy
                                for (int j = 0; j < 10; ++j) {
                                    int idx = ((operations_done + j) % ACTUAL_OPS) * 10 + 1;
                                    table->get(idx);
                                }
                                break;
                            }
                        }
                        
                        operations_done++;
                        
                        // Occasionally yield to simulate real-world conditions
                        if (operations_done % 20 == 0) {
                            std::this_thread::yield();
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in thread " << t << " (op type " << static_cast<int>(opType) 
                                  << "): " << e.what() << std::endl;
                        error_count.fetch_add(1);
                    } catch (...) {
                        std::cerr << "Unknown exception in thread " << t << std::endl;
                        error_count.fetch_add(1);
                    }
                }
                
                // Thread completed successfully
                completed_threads.fetch_add(1);
                
            } catch (const std::exception& e) {
                std::cerr << "Thread exception: " << e.what() << std::endl;
                error_count.fetch_add(1);
            } catch (...) {
                std::cerr << "Unknown thread exception" << std::endl;
                error_count.fetch_add(1);
            }
        });
    }
    
    // Wait for all threads to be ready with timeout
    auto wait_start = std::chrono::steady_clock::now();
    while (ready_count.load() < NUM_THREADS && 
          std::chrono::steady_clock::now() - wait_start < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Start threads
    start_flag.store(true);
    
    // Set a time limit for the entire test
    auto test_start = std::chrono::steady_clock::now();
    auto max_test_duration = std::chrono::seconds(3); // Shorter duration to avoid test hanging
    
    // Join threads with a maximum timeout per thread
    for (auto& thread : threads) {
        if (thread.joinable()) {
            // Try to join with a short timeout
            std::thread joiner([&thread]() {
                if (thread.joinable()) thread.join();
            });
            
            // Wait for joiner with a short timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // If joining timed out, detach the joiner
            if (joiner.joinable()) joiner.detach();
            
            // If thread still hasn't joined, detach it
            if (thread.joinable()) thread.detach();
        }
    }
    
    // Give any detached threads a moment to clean up
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Verify all threads completed
    EXPECT_EQ(completed_threads.load(), NUM_THREADS) 
        << "Not all threads completed successfully";
    
    // Verify no errors occurred
    EXPECT_EQ(error_count.load(), 0) 
        << "Errors occurred during thread execution";
    
    // Table should not crash and should contain entries (unless a clear happened last)
    if (table->size() == 0) {
        // If size is 0, make sure we can still use the table
        auto state = std::make_unique<MockGameState>(12345);
        auto node = std::make_unique<mcts::MCTSNode>(std::move(state));
        table->store(12345, node.get(), 0);
        EXPECT_EQ(table->get(12345), node.get());
    } else {
        // If size is not 0, verify the table has entries
        EXPECT_GT(table->size(), 0);
    }
    
    // Additional verification - check if we can do a stress operation after the test
    table->clear();
    EXPECT_EQ(table->size(), 0);
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