#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>

#include "mcts/burst_coordinator.h"
#include "games/gomoku/gomoku_state.h"
#include "nn/neural_network_factory.h"
#include "core/game_export.h"

namespace alphazero {
namespace mcts {
namespace test {

class MockUnifiedInferenceServer : public UnifiedInferenceServer {
public:
    MockUnifiedInferenceServer() 
        : UnifiedInferenceServer(nullptr, [](){
            UnifiedInferenceServer::ServerConfig config;
            config.target_batch_size = 32;
            config.min_batch_size = 4;
            config.max_batch_size = 128;
            config.max_batch_wait = std::chrono::milliseconds(10);
            config.min_batch_wait = std::chrono::milliseconds(1);
            config.num_worker_threads = 4;
            return config;
        }()) {
        batch_count_ = 0;
        total_requests_ = 0;
    }

    std::vector<NetworkOutput> processRequests(const std::vector<InferenceRequest>& requests) {
        batch_count_++;
        total_requests_ += requests.size();
        
        // Track batch sizes for verification
        batch_sizes_.push_back(requests.size());
        
        // Simulate processing time based on batch size
        auto processing_time = std::chrono::milliseconds(requests.size() / 4 + 2);
        std::this_thread::sleep_for(processing_time);
        
        std::vector<NetworkOutput> outputs;
        outputs.reserve(requests.size());
        
        for (size_t i = 0; i < requests.size(); ++i) {
            NetworkOutput output;
            output.value = (i % 2 == 0) ? 0.7f : -0.7f;
            output.policy.resize(225, 1.0f / 225.0f); // Uniform policy for 15x15 board
            outputs.push_back(std::move(output));
        }
        
        return outputs;
    }
    
    size_t getBatchCount() const { return batch_count_; }
    size_t getTotalRequests() const { return total_requests_; }
    const std::vector<size_t>& getBatchSizes() const { return batch_sizes_; }
    
private:
    std::atomic<size_t> batch_count_;
    std::atomic<size_t> total_requests_;
    std::vector<size_t> batch_sizes_;
};

class BurstCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        // Create mock inference server
        mock_server_ = std::make_shared<MockUnifiedInferenceServer>();
        
        // Create burst coordinator with optimized settings
        BurstCoordinator::BurstConfig config;
        config.target_burst_size = 32;
        config.min_burst_size = 4;
        config.collection_timeout = std::chrono::milliseconds(10);
        config.evaluation_timeout = std::chrono::milliseconds(50);
        config.max_parallel_threads = 8;
        
        coordinator_ = std::make_unique<BurstCoordinator>(mock_server_, config);
        
        // Create test game state
        test_state_ = std::make_unique<games::gomoku::GomokuState>();
    }
    
    void TearDown() override {
        coordinator_.reset();
        mock_server_.reset();
    }
    
    std::shared_ptr<MockUnifiedInferenceServer> mock_server_;
    std::unique_ptr<BurstCoordinator> coordinator_;
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
};

TEST_F(BurstCoordinatorTest, BasicBurstCollection) {
    // Create test nodes and requests
    std::vector<BurstCoordinator::BurstRequest> requests;
    
    for (int i = 0; i < 16; ++i) {
        BurstCoordinator::BurstRequest request;
        request.leaf = MCTSNode::create(test_state_->clone(), nullptr);
        request.state = test_state_->clone();
        requests.push_back(std::move(request));
    }
    
    // Execute burst collection
    auto start_time = std::chrono::steady_clock::now();
    auto results = coordinator_->collectAndEvaluate(requests, 32);
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify results
    EXPECT_EQ(results.size(), 16) << "Should return results for all requests";
    EXPECT_EQ(mock_server_->getTotalRequests(), 16) << "Server should process all requests";
    EXPECT_GE(mock_server_->getBatchCount(), 1) << "Should have at least one batch";
    EXPECT_LE(duration, std::chrono::milliseconds(100)) << "Should complete within reasonable time";
    
    // Verify batch efficiency
    const auto& batch_sizes = mock_server_->getBatchSizes();
    for (size_t batch_size : batch_sizes) {
        EXPECT_GE(batch_size, 4) << "Batch size should meet minimum threshold";
        EXPECT_LE(batch_size, 128) << "Batch size should not exceed maximum";
    }
}

TEST_F(BurstCoordinatorTest, LargeBurstOptimization) {
    // Test with large burst that should be efficiently batched
    std::vector<BurstCoordinator::BurstRequest> requests;
    const int REQUEST_COUNT = 96; // 3x target batch size
    
    for (int i = 0; i < REQUEST_COUNT; ++i) {
        BurstCoordinator::BurstRequest request;
        request.leaf = MCTSNode::create(test_state_->clone(), nullptr);
        request.state = test_state_->clone();
        requests.push_back(std::move(request));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto results = coordinator_->collectAndEvaluate(requests, 96);
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify efficient batching
    EXPECT_EQ(results.size(), REQUEST_COUNT);
    EXPECT_EQ(mock_server_->getTotalRequests(), REQUEST_COUNT);
    
    // Should use optimal batch sizes close to target (32)
    const auto& batch_sizes = mock_server_->getBatchSizes();
    double avg_batch_size = 0.0;
    for (size_t batch_size : batch_sizes) {
        avg_batch_size += batch_size;
    }
    avg_batch_size /= batch_sizes.size();
    
    EXPECT_GE(avg_batch_size, 24.0) << "Average batch size should be reasonably large";
    EXPECT_LE(avg_batch_size, 48.0) << "Average batch size should not be too large";
    
    // Efficiency check - should complete quickly with good batching
    EXPECT_LE(duration, std::chrono::milliseconds(200)) << "Large burst should be processed efficiently";
}

TEST_F(BurstCoordinatorTest, ConcurrentBurstHandling) {
    // Test concurrent burst collection from multiple threads
    const int THREAD_COUNT = 4;
    const int REQUESTS_PER_THREAD = 24;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<NetworkOutput>> thread_results(THREAD_COUNT);
    std::atomic<int> completed_threads{0};
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Launch concurrent burst collections
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<BurstCoordinator::BurstRequest> requests;
            
            for (int i = 0; i < REQUESTS_PER_THREAD; ++i) {
                BurstCoordinator::BurstRequest request;
                request.leaf = MCTSNode::create(test_state_->clone(), nullptr);
                request.state = test_state_->clone();
                requests.push_back(std::move(request));
            }
            
            thread_results[t] = coordinator_->collectAndEvaluate(requests, REQUESTS_PER_THREAD);
            completed_threads++;
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify all threads completed successfully
    EXPECT_EQ(completed_threads.load(), THREAD_COUNT);
    
    for (int t = 0; t < THREAD_COUNT; ++t) {
        EXPECT_EQ(thread_results[t].size(), REQUESTS_PER_THREAD) 
            << "Thread " << t << " should have correct result count";
    }
    
    // Verify total processing
    size_t total_expected = THREAD_COUNT * REQUESTS_PER_THREAD;
    EXPECT_EQ(mock_server_->getTotalRequests(), total_expected) 
        << "Server should process all requests from all threads";
    
    // Concurrent execution should be faster than sequential
    EXPECT_LE(duration, std::chrono::milliseconds(500)) 
        << "Concurrent burst handling should be efficient";
}

TEST_F(BurstCoordinatorTest, EmptyCollectionHandling) {
    // Test behavior with empty request collection
    std::vector<BurstCoordinator::BurstRequest> empty_requests;
    
    auto results = coordinator_->collectAndEvaluate(empty_requests, 32);
    
    EXPECT_TRUE(results.empty()) << "Empty collection should return empty results";
    EXPECT_EQ(mock_server_->getTotalRequests(), 0) << "No requests should be sent to server";
    EXPECT_EQ(mock_server_->getBatchCount(), 0) << "No batches should be processed";
}

TEST_F(BurstCoordinatorTest, BatchSizeEfficiency) {
    // Test with various request counts to verify batch size optimization
    std::vector<int> test_sizes = {1, 4, 8, 16, 32, 48, 64, 96, 128};
    
    for (int request_count : test_sizes) {
        // Reset mock server counters
        mock_server_ = std::make_shared<MockUnifiedInferenceServer>();
        
        // Recreate coordinator with new server
        BurstCoordinator::BurstConfig config;
        config.target_burst_size = 32;
        config.min_burst_size = 4;
        config.collection_timeout = std::chrono::milliseconds(10);
        config.evaluation_timeout = std::chrono::milliseconds(50);
        config.max_parallel_threads = 8;
        
        coordinator_ = std::make_unique<BurstCoordinator>(mock_server_, config);
        
        // Create requests
        std::vector<BurstCoordinator::BurstRequest> requests;
        for (int i = 0; i < request_count; ++i) {
            BurstCoordinator::BurstRequest request;
            request.leaf = MCTSNode::create(test_state_->clone(), nullptr);
            request.state = test_state_->clone();
            requests.push_back(std::move(request));
        }
        
        auto results = coordinator_->collectAndEvaluate(requests, request_count);
        
        EXPECT_EQ(results.size(), request_count) 
            << "Request count " << request_count << " should return correct results";
            
        // Verify batch efficiency based on request count
        const auto& batch_sizes = mock_server_->getBatchSizes();
        if (request_count >= 4) {
            // Should have at least one reasonable batch
            EXPECT_GE(batch_sizes.size(), 1) << "Should have processed at least one batch";
            
            size_t total_processed = 0;
            for (size_t batch_size : batch_sizes) {
                total_processed += batch_size;
            }
            EXPECT_EQ(total_processed, request_count) 
                << "All requests should be processed exactly once";
        }
    }
}

TEST_F(BurstCoordinatorTest, TimeoutHandling) {
    // Test timeout behavior with slow processing
    BurstCoordinator::BurstConfig config;
    config.target_burst_size = 32;
    config.min_burst_size = 4;
    config.collection_timeout = std::chrono::milliseconds(5);  // Very short timeout
    config.evaluation_timeout = std::chrono::milliseconds(20); // Short evaluation timeout
    config.max_parallel_threads = 8;
    
    coordinator_ = std::make_unique<BurstCoordinator>(mock_server_, config);
    
    std::vector<BurstCoordinator::BurstRequest> requests;
    for (int i = 0; i < 8; ++i) {
        BurstCoordinator::BurstRequest request;
        request.leaf = MCTSNode::create(test_state_->clone(), nullptr);
        request.state = test_state_->clone();
        requests.push_back(std::move(request));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto results = coordinator_->collectAndEvaluate(requests, 8);
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Should still return results even with timeout
    EXPECT_EQ(results.size(), 8) << "Should return results despite timeout";
    
    // Should respect timeout bounds (with some tolerance for processing)
    EXPECT_LE(duration, std::chrono::milliseconds(100)) 
        << "Should respect timeout constraints";
}

} // namespace test
} // namespace mcts
} // namespace alphazero