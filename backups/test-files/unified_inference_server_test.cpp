#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

#include "mcts/unified_inference_server.h"
#include "games/gomoku/gomoku_state.h"
#include "nn/neural_network_factory.h"
#include "core/game_export.h"

namespace alphazero {
namespace mcts {
namespace test {

class MockNeuralNetwork : public nn::NeuralNetwork {
public:
    MockNeuralNetwork() : batch_count_(0), total_inferences_(0) {}
    
    std::vector<int64_t> getInputShape() const override {
        return {15, 15}; // 15x15 Gomoku board
    }
    
    int64_t getPolicySize() const override {
        return 225; // 15x15 board positions
    }
    
    std::vector<NetworkOutput> inference(const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        batch_count_++;
        total_inferences_ += states.size();
        batch_sizes_.push_back(states.size());
        
        // Simulate GPU processing time proportional to batch size
        auto processing_time = std::chrono::milliseconds(states.size() / 8 + 5);
        std::this_thread::sleep_for(processing_time);
        
        std::vector<NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); ++i) {
            NetworkOutput output;
            output.value = (i % 3 == 0) ? 0.8f : ((i % 3 == 1) ? -0.3f : 0.1f);
            output.policy.resize(225, 1.0f / 225.0f); // 15x15 Gomoku board
            
            // Add some variation to the policy
            if (i < 225) {
                output.policy[i] = 0.05f + (i % 5) * 0.01f;
            }
            
            outputs.push_back(std::move(output));
        }
        
        return outputs;
    }
    
    void save(const std::string& /*path*/) override {}
    void load(const std::string& /*path*/) override {}
    
    size_t getBatchCount() const { return batch_count_.load(); }
    size_t getTotalInferences() const { return total_inferences_.load(); }
    const std::vector<size_t>& getBatchSizes() const { return batch_sizes_; }
    
    void resetCounters() {
        batch_count_ = 0;
        total_inferences_ = 0;
        batch_sizes_.clear();
    }
    
private:
    std::atomic<size_t> batch_count_;
    std::atomic<size_t> total_inferences_;
    std::vector<size_t> batch_sizes_;
};

class UnifiedInferenceServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        mock_network_ = std::make_shared<MockNeuralNetwork>();
        
        // Create test game state
        test_state_ = std::make_unique<games::gomoku::GomokuState>();
    }
    
    void createServer(UnifiedInferenceServer::ServerConfig config) {
        server_ = std::make_unique<UnifiedInferenceServer>(mock_network_, config);
        server_->start();
        
        // Give server time to start up
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    void TearDown() override {
        if (server_) {
            server_->stop();
            server_.reset();
        }
        mock_network_.reset();
    }
    
    std::shared_ptr<MockNeuralNetwork> mock_network_;
    std::unique_ptr<UnifiedInferenceServer> server_;
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
};

TEST_F(UnifiedInferenceServerTest, BasicBatchProcessing) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 16;
    config.min_batch_size = 4;
    config.max_batch_size = 64;
    config.max_batch_wait = std::chrono::milliseconds(20);
    config.min_batch_wait = std::chrono::milliseconds(2);
    config.num_worker_threads = 2;
    
    createServer(config);
    
    // Submit individual requests that should be batched
    std::vector<std::future<NetworkOutput>> futures;
    for (int i = 0; i < 12; ++i) {
        auto node = MCTSNode::create(test_state_->clone(), nullptr);
        auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
        
        futures.push_back(server_->submitRequest(node, state, {}));
    }
    
    // Wait for all results
    std::vector<NetworkOutput> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    // Verify results
    EXPECT_EQ(results.size(), 12) << "Should return results for all requests";
    EXPECT_EQ(mock_network_->getTotalInferences(), 12) << "Network should process all requests";
    
    // Should efficiently batch the requests
    EXPECT_LE(mock_network_->getBatchCount(), 3) << "Should use efficient batching";
    EXPECT_GE(mock_network_->getBatchCount(), 1) << "Should process at least one batch";
    
    // Verify all results are valid
    for (const auto& result : results) {
        EXPECT_EQ(result.policy.size(), 225) << "Policy should have correct size";
        EXPECT_GE(result.value, -1.0f) << "Value should be in valid range";
        EXPECT_LE(result.value, 1.0f) << "Value should be in valid range";
    }
}

TEST_F(UnifiedInferenceServerTest, OptimalBatchSizeTargeting) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 32;
    config.min_batch_size = 8;
    config.max_batch_size = 128;
    config.max_batch_wait = std::chrono::milliseconds(15);
    config.min_batch_wait = std::chrono::milliseconds(1);
    config.num_worker_threads = 4;
    
    createServer(config);
    
    // Submit exactly target batch size worth of requests
    std::vector<std::future<NetworkOutput>> futures;
    for (int i = 0; i < 32; ++i) {
        auto node = MCTSNode::create(test_state_->clone(), nullptr);
        auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
        
        futures.push_back(server_->submitRequest(node, state, {}));
    }
    
    // Wait for results
    for (auto& future : futures) {
        future.get();
    }
    
    // Should achieve target batch size efficiently
    const auto& batch_sizes = mock_network_->getBatchSizes();
    EXPECT_GE(batch_sizes.size(), 1) << "Should process at least one batch";
    
    // Check if we hit the target batch size
    bool found_target_batch = false;
    for (size_t batch_size : batch_sizes) {
        if (batch_size == 32) {
            found_target_batch = true;
            break;
        }
    }
    
    if (!found_target_batch) {
        // If not exact target, should be close to target in total
        size_t total_processed = 0;
        for (size_t batch_size : batch_sizes) {
            total_processed += batch_size;
            EXPECT_GE(batch_size, 8) << "All batches should meet minimum size";
            EXPECT_LE(batch_size, 128) << "All batches should be within max size";
        }
        EXPECT_EQ(total_processed, 32) << "Should process exactly all requests";
    }
}

TEST_F(UnifiedInferenceServerTest, ConcurrentRequestHandling) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 24;
    config.min_batch_size = 6;
    config.max_batch_size = 96;
    config.max_batch_wait = std::chrono::milliseconds(25);
    config.min_batch_wait = std::chrono::milliseconds(2);
    config.num_worker_threads = 6;
    
    createServer(config);
    
    const int THREAD_COUNT = 8;
    const int REQUESTS_PER_THREAD = 16;
    
    std::vector<std::thread> threads;
    std::vector<std::atomic<bool>> thread_success(THREAD_COUNT);
    std::vector<std::vector<NetworkOutput>> thread_results(THREAD_COUNT);
    
    for (int t = 0; t < THREAD_COUNT; ++t) {
        thread_success[t] = false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Launch concurrent request threads
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            try {
                std::vector<std::future<NetworkOutput>> futures;
                
                // Submit requests
                for (int i = 0; i < REQUESTS_PER_THREAD; ++i) {
                    auto node = MCTSNode::create(test_state_->clone(), nullptr);
                    auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
                    
                    futures.push_back(server_->submitRequest(node, state, {}));
                }
                
                // Collect results
                thread_results[t].clear();
                thread_results[t].reserve(REQUESTS_PER_THREAD);
                
                for (auto& future : futures) {
                    thread_results[t].push_back(future.get());
                }
                
                thread_success[t] = true;
            } catch (const std::exception& e) {
                // Thread failed
                thread_success[t] = false;
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify all threads succeeded
    for (int t = 0; t < THREAD_COUNT; ++t) {
        EXPECT_TRUE(thread_success[t].load()) << "Thread " << t << " should succeed";
        EXPECT_EQ(thread_results[t].size(), REQUESTS_PER_THREAD) 
            << "Thread " << t << " should get all results";
    }
    
    // Verify total processing
    size_t expected_total = THREAD_COUNT * REQUESTS_PER_THREAD;
    EXPECT_EQ(mock_network_->getTotalInferences(), expected_total) 
        << "Should process all requests from all threads";
    
    // Concurrent processing should be reasonably fast
    EXPECT_LE(duration, std::chrono::milliseconds(1000)) 
        << "Concurrent processing should be efficient";
    
    // Verify batch efficiency under concurrency
    const auto& batch_sizes = mock_network_->getBatchSizes();
    double avg_batch_size = 0.0;
    for (size_t batch_size : batch_sizes) {
        avg_batch_size += batch_size;
    }
    avg_batch_size /= batch_sizes.size();
    
    EXPECT_GE(avg_batch_size, 12.0) << "Should maintain good batch sizes under concurrency";
}

TEST_F(UnifiedInferenceServerTest, TimeoutBehavior) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 64; // Large target
    config.min_batch_size = 4;
    config.max_batch_size = 128;
    config.max_batch_wait = std::chrono::milliseconds(30); // Moderate timeout
    config.min_batch_wait = std::chrono::milliseconds(5);
    config.num_worker_threads = 2;
    
    createServer(config);
    
    // Submit fewer requests than target batch size
    std::vector<std::future<NetworkOutput>> futures;
    for (int i = 0; i < 8; ++i) { // Much less than target of 64
        auto node = MCTSNode::create(test_state_->clone(), nullptr);
        auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
        
        futures.push_back(server_->submitRequest(node, state, {}));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Wait for results
    std::vector<NetworkOutput> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Should get all results despite not reaching target batch size
    EXPECT_EQ(results.size(), 8) << "Should return all results";
    EXPECT_EQ(mock_network_->getTotalInferences(), 8) << "Should process all requests";
    
    // Should respect timeout and not wait indefinitely
    EXPECT_LE(duration, std::chrono::milliseconds(100)) 
        << "Should timeout and process smaller batch";
    
    // Should have processed as a smaller batch due to timeout
    const auto& batch_sizes = mock_network_->getBatchSizes();
    EXPECT_GE(batch_sizes.size(), 1) << "Should have processed at least one batch";
    
    bool found_small_batch = false;
    for (size_t batch_size : batch_sizes) {
        if (batch_size <= 8) {
            found_small_batch = true;
            break;
        }
    }
    EXPECT_TRUE(found_small_batch) << "Should have processed a small batch due to timeout";
}

TEST_F(UnifiedInferenceServerTest, LargeBatchOptimization) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 48;
    config.min_batch_size = 8;
    config.max_batch_size = 128;
    config.max_batch_wait = std::chrono::milliseconds(20);
    config.min_batch_wait = std::chrono::milliseconds(1);
    config.num_worker_threads = 4;
    
    createServer(config);
    
    // Submit large number of requests
    const int LARGE_REQUEST_COUNT = 144; // 3x target batch size
    std::vector<std::future<NetworkOutput>> futures;
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (int i = 0; i < LARGE_REQUEST_COUNT; ++i) {
        auto node = MCTSNode::create(test_state_->clone(), nullptr);
        auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
        
        futures.push_back(server_->submitRequest(node, state, {}));
    }
    
    // Wait for all results
    std::vector<NetworkOutput> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify all requests processed
    EXPECT_EQ(results.size(), LARGE_REQUEST_COUNT);
    EXPECT_EQ(mock_network_->getTotalInferences(), LARGE_REQUEST_COUNT);
    
    // Should achieve good batch efficiency
    const auto& batch_sizes = mock_network_->getBatchSizes();
    
    // Calculate efficiency metrics
    double avg_batch_size = 0.0;
    size_t max_batch_size = 0;
    for (size_t batch_size : batch_sizes) {
        avg_batch_size += batch_size;
        max_batch_size = std::max(max_batch_size, batch_size);
    }
    avg_batch_size /= batch_sizes.size();
    
    EXPECT_GE(avg_batch_size, 32.0) << "Should achieve high average batch size";
    EXPECT_LE(max_batch_size, 128) << "Should respect maximum batch size";
    
    // Large batch processing should be efficient
    EXPECT_LE(duration, std::chrono::milliseconds(800)) 
        << "Large batch processing should be fast";
}

TEST_F(UnifiedInferenceServerTest, ServerStartStopLifecycle) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 16;
    config.min_batch_size = 4;
    config.max_batch_size = 64;
    config.max_batch_wait = std::chrono::milliseconds(10);
    config.min_batch_wait = std::chrono::milliseconds(1);
    config.num_worker_threads = 2;
    
    // Test start/stop cycle
    createServer(config);
    
    // Submit a request to verify server is running
    auto node = MCTSNode::create(test_state_->clone(), nullptr);
    auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
    
    auto future = server_->submitRequest(node, state, {});
    auto result = future.get();
    
    EXPECT_EQ(result.policy.size(), 225) << "Server should be functional after start";
    
    // Stop server
    server_->stop();
    
    // Verify clean shutdown
    EXPECT_GE(mock_network_->getTotalInferences(), 1) << "Should have processed at least one request";
    
    // Starting a new server should work
    mock_network_->resetCounters();
    createServer(config);
    
    // Submit another request
    auto node2 = MCTSNode::create(test_state_->clone(), nullptr);
    auto state2 = std::shared_ptr<core::IGameState>(test_state_->clone());
    
    auto future2 = server_->submitRequest(node2, state2, {});
    auto result2 = future2.get();
    
    EXPECT_EQ(result2.policy.size(), 225) << "Restarted server should be functional";
    EXPECT_EQ(mock_network_->getTotalInferences(), 1) << "New server should start fresh";
}

TEST_F(UnifiedInferenceServerTest, EmptyRequestHandling) {
    UnifiedInferenceServer::ServerConfig config;
    config.target_batch_size = 16;
    config.min_batch_size = 1; // Allow single requests
    config.max_batch_size = 64;
    config.max_batch_wait = std::chrono::milliseconds(10);
    config.min_batch_wait = std::chrono::milliseconds(1);
    config.num_worker_threads = 2;
    
    createServer(config);
    
    // Test with no requests for a period
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    EXPECT_EQ(mock_network_->getTotalInferences(), 0) 
        << "No requests should be processed when none submitted";
    
    // Then submit a single request
    auto node = MCTSNode::create(test_state_->clone(), nullptr);
    auto state = std::shared_ptr<core::IGameState>(test_state_->clone());
    
    auto future = server_->submitRequest(node, state, {});
    auto result = future.get();
    
    EXPECT_EQ(result.policy.size(), 225) << "Single request should be handled correctly";
    EXPECT_EQ(mock_network_->getTotalInferences(), 1) << "Should process exactly one request";
}

} // namespace test
} // namespace mcts
} // namespace alphazero