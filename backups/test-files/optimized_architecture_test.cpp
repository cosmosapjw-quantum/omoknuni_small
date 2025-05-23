#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>
#include <iterator>

#include "mcts/mcts_engine.h"
#include "mcts/unified_inference_server.h"
#include "mcts/burst_coordinator.h"
#include "mcts/lock_free_batch_accumulator.h"
#include "mcts/optimized_external_queue_processor.h"
#include "mcts/advanced_memory_pool.h"
#include "mcts/cuda_stream_optimizer.h"
#include "games/gomoku/gomoku_state.h"
#include "nn/neural_network_factory.h"
#include "core/game_export.h"

namespace alphazero {
namespace integration {
namespace test {

class MockOptimizedNeuralNetwork : public nn::NeuralNetwork {
public:
    MockOptimizedNeuralNetwork() : inference_count_(0), total_batch_size_(0) {}
    
    std::vector<int64_t> getInputShape() const override {
        return {15, 15}; // 15x15 Gomoku board
    }
    
    int64_t getPolicySize() const override {
        return 225; // 15x15 board positions
    }
    
    std::vector<mcts::NetworkOutput> inference(const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        inference_count_++;
        total_batch_size_ += states.size();
        batch_sizes_.push_back(states.size());
        
        // Track timing for performance validation
        auto start_time = std::chrono::steady_clock::now();
        
        // Simulate realistic GPU processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(states.size() / 16 + 3));
        
        auto end_time = std::chrono::steady_clock::now();
        inference_times_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
        
        std::vector<mcts::NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); ++i) {
            mcts::NetworkOutput output;
            output.value = (i % 4 == 0) ? 0.7f : ((i % 4 == 1) ? -0.2f : ((i % 4 == 2) ? 0.4f : -0.1f));
            output.policy.resize(225, 1.0f / 225.0f);
            
            // Add realistic policy variation
            for (size_t j = 0; j < std::min(output.policy.size(), static_cast<size_t>(20)); ++j) {
                output.policy[j] = 0.01f + (j % 7) * 0.005f;
            }
            
            outputs.push_back(std::move(output));
        }
        
        return outputs;
    }
    
    void save(const std::string& /*path*/) override {}
    void load(const std::string& /*path*/) override {}
    
    // Performance metrics
    size_t getInferenceCount() const { return inference_count_.load(); }
    size_t getTotalBatchSize() const { return total_batch_size_.load(); }
    size_t getTotalInferences() const { return total_batch_size_.load(); } // Total individual inferences processed
    const std::vector<size_t>& getBatchSizes() const { return batch_sizes_; }
    const std::vector<long>& getInferenceTimes() const { return inference_times_; }
    
    double getAverageBatchSize() const {
        return inference_count_ > 0 ? static_cast<double>(total_batch_size_) / inference_count_ : 0.0;
    }
    
    void resetMetrics() {
        inference_count_ = 0;
        total_batch_size_ = 0;
        batch_sizes_.clear();
        inference_times_.clear();
    }
    
private:
    std::atomic<size_t> inference_count_;
    std::atomic<size_t> total_batch_size_;
    std::vector<size_t> batch_sizes_;
    std::vector<long> inference_times_;
};

class OptimizedArchitectureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        mock_neural_net_ = std::make_shared<MockOptimizedNeuralNetwork>();
        test_state_ = std::make_unique<games::gomoku::GomokuState>();
    }
    
    void TearDown() override {
        mock_neural_net_.reset();
    }
    
    std::shared_ptr<MockOptimizedNeuralNetwork> mock_neural_net_;
    std::unique_ptr<games::gomoku::GomokuState> test_state_;
};

TEST_F(OptimizedArchitectureTest, BurstCoordinatorIntegration) {
    // Test that BurstCoordinator + UnifiedInferenceServer achieves target batch efficiency
    
    mcts::UnifiedInferenceServer::ServerConfig inference_config;
    inference_config.target_batch_size = 32;
    inference_config.min_batch_size = 8;
    inference_config.max_batch_size = 96;
    inference_config.max_batch_wait = std::chrono::milliseconds(15);
    inference_config.min_batch_wait = std::chrono::milliseconds(2);
    inference_config.num_worker_threads = 4;
    
    auto inference_server = std::make_shared<mcts::UnifiedInferenceServer>(
        mock_neural_net_, inference_config);
    inference_server->start();
    
    mcts::BurstCoordinator::BurstConfig burst_config;
    burst_config.target_burst_size = 32;
    burst_config.min_burst_size = 8;
    // max_size doesn't exist in BurstConfig - removed
    burst_config.collection_timeout = std::chrono::milliseconds(10);
    burst_config.evaluation_timeout = std::chrono::milliseconds(40);
    burst_config.max_parallel_threads = 8;
    
    auto burst_coordinator = std::make_unique<mcts::BurstCoordinator>(
        inference_server, burst_config);
    
    // Test large burst collection
    const int TOTAL_REQUESTS = 128;
    std::vector<mcts::BurstCoordinator::BurstRequest> requests;
    
    for (int i = 0; i < TOTAL_REQUESTS; ++i) {
        mcts::BurstCoordinator::BurstRequest request;
        request.leaf = mcts::MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        request.state = test_state_->clone();
        requests.push_back(std::move(request));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto results = burst_coordinator->collectAndEvaluate(requests, TOTAL_REQUESTS);
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    inference_server->stop();
    
    // Verify integration results
    EXPECT_EQ(results.size(), TOTAL_REQUESTS) << "Should process all requests";
    EXPECT_EQ(mock_neural_net_->getTotalBatchSize(), TOTAL_REQUESTS) 
        << "Neural network should see all requests";
    
    // Verify batch efficiency - should achieve good average batch sizes
    double avg_batch_size = mock_neural_net_->getAverageBatchSize();
    EXPECT_GE(avg_batch_size, 20.0) << "Should achieve good average batch size";
    EXPECT_LE(avg_batch_size, 64.0) << "Should not exceed reasonable limits";
    
    // Performance check - should be efficient
    EXPECT_LE(duration, std::chrono::milliseconds(500)) 
        << "Integration should be performant";
    
    // Verify batch size distribution
    const auto& batch_sizes = mock_neural_net_->getBatchSizes();
    for (size_t batch_size : batch_sizes) {
        EXPECT_GE(batch_size, 8) << "All batches should meet minimum size";
        EXPECT_LE(batch_size, 96) << "All batches should respect maximum size";
    }
}

TEST_F(OptimizedArchitectureTest, LockFreeBatchAccumulatorTest) {
    // Test the lock-free batch accumulator component
    
    mcts::LockFreeBatchConfig config;
    config.target_batch_size = 32;
    config.max_wait_time = std::chrono::milliseconds(15);
    
    auto batch_accumulator = std::make_unique<mcts::LockFreeBatchAccumulator>(config);
    
    // Submit sample requests
    const int TOTAL_REQUESTS = 100;
    
    for (int i = 0; i < TOTAL_REQUESTS; ++i) {
        mcts::PendingEvaluation eval;
        eval.node = mcts::MCTSNode::create(std::unique_ptr<core::IGameState>(test_state_->clone()), nullptr);
        eval.state = test_state_->clone();
        eval.batch_id = i;
        eval.request_id = i * 100;
        
        batch_accumulator->submitRequest(std::move(eval));
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Collect batches
    std::vector<mcts::PendingEvaluation> collected;
    
    while (collected.size() < TOTAL_REQUESTS) {
        auto batch = batch_accumulator->collectBatch();
        if (batch.empty()) {
            // Give a little more time for processing
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else {
            collected.insert(collected.end(), std::make_move_iterator(batch.begin()), std::make_move_iterator(batch.end()));
        }
        
        // Prevent infinite loop in case of failures
        if (collected.size() == TOTAL_REQUESTS || 
            batch_accumulator->pendingCount() == 0 && batch_accumulator->readyCount() == 0) {
            break;
        }
    }
    
    // Verify batch accumulator behavior
    EXPECT_GE(collected.size(), 0.9 * TOTAL_REQUESTS) 
        << "Should process most of the requests";
    
    EXPECT_LE(batch_accumulator->pendingCount() + batch_accumulator->readyCount(), 
             TOTAL_REQUESTS - collected.size()) 
        << "All requests should be accounted for";
}

TEST_F(OptimizedArchitectureTest, MCTSEngineOptimizedPipeline) {
    // Test full MCTS engine with optimized architecture
    
    mcts::MCTSSettings settings;
    settings.num_simulations = 400; // Moderate simulation count for testing
    settings.num_threads = 8;
    settings.batch_size = 32;
    settings.batch_timeout = std::chrono::milliseconds(20);
    settings.exploration_constant = 1.4f;
    settings.virtual_loss = 1.0f; // Optimized virtual loss
    settings.use_progressive_widening = true;
    settings.progressive_widening_c = 2.0f; // Optimized parameters
    settings.progressive_widening_k = 6.0f;
    
    mcts::MCTSEngine engine(mock_neural_net_, settings);
    
    auto start_time = std::chrono::steady_clock::now();
    engine.search(*test_state_);
    auto duration = std::chrono::steady_clock::now() - start_time;
    
    // Verify search completed successfully (check that it ran without errors)
    EXPECT_GT(duration.count(), 0) << "Search should take some time to complete";
    
    // Verify that neural network was used during search
    EXPECT_GT(mock_neural_net_->getTotalInferences(), 10) << "Should have performed neural network inferences";
    
    // Verify batch efficiency in full MCTS context
    double avg_batch_size = mock_neural_net_->getAverageBatchSize();
    EXPECT_GE(avg_batch_size, 8.0) << "MCTS search should achieve reasonable batch sizes";
    
    // Verify that the search ran successfully by checking inference metrics
    EXPECT_GT(mock_neural_net_->getInferenceCount(), 1) << "Search should have triggered neural network calls";
    EXPECT_GT(mock_neural_net_->getTotalInferences(), 10) << "Search should have processed multiple positions";
    
    // Performance validation
    EXPECT_LE(duration, std::chrono::milliseconds(2000)) 
        << "Optimized MCTS should complete within reasonable time";
    
    // Verify improved utilization - should use most of the target simulations
    float utilization = static_cast<float>(mock_neural_net_->getTotalBatchSize()) / settings.num_simulations;
    EXPECT_GE(utilization, 0.6f) 
        << "Should achieve good simulation utilization with optimized architecture";
}

TEST_F(OptimizedArchitectureTest, PerformanceRegression) {
    // Ensure optimized architecture performs better than theoretical baseline
    
    const int BENCHMARK_SIMULATIONS = 200;
    const int BENCHMARK_THREADS = 4;
    
    mcts::MCTSSettings optimized_settings;
    optimized_settings.num_simulations = BENCHMARK_SIMULATIONS;
    optimized_settings.num_threads = BENCHMARK_THREADS;
    optimized_settings.batch_size = 32;
    optimized_settings.batch_timeout = std::chrono::milliseconds(15);
    optimized_settings.virtual_loss = 1.0f; // Optimized
    optimized_settings.progressive_widening_c = 2.0f; // Optimized
    optimized_settings.progressive_widening_k = 6.0f; // Optimized
    
    // Run optimized search
    mcts::MCTSEngine optimized_engine(mock_neural_net_, optimized_settings);
    
    auto opt_start = std::chrono::steady_clock::now();
    optimized_engine.search(*test_state_);
    auto opt_duration = std::chrono::steady_clock::now() - opt_start;
    
    // Collect optimized metrics
    double opt_avg_batch = mock_neural_net_->getAverageBatchSize();
    size_t opt_total_batches = mock_neural_net_->getInferenceCount();
    size_t opt_utilization = mock_neural_net_->getTotalBatchSize();
    
    // Reset for unoptimized test
    mock_neural_net_->resetMetrics();
    
    // Run with less optimized settings (simulating old behavior)
    mcts::MCTSSettings unoptimized_settings = optimized_settings;
    unoptimized_settings.virtual_loss = 3.0f; // Old aggressive virtual loss
    unoptimized_settings.progressive_widening_c = 1.0f; // Old restrictive
    unoptimized_settings.progressive_widening_k = 10.0f; // Old restrictive
    unoptimized_settings.batch_size = 16; // Smaller batches
    
    mcts::MCTSEngine unoptimized_engine(mock_neural_net_, unoptimized_settings);
    
    auto unopt_start = std::chrono::steady_clock::now();
    unoptimized_engine.search(*test_state_);
    auto unopt_duration = std::chrono::steady_clock::now() - unopt_start;
    
    // Collect unoptimized metrics
    double unopt_avg_batch = mock_neural_net_->getAverageBatchSize();
    size_t unopt_total_batches = mock_neural_net_->getInferenceCount();
    size_t unopt_utilization = mock_neural_net_->getTotalBatchSize();
    
    // Performance comparisons
    EXPECT_GE(opt_avg_batch, unopt_avg_batch * 0.8) 
        << "Optimized should maintain good batch sizes";
    
    EXPECT_LE(opt_total_batches, unopt_total_batches * 1.2) 
        << "Optimized should not require significantly more batches";
    
    EXPECT_GE(opt_utilization, unopt_utilization * 0.9) 
        << "Optimized should achieve equal or better simulation utilization";
    
    // Timing comparison - optimized should not be significantly slower
    auto opt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(opt_duration).count();
    auto unopt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(unopt_duration).count();
    
    EXPECT_LE(opt_ms, unopt_ms * 1.3) 
        << "Optimized architecture should maintain reasonable performance";
}

TEST_F(OptimizedArchitectureTest, ConcurrentSearchStability) {
    // Test that optimized architecture is stable under concurrent load
    
    const int CONCURRENT_SEARCHES = 4;
    const int SIMULATIONS_PER_SEARCH = 150;
    
    mcts::MCTSSettings settings;
    settings.num_simulations = SIMULATIONS_PER_SEARCH;
    settings.num_threads = 6;
    settings.batch_size = 24;
    settings.batch_timeout = std::chrono::milliseconds(20);
    settings.virtual_loss = 1.0f;
    settings.progressive_widening_c = 2.0f;
    settings.progressive_widening_k = 6.0f;
    
    std::vector<std::thread> search_threads;
    std::vector<std::atomic<bool>> search_success(CONCURRENT_SEARCHES);
    std::vector<std::chrono::milliseconds> search_times(CONCURRENT_SEARCHES);
    
    for (int i = 0; i < CONCURRENT_SEARCHES; ++i) {
        search_success[i] = false;
    }
    
    auto overall_start = std::chrono::steady_clock::now();
    
    // Launch concurrent searches
    for (int i = 0; i < CONCURRENT_SEARCHES; ++i) {
        search_threads.emplace_back([&, i]() {
            try {
                auto local_state = std::make_unique<games::gomoku::GomokuState>();
                mcts::MCTSEngine engine(mock_neural_net_, settings);
                
                auto start = std::chrono::steady_clock::now();
                engine.search(*local_state);
                auto end = std::chrono::steady_clock::now();
                
                search_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                search_success[i] = true;
                
            } catch (const std::exception& e) {
                search_success[i] = false;
            }
        });
    }
    
    // Wait for all searches to complete
    for (auto& thread : search_threads) {
        thread.join();
    }
    
    auto overall_duration = std::chrono::steady_clock::now() - overall_start;
    
    // Verify all searches completed successfully
    for (int i = 0; i < CONCURRENT_SEARCHES; ++i) {
        EXPECT_TRUE(search_success[i].load()) 
            << "Concurrent search " << i << " should complete successfully";
    }
    
    // Verify reasonable performance under concurrent load
    EXPECT_LE(overall_duration, std::chrono::milliseconds(5000)) 
        << "Concurrent searches should complete within reasonable time";
    
    // Check that we achieved good batch efficiency under concurrency
    double avg_batch_size = mock_neural_net_->getAverageBatchSize();
    EXPECT_GE(avg_batch_size, 12.0) 
        << "Should maintain decent batch sizes under concurrent load";
    
    // Verify total utilization
    size_t expected_total_utilization = CONCURRENT_SEARCHES * SIMULATIONS_PER_SEARCH;
    size_t actual_utilization = mock_neural_net_->getTotalBatchSize();
    
    // Should utilize most of the target simulations (allowing for some search variation)
    float utilization_ratio = static_cast<float>(actual_utilization) / expected_total_utilization;
    EXPECT_GE(utilization_ratio, 0.5f) 
        << "Should achieve reasonable total simulation utilization";
}

} // namespace test
} // namespace integration
} // namespace alphazero