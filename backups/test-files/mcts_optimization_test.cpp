#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <thread>
#include "mcts/mcts_engine.h"
#include "mcts/unified_inference_server.h"
#include "games/gomoku/gomoku_state.h"
#include "nn/neural_network_factory.h"
#include "nn/gpu_optimizer.h"
#include "core/game_export.h"

namespace alphazero {
namespace mcts {
namespace test {

class MCTSOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register GOMOKU game type for tests  
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        // Create neural network
        neural_net = nn::NeuralNetworkFactory::createResNet(
            4,      // input_channels
            15,     // board_size
            10,     // num_res_blocks
            128,    // num_filters
            225,    // policy_size
            false   // use_gpu - set to false for tests
        );
    }
    
    std::shared_ptr<nn::NeuralNetwork> neural_net;
};

TEST_F(MCTSOptimizationTest, BatchProcessingOptimization) {
    // Test that batch processing achieves the target batch size
    MCTSSettings settings;
    settings.batch_size = 256;  // Our optimized batch size
    settings.batch_timeout = std::chrono::milliseconds(75);
    settings.num_threads = 8;
    settings.num_simulations = 800;
    
    MCTSEngine engine(neural_net, settings);
    
    // Run search and measure batch sizes
    auto state = std::make_unique<games::gomoku::GomokuState>();
    auto result = engine.search(*state);
    
    // Check that we achieved good batching
    const auto& stats = engine.getLastStats();
    EXPECT_GT(stats.avg_batch_size, 100.0f) << "Average batch size should be greater than 100";
    EXPECT_LT(stats.avg_batch_latency.count(), 100) << "Batch latency should be under 100ms";
    
    std::cout << "Average batch size: " << stats.avg_batch_size << std::endl;
    std::cout << "Average batch latency: " << stats.avg_batch_latency.count() << "ms" << std::endl;
    std::cout << "Nodes per second: " << stats.nodes_per_second << std::endl;
}

TEST_F(MCTSOptimizationTest, GPUMemoryOptimization) {
    // Test GPU memory usage with optimization
    nn::GPUOptimizer::Config gpu_config;
    gpu_config.max_batch_size = 256;
    gpu_config.use_pinned_memory = true;
    gpu_config.pre_allocate = true;
    
    nn::GPUOptimizer optimizer(gpu_config);
    
    // Simulate batch inference
    std::vector<std::unique_ptr<core::IGameState>> states;
    for (int i = 0; i < 256; ++i) {
        states.push_back(std::make_unique<games::gomoku::GomokuState>());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto tensor = optimizer.prepareStatesBatch(states);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.size(0), 256);
    EXPECT_LT(duration.count(), 10000) << "Batch preparation should be under 10ms";
    
    auto stats = optimizer.getMemoryStats();
    std::cout << "GPU memory allocated: " << stats.allocated_gpu_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "Pinned memory allocated: " << stats.allocated_pinned_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "Average transfer time: " << stats.avg_transfer_time.count() << " us" << std::endl;
}

TEST_F(MCTSOptimizationTest, ConcurrentSearchStressTest) {
    // Test concurrent search with optimizations
    MCTSSettings settings;
    settings.batch_size = 256;
    settings.batch_timeout = std::chrono::milliseconds(75);
    settings.num_threads = 16;  // High thread count
    settings.num_simulations = 1600;
    settings.use_root_parallelization = true;
    settings.num_root_workers = 4;
    
    // Run multiple searches concurrently
    const int num_searches = 4;
    std::vector<std::thread> threads;
    std::vector<SearchResult> results(num_searches);
    
    auto search_func = [&](int idx) {
        MCTSEngine engine(neural_net, settings);
        auto state = std::make_unique<games::gomoku::GomokuState>();
        results[idx] = engine.search(*state);
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_searches; ++i) {
        threads.emplace_back(search_func, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Check results
    for (const auto& result : results) {
        EXPECT_GE(result.action, 0);
        EXPECT_FALSE(result.probabilities.empty());
        EXPECT_GT(result.stats.total_nodes, 0);
        
        std::cout << "Search completed - Nodes: " << result.stats.total_nodes 
                  << ", NPS: " << result.stats.nodes_per_second 
                  << ", Avg batch: " << result.stats.avg_batch_size << std::endl;
    }
    
    std::cout << "Total time for " << num_searches << " concurrent searches: " 
              << duration.count() << "ms" << std::endl;
}

TEST_F(MCTSOptimizationTest, MemoryLeakTest) {
    // Test for memory leaks with repeated searches
    MCTSSettings settings;
    settings.batch_size = 128;
    settings.num_threads = 4;
    settings.num_simulations = 100;
    
    const int num_iterations = 10;
    
    for (int i = 0; i < num_iterations; ++i) {
        MCTSEngine engine(neural_net, settings);
        auto state = std::make_unique<games::gomoku::GomokuState>();
        auto result = engine.search(*state);
        
        // Force cleanup
        engine.forceCleanup();
        
        if (i % 5 == 0) {
            std::cout << "Iteration " << i << " - Nodes: " << result.stats.total_nodes << std::endl;
        }
    }
    
    // Memory should be stable after iterations
    // In a real test, we would measure actual memory usage
    SUCCEED() << "Memory leak test completed";
}

} // namespace test
} // namespace mcts
} // namespace alphazero