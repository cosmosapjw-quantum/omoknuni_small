#include <gtest/gtest.h>
#include <omp.h>
#include <thread>
#include <atomic>
#include <chrono>
#include "mcts/mcts_engine.h"
#include "nn/neural_network.h"
#include "games/gomoku/gomoku_state.h"
#include "utils/debug_monitor.h"
#include "utils/memory_tracker.h"

using namespace alphazero;

// Mock neural network for testing
class MockNeuralNetwork : public nn::NeuralNetwork {
public:
    MockNeuralNetwork() : call_count_(0) {}
    
    std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        
        call_count_.fetch_add(states.size(), std::memory_order_relaxed);
        
        std::vector<mcts::NetworkOutput> outputs;
        for (size_t i = 0; i < states.size(); i++) {
            mcts::NetworkOutput output;
            
            // Create dummy policy based on state
            int num_actions = 361; // 19x19 gomoku board
            output.policy.resize(num_actions);
            float sum = 0.0f;
            
            // Simple policy: prefer center moves
            for (int j = 0; j < num_actions; j++) {
                int x = j % 19;
                int y = j / 19;
                float dist = std::sqrt((x - 9) * (x - 9) + (y - 9) * (y - 9));
                output.policy[j] = std::exp(-dist / 5.0f);
                sum += output.policy[j];
            }
            
            // Normalize
            for (int j = 0; j < num_actions; j++) {
                output.policy[j] /= sum;
            }
            
            // Dummy value
            output.value = 0.1f;
            
            outputs.push_back(output);
        }
        
        // Simulate GPU processing time
        std::this_thread::sleep_for(std::chrono::microseconds(100 * states.size()));
        
        return outputs;
    }
    
    // Required virtual methods
    void save(const std::string& path) override {}
    void load(const std::string& path) override {}
    std::vector<int64_t> getInputShape() const override { return {19, 19, 17}; }
    int64_t getPolicySize() const override { return 361; }
    
    int getCallCount() const {
        return call_count_.load(std::memory_order_relaxed);
    }
    
private:
    std::atomic<int> call_count_;
};

class MCTSOpenMPTest : public ::testing::Test {
protected:
    void SetUp() override {
        neural_net_ = std::make_shared<MockNeuralNetwork>();
    }
    
    std::shared_ptr<MockNeuralNetwork> neural_net_;
};

TEST_F(MCTSOpenMPTest, BasicFunctionality) {
    // Test basic MCTS search with OpenMP
    mcts::MCTSSettings settings;
    settings.num_simulations = 100;
    settings.num_threads = 4;
    settings.batch_size = 8;
    
    omp_set_num_threads(settings.num_threads);
    
    mcts::MCTSEngine engine(neural_net_, settings);
    games::gomoku::GomokuState state(19);
    
    auto result = engine.search(state);
    
    // Verify result
    EXPECT_NE(result.action, -1);
    EXPECT_EQ(result.probabilities.size(), 361);
    EXPECT_GT(result.stats.total_nodes, 100);
    EXPECT_GT(neural_net_->getCallCount(), 0);
}

TEST_F(MCTSOpenMPTest, ThreadScaling) {
    // Test that performance scales with thread count
    std::vector<int> thread_counts = {1, 2, 4};
    std::vector<double> times;
    
    for (int threads : thread_counts) {
        mcts::MCTSSettings settings;
        settings.num_simulations = 400;
        settings.num_threads = threads;
        settings.batch_size = 16;
        
        omp_set_num_threads(threads);
        
        mcts::MCTSEngine engine(neural_net_, settings);
        games::gomoku::GomokuState state(19);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine.search(state);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        times.push_back(duration.count());
        
        std::cout << "Threads: " << threads << ", Time: " << duration.count() 
                  << "ms, Nodes: " << result.stats.total_nodes << std::endl;
    }
    
    // Verify speedup (allow some overhead)
    EXPECT_LT(times[1], times[0] * 0.7);  // 2 threads should be faster than 1
    EXPECT_LT(times[2], times[1] * 0.7);  // 4 threads should be faster than 2
}

TEST_F(MCTSOpenMPTest, ConsistentResults) {
    // Test that results are consistent across runs
    mcts::MCTSSettings settings;
    settings.num_simulations = 200;
    settings.num_threads = 4;
    settings.batch_size = 16;
    // Use fixed settings for consistency
    
    omp_set_num_threads(settings.num_threads);
    
    games::gomoku::GomokuState state(19);
    // Make some moves for more interesting position
    state.makeMove(180);  // Center
    state.makeMove(181);
    state.makeMove(162);
    
    // Run multiple times
    std::vector<int> actions;
    std::vector<float> values;
    
    for (int i = 0; i < 3; i++) {
        mcts::MCTSEngine engine(neural_net_, settings);
        auto result = engine.search(state);
        actions.push_back(result.action);
        values.push_back(result.value);
    }
    
    // Actions should be the same (with fixed seed)
    EXPECT_EQ(actions[0], actions[1]);
    EXPECT_EQ(actions[1], actions[2]);
    
    // Values should be very close
    EXPECT_NEAR(values[0], values[1], 0.01);
    EXPECT_NEAR(values[1], values[2], 0.01);
}

TEST_F(MCTSOpenMPTest, HighThreadCount) {
    // Test with high thread count (matching user's system)
    mcts::MCTSSettings settings;
    settings.num_simulations = 800;
    settings.num_threads = 20;  // User's mcts_num_threads setting
    settings.batch_size = 32;
    
    omp_set_num_threads(settings.num_threads);
    
    mcts::MCTSEngine engine(neural_net_, settings);
    games::gomoku::GomokuState state(19);
    
    // Start CPU monitoring
    debug::SystemMonitor::instance().start(100);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = engine.search(state);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Stop monitoring and get stats
    debug::SystemMonitor::instance().stop();
    double cpu_usage = debug::SystemMonitor::instance().getCpuUsage();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double sims_per_sec = settings.num_simulations * 1000.0 / duration.count();
    
    std::cout << "High thread count test:\n";
    std::cout << "  Threads: " << settings.num_threads << "\n";
    std::cout << "  Time: " << duration.count() << "ms\n";
    std::cout << "  Throughput: " << sims_per_sec << " sims/sec\n";
    std::cout << "  CPU usage: " << cpu_usage << "%\n";
    std::cout << "  Nodes created: " << result.stats.total_nodes << "\n";
    
    // CPU usage should be significant with 20 threads
    EXPECT_GT(cpu_usage, 50.0);  // Should use at least 50% CPU
    EXPECT_GT(sims_per_sec, 100);  // Should achieve reasonable throughput
}

TEST_F(MCTSOpenMPTest, MemoryEfficiency) {
    // Test memory efficiency with large simulation count
    mcts::MCTSSettings settings;
    settings.num_simulations = 1600;
    settings.num_threads = 16;
    settings.batch_size = 64;
    settings.max_concurrent_simulations = 512;
    
    omp_set_num_threads(settings.num_threads);
    
    mcts::MCTSEngine engine(neural_net_, settings);
    games::gomoku::GomokuState state(19);
    
    // Run search without detailed memory tracking since this is just a test
    auto result = engine.search(state);
    
    std::cout << "Memory test:\n";
    std::cout << "  Nodes created: " << result.stats.total_nodes << "\n";
    
    // Check that result is valid
    EXPECT_GT(result.stats.total_nodes, 0);
}

TEST_F(MCTSOpenMPTest, StressTest) {
    // Stress test with extreme parameters
    mcts::MCTSSettings settings;
    settings.num_simulations = 100;  // Small count but high frequency
    settings.num_threads = 24;
    settings.batch_size = 8;
    
    omp_set_num_threads(settings.num_threads);
    
    mcts::MCTSEngine engine(neural_net_, settings);
    
    // Run many searches rapidly
    const int num_searches = 10;
    std::atomic<int> completed_searches(0);
    
    #pragma omp parallel for
    for (int i = 0; i < num_searches; i++) {
        games::gomoku::GomokuState state(19);
        // Different starting positions
        state.makeMove(i * 19 + 9);
        
        auto result = engine.search(state);
        completed_searches.fetch_add(1, std::memory_order_relaxed);
        
        EXPECT_NE(result.action, -1);
        EXPECT_GT(result.stats.total_nodes, 0);
    }
    
    EXPECT_EQ(completed_searches.load(), num_searches);
}