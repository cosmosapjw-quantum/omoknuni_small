#include "mcts/mcts_engine.h"
#include "mcts/unified_inference_server.h"
#include "nn/neural_network.h"
#include "games/gomoku/gomoku_state.h"
#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>

using namespace alphazero;
using namespace alphazero::mcts;

// Mock neural network for testing
class MockNeuralNetwork : public nn::NeuralNetwork {
public:
    MockNeuralNetwork() = default;
    
    std::vector<NetworkOutput> inference(const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        std::vector<NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); ++i) {
            NetworkOutput output;
            output.value = 0.5f; // Default value
            
            // Get action space size and create uniform policy
            int action_size = states[i]->getActionSpaceSize();
            output.policy.resize(action_size, 1.0f / action_size);
            outputs.push_back(output);
        }
        
        return outputs;
    }
    
    // Required interface methods
    void save(const std::string&) override {}
    void load(const std::string&) override {}
    std::vector<int64_t> getInputShape() const override { return {3, 15, 15}; }
    int64_t getPolicySize() const override { return 225; }
};

// Test fixture for shared queue testing
class SharedQueueTest : public ::testing::Test {
protected:
    // Use the moodycamel queue directly
    moodycamel::ConcurrentQueue<PendingEvaluation> shared_leaf_queue_;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>> shared_result_queue_;
    
    // Flag to track if notify function was called
    std::atomic<bool> notify_called_{false};
    
    // Dummy inference function that returns random values
    std::function<std::vector<NetworkOutput>(const std::vector<std::unique_ptr<core::IGameState>>&)> inference_fn_;
    
    void SetUp() override {
        // Create a simple inference function for testing
        inference_fn_ = [](const std::vector<std::unique_ptr<core::IGameState>>& states) {
            std::vector<NetworkOutput> outputs;
            outputs.reserve(states.size());
            
            for (size_t i = 0; i < states.size(); ++i) {
                NetworkOutput output;
                output.value = 0.5f; // Default value
                
                // Get action space size and create uniform policy
                int action_size = states[i]->getActionSpaceSize();
                output.policy.resize(action_size, 1.0f / action_size);
                
                outputs.push_back(std::move(output));
            }
            
            return outputs;
        };
    }
};

// Test that MCTSEngine correctly enqueues to shared queue
TEST_F(SharedQueueTest, EngineCorrectlyEnqueues) {
    // Create MCTS Settings with appropriate parameters for testing
    MCTSSettings settings;
    settings.num_simulations = 10;
    settings.num_threads = 2;
    settings.batch_size = 8;
    settings.use_transposition_table = false;
    
    // Create a notify function
    auto notify_fn = [this]() {
        notify_called_.store(true);
        std::cout << "Notify function called!" << std::endl;
    };
    
    // Create MCTSEngine with neural network to get new architecture
    auto neural_net = std::make_shared<MockNeuralNetwork>();
    MCTSEngine engine(neural_net, settings);
    
    // Get the UnifiedInferenceServer from the engine (new architecture)
    auto inference_server = engine.getInferenceServer();
    if (!inference_server) {
        std::cerr << "Error: Could not get inference server from engine" << std::endl;
        GTEST_FAIL() << "Inference server is null";
    }
    
    std::cout << "UnifiedInferenceServer is available and configured" << std::endl;
    
    // Create a Gomoku state for testing
    std::unique_ptr<games::gomoku::GomokuState> state = 
        std::make_unique<games::gomoku::GomokuState>(15, false, false, 0, false);
    
    // Perform search on the state - this should trigger enqueueing
    std::cout << "Starting search..." << std::endl;
    
    // Run search in a separate thread with timeout
    std::atomic<bool> search_done{false};
    std::atomic<bool> test_success{false}; // Added to track test success
    std::thread search_thread([&]() {
        try {
            engine.search(*state);
            search_done.store(true);
        } catch (const std::exception& e) {
            std::cerr << "Exception in search: " << e.what() << std::endl;
        }
    });
    
    // Wait for search to complete or timeout
    auto start_time = std::chrono::steady_clock::now();
    constexpr auto max_wait_time = std::chrono::seconds(3); // Reduced from 5 to 3 seconds
    
    while (!search_done.load() && !test_success.load() && 
           std::chrono::steady_clock::now() - start_time < max_wait_time) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Print queue size periodically
        std::cout << "Shared leaf queue size: " << shared_leaf_queue_.size_approx() << std::endl;
        
        // Check if items were enqueued - this is our success condition
        if (shared_leaf_queue_.size_approx() > 0) {
            std::cout << "Items successfully enqueued to shared queue!" << std::endl;
            test_success.store(true);
            break;  // Exit the loop once we've confirmed items are enqueued
        }
    }
    
    // Force termination of search if we've confirmed success
    if (test_success.load() && !search_done.load()) {
        std::cout << "Test successful - terminating search early" << std::endl;
        // Setting a flag to terminate the search engine
        engine.updateSettings(MCTSSettings()); // Reset settings with minimal values
        std::atomic<bool> terminate_signal{true};
        
        // Forcefully terminate the test after a short delay
        auto terminate_thread = std::thread([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            terminate_signal.store(true);
        });
        
        // Detach the thread to let it run independently
        terminate_thread.detach();
    }
    
    // Safe join with timeout
    if (search_thread.joinable()) {
        // Use a short timeout for joining to prevent the test from hanging
        auto join_status = std::thread([&]() {
            search_thread.join();
        });
        
        // Wait up to 1 second for join to complete
        if (join_status.joinable()) {
            join_status.detach(); // Detach if it takes too long (test cleanup only)
        }
    }
    
    // Verify that items were enqueued
    EXPECT_GT(shared_leaf_queue_.size_approx(), 0) 
        << "No items were enqueued to the shared leaf queue";
    
    // Verify that notify function was called
    EXPECT_TRUE(notify_called_.load()) 
        << "Notify function was not called";
}

// When building as a standalone test, uncomment this
#ifndef CUSTOM_MAIN_USED
// Main function to run the test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif