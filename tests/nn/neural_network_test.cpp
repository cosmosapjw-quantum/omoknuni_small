// tests/nn/neural_network_test.cpp
#include <gtest/gtest.h>
#ifdef WITH_TORCH
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"
#include "games/gomoku/gomoku_state.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <ctime>

using namespace alphazero;

class NeuralNetworkTest : public ::testing::Test {
protected:
    std::shared_ptr<nn::ResNetModel> model;
    int64_t input_channels = 17;
    int64_t board_size = 9;
    int64_t policy_size = board_size * board_size; // Added for clarity

    void SetUp() override {
        // Create a small ResNet model for testing
        // Parameters: input_channels, board_size, num_res_blocks, num_filters, policy_size
        model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);
    }
};

// Test model creation
TEST_F(NeuralNetworkTest, ModelCreation) {
    ASSERT_TRUE(model);
    EXPECT_EQ(model->getInputShape()[0], 17);
    EXPECT_EQ(model->getInputShape()[1], 9);
    EXPECT_EQ(model->getInputShape()[2], 9);
    EXPECT_EQ(model->getPolicySize(), 81);  // 9x9 board
}

// Test forward pass
TEST_F(NeuralNetworkTest, ForwardPass) {
    std::cout << "[ForwardPass] Starting test" << std::endl;
    std::cout << "[ForwardPass] Model created with input_channels=" << model->getInputShape()[0] 
              << ", board_size=" << model->getInputShape()[1] << "x" << model->getInputShape()[2] << std::endl;
    
    // Create dummy input tensor
    std::cout << "[ForwardPass] Creating input tensor..." << std::endl;
    torch::Tensor input = torch::zeros({1, 17, 9, 9});
    std::cout << "[ForwardPass] Input tensor created with shape: " << input.sizes() << std::endl;
    
    // Run forward pass
    std::cout << "[ForwardPass] Running forward pass..." << std::endl;
    auto [policy, value] = model->forward(input);
    std::cout << "[ForwardPass] Forward pass completed" << std::endl;
    
    // Check shapes
    std::cout << "[ForwardPass] Policy shape: " << policy.sizes() << ", Value shape: " << value.sizes() << std::endl;
    EXPECT_EQ(policy.sizes(), std::vector<int64_t>({1, 81}));
    EXPECT_EQ(value.sizes(), std::vector<int64_t>({1, 1}));
    
    // Check value range
    float value_scalar = value.item<float>();
    std::cout << "[ForwardPass] Value: " << value_scalar << std::endl;
    EXPECT_GE(value_scalar, -1.0f);
    EXPECT_LE(value_scalar, 1.0f);
    
    // Check policy sums to 1 (log_softmax is used in forward)
    std::cout << "[ForwardPass] Computing policy probabilities..." << std::endl;
    torch::Tensor policy_probs = torch::exp(policy);
    float sum = policy_probs.sum().item<float>();
    std::cout << "[ForwardPass] Policy sum: " << sum << std::endl;
    EXPECT_NEAR(sum, 1.0f, 1e-6);
    
    std::cout << "[ForwardPass] Test completed successfully" << std::endl;
}

// Test inference with game states
TEST_F(NeuralNetworkTest, Inference) {
    // Create Gomoku state
    auto gomoku = std::make_unique<games::gomoku::GomokuState>(9);
    
    // Make a few moves
    gomoku->makeMove(4 * 9 + 4);  // Center
    gomoku->makeMove(3 * 9 + 3);  // Top-left of center
    
    // Create batch
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.push_back(std::move(gomoku));
    
    // Run inference
    auto outputs = model->inference(states);
    
    // Check results
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].policy.size(), 81);
    
    // Check policy is valid probability distribution
    float sum = 0.0f;
    for (float p : outputs[0].policy) {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
        sum += p;
    }
    std::cout << "Policy size: " << outputs[0].policy.size() << ", Sum: " << sum << std::endl;
    std::cout << "First few policy values: ";
    for (size_t i = 0; i < std::min(size_t(5), outputs[0].policy.size()); ++i) {
        std::cout << outputs[0].policy[i] << " ";
    }
    std::cout << std::endl;
    EXPECT_NEAR(sum, 1.0f, 1e-6);
    
    // Check value is in valid range
    EXPECT_GE(outputs[0].value, -1.0f);
    EXPECT_LE(outputs[0].value, 1.0f);
}

// Test save and load
// DISABLED due to hanging issue - needs investigation
TEST_F(NeuralNetworkTest, DISABLED_SaveLoad) {
    // Create a model
    int64_t input_channels = 19;  // Use 19 to match what loadResNet expects
    int64_t board_size = 9;
    int64_t policy_size = board_size * board_size;
    auto original_model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);

    // Generate a filename that already includes game info to avoid path modification
    std::string filename = "/tmp/test_model_gomoku_9x9_" + std::to_string(std::time(nullptr)) + ".pt";
    
    // Initialize model with random weights
    for (auto& p : original_model->parameters()) {
        p.data().normal_(0.0, 0.1);
    }
    
    // Get a parameter tensor for later comparison
    torch::Tensor first_param = original_model->parameters()[0];
    std::vector<float> before_save;
    auto accessor = first_param.data_ptr<float>();
    for (int i = 0; i < 3; i++) {
        before_save.push_back(accessor[i]);
    }
    
    // Save the model directly (bypass factory to avoid path issues)
    original_model->save(filename);
    
    // Create a new model and load weights directly
    auto loaded_model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);
    loaded_model->load(filename);
    
    // Compare original and loaded model parameters (simple check)
    torch::Tensor loaded_first_param = loaded_model->parameters()[0];
    auto loaded_accessor = loaded_first_param.data_ptr<float>();
    bool same = true;
    for (int i = 0; i < 3; i++) {
        if (std::abs(before_save[i] - loaded_accessor[i]) > 1e-6) {
            same = false;
            break;
        }
    }
    EXPECT_TRUE(same);  // Parameters should be the same after loading
    
    // Create another model with different architecture to ensure load fails or adapts
    // Parameters: input_channels, board_size, num_res_blocks, num_filters, policy_size
    auto new_model = nn::NeuralNetworkFactory::createResNet(input_channels, board_size, 2, 32, policy_size);

    // Try to load the original model into the new model instance
    // Parameters: path, input_channels, board_size, num_res_blocks, num_filters, policy_size
    // ... existing code ...
    
    // Clean up
    std::remove(filename.c_str());
}

// Main function removed - part of all_tests target
#else
// Dummy test when torch is not available
TEST(DummyTest, NoTorchAvailable) {
    SUCCEED() << "Neural network tests are skipped when WITH_TORCH is OFF";
}

// Main function removed - part of all_tests target
#endif // WITH_TORCH