// include/nn/resnet_model.h
#ifndef ALPHAZERO_NN_RESNET_MODEL_H
#define ALPHAZERO_NN_RESNET_MODEL_H

#include <torch/torch.h>
#include "nn/neural_network.h"
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

/**
 * @brief A residual block for the ResNet architecture
 */
class ALPHAZERO_API ResNetResidualBlock : public torch::nn::Module {
public:
    ResNetResidualBlock(int64_t channels);
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
};

/**
 * @brief ResNet model for AlphaZero
 */
class ALPHAZERO_API ResNetModel : public NeuralNetwork, public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param input_channels Number of input channels
     * @param board_size Board size (assumed square)
     * @param num_res_blocks Number of residual blocks
     * @param num_filters Number of filters in convolutional layers
     * @param policy_size Size of policy output (action space size)
     * 
     * @note IMPORTANT: The board_size parameter locks the model to a specific board size.
     * During inference, game states MUST have the same board size as specified here,
     * otherwise matrix dimension mismatches will occur. This is because the policy head
     * fully-connected layer dimensions depend on the board size.
     */
    ResNetModel(int64_t input_channels, int64_t board_size, 
                int64_t num_res_blocks = 10, int64_t num_filters = 128,
                int64_t policy_size = 0);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Tuple of policy and value tensors
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    
    /**
     * @brief Perform batch inference on a set of game states
     * 
     * @param states Vector of game states
     * @return Vector of network outputs (policy and value)
     */
    std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override;
    
    /**
     * @brief Save the model to a file
     * 
     * @param path File path
     */
    void save(const std::string& path) override;
    
    /**
     * @brief Load the model from a file
     * 
     * @param path File path
     */
    void load(const std::string& path) override;
    
    /**
     * @brief Get the input shape that the network expects
     * 
     * @return Vector of dimensions [channels, height, width]
     */
    std::vector<int64_t> getInputShape() const override;
    
    /**
     * @brief Get the policy output size
     * 
     * @return Number of policy outputs
     */
    int64_t getPolicySize() const override;
    
    /**
     * @brief Clean up tensor pool to free memory
     */
    void cleanupTensorPool();
    
private:
    int64_t input_channels_;
    int64_t board_size_;
    int64_t policy_size_;
    
    torch::nn::Conv2d input_conv_{nullptr};
    torch::nn::BatchNorm2d input_bn_{nullptr};
    torch::nn::ModuleList res_blocks_{nullptr};
    
    // Policy head
    torch::nn::Conv2d policy_conv_{nullptr};
    torch::nn::BatchNorm2d policy_bn_{nullptr};
    torch::nn::Linear policy_fc_{nullptr};
    
    // Value head
    torch::nn::Conv2d value_conv_{nullptr};
    torch::nn::BatchNorm2d value_bn_{nullptr};
    torch::nn::Linear value_fc1_{nullptr};
    torch::nn::Linear value_fc2_{nullptr};
    
    // Tensor preparation - original signature
    torch::Tensor prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states);
    // Tensor preparation - overloaded signature with target device
    torch::Tensor prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states, torch::Device target_device);
    
    // Tensor pool for pre-allocated GPU tensors
    struct TensorPool {
        std::vector<torch::Tensor> cpu_tensors;
        std::vector<torch::Tensor> gpu_tensors;
        std::atomic<size_t> current_idx{0};
        size_t pool_size{4};
        bool initialized{false};
        
        void init(int64_t batch_size, int64_t channels, int64_t height, int64_t width, torch::Device device);
        torch::Tensor getCPUTensor(size_t batch_size);
        torch::Tensor getGPUTensor(size_t batch_size);
        void cleanup();  // Add cleanup method
    };
    
    TensorPool tensor_pool_;
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_RESNET_MODEL_H