// include/nn/resnet_model.h
#ifndef ALPHAZERO_NN_RESNET_MODEL_H
#define ALPHAZERO_NN_RESNET_MODEL_H

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

#include "nn/neural_network.h"
#include "core/export_macros.h"
#include "mcts/gpu_memory_pool.h"

namespace alphazero {
namespace nn {

#ifdef WITH_TORCH
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
    
    // Tensor pool cleanup method removed - no longer using tensor pools
    
    /**
     * @brief Set GPU memory pool for efficient tensor allocation
     * @param pool Shared pointer to GPU memory pool
     */
    void setGPUMemoryPool(std::shared_ptr<mcts::GPUMemoryPool> pool) {
        gpu_memory_pool_ = pool;
    }
    
    // GPU optimization support
    bool isDeterministic() const override { return true; }  // ResNet is deterministic
    
    void enableGPUOptimizations(
        bool enable_cuda_graphs = true,
        bool enable_persistent_kernels = true, 
        bool enable_torch_script = true,
        int cuda_stream_priority = -1
    ) override;
    
    GPUOptimizationStatus getGPUOptimizationStatus() const override;
    
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
    
    // Tensor pool removed - direct allocation is used instead
    
    // GPU memory pool for efficient tensor allocation
    std::shared_ptr<mcts::GPUMemoryPool> gpu_memory_pool_;
    
    // GPU optimization state
    mutable GPUOptimizationStatus gpu_opt_status_;
    std::shared_ptr<torch::jit::Module> torch_script_model_;
    bool use_torch_script_ = false;
};

#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_RESNET_MODEL_H