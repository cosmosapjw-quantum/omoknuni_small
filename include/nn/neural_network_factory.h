// include/nn/neural_network_factory.h
#ifndef ALPHAZERO_NN_NEURAL_NETWORK_FACTORY_H
#define ALPHAZERO_NN_NEURAL_NETWORK_FACTORY_H

#include <memory>
#include <string>
#include "nn/neural_network.h"
#ifdef WITH_TORCH
#include "nn/resnet_model.h"
#endif
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

/**
 * @brief Factory for creating neural network models
 */
class ALPHAZERO_API NeuralNetworkFactory {
public:
#ifdef WITH_TORCH
    /**
     * @brief Check if CUDA is available and working
     *
     * @return True if CUDA is available and verified working
     */
    static bool isCudaAvailable();

    /**
     * @brief Get the appropriate device for tensor operations
     *
     * @param force_cpu Force CPU usage even if CUDA is available
     * @return Device to use for tensor operations
     */
    static torch::Device getDevice(bool force_cpu = false);

    /**
     * @brief Create a ResNet model
     *
     * @param input_channels Number of input channels
     * @param board_size Board size (assumed square)
     * @param num_res_blocks Number of residual blocks
     * @param num_filters Number of filters in convolutional layers
     * @param policy_size Size of policy output (action space size)
     * @param use_gpu Whether to use GPU for the model if available
     * @return Shared pointer to the model
     */
    static std::shared_ptr<ResNetModel> createResNet(
        int64_t input_channels, int64_t board_size,
        int64_t num_res_blocks, int64_t num_filters,
        int64_t policy_size,
        bool use_gpu = true);

    /**
     * @brief Load a model from a file
     *
     * @param path File path
     * @param input_channels Number of input channels (needed for initialization)
     * @param board_size Board size (needed for initialization)
     * @param num_res_blocks Number of residual blocks (added)
     * @param num_filters Number of filters in convolutional layers (added)
     * @param policy_size Policy size (needed for initialization)
     * @param use_gpu Whether to use GPU for the model if available
     * @return Shared pointer to the loaded model
     */
    static std::shared_ptr<ResNetModel> loadResNet(
        const std::string& path, int64_t input_channels, int64_t board_size,
        int64_t num_res_blocks, int64_t num_filters,
        int64_t policy_size = 0,
        bool use_gpu = true);
#endif // WITH_TORCH
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_NEURAL_NETWORK_FACTORY_H