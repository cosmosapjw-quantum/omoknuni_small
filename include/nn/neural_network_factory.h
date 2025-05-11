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
     * @brief Create a ResNet model
     * 
     * @param input_channels Number of input channels
     * @param board_size Board size (assumed square)
     * @param num_res_blocks Number of residual blocks
     * @param num_filters Number of filters in convolutional layers
     * @param policy_size Size of policy output (action space size)
     * @return Shared pointer to the model
     */
    static std::shared_ptr<ResNetModel> createResNet(
        int64_t input_channels, int64_t board_size, 
        int64_t num_res_blocks = 10, int64_t num_filters = 128,
        int64_t policy_size = 0);
    
    /**
     * @brief Load a model from a file
     * 
     * @param path File path
     * @param input_channels Number of input channels (needed for initialization)
     * @param board_size Board size (needed for initialization)
     * @param policy_size Policy size (needed for initialization)
     * @return Shared pointer to the loaded model
     */
    static std::shared_ptr<ResNetModel> loadResNet(
        const std::string& path, int64_t input_channels, int64_t board_size, 
        int64_t policy_size = 0);
#endif // WITH_TORCH
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_NEURAL_NETWORK_FACTORY_H