// src/nn/neural_network_factory.cpp
#include "nn/neural_network_factory.h"

namespace alphazero {
namespace nn {

#ifdef WITH_TORCH
std::shared_ptr<ResNetModel> NeuralNetworkFactory::createResNet(
    int64_t input_channels, int64_t board_size, 
    int64_t num_res_blocks, int64_t num_filters,
    int64_t policy_size) {
    
    auto model = std::make_shared<ResNetModel>(
        input_channels, board_size, num_res_blocks, num_filters, policy_size);
    
    return model;
}

std::shared_ptr<ResNetModel> NeuralNetworkFactory::loadResNet(
    const std::string& path, int64_t input_channels, int64_t board_size, 
    int64_t policy_size) {
    
    // Create a model with the correct dimensions
    auto model = createResNet(input_channels, board_size, 10, 128, policy_size);
    
    // Load weights
    model->load(path);
    
    return model;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero