// src/nn/neural_network_factory.cpp
#include "nn/neural_network_factory.h"

namespace alphazero {
namespace nn {

#ifdef WITH_TORCH
// Check if CUDA is available and working
bool NeuralNetworkFactory::isCudaAvailable() {
    bool cuda_available = false;
    try {
        // Check if CUDA is available through PyTorch
        cuda_available = torch::cuda::is_available();

        if (cuda_available) {
            // Additional verification: Try to create a small tensor on CUDA
            try {
                torch::Tensor test_tensor = torch::ones({1, 1}, torch::kCUDA);
                // If we got here, CUDA is working
            } catch (const std::exception&) {
                cuda_available = false;
            }
        }
    } catch (const std::exception&) {
        cuda_available = false;
    }

    return cuda_available;
}

// Get device for tensor operations
torch::Device NeuralNetworkFactory::getDevice(bool force_cpu) {
    if (force_cpu) {
        return torch::kCPU;
    }

    // Always try CUDA but fall back to CPU if unavailable
    if (isCudaAvailable()) {
        return torch::kCUDA;
    } else {
        return torch::kCPU;
    }
}

std::shared_ptr<ResNetModel> NeuralNetworkFactory::createResNet(
    int64_t input_channels, int64_t board_size,
    int64_t num_res_blocks, int64_t num_filters,
    int64_t policy_size,
    bool use_gpu) {

    // Create the model
    auto model = std::make_shared<ResNetModel>(
        input_channels, board_size, num_res_blocks, num_filters, policy_size);

    // Move to appropriate device
    torch::Device device = getDevice(!use_gpu);
    model->to(device);


    return model;
}

std::shared_ptr<ResNetModel> NeuralNetworkFactory::loadResNet(
    const std::string& path, int64_t input_channels, int64_t board_size,
    int64_t policy_size,
    bool use_gpu) {

    // Create a model with the correct dimensions
    // For Gomoku games, ensure we're using 17 channels for the enhanced representation
    if (input_channels == 3 && policy_size == board_size * board_size) {
        // This is likely a Gomoku model, which should use 17 channels
        input_channels = 17;
    }
    
    // Use smaller dimensions to avoid memory issues
    auto model = createResNet(input_channels, board_size, 5, 64, policy_size, use_gpu);

    // Load weights with error handling
    try {
        model->load(path);
    } catch (const std::exception&) {
        // Try again with CPU if we were using GPU
        if (use_gpu && model->parameters().begin()->device().is_cuda()) {
            model = createResNet(input_channels, board_size, 5, 64, policy_size, false);
            try {
                model->load(path);
            } catch (const std::exception&) {
                // Silently continue with initialized model
            }
        }
    }

    return model;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero