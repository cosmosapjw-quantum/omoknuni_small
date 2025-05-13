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
                std::cout << "CUDA is available and working" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "CUDA reported as available but failed verification: " << e.what() << std::endl;
                cuda_available = false;
            }
        } else {
            std::cout << "CUDA is not available, using CPU only" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error checking CUDA availability: " << e.what() << std::endl;
        cuda_available = false;
    }

    return cuda_available;
}

// Get device for tensor operations
torch::Device NeuralNetworkFactory::getDevice(bool force_cpu) {
    if (force_cpu) {
        return torch::kCPU;
    }

#ifdef ENABLE_CUDA_FALLBACK
    // With fallback enabled, try CUDA but fall back to CPU if unavailable
    if (isCudaAvailable()) {
        return torch::kCUDA;
    } else {
        return torch::kCPU;
    }
#else
    // Without fallback, always try to use CUDA and throw if unavailable
    if (torch::cuda::is_available()) {
        return torch::kCUDA;
    } else {
        throw std::runtime_error("CUDA requested but not available");
    }
#endif
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

    std::cout << "Created ResNet model on device: "
              << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    return model;
}

std::shared_ptr<ResNetModel> NeuralNetworkFactory::loadResNet(
    const std::string& path, int64_t input_channels, int64_t board_size,
    int64_t policy_size,
    bool use_gpu) {

    // Create a model with the correct dimensions
    // Use smaller dimensions to avoid memory issues
    auto model = createResNet(input_channels, board_size, 5, 64, policy_size, use_gpu);

    // Load weights with error handling
    try {
        model->load(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << path << ": " << e.what() << std::endl;

        // In fallback mode, try again with CPU if we were using GPU
        if (use_gpu && model->parameters().begin()->device().is_cuda()) {
            std::cout << "Attempting to load model on CPU instead..." << std::endl;
            model = createResNet(input_channels, board_size, 5, 64, policy_size, false);
            model->load(path);
        } else {
            // Re-throw if fallback doesn't apply
            throw;
        }
    }

    return model;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero