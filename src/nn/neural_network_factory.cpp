// src/nn/neural_network_factory.cpp
#include "nn/neural_network_factory.h"
#include <memory> // For std::shared_ptr
#include <string> // For std::string
#include <stdexcept> // For std::exception (though likely brought in by other headers)
#include <iostream> // For std::cerr

#ifdef WITH_TORCH
#include "utils/device_utils.h" // Moved to global scope
#endif // WITH_TORCH

namespace alphazero {
namespace nn {

// Define method implementations based on WITH_TORCH flag
#ifdef WITH_TORCH
// utils/device_utils.h was moved up

// Use our robust device detection utilities
bool NeuralNetworkFactory::isCudaAvailable() {
    return alphazero::utils::DeviceUtils::isCudaAvailable();
}

// Get device for tensor operations, safely handling device issues
torch::Device NeuralNetworkFactory::getDevice(bool force_cpu) {
    return alphazero::utils::DeviceUtils::getDevice(force_cpu);
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
    int64_t num_res_blocks, int64_t num_filters,
    int64_t policy_size,
    bool use_gpu) {

    // Create a model with the correct dimensions
    // For Gomoku games, ensure we're using 17 channels for the enhanced representation
    if (input_channels == 3 && policy_size == board_size * board_size) {
        // This is likely a Gomoku model, which should use 17 channels
        input_channels = 17;
    }
    
    // Use the provided architectural parameters
    auto model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);

    // Load weights with error handling
    try {
        model->load(path);
    } catch (const std::exception& load_exception) {
        std::cerr << "Warning: Exception during primary model load attempt: " << load_exception.what() << std::endl;
        
        if (use_gpu) { 
            std::cerr << "Primary load (potentially on GPU) failed. Attempting to create and load model on CPU with same architecture." << std::endl;
            // Create a new model instance explicitly for CPU, using the same architecture parameters
            model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, false); 
            try {
                model->load(path); 
                std::cerr << "Successfully created and loaded model on CPU after initial failure." << std::endl;
            } catch (const std::exception& cpu_load_exception) {
                std::cerr << "Warning: Failed to load model on CPU as well: " << cpu_load_exception.what() << std::endl;
                std::cerr << "Proceeding with a freshly initialized model (random weights) on CPU." << std::endl;
            }
        } else {
            std::cerr << "Model load on CPU failed. Proceeding with a freshly initialized model (random weights)." << std::endl;
        }
    }

    return model;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero