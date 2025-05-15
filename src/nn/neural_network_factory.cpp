// src/nn/neural_network_factory.cpp
#include "nn/neural_network_factory.h"
#include <memory> // For std::shared_ptr
#include <string> // For std::string
#include <stdexcept> // For std::exception (though likely brought in by other headers)
#include <iostream> // For std::cerr
#include <fstream> // For std::ifstream

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

    // Validate parameters to prevent extremely large models
    if (num_res_blocks > 50) {
        std::cout << "WARNING: Extremely large num_res_blocks value detected (" << num_res_blocks 
                 << "). This may be a parameter error. Limiting to 20 blocks." << std::endl;
        num_res_blocks = 20; // Cap to a reasonable number
    }
    
    if (num_filters < 1) {
        std::cout << "WARNING: Invalid num_filters value detected (" << num_filters 
                 << "). Setting to default value of 128." << std::endl;
        num_filters = 128; // Set to a reasonable default
    }

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

    std::cout << "NeuralNetworkFactory::loadResNet - Loading model from: " << path << std::endl;
    std::cout << "Parameters: input_channels=" << input_channels 
             << ", board_size=" << board_size 
             << ", num_res_blocks=" << num_res_blocks 
             << ", num_filters=" << num_filters 
             << ", policy_size=" << policy_size 
             << ", use_gpu=" << (use_gpu ? "true" : "false") << std::endl;
             
    // Validate parameters to prevent extremely large models
    if (num_res_blocks > 50) {
        std::cout << "WARNING: Extremely large num_res_blocks value detected (" << num_res_blocks 
                 << "). This may be a parameter error. Limiting to 20 blocks." << std::endl;
        num_res_blocks = 20; // Cap to a reasonable number
    }
    
    if (num_filters < 1) {
        std::cout << "WARNING: Invalid num_filters value detected (" << num_filters 
                 << "). Setting to default value of 128." << std::endl;
        num_filters = 128; // Set to a reasonable default
    }

    // Create a model with the correct dimensions
    // For Gomoku games, ensure we're using 17 channels for the enhanced representation
    if (input_channels == 3 && policy_size == board_size * board_size) {
        std::cout << "NeuralNetworkFactory::loadResNet - Detected Gomoku model, using 17 channels for enhanced representation" << std::endl;
        // This is likely a Gomoku model, which should use 17 channels
        input_channels = 17;
    }
    
    // Try to see if the file exists first, otherwise don't waste time creating a model
    {
        std::ifstream file_check(path);
        if (!file_check.good()) {
            std::cout << "NeuralNetworkFactory::loadResNet - File not found: " << path << std::endl;
            std::cout << "NeuralNetworkFactory::loadResNet - Creating a new model with random weights" << std::endl;
            // Just create a new model with random weights
            auto model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
            
            // Create directories if needed
            size_t last_slash = path.find_last_of('/');
            if (last_slash != std::string::npos) {
                std::string dir_path = path.substr(0, last_slash);
                std::cout << "Creating directory: \"" << dir_path << "\"" << std::endl;
                std::string mkdir_cmd = "mkdir -p \"" + dir_path + "\"";
                int result = std::system(mkdir_cmd.c_str());
                if (result != 0) {
                    std::cerr << "Warning: Failed to create directory: " << dir_path << std::endl;
                }
            }
            
            // Save the new model
            try {
                std::cout << "NeuralNetworkFactory::loadResNet - Saving initial model to: " << path << std::endl;
                model->save(path);
            } catch (const std::exception& save_exception) {
                std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
            }
            
            return model;
        }
        // file_check automatically closed when it goes out of scope
    }
    
    // Use the provided architectural parameters
    std::cout << "NeuralNetworkFactory::loadResNet - Creating ResNet model with specified parameters" << std::endl;
    auto model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);

    // Load weights with error handling
    try {
        std::cout << "NeuralNetworkFactory::loadResNet - Attempting to load model from: " << path << std::endl;
        try {
            model->load(path);
            std::cout << "NeuralNetworkFactory::loadResNet - Model loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "NeuralNetworkFactory::loadResNet - Exception during model load: " << e.what() << std::endl;
            std::cerr << "NeuralNetworkFactory::loadResNet - Creating new model" << std::endl;
            // Create a new model instead
            model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
            
            // Save the new model
            try {
                std::cout << "NeuralNetworkFactory::loadResNet - Saving initial model to: " << path << std::endl;
                model->save(path);
            } catch (const std::exception& save_exception) {
                std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
            }
        }
    } catch (const std::exception& load_exception) {
        std::cerr << "NeuralNetworkFactory::loadResNet - Exception during primary model load attempt: " 
                 << load_exception.what() << std::endl;
        
        if (use_gpu) { 
            std::cerr << "NeuralNetworkFactory::loadResNet - Primary load (potentially on GPU) failed. "
                     << "Attempting to create and load model on CPU with same architecture." << std::endl;
            
            // Create a new model instance explicitly for CPU, using the same architecture parameters
            model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, false); 
            try {
                try {
                    model->load(path);
                    std::cout << "NeuralNetworkFactory::loadResNet - Successfully loaded model on CPU after initial failure" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "NeuralNetworkFactory::loadResNet - Error loading model on CPU: " << e.what() << std::endl;
                    std::cerr << "NeuralNetworkFactory::loadResNet - Creating new model on CPU" << std::endl;
                    // Create a new model instead
                    model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, false);
                    
                    // Save the new model
                    try {
                        std::cout << "NeuralNetworkFactory::loadResNet - Saving initial model to: " << path << std::endl;
                        model->save(path);
                    } catch (const std::exception& save_exception) {
                        std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
                    }
                }
            } catch (const std::exception& cpu_load_exception) {
                std::cerr << "NeuralNetworkFactory::loadResNet - Failed to load model on CPU as well: " 
                         << cpu_load_exception.what() << std::endl;
                std::cerr << "NeuralNetworkFactory::loadResNet - Proceeding with a freshly initialized model (random weights) on CPU." << std::endl;
                
                // Save the new model
                try {
                    std::cout << "NeuralNetworkFactory::loadResNet - Saving initial model to: " << path << std::endl;
                    model->save(path);
                } catch (const std::exception& save_exception) {
                    std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
                }
            }
        } else {
            std::cerr << "NeuralNetworkFactory::loadResNet - Model load on CPU failed. "
                    << "Proceeding with a freshly initialized model (random weights)." << std::endl;
            
            // Save the new model
            try {
                std::cout << "NeuralNetworkFactory::loadResNet - Saving initial model to: " << path << std::endl;
                model->save(path);
            } catch (const std::exception& save_exception) {
                std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
            }
        }
    }

    return model;
}
#endif // WITH_TORCH

} // namespace nn
} // namespace alphazero