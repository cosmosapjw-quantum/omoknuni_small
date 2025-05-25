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

    // Determine game type from board size and policy size
    std::string game_type = "unknown";
    if (board_size == 8 && policy_size == 64) {
        game_type = "chess";
        // For Chess, always use 8x8 board
        board_size = 8;
        policy_size = 64;
    } else if (board_size >= 5 && board_size <= 19 && policy_size == board_size * board_size) {
        // Could be Gomoku or Go - check if path contains hints
        if (board_size <= 15) {
            game_type = "gomoku";
        } else {
            game_type = "go";
        }
    }
    
    // Modify path to include game type and board size
    std::string actual_path = path;
    
    // Check if the path already contains game type and board size info
    bool already_has_game_info = (path.find("_chess") != std::string::npos) ||
                                 (path.find("_gomoku_") != std::string::npos) ||
                                 (path.find("_go_") != std::string::npos);
    
    if (!already_has_game_info) {
        // Only modify if the path doesn't already have game info
        size_t dot_pos = path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            if (game_type == "chess") {
                // For chess: model.pt -> model_chess.pt
                actual_path = path.substr(0, dot_pos) + "_chess" + path.substr(dot_pos);
            } else if (game_type != "unknown") {
                // For Gomoku/Go: model.pt -> model_gomoku_15x15.pt
                actual_path = path.substr(0, dot_pos) + "_" + game_type + "_" + 
                             std::to_string(board_size) + "x" + std::to_string(board_size) + path.substr(dot_pos);
            }
        }
    }

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
        // This is likely a Gomoku model, which should use 17 channels
        input_channels = 17;
    }
    
    // Log the actual model path being used
    if (actual_path != path) {
        std::cerr << "NeuralNetworkFactory: Using board-specific model: " << actual_path << std::endl;
    }
    
    // Try to see if the file exists first, otherwise don't waste time creating a model
    {
        std::ifstream file_check(actual_path);
        if (!file_check.good()) {
            // Just create a new model with random weights
            auto model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
            
            // Create directories if needed
            size_t last_slash = actual_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                std::string dir_path = actual_path.substr(0, last_slash);
                std::string mkdir_cmd = "mkdir -p \"" + dir_path + "\"";
                int result = std::system(mkdir_cmd.c_str());
                if (result != 0) {
                    std::cerr << "Warning: Failed to create directory: " << dir_path << std::endl;
                }
            }
            
            // Save the new model
            try {
                model->save(actual_path);
            } catch (const std::exception& save_exception) {
                std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
            }
            
            return model;
        }
        // file_check automatically closed when it goes out of scope
    }
    
    // Use the provided architectural parameters
    auto model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);

    // Load weights with error handling
    try {
        try {
            model->load(actual_path);
        } catch (const std::exception& e) {
            std::cerr << "NeuralNetworkFactory::loadResNet - Exception during model load: " << e.what() << std::endl;
            
            // Check if this is a shape mismatch error
            std::string error_msg = e.what();
            if (error_msg.find("shapes cannot be multiplied") != std::string::npos) {
                std::cerr << "ERROR: Model architecture mismatch!" << std::endl;
                std::cerr << "The saved model was trained for a different board size." << std::endl;
                std::cerr << "Expected board size: " << board_size << "x" << board_size << std::endl;
                std::cerr << "Creating a new model with the correct architecture..." << std::endl;
            } else {
                std::cerr << "NeuralNetworkFactory::loadResNet - Creating new model" << std::endl;
            }
            
            // Create a new model instead
            model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, use_gpu);
            
            // Save the new model
            try {
                model->save(actual_path);
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
                    model->load(actual_path);
                } catch (const std::exception& e) {
                    std::cerr << "NeuralNetworkFactory::loadResNet - Error loading model on CPU: " << e.what() << std::endl;
                    std::cerr << "NeuralNetworkFactory::loadResNet - Creating new model on CPU" << std::endl;
                    // Create a new model instead
                    model = createResNet(input_channels, board_size, num_res_blocks, num_filters, policy_size, false);
                    
                    // Save the new model
                    try {
                                model->save(actual_path);
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
                        model->save(actual_path);
                } catch (const std::exception& save_exception) {
                    std::cerr << "Warning: Failed to save initial model: " << save_exception.what() << std::endl;
                }
            }
        } else {
            std::cerr << "NeuralNetworkFactory::loadResNet - Model load on CPU failed. "
                    << "Proceeding with a freshly initialized model (random weights)." << std::endl;
            
            // Save the new model
            try {
                model->save(actual_path);
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