#ifndef ALPHAZERO_UTILS_CUDA_UTILS_H
#define ALPHAZERO_UTILS_CUDA_UTILS_H

#include <torch/torch.h>
#include <string>
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace alphazero {
namespace cuda {

/**
 * @brief Safely synchronize CUDA operations
 */
inline void safe_synchronize() {
    try {
        if (torch::cuda::is_available()) {
            // Synchronize all CUDA streams using PyTorch's API
            torch::cuda::synchronize();
            
            // For newer PyTorch versions that might not expose cudaDeviceSynchronize,
            // we use the CUDA Runtime API directly
            cudaDeviceSynchronize();
        }
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error synchronizing CUDA: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error synchronizing CUDA: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error synchronizing CUDA" << std::endl;
    }
}

/**
 * @brief Calculate memory required for tensors
 * @param batch_size Batch size
 * @param channels Number of channels
 * @param height Height
 * @param width Width
 * @param policy_size Policy size
 * @return Memory required in bytes
 */
inline size_t calculate_tensor_memory(
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width,
    size_t policy_size) {
    
    // Calculate memory for one example
    // State tensor: batch_size * channels * height * width * sizeof(float)
    size_t state_memory = batch_size * channels * height * width * sizeof(float);
    
    // Policy tensor: batch_size * policy_size * sizeof(float)
    size_t policy_memory = batch_size * policy_size * sizeof(float);
    
    // Value tensor: batch_size * sizeof(float)
    size_t value_memory = batch_size * sizeof(float);
    
    // PyTorch overhead (approximately 2x)
    return (state_memory + policy_memory + value_memory) * 2;
}

/**
 * @brief Get optimal batch size based on available CUDA memory
 * @param channels Number of channels in state
 * @param height Height of state
 * @param width Width of state
 * @param policy_size Size of policy vector
 * @param min_batch_size Minimum batch size to return
 * @param max_batch_size Maximum batch size to return
 * @param memory_limit Memory limit as fraction of available memory (0.0-1.0)
 * @return Optimal batch size
 */
inline size_t get_optimal_batch_size(
    size_t channels,
    size_t height,
    size_t width,
    size_t policy_size,
    size_t min_batch_size = 16,
    size_t max_batch_size = 1024,
    float memory_limit = 0.7) {
    
    // Default batch size if can't determine
    size_t default_batch = 128;
    
    // If CUDA is not available, use a conservative batch size
    if (!torch::cuda::is_available()) {
        return std::min(default_batch, max_batch_size);
    }
    
    try {
        // Get free CUDA memory using CUDA Runtime API
        size_t free_memory = 0;
        size_t total_memory = 0;
        
        // Initialize CUDA runtime
        cudaFree(nullptr);
        
        // Get memory info directly from CUDA
        cudaMemGetInfo(&free_memory, &total_memory);
        
        // Get device properties using updated PyTorch API
        // auto props = at::cuda::getDeviceProperties(0); // Unused variable
        
        // Apply memory limit
        free_memory = static_cast<size_t>(free_memory * memory_limit);
        
        // Calculate memory per example
        size_t example_memory = calculate_tensor_memory(1, channels, height, width, policy_size);
        
        // Calculate optimal batch size
        size_t optimal_batch = free_memory / example_memory;
        
        // Limit to reasonable range
        optimal_batch = std::max(optimal_batch, min_batch_size);
        optimal_batch = std::min(optimal_batch, max_batch_size);
        
        std::cout << "Optimal batch size: " << optimal_batch
                  << " (free memory: " << (free_memory / (1024*1024)) << " MB"
                  << ", memory per example: " << (example_memory / 1024) << " KB)"
                  << std::endl;
        
        return optimal_batch;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error calculating optimal batch size: " << e.what() << std::endl;
        return default_batch;
    } catch (const std::exception& e) {
        std::cerr << "Error calculating optimal batch size: " << e.what() << std::endl;
        return default_batch;
    } catch (...) {
        std::cerr << "Unknown error calculating optimal batch size" << std::endl;
        return default_batch;
    }
}

/**
 * @brief Create a properly disposed tensor
 * @param sizes Tensor sizes
 * @param options Tensor options
 * @return Tensor
 * 
 * Creates a tensor that's properly initialized and managed to avoid device leaks
 */
inline torch::Tensor create_tensor(torch::IntArrayRef sizes, const torch::TensorOptions& options) {
    try {
        // Detect device issues before creating the tensor
        if (options.device_opt().has_value()) {
            auto device = options.device_opt().value();
            if (device.is_cuda() && !torch::cuda::is_available()) {
                std::cerr << "Warning: Creating tensor on CUDA device but CUDA is not available, falling back to CPU" << std::endl;
                return torch::zeros(sizes, options.device(torch::kCPU));
            }
            
            if (device.is_cuda() && device.index() >= torch::cuda::device_count()) {
                std::cerr << "Warning: CUDA device index " << device.index() << " is out of range (max "
                          << (torch::cuda::device_count() - 1) << "), falling back to device 0" << std::endl;
                return torch::zeros(sizes, options.device(torch::Device(torch::kCUDA, 0)));
            }
        }
        
        // Create the tensor
        return torch::zeros(sizes, options);
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error creating tensor: " << e.what() << std::endl;
        // Fall back to CPU
        return torch::zeros(sizes, torch::TensorOptions().device(torch::kCPU));
    } catch (const std::exception& e) {
        std::cerr << "Error creating tensor: " << e.what() << std::endl;
        // Fall back to CPU
        return torch::zeros(sizes, torch::TensorOptions().device(torch::kCPU));
    }
}

/**
 * @brief Process tensors in chunks to avoid OOM errors
 * @param process_func Function to process each chunk
 * @param tensors Vector of tensors to process
 * @param max_chunk_size Maximum chunk size
 */
template<typename Func>
inline void process_in_chunks(
    Func process_func,
    const std::vector<torch::Tensor>& tensors,
    size_t max_chunk_size = 128) {
    
    size_t num_tensors = tensors.size();
    size_t num_chunks = (num_tensors + max_chunk_size - 1) / max_chunk_size;
    
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * max_chunk_size;
        size_t end = std::min(start + max_chunk_size, num_tensors);
        
        // Create vector for this chunk
        std::vector<torch::Tensor> chunk_tensors(tensors.begin() + start, tensors.begin() + end);
        
        // Process the chunk
        process_func(chunk_tensors, chunk, start, end);
        
        // Clean up after processing
        safe_synchronize();
    }
}

/**
 * @brief Get all available CUDA devices
 * @return Vector of device indices
 */
inline std::vector<int> get_available_devices() {
    std::vector<int> devices;
    
    if (torch::cuda::is_available()) {
        int device_count = torch::cuda::device_count();
        for (int i = 0; i < device_count; i++) {
            devices.push_back(i);
        }
    }
    
    return devices;
}

/**
 * @brief Get memory usage on CUDA device
 * @param device_index Device index
 * @return Pair of (used memory in MB, total memory in MB)
 */
inline std::pair<float, float> get_memory_usage(int device_index = 0) {
    if (!torch::cuda::is_available()) {
        return {0.0f, 0.0f};
    }
    
    try {
        // Validate device index
        int device_count = torch::cuda::device_count();
        if (device_index >= device_count) {
            std::cerr << "Invalid device index " << device_index << " (max " << (device_count-1) << ")" << std::endl;
            return {0.0f, 0.0f};
        }
        
        // Use cudaSetDevice directly instead of torch::cuda::set_device
        cudaSetDevice(device_index);
        
        // Get memory info directly from CUDA Runtime API
        size_t free_memory = 0;
        size_t total_memory = 0;
        
        cudaMemGetInfo(&free_memory, &total_memory);
        
        // Get device properties
        // auto props = at::cuda::getDeviceProperties(device_index); // Unused variable
        
        // Used memory is total minus free
        size_t used_memory = total_memory - free_memory;
        
        // Convert to MB
        float used_mb = static_cast<float>(used_memory) / (1024.0f * 1024.0f);
        float total_mb = static_cast<float>(total_memory) / (1024.0f * 1024.0f);
        
        return {used_mb, total_mb};
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error getting memory usage: " << e.what() << std::endl;
        return {0.0f, 0.0f};
    } catch (const std::exception& e) {
        std::cerr << "Error getting memory usage: " << e.what() << std::endl;
        return {0.0f, 0.0f};
    } catch (...) {
        std::cerr << "Unknown error getting memory usage" << std::endl;
        return {0.0f, 0.0f};
    }
}

/**
 * @brief Check if a tensor has device issues
 * @param tensor Tensor to check
 * @return true if there's a potential device issue
 */
inline bool has_device_issue(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return false;  // No device to check
    }
    
    try {
        auto device = tensor.device();
        if (device.is_cuda() && !torch::cuda::is_available()) {
            return true;  // CUDA device but CUDA not available
        }
        
        if (device.is_cuda() && device.index() >= torch::cuda::device_count()) {
            return true;  // CUDA device index out of range
        }
        
        // Attempt to access the data to verify device validity
        tensor.item<float>();
        return false;  // No issues detected
    } catch (const c10::Error& e) {
        std::string error_msg = e.what();
        // Check for known device-related error messages
        if (error_msg.find("CUDA error") != std::string::npos ||
            error_msg.find("device") != std::string::npos ||
            error_msg.find("101") != std::string::npos) {
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        return true;  // Any other exception might indicate a device issue
    }
}

/**
 * @brief Fix tensor device issues by moving to a valid device
 * @param tensor Tensor to fix
 * @return Fixed tensor
 */
inline torch::Tensor fix_device_issues(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return tensor;
    }
    
    if (!has_device_issue(tensor)) {
        return tensor;  // No issues to fix
    }
    
    try {
        // Move to CPU first as a safe option
        auto cpu_tensor = tensor.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true);
        
        // Move back to CUDA if available
        if (torch::cuda::is_available()) {
            return cpu_tensor.to(torch::Device(torch::kCUDA, 0));
        }
        
        return cpu_tensor;
    } catch (const std::exception& e) {
        std::cerr << "Error fixing tensor device: " << e.what() << std::endl;
        // Create a new tensor on CPU as a last resort
        return torch::zeros_like(tensor, torch::TensorOptions().device(torch::kCPU));
    }
}

} // namespace cuda
} // namespace alphazero

#endif // ALPHAZERO_UTILS_CUDA_UTILS_H