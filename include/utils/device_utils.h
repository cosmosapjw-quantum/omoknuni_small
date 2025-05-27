#ifndef ALPHAZERO_UTILS_DEVICE_UTILS_H
#define ALPHAZERO_UTILS_DEVICE_UTILS_H

#include <torch/torch.h>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace alphazero {
namespace utils {

/**
 * @brief Utility functions for device handling
 */
class DeviceUtils {
public:
    /**
     * @brief Check if CUDA is available and working
     * @return true if CUDA is available, false otherwise
     */
    static bool isCudaAvailable() {
        bool cuda_available_basic = false;
        try {
            // Step 1: Basic check using PyTorch's API. This is usually lightweight.
            cuda_available_basic = torch::cuda::is_available() && torch::cuda::device_count() > 0;
        } catch (const c10::Error& e) { // Catch PyTorch specific errors first
            std::cerr << "PyTorch (c10::Error) during basic torch::cuda::is_available() check: " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) { // Catch standard C++ errors
            std::cerr << "Standard exception during basic torch::cuda::is_available() check: " << e.what() << std::endl;
            return false;
        } catch (...) { // Catch any other type of exception
            std::cerr << "Unknown error during basic torch::cuda::is_available() check." << std::endl;
            return false;
        }

        if (!cuda_available_basic) {
            // std::cout << "CUDA not available based on torch::cuda::is_available() or device_count is 0." << std::endl; // Optional: for debugging
            return false;
        }

        // Step 2: Functional verification by trying a small CUDA operation.
        // This is a more robust check but can also be where environment issues surface.
        bool cuda_verified_functional = false;
        try {
            torch::Tensor test_tensor = torch::ones({1, 1}, torch::TensorOptions().device(torch::kCUDA));
            
            // Ensure the tensor is indeed on a CUDA device and accessible.
            // Accessing an item forces synchronization and data retrieval.
            test_tensor[0][0].item<float>(); // This will synchronize and might throw if CUDA context is bad.

            cuda_verified_functional = true;
            // Tensor 'test_tensor' goes out of scope here. PyTorch's caching allocator handles its memory.
        } catch (const c10::Error& e) {
            std::cerr << "CUDA verification (tensor op) failed with c10::Error: " << e.what() << std::endl;
            // cuda_verified_functional remains false
        } catch (const std::exception& e) {
            std::cerr << "CUDA verification (tensor op) failed with std::exception: " << e.what() << std::endl;
            // cuda_verified_functional remains false
        } catch (...) {
            std::cerr << "CUDA verification (tensor op) failed with unknown error." << std::endl;
            // cuda_verified_functional remains false
        }
        
        if (!cuda_verified_functional) {
            std::cerr << "CUDA reported as available by PyTorch, but functional test (tensor operation) failed." << std::endl;
        }

        return cuda_verified_functional;
    }
    
    /**
     * @brief Get device for tensor operations
     * @param force_cpu Whether to force CPU usage
     * @return Device to use
     */
    static torch::Device getDevice(bool force_cpu = false) {
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
    
    /**
     * @brief Safe method to get the device of a tensor
     * @param tensor Tensor to get device from
     * @return Device of the tensor, or CPU if not defined or error
     */
    static torch::Device getDeviceSafe(const torch::Tensor& tensor) {
        if (!tensor.defined()) {
            return torch::kCPU;
        }
        
        try {
            return tensor.device();
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error getting tensor device: " << e.what() << std::endl;
            return torch::kCPU;
        } catch (const std::exception& e) {
            std::cerr << "Error getting tensor device: " << e.what() << std::endl;
            return torch::kCPU;
        } catch (...) {
            std::cerr << "Unknown error getting tensor device" << std::endl;
            return torch::kCPU;
        }
    }
    
    /**
     * @brief Safe method to get the device from model parameters
     * @param model Model to get device from
     * @return Device of the model, or CPU if not defined or error
     */
    template<typename ModelType>
    static torch::Device getModelDeviceSafe(const std::shared_ptr<ModelType>& model) {
        if (!model) {
            return torch::kCPU;
        }
        
        try {
            auto params = model->parameters();
            if (params.empty()) {
                return torch::kCPU;
            }
            
            // Directly use the first parameter's device
            return getDeviceSafe(params[0]);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error getting model device: " << e.what() << std::endl;
            return torch::kCPU;
        } catch (const std::exception& e) {
            std::cerr << "Error getting model device: " << e.what() << std::endl;
            return torch::kCPU;
        } catch (...) {
            std::cerr << "Unknown error getting model device" << std::endl;
            return torch::kCPU;
        }
    }
    
    /**
     * @brief Safely move a tensor to a device
     * @param tensor Tensor to move
     * @param device Device to move to
     * @return Tensor on the target device, or original tensor if error
     */
    static torch::Tensor toDeviceSafe(const torch::Tensor& tensor, const torch::Device& device) {
        if (!tensor.defined()) {
            return tensor;
        }
        
        try {
            // First check if tensor is already on the target device
            if (tensor.device() == device) {
                return tensor;
            }
            
            return tensor.to(device);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error moving tensor to device " << device << ": " << e.what() << std::endl;
            return tensor;
        } catch (const std::exception& e) {
            std::cerr << "Error moving tensor to device " << device << ": " << e.what() << std::endl;
            return tensor;
        } catch (...) {
            std::cerr << "Unknown error moving tensor to device " << device << std::endl;
            return tensor;
        }
    }
    
    /**
     * @brief Safely move a model to a device
     * @param model Model to move
     * @param device Device to move to
     * @return true if successful, false otherwise
     */
    template<typename ModelType>
    static bool toDeviceSafe(std::shared_ptr<ModelType>& model, const torch::Device& device) {
        if (!model) {
            return false;
        }
        
        try {
            model->to(device);
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error moving model to device " << device << ": " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Error moving model to device " << device << ": " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "Unknown error moving model to device " << device << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Get the CUDA device properties
     * @param device_index CUDA device index
     * @return String with device properties
     */
    static std::string getCudaDeviceProperties(int device_index = 0) {
        if (!isCudaAvailable()) {
            return "CUDA not available";
        }
        
        try {
            int num_devices = torch::cuda::device_count();
            if (device_index >= num_devices) {
                return "Invalid device index (max " + std::to_string(num_devices-1) + ")";
            }
            
            // Use PyTorch's API to get device properties
            auto props = at::cuda::getDeviceProperties(device_index);
            
            std::ostringstream ss;
            ss << "CUDA Device " << device_index << ": " << props->name << "\n"
               << "  Compute capability: " << props->major << "." << props->minor << "\n"
               << "  Total memory: " << (props->totalGlobalMem / (1024*1024)) << " MB";
               
            return ss.str();
        } catch (const std::exception& e) {
            return "Error getting CUDA device properties: " + std::string(e.what());
        }
    }
    
    /**
     * @brief Get optimal batch size based on available CUDA memory
     * @param tensor_size Size of each tensor in bytes
     * @param reserved_memory Memory to reserve for other operations (in MB)
     * @return Optimal batch size or default (128) if estimation fails
     */
    static size_t getOptimalBatchSize(size_t tensor_size, size_t reserved_memory = 1024) {
        if (!isCudaAvailable()) {
            return 64;  // Default for CPU
        }
        
        try {
            // Get free CUDA memory using current API
            // size_t total_memory = 0; // Unused variable
            size_t free_memory = 0;
            
            // Get total memory from device properties
            // auto props = at::cuda::getDeviceProperties(0); // Unused variable
            // total_memory = props->totalGlobalMem / (1024 * 1024); // Convert to MB - This line is also part of the unused variable
            
            // In newer PyTorch versions, use cudaMemGetInfo directly with C++ CUDA runtime API
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            
            // Initialize CUDA runtime
            cudaFree(nullptr);
            
            // Get memory info directly from CUDA runtime
            cudaMemGetInfo(&free_bytes, &total_bytes);
            
            // Convert to MB
            free_memory = free_bytes / (1024 * 1024);
            
            // Reserve some memory
            free_memory = (free_memory > reserved_memory) ? (free_memory - reserved_memory) : 0;
            
            // Calculate optimal batch size
            size_t tensor_size_mb = tensor_size / (1024 * 1024);
            if (tensor_size_mb == 0) tensor_size_mb = 1;  // Avoid division by zero
            
            size_t optimal_batch_size = free_memory / tensor_size_mb;
            
            // Cap to reasonable values
            optimal_batch_size = std::min(optimal_batch_size, size_t(1024));
            optimal_batch_size = std::max(optimal_batch_size, size_t(16));
            
            std::cout << "Optimal batch size: " << optimal_batch_size 
                      << " (free memory: " << free_memory << " MB, tensor size: " 
                      << tensor_size_mb << " MB)" << std::endl;
            
            return optimal_batch_size;
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error calculating optimal batch size: " << e.what() << std::endl;
            return 128;  // Default fallback
        } catch (const std::exception& e) {
            std::cerr << "Error calculating optimal batch size: " << e.what() << std::endl;
            return 128;  // Default fallback
        } catch (...) {
            std::cerr << "Unknown error calculating optimal batch size" << std::endl;
            return 128;  // Default fallback
        }
    }
};

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_UTILS_DEVICE_UTILS_H