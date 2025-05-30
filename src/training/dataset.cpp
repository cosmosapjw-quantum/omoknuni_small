#include "training/dataset.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <random> // For std::random_device and std::mt19937

#ifdef WITH_TORCH
// Add necessary headers for CUDA API calls
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#endif

namespace alphazero {
namespace training {

#ifdef WITH_TORCH

AlphaZeroDataset::AlphaZeroDataset(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& states,
    const std::vector<std::vector<float>>& policies,
    const std::vector<float>& values,
    const torch::Device& device)
{
    // Validate input dimensions
    if (states.empty() || policies.empty() || values.empty()) {
        throw std::invalid_argument("Dataset inputs cannot be empty");
    }
    
    if (states.size() != policies.size() || states.size() != values.size()) {
        throw std::invalid_argument("Dataset inputs must have the same length");
    }
    
    // Get dimensions from the first example
    size_t num_examples = states.size();
    size_t channels = states[0].size();
    size_t height = states[0][0].size();
    size_t width = states[0][0][0].size();
    size_t policy_size = policies[0].size();
    
    std::cout << "Creating dataset with " << num_examples << " examples, "
              << "state shape: [" << channels << ", " << height << ", " << width << "], "
              << "policy size: " << policy_size << std::endl;
    
    // Debug device
    std::cout << "Dataset target device: " << device << std::endl;
    std::cout << "Is CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    }
    
    // Create tensor options with appropriate device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    // Initialize tensors
    states_ = torch::zeros({static_cast<int64_t>(num_examples), 
                           static_cast<int64_t>(channels), 
                           static_cast<int64_t>(height), 
                           static_cast<int64_t>(width)}, options);
                           
    policies_ = torch::zeros({static_cast<int64_t>(num_examples), 
                             static_cast<int64_t>(policy_size)}, options);
                             
    values_ = torch::zeros({static_cast<int64_t>(num_examples), 1}, options);
    
    // Copy data to tensors
    try {
        // Temporary tensors on CPU for safety during construction
        auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto temp_states = torch::zeros({static_cast<int64_t>(num_examples), 
                                       static_cast<int64_t>(channels), 
                                       static_cast<int64_t>(height), 
                                       static_cast<int64_t>(width)}, cpu_options);
        auto temp_policies = torch::zeros({static_cast<int64_t>(num_examples), 
                                         static_cast<int64_t>(policy_size)}, cpu_options);
        auto temp_values = torch::zeros({static_cast<int64_t>(num_examples), 1}, cpu_options);
        
        // Copy states with safer memory access
        for (size_t i = 0; i < num_examples; i++) {
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    // Use contiguous memory for each row
                    if (width == states[i][c][h].size()) {
                        std::memcpy(temp_states[i][c][h].data_ptr(), 
                                   states[i][c][h].data(), 
                                   width * sizeof(float));
                    } else {
                        // Fallback to element-wise copy if sizes don't match
                        for (size_t w = 0; w < width; w++) {
                            if (w < states[i][c][h].size()) {
                                temp_states[i][c][h][w] = states[i][c][h][w];
                            }
                        }
                    }
                }
            }
            
            // Copy policy with safer memory access
            if (policy_size == policies[i].size()) {
                std::memcpy(temp_policies[i].data_ptr(), 
                           policies[i].data(), 
                           policy_size * sizeof(float));
            } else {
                // Fallback to element-wise copy
                for (size_t j = 0; j < policy_size && j < policies[i].size(); j++) {
                    temp_policies[i][j] = policies[i][j];
                }
            }
            
            // Copy value
            temp_values[i][0] = values[i];
        }
        
        // Now safely move to target device with error handling
        try {
            torch::Device effective_final_device = device; // Local variable for effective device

            // Check GPU memory before movement (based on original requested device)
            if (device.is_cuda()) {
                std::cout << "Before moving to GPU (requested device " << device << "): " 
                          << "GPU memory allocated=" << c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).allocated_bytes[static_cast<size_t>(0)].current 
                          << ", reserved=" << c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).reserved_bytes[static_cast<size_t>(0)].current << std::endl;
                
                // Calculate tensor sizes
                size_t total_elements = static_cast<size_t>(temp_states.numel()) + 
                                      static_cast<size_t>(temp_policies.numel()) + 
                                      static_cast<size_t>(temp_values.numel());
                size_t approx_bytes = total_elements * 4; // float32 = 4 bytes
                std::cout << "Estimated tensor memory for dataset: " << approx_bytes << " bytes" << std::endl;
                
                // Check if we should move to GPU
                if (approx_bytes > 0.9 * at::cuda::getDeviceProperties(device.index())->totalGlobalMem) {
                    std::cerr << "WARNING: Tensor size too large for GPU memory. Falling back to CPU for dataset tensors." << std::endl;
                    effective_final_device = torch::kCPU; // Fallback to CPU
                }
            }
            
            std::cout << "Moving dataset tensors to final device: " << effective_final_device << std::endl;
            
            // Move tensors individually with verbose error handling
            try {
                states_ = temp_states.to(effective_final_device);
                std::cout << "States tensor moved successfully to " << effective_final_device << std::endl;
            } catch (const torch::Error& e) { // Use torch::Error
                std::cerr << "Error moving states tensor: " << e.what() << std::endl;
                throw;
            }
            
            try {
                policies_ = temp_policies.to(effective_final_device);
                std::cout << "Policies tensor moved successfully to " << effective_final_device << std::endl;
            } catch (const torch::Error& e) { // Use torch::Error
                std::cerr << "Error moving policies tensor: " << e.what() << std::endl;
                throw;
            }
            
            try {
                values_ = temp_values.to(effective_final_device);
                std::cout << "Values tensor moved successfully to " << effective_final_device << std::endl;
            } catch (const torch::Error& e) { // Use torch::Error
                std::cerr << "Error moving values tensor: " << e.what() << std::endl;
                throw;
            }
            
            // Verify tensors are on the correct device
            std::cout << "Final tensor devices - States: " << states_.device() 
                     << ", Policies: " << policies_.device()
                     << ", Values: " << values_.device() << std::endl;
            
            if (effective_final_device.is_cuda()) {
                std::cout << "After moving to GPU (device " << effective_final_device << "): "
                          << "GPU memory allocated=" << c10::cuda::CUDACachingAllocator::getDeviceStats(effective_final_device.index()).allocated_bytes[static_cast<size_t>(0)].current
                          << ", reserved=" << c10::cuda::CUDACachingAllocator::getDeviceStats(effective_final_device.index()).reserved_bytes[static_cast<size_t>(0)].current << std::endl;
            }
        } catch (const torch::Error& e) { // Use torch::Error
            std::cerr << "PyTorch error moving tensors to device: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU for dataset tensors" << std::endl;
            // Keep the CPU tensors as-is if we can't move them
            states_ = temp_states;
            policies_ = temp_policies;
            values_ = temp_values;
        }
        
        // Clear CPU tensors to free memory if we successfully moved to a different device (GPU)
        if (states_.device() != torch::kCPU) { // Check actual device of states_
            temp_states = torch::Tensor();
            temp_policies = torch::Tensor();
            temp_values = torch::Tensor();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error copying data to tensors: " << e.what() << std::endl;
        throw;
    }
    
    // Initialize indices for shuffling
    indices_.resize(num_examples);
    for (size_t i = 0; i < num_examples; i++) {
        indices_[i] = i;
    }
    
    // Initialize random number generator with random seed
    std::random_device rd;
    rng_ = std::mt19937(rd());
    
    std::cout << "Dataset created successfully on device: " << device << std::endl;
}

AlphaZeroDataset::AlphaZeroDataset(
    torch::Tensor states,
    torch::Tensor policies,
    torch::Tensor values)
    : states_(states), policies_(policies), values_(values)
{
    // Validate input dimensions
    if (states_.dim() != 4) {
        throw std::invalid_argument("States tensor must have 4 dimensions [N, C, H, W]");
    }
    
    if (policies_.dim() != 2) {
        throw std::invalid_argument("Policies tensor must have 2 dimensions [N, action_space]");
    }
    
    if (values_.dim() != 2 || values_.size(1) != 1) {
        // Reshape if needed
        values_ = values_.reshape({values_.size(0), 1});
    }
    
    // Ensure all tensors have the same number of examples
    int64_t num_examples = states_.size(0);
    if (policies_.size(0) != num_examples || values_.size(0) != num_examples) {
        throw std::invalid_argument("All tensors must have the same number of examples");
    }
    
    // Ensure all tensors are on the same device
    if (states_.device() != policies_.device() || states_.device() != values_.device()) {
        throw std::invalid_argument("All tensors must be on the same device");
    }
    
    // Initialize indices for shuffling
    indices_.resize(num_examples);
    for (size_t i = 0; i < num_examples; i++) {
        indices_[i] = i;
    }
    
    // Initialize random number generator with random seed
    std::random_device rd;
    rng_ = std::mt19937(rd());
    
    std::cout << "Dataset created with pre-made tensors (" << num_examples << " examples) on device: " 
              << states_.device() << std::endl;
}

size_t AlphaZeroDataset::size() const {
    return states_.size(0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AlphaZeroDataset::get(size_t index) {
    // Apply shuffling via indices_
    size_t shuffled_index = indices_[index];
    
    // Extract slices from the tensors
    auto state = states_[shuffled_index];
    auto policy = policies_[shuffled_index];
    auto value = values_[shuffled_index];
    
    return {state, policy, value};
}

void AlphaZeroDataset::shuffle() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

std::shared_ptr<AlphaZeroDataset> AlphaZeroDataset::subset(size_t start, size_t end) const {
    if (start >= end || end > size()) {
        throw std::out_of_range("Invalid subset range");
    }
    
    // Extract slices from the tensors
    auto subset_states = states_.slice(0, start, end);
    auto subset_policies = policies_.slice(0, start, end);
    auto subset_values = values_.slice(0, start, end);
    
    // Create a new dataset with the slices
    return std::make_shared<AlphaZeroDataset>(subset_states, subset_policies, subset_values);
}

torch::Device AlphaZeroDataset::device() const {
    return states_.device();
}

AlphaZeroDataset& AlphaZeroDataset::to(const torch::Device& device) {
    try {
        states_ = states_.to(device);
        policies_ = policies_.to(device);
        values_ = values_.to(device);
    } catch (const torch::Error& e) { // Use torch::Error
        std::cerr << "PyTorch error moving dataset to device " << device << ": " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error moving dataset to device " << device << ": " << e.what() << std::endl;
        throw;
    }
    
    return *this;
}

#endif // WITH_TORCH

} // namespace training
} // namespace alphazero