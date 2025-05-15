#include "training/dataset.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace alphazero {
namespace training {

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
            states_ = temp_states.to(device);
            policies_ = temp_policies.to(device);
            values_ = temp_values.to(device);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error moving tensors to device " << device << ": " << e.what() << std::endl;
            std::cerr << "Falling back to CPU" << std::endl;
            // Keep the CPU tensors as-is if we can't move them
            states_ = temp_states;
            policies_ = temp_policies;
            values_ = temp_values;
        }
        
        // Clear CPU tensors to free memory if we successfully moved to device
        if (device != torch::kCPU && states_.device() == device) {
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
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error moving dataset to device " << device << ": " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error moving dataset to device " << device << ": " << e.what() << std::endl;
        throw;
    }
    
    return *this;
}

} // namespace training
} // namespace alphazero