#ifndef ALPHAZERO_TRAINING_DATASET_H
#define ALPHAZERO_TRAINING_DATASET_H

#include <vector>
#include <random>
#include <memory>
#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

namespace alphazero {
namespace training {

#ifdef WITH_TORCH

/**
 * @brief Abstract dataset interface for AlphaZero training
 */
class Dataset {
public:
    virtual ~Dataset() = default;
    
    /**
     * @brief Get the number of examples in the dataset
     * @return Size of the dataset
     */
    virtual size_t size() const = 0;
    
    /**
     * @brief Get a single example from the dataset
     * @param index Index of the example to get
     * @return Tuple of (state, policy, value)
     */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get(size_t index) = 0;
    
    /**
     * @brief Shuffle the dataset
     */
    virtual void shuffle() = 0;
};

/**
 * @brief Dataset for AlphaZero training data in memory
 */
class AlphaZeroDataset : public Dataset {
public:
    /**
     * @brief Constructor for AlphaZeroDataset
     * 
     * @param states State tensors (game states) [N, C, H, W]
     * @param policies Policy tensors (move probabilities) [N, action_space]
     * @param values Value tensors (game outcomes) [N]
     * @param device Device to store tensors on
     */
    AlphaZeroDataset(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& states,
        const std::vector<std::vector<float>>& policies,
        const std::vector<float>& values,
        const torch::Device& device = torch::kCPU);
    
    /**
     * @brief Constructor for AlphaZeroDataset with pre-made tensors
     * 
     * @param states State tensors
     * @param policies Policy tensors
     * @param values Value tensors
     */
    AlphaZeroDataset(
        torch::Tensor states,
        torch::Tensor policies,
        torch::Tensor values);
    
    /**
     * @brief Get the number of examples in the dataset
     * @return Size of the dataset
     */
    size_t size() const override;
    
    /**
     * @brief Get a single example from the dataset
     * @param index Index of the example to get
     * @return Tuple of (state, policy, value)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get(size_t index) override;
    
    /**
     * @brief Shuffle the dataset
     */
    void shuffle() override;
    
    /**
     * @brief Create a subset of the dataset
     * @param start Start index
     * @param end End index (exclusive)
     * @return New dataset containing the subset
     */
    std::shared_ptr<AlphaZeroDataset> subset(size_t start, size_t end) const;
    
    /**
     * @brief Get the device the dataset is on
     * @return Device
     */
    torch::Device device() const;
    
    /**
     * @brief Move the dataset to a different device
     * @param device New device
     * @return Reference to this dataset
     */
    AlphaZeroDataset& to(const torch::Device& device);
    
private:
    // Tensors that store the data
    torch::Tensor states_;     // [N, C, H, W]
    torch::Tensor policies_;   // [N, action_space]
    torch::Tensor values_;     // [N, 1]
    
    // Indices for shuffling
    std::vector<size_t> indices_;
    
    // Random number generator for shuffling
    std::mt19937 rng_;
};

#else // !WITH_TORCH
// Dummy classes when torch is not available
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const { return 0; }
};

class AlphaZeroDataset : public Dataset {
public:
    AlphaZeroDataset() {}
    void add(const std::vector<std::vector<std::vector<float>>>&,
            const std::vector<float>&, float) {}
    void add(void*, void*, float) {}
    void shuffle() {}
    AlphaZeroDataset& to(const void*) { return *this; }
};
#endif // WITH_TORCH

} // namespace training
} // namespace alphazero

#endif // ALPHAZERO_TRAINING_DATASET_H