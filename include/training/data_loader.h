#ifndef ALPHAZERO_TRAINING_DATA_LOADER_H
#define ALPHAZERO_TRAINING_DATA_LOADER_H

#include <vector>
#include <memory>
#include <random>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <future>
#ifdef WITH_TORCH
#include <torch/torch.h>
#endif
#include "training/dataset.h"

namespace alphazero {
namespace training {

#ifdef WITH_TORCH

/**
 * @brief Batch of training data
 */
struct Batch {
    torch::Tensor states;    // [batch_size, channels, height, width]
    torch::Tensor policies;  // [batch_size, action_space]
    torch::Tensor values;    // [batch_size, 1]
    
    // Get the device of the batch
    torch::Device device() const {
        return states.device();
    }
    
    // Move the batch to a different device
    Batch& to(const torch::Device& device) {
        states = states.to(device);
        policies = policies.to(device);
        values = values.to(device);
        return *this;
    }
    
    // Get the size of the batch
    size_t size() const {
        return states.size(0);
    }
};

/**
 * @brief Data loader for AlphaZero training, similar to PyTorch's DataLoader
 */
class DataLoader {
public:
    /**
     * @brief Constructor for DataLoader
     * 
     * @param dataset Dataset to load from
     * @param batch_size Number of examples per batch
     * @param shuffle Whether to shuffle the data between epochs
     * @param num_workers Number of worker threads to use for loading
     * @param pin_memory Whether to pin memory for faster CPU-to-GPU transfers
     * @param drop_last Whether to drop the last incomplete batch
     */
    DataLoader(
        std::shared_ptr<Dataset> dataset,
        size_t batch_size,
        bool shuffle = true,
        size_t num_workers = 0,
        bool pin_memory = false,
        bool drop_last = false);
    
    /**
     * @brief Destructor
     */
    ~DataLoader();
    
    /**
     * @brief Iterator for DataLoader
     */
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t index);
        
        // Iterator requirements
        Iterator& operator++();
        Batch operator*();
        bool operator!=(const Iterator& other) const;
        
    private:
        DataLoader* loader_;
        size_t index_;
    };
    
    /**
     * @brief Begin iterator
     */
    Iterator begin();
    
    /**
     * @brief End iterator
     */
    Iterator end();
    
    /**
     * @brief Get the number of batches
     */
    size_t size() const;
    
    /**
     * @brief Get the batch size
     */
    size_t batch_size() const;
    
    /**
     * @brief Reset the data loader to the beginning
     */
    void reset();
    
public:
    /**
     * @brief Load a batch at a specific index
     * @param batch_index Batch index to load
     * @return Batch of data
     */
    Batch load_batch(size_t batch_index);
    
private:
    // Dataset to load from
    std::shared_ptr<Dataset> dataset_;
    
    // Batch size
    size_t batch_size_;
    
    // Whether to shuffle the data
    bool shuffle_;
    
    // Number of worker threads
    size_t num_workers_;
    
    // Whether to pin memory
    bool pin_memory_;
    
    // Whether to drop the last incomplete batch
    bool drop_last_;
    
    // Current index for single-threaded loading
    size_t current_index_;
    
    // Total number of batches
    size_t num_batches_;
    
    // Flag to check if the dataset is prepared for the current epoch
    bool prepared_;
    
    // Prepare the dataset for the current epoch
    void prepare();
    
    // Multi-threading components
    std::vector<std::thread> workers_;
    std::queue<std::future<Batch>> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_workers_;
    std::atomic<size_t> next_batch_index_;
    
    // Start worker threads
    void start_workers();
    
    // Stop worker threads
    void stop_workers();
    
    // Worker thread function
    void worker_function();
};

#else // !WITH_TORCH
// Dummy class when torch is not available
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset>, size_t = 32, size_t = 4, bool = true) {}
};
#endif // WITH_TORCH

} // namespace training
} // namespace alphazero

#endif // ALPHAZERO_TRAINING_DATA_LOADER_H