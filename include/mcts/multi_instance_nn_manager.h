#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <future>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include "core/export_macros.h"
#include "nn/neural_network.h"
#include "mcts/evaluation_types.h"

namespace alphazero {

namespace nn {
// Forward declaration
struct NeuralNetworkConfig {
    int input_channels;
    int board_size;
    int num_res_blocks;
    int num_filters;
};
}

namespace mcts {

/**
 * @brief Manages multiple independent neural network instances for true parallel inference
 * 
 * Key features:
 * - Each MCTS engine gets its own neural network instance
 * - Independent CUDA streams for each instance
 * - No shared state or synchronization between instances
 * - Async inference pipeline with futures
 */
class ALPHAZERO_API MultiInstanceNNManager {
public:
    struct NNInstance {
        std::shared_ptr<nn::NeuralNetwork> network;
        cudaStream_t cuda_stream;
        int device_id;
        std::atomic<bool> is_busy{false};
        
        // Performance metrics
        std::atomic<size_t> inference_count{0};
        std::atomic<double> total_inference_time{0.0};
    };
    
    /**
     * @brief Initialize manager with multiple neural network instances
     * @param model_path Path to the model file
     * @param num_instances Number of independent instances to create
     * @param config Neural network configuration
     */
    MultiInstanceNNManager(const std::string& model_path, 
                          int num_instances,
                          const nn::NeuralNetworkConfig& config);
    
    ~MultiInstanceNNManager();
    
    /**
     * @brief Get a neural network instance for an MCTS engine
     * @param engine_id Unique identifier for the MCTS engine
     * @return Shared pointer to an independent neural network
     */
    std::shared_ptr<nn::NeuralNetwork> getInstance(int engine_id);
    
    /**
     * @brief Perform async inference on a specific instance
     * @param engine_id Engine requesting inference
     * @param states Game states to evaluate
     * @return Future containing the evaluation results
     */
    std::future<std::vector<NetworkOutput>> asyncInference(
        int engine_id,
        const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    /**
     * @brief Get performance statistics
     */
    void printStatistics() const;
    
private:
    std::vector<std::unique_ptr<NNInstance>> instances_;
    std::string model_path_;
    nn::NeuralNetworkConfig config_;
    
    // Threads for async operations
    std::vector<std::thread> inference_threads_;
    
    // Performance monitoring
    std::atomic<size_t> total_requests_{0};
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace mcts
} // namespace alphazero