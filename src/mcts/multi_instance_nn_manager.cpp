#include "mcts/multi_instance_nn_manager.h"
#include "nn/neural_network_factory.h"
#include "utils/logger.h"
#include <thread>
#include <future>
#include <chrono>

// Use neural network logger
#define LOG_INFO(...) LOG_NN_INFO(__VA_ARGS__)
#define LOG_WARNING(...) LOG_NN_WARN(__VA_ARGS__)
#define LOG_ERROR(...) LOG_NN_ERROR(__VA_ARGS__)
#define LOG_DEBUG(...) LOG_NN_DEBUG(__VA_ARGS__)

namespace alphazero {
namespace mcts {

MultiInstanceNNManager::MultiInstanceNNManager(
    const std::string& model_path, 
    int num_instances,
    const nn::NeuralNetworkConfig& config) 
    : model_path_(model_path)
    , config_(config)
    , start_time_(std::chrono::steady_clock::now()) {
    
    LOG_INFO("Creating {} independent neural network instances", num_instances);
    
    // Reserve space for inference threads (created on demand)
    inference_threads_.reserve(num_instances);
    
    // Initialize instances
    instances_.reserve(num_instances);
    
    for (int i = 0; i < num_instances; ++i) {
        auto instance = std::make_unique<NNInstance>();
        
        // Distribute instances across available GPUs
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        instance->device_id = (num_gpus > 0) ? (i % num_gpus) : 0;
        
        // Set device for this instance
        cudaSetDevice(instance->device_id);
        
        // Create independent CUDA stream
        cudaStreamCreate(&instance->cuda_stream);
        
        // Load independent copy of the model
        LOG_INFO("Loading instance {} on GPU {}", i, instance->device_id);
        
        instance->network = nn::NeuralNetworkFactory::loadResNet(
            model_path,
            config.input_channels,
            config.board_size,
            config.num_res_blocks,
            config.num_filters,
            config.board_size * config.board_size,  // policy_size
            true  // use_gpu
        );
        
        // Note: setCudaStream would need to be implemented in the neural network class
        // For now, we'll handle stream management in the inference method
        
        instances_.push_back(std::move(instance));
    }
    
    LOG_INFO("All instances initialized successfully");
}

MultiInstanceNNManager::~MultiInstanceNNManager() {
    // Clean up CUDA streams
    for (auto& instance : instances_) {
        if (instance->cuda_stream) {
            cudaStreamDestroy(instance->cuda_stream);
        }
    }
}

std::shared_ptr<nn::NeuralNetwork> MultiInstanceNNManager::getInstance(int engine_id) {
    if (engine_id < 0 || engine_id >= static_cast<int>(instances_.size())) {
        LOG_ERROR("Invalid engine_id: {}", engine_id);
        return nullptr;
    }
    
    return instances_[engine_id]->network;
}

std::future<std::vector<NetworkOutput>> MultiInstanceNNManager::asyncInference(
    int engine_id,
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    if (engine_id < 0 || engine_id >= static_cast<int>(instances_.size())) {
        LOG_ERROR("Invalid engine_id: {}", engine_id);
        // Return empty result
        std::promise<std::vector<NetworkOutput>> promise;
        promise.set_value(std::vector<NetworkOutput>());
        return promise.get_future();
    }
    
    total_requests_++;
    
    // Create async task
    return std::async(std::launch::async, [this, engine_id, &states]() {
        auto& instance = *instances_[engine_id];
        
        // Mark instance as busy
        bool expected = false;
        if (!instance.is_busy.compare_exchange_strong(expected, true)) {
            LOG_WARNING("Instance {} is already busy!", engine_id);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Set device context
        cudaSetDevice(instance.device_id);
        
        // Perform inference on dedicated instance
        auto results = instance.network->inference(states);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        // Update metrics
        instance.inference_count++;
        double old_time = instance.total_inference_time.load();
        instance.total_inference_time.store(old_time + duration);
        
        // Mark instance as available
        instance.is_busy = false;
        
        return results;
    });
}

void MultiInstanceNNManager::printStatistics() const {
    auto now = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(now - start_time_).count();
    
    LOG_INFO("=== Performance Statistics ===");
    LOG_INFO("Total requests: {}", total_requests_.load());
    LOG_INFO("Total time: {} seconds", total_time);
    LOG_INFO("Requests/sec: {}", total_requests_.load() / total_time);
    
    for (size_t i = 0; i < instances_.size(); ++i) {
        const auto& instance = *instances_[i];
        double avg_time = (instance.inference_count > 0) 
            ? instance.total_inference_time / instance.inference_count 
            : 0.0;
        
        LOG_INFO("Instance {}: GPU={} Inferences={} AvgTime={}ms Busy={}",
                 i, instance.device_id, instance.inference_count.load(),
                 avg_time * 1000, instance.is_busy.load());
    }
}

} // namespace mcts
} // namespace alphazero