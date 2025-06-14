// include/nn/neural_network.h
#ifndef ALPHAZERO_NN_NEURAL_NETWORK_H
#define ALPHAZERO_NN_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include "core/igamestate.h"
#include "mcts/evaluation_types.h"
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

/**
 * @brief Interface for neural network models used in AlphaZero
 */
class ALPHAZERO_API NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;
    
    /**
     * @brief Perform batch inference on a set of game states
     * 
     * @param states Vector of game states
     * @return Vector of network outputs (policy and value)
     */
    virtual std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) = 0;
        
    /**
     * @brief Evaluate a single game state
     * 
     * @param state The game state to evaluate
     * @return Network output (policy and value)
     */
    virtual mcts::NetworkOutput evaluate(const core::IGameState& state) {
        // Default implementation using inference
        auto state_clone = state.clone();
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.push_back(std::move(state_clone));
        
        auto results = inference(states);
        if (results.empty()) {
            // Return default if inference failed
            mcts::NetworkOutput default_output;
            default_output.value = 0.0f;
            default_output.policy.resize(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
            return default_output;
        }
        
        return results[0];
    }
    
    /**
     * @brief Check if the neural network supports batch evaluation
     * 
     * @return true if evaluateBatch is supported
     */
    virtual bool supportsEvaluateBatch() const {
        return false; // Default implementation does not support evaluateBatch
    }
    
    /**
     * @brief Perform batch evaluation directly (optimized implementation)
     * 
     * @param states Vector of game states
     * @return Vector of network outputs
     */
    virtual std::vector<mcts::NetworkOutput> evaluateBatch(
        const std::vector<std::unique_ptr<core::IGameState>>& states) {
        // Default implementation falls back to inference
        return inference(states);
    }
    
    /**
     * @brief Check if model is deterministic (no randomness during inference)
     * 
     * @return true if model has fixed computation graph
     */
    virtual bool isDeterministic() const {
        return true; // Most models are deterministic by default
    }
    
    /**
     * @brief Enable GPU optimization features
     * 
     * @param enable_cuda_graphs Use CUDA graphs for deterministic models
     * @param enable_persistent_kernels Keep data on GPU between operations
     * @param enable_torch_script Convert to TorchScript for optimization
     * @param cuda_stream_priority Priority for CUDA stream (-1 for default)
     */
    virtual void enableGPUOptimizations(
        bool enable_cuda_graphs = true,
        bool enable_persistent_kernels = true, 
        bool enable_torch_script = true,
        int cuda_stream_priority = -1
    ) {
        // Default implementation does nothing
    }
    
    /**
     * @brief Get current GPU optimization status
     */
    struct GPUOptimizationStatus {
        bool cuda_graphs_enabled = false;
        bool cuda_graphs_supported = false;
        bool persistent_kernels_enabled = false;
        bool torch_script_enabled = false;
        bool model_is_deterministic = true;
        int cuda_stream_priority = -1;
        size_t allocated_memory_mb = 0;
        size_t reserved_memory_mb = 0;
    };
    
    virtual GPUOptimizationStatus getGPUOptimizationStatus() const {
        return GPUOptimizationStatus();
    }
    
    /**
     * @brief Save the model to a file
     * 
     * @param path File path
     */
    virtual void save(const std::string& path) = 0;
    
    /**
     * @brief Load the model from a file
     * 
     * @param path File path
     */
    virtual void load(const std::string& path) = 0;
    
    /**
     * @brief Get the input shape that the network expects
     * 
     * @return Vector of dimensions [channels, height, width]
     */
    virtual std::vector<int64_t> getInputShape() const = 0;
    
    /**
     * @brief Get the policy output size
     * 
     * @return Number of policy outputs
     */
    virtual int64_t getPolicySize() const = 0;
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_NEURAL_NETWORK_H