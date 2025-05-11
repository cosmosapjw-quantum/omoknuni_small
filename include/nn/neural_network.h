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