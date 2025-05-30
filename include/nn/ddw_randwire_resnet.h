#ifndef ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H
#define ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <memory>
#include <tuple>
#include "nn/neural_network.h"
#include "mcts/gpu_memory_pool.h"
#include "core/export_macros.h"

namespace alphazero {
namespace nn {

/**
 * @brief Graph generation method for RandWire networks
 */
enum class GraphGenMethod {
    WATTS_STROGATZ,  // Small-world graph
    ERDOS_RENYI,     // Random graph
    BARABASI_ALBERT  // Scale-free graph
};

/**
 * @brief Squeeze-and-Excitation Block for attention mechanism
 */
class SEBlock : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param channels Number of input and output channels
     * @param reduction Reduction ratio for the bottleneck
     */
    SEBlock(int64_t channels, int64_t reduction = 16);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Output tensor with channel attention applied
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::AdaptiveAvgPool2d squeeze{nullptr};
    torch::nn::Sequential excitation{nullptr};
};

/**
 * @brief Residual block with batch normalization
 */
class ResidualBlock : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param channels Number of input and output channels
     */
    ResidualBlock(int64_t channels);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Output tensor with residual connection
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    std::shared_ptr<SEBlock> se{nullptr};
};

/**
 * @brief Router module for dynamic wiring
 */
class RouterModule : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     */
    RouterModule(int64_t in_channels, int64_t out_channels);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Output tensor with routing applied
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};

/**
 * @brief Dynamic routing gate for instance-aware connections
 */
class DynamicRoutingGate : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param channels Number of channels
     * @param num_nodes Number of nodes in the graph
     */
    DynamicRoutingGate(int64_t channels, int64_t num_nodes);
    
    /**
     * @brief Compute dynamic routing weights
     * 
     * @param x Input tensor
     * @param edge_features Edge feature tensor
     * @return Routing weights for each edge
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_features);

private:
    torch::nn::Conv2d feature_extractor{nullptr};
    torch::nn::Linear edge_scorer{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};

/**
 * @brief Directed graph edge for random wiring
 */
struct Edge {
    int from;
    int to;
    
    bool operator==(const Edge& other) const {
        return from == other.from && to == other.to;
    }
};

/**
 * @brief Custom hash for Edge struct
 */
struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        return std::hash<int>()(edge.from) ^ std::hash<int>()(edge.to);
    }
};

/**
 * @brief Simple directed graph implementation
 */
class DiGraph {
public:
    /**
     * @brief Add a node to the graph
     * 
     * @param node Node identifier
     */
    void add_node(int node);
    
    /**
     * @brief Add an edge to the graph
     * 
     * @param from Source node
     * @param to Target node
     */
    void add_edge(int from, int to);
    
    /**
     * @brief Get all nodes in the graph
     * 
     * @return Vector of node identifiers
     */
    std::vector<int> nodes() const;
    
    /**
     * @brief Get predecessors of a node
     * 
     * @param node Node identifier
     * @return Vector of predecessor node identifiers
     */
    std::vector<int> predecessors(int node) const;
    
    /**
     * @brief Get successors of a node
     * 
     * @param node Node identifier
     * @return Vector of successor node identifiers
     */
    std::vector<int> successors(int node) const;
    
    /**
     * @brief Get in-degree of a node
     * 
     * @param node Node identifier
     * @return Number of incoming edges
     */
    int in_degree(int node) const;
    
    /**
     * @brief Get out-degree of a node
     * 
     * @param node Node identifier
     * @return Number of outgoing edges
     */
    int out_degree(int node) const;
    
    /**
     * @brief Get topological sort of nodes
     * 
     * @return Vector of node identifiers in topological order
     */
    std::vector<int> topological_sort() const;
    
    /**
     * @brief Get all edges in the graph
     * 
     * @return Vector of edges
     */
    std::vector<Edge> edges() const;
    
    /**
     * @brief Get the number of nodes
     * 
     * @return Number of nodes
     */
    size_t size() const;

private:
    std::unordered_set<int> nodes_;
    std::unordered_map<int, std::vector<int>> adjacency_list_;
    std::unordered_map<int, std::vector<int>> reverse_adjacency_list_;
    
    /**
     * @brief Depth-first topological sort helper
     * 
     * @param node Current node
     * @param visited Set of visited nodes
     * @param temp_visited Set of temporarily visited nodes (for cycle detection)
     * @param result Resulting topological sort
     * @return true if successful, false if cycles detected
     */
    bool _dfs_topo_sort(int node, std::unordered_set<int>& visited, 
                        std::unordered_set<int>& temp_visited, 
                        std::vector<int>& result) const;
};

/**
 * @brief Configuration for RandWire block
 */
struct RandWireConfig {
    int64_t num_nodes = 32;
    GraphGenMethod method = GraphGenMethod::WATTS_STROGATZ;
    double p = 0.75;  // For WS model
    double edge_prob = 0.1;  // For ER model
    int64_t m = 5;  // For BA model
    int64_t k = 4;  // For WS model
    int64_t seed = -1;
    bool use_dynamic_routing = true;
};

/**
 * @brief Random wiring block with dynamic connections
 */
class RandWireBlock : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param channels Number of input and output channels
     * @param config Configuration for the block
     */
    RandWireBlock(int64_t channels, const RandWireConfig& config = RandWireConfig());
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Output tensor after graph processing
     */
    torch::Tensor forward(torch::Tensor x);
    
    /**
     * @brief Forward pass with dynamic routing
     * 
     * @param x Input tensor
     * @param use_dynamic Whether to use dynamic routing for this forward pass
     * @return Output tensor after graph processing
     */
    torch::Tensor forward(torch::Tensor x, bool use_dynamic);

private:
    int64_t channels_;
    RandWireConfig config_;
    DiGraph graph_;
    std::vector<int> input_nodes_;
    std::vector<int> output_nodes_;
    std::vector<int> active_nodes_;  // Only nodes that are actually connected
    std::unordered_map<std::string, std::shared_ptr<RouterModule>> routers_;  // Legacy, kept for compatibility
    std::unordered_map<std::string, std::shared_ptr<RouterModule>> adaptive_routers_;  // Dynamic routers created on-demand
    std::unordered_map<std::string, std::shared_ptr<ResidualBlock>> blocks_;
    std::shared_ptr<RouterModule> output_router_{nullptr};
    std::shared_ptr<DynamicRoutingGate> routing_gate_{nullptr};
    
    /**
     * @brief Generate a graph based on the selected method
     * 
     * @return Directed graph
     */
    DiGraph _generate_graph();
    
    /**
     * @brief Generate Watts-Strogatz small-world graph
     */
    DiGraph _generate_ws_graph();
    
    /**
     * @brief Generate Erdos-Renyi random graph
     */
    DiGraph _generate_er_graph();
    
    /**
     * @brief Generate Barabasi-Albert scale-free graph
     */
    DiGraph _generate_ba_graph();
    
    /**
     * @brief Find active nodes (nodes that are actually connected)
     */
    void _find_active_nodes();
};

/**
 * @brief Configuration for DDWRandWireResNet
 */
struct ALPHAZERO_API DDWRandWireResNetConfig {
    int64_t input_channels;
    int64_t output_size;
    int64_t board_height;
    int64_t board_width;
    int64_t channels = 128;
    int64_t num_blocks = 20;
    RandWireConfig randwire_config;
    bool use_dynamic_routing = true;
};

/**
 * @brief Dynamic Dense-Wired Random-Wire ResNet for AlphaZero
 */
class ALPHAZERO_API DDWRandWireResNet : public nn::NeuralNetwork, public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param config Configuration for the network
     */
    DDWRandWireResNet(const DDWRandWireResNetConfig& config);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Tuple of policy and value tensors
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    
    /**
     * @brief Forward pass with dynamic routing control
     * 
     * @param x Input tensor
     * @param use_dynamic Whether to use dynamic routing
     * @return Tuple of policy and value tensors
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, bool use_dynamic);
    
    /**
     * @brief Save the model to a file
     * 
     * @param path File path
     */
    void save(const std::string& path) override;
    
    /**
     * @brief Load the model from a file
     * 
     * @param path File path
     */
    void load(const std::string& path) override;
    
    /**
     * @brief Export the model to TorchScript format
     * 
     * @param path File path
     * @param input_shape Input tensor shape for tracing
     */
    void export_to_torchscript(const std::string& path, 
                              std::vector<int64_t> input_shape = {1, 0, 0, 0});

    // Implement NeuralNetwork interface
    std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override;
    
    std::vector<int64_t> getInputShape() const override;
    
    int64_t getPolicySize() const override;
    
    // Tensor pool cleanup method removed - no longer using tensor pools
    
    /**
     * @brief Set GPU memory pool for efficient tensor allocation
     * @param pool Shared pointer to GPU memory pool
     */
    void setGPUMemoryPool(std::shared_ptr<mcts::GPUMemoryPool> pool) {
        gpu_memory_pool_ = pool;
    }
    
    /**
     * @brief Override to method to track device changes
     */
    void to(torch::Device device, bool non_blocking = false) {
        torch::nn::Module::to(device, non_blocking);
        device_ = device;
    }
    
    void to(torch::ScalarType dtype, bool non_blocking = false) {
        torch::nn::Module::to(dtype, non_blocking);
    }
    
    void to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false) {
        torch::nn::Module::to(device, dtype, non_blocking);
        device_ = device;
    }

private:
    DDWRandWireResNetConfig config_;
    torch::nn::Conv2d input_conv_{nullptr};
    torch::nn::BatchNorm2d input_bn_{nullptr};
    torch::nn::ModuleList rand_wire_blocks_{nullptr};
    
    // Policy head
    torch::nn::Conv2d policy_conv_{nullptr};
    torch::nn::BatchNorm2d policy_bn_{nullptr};
    torch::nn::Linear policy_fc_{nullptr};
    
    // Value head
    torch::nn::Conv2d value_conv_{nullptr};
    torch::nn::BatchNorm2d value_bn_{nullptr};
    torch::nn::Linear value_fc1_{nullptr};
    torch::nn::Linear value_fc2_{nullptr};
    
    /**
     * @brief Initialize weights using Kaiming initialization
     */
    void _initialize_weights();
    
    /**
     * @brief Prepare input tensor for batch inference
     * @param states Vector of game states
     * @return Prepared input tensor
     */
    torch::Tensor prepareInputTensor(
        const std::vector<std::unique_ptr<core::IGameState>>& states);
    
    /**
     * @brief Prepare input tensor with target device
     * @param states Vector of game states
     * @param target_device Target device for the tensor
     * @return Prepared input tensor
     */
    torch::Tensor prepareInputTensor(
        const std::vector<std::unique_ptr<core::IGameState>>& states,
        torch::Device target_device);
    
    // Tensor pool removed - direct allocation is used instead
    
    // GPU memory pool for efficient tensor allocation
    std::shared_ptr<mcts::GPUMemoryPool> gpu_memory_pool_;
    
    // Device for the model
    torch::Device device_ = torch::kCPU;
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H