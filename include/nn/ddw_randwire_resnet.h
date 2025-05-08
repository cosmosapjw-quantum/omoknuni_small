#ifndef ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H
#define ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <memory>
#include <tuple>

namespace alphazero {
namespace nn {

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
 * @brief Random wiring block with dynamic connections
 */
class RandWireBlock : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param channels Number of input and output channels
     * @param num_nodes Number of nodes in the random graph
     * @param p Rewiring probability (Watts-Strogatz model)
     * @param seed Random seed for reproducibility
     */
    RandWireBlock(int64_t channels, int64_t num_nodes = 32, 
                 double p = 0.75, int64_t seed = -1);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Output tensor after graph processing
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t channels_;
    int64_t num_nodes_;
    DiGraph graph_;
    std::vector<int> input_nodes_;
    std::vector<int> output_nodes_;
    std::unordered_map<std::string, std::shared_ptr<RouterModule>> routers_;
    std::unordered_map<std::string, std::shared_ptr<ResidualBlock>> blocks_;
    std::shared_ptr<RouterModule> output_router_{nullptr};
    
    /**
     * @brief Generate a small-world graph using Watts-Strogatz model
     * 
     * @param num_nodes Number of nodes
     * @param p Rewiring probability
     * @param seed Random seed
     * @return Directed graph
     */
    DiGraph _generate_graph(int64_t num_nodes, double p, int64_t seed);
};

/**
 * @brief Dynamic Dense-Wired Random-Wire ResNet for AlphaZero
 */
class DDWRandWireResNet : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * 
     * @param input_channels Number of input channels
     * @param output_size Size of policy output
     * @param channels Number of channels in the network
     * @param num_blocks Number of random wire blocks
     */
    DDWRandWireResNet(int64_t input_channels, int64_t output_size, 
                      int64_t channels = 128, int64_t num_blocks = 20);
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return Tuple of policy and value tensors
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    
    /**
     * @brief Save the model to a file
     * 
     * @param path File path
     */
    void save(const std::string& path);
    
    /**
     * @brief Load the model from a file
     * 
     * @param path File path
     */
    void load(const std::string& path);
    
    /**
     * @brief Export the model to TorchScript format
     * 
     * @param path File path
     * @param input_shape Input tensor shape for tracing
     */
    void export_to_torchscript(const std::string& path, 
                              std::vector<int64_t> input_shape = {1, 0, 0, 0});

private:
    int64_t input_channels_;
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
};

} // namespace nn
} // namespace alphazero

#endif // ALPHAZERO_NN_DDW_RANDWIRE_RESNET_H