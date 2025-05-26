#include "nn/ddw_randwire_resnet.h"
#include <fstream>
#include <queue>

namespace alphazero {
namespace nn {

// ---------- SEBlock Implementation ----------
SEBlock::SEBlock(int64_t channels, int64_t reduction) {
    squeeze = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
    
    excitation = torch::nn::Sequential(
        torch::nn::Linear(channels, channels / reduction),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(channels / reduction, channels),
        torch::nn::Sigmoid()
    );
    
    register_module("squeeze", squeeze);
    register_module("excitation", excitation);
}

torch::Tensor SEBlock::forward(torch::Tensor x) {
    int64_t b_size = x.size(0);
    int64_t c_size = x.size(1);
    
    torch::Tensor y = squeeze(x).view({b_size, c_size});
    y = excitation->forward(y).view({b_size, c_size, 1, 1});
    
    return x * y;
}

// ---------- ResidualBlock Implementation ----------
ResidualBlock::ResidualBlock(int64_t channels) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1).bias(false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    se = std::make_shared<SEBlock>(channels);
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("se", se);
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    torch::Tensor out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    out = se->forward(out);
    out += residual;
    out = torch::relu(out);
    return out;
}

// ---------- RouterModule Implementation ----------
RouterModule::RouterModule(int64_t in_channels, int64_t out_channels) {
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                             .bias(false));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
    
    register_module("conv", conv);
    register_module("bn", bn);
}

torch::Tensor RouterModule::forward(torch::Tensor x) {
    return torch::relu(bn(conv(x)));
}

// ---------- DiGraph Implementation ----------
void DiGraph::add_node(int node) {
    nodes_.insert(node);
    if (adjacency_list_.find(node) == adjacency_list_.end()) {
        adjacency_list_[node] = std::vector<int>();
    }
    if (reverse_adjacency_list_.find(node) == reverse_adjacency_list_.end()) {
        reverse_adjacency_list_[node] = std::vector<int>();
    }
}

void DiGraph::add_edge(int from, int to) {
    add_node(from);
    add_node(to);
    
    adjacency_list_[from].push_back(to);
    reverse_adjacency_list_[to].push_back(from);
}

std::vector<int> DiGraph::nodes() const {
    std::vector<int> result;
    result.reserve(nodes_.size());
    for (int node : nodes_) {
        result.push_back(node);
    }
    return result;
}

std::vector<int> DiGraph::predecessors(int node) const {
    auto it = reverse_adjacency_list_.find(node);
    if (it != reverse_adjacency_list_.end()) {
        return it->second;
    }
    return std::vector<int>();
}

std::vector<int> DiGraph::successors(int node) const {
    auto it = adjacency_list_.find(node);
    if (it != adjacency_list_.end()) {
        return it->second;
    }
    return std::vector<int>();
}

int DiGraph::in_degree(int node) const {
    auto it = reverse_adjacency_list_.find(node);
    if (it != reverse_adjacency_list_.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

int DiGraph::out_degree(int node) const {
    auto it = adjacency_list_.find(node);
    if (it != adjacency_list_.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

std::vector<Edge> DiGraph::edges() const {
    std::vector<Edge> result;
    for (const auto& [from, to_list] : adjacency_list_) {
        for (int to : to_list) {
            result.push_back({from, to});
        }
    }
    return result;
}

size_t DiGraph::size() const {
    return nodes_.size();
}

bool DiGraph::_dfs_topo_sort(int node, std::unordered_set<int>& visited, 
                           std::unordered_set<int>& temp_visited, 
                           std::vector<int>& result) const {
    if (temp_visited.find(node) != temp_visited.end()) {
        // Cycle detected
        return false;
    }
    
    if (visited.find(node) == visited.end()) {
        temp_visited.insert(node);
        
        auto it = adjacency_list_.find(node);
        if (it != adjacency_list_.end()) {
            for (int successor : it->second) {
                if (!_dfs_topo_sort(successor, visited, temp_visited, result)) {
                    return false;
                }
            }
        }
        
        temp_visited.erase(node);
        visited.insert(node);
        result.push_back(node);
    }
    
    return true;
}

std::vector<int> DiGraph::topological_sort() const {
    std::vector<int> result;
    std::unordered_set<int> visited;
    std::unordered_set<int> temp_visited;
    
    // Run DFS from each unvisited node
    for (int node : nodes_) {
        if (visited.find(node) == visited.end()) {
            if (!_dfs_topo_sort(node, visited, temp_visited, result)) {
                // Cycle detected, return empty vector
                return std::vector<int>();
            }
        }
    }
    
    // Reverse the result for correct order
    std::reverse(result.begin(), result.end());
    return result;
}

// ---------- RandWireBlock Implementation ----------
RandWireBlock::RandWireBlock(int64_t channels, int64_t num_nodes, double p, int64_t seed)
    : channels_(channels), num_nodes_(num_nodes) {
    
    // Generate random graph
    graph_ = _generate_graph(num_nodes, p, seed);
    
    // Find input and output nodes
    for (int node : graph_.nodes()) {
        if (graph_.in_degree(node) == 0) {
            input_nodes_.push_back(node);
        }
        if (graph_.out_degree(node) == 0) {
            output_nodes_.push_back(node);
        }
    }
    
    // Ensure at least one input and output node
    if (input_nodes_.empty()) {
        input_nodes_.push_back(0);
    }
    if (output_nodes_.empty()) {
        output_nodes_.push_back(num_nodes - 1);
    }
    
    // Create router modules and register them
    for (int node : graph_.nodes()) {
        int in_degree = graph_.in_degree(node);
        if (in_degree > 0) {
            auto router = std::make_shared<RouterModule>(in_degree * channels, channels);
            register_module("router_" + std::to_string(node), router);
            routers_.emplace(std::to_string(node), router);
        }
    }
    
    // Create residual blocks and register them
    for (int node : graph_.nodes()) {
        auto block = std::make_shared<ResidualBlock>(channels);
        register_module("block_" + std::to_string(node), block);
        blocks_.emplace(std::to_string(node), block);
    }
    
    // Create output router if needed
    if (output_nodes_.size() > 1) {
        output_router_ = std::make_shared<RouterModule>(output_nodes_.size() * channels, channels);
        register_module("output_router", output_router_);
    }
}

DiGraph RandWireBlock::_generate_graph(int64_t num_nodes, double p, int64_t seed) {
    // Set random seed for reproducibility
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(static_cast<unsigned int>(seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    // Create Watts-Strogatz small-world graph
    int k = 4;  // Each node is connected to k nearest neighbors
    
    // Create ring lattice
    DiGraph G;
    for (int i = 0; i < num_nodes; i++) {
        G.add_node(i);
    }
    
    // Add initial edges to k nearest neighbors
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int target = (i + j) % num_nodes;
            // Make edges directed to avoid cycles
            if (i < target) {
                G.add_edge(i, target);
            } else {
                G.add_edge(target, i);
            }
        }
    }
    
    // Rewire edges with probability p
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    
    // Get all edges
    std::vector<Edge> edges = G.edges();
    
    // Create a new graph for rewiring
    DiGraph DG;
    for (int i = 0; i < num_nodes; i++) {
        DG.add_node(i);
    }
    
    // Rewire each edge with probability p
    for (const Edge& edge : edges) {
        int u = edge.from;
        int v = edge.to;
        
        if (dist(gen) < p) {
            // Rewire edge
            int w = node_dist(gen);
            while (w == u || w == v || (u < w && DG.successors(u).end() != std::find(DG.successors(u).begin(), DG.successors(u).end(), w)) ||
                  (w < u && DG.successors(w).end() != std::find(DG.successors(w).begin(), DG.successors(w).end(), u))) {
                w = node_dist(gen);
            }
            
            // Add new edge, ensuring it flows from lower to higher index
            if (u < w) {
                DG.add_edge(u, w);
            } else {
                DG.add_edge(w, u);
            }
        } else {
            // Keep original edge
            DG.add_edge(u, v);
        }
    }
    
    return DG;
}

torch::Tensor RandWireBlock::forward(torch::Tensor x) {
    // Node outputs map
    std::unordered_map<int, torch::Tensor> node_outputs;
    
    // Process input nodes
    for (int node : input_nodes_) {
        auto it = blocks_.find(std::to_string(node));
        if (it != blocks_.end()) {
            node_outputs[node] = it->second->forward(x);
        }
    }
    
    // Process nodes in topological order
    std::vector<int> topo_order = graph_.topological_sort();
    for (int node : topo_order) {
        // Skip input nodes
        if (std::find(input_nodes_.begin(), input_nodes_.end(), node) != input_nodes_.end()) {
            continue;
        }
        
        // Get inputs from predecessor nodes
        std::vector<int> predecessors = graph_.predecessors(node);
        if (predecessors.empty()) {
            continue;
        }
        
        // Concatenate inputs
        std::vector<torch::Tensor> inputs;
        for (int pred : predecessors) {
            inputs.push_back(node_outputs[pred]);
        }
        
        torch::Tensor routed;
        if (inputs.size() > 1) {
            torch::Tensor combined = torch::cat(inputs, 1);
            auto it = routers_.find(std::to_string(node));
            if (it != routers_.end()) {
                routed = it->second->forward(combined);
            } else {
                routed = combined;
            }
        } else {
            routed = inputs[0];
        }
        
        // Process through residual block
        auto it = blocks_.find(std::to_string(node));
        if (it != blocks_.end()) {
            node_outputs[node] = it->second->forward(routed);
        }
    }
    
    // Combine outputs
    if (output_nodes_.size() > 1) {
        std::vector<torch::Tensor> outputs;
        for (int node : output_nodes_) {
            outputs.push_back(node_outputs[node]);
        }
        torch::Tensor combined = torch::cat(outputs, 1);
        return output_router_->forward(combined);
    } else {
        return node_outputs[output_nodes_[0]];
    }
}

// ---------- DDWRandWireResNet Implementation ----------
DDWRandWireResNet::DDWRandWireResNet(int64_t input_channels, int64_t output_size, 
                                   int64_t channels, int64_t num_blocks)
    : input_channels_(input_channels) {
    
    // Input layer
    input_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, channels, 3)
                                   .padding(1).bias(false));
    input_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    // Random wire blocks
    rand_wire_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < num_blocks; ++i) {
        rand_wire_blocks_->push_back(std::make_shared<RandWireBlock>(channels, 32, 0.75, i));
    }
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    policy_fc_ = torch::nn::Linear(32 * 8 * 8, output_size);
    
    // Value head
    value_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 32, 1).bias(false));
    value_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    value_fc1_ = torch::nn::Linear(32 * 8 * 8, 256);
    value_fc2_ = torch::nn::Linear(256, 1);
    
    // Register modules
    register_module("input_conv", input_conv_);
    register_module("input_bn", input_bn_);
    register_module("rand_wire_blocks", rand_wire_blocks_);
    register_module("policy_conv", policy_conv_);
    register_module("policy_bn", policy_bn_);
    register_module("policy_fc", policy_fc_);
    register_module("value_conv", value_conv_);
    register_module("value_bn", value_bn_);
    register_module("value_fc1", value_fc1_);
    register_module("value_fc2", value_fc2_);
    
    // Initialize weights
    _initialize_weights();
    
    // Set device
    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        this->to(device_);
    }
    
    // Initialize tensor pool
    std::vector<int64_t> default_shape = {32, input_channels, 19, 19}; // Default shape
    tensor_pool_.init(default_shape, 4);
}

std::tuple<torch::Tensor, torch::Tensor> DDWRandWireResNet::forward(torch::Tensor x) {
    // Input layer
    x = torch::relu(input_bn_(input_conv_(x)));
    
    // Random wire blocks
    for (const auto& block : *rand_wire_blocks_) {
        x = block->as<RandWireBlock>()->forward(x);
    }
    
    // Adaptive pooling to handle different board sizes
    auto sizes = x.sizes();
    int64_t height = sizes[2];
    int64_t width = sizes[3];
    
    // Target size of 8x8
    int64_t target_size = 8;
    target_size = std::min(target_size, std::min(height, width));
    
    torch::Tensor x_pooled;
    if (height != target_size || width != target_size) {
        x_pooled = torch::adaptive_avg_pool2d(x, {target_size, target_size});
    } else {
        x_pooled = x;
    }
    
    // Policy head
    torch::Tensor policy = torch::relu(policy_bn_(policy_conv_(x_pooled)));
    policy = policy.view({policy.size(0), -1});
    policy = policy_fc_(policy);
    
    // Value head
    torch::Tensor value = torch::relu(value_bn_(value_conv_(x_pooled)));
    value = value.view({value.size(0), -1});
    value = torch::relu(value_fc1_(value));
    value = torch::tanh(value_fc2_(value));
    
    return {policy, value};
}

void DDWRandWireResNet::_initialize_weights() {
    // He initialization for all layers
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* conv = module->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(
                conv->weight, 0.0, torch::kFanOut, torch::kReLU);
            if (conv->options.bias()) {
                torch::nn::init::constant_(conv->bias, 0.0);
            }
        } else if (auto* bn = module->as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::constant_(bn->weight, 1.0);
            torch::nn::init::constant_(bn->bias, 0.0);
        } else if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::kaiming_normal_(
                linear->weight, 0.0, torch::kFanOut, torch::kReLU);
            torch::nn::init::constant_(linear->bias, 0.0);
        }
    }
}

void DDWRandWireResNet::save(const std::string& path) {
    auto self = shared_from_this();
    torch::save(self, path);
}

void DDWRandWireResNet::load(const std::string& path) {
    auto self = shared_from_this();
    torch::load(self, path);
}

void DDWRandWireResNet::export_to_torchscript(const std::string& path, std::vector<int64_t> input_shape) {
    // Set model to evaluation mode
    eval();
    
    // Create dummy input for tracing
    if (input_shape[1] == 0) {
        input_shape[1] = input_channels_;
    }
    if (input_shape[2] == 0 || input_shape[3] == 0) {
        input_shape[2] = input_shape[3] = 8;  // Default board size
    }
    
    torch::Tensor dummy_input = torch::zeros(input_shape);
    
    // Save model
    try {
        // Try direct save first
        torch::jit::script::Module traced_module;
        traced_module.save(path);
    } catch (const c10::Error&) {
        // Fall back to alternative save method
        auto model_copy = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
        if (model_copy) {
            torch::save(model_copy, path);
        }
    }
}

// Implementations for NeuralNetwork interface
std::vector<mcts::NetworkOutput> DDWRandWireResNet::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    // TODO: Implement actual batch inference logic for DDWRandWireResNet
    // This will involve:
    // 1. Preparing input tensors from game states (similar to ResNetModel::prepareInputTensor)
    // 2. Running the forward pass of the DDWRandWireResNet model
    // 3. Converting output tensors to std::vector<mcts::NetworkOutput>
    
    if (states.empty()) {
        return {};
    }
    
    // Placeholder: throw error or return dummy data
    throw std::runtime_error("DDWRandWireResNet::inference() not yet implemented.");
    // Example of returning dummy data (replace with actual implementation):
    /*
    std::vector<mcts::NetworkOutput> outputs;
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput out;
        out.policy.assign(getPolicySize(), 1.0f / getPolicySize()); // Uniform policy
        out.value = 0.0f; // Zero value
        outputs.push_back(out);
    }
    return outputs;
    */
}

std::vector<int64_t> DDWRandWireResNet::getInputShape() const {
    // TODO: Return the actual input shape your DDWRandWireResNet expects.
    // Example: {input_channels_, board_height, board_width}
    // For now, returning a placeholder. You'll need to store board dimensions
    // or have a way to determine them if they are dynamic.
    // Assuming a fixed board size for now, e.g., 8x8, like in policy_fc_ layer calculation
    // This needs to be consistent with how you'd prepare input tensors in inference().
    if (input_channels_ == 0) { // Should be set in constructor
        throw std::logic_error("DDWRandWireResNet input_channels_ is not set.");
    }
     // Placeholder - assuming policy head uses 8x8 feature maps before FC layer.
     // This is a guess and needs to be confirmed from your model architecture.
     // A more robust way is to store board_height and board_width if they are fixed,
     // or define a convention if they can vary. For AlphaZero, they are usually fixed per game.
    return {input_channels_, 8, 8}; 
}

int64_t DDWRandWireResNet::getPolicySize() const {
    // TODO: Return the actual policy size.
    // This should match the output_size parameter passed to the constructor and used by policy_fc_.
    // Assuming policy_fc_ is torch::nn::Linear(32 * 8 * 8, output_size);
    // The 'output_size' here is the policy size.
    // You'll need to store this output_size in the class if it's not already.
    
    // Let's assume there's a member variable output_size_ that stores this.
    // For now, as a placeholder, let's try to deduce from policy_fc_ if possible, or throw.
    if (!policy_fc_) {
        throw std::logic_error("DDWRandWireResNet policy_fc_ is null, cannot determine policy size.");
    }
    // This gets the out_features of the Linear layer
    return policy_fc_->options.out_features(); 
}

void DDWRandWireResNet::cleanupTensorPool() {
    tensor_pool_.cleanup();
}

torch::Tensor DDWRandWireResNet::prepareInputTensor(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    return prepareInputTensor(states, device_);
}

torch::Tensor DDWRandWireResNet::prepareInputTensor(
    const std::vector<std::unique_ptr<core::IGameState>>& states,
    torch::Device target_device) {
    
    if (states.empty()) {
        throw std::invalid_argument("No states provided for inference");
    }
    
    // Get input shape from first state
    auto input_shape = states[0]->getTensorRepresentation();
    int channels = input_shape.size();
    int height = 0, width = 0;
    
    if (channels > 0 && !input_shape[0].empty()) {
        height = input_shape[0].size();
        width = input_shape[0][0].size();
    }
    
    std::vector<int64_t> tensor_shape = {
        static_cast<int64_t>(states.size()),
        static_cast<int64_t>(channels),
        static_cast<int64_t>(height),
        static_cast<int64_t>(width)
    };
    
    // Use tensor pool if available
    torch::Tensor input_tensor;
    if (target_device.is_cuda() && gpu_memory_pool_) {
        input_tensor = tensor_pool_.getGPUTensor(tensor_shape, target_device);
    } else {
        input_tensor = tensor_pool_.getCPUTensor(tensor_shape);
    }
    
    // Fill tensor with state observations
    auto input_accessor = input_tensor.accessor<float, 4>();
    for (size_t i = 0; i < states.size(); ++i) {
        auto obs = states[i]->getTensorRepresentation();
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    input_accessor[i][c][h][w] = obs[c][h][w];
                }
            }
        }
    }
    
    return input_tensor.to(target_device);
}

// TensorPool implementation
void DDWRandWireResNet::TensorPool::init(const std::vector<int64_t>& shape, size_t size) {
    pool_size = size;
    cpu_tensors.clear();
    gpu_tensors.clear();
    
    // Pre-allocate CPU tensors
    for (size_t i = 0; i < pool_size; ++i) {
        cpu_tensors.push_back(torch::empty(shape, torch::kFloat32));
    }
    
    // GPU tensors will be allocated on demand
    current_cpu_idx = 0;
    current_gpu_idx = 0;
}

torch::Tensor DDWRandWireResNet::TensorPool::getCPUTensor(const std::vector<int64_t>& shape) {
    if (cpu_tensors.empty() || current_cpu_idx >= cpu_tensors.size()) {
        return torch::empty(shape, torch::kFloat32);
    }
    
    auto& tensor = cpu_tensors[current_cpu_idx];
    current_cpu_idx = (current_cpu_idx + 1) % pool_size;
    
    // Resize if necessary
    if (tensor.sizes() != shape) {
        tensor.resize_(shape);
    }
    
    return tensor;
}

torch::Tensor DDWRandWireResNet::TensorPool::getGPUTensor(const std::vector<int64_t>& shape, torch::Device device) {
    if (gpu_tensors.empty()) {
        // Allocate GPU tensors on first use
        for (size_t i = 0; i < pool_size; ++i) {
            gpu_tensors.push_back(torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        }
    }
    
    if (current_gpu_idx >= gpu_tensors.size()) {
        return torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    
    auto& tensor = gpu_tensors[current_gpu_idx];
    current_gpu_idx = (current_gpu_idx + 1) % pool_size;
    
    // Resize if necessary
    if (tensor.sizes() != shape) {
        tensor.resize_(shape);
    }
    
    return tensor;
}

void DDWRandWireResNet::TensorPool::cleanup() {
    cpu_tensors.clear();
    gpu_tensors.clear();
    current_cpu_idx = 0;
    current_gpu_idx = 0;
}

} // namespace nn
} // namespace alphazero