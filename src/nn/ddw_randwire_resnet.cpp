#include "nn/ddw_randwire_resnet.h"
#include <fstream>
#include <queue>
#include <algorithm>
#include <numeric>

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

// ---------- DynamicRoutingGate Implementation ----------
DynamicRoutingGate::DynamicRoutingGate(int64_t channels, int64_t num_nodes) {
    // Feature extractor to get global context
    feature_extractor = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channels, channels / 4, 1).bias(false));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels / 4));
    
    // Edge scorer to compute routing weights
    int64_t max_edges = num_nodes * num_nodes;  // Upper bound
    edge_scorer = torch::nn::Linear(channels / 4, max_edges);
    
    register_module("feature_extractor", feature_extractor);
    register_module("bn", bn);
    register_module("edge_scorer", edge_scorer);
}

torch::Tensor DynamicRoutingGate::forward(torch::Tensor x, torch::Tensor edge_features) {
    // Extract global features from input
    torch::Tensor feat = torch::relu(bn(feature_extractor(x)));
    
    // Global average pooling
    feat = torch::adaptive_avg_pool2d(feat, {1, 1});
    feat = feat.view({feat.size(0), -1});
    
    // Compute edge scores
    torch::Tensor scores = edge_scorer(feat);
    
    // Apply sigmoid for gating
    return torch::sigmoid(scores);
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
RandWireBlock::RandWireBlock(int64_t channels, const RandWireConfig& config)
    : channels_(channels), config_(config) {
    
    // Generate random graph
    graph_ = _generate_graph();
    
    // Find active nodes
    _find_active_nodes();
    
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
    if (input_nodes_.empty() && !active_nodes_.empty()) {
        input_nodes_.push_back(active_nodes_[0]);
    }
    if (output_nodes_.empty() && !active_nodes_.empty()) {
        output_nodes_.push_back(active_nodes_.back());
    }
    
    // Don't pre-create router modules - we'll use adaptive routing instead
    // This allows flexible input dimensions during forward pass
    
    // Create residual blocks only for active nodes
    for (int node : active_nodes_) {
        auto block = std::make_shared<ResidualBlock>(channels);
        register_module("block_" + std::to_string(node), block);
        blocks_.emplace(std::to_string(node), block);
    }
    
    // Create output router if needed
    if (output_nodes_.size() > 1) {
        output_router_ = std::make_shared<RouterModule>(
            static_cast<int64_t>(output_nodes_.size()) * channels, channels);
        register_module("output_router", output_router_);
    }
    
    // Create dynamic routing gate if enabled
    if (config_.use_dynamic_routing) {
        routing_gate_ = std::make_shared<DynamicRoutingGate>(channels, config_.num_nodes);
        register_module("routing_gate", routing_gate_);
    }
}

DiGraph RandWireBlock::_generate_graph() {
    switch (config_.method) {
        case GraphGenMethod::WATTS_STROGATZ:
            return _generate_ws_graph();
        case GraphGenMethod::ERDOS_RENYI:
            return _generate_er_graph();
        case GraphGenMethod::BARABASI_ALBERT:
            return _generate_ba_graph();
        default:
            return _generate_ws_graph();
    }
}

DiGraph RandWireBlock::_generate_ws_graph() {
    // Set random seed for reproducibility
    std::mt19937 gen;
    if (config_.seed >= 0) {
        gen.seed(static_cast<unsigned int>(config_.seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    int64_t num_nodes = config_.num_nodes;
    int k = static_cast<int>(config_.k);
    double p = config_.p;
    
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
    std::uniform_int_distribution<int> node_dist(0, static_cast<int>(num_nodes - 1));
    
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
            // Rewire edge with a maximum number of attempts
            int w = node_dist(gen);
            int attempts = 0;
            const int max_attempts = 100;
            
            while (attempts < max_attempts && 
                   (w == u || w == v || 
                    (u < w && std::find(DG.successors(u).begin(), DG.successors(u).end(), w) != DG.successors(u).end()) ||
                    (w < u && std::find(DG.successors(w).begin(), DG.successors(w).end(), u) != DG.successors(w).end()))) {
                w = node_dist(gen);
                attempts++;
            }
            
            // Only add edge if we found a valid target
            if (attempts < max_attempts) {
                // Add new edge, ensuring it flows from lower to higher index
                if (u < w) {
                    DG.add_edge(u, w);
                } else {
                    DG.add_edge(w, u);
                }
            } else {
                // If we couldn't find a valid rewiring, keep the original edge
                DG.add_edge(u, v);
            }
        } else {
            // Keep original edge
            DG.add_edge(u, v);
        }
    }
    
    return DG;
}

DiGraph RandWireBlock::_generate_er_graph() {
    std::mt19937 gen;
    if (config_.seed >= 0) {
        gen.seed(static_cast<unsigned int>(config_.seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    DiGraph G;
    int64_t num_nodes = config_.num_nodes;
    double edge_prob = config_.edge_prob;
    
    // Add all nodes
    for (int i = 0; i < num_nodes; i++) {
        G.add_node(i);
    }
    
    // Add edges with probability edge_prob
    for (int i = 0; i < num_nodes; i++) {
        for (int j = i + 1; j < num_nodes; j++) {
            if (dist(gen) < edge_prob) {
                G.add_edge(i, j);
            }
        }
    }
    
    return G;
}

DiGraph RandWireBlock::_generate_ba_graph() {
    std::mt19937 gen;
    if (config_.seed >= 0) {
        gen.seed(static_cast<unsigned int>(config_.seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    DiGraph G;
    int64_t num_nodes = config_.num_nodes;
    int64_t m = config_.m;  // Number of edges to attach from new node
    
    // Start with a complete graph of m+1 nodes
    for (int i = 0; i <= m; i++) {
        G.add_node(i);
        for (int j = 0; j < i; j++) {
            G.add_edge(j, i);
        }
    }
    
    // Add remaining nodes using preferential attachment
    std::vector<int> degrees(num_nodes, 0);
    for (int i = 0; i <= m; i++) {
        degrees[i] = static_cast<int>(m);
    }
    
    for (int i = static_cast<int>(m + 1); i < num_nodes; i++) {
        G.add_node(i);
        
        // Calculate total degree
        int total_degree = std::accumulate(degrees.begin(), degrees.begin() + i, 0);
        
        // Select m nodes to connect to
        std::set<int> targets;
        while (targets.size() < static_cast<size_t>(m)) {
            std::uniform_int_distribution<int> dist(0, total_degree - 1);
            int rand_val = dist(gen);
            
            // Find node by preferential attachment
            int cumsum = 0;
            for (int j = 0; j < i; j++) {
                cumsum += degrees[j];
                if (rand_val < cumsum) {
                    targets.insert(j);
                    break;
                }
            }
        }
        
        // Add edges
        for (int target : targets) {
            G.add_edge(target, i);
            degrees[target]++;
            degrees[i]++;
        }
    }
    
    return G;
}

void RandWireBlock::_find_active_nodes() {
    // Find nodes that are actually connected in the graph
    active_nodes_.clear();
    
    std::vector<int> all_nodes = graph_.nodes();
    for (int node : all_nodes) {
        if (graph_.in_degree(node) > 0 || graph_.out_degree(node) > 0) {
            active_nodes_.push_back(node);
        }
    }
    
    // Sort for consistent ordering
    std::sort(active_nodes_.begin(), active_nodes_.end());
}

torch::Tensor RandWireBlock::forward(torch::Tensor x) {
    return forward(x, config_.use_dynamic_routing);
}

torch::Tensor RandWireBlock::forward(torch::Tensor x, bool use_dynamic) {
    // Node outputs map
    std::unordered_map<int, torch::Tensor> node_outputs;
    
    // Get dynamic routing weights if enabled
    torch::Tensor routing_weights;
    if (use_dynamic && routing_gate_) {
        // Create edge features tensor (placeholder - could be learned)
        torch::Tensor edge_features = torch::ones({x.size(0), 1}, x.options());
        routing_weights = routing_gate_->forward(x, edge_features);
    }
    
    // Process input nodes
    for (int node : input_nodes_) {
        auto it = blocks_.find(std::to_string(node));
        if (it != blocks_.end()) {
            node_outputs[node] = it->second->forward(x);
        }
    }
    
    // Process nodes in topological order
    std::vector<int> topo_order = graph_.topological_sort();
    int edge_idx = 0;
    
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
        
        // Collect and weight inputs
        std::vector<torch::Tensor> inputs;
        for (size_t i = 0; i < predecessors.size(); i++) {
            int pred = predecessors[i];
            // Check if predecessor has output
            if (node_outputs.find(pred) == node_outputs.end()) {
                continue;  // Skip if predecessor hasn't been processed yet
            }
            torch::Tensor input = node_outputs[pred];
            
            // Apply dynamic routing weights if available
            if (use_dynamic && routing_weights.defined() && edge_idx < routing_weights.size(1)) {
                torch::Tensor weight = routing_weights.index({torch::indexing::Slice(), edge_idx})
                    .unsqueeze(1).unsqueeze(2).unsqueeze(3);
                input = input * weight;
                edge_idx++;
            }
            
            inputs.push_back(input);
        }
        
        // Skip this node if no inputs are available yet
        if (inputs.empty()) {
            continue;
        }
        
        torch::Tensor routed;
        if (inputs.size() > 1) {
            // Use adaptive routing - create router on-demand or use pooling
            std::string router_key = "router_" + std::to_string(node) + "_" + std::to_string(inputs.size());
            
            // Check if we have a router for this specific input count
            auto it = adaptive_routers_.find(router_key);
            if (it != adaptive_routers_.end()) {
                torch::Tensor combined = torch::cat(inputs, 1);
                routed = it->second->forward(combined);
            } else {
                // Create router on-demand if not exists
                int64_t input_channels = inputs.size() * channels_;
                auto router = std::make_shared<RouterModule>(input_channels, channels_);
                
                // Move router to same device as inputs
                if (inputs[0].is_cuda()) {
                    router->to(inputs[0].device());
                }
                
                // Register and use the router
                register_module(router_key, router);
                adaptive_routers_[router_key] = router;
                
                torch::Tensor combined = torch::cat(inputs, 1);
                routed = router->forward(combined);
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
            if (node_outputs.find(node) != node_outputs.end()) {
                outputs.push_back(node_outputs[node]);
            }
        }
        if (!outputs.empty()) {
            torch::Tensor combined = torch::cat(outputs, 1);
            return output_router_->forward(combined);
        }
    } else if (!output_nodes_.empty() && node_outputs.find(output_nodes_[0]) != node_outputs.end()) {
        return node_outputs[output_nodes_[0]];
    }
    
    // Fallback: return input
    return x;
}

// ---------- DDWRandWireResNet Implementation ----------
DDWRandWireResNet::DDWRandWireResNet(const DDWRandWireResNetConfig& config)
    : config_(config) {
    // Set PyTorch manual seed if specified to ensure deterministic model creation
    if (config.randwire_config.seed >= 0) {
        torch::manual_seed(config.randwire_config.seed);
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed(config.randwire_config.seed);
            torch::cuda::manual_seed_all(config.randwire_config.seed);
        }
    }
    
    // Input layer
    input_conv_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.input_channels, config.channels, 3)
        .padding(1).bias(false));
    input_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(config.channels));
    
    // Random wire blocks
    rand_wire_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < config.num_blocks; ++i) {
        // Use different seeds for each block
        RandWireConfig block_config = config.randwire_config;
        if (block_config.seed >= 0) {
            block_config.seed = block_config.seed + i;
        }
        rand_wire_blocks_->push_back(
            std::make_shared<RandWireBlock>(config.channels, block_config));
    }
    
    // Calculate policy head input size based on board dimensions
    int64_t policy_spatial_size = config.board_height * config.board_width;
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.channels, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    policy_fc_ = torch::nn::Linear(32 * policy_spatial_size, config.output_size);
    
    // Value head
    value_conv_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.channels, 32, 1).bias(false));
    value_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    value_fc1_ = torch::nn::Linear(32 * policy_spatial_size, 256);
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
    
    // Set device to CPU by default (will be moved by factory if needed)
    device_ = torch::kCPU;
    
    // Initialize tensor pool
    std::vector<int64_t> default_shape = {
        32, config.input_channels, config.board_height, config.board_width
    };
    tensor_pool_.init(default_shape, 4);
}

std::tuple<torch::Tensor, torch::Tensor> DDWRandWireResNet::forward(torch::Tensor x) {
    return forward(x, config_.use_dynamic_routing);
}

std::tuple<torch::Tensor, torch::Tensor> DDWRandWireResNet::forward(
    torch::Tensor x, bool use_dynamic) {
    // Input layer
    x = torch::relu(input_bn_(input_conv_(x)));
    
    // Random wire blocks with dynamic routing
    for (const auto& block : *rand_wire_blocks_) {
        x = block->as<RandWireBlock>()->forward(x, use_dynamic);
    }
    
    // Check current spatial dimensions
    auto sizes = x.sizes();
    int64_t current_height = sizes[2];
    int64_t current_width = sizes[3];
    
    // Apply adaptive pooling if dimensions don't match expected board size
    if (current_height != config_.board_height || current_width != config_.board_width) {
        x = torch::nn::functional::adaptive_avg_pool2d(
            x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions({config_.board_height, config_.board_width}));
    }
    
    // Policy head
    torch::Tensor policy = torch::relu(policy_bn_(policy_conv_(x)));
    policy = policy.view({policy.size(0), -1});
    policy = policy_fc_(policy);
    
    // Value head
    torch::Tensor value = torch::relu(value_bn_(value_conv_(x)));
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

void DDWRandWireResNet::export_to_torchscript(const std::string& path, 
                                             std::vector<int64_t> input_shape) {
    // Set model to evaluation mode
    eval();
    
    // Create dummy input for tracing
    if (input_shape[1] == 0) {
        input_shape[1] = config_.input_channels;
    }
    if (input_shape[2] == 0 || input_shape[3] == 0) {
        input_shape[2] = config_.board_height;
        input_shape[3] = config_.board_width;
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
    
    if (states.empty()) {
        return {};
    }
    
    // Set model to evaluation mode
    eval();
    
    // Prepare input tensor on the same device as the model
    torch::Tensor input_tensor = prepareInputTensor(states, device_);
    
    // Disable gradient computation for inference
    torch::NoGradGuard no_grad;
    
    // Forward pass
    auto [policy_logits, value] = forward(input_tensor);
    
    // Move tensors to CPU for processing
    policy_logits = policy_logits.to(torch::kCPU);
    value = value.to(torch::kCPU);
    
    // Apply softmax to policy
    torch::Tensor policy_probs = torch::softmax(policy_logits, /*dim=*/1);
    
    // Convert to vector of NetworkOutput
    std::vector<mcts::NetworkOutput> outputs;
    outputs.reserve(states.size());
    
    auto policy_accessor = policy_probs.accessor<float, 2>();
    auto value_accessor = value.accessor<float, 2>();
    
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput output;
        
        // Extract policy
        output.policy.resize(config_.output_size);
        for (int64_t j = 0; j < config_.output_size; ++j) {
            output.policy[j] = policy_accessor[i][j];
        }
        
        // Extract value
        output.value = value_accessor[i][0];
        
        outputs.push_back(output);
    }
    
    return outputs;
}

std::vector<int64_t> DDWRandWireResNet::getInputShape() const {
    return {config_.input_channels, config_.board_height, config_.board_width};
}

int64_t DDWRandWireResNet::getPolicySize() const {
    return config_.output_size;
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
    // Use enhanced representation if model expects more than 3 channels
    auto input_shape = (config_.input_channels > 3) ? 
        states[0]->getEnhancedTensorRepresentation() : 
        states[0]->getTensorRepresentation();
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
        // GPUMemoryPool requires dtype, device_id, and optional stream
        int device_id = target_device.index();
        input_tensor = gpu_memory_pool_->allocateTensor(
            tensor_shape, torch::kFloat32, device_id, nullptr);
    } else {
        input_tensor = tensor_pool_.getCPUTensor(tensor_shape);
    }
    
    // Fill tensor with state observations
    auto input_accessor = input_tensor.accessor<float, 4>();
    for (size_t i = 0; i < states.size(); ++i) {
        // Use enhanced representation if model expects more than 3 channels
        auto obs = (config_.input_channels > 3) ? 
            states[i]->getEnhancedTensorRepresentation() : 
            states[i]->getTensorRepresentation();
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

torch::Tensor DDWRandWireResNet::TensorPool::getGPUTensor(
    const std::vector<int64_t>& shape, torch::Device device) {
    if (gpu_tensors.empty()) {
        // Allocate GPU tensors on first use
        for (size_t i = 0; i < pool_size; ++i) {
            gpu_tensors.push_back(
                torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
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