#include "mcts/gpu_enhanced_mcts_engine.h"
#include "mcts/shared_eval_wrapper.h"
#include "utils/logger.h"
#include <chrono>
#include <random>

namespace alphazero {
namespace mcts {

GPUEnhancedMCTSEngine::GPUEnhancedMCTSEngine(
    std::shared_ptr<nn::NeuralNetwork> network,
    const MCTSSettings& settings,
    const GPUConfig& gpu_config)
    : MCTSEngine(network, settings), gpu_config_(gpu_config) {
    
    LOG_MCTS_INFO("Initializing GPU-Enhanced MCTS Engine");
    LOG_MCTS_INFO("  GPU tree storage: {}", gpu_config.use_gpu_tree_storage);
    LOG_MCTS_INFO("  GPU tree traversal: {}", gpu_config.use_gpu_tree_traversal);
    LOG_MCTS_INFO("  CUDA graph: {}", gpu_config.use_cuda_graph);
    LOG_MCTS_INFO("  GPU node selection: {}", gpu_config.use_gpu_node_selection);
    
    initializeGPUComponents();
}

GPUEnhancedMCTSEngine::~GPUEnhancedMCTSEngine() {
    // GPU components cleanup handled by destructors
}

void GPUEnhancedMCTSEngine::initializeGPUComponents() {
    // Initialize GPU batch evaluator
    if (!SharedEvalWrapper::isEnabled()) {
        // Only create our own evaluator if not using shared evaluation
        GPUBatchEvaluator::Config eval_config;
        eval_config.max_batch_size = gpu_config_.gpu_batch_size;
        eval_config.min_batch_size = gpu_config_.gpu_min_batch_size;
        eval_config.batch_timeout_us = static_cast<int64_t>(gpu_config_.gpu_batch_timeout_ms * 1000);
        eval_config.num_cuda_streams = gpu_config_.gpu_cuda_streams;
        
        gpu_evaluator_ = std::make_unique<GPUBatchEvaluator>(
            neural_network_, 
            eval_config
        );
    }
    
    // Initialize GPU tree storage if enabled
    if (gpu_config_.use_gpu_tree_storage) {
        GPUTreeStorage::Config tree_config;
        tree_config.max_nodes = gpu_config_.gpu_tree_max_nodes;
        tree_config.max_actions = gpu_config_.gpu_tree_max_actions;
        tree_config.use_half_precision = gpu_config_.gpu_tree_half_precision;
        // tree_config doesn't have enable_tree_traversal field
        
        gpu_tree_storage_ = std::make_unique<GPUTreeStorage>(tree_config);
    }
    
    // Initialize CUDA graph manager if enabled
    if (gpu_config_.use_cuda_graph && neural_network_) {
        MultiGraphManager::Config graph_config;
        graph_config.batch_sizes = gpu_config_.cuda_graph_batch_sizes;
        graph_config.auto_capture = true;
        
        graph_manager_ = std::make_unique<MultiGraphManager>(graph_config);
        // Neural network doesn't have getModel method - skip CUDA graph for now
        // graph_manager_->initialize(*neural_network_->getModel());
    }
    
    LOG_MCTS_INFO("GPU components initialized successfully");
}

SearchResult GPUEnhancedMCTSEngine::searchGPU(const core::IGameState& state) {
    
    // Create root node
    auto root = MCTSNode::create(state.clone());
    
    // Attach GPU metadata if using GPU features
    if (gpu_config_.use_gpu_node_selection) {
        auto metadata = std::make_unique<GPUNodeMetadata>();
        metadata->use_gpu_selection = true;
        GPUNodeManager::attachGPUMetadata(root, std::move(metadata));
    }
    
    // Run GPU-accelerated simulations
    if (gpu_config_.use_gpu_tree_storage) {
        uploadTreeToGPU(root);
        runSimulationsGPU(root, settings_.num_simulations);
        downloadTreeFromGPU();
    } else {
        // Fall back to standard CPU search
        runSimulations(root, settings_.num_simulations);
    }
    
    // Get search result
    SearchResult result;
    result.action = selectBestMove(root);
    result.probabilities = getPolicyDistribution(root, settings_.temperature);
    result.value = root->getVisitCount() > 0 ? root->getValue() : 0.0f;
    
    // Set basic stats
    result.stats.total_evaluations = settings_.num_simulations;
    
    // Log GPU utilization
    if (gpu_config_.use_gpu_node_selection) {
        float gpu_util = GPUNodeManager::getGPUUtilization(root);
        // LOG_MCTS_DEBUG("GPU node selection utilization: {:.1f}%", gpu_util * 100);
        (void)gpu_util; // Suppress unused variable warning if logging is disabled
    }
    
    return result;
}

// Required virtual methods from base class
void GPUEnhancedMCTSEngine::runSimulations(std::shared_ptr<MCTSNode> root, int num_simulations) {
    if (gpu_config_.use_gpu_tree_storage) {
        runSimulationsGPU(root, num_simulations);
    } else {
        // Run standard CPU simulations
        for (int i = 0; i < num_simulations; ++i) {
            // Standard MCTS simulation
            auto current = root;
            std::vector<std::shared_ptr<MCTSNode>> path;
            
            // Selection
            while (!current->isLeaf()) {
                path.push_back(current);
                std::mt19937 rng(std::random_device{}());
                current = current->selectBestChildUCB(settings_.exploration_constant, rng);
            }
            path.push_back(current);
            
            // Expansion and evaluation
            float value = 0.0f;
            if (!current->isTerminal() && current->getVisitCount() > 0) {
                current->expand();
                if (!current->getChildren().empty()) {
                    current = current->getChildren()[0];
                    path.push_back(current);
                }
            }
            
            if (current->isTerminal()) {
                value = current->getTerminalValue();
            } else {
                value = evaluateNode(current);
            }
            
            // Backpropagation
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                (*it)->update(value);
                value = 1.0f - value;  // Flip for opponent
            }
        }
    }
}

int GPUEnhancedMCTSEngine::selectBestMove(std::shared_ptr<MCTSNode> root) const {
    // Select child with most visits
    const auto& children = root->getChildren();
    if (children.empty()) return -1;
    
    int best_idx = 0;
    int max_visits = 0;
    
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getVisitCount() > max_visits) {
            max_visits = children[i]->getVisitCount();
            best_idx = i;
        }
    }
    
    return root->getActions()[best_idx];
}

std::vector<int> GPUEnhancedMCTSEngine::getChildVisitCounts(std::shared_ptr<MCTSNode> root) const {
    std::vector<int> visit_counts;
    const auto& children = root->getChildren();
    
    visit_counts.reserve(children.size());
    for (const auto& child : children) {
        visit_counts.push_back(child->getVisitCount());
    }
    
    return visit_counts;
}

std::vector<float> GPUEnhancedMCTSEngine::getPolicyDistribution(
    std::shared_ptr<MCTSNode> root, float temperature) const {
    const auto& children = root->getChildren();
    const auto& actions = root->getActions();
    
    // Get max action value to create full policy vector
    int max_action = 0;
    for (int action : actions) {
        max_action = std::max(max_action, action);
    }
    
    std::vector<float> policy(max_action + 1, 0.0f);
    
    if (temperature == 0.0f) {
        // Deterministic: assign 1.0 to best move
        int best_idx = 0;
        int max_visits = 0;
        for (size_t i = 0; i < children.size(); ++i) {
            if (children[i]->getVisitCount() > max_visits) {
                max_visits = children[i]->getVisitCount();
                best_idx = i;
            }
        }
        policy[actions[best_idx]] = 1.0f;
    } else {
        // Apply temperature
        std::vector<float> visits_with_temp;
        float sum = 0.0f;
        
        for (const auto& child : children) {
            float v = std::pow(child->getVisitCount(), 1.0f / temperature);
            visits_with_temp.push_back(v);
            sum += v;
        }
        
        // Normalize
        for (size_t i = 0; i < actions.size(); ++i) {
            policy[actions[i]] = visits_with_temp[i] / sum;
        }
    }
    
    return policy;
}

float GPUEnhancedMCTSEngine::evaluateNode(std::shared_ptr<MCTSNode> node) {
    // Use GPU evaluation if available
    if (gpu_evaluator_) {
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.push_back(node->getState().clone());
        
        auto future = gpu_evaluator_->submitBatch(std::move(states));
        auto outputs = future.get();
        
        if (!outputs.empty()) {
            // Apply policy to node if not yet expanded
            if (!node->isExpanded() && !node->isTerminal()) {
                const auto& legal_actions = node->getState().getLegalMoves();
                std::vector<float> action_probs(legal_actions.size());
                for (size_t i = 0; i < legal_actions.size(); ++i) {
                    action_probs[i] = outputs[0].policy[legal_actions[i]];
                }
                // Note: Node expansion should happen separately with these priors
            }
            return outputs[0].value;
        }
    }
    
    // Fall back to standard evaluation using neural network
    if (neural_network_) {
        auto states = std::vector<std::unique_ptr<core::IGameState>>();
        states.push_back(node->getState().clone());
        auto outputs = neural_network_->inference(states);
        if (!outputs.empty()) {
            // Apply policy to node
            const auto& legal_actions = node->getState().getLegalMoves();
            std::vector<float> action_probs(legal_actions.size());
            for (size_t i = 0; i < legal_actions.size(); ++i) {
                action_probs[i] = outputs[0].policy[legal_actions[i]];
            }
            // Note: Node expansion happens separately
            return outputs[0].value;
        }
    }
    return 0.0f;
}

void GPUEnhancedMCTSEngine::runSimulationsGPU(
    std::shared_ptr<MCTSNode> root, 
    int num_simulations) {
    
    // Initialize GPU tree with root
    gpu_tree_storage_->allocateTrees(1);
    
    // Run simulations using GPU acceleration
    for (int sim = 0; sim < num_simulations; ++sim) {
        // Phase 1: GPU tree traversal to find leaf nodes
        std::vector<std::shared_ptr<MCTSNode>> leaves;
        
        if (gpu_config_.use_gpu_tree_traversal) {
            // Use GPU kernel for parallel traversal
            auto root_indices = torch::tensor({0}, torch::kInt32).cuda();
            auto paths = torch::zeros({1, 256, 2}, torch::kInt32).cuda();
            auto leaf_indices = torch::zeros({1}, torch::kInt32).cuda();
            
            gpu_tree_storage_->launchTreeTraversal(
                root_indices, paths, leaf_indices, 
                settings_.exploration_constant
            );
            
            // Convert GPU results to CPU nodes
            // This is simplified - real implementation would map indices to nodes
            leaves.push_back(root);  // Placeholder
        } else {
            // CPU tree traversal
            auto node = root;
            while (!node->isTerminal() && node->isExpanded()) {
                std::mt19937 rng(std::random_device{}());
                node = node->selectBestChildUCB(settings_.exploration_constant, rng);
            }
            leaves.push_back(node);
        }
        
        // Phase 2: Expand and evaluate leaves
        for (auto& leaf : leaves) {
            if (!leaf->isTerminal() && !leaf->isExpanded()) {
                expandNodeGPU(leaf);
            }
        }
        
        // Phase 3: Backpropagate values
        // In real implementation, would use GPU kernel
        for (auto& leaf : leaves) {
            if (!leaf->isTerminal()) {
                auto value = evaluateNode(leaf);
                leaf->updateRecursive(value);
            }
        }
    }
}

void GPUEnhancedMCTSEngine::expandNodeGPU(std::shared_ptr<MCTSNode> node) {
    // Get legal moves
    auto legal_moves = node->getState().getLegalMoves();
    if (legal_moves.empty()) return;
    
    // Evaluate state to get policy
    std::vector<float> policy;
    float value;  // Used to set in NetworkOutput
    
    if (false) {
        // CUDA graph disabled for now
    } else {
        // Standard evaluation
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.push_back(node->getState().clone());
        auto outputs = neural_network_->inference(states);
        auto output = outputs[0];
        policy = output.policy;
        value = output.value;
    }
    
    // Expand node with policy
    node->expand();
    node->setPriorProbabilities(policy);
    
    // Update GPU tree storage if enabled
    if (gpu_config_.use_gpu_tree_storage) {
        // Get parent index if it exists
        int32_t parent_idx = -1;
        if (node->getParent()) {
            // In real implementation, would need node-to-index mapping
            parent_idx = 0;  // Placeholder
        }
        
        gpu_tree_storage_->addNode(
            0,  // tree_idx
            parent_idx,
            node->getState().getHash(),
            policy
        );
    }
    
    gpu_evaluations_.fetch_add(1);
}

void GPUEnhancedMCTSEngine::backpropagateGPU(
    const std::vector<std::shared_ptr<MCTSNode>>& path, 
    float value) {
    
    if (gpu_config_.use_gpu_tree_traversal && gpu_tree_storage_) {
        // Use GPU kernel for backpropagation
        // This is a placeholder - real implementation would convert path to GPU format
        auto paths = torch::zeros({1, static_cast<int64_t>(path.size()), 2}, torch::kInt32).cuda();
        auto path_lengths = torch::tensor({static_cast<int>(path.size())}, torch::kInt32).cuda();
        auto leaf_values = torch::tensor({value}, torch::kFloat32).cuda();
        
        gpu_tree_storage_->launchBackpropagation(paths, path_lengths, leaf_values);
    } else {
        // CPU backpropagation
        for (auto& node : path) {
            node->update(value);
            value = -value;  // Flip for opponent
        }
    }
}

void GPUEnhancedMCTSEngine::uploadTreeToGPU(std::shared_ptr<MCTSNode> root) {
    if (!gpu_tree_storage_) return;
    
    // Initialize tree storage
    gpu_tree_storage_->allocateTrees(1);
    
    // Add root node - get action size from game state
    int action_size = root->getState().getLegalMoves().size();
    std::vector<float> dummy_policy(action_size, 1.0f / action_size);
    gpu_tree_storage_->addNode(
        0,  // tree_idx
        -1,  // no parent
        root->getState().getHash(),
        dummy_policy
    );
    
    // TODO: Recursively upload entire tree
}

void GPUEnhancedMCTSEngine::downloadTreeFromGPU() {
    if (!gpu_tree_storage_) return;
    
    // Sync GPU memory back to CPU
    gpu_tree_storage_->syncFromGPU();
    
    // TODO: Update CPU nodes with GPU results
}

std::vector<MCTSNode*> GPUEnhancedMCTSEngine::selectBestChildrenGPU(
    const std::vector<std::shared_ptr<MCTSNode>>& nodes,
    float exploration_constant) {
    
    std::vector<MCTSNode*> best_children;
    
    if (gpu_config_.use_gpu_node_selection) {
        // Batch GPU selection
        for (const auto& node : nodes) {
            auto* best = GPUNodeManager::selectBestChildGPU(
                node, exploration_constant, true
            );
            best_children.push_back(best);
            gpu_selections_.fetch_add(1);
        }
    } else {
        // CPU selection
        for (const auto& node : nodes) {
            std::mt19937 rng(std::random_device{}());
            auto best = node->selectBestChildUCB(exploration_constant, rng);
            best_children.push_back(best.get());
            cpu_selections_.fetch_add(1);
        }
    }
    
    return best_children;
}

torch::Tensor GPUEnhancedMCTSEngine::createInputTensor(const core::IGameState& state) {
    // Convert game state to tensor using neural network
    // This uses the neural network's preprocessing
    if (neural_network_) {
        auto clone = state.clone();
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.push_back(std::move(clone));
        
        // Get preprocessed input from neural network
        // Most networks have a method to convert states to tensors
        // For now, we'll create a dummy tensor
        // In real implementation, this would use the network's preprocessing
        int channels = 17;  // Typical for games like Go
        int board_size = 19;  // Default board size
        
        auto tensor = torch::zeros({1, channels, board_size, board_size}, torch::kFloat32);
        return tensor.to(torch::kCUDA);
    }
    
    // Fallback: create dummy tensor
    return torch::zeros({1, 17, 19, 19}, torch::kFloat32).to(torch::kCUDA);
}

GPUEnhancedMCTSEngine::GPUStats GPUEnhancedMCTSEngine::getGPUStats() const {
    GPUStats stats{};
    
    // Tree storage stats
    if (gpu_tree_storage_) {
        auto tree_stats = gpu_tree_storage_->getMemoryStats();
        stats.gpu_tree_nodes = tree_stats.active_nodes;
        stats.gpu_tree_memory_mb = tree_stats.gpu_memory_usage_mb;
        stats.tree_compression_ratio = tree_stats.compression_ratio;
    }
    
    // Evaluation stats
    stats.total_gpu_evaluations = gpu_evaluations_.load();
    
    // Selection stats
    stats.gpu_selections = gpu_selections_.load();
    stats.cpu_selections = cpu_selections_.load();
    if (stats.gpu_selections + stats.cpu_selections > 0) {
        stats.gpu_selection_ratio = static_cast<float>(stats.gpu_selections) / 
            (stats.gpu_selections + stats.cpu_selections);
    }
    
    // CUDA graph stats
    if (graph_manager_) {
        auto graph_stats = graph_manager_->getStats();
        stats.graph_hits = graph_stats.total_graphs;
        stats.graph_speedup = graph_stats.overall_speedup;
    }
    
    return stats;
}

}  // namespace mcts
}  // namespace alphazero