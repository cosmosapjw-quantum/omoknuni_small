#include "mcts/gpu_node.h"
#include "utils/logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>

namespace alphazero {
namespace mcts {

// Static members definition
std::unordered_map<MCTSNode*, std::unique_ptr<GPUNodeMetadata>> GPUNodeManager::gpu_metadata_map_;
std::mutex GPUNodeManager::map_mutex_;

void GPUNodeManager::attachGPUMetadata(std::shared_ptr<MCTSNode> node, 
                                      std::unique_ptr<GPUNodeMetadata> metadata) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    gpu_metadata_map_[node.get()] = std::move(metadata);
}

GPUNodeMetadata* GPUNodeManager::getGPUMetadata(const std::shared_ptr<MCTSNode>& node) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = gpu_metadata_map_.find(node.get());
    if (it != gpu_metadata_map_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool GPUNodeManager::hasGPUMetadata(const std::shared_ptr<MCTSNode>& node) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return gpu_metadata_map_.find(node.get()) != gpu_metadata_map_.end();
}

void GPUNodeManager::assignToGPUBatch(const std::shared_ptr<MCTSNode>& node,
                                     int batch_idx, int node_idx, int tree_idx) {
    auto* metadata = getGPUMetadata(node);
    if (metadata) {
        metadata->tensor_batch_idx = batch_idx;
        metadata->tensor_node_idx = node_idx;
        metadata->tensor_tree_idx = tree_idx;
        metadata->is_in_gpu_batch = true;
    }
}

void GPUNodeManager::syncFromGPU(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    if (!metadata || !metadata->is_in_gpu_batch || !metadata->needs_gpu_sync) {
        return;
    }

    // Sync Q values and visit counts from GPU memory
    if (metadata->gpu_q_values && metadata->gpu_visit_counts) {
        auto& children = node->getChildren();
        int num_children = children.size();
        for (int i = 0; i < num_children; ++i) {
            if (children[i]) {
                // Update child's values from GPU memory
                int visits = metadata->gpu_visit_counts[i];
                float q_value = metadata->gpu_q_values[i];
                if (visits > 0) {
                    // We need to update the node's statistics
                    // This is a simplified approach - in practice we'd need proper synchronization
                    for (int v = 0; v < visits; ++v) {
                        children[i]->update(q_value);
                    }
                }
            }
        }
    }

    metadata->needs_gpu_sync = false;
}

void GPUNodeManager::syncToGPU(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    if (!metadata || !metadata->is_in_gpu_batch) {
        return;
    }

    // Sync Q values and visit counts to GPU memory
    if (metadata->gpu_q_values && metadata->gpu_visit_counts) {
        auto& children = node->getChildren();
        int num_children = children.size();
        for (int i = 0; i < num_children; ++i) {
            if (children[i]) {
                int visits = children[i]->getVisitCount();
                if (visits > 0) {
                    // Calculate Q as average value
                    metadata->gpu_q_values[i] = children[i]->getValue();
                    metadata->gpu_visit_counts[i] = visits;
                } else {
                    metadata->gpu_q_values[i] = 0.0f;
                    metadata->gpu_visit_counts[i] = 0;
                }
            }
        }
    }
}

bool GPUNodeManager::shouldUseGPU(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    if (!metadata) return false;
    
    return metadata->use_gpu_selection && 
           metadata->is_in_gpu_batch &&
           node->getChildren().size() >= gpu_selection_threshold_;
}

MCTSNode* GPUNodeManager::selectBestChildGPU(const std::shared_ptr<MCTSNode>& node,
                                            float exploration_constant, 
                                            bool add_noise) {
    auto* metadata = getGPUMetadata(node);
    if (!metadata) return nullptr;
    
    // Check if we should use GPU selection
    auto& children = node->getChildren();
    int num_children = children.size();
    
    if (metadata->use_gpu_selection && 
        metadata->is_in_gpu_batch && 
        num_children > gpu_selection_threshold_) {
        
        metadata->gpu_selections.fetch_add(1);
        // GPU selection will be handled by GPUTreeStorage batch kernel
        // For now, fall back to CPU selection
        // TODO: Implement GPU batch selection coordination
    }
    
    metadata->cpu_selections.fetch_add(1);
    
    // Use standard CPU selection
    if (num_children == 0) {
        return nullptr;
    }
    
    float best_value = -std::numeric_limits<float>::infinity();
    MCTSNode* best_child = nullptr;
    
    int parent_visits = node->getVisitCount();
    float parent_visits_sqrt = std::sqrt(static_cast<float>(parent_visits));
    
    for (auto& child : children) {
        if (!child) continue;
        
        float q_value = child->getValue();
        int child_visits = child->getVisitCount();
        float prior = child->getPriorProbability();
        
        float exploration_term = exploration_constant * prior * parent_visits_sqrt / (1.0f + child_visits);
        float ucb_value = q_value + exploration_term;
        
        if (add_noise && child_visits == 0) {
            // Add small noise to break ties
            ucb_value += 0.001f * (rand() / static_cast<float>(RAND_MAX));
        }
        
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_child = child.get();
        }
    }
    
    return best_child;
}

int GPUNodeManager::getGPUSelections(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    return metadata ? metadata->gpu_selections.load() : 0;
}

int GPUNodeManager::getCPUSelections(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    return metadata ? metadata->cpu_selections.load() : 0;
}

float GPUNodeManager::getGPUUtilization(const std::shared_ptr<MCTSNode>& node) {
    auto* metadata = getGPUMetadata(node);
    if (!metadata) return 0.0f;
    
    int total = metadata->gpu_selections.load() + metadata->cpu_selections.load();
    if (total == 0) return 0.0f;
    return static_cast<float>(metadata->gpu_selections.load()) / total;
}

}  // namespace mcts
}  // namespace alphazero