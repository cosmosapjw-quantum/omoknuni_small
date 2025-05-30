#pragma once

#include "mcts/mcts_node.h"
#include <atomic>
#include <memory>

namespace alphazero {
namespace mcts {

// Forward declaration
using Move = int;

/**
 * GPU metadata for MCTS nodes
 * 
 * This class provides GPU-specific metadata that can be attached to regular MCTSNode instances
 * to enable GPU acceleration without modifying the base node class.
 */
class GPUNodeMetadata {
public:
    // GPU-specific metadata
    int32_t tensor_batch_idx = -1;    // Batch index in GPU tensors
    int32_t tensor_node_idx = -1;     // Node index within batch
    int32_t tensor_tree_idx = -1;     // Tree index for multi-tree batching
    
    // GPU memory pointers for direct access
    float* gpu_q_values = nullptr;    // Direct pointer to Q values in GPU memory
    int* gpu_visit_counts = nullptr;  // Direct pointer to visit counts
    float* gpu_priors = nullptr;      // Direct pointer to prior probabilities
    
    // GPU selection flags
    bool use_gpu_selection = false;   // Whether to use GPU for child selection
    bool is_in_gpu_batch = false;     // Whether this node is in current GPU batch
    bool needs_gpu_sync = false;      // Whether GPU data needs sync to CPU
    
    // Statistics
    std::atomic<int> gpu_selections{0};   // Number of times GPU was used
    std::atomic<int> cpu_selections{0};   // Number of times CPU was used
};

/**
 * GPU node manager that associates GPU metadata with regular MCTS nodes
 * 
 * This approach uses composition instead of inheritance to work with the
 * existing MCTSNode private constructor pattern.
 */
class GPUNodeManager {
public:
    // Associate GPU metadata with a node
    static void attachGPUMetadata(std::shared_ptr<MCTSNode> node, std::unique_ptr<GPUNodeMetadata> metadata);
    
    // Get GPU metadata for a node
    static GPUNodeMetadata* getGPUMetadata(const std::shared_ptr<MCTSNode>& node);
    
    // Check if node has GPU metadata
    static bool hasGPUMetadata(const std::shared_ptr<MCTSNode>& node);
    
    // GPU batch assignment
    static void assignToGPUBatch(const std::shared_ptr<MCTSNode>& node, 
                                 int batch_idx, int node_idx, int tree_idx);
    
    // Synchronization methods
    static void syncFromGPU(const std::shared_ptr<MCTSNode>& node);
    static void syncToGPU(const std::shared_ptr<MCTSNode>& node);
    
    // GPU selection helpers
    static bool shouldUseGPU(const std::shared_ptr<MCTSNode>& node);
    static MCTSNode* selectBestChildGPU(const std::shared_ptr<MCTSNode>& node, 
                                       float exploration_constant, bool add_noise = false);
    
    // Statistics
    static int getGPUSelections(const std::shared_ptr<MCTSNode>& node);
    static int getCPUSelections(const std::shared_ptr<MCTSNode>& node);
    static float getGPUUtilization(const std::shared_ptr<MCTSNode>& node);
    
private:
    // Static map to associate nodes with GPU metadata
    static std::unordered_map<MCTSNode*, std::unique_ptr<GPUNodeMetadata>> gpu_metadata_map_;
    static std::mutex map_mutex_;
    
    static constexpr int gpu_selection_threshold_ = 64;
};

}  // namespace mcts
}  // namespace alphazero