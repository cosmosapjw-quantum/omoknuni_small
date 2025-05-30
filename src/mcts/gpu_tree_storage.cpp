#include "mcts/gpu_tree_storage.h"
#include "mcts/gpu_kernels.h"
#include "mcts/half_precision_utils.h"
#include "utils/logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace alphazero {
namespace mcts {

// Forward declarations of CUDA kernels
// TODO: Implement these kernels when GPU tree storage is fully developed
/*
extern "C" {
    void launchTreeTraversalKernel(
        const void* node_metadata,
        const void* edge_data,
        const int* root_indices,
        int* paths,
        int* leaf_indices,
        float c_puct,
        int batch_size,
        int max_nodes,
        int max_depth,
        cudaStream_t stream
    );
    
    void launchCompactTreeKernel(
        void* compact_data,
        const void* sparse_data,
        int num_nodes,
        cudaStream_t stream
    );
}
*/

GPUTreeStorage::GPUTreeStorage(const Config& config) : config_(config), pinned_memory_pool_(nullptr), pinned_memory_size_(0) {
    // Allocate pinned memory for fast CPU-GPU transfers
    if (config.use_unified_memory) {
        // Use CUDA unified memory
        size_t total_size = config.max_trees * config.max_nodes * 
                          (sizeof(CompactNode) + sizeof(CompactEdge) * 8);
        cudaMallocManaged(&pinned_memory_pool_, total_size);
        pinned_memory_size_ = total_size;
    } else {
        // Use regular pinned memory
        pinned_memory_size_ = config.max_trees * config.max_nodes * 64; // Conservative estimate
        cudaMallocHost(&pinned_memory_pool_, pinned_memory_size_);
    }
    
    // Pre-allocate CPU storage
    cpu_nodes_.resize(config.max_trees);
    cpu_edges_.resize(config.max_trees);
    transposition_tables_.resize(config.max_trees);
    
    // LOG_SYSTEM_INFO("GPU Tree Storage initialized:");
    // LOG_SYSTEM_INFO("  - Max nodes: {}", config.max_nodes);
    // LOG_SYSTEM_INFO("  - Max actions: {}", config.max_actions);
    // LOG_SYSTEM_INFO("  - Max trees: {}", config.max_trees);
    // LOG_SYSTEM_INFO("  - Half precision: {}", config.use_half_precision);
    // LOG_SYSTEM_INFO("  - Unified memory: {}", config.use_unified_memory);
    // LOG_SYSTEM_INFO("  - Pinned memory: {:.2f} MB", pinned_memory_size_ / (1024.0 * 1024.0));
}

GPUTreeStorage::~GPUTreeStorage() {
    if (config_.use_unified_memory) {
        cudaFree(pinned_memory_pool_);
    } else {
        cudaFreeHost(pinned_memory_pool_);
    }
}

void GPUTreeStorage::allocateTrees(int batch_size) {
    // Initialize batch tensors
    initializeTensors(batch_size);
    
    // Clear CPU storage
    for (int i = 0; i < batch_size; ++i) {
        cpu_nodes_[i].clear();
        cpu_nodes_[i].reserve(config_.max_nodes / batch_size);
        
        cpu_edges_[i].clear();
        cpu_edges_[i].reserve(config_.max_nodes * 4 / batch_size);
        
        transposition_tables_[i].clear();
    }
}

void GPUTreeStorage::initializeTensors(int batch_size) {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    
    // Node metadata tensor (SoA layout)
    batch_tensors_.node_metadata = torch::zeros(
        {batch_size, static_cast<long>(config_.max_nodes), 4},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    
    // Node hashes
    batch_tensors_.node_hashes = torch::zeros(
        {batch_size, static_cast<long>(config_.max_nodes)},
        torch::TensorOptions().dtype(torch::kInt64).device(device)
    );
    
    // Edge data (using appropriate precision)
    auto edge_dtype = config_.use_half_precision ? torch::kFloat16 : torch::kFloat32;
    size_t max_edges = config_.max_nodes * 8; // Average branching factor estimate
    
    batch_tensors_.edge_actions = torch::zeros(
        {batch_size, static_cast<long>(max_edges)},
        torch::TensorOptions().dtype(torch::kInt16).device(device)
    );
    
    batch_tensors_.edge_visits = torch::zeros(
        {batch_size, static_cast<long>(max_edges)},
        torch::TensorOptions().dtype(torch::kInt16).device(device)
    );
    
    batch_tensors_.edge_priors = torch::zeros(
        {batch_size, static_cast<long>(max_edges)},
        torch::TensorOptions().dtype(edge_dtype).device(device)
    );
    
    batch_tensors_.edge_q_values = torch::zeros(
        {batch_size, static_cast<long>(max_edges)},
        torch::TensorOptions().dtype(edge_dtype).device(device)
    );
    
    // Sparse storage for wide nodes
    size_t sparse_capacity = config_.max_nodes * config_.sparse_threshold / 4;
    
    batch_tensors_.sparse_indices = torch::zeros(
        {batch_size, static_cast<long>(sparse_capacity)},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    
    batch_tensors_.sparse_ptr = torch::zeros(
        {batch_size, static_cast<long>(config_.max_nodes + 1)},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    
    batch_tensors_.sparse_values = torch::zeros(
        {batch_size, static_cast<long>(sparse_capacity)},
        torch::TensorOptions().dtype(edge_dtype).device(device)
    );
    
    // Active node tracking
    batch_tensors_.active_nodes = torch::zeros(
        {batch_size, static_cast<long>(config_.max_nodes)},
        torch::TensorOptions().dtype(torch::kBool).device(device)
    );
    
    batch_tensors_.node_counts = torch::zeros(
        {batch_size},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
}

int32_t GPUTreeStorage::addNode(
    int tree_idx,
    int32_t parent_idx,
    uint64_t state_hash,
    const std::vector<float>& priors) {
    
    // Check transposition table
    auto& tt = transposition_tables_[tree_idx];
    auto it = tt.find(state_hash);
    if (it != tt.end()) {
        return it->second; // Return existing node
    }
    
    // Allocate new node
    auto& nodes = cpu_nodes_[tree_idx];
    auto& edges = cpu_edges_[tree_idx];
    
    int32_t node_idx = nodes.size();
    if (node_idx >= static_cast<int32_t>(config_.max_nodes)) {
        LOG_SYSTEM_WARN("Tree {} reached maximum nodes", tree_idx);
        return -1;
    }
    
    // Create compact node
    CompactNode node;
    node.parent_idx = parent_idx;
    node.first_child_idx = edges.size();
    node.num_children = priors.size();
    node.depth = (parent_idx >= 0 && parent_idx < static_cast<int32_t>(nodes.size())) 
                 ? nodes[parent_idx].depth + 1 : 0;
    node.state_hash = state_hash;
    
    nodes.push_back(node);
    
    // Add edges for children
    bool use_sparse = priors.size() > config_.sparse_threshold;
    
    if (!use_sparse) {
        // Dense storage
        for (size_t i = 0; i < priors.size(); ++i) {
            CompactEdge edge;
            edge.action = i;
            edge.visit_count = 0;
            edge.prior = float_to_half(priors[i]);
            edge.q_value = float_to_half(0.0f);
            edges.push_back(edge);
        }
    } else {
        // Sparse storage - only store high-prior actions initially
        std::vector<std::pair<float, size_t>> prior_actions;
        for (size_t i = 0; i < priors.size(); ++i) {
            if (priors[i] > 0.01f) { // Threshold for sparse storage
                prior_actions.push_back({priors[i], i});
            }
        }
        
        // Sort by prior (descending)
        std::sort(prior_actions.begin(), prior_actions.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Store top actions
        size_t num_to_store = std::min(prior_actions.size(), 
                                      static_cast<size_t>(config_.sparse_threshold));
        
        for (size_t i = 0; i < num_to_store; ++i) {
            CompactEdge edge;
            edge.action = prior_actions[i].second;
            edge.visit_count = 0;
            edge.prior = float_to_half(prior_actions[i].first);
            edge.q_value = float_to_half(0.0f);
            edges.push_back(edge);
        }
        
        node.num_children = num_to_store;
    }
    
    // Update transposition table
    tt[state_hash] = node_idx;
    
    return node_idx;
}

void GPUTreeStorage::updateNode(
    int tree_idx,
    int32_t node_idx,
    int32_t child_action,
    float value,
    int visit_increment) {
    
    auto& nodes = cpu_nodes_[tree_idx];
    auto& edges = cpu_edges_[tree_idx];
    
    if (node_idx < 0 || node_idx >= static_cast<int32_t>(nodes.size())) {
        return;
    }
    
    const auto& node = nodes[node_idx];
    
    // Find edge for child action
    for (int i = 0; i < node.num_children; ++i) {
        int edge_idx = node.first_child_idx + i;
        if (edge_idx < static_cast<int>(edges.size()) && 
            edges[edge_idx].action == child_action) {
            
            // Update edge statistics
            auto& edge = edges[edge_idx];
            
            // Decompress visit count
            int visits = decompressVisitCount(edge.visit_count);
            visits += visit_increment;
            
            // Update Q value incrementally
            float old_q = half_to_float(edge.q_value);
            float new_q = old_q + (value - old_q) / visits;
            
            // Store updated values
            compressVisitCount(edge.visit_count, visits);
            edge.q_value = float_to_half(new_q);
            
            break;
        }
    }
}

void GPUTreeStorage::syncToGPU(cudaStream_t stream) {
    // Prepare data in pinned memory
    uint8_t* pinned_ptr = pinned_memory_pool_;
    size_t offset = 0;
    
    for (int tree_idx = 0; tree_idx < static_cast<int>(cpu_nodes_.size()); ++tree_idx) {
        const auto& nodes = cpu_nodes_[tree_idx];
        const auto& edges = cpu_edges_[tree_idx];
        
        if (nodes.empty()) continue;
        
        // Copy node data to pinned memory
        size_t node_size = nodes.size() * sizeof(CompactNode);
        std::memcpy(pinned_ptr + offset, nodes.data(), node_size);
        
        // Update GPU tensors
        auto metadata_accessor = batch_tensors_.node_metadata.accessor<int32_t, 3>();
        auto hash_accessor = batch_tensors_.node_hashes.accessor<int64_t, 2>();
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            metadata_accessor[tree_idx][i][0] = node.parent_idx;
            metadata_accessor[tree_idx][i][1] = node.first_child_idx;
            metadata_accessor[tree_idx][i][2] = node.num_children;
            metadata_accessor[tree_idx][i][3] = node.depth;
            hash_accessor[tree_idx][i] = node.state_hash;
        }
        
        // Copy edge data
        if (!edges.empty()) {
            auto action_accessor = batch_tensors_.edge_actions.accessor<int16_t, 2>();
            auto visit_accessor = batch_tensors_.edge_visits.accessor<int16_t, 2>();
            
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = edges[i];
                action_accessor[tree_idx][i] = edge.action;
                visit_accessor[tree_idx][i] = edge.visit_count;
                
                // Copy half precision values
                if (config_.use_half_precision) {
                    auto prior_ptr = batch_tensors_.edge_priors.accessor<at::Half, 2>();
                    auto q_ptr = batch_tensors_.edge_q_values.accessor<at::Half, 2>();
                    prior_ptr[tree_idx][i] = at::Half(half_to_float(edge.prior));
                    q_ptr[tree_idx][i] = at::Half(half_to_float(edge.q_value));
                } else {
                    auto prior_ptr = batch_tensors_.edge_priors.accessor<float, 2>();
                    auto q_ptr = batch_tensors_.edge_q_values.accessor<float, 2>();
                    prior_ptr[tree_idx][i] = half_to_float(edge.prior);
                    q_ptr[tree_idx][i] = half_to_float(edge.q_value);
                }
            }
        }
        
        // Update node count
        batch_tensors_.node_counts[tree_idx] = nodes.size();
        batch_tensors_.active_nodes[tree_idx].slice(0, 0, nodes.size()).fill_(true);
        
        offset += node_size;
    }
    
    // Async copy to GPU if stream provided
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

void GPUTreeStorage::syncFromGPU(cudaStream_t stream) {
    // Synchronize stream if provided
    if (stream) {
        cudaStreamSynchronize(stream);
    }
    
    // Copy updated statistics back to CPU
    auto visit_accessor = batch_tensors_.edge_visits.accessor<int16_t, 2>();
    
    for (int tree_idx = 0; tree_idx < static_cast<int>(cpu_edges_.size()); ++tree_idx) {
        auto& edges = cpu_edges_[tree_idx];
        
        for (size_t i = 0; i < edges.size(); ++i) {
            edges[i].visit_count = visit_accessor[tree_idx][i];
            
            if (config_.use_half_precision) {
                auto q_ptr = batch_tensors_.edge_q_values.accessor<at::Half, 2>();
                float q_float = q_ptr[tree_idx][i].operator float();
                edges[i].q_value = float_to_half(q_float);
            } else {
                auto q_ptr = batch_tensors_.edge_q_values.accessor<float, 2>();
                edges[i].q_value = float_to_half(q_ptr[tree_idx][i]);
            }
        }
    }
}

void GPUTreeStorage::compressVisitCount(uint16_t& compressed, int full_count) {
    // Logarithmic compression for large visit counts
    if (full_count < 65536) {
        compressed = full_count;
    } else {
        // Use top bits for exponent, bottom bits for mantissa
        int exp = 0;
        int mantissa = full_count;
        while (mantissa >= 32768) {
            mantissa >>= 1;
            exp++;
        }
        compressed = (exp << 13) | (mantissa & 0x1FFF);
    }
}

int GPUTreeStorage::decompressVisitCount(uint16_t compressed) {
    if (compressed < 32768) {
        return compressed;
    } else {
        int exp = compressed >> 13;
        int mantissa = compressed & 0x1FFF;
        return mantissa << exp;
    }
}

GPUTreeStorage::MemoryStats GPUTreeStorage::getMemoryStats() const {
    MemoryStats stats;
    
    stats.total_allocated_bytes = pinned_memory_size_;
    
    // Count active nodes
    stats.active_nodes = 0;
    stats.sparse_nodes = 0;
    
    for (const auto& nodes : cpu_nodes_) {
        stats.active_nodes += nodes.size();
        for (const auto& node : nodes) {
            if (node.num_children > config_.sparse_threshold) {
                stats.sparse_nodes++;
            }
        }
    }
    
    // Calculate compression ratio
    size_t uncompressed_size = stats.active_nodes * 
        (sizeof(CompactNode) + config_.max_actions * sizeof(CompactEdge));
    size_t compressed_size = stats.total_allocated_bytes;
    
    stats.compression_ratio = static_cast<float>(uncompressed_size) / compressed_size;
    
    // GPU memory usage
    size_t gpu_bytes = 0;
    gpu_bytes += batch_tensors_.node_metadata.numel() * sizeof(int32_t);
    gpu_bytes += batch_tensors_.node_hashes.numel() * sizeof(int64_t);
    gpu_bytes += batch_tensors_.edge_actions.numel() * sizeof(int16_t);
    gpu_bytes += batch_tensors_.edge_visits.numel() * sizeof(int16_t);
    
    if (config_.use_half_precision) {
        gpu_bytes += batch_tensors_.edge_priors.numel() * sizeof(half);
        gpu_bytes += batch_tensors_.edge_q_values.numel() * sizeof(half);
    } else {
        gpu_bytes += batch_tensors_.edge_priors.numel() * sizeof(float);
        gpu_bytes += batch_tensors_.edge_q_values.numel() * sizeof(float);
    }
    
    stats.gpu_memory_usage_mb = gpu_bytes / (1024.0f * 1024.0f);
    
    return stats;
}

void GPUTreeStorage::launchTreeTraversal(
    const torch::Tensor& root_indices,
    torch::Tensor& paths,
    torch::Tensor& leaf_indices,
    float c_puct,
    cudaStream_t stream) {
    
    int batch_size = root_indices.size(0);
    [[maybe_unused]] int max_depth = paths.size(1);
    
    // Allocate temporary buffers for kernel
    int* path_lengths;
    cudaMalloc(&path_lengths, batch_size * sizeof(int));
    
    // Get tensor accessors for kernel call (currently unused, will be used when fully implemented)
    [[maybe_unused]] auto edge_visits = batch_tensors_.edge_visits.accessor<int16_t, 2>();
    [[maybe_unused]] auto edge_q_values = batch_tensors_.edge_q_values.accessor<at::Half, 2>();
    [[maybe_unused]] auto edge_priors = batch_tensors_.edge_priors.accessor<at::Half, 2>();
    [[maybe_unused]] auto node_metadata = batch_tensors_.node_metadata.accessor<int32_t, 3>();
    [[maybe_unused]] auto active_nodes = batch_tensors_.active_nodes.accessor<bool, 2>();
    
    // Extract raw pointers from tensors
    float* Q_values = nullptr;
    int* N_values = nullptr;
    float* P_values = nullptr;
    int* N_total = nullptr;
    int* parent_indices = nullptr;
    int* child_indices = nullptr;
    bool* action_masks = nullptr;
    bool* is_terminal = nullptr;
    int* virtual_loss = nullptr;
    
    // Allocate temporary buffers and convert data
    // Note: In a real implementation, these would be pre-allocated
    size_t total_edges = batch_size * config_.max_nodes * config_.max_actions;
    cudaMalloc(&Q_values, total_edges * sizeof(float));
    cudaMalloc(&N_values, total_edges * sizeof(int));
    cudaMalloc(&P_values, total_edges * sizeof(float));
    cudaMalloc(&N_total, batch_size * config_.max_nodes * sizeof(int));
    cudaMalloc(&parent_indices, batch_size * config_.max_nodes * sizeof(int));
    cudaMalloc(&child_indices, total_edges * sizeof(int));
    cudaMalloc(&action_masks, total_edges * sizeof(bool));
    cudaMalloc(&is_terminal, batch_size * config_.max_nodes * sizeof(bool));
    cudaMalloc(&virtual_loss, batch_size * config_.max_nodes * sizeof(int));
    
    // Initialize arrays to zero
    cudaMemset(Q_values, 0, total_edges * sizeof(float));
    cudaMemset(N_values, 0, total_edges * sizeof(int));
    cudaMemset(P_values, 0, total_edges * sizeof(float));
    cudaMemset(N_total, 0, batch_size * config_.max_nodes * sizeof(int));
    cudaMemset(virtual_loss, 0, batch_size * config_.max_nodes * sizeof(int));
    
    // TODO: Copy actual data from batch_tensors_ to these arrays
    // This is a placeholder implementation
    
    // Call the kernel with appropriate tensor pointers
    launchTreeTraversalKernel(
        Q_values,                   // Q values tensor
        N_values,                   // Visit counts tensor  
        P_values,                   // Prior probabilities
        N_total,                    // Total visits per node
        parent_indices,             // Parent node indices
        child_indices,              // Child node indices
        action_masks,               // Valid action masks
        is_terminal,                // Terminal node flags
        virtual_loss,               // Virtual loss counts
        paths.data_ptr<int>(),      // Output paths
        path_lengths,               // Output path lengths
        nullptr,                    // Leaf values (filled by evaluation)
        c_puct,                     // Exploration constant
        batch_size,                 // Batch size
        config_.max_nodes,          // Max nodes per tree
        config_.max_actions,        // Max actions per node
        1,                          // Single simulation per batch for now
        true,                       // Use warp optimization
        stream                      // CUDA stream
    );
    
    // Cleanup temporary buffers
    cudaFree(Q_values);
    cudaFree(N_values);
    cudaFree(P_values);
    cudaFree(N_total);
    cudaFree(parent_indices);
    cudaFree(child_indices);
    cudaFree(action_masks);
    cudaFree(is_terminal);
    cudaFree(virtual_loss);
    
    // Cleanup
    cudaFree(path_lengths);
}

void GPUTreeStorage::launchBackpropagation(
    const torch::Tensor& paths,
    const torch::Tensor& path_lengths,
    const torch::Tensor& leaf_values,
    cudaStream_t stream) {
    
    int batch_size = paths.size(0);
    int max_depth = paths.size(1);
    
    // Similar to launchTreeTraversal, we need to extract raw pointers
    // In a real implementation, these would be persistent allocations
    size_t total_edges = batch_size * config_.max_nodes * config_.max_actions;
    float* Q_values = nullptr;
    int* N_values = nullptr;
    int* N_total = nullptr;
    int* virtual_loss = nullptr;
    
    cudaMalloc(&Q_values, total_edges * sizeof(float));
    cudaMalloc(&N_values, total_edges * sizeof(int));
    cudaMalloc(&N_total, batch_size * config_.max_nodes * sizeof(int));
    cudaMalloc(&virtual_loss, batch_size * config_.max_nodes * sizeof(int));
    
    // Initialize arrays
    cudaMemset(Q_values, 0, total_edges * sizeof(float));
    cudaMemset(N_values, 0, total_edges * sizeof(int));
    cudaMemset(N_total, 0, batch_size * config_.max_nodes * sizeof(int));
    cudaMemset(virtual_loss, 0, batch_size * config_.max_nodes * sizeof(int));
    
    // TODO: Copy actual data from batch_tensors_ to these arrays
    
    // Call the backpropagation kernel
    launchBackpropagationKernel(
        Q_values,                    // Q values to update
        N_values,                    // Visit counts to update
        N_total,                     // Total visits per node
        virtual_loss,                // Virtual loss to remove
        paths.data_ptr<int>(),        // Selected paths
        path_lengths.data_ptr<int>(), // Path lengths
        leaf_values.data_ptr<float>(),// Leaf evaluation values
        batch_size,                   // Batch size
        config_.max_nodes,            // Max nodes
        config_.max_actions,          // Max actions
        max_depth,                    // Max depth
        stream                        // CUDA stream
    );
    
    // Cleanup
    cudaFree(Q_values);
    cudaFree(N_values);
    cudaFree(N_total);
    cudaFree(virtual_loss);
}

// GPUNodePool implementation

GPUNodePool::GPUNodePool(size_t initial_capacity) 
    : capacity_(initial_capacity), allocated_(0) {
    
    // Allocate GPU memory
    cudaMalloc(&d_memory_, capacity_ * sizeof(GPUTreeStorage::CompactNode));
    
    // Initialize free list
    free_list_.reserve(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        free_list_.push_back(i);
    }
}

GPUNodePool::~GPUNodePool() {
    if (d_memory_) {
        cudaFree(d_memory_);
    }
    
    // LOG_SYSTEM_INFO("GPU Node Pool stats:");
    // LOG_SYSTEM_INFO("  - Total allocations: {}", total_allocations_.load());
    // LOG_SYSTEM_INFO("  - Total deallocations: {}", total_deallocations_.load());
    // LOG_SYSTEM_INFO("  - Final allocated: {}", allocated_);
}

std::vector<int32_t> GPUNodePool::allocateBatch(size_t count) {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    
    std::vector<int32_t> indices;
    indices.reserve(count);
    
    for (size_t i = 0; i < count && !free_list_.empty(); ++i) {
        indices.push_back(free_list_.back());
        free_list_.pop_back();
        allocated_++;
    }
    
    total_allocations_ += indices.size();
    
    return indices;
}

void GPUNodePool::freeBatch(const std::vector<int32_t>& indices) {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    
    for (int32_t idx : indices) {
        free_list_.push_back(idx);
        allocated_--;
    }
    
    total_deallocations_ += indices.size();
}

void GPUNodePool::defragment() {
    // TODO: Implement memory defragmentation
    // This would involve:
    // 1. Sorting allocated nodes by index
    // 2. Compacting them to the beginning of memory
    // 3. Updating all references
    // 4. Rebuilding the free list
}

} // namespace mcts
} // namespace alphazero