#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace alphazero {
namespace mcts {

// Constants for CUDA kernels
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 256;

/**
 * Sparse UCB Kernel for Large Branching Factors
 * Only computes UCB for valid actions indicated by mask
 */
__global__ void computeSparseUCBKernel(
    const float* __restrict__ Q,           // [batch, nodes, actions]
    const int* __restrict__ N,             // [batch, nodes, actions] 
    const float* __restrict__ P,           // [batch, nodes, actions]
    const int* __restrict__ N_total,       // [batch, nodes]
    const bool* __restrict__ action_mask,  // [batch, nodes, actions]
    float* __restrict__ UCB,               // [batch, nodes, actions]
    int* __restrict__ topk_indices,        // [batch, nodes, k]
    float* __restrict__ topk_values,       // [batch, nodes, k]
    float c_puct,
    int batch_size,
    int num_nodes,
    int num_actions,
    int topk
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_nodes * num_actions);
    int node_idx = (tid / num_actions) % num_nodes;
    int action_idx = tid % num_actions;
    
    if (batch_idx >= batch_size || node_idx >= num_nodes || action_idx >= num_actions) return;
    
    int global_idx = batch_idx * num_nodes * num_actions + node_idx * num_actions + action_idx;
    
    // Only compute UCB for valid actions
    if (!action_mask[global_idx]) {
        UCB[global_idx] = -INFINITY;
        return;
    }
    
    float q = Q[global_idx];
    int n = N[global_idx];
    float p = P[global_idx];
    int n_total = N_total[batch_idx * num_nodes + node_idx];
    
    float exploration = c_puct * p * sqrtf((float)n_total) / (1.0f + n);
    UCB[global_idx] = q + exploration;
}

/**
 * UCB Calculation Kernel
 * Computes UCB scores for all actions in all nodes
 * UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)
 */
__global__ void computeUCBKernel(
    const float* __restrict__ Q,      // [batch, nodes, actions]
    const int* __restrict__ N,        // [batch, nodes, actions]
    const float* __restrict__ P,      // [batch, nodes, actions]
    const int* __restrict__ N_total,  // [batch, nodes]
    float* __restrict__ UCB,          // [batch, nodes, actions]
    float c_puct,
    int batch_size,
    int num_nodes,
    int num_actions
) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_nodes * num_actions;
    
    if (tid >= total_elements) return;
    
    // Calculate indices
    int idx = tid;
    int node_idx = tid / num_actions;  // Node index for N_total lookup
    
    // Get values
    float q_value = Q[idx];
    int n_value = N[idx];
    float p_value = P[idx];
    int n_parent = N_total[node_idx];
    
    // Compute UCB
    float exploration = c_puct * p_value * sqrtf(static_cast<float>(n_parent)) / (1.0f + n_value);
    UCB[idx] = q_value + exploration;
}

/**
 * Sparse UCB Calculation Kernel for large branching factors
 * Only computes UCB for visited actions and high-prior unvisited actions
 */
__global__ void computeUCBSparseKernel(
    const float* __restrict__ Q_sparse,    // Sparse Q values
    const int* __restrict__ N_sparse,      // Sparse N values
    const int* __restrict__ indices,       // Action indices
    const int* __restrict__ ptr,           // CSR row pointers
    const float* __restrict__ P_dense,     // Dense prior probabilities
    const int* __restrict__ N_total,       // Total visits per node
    float* __restrict__ UCB,               // Output UCB scores
    float c_puct,
    float prior_threshold,
    int batch_size,
    int num_nodes,
    int num_actions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = tid / num_actions;
    int action = tid % num_actions;
    
    if (node_idx >= batch_size * num_nodes) return;
    
    // Removed unused batch and node variables
    
    // Get sparse range for this node
    int start = ptr[node_idx];
    int end = ptr[node_idx + 1];
    
    // Binary search for action in sparse data
    int sparse_idx = -1;
    int left = start;
    int right = end - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (indices[mid] == action) {
            sparse_idx = mid;
            break;
        } else if (indices[mid] < action) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    float ucb_value;
    int n_parent = N_total[node_idx];
    float sqrt_n_parent = sqrtf(static_cast<float>(n_parent));
    
    if (sparse_idx >= 0) {
        // Visited action - use stored Q and N
        float q = Q_sparse[sparse_idx];
        int n = N_sparse[sparse_idx];
        float p = P_dense[tid];
        ucb_value = q + c_puct * p * sqrt_n_parent / (1.0f + n);
    } else {
        // Unvisited action - check if prior is high enough
        float p = P_dense[tid];
        if (p > prior_threshold) {
            ucb_value = c_puct * p * sqrt_n_parent;
        } else {
            ucb_value = -INFINITY;  // Will not be selected
        }
    }
    
    UCB[tid] = ucb_value;
}

/**
 * Warp-level reduction for finding maximum value and index
 */
__device__ void warpReduceMaxWithIndex(float& val, int& idx) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

/**
 * Path Selection Kernel
 * Selects best actions based on UCB scores with virtual loss
 */
__global__ void selectPathsKernel(
    const float* __restrict__ UCB,          // [batch, nodes, actions]
    const bool* __restrict__ children_mask, // [batch, nodes, actions]
    int* __restrict__ selected_paths,       // [batch, max_depth, 2]
    int* __restrict__ virtual_loss,         // [batch, nodes, actions]
    int batch_size,
    int num_nodes,
    int num_actions,
    int max_depth,
    int virtual_loss_value = 3
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ char shared_mem[];
    float* shared_ucb = (float*)shared_mem;
    int* shared_idx = (int*)&shared_ucb[blockDim.x];
    
    // Start from root node
    int current_node = 0;
    int depth = 0;
    
    while (depth < max_depth && current_node >= 0) {
        // Each thread handles different actions
        int tid = threadIdx.x;
        
        float max_ucb = -INFINITY;
        int best_action = -1;
        
        // Process actions in chunks
        for (int action = tid; action < num_actions; action += blockDim.x) {
            int idx = batch_idx * num_nodes * num_actions + 
                     current_node * num_actions + action;
            
            if (children_mask[idx]) {
                float ucb = UCB[idx];
                if (ucb > max_ucb) {
                    max_ucb = ucb;
                    best_action = action;
                }
            }
        }
        
        // Store in shared memory
        shared_ucb[tid] = max_ucb;
        shared_idx[tid] = best_action;
        __syncthreads();
        
        // Block-level reduction
        for (int s = blockDim.x / 2; s > 0; s /= 2) {
            if (tid < s) {
                if (shared_ucb[tid + s] > shared_ucb[tid]) {
                    shared_ucb[tid] = shared_ucb[tid + s];
                    shared_idx[tid] = shared_idx[tid + s];
                }
            }
            __syncthreads();
        }
        
        // Thread 0 writes result and applies virtual loss
        if (tid == 0) {
            if (shared_idx[0] >= 0) {
                // Store selected path
                selected_paths[batch_idx * max_depth * 2 + depth * 2] = current_node;
                selected_paths[batch_idx * max_depth * 2 + depth * 2 + 1] = shared_idx[0];
                
                // Apply virtual loss
                int vl_idx = batch_idx * num_nodes * num_actions + 
                            current_node * num_actions + shared_idx[0];
                atomicAdd(&virtual_loss[vl_idx], virtual_loss_value);
                
                // Move to next node (simplified - actual implementation needs child lookup)
                // For now, assume linear indexing
                current_node = current_node * num_actions + shared_idx[0] + 1;
                if (current_node >= num_nodes) {
                    current_node = -1;  // Leaf node
                }
            } else {
                current_node = -1;  // No valid action
            }
        }
        
        // Ensure thread 0 has finished updating current_node before broadcast
        __syncthreads();
        
        // Broadcast next node to all threads
        current_node = __shfl_sync(0xFFFFFFFF, current_node, 0);
        depth++;
        __syncthreads();
    }
}

/**
 * Batch Backup Kernel
 * Updates W (cumulative values) and N (visit counts) along selected paths
 */
__global__ void batchBackupKernel(
    float* __restrict__ W,              // [batch, nodes, actions]
    int* __restrict__ N,                // [batch, nodes, actions]
    int* __restrict__ virtual_loss,     // [batch, nodes, actions]
    const float* __restrict__ values,   // [batch]
    const int* __restrict__ paths,      // [batch, max_depth, 2]
    int batch_size,
    int num_nodes,
    int num_actions,
    int max_depth
) {
    int batch_idx = blockIdx.x;
    int depth_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || depth_idx >= max_depth) return;
    
    // Get path for this depth
    int node = paths[batch_idx * max_depth * 2 + depth_idx * 2];
    int action = paths[batch_idx * max_depth * 2 + depth_idx * 2 + 1];
    
    if (node < 0 || action < 0) return;  // Invalid path element
    
    // Calculate index
    int idx = batch_idx * num_nodes * num_actions + 
              node * num_actions + action;
    
    // Get value (alternating signs for two-player games)
    float value = values[batch_idx];
    if (depth_idx % 2 == 1) value = -value;
    
    // Update statistics atomically
    atomicAdd(&W[idx], value);
    atomicAdd(&N[idx], 1);
    
    // Remove virtual loss
    atomicAdd(&virtual_loss[idx], -3);  // Assuming virtual loss of 3
}

/**
 * Advanced backup kernel with warp-level optimizations
 */
__global__ void advancedBackupKernel(
    float* __restrict__ W,
    int* __restrict__ N,
    const float* __restrict__ values,
    const int* __restrict__ paths,
    const int* __restrict__ path_lengths,
    int batch_size,
    int num_nodes,
    int num_actions,
    int max_depth
) {
    // Use warps to handle different paths
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= batch_size) return;
    
    float value = values[warp_id];
    int path_len = path_lengths[warp_id];
    
    // Each lane handles different depth levels
    for (int d = lane_id; d < path_len; d += WARP_SIZE) {
        int node = paths[warp_id * max_depth * 2 + d * 2];
        int action = paths[warp_id * max_depth * 2 + d * 2 + 1];
        
        if (node >= 0 && action >= 0) {
            int idx = warp_id * num_nodes * num_actions + 
                     node * num_actions + action;
            
            // Alternate value sign for two-player games
            float update_value = (d % 2 == 0) ? value : -value;
            
            // Atomic updates
            atomicAdd(&W[idx], update_value);
            atomicAdd(&N[idx], 1);
        }
    }
}

// Kernel launcher functions (called from C++)
extern "C" {

void launchUCBKernel(
    const float* Q,
    const int* N,
    const float* P,
    const int* N_total,
    float* UCB,
    float c_puct,
    int batch_size,
    int num_nodes,
    int num_actions,
    cudaStream_t stream
) {
    int total_elements = batch_size * num_nodes * num_actions;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    computeUCBKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        Q, N, P, N_total, UCB, c_puct,
        batch_size, num_nodes, num_actions
    );
}

void launchPathSelectionKernel(
    const float* UCB,
    const bool* children_mask,
    int* selected_paths,
    int* virtual_loss,
    int batch_size,
    int num_nodes,
    int num_actions,
    int max_depth,
    cudaStream_t stream
) {
    // One block per batch element
    int threads_per_block = (num_actions < MAX_THREADS_PER_BLOCK) ? num_actions : MAX_THREADS_PER_BLOCK;
    size_t shared_mem_size = threads_per_block * (sizeof(float) + sizeof(int));
    
    selectPathsKernel<<<batch_size, threads_per_block, shared_mem_size, stream>>>(
        UCB, children_mask, selected_paths, virtual_loss,
        batch_size, num_nodes, num_actions, max_depth
    );
}

void launchBackupKernel(
    float* W,
    int* N,
    int* virtual_loss,
    const float* values,
    const int* paths,
    int batch_size,
    int num_nodes,
    int num_actions,
    int max_depth,
    cudaStream_t stream
) {
    // One block per batch, threads handle depths
    int threads_per_block = (max_depth < MAX_THREADS_PER_BLOCK) ? max_depth : MAX_THREADS_PER_BLOCK;
    
    batchBackupKernel<<<batch_size, threads_per_block, 0, stream>>>(
        W, N, virtual_loss, values, paths,
        batch_size, num_nodes, num_actions, max_depth
    );
}

} // extern "C"

} // namespace mcts
} // namespace alphazero