#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace alphazero {
namespace mcts {

/**
 * CUDA kernel to apply virtual loss to paths
 * Uses atomic operations to handle concurrent updates
 */
__global__ void applyVirtualLossKernel(
    const int* __restrict__ paths,      // [batch_size, max_depth, 2]
    int* __restrict__ virtual_loss,     // [batch_size, num_nodes, num_actions]
    const int* __restrict__ path_lengths,
    int batch_size,
    int max_depth,
    int num_nodes,
    int num_actions,
    int virtual_loss_value
) {
    int batch_idx = blockIdx.x;
    int depth_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || depth_idx >= max_depth) return;
    
    // Check if this depth is valid for this path
    if (depth_idx >= path_lengths[batch_idx]) return;
    
    // Get node and action from path
    int path_idx = batch_idx * max_depth * 2 + depth_idx * 2;
    int node_idx = paths[path_idx];
    int action_idx = paths[path_idx + 1];
    
    // Validate indices
    if (node_idx < 0 || node_idx >= num_nodes || 
        action_idx < 0 || action_idx >= num_actions) return;
    
    // Calculate virtual loss tensor index
    int vl_idx = batch_idx * num_nodes * num_actions + 
                 node_idx * num_actions + action_idx;
    
    // Apply virtual loss atomically
    atomicAdd(&virtual_loss[vl_idx], virtual_loss_value);
}

/**
 * Advanced virtual loss kernel with warp-level optimizations
 */
__global__ void applyVirtualLossWarpOptimized(
    const int* __restrict__ paths,
    int* __restrict__ virtual_loss,
    const int* __restrict__ path_lengths,
    const float* __restrict__ adaptive_factors,  // Per-node adaptive factors
    int batch_size,
    int max_depth,
    int num_nodes,
    int num_actions,
    int base_virtual_loss
) {
    const int WARP_SIZE = 32;
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= batch_size) return;
    
    int path_length = path_lengths[warp_id];
    
    // Each lane in the warp handles different depth levels
    for (int d = lane_id; d < path_length; d += WARP_SIZE) {
        int path_idx = warp_id * max_depth * 2 + d * 2;
        int node_idx = paths[path_idx];
        int action_idx = paths[path_idx + 1];
        
        if (node_idx >= 0 && node_idx < num_nodes && 
            action_idx >= 0 && action_idx < num_actions) {
            
            // Calculate adaptive virtual loss
            int vl_value = base_virtual_loss;
            if (adaptive_factors != nullptr && node_idx < num_nodes) {
                vl_value = __float2int_rn(base_virtual_loss * adaptive_factors[node_idx]);
            }
            
            int vl_idx = warp_id * num_nodes * num_actions + 
                        node_idx * num_actions + action_idx;
            
            atomicAdd(&virtual_loss[vl_idx], vl_value);
        }
    }
}

/**
 * CUDA kernel to remove virtual loss from paths
 */
__global__ void removeVirtualLossKernel(
    const int* __restrict__ paths,
    int* __restrict__ virtual_loss,
    const int* __restrict__ path_lengths,
    int batch_size,
    int max_depth,
    int num_nodes,
    int num_actions,
    int virtual_loss_value
) {
    int batch_idx = blockIdx.x;
    int depth_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || depth_idx >= max_depth) return;
    
    if (depth_idx >= path_lengths[batch_idx]) return;
    
    int path_idx = batch_idx * max_depth * 2 + depth_idx * 2;
    int node_idx = paths[path_idx];
    int action_idx = paths[path_idx + 1];
    
    if (node_idx < 0 || node_idx >= num_nodes || 
        action_idx < 0 || action_idx >= num_actions) return;
    
    int vl_idx = batch_idx * num_nodes * num_actions + 
                 node_idx * num_actions + action_idx;
    
    // Remove virtual loss atomically, ensuring non-negative
    int old_val = atomicSub(&virtual_loss[vl_idx], virtual_loss_value);
    
    // If we went negative, correct it
    if (old_val < virtual_loss_value) {
        atomicMax(&virtual_loss[vl_idx], 0);
    }
}

/**
 * Apply virtual loss penalty to UCB scores
 */
__global__ void applyVirtualLossToUCBKernel(
    float* __restrict__ Q_values,
    const int* __restrict__ virtual_loss,
    float virtual_loss_weight,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Subtract weighted virtual loss from Q value
    float vl_penalty = virtual_loss[idx] * virtual_loss_weight;
    Q_values[idx] -= vl_penalty;
}

/**
 * Batch update virtual losses with conflict resolution
 * Uses shared memory to reduce atomic contention
 */
__global__ void batchUpdateVirtualLossWithConflictResolution(
    const int* __restrict__ updates,     // [num_updates, 3] (batch, node, action)
    const int* __restrict__ amounts,     // [num_updates] virtual loss amounts
    int* __restrict__ virtual_loss,      // [batch, nodes, actions]
    int num_updates,
    int num_nodes,
    int num_actions
) {
    extern __shared__ int shared_updates[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int updates_per_block = blockDim.x;
    
    // Each block processes a chunk of updates
    int start_idx = bid * updates_per_block;
    int end_idx = ((start_idx + updates_per_block) < num_updates) ? (start_idx + updates_per_block) : num_updates;
    
    // Load updates into shared memory
    if (start_idx + tid < end_idx) {
        int update_idx = start_idx + tid;
        shared_updates[tid * 4] = updates[update_idx * 3];      // batch
        shared_updates[tid * 4 + 1] = updates[update_idx * 3 + 1]; // node
        shared_updates[tid * 4 + 2] = updates[update_idx * 3 + 2]; // action
        shared_updates[tid * 4 + 3] = amounts[update_idx];      // amount
    }
    __syncthreads();
    
    // Process updates with conflict detection
    for (int i = 0; i < updates_per_block && start_idx + i < end_idx; ++i) {
        if (tid == 0) {
            int batch = shared_updates[i * 4];
            int node = shared_updates[i * 4 + 1];
            int action = shared_updates[i * 4 + 2];
            int amount = shared_updates[i * 4 + 3];
            
            if (batch >= 0 && node >= 0 && action >= 0) {
                int vl_idx = batch * num_nodes * num_actions + 
                            node * num_actions + action;
                atomicAdd(&virtual_loss[vl_idx], amount);
            }
        }
        __syncthreads();
    }
}

// External C interface for calling from C++
extern "C" {

void launchApplyVirtualLossKernelCuda(
    const int* paths,
    int* virtual_loss_tensor,
    const int* path_lengths,
    int batch_size,
    int max_depth,
    int num_nodes,
    int num_actions,
    int virtual_loss_value,
    cudaStream_t stream
) {
    // One block per batch element, threads handle depths
    int threads_per_block = (max_depth < 256) ? max_depth : 256;
    
    applyVirtualLossKernel<<<batch_size, threads_per_block, 0, stream>>>(
        paths, virtual_loss_tensor, path_lengths,
        batch_size, max_depth, num_nodes, num_actions, virtual_loss_value
    );
}

void launchRemoveVirtualLossKernelCuda(
    const int* paths,
    int* virtual_loss_tensor,
    const int* path_lengths,
    int batch_size,
    int max_depth,
    int num_nodes,
    int num_actions,
    cudaStream_t stream
) {
    // One block per batch element, threads handle depths
    int threads_per_block = (max_depth < 256) ? max_depth : 256;
    
    // Use default virtual loss value of 3
    removeVirtualLossKernel<<<batch_size, threads_per_block, 0, stream>>>(
        paths, virtual_loss_tensor, path_lengths,
        batch_size, max_depth, num_nodes, num_actions, 3
    );
}

void launchApplyVirtualLossToUCBKernel(
    float* Q_tensor,
    const int* virtual_loss_tensor,
    float virtual_loss_weight,
    int total_elements,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    applyVirtualLossToUCBKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        Q_tensor, virtual_loss_tensor, virtual_loss_weight, total_elements
    );
}

} // extern "C"

} // namespace mcts
} // namespace alphazero