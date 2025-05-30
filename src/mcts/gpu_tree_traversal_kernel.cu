#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace alphazero {
namespace mcts {

// Constants for tree traversal
constexpr int WARP_SIZE = 32;
constexpr int MAX_DEPTH = 256;
// constexpr int MAX_CHILDREN = 362;  // Unused - max actions passed as parameter

// Kernel for parallel tree traversal with virtual loss
__global__ void treeTraversalKernel(
    const float* __restrict__ Q_values,        // [batch_size, max_nodes, max_actions]
    const int* __restrict__ N_values,          // [batch_size, max_nodes, max_actions]
    const float* __restrict__ P_values,        // [batch_size, max_nodes, max_actions]
    const int* __restrict__ N_total,           // [batch_size, max_nodes]
    const int* __restrict__ parent_indices,    // [batch_size, max_nodes]
    const int* __restrict__ child_indices,     // [batch_size, max_nodes, max_actions]
    const bool* __restrict__ action_mask,      // [batch_size, max_nodes, max_actions]
    const bool* __restrict__ is_terminal,      // [batch_size, max_nodes]
    int* __restrict__ virtual_loss,            // [batch_size, max_nodes, max_actions]
    int* __restrict__ selected_paths,          // [batch_size, max_depth, 2] (node_idx, action_idx)
    int* __restrict__ path_lengths,            // [batch_size]
    float* __restrict__ leaf_values,           // [batch_size] - output leaf evaluations
    const float c_puct,
    const int batch_size,
    const int max_nodes,
    const int max_actions,
    const int max_simulations) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = tid / max_simulations;
    const int sim_idx = tid % max_simulations;
    
    if (batch_idx >= batch_size) return;
    
    // Thread-local path storage
    int local_path[MAX_DEPTH][2];  // (node_idx, action_idx)
    int local_path_length = 0;
    
    // Start from root
    int current_node = 0;
    bool found_leaf = false;
    
    // Traverse down the tree
    for (int depth = 0; depth < MAX_DEPTH && !found_leaf; ++depth) {
        // Check if current node is terminal
        if (is_terminal[batch_idx * max_nodes + current_node]) {
            found_leaf = true;
            break;
        }
        
        // Select best child using PUCT
        float best_ucb = -1e10f;
        int best_action = -1;
        int best_child = -1;
        
        // Get total visits for current node
        int n_total = N_total[batch_idx * max_nodes + current_node];
        float sqrt_n_total = sqrtf(static_cast<float>(n_total + 1));
        
        // Evaluate all actions
        for (int action = 0; action < max_actions; ++action) {
            int idx = (batch_idx * max_nodes + current_node) * max_actions + action;
            
            // Skip invalid actions
            if (!action_mask[idx]) continue;
            
            // Get child node
            int child_node = child_indices[idx];
            if (child_node < 0) {
                // Unexpanded child - this is a leaf
                best_action = action;
                best_child = -1;
                found_leaf = true;
                break;
            }
            
            // Calculate UCB with virtual loss
            int n = N_values[idx];
            int vl = virtual_loss[idx];
            int n_with_vl = n + vl;
            
            float q = (n_with_vl > 0) ? Q_values[idx] : 0.0f;
            float p = P_values[idx];
            
            float ucb = q + c_puct * p * sqrt_n_total / (1.0f + n_with_vl);
            
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_action = action;
                best_child = child_node;
            }
        }
        
        // Store selected action in path
        if (best_action >= 0) {
            local_path[local_path_length][0] = current_node;
            local_path[local_path_length][1] = best_action;
            local_path_length++;
            
            // Apply virtual loss
            int vl_idx = (batch_idx * max_nodes + current_node) * max_actions + best_action;
            atomicAdd(&virtual_loss[vl_idx], 1);
            
            // Move to child node
            if (best_child >= 0) {
                current_node = best_child;
            } else {
                found_leaf = true;
            }
        } else {
            // No valid actions - terminal node
            found_leaf = true;
        }
    }
    
    // Store the selected path (only first simulation per batch writes)
    if (sim_idx == 0 && local_path_length > 0) {
        path_lengths[batch_idx] = local_path_length;
        for (int i = 0; i < local_path_length; ++i) {
            int path_idx = batch_idx * MAX_DEPTH * 2 + i * 2;
            selected_paths[path_idx] = local_path[i][0];
            selected_paths[path_idx + 1] = local_path[i][1];
        }
    }
}

// Kernel for backpropagation after evaluation
__global__ void backpropagationKernel(
    float* __restrict__ Q_values,              // [batch_size, max_nodes, max_actions]
    int* __restrict__ N_values,                // [batch_size, max_nodes, max_actions]
    int* __restrict__ N_total,                 // [batch_size, max_nodes]
    int* __restrict__ virtual_loss,            // [batch_size, max_nodes, max_actions]
    const int* __restrict__ selected_paths,    // [batch_size, max_depth, 2]
    const int* __restrict__ path_lengths,      // [batch_size]
    const float* __restrict__ leaf_values,     // [batch_size]
    const int batch_size,
    const int max_nodes,
    const int max_actions,
    const int max_depth) {
    
    const int batch_idx = blockIdx.x;
    const int depth_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int path_length = path_lengths[batch_idx];
    if (depth_idx >= path_length) return;
    
    // Get the node and action from path
    int path_offset = batch_idx * max_depth * 2 + depth_idx * 2;
    int node_idx = selected_paths[path_offset];
    int action_idx = selected_paths[path_offset + 1];
    
    // Get the value to backpropagate
    float value = leaf_values[batch_idx];
    
    // Update Q and N values
    int idx = (batch_idx * max_nodes + node_idx) * max_actions + action_idx;
    
    // Atomic updates
    atomicAdd(&N_values[idx], 1);
    atomicAdd(&N_total[batch_idx * max_nodes + node_idx], 1);
    
    // Update Q value (running average)
    int n = N_values[idx];
    float old_q = Q_values[idx];
    float new_q = old_q + (value - old_q) / n;
    Q_values[idx] = new_q;
    
    // Remove virtual loss
    atomicAdd(&virtual_loss[idx], -1);
}

// Optimized kernel for large branching factors using warp-level primitives
__global__ void treeTraversalWarpOptimizedKernel(
    const float* __restrict__ Q_values,
    const int* __restrict__ N_values,
    const float* __restrict__ P_values,
    const int* __restrict__ N_total,
    const int* __restrict__ parent_indices,
    const int* __restrict__ child_indices,
    const bool* __restrict__ action_mask,
    const bool* __restrict__ is_terminal,
    int* __restrict__ virtual_loss,
    int* __restrict__ selected_paths,
    int* __restrict__ path_lengths,
    float* __restrict__ leaf_values,
    const float c_puct,
    const int batch_size,
    const int max_nodes,
    const int max_actions,
    const int max_simulations) {
    
    // Use cooperative groups for warp-level operations
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_idx = warp_id / (max_simulations / WARP_SIZE);
    
    if (batch_idx >= batch_size) return;
    
    // Removed unused shared memory declarations
    // Warp-level operations use shuffle instructions instead
    
    int current_node = 0;
    int path_length = 0;
    
    // Main traversal loop
    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        if (is_terminal[batch_idx * max_nodes + current_node]) break;
        
        // Parallel UCB calculation across warp
        float best_ucb = -1e10f;
        int best_action = -1;
        
        int n_total = N_total[batch_idx * max_nodes + current_node];
        float sqrt_n_total = sqrtf(static_cast<float>(n_total + 1));
        
        // Each thread in warp handles different actions
        for (int action = lane_id; action < max_actions; action += WARP_SIZE) {
            int idx = (batch_idx * max_nodes + current_node) * max_actions + action;
            
            if (action_mask[idx]) {
                int n = N_values[idx] + virtual_loss[idx];
                float q = (n > 0) ? Q_values[idx] : 0.0f;
                float p = P_values[idx];
                float ucb = q + c_puct * p * sqrt_n_total / (1.0f + n);
                
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best_action = action;
                }
            }
        }
        
        // Warp-level reduction to find best action using shuffle operations only
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_ucb = warp.shfl_down(best_ucb, offset);
            int other_action = warp.shfl_down(best_action, offset);
            
            if (other_ucb > best_ucb) {
                best_ucb = other_ucb;
                best_action = other_action;
            }
        }
        
        // Lane 0 has the best action
        best_action = warp.shfl(best_action, 0);
        if (best_action < 0) break;
        
        // Apply virtual loss (only lane 0)
        if (lane_id == 0) {
            int vl_idx = (batch_idx * max_nodes + current_node) * max_actions + best_action;
            atomicAdd(&virtual_loss[vl_idx], 1);
            
            // Store in path
            if (warp_id % (max_simulations / WARP_SIZE) == 0) {
                int path_idx = batch_idx * MAX_DEPTH * 2 + path_length * 2;
                selected_paths[path_idx] = current_node;
                selected_paths[path_idx + 1] = best_action;
                path_length++;
            }
        }
        
        // Move to next node
        int child_idx = (batch_idx * max_nodes + current_node) * max_actions + best_action;
        int next_node = child_indices[child_idx];
        if (next_node < 0) break;
        
        current_node = next_node;
        warp.sync();
    }
    
    // Store path length (lane 0 only)
    if (lane_id == 0 && warp_id % (max_simulations / WARP_SIZE) == 0) {
        path_lengths[batch_idx] = path_length;
    }
}

// Host wrapper function
void launchTreeTraversalKernel(
    const float* Q_values,
    const int* N_values,
    const float* P_values,
    const int* N_total,
    const int* parent_indices,
    const int* child_indices,
    const bool* action_mask,
    const bool* is_terminal,
    int* virtual_loss,
    int* selected_paths,
    int* path_lengths,
    float* leaf_values,
    float c_puct,
    int batch_size,
    int max_nodes,
    int max_actions,
    int max_simulations,
    bool use_warp_optimized,
    cudaStream_t stream) {
    
    if (use_warp_optimized && max_actions > 64) {
        // Use warp-optimized kernel for large action spaces
        int warps_per_batch = (max_simulations + WARP_SIZE - 1) / WARP_SIZE;
        int total_warps = batch_size * warps_per_batch;
        int threads_per_block = 256;
        int blocks = (total_warps * WARP_SIZE + threads_per_block - 1) / threads_per_block;
        
        treeTraversalWarpOptimizedKernel<<<blocks, threads_per_block, 0, stream>>>(
            Q_values, N_values, P_values, N_total,
            parent_indices, child_indices, action_mask, is_terminal,
            virtual_loss, selected_paths, path_lengths, leaf_values,
            c_puct, batch_size, max_nodes, max_actions, max_simulations
        );
    } else {
        // Use standard kernel
        int total_threads = batch_size * max_simulations;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        
        treeTraversalKernel<<<blocks, threads_per_block, 0, stream>>>(
            Q_values, N_values, P_values, N_total,
            parent_indices, child_indices, action_mask, is_terminal,
            virtual_loss, selected_paths, path_lengths, leaf_values,
            c_puct, batch_size, max_nodes, max_actions, max_simulations
        );
    }
}

// Host wrapper for backpropagation
void launchBackpropagationKernel(
    float* Q_values,
    int* N_values,
    int* N_total,
    int* virtual_loss,
    const int* selected_paths,
    const int* path_lengths,
    const float* leaf_values,
    int batch_size,
    int max_nodes,
    int max_actions,
    int max_depth,
    cudaStream_t stream) {
    
    // One block per batch element, threads handle different depths
    dim3 blocks(batch_size);
    dim3 threads(max_depth);
    
    backpropagationKernel<<<blocks, threads, 0, stream>>>(
        Q_values, N_values, N_total, virtual_loss,
        selected_paths, path_lengths, leaf_values,
        batch_size, max_nodes, max_actions, max_depth
    );
}

}  // namespace mcts
}  // namespace alphazero