#pragma once

#include <cuda_runtime.h>

namespace alphazero {
namespace mcts {

// Tree traversal kernel launcher
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
    cudaStream_t stream);

// Backpropagation kernel launcher
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
    cudaStream_t stream);

}  // namespace mcts
}  // namespace alphazero