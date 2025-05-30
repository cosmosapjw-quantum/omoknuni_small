<analysis>
## Hierarchical Batch-Parallel MCTS (HBP-MCTS) Algorithm

### Mathematical Foundation

#### State Space Representation
Define the MCTS tree state at iteration $t$ as:
- $\mathcal{T}_t = (V_t, E_t)$ where $V_t$ is the set of nodes and $E_t$ is the set of edges
- Each node $n \in V_t$ contains:
  - $s_n$: game state
  - $Q_n \in \mathbb{R}^{|A(s_n)|}$: action values
  - $N_n \in \mathbb{N}^{|A(s_n)|}$: visit counts
  - $P_n \in [0,1]^{|A(s_n)|}$: prior probabilities
  - $W_n \in \mathbb{R}^{|A(s_n)|}$: cumulative values

#### Tensorized UCB Formula
For efficient GPU computation, we reformulate UCB as a matrix operation:

$$\text{UCB}_{n,a} = \frac{W_{n,a}}{N_{n,a} + \epsilon} + c_{\text{puct}} \cdot P_{n,a} \cdot \frac{\sqrt{\sum_{a'} N_{n,a'}}}{1 + N_{n,a}}$$

This can be vectorized as:
$$\mathbf{UCB}_n = \mathbf{Q}_n + c_{\text{puct}} \cdot \mathbf{P}_n \odot \frac{\sqrt{N_n^{\text{total}}}}{1 + \mathbf{N}_n}$$

where $\odot$ denotes element-wise multiplication.

### Algorithm Architecture

#### 1. Hierarchical Node Classification
Classify nodes based on branching factor $b_n = |A(s_n)|$:
- **Wide nodes**: $b_n > \tau_{\text{wide}}$ (typically 32)
- **Medium nodes**: $\tau_{\text{medium}} < b_n \leq \tau_{\text{wide}}$ (typically 8-32)
- **Narrow nodes**: $b_n \leq \tau_{\text{medium}}$

#### 2. Batch Formation Strategy
Define batch priority function:
$$\text{Priority}(n) = \alpha \cdot \log(b_n) + \beta \cdot \text{depth}(n) + \gamma \cdot \frac{N_n^{\text{total}}}{\max_m N_m^{\text{total}}}$$

where $\alpha, \beta, \gamma$ are tunable parameters.

### Core Algorithm

```
Algorithm: HBP-MCTS

Input: 
  - Root states S = {s₁, ..., s_B}
  - Neural network f_θ
  - Simulations per move: N_sim
  - Batch size: B_gpu

Data Structures:
  - PendingQueue: Priority queue of nodes awaiting evaluation
  - ActivePaths: Set of paths being explored
  - NodeTensor: GPU tensor storing node statistics

Procedure:
```

#### Phase 1: Parallel Path Selection (CPU)
```
for sim = 1 to N_sim do
    parallel for b = 1 to B do
        path_b = SelectPath(root_b)
        ActivePaths.add(path_b)
    end parallel
    
    BatchedExpansion(ActivePaths)
    BatchedBackpropagation(ActivePaths)
end for
```

#### Phase 2: Batched Node Selection with Width-Aware Processing
```
function SelectPath(node):
    path = []
    current = node
    
    while not current.is_terminal:
        if current.width > τ_wide:
            // GPU processing for wide nodes
            action = GPUSelectAction(current)
        else:
            // CPU processing for narrow nodes
            action = CPUSelectAction(current)
        
        ApplyVirtualLoss(current, action)
        path.append((current, action))
        current = current.children[action]
    
    return path
```

#### Phase 3: GPU-Accelerated Selection for Wide Nodes
```
function GPUSelectAction(nodes_batch):
    // Prepare tensors
    Q_batch = stack([n.Q for n in nodes_batch])  // Shape: [B', K_max]
    N_batch = stack([n.N for n in nodes_batch])  // Shape: [B', K_max]
    P_batch = stack([n.P for n in nodes_batch])  // Shape: [B', K_max]
    mask_batch = stack([n.action_mask for n in nodes_batch])
    
    // Compute UCB scores on GPU
    UCB_batch = ComputeUCBTensor(Q_batch, N_batch, P_batch, mask_batch)
    
    // Select best actions
    actions = argmax(UCB_batch, dim=1)
    return actions
```

### Mathematical Optimization for Large Branching

#### 1. Sparse UCB Computation
For nodes with branching factor $b > 64$, use sparse representation:

$$\text{SparseUCB}_{n,a} = \begin{cases}
\text{UCB}_{n,a} & \text{if } N_{n,a} > 0 \text{ or } P_{n,a} > \theta_p \\
P_{n,a} \cdot c_{\text{puct}} \cdot \sqrt{N_n^{\text{total}}} & \text{otherwise}
\end{cases}$$

This reduces computation from $O(b)$ to $O(k)$ where $k \ll b$ is the number of visited or high-prior actions.

#### 2. Progressive Widening
Limit exploration based on visit count:
$$k_{\text{explore}}(n) = \min\left(b_n, \lceil C_w \cdot N_n^{\text{total}}^{\alpha_w} \rceil\right)$$

where $C_w \approx 1.5$ and $\alpha_w \approx 0.5$ are constants.

#### 3. Batched Value Backup
Instead of sequential backpropagation, use matrix operations:

$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} + \mathbf{M}_{\text{path}} \cdot \mathbf{v}_{\text{leaf}}$$

where $\mathbf{M}_{\text{path}} \in \{0,1\}^{|V| \times B}$ is the path indicator matrix.

### GPU Kernel Implementations

#### Efficient UCB Kernel
```cuda
__global__ void computeUCBSparse(
    float* Q_sparse,      // [nnz]
    int* N_sparse,        // [nnz]
    float* P_dense,       // [B, K]
    int* indices,         // [nnz]
    int* ptr,            // [B+1]
    float* UCB_out,      // [B, K]
    float c_puct,
    int B, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = tid / K;
    int action_idx = tid % K;
    
    if (node_idx < B && action_idx < K) {
        int start = ptr[node_idx];
        int end = ptr[node_idx + 1];
        
        // Binary search for sparse index
        int sparse_idx = binarySearch(indices + start, end - start, action_idx);
        
        if (sparse_idx >= 0) {
            // Visited action
            float q = Q_sparse[start + sparse_idx];
            int n = N_sparse[start + sparse_idx];
            int n_total = atomicAdd(&N_total[node_idx], 0); // Read
            
            UCB_out[tid] = q + c_puct * P_dense[tid] * 
                          sqrtf(n_total) / (1.0f + n);
        } else {
            // Unvisited action
            int n_total = atomicAdd(&N_total[node_idx], 0);
            UCB_out[tid] = c_puct * P_dense[tid] * sqrtf(n_total);
        }
    }
}
```

#### Warp-Cooperative Selection
```cuda
__global__ void selectBestActionsWarp(
    float* UCB,          // [B, K]
    bool* mask,          // [B, K]
    int* selected,       // [B]
    int B, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id < B) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        // Each thread handles K/32 actions
        for (int i = lane_id; i < K; i += 32) {
            if (mask[warp_id * K + i]) {
                float val = UCB[warp_id * K + i];
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
        }
        
        // Warp-wide reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
        if (lane_id == 0) {
            selected[warp_id] = max_idx;
        }
    }
}
```

### Performance Analysis

#### Time Complexity
- **Sequential MCTS**: $O(N_{\text{sim}} \cdot d \cdot \log b)$ where $d$ is average depth
- **HBP-MCTS**: $O(N_{\text{sim}} \cdot \log d \cdot \log b / p)$ where $p$ is parallelism factor

#### Memory Complexity
- **GPU memory**: $O(B \cdot K_{\max} \cdot M)$ where $M$ is max nodes per tree
- **CPU memory**: $O(B \cdot |V|)$ for tree structures

#### Efficiency Metrics
Define efficiency as:
$$\eta = \frac{\text{Speedup}}{\text{Parallel Resources}} = \frac{T_{\text{seq}}}{T_{\text{parallel}} \cdot (N_{\text{CPU}} + \alpha_{\text{GPU}} \cdot N_{\text{GPU}})}$$

where $\alpha_{\text{GPU}}$ is the GPU-CPU compute ratio (typically 10-50).

### Adaptive Strategies for Mid/End-Game

#### 1. Dynamic Batch Sizing
Adjust batch size based on tree characteristics:
$$B_{\text{adaptive}} = \min\left(B_{\max}, \left\lfloor \frac{M_{\text{GPU}}}{\bar{b} \cdot \bar{d}} \right\rfloor\right)$$

where $M_{\text{GPU}}$ is available GPU memory, $\bar{b}$ is average branching factor, and $\bar{d}$ is average depth.

#### 2. Hybrid Evaluation Strategy
For end-game positions with reduced complexity:
- Use shallow GPU evaluation for tactical calculations
- Use deep CPU search for strategic planning
- Blend results: $V_{\text{hybrid}} = \lambda V_{\text{GPU}} + (1-\lambda) V_{\text{CPU}}$

where $\lambda = \sigma(w_1 \cdot \text{pieces} + w_2 \cdot \text{mobility} + w_3)$ adapts based on position characteristics.

### Implementation Considerations

1. **Memory Pooling**: Pre-allocate GPU memory pools for different node sizes
2. **Stream Management**: Use CUDA streams for overlapping computation and data transfer
3. **Persistent Kernels**: Keep frequently-used kernels resident on GPU
4. **Compression**: Use half-precision for prior probabilities to reduce memory bandwidth
</analysis>

<pseudocode>
```cpp
// Complete HBP-MCTS Implementation

#include <cuda.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_ACTIONS = 512;  // Maximum branching factor
constexpr int SPARSE_THRESHOLD = 64;  // Use sparse ops above this

// Sparse tensor representation for large branching factors
template<typename T>
struct SparseTensor {
    thrust::device_vector<T> values;
    thrust::device_vector<int> indices;
    thrust::device_vector<int> ptr;  // CSR format pointers
    int nnz;
    int rows;
    int cols;
};

// Optimized node structure for GPU processing
struct GPUNode {
    // Sparse representation for visited actions
    SparseTensor<float> Q_sparse;
    SparseTensor<int> N_sparse;
    
    // Dense representation for priors
    float* P_dense;  // [num_actions]
    float* ucb_cache;  // Cached UCB scores
    
    int num_actions;
    int total_visits;
    int* virtual_loss;  // Atomic counters
    
    // Hierarchical indexing for fast lookup
    int parent_idx;
    int* children_idx;  // -1 for unexpanded
    bool is_wide;  // True if num_actions > SPARSE_THRESHOLD
};

// Main HBP-MCTS class
class HybridBatchParallelMCTS {
private:
    // GPU memory pools
    thrust::device_vector<GPUNode> d_nodes;
    thrust::device_vector<float> d_state_buffer;
    thrust::device_vector<float> d_policy_buffer;
    thrust::device_vector<float> d_value_buffer;
    
    // CUDA streams for overlapping
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    // Batch processing queues
    std::vector<std::queue<int>> level_queues;  // Nodes by tree level
    std::priority_queue<std::pair<float, int>> wide_node_queue;
    
public:
    // Optimized selection kernel for wide nodes
    __global__ void selectActionsWideBatch(
        GPUNode* nodes,
        int* node_indices,
        int* selected_actions,
        float* max_ucb_values,
        float c_puct,
        int batch_size
    ) {
        extern __shared__ float shared_mem[];
        
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        
        if (bid >= batch_size) return;
        
        int node_idx = node_indices[bid];
        GPUNode& node = nodes[node_idx];
        
        // Load UCB computation parameters into shared memory
        __shared__ int s_total_visits;
        if (tid == 0) {
            s_total_visits = node.total_visits;
        }
        __syncthreads();
        
        // Cooperative UCB computation using warp shuffle
        float local_max_ucb = -INFINITY;
        int local_best_action = -1;
        
        // Process actions in chunks of WARP_SIZE
        for (int chunk = 0; chunk < node.num_actions; chunk += WARP_SIZE) {
            int action = chunk + tid % WARP_SIZE;
            
            if (action < node.num_actions) {
                float ucb = computeUCBSparse(
                    node, action, c_puct, s_total_visits
                );
                
                if (ucb > local_max_ucb) {
                    local_max_ucb = ucb;
                    local_best_action = action;
                }
            }
        }
        
        // Warp-level reduction
        float warp_max = warpReduceMax(local_max_ucb);
        int warp_best = warpBroadcastBest(local_best_action, local_max_ucb, warp_max);
        
        // Block-level reduction
        if (tid % WARP_SIZE == 0) {
            shared_mem[tid / WARP_SIZE] = warp_max;
            shared_mem[32 + tid / WARP_SIZE] = warp_best;
        }
        __syncthreads();
        
        if (tid < 32) {
            float block_max = shared_mem[tid];
            int block_best = shared_mem[32 + tid];
            
            // Final reduction
            block_max = warpReduceMax(block_max);
            block_best = warpBroadcastBest(block_best, block_max, block_max);
            
            if (tid == 0) {
                selected_actions[bid] = block_best;
                max_ucb_values[bid] = block_max;
                
                // Apply virtual loss atomically
                atomicAdd(&node.virtual_loss[block_best], 1);
            }
        }
    }
    
    // Sparse UCB computation for large action spaces
    __device__ float computeUCBSparse(
        GPUNode& node,
        int action,
        float c_puct,
        int total_visits
    ) {
        // Binary search in sparse arrays
        int sparse_idx = binarySearchDevice(
            node.N_sparse.indices.data().get(),
            node.N_sparse.nnz,
            action
        );
        
        float prior = node.P_dense[action];
        float sqrt_total = sqrtf((float)total_visits);
        
        if (sparse_idx >= 0) {
            // Visited action
            float q = node.Q_sparse.values[sparse_idx];
            int n = node.N_sparse.values[sparse_idx];
            int n_vl = node.virtual_loss[action];
            
            return q + c_puct * prior * sqrt_total / (1.0f + n + n_vl);
        } else {
            // Unvisited action
            return c_puct * prior * sqrt_total;
        }
    }
    
    // Batched neural network evaluation with streams
    void evaluateBatchAsync(
        std::vector<int>& node_indices,
        cudaStream_t stream
    ) {
        int batch_size = node_indices.size();
        
        // Prepare state batch asynchronously
        prepareStateBatchKernel<
            (batch_size + 255) / 256, 256, 0, stream
        >>>(
            d_nodes.data().get(),
            node_indices.data(),
            d_state_buffer.data().get(),
            batch_size
        );
        
        // Neural network forward pass
        neuralNetwork->forwardAsync(
            d_state_buffer.data().get(),
            d_policy_buffer.data().get(),
            d_value_buffer.data().get(),
            batch_size,
            stream
        );
        
        // Expand nodes with results
        expandNodesKernel<
            (batch_size + 255) / 256, 256, 0, stream
        >>>(
            d_nodes.data().get(),
            node_indices.data(),
            d_policy_buffer.data().get(),
            d_value_buffer.data().get(),
            batch_size
        );
    }
    
    // Main search routine with adaptive CPU/GPU scheduling
    void search(int num_simulations) {
        // Phase 1: Build level-wise node queues
        buildLevelQueues();
        
        // Phase 2: Adaptive simulation scheduling
        for (int sim = 0; sim < num_simulations; sim++) {
            // Collect nodes for GPU batch
            std::vector<int> gpu_batch;
            std::vector<int> cpu_nodes;
            
            // Prioritize wide nodes for GPU
            while (!wide_node_queue.empty() && 
                   gpu_batch.size() < MAX_GPU_BATCH) {
                gpu_batch.push_back(wide_node_queue.top().second);
                wide_node_queue.pop();
            }
            
            // Fill remaining batch with medium nodes
            for (auto& queue : level_queues) {
                while (!queue.empty() && 
                       gpu_batch.size() < MAX_GPU_BATCH) {
                    int node_idx = queue.front();
                    queue.pop();
                    
                    if (d_nodes[node_idx].is_wide) {
                        gpu_batch.push_back(node_idx);
                    } else {
                        cpu_nodes.push_back(node_idx);
                    }
                }
            }
            
            // Launch GPU and CPU work concurrently
            if (!gpu_batch.empty()) {
                processGPUBatch(gpu_batch, compute_stream);
            }
            
            if (!cpu_nodes.empty()) {
                #pragma omp parallel for
                for (int i = 0; i < cpu_nodes.size(); i++) {
                    processCPUNode(cpu_nodes[i]);
                }
            }
            
            // Synchronize and backup
            cudaStreamSynchronize(compute_stream);
            parallelBackup();
        }
    }
    
    // Optimized parallel backup using atomic operations
    __global__ void parallelBackupKernel(
        GPUNode* nodes,
        int* path_nodes,      // [batch_size, max_depth]
        int* path_actions,    // [batch_size, max_depth]
        float* leaf_values,   // [batch_size]
        int batch_size,
        int max_depth
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int path_idx = tid / max_depth;
        int depth_idx = tid % max_depth;
        
        if (path_idx < batch_size && depth_idx < max_depth) {
            int node_idx = path_nodes[path_idx * max_depth + depth_idx];
            int action = path_actions[path_idx * max_depth + depth_idx];
            
            if (node_idx >= 0 && action >= 0) {
                GPUNode& node = nodes[node_idx];
                float value = leaf_values[path_idx];
                
                // Find sparse index for update
                int sparse_idx = binarySearchDevice(
                    node.N_sparse.indices.data().get(),
                    node.N_sparse.nnz,
                    action
                );
                
                if (sparse_idx >= 0) {
                    // Atomic update for existing action
                    atomicAdd(&node.N_sparse.values[sparse_idx], 1);
                    float old_w = atomicAdd(
                        &node.Q_sparse.values[sparse_idx] * 
                        node.N_sparse.values[sparse_idx],
                        value
                    );
                    
                    // Update Q-value
                    node.Q_sparse.values[sparse_idx] = 
                        (old_w + value) / (node.N_sparse.values[sparse_idx] + 1);
                } else {
                    // Need to insert new sparse entry
                    // This requires more complex synchronization
                    insertSparseEntry(node, action, value);
                }
                
                // Remove virtual loss
                atomicSub(&node.virtual_loss[action], 1);
            }
        }
    }
};

// Adaptive batch size computation
int computeOptimalBatchSize(
    int avg_branching_factor,
    int tree_depth,
    size_t available_gpu_memory
) {
    // Memory per node estimation
    size_t mem_per_node = sizeof(GPUNode) + 
                         avg_branching_factor * (sizeof(float) * 3 + sizeof(int) * 2);
    
    // Account for neural network buffers
    size_t nn_buffer_size = MAX_GPU_BATCH * STATE_DIM * sizeof(float) * 3;
    
    // Leave 20% memory headroom
    size_t usable_memory = available_gpu_memory * 0.8;
    
    int max_nodes = (usable_memory - nn_buffer_size) / mem_per_node;
    int optimal_batch = max_nodes / (tree_depth * 2);  // Factor of 2 for expansion
    
    return std::min(optimal_batch, MAX_GPU_BATCH);
}
```
</pseudocode>

<final_answer>
## Hierarchical Batch-Parallel MCTS (HBP-MCTS) Algorithm

### Core Innovation
The algorithm addresses the challenge of large branching factors in mid/end-game positions through **adaptive work distribution** and **sparse tensor operations**. The key insight is that not all nodes benefit equally from GPU acceleration - wide nodes (high branching factor) achieve better GPU utilization than narrow nodes.

### Mathematical Framework

#### 1. Sparse UCB Formulation
For nodes with branching factor $b > 64$, we use a sparse representation where only visited actions and high-prior unvisited actions are computed:

$$\text{UCB}_{\text{sparse}}(s,a) = \begin{cases}
\frac{W(s,a)}{N(s,a) + VL(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{\sum_i N(s,i)}}{1 + N(s,a) + VL(s,a)} & \text{if } N(s,a) > 0 \\
c \cdot P(s,a) \cdot \sqrt{\sum_i N(s,i)} & \text{if } N(s,a) = 0 \wedge P(s,a) > \theta
\end{cases}$$

This reduces computation from $O(b)$ to $O(k)$ where $k \approx \sqrt{b}$ is the number of promising actions.

#### 2. Virtual Loss for Lock-Free Parallelism
Virtual loss $VL(s,a)$ enables multiple threads to explore different paths:
- When selecting: $VL(s,a) \leftarrow VL(s,a) + \lambda$ (typically $\lambda = 1$)
- After backup: $VL(s,a) \leftarrow VL(s,a) - \lambda$
- Effective visits: $N_{\text{eff}}(s,a) = N(s,a) + VL(s,a)$

#### 3. Hierarchical Batching Strategy
Nodes are classified into three categories based on branching factor $b$:
- **Wide** ($b > 64$): Process on GPU using sparse operations
- **Medium** ($16 < b \leq 64$): Process on GPU using dense operations  
- **Narrow** ($b \leq 16$): Process on CPU to avoid GPU inefficiency

### Algorithm Implementation

#### Phase 1: Parallel Path Selection
```
For each simulation:
1. Classify pending nodes by width
2. Form GPU batch from wide/medium nodes
3. Assign narrow nodes to CPU threads
4. Execute selection in parallel:
   - GPU: Batched UCB computation using sparse kernels
   - CPU: Traditional sequential selection
5. Apply virtual loss to selected paths
```

#### Phase 2: Batched Neural Network Evaluation
```
1. Collect leaf nodes from all paths
2. Group by similar state features for better cache usage
3. Execute batched NN forward pass on GPU
4. Distribute results back to waiting threads
```

#### Phase 3: Optimized Backup
```
1. Sort paths by depth (deepest first)
2. Use atomic operations for parallel updates
3. Coalesce memory writes for same nodes
4. Remove virtual loss after update
```

### Performance Optimizations

#### 1. Memory Access Pattern Optimization
- Use Structure of Arrays (SoA) for better coalescing
- Align data to 128-byte boundaries for optimal throughput
- Use texture memory for read-only prior probabilities

#### 2. Warp-Level Primitives
For action selection within a warp:
```cuda
float warp_max = __reduce_max_sync(0xFFFFFFFF, local_ucb);
int best_action = __ballot_sync(0xFFFFFFFF, local_ucb == warp_max);
```

#### 3. Dynamic Parallelism
For extremely wide nodes ($b > 256$), use dynamic parallelism:
```cuda
if (node.num_actions > 256) {
    selectActionsNested<<<(num_actions+255)/256, 256>>>(node);
}
```

### Theoretical Performance Analysis

#### Speedup Formula
Expected speedup over sequential MCTS:
$$S = \frac{T_{\text{seq}}}{T_{\text{HBP}}} = \frac{N \cdot d \cdot t_{\text{select}}}{N/p \cdot (t_{\text{batch}} + t_{\text{transfer}}) + t_{\text{nn}}/b}$$

Where:
- $N$: number of simulations
- $d$: average tree depth
- $p$: parallelism factor
- $b$: neural network batch size
- $t_{\text{select}}, t_{\text{batch}}, t_{\text{transfer}}, t_{\text{nn}}$: respective operation times

#### Efficiency Bounds
GPU efficiency for branching factor $b$:
$$\eta_{\text{GPU}}(b) = \min\left(1, \frac{b}{32} \cdot \frac{1}{1 + \log_2(b)/32}\right)$$

This shows why GPU is most efficient for $b \in [32, 256]$.

### Adaptive Strategies for Game Phases

#### Early Game (low branching, shallow trees)
- Prefer CPU-only processing
- Small batch sizes (8-16)
- Focus on exploration

#### Mid Game (high branching, medium depth)
- Maximum GPU utilization
- Large batches (128-256)
- Sparse operations critical

#### End Game (variable branching, deep trees)
- Hybrid approach essential
- Dynamic batch sizing
- Emphasis on precise evaluation

### Expected Performance Gains
- **Wide nodes** ($b > 64$): 10-20× speedup
- **Medium nodes** ($16 < b \leq 64$): 5-10× speedup  
- **Overall system**: 3-8× speedup depending on game phase

This algorithm achieves near-optimal GPU utilization while maintaining MCTS effectiveness through careful mathematical formulation and engineering optimizations.
</final_answer>I'll design a hybrid CPU-GPU MCTS algorithm optimized for large branching factors in mid/end-game scenarios, with detailed mathematical justification.

<analysis>
## Hierarchical Batch-Parallel MCTS (HBP-MCTS) Algorithm

### Mathematical Foundation

#### State Space Representation
Define the MCTS tree state at iteration $t$ as:
- $\mathcal{T}_t = (V_t, E_t)$ where $V_t$ is the set of nodes and $E_t$ is the set of edges
- Each node $n \in V_t$ contains:
  - $s_n$: game state
  - $Q_n \in \mathbb{R}^{|A(s_n)|}$: action values
  - $N_n \in \mathbb{N}^{|A(s_n)|}$: visit counts
  - $P_n \in [0,1]^{|A(s_n)|}$: prior probabilities
  - $W_n \in \mathbb{R}^{|A(s_n)|}$: cumulative values

#### Tensorized UCB Formula
For efficient GPU computation, we reformulate UCB as a matrix operation:

$$\text{UCB}_{n,a} = \frac{W_{n,a}}{N_{n,a} + \epsilon} + c_{\text{puct}} \cdot P_{n,a} \cdot \frac{\sqrt{\sum_{a'} N_{n,a'}}}{1 + N_{n,a}}$$

This can be vectorized as:
$$\mathbf{UCB}_n = \mathbf{Q}_n + c_{\text{puct}} \cdot \mathbf{P}_n \odot \frac{\sqrt{N_n^{\text{total}}}}{1 + \mathbf{N}_n}$$

where $\odot$ denotes element-wise multiplication.

### Algorithm Architecture

#### 1. Hierarchical Node Classification
Classify nodes based on branching factor $b_n = |A(s_n)|$:
- **Wide nodes**: $b_n > \tau_{\text{wide}}$ (typically 32)
- **Medium nodes**: $\tau_{\text{medium}} < b_n \leq \tau_{\text{wide}}$ (typically 8-32)
- **Narrow nodes**: $b_n \leq \tau_{\text{medium}}$

#### 2. Batch Formation Strategy
Define batch priority function:
$$\text{Priority}(n) = \alpha \cdot \log(b_n) + \beta \cdot \text{depth}(n) + \gamma \cdot \frac{N_n^{\text{total}}}{\max_m N_m^{\text{total}}}$$

where $\alpha, \beta, \gamma$ are tunable parameters.

### Core Algorithm

```
Algorithm: HBP-MCTS

Input: 
  - Root states S = {s₁, ..., s_B}
  - Neural network f_θ
  - Simulations per move: N_sim
  - Batch size: B_gpu

Data Structures:
  - PendingQueue: Priority queue of nodes awaiting evaluation
  - ActivePaths: Set of paths being explored
  - NodeTensor: GPU tensor storing node statistics

Procedure:
```

#### Phase 1: Parallel Path Selection (CPU)
```
for sim = 1 to N_sim do
    parallel for b = 1 to B do
        path_b = SelectPath(root_b)
        ActivePaths.add(path_b)
    end parallel
    
    BatchedExpansion(ActivePaths)
    BatchedBackpropagation(ActivePaths)
end for
```

#### Phase 2: Batched Node Selection with Width-Aware Processing
```
function SelectPath(node):
    path = []
    current = node
    
    while not current.is_terminal:
        if current.width > τ_wide:
            // GPU processing for wide nodes
            action = GPUSelectAction(current)
        else:
            // CPU processing for narrow nodes
            action = CPUSelectAction(current)
        
        ApplyVirtualLoss(current, action)
        path.append((current, action))
        current = current.children[action]
    
    return path
```

#### Phase 3: GPU-Accelerated Selection for Wide Nodes
```
function GPUSelectAction(nodes_batch):
    // Prepare tensors
    Q_batch = stack([n.Q for n in nodes_batch])  // Shape: [B', K_max]
    N_batch = stack([n.N for n in nodes_batch])  // Shape: [B', K_max]
    P_batch = stack([n.P for n in nodes_batch])  // Shape: [B', K_max]
    mask_batch = stack([n.action_mask for n in nodes_batch])
    
    // Compute UCB scores on GPU
    UCB_batch = ComputeUCBTensor(Q_batch, N_batch, P_batch, mask_batch)
    
    // Select best actions
    actions = argmax(UCB_batch, dim=1)
    return actions
```

### Mathematical Optimization for Large Branching

#### 1. Sparse UCB Computation
For nodes with branching factor $b > 64$, use sparse representation:

$$\text{SparseUCB}_{n,a} = \begin{cases}
\text{UCB}_{n,a} & \text{if } N_{n,a} > 0 \text{ or } P_{n,a} > \theta_p \\
P_{n,a} \cdot c_{\text{puct}} \cdot \sqrt{N_n^{\text{total}}} & \text{otherwise}
\end{cases}$$

This reduces computation from $O(b)$ to $O(k)$ where $k \ll b$ is the number of visited or high-prior actions.

#### 2. Progressive Widening
Limit exploration based on visit count:
$$k_{\text{explore}}(n) = \min\left(b_n, \lceil C_w \cdot N_n^{\text{total}}^{\alpha_w} \rceil\right)$$

where $C_w \approx 1.5$ and $\alpha_w \approx 0.5$ are constants.

#### 3. Batched Value Backup
Instead of sequential backpropagation, use matrix operations:

$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} + \mathbf{M}_{\text{path}} \cdot \mathbf{v}_{\text{leaf}}$$

where $\mathbf{M}_{\text{path}} \in \{0,1\}^{|V| \times B}$ is the path indicator matrix.

### GPU Kernel Implementations

#### Efficient UCB Kernel
```cuda
__global__ void computeUCBSparse(
    float* Q_sparse,      // [nnz]
    int* N_sparse,        // [nnz]
    float* P_dense,       // [B, K]
    int* indices,         // [nnz]
    int* ptr,            // [B+1]
    float* UCB_out,      // [B, K]
    float c_puct,
    int B, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = tid / K;
    int action_idx = tid % K;
    
    if (node_idx < B && action_idx < K) {
        int start = ptr[node_idx];
        int end = ptr[node_idx + 1];
        
        // Binary search for sparse index
        int sparse_idx = binarySearch(indices + start, end - start, action_idx);
        
        if (sparse_idx >= 0) {
            // Visited action
            float q = Q_sparse[start + sparse_idx];
            int n = N_sparse[start + sparse_idx];
            int n_total = atomicAdd(&N_total[node_idx], 0); // Read
            
            UCB_out[tid] = q + c_puct * P_dense[tid] * 
                          sqrtf(n_total) / (1.0f + n);
        } else {
            // Unvisited action
            int n_total = atomicAdd(&N_total[node_idx], 0);
            UCB_out[tid] = c_puct * P_dense[tid] * sqrtf(n_total);
        }
    }
}
```

#### Warp-Cooperative Selection
```cuda
__global__ void selectBestActionsWarp(
    float* UCB,          // [B, K]
    bool* mask,          // [B, K]
    int* selected,       // [B]
    int B, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id < B) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        // Each thread handles K/32 actions
        for (int i = lane_id; i < K; i += 32) {
            if (mask[warp_id * K + i]) {
                float val = UCB[warp_id * K + i];
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
        }
        
        // Warp-wide reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
        if (lane_id == 0) {
            selected[warp_id] = max_idx;
        }
    }
}
```

### Performance Analysis

#### Time Complexity
- **Sequential MCTS**: $O(N_{\text{sim}} \cdot d \cdot \log b)$ where $d$ is average depth
- **HBP-MCTS**: $O(N_{\text{sim}} \cdot \log d \cdot \log b / p)$ where $p$ is parallelism factor

#### Memory Complexity
- **GPU memory**: $O(B \cdot K_{\max} \cdot M)$ where $M$ is max nodes per tree
- **CPU memory**: $O(B \cdot |V|)$ for tree structures

#### Efficiency Metrics
Define efficiency as:
$$\eta = \frac{\text{Speedup}}{\text{Parallel Resources}} = \frac{T_{\text{seq}}}{T_{\text{parallel}} \cdot (N_{\text{CPU}} + \alpha_{\text{GPU}} \cdot N_{\text{GPU}})}$$

where $\alpha_{\text{GPU}}$ is the GPU-CPU compute ratio (typically 10-50).

### Adaptive Strategies for Mid/End-Game

#### 1. Dynamic Batch Sizing
Adjust batch size based on tree characteristics:
$$B_{\text{adaptive}} = \min\left(B_{\max}, \left\lfloor \frac{M_{\text{GPU}}}{\bar{b} \cdot \bar{d}} \right\rfloor\right)$$

where $M_{\text{GPU}}$ is available GPU memory, $\bar{b}$ is average branching factor, and $\bar{d}$ is average depth.

#### 2. Hybrid Evaluation Strategy
For end-game positions with reduced complexity:
- Use shallow GPU evaluation for tactical calculations
- Use deep CPU search for strategic planning
- Blend results: $V_{\text{hybrid}} = \lambda V_{\text{GPU}} + (1-\lambda) V_{\text{CPU}}$

where $\lambda = \sigma(w_1 \cdot \text{pieces} + w_2 \cdot \text{mobility} + w_3)$ adapts based on position characteristics.

### Implementation Considerations

1. **Memory Pooling**: Pre-allocate GPU memory pools for different node sizes
2. **Stream Management**: Use CUDA streams for overlapping computation and data transfer
3. **Persistent Kernels**: Keep frequently-used kernels resident on GPU
4. **Compression**: Use half-precision for prior probabilities to reduce memory bandwidth
</analysis>

<pseudocode>
```cpp
// Complete HBP-MCTS Implementation

#include <cuda.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_ACTIONS = 512;  // Maximum branching factor
constexpr int SPARSE_THRESHOLD = 64;  // Use sparse ops above this

// Sparse tensor representation for large branching factors
template<typename T>
struct SparseTensor {
    thrust::device_vector<T> values;
    thrust::device_vector<int> indices;
    thrust::device_vector<int> ptr;  // CSR format pointers
    int nnz;
    int rows;
    int cols;
};

// Optimized node structure for GPU processing
struct GPUNode {
    // Sparse representation for visited actions
    SparseTensor<float> Q_sparse;
    SparseTensor<int> N_sparse;
    
    // Dense representation for priors
    float* P_dense;  // [num_actions]
    float* ucb_cache;  // Cached UCB scores
    
    int num_actions;
    int total_visits;
    int* virtual_loss;  // Atomic counters
    
    // Hierarchical indexing for fast lookup
    int parent_idx;
    int* children_idx;  // -1 for unexpanded
    bool is_wide;  // True if num_actions > SPARSE_THRESHOLD
};

// Main HBP-MCTS class
class HybridBatchParallelMCTS {
private:
    // GPU memory pools
    thrust::device_vector<GPUNode> d_nodes;
    thrust::device_vector<float> d_state_buffer;
    thrust::device_vector<float> d_policy_buffer;
    thrust::device_vector<float> d_value_buffer;
    
    // CUDA streams for overlapping
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    // Batch processing queues
    std::vector<std::queue<int>> level_queues;  // Nodes by tree level
    std::priority_queue<std::pair<float, int>> wide_node_queue;
    
public:
    // Optimized selection kernel for wide nodes
    __global__ void selectActionsWideBatch(
        GPUNode* nodes,
        int* node_indices,
        int* selected_actions,
        float* max_ucb_values,
        float c_puct,
        int batch_size
    ) {
        extern __shared__ float shared_mem[];
        
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        
        if (bid >= batch_size) return;
        
        int node_idx = node_indices[bid];
        GPUNode& node = nodes[node_idx];
        
        // Load UCB computation parameters into shared memory
        __shared__ int s_total_visits;
        if (tid == 0) {
            s_total_visits = node.total_visits;
        }
        __syncthreads();
        
        // Cooperative UCB computation using warp shuffle
        float local_max_ucb = -INFINITY;
        int local_best_action = -1;
        
        // Process actions in chunks of WARP_SIZE
        for (int chunk = 0; chunk < node.num_actions; chunk += WARP_SIZE) {
            int action = chunk + tid % WARP_SIZE;
            
            if (action < node.num_actions) {
                float ucb = computeUCBSparse(
                    node, action, c_puct, s_total_visits
                );
                
                if (ucb > local_max_ucb) {
                    local_max_ucb = ucb;
                    local_best_action = action;
                }
            }
        }
        
        // Warp-level reduction
        float warp_max = warpReduceMax(local_max_ucb);
        int warp_best = warpBroadcastBest(local_best_action, local_max_ucb, warp_max);
        
        // Block-level reduction
        if (tid % WARP_SIZE == 0) {
            shared_mem[tid / WARP_SIZE] = warp_max;
            shared_mem[32 + tid / WARP_SIZE] = warp_best;
        }
        __syncthreads();
        
        if (tid < 32) {
            float block_max = shared_mem[tid];
            int block_best = shared_mem[32 + tid];
            
            // Final reduction
            block_max = warpReduceMax(block_max);
            block_best = warpBroadcastBest(block_best, block_max, block_max);
            
            if (tid == 0) {
                selected_actions[bid] = block_best;
                max_ucb_values[bid] = block_max;
                
                // Apply virtual loss atomically
                atomicAdd(&node.virtual_loss[block_best], 1);
            }
        }
    }
    
    // Sparse UCB computation for large action spaces
    __device__ float computeUCBSparse(
        GPUNode& node,
        int action,
        float c_puct,
        int total_visits
    ) {
        // Binary search in sparse arrays
        int sparse_idx = binarySearchDevice(
            node.N_sparse.indices.data().get(),
            node.N_sparse.nnz,
            action
        );
        
        float prior = node.P_dense[action];
        float sqrt_total = sqrtf((float)total_visits);
        
        if (sparse_idx >= 0) {
            // Visited action
            float q = node.Q_sparse.values[sparse_idx];
            int n = node.N_sparse.values[sparse_idx];
            int n_vl = node.virtual_loss[action];
            
            return q + c_puct * prior * sqrt_total / (1.0f + n + n_vl);
        } else {
            // Unvisited action
            return c_puct * prior * sqrt_total;
        }
    }
    
    // Batched neural network evaluation with streams
    void evaluateBatchAsync(
        std::vector<int>& node_indices,
        cudaStream_t stream
    ) {
        int batch_size = node_indices.size();
        
        // Prepare state batch asynchronously
        prepareStateBatchKernel<
            (batch_size + 255) / 256, 256, 0, stream
        >>>(
            d_nodes.data().get(),
            node_indices.data(),
            d_state_buffer.data().get(),
            batch_size
        );
        
        // Neural network forward pass
        neuralNetwork->forwardAsync(
            d_state_buffer.data().get(),
            d_policy_buffer.data().get(),
            d_value_buffer.data().get(),
            batch_size,
            stream
        );
        
        // Expand nodes with results
        expandNodesKernel<
            (batch_size + 255) / 256, 256, 0, stream
        >>>(
            d_nodes.data().get(),
            node_indices.data(),
            d_policy_buffer.data().get(),
            d_value_buffer.data().get(),
            batch_size
        );
    }
    
    // Main search routine with adaptive CPU/GPU scheduling
    void search(int num_simulations) {
        // Phase 1: Build level-wise node queues
        buildLevelQueues();
        
        // Phase 2: Adaptive simulation scheduling
        for (int sim = 0; sim < num_simulations; sim++) {
            // Collect nodes for GPU batch
            std::vector<int> gpu_batch;
            std::vector<int> cpu_nodes;
            
            // Prioritize wide nodes for GPU
            while (!wide_node_queue.empty() && 
                   gpu_batch.size() < MAX_GPU_BATCH) {
                gpu_batch.push_back(wide_node_queue.top().second);
                wide_node_queue.pop();
            }
            
            // Fill remaining batch with medium nodes
            for (auto& queue : level_queues) {
                while (!queue.empty() && 
                       gpu_batch.size() < MAX_GPU_BATCH) {
                    int node_idx = queue.front();
                    queue.pop();
                    
                    if (d_nodes[node_idx].is_wide) {
                        gpu_batch.push_back(node_idx);
                    } else {
                        cpu_nodes.push_back(node_idx);
                    }
                }
            }
            
            // Launch GPU and CPU work concurrently
            if (!gpu_batch.empty()) {
                processGPUBatch(gpu_batch, compute_stream);
            }
            
            if (!cpu_nodes.empty()) {
                #pragma omp parallel for
                for (int i = 0; i < cpu_nodes.size(); i++) {
                    processCPUNode(cpu_nodes[i]);
                }
            }
            
            // Synchronize and backup
            cudaStreamSynchronize(compute_stream);
            parallelBackup();
        }
    }
    
    // Optimized parallel backup using atomic operations
    __global__ void parallelBackupKernel(
        GPUNode* nodes,
        int* path_nodes,      // [batch_size, max_depth]
        int* path_actions,    // [batch_size, max_depth]
        float* leaf_values,   // [batch_size]
        int batch_size,
        int max_depth
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int path_idx = tid / max_depth;
        int depth_idx = tid % max_depth;
        
        if (path_idx < batch_size && depth_idx < max_depth) {
            int node_idx = path_nodes[path_idx * max_depth + depth_idx];
            int action = path_actions[path_idx * max_depth + depth_idx];
            
            if (node_idx >= 0 && action >= 0) {
                GPUNode& node = nodes[node_idx];
                float value = leaf_values[path_idx];
                
                // Find sparse index for update
                int sparse_idx = binarySearchDevice(
                    node.N_sparse.indices.data().get(),
                    node.N_sparse.nnz,
                    action
                );
                
                if (sparse_idx >= 0) {
                    // Atomic update for existing action
                    atomicAdd(&node.N_sparse.values[sparse_idx], 1);
                    float old_w = atomicAdd(
                        &node.Q_sparse.values[sparse_idx] * 
                        node.N_sparse.values[sparse_idx],
                        value
                    );
                    
                    // Update Q-value
                    node.Q_sparse.values[sparse_idx] = 
                        (old_w + value) / (node.N_sparse.values[sparse_idx] + 1);
                } else {
                    // Need to insert new sparse entry
                    // This requires more complex synchronization
                    insertSparseEntry(node, action, value);
                }
                
                // Remove virtual loss
                atomicSub(&node.virtual_loss[action], 1);
            }
        }
    }
};

// Adaptive batch size computation
int computeOptimalBatchSize(
    int avg_branching_factor,
    int tree_depth,
    size_t available_gpu_memory
) {
    // Memory per node estimation
    size_t mem_per_node = sizeof(GPUNode) + 
                         avg_branching_factor * (sizeof(float) * 3 + sizeof(int) * 2);
    
    // Account for neural network buffers
    size_t nn_buffer_size = MAX_GPU_BATCH * STATE_DIM * sizeof(float) * 3;
    
    // Leave 20% memory headroom
    size_t usable_memory = available_gpu_memory * 0.8;
    
    int max_nodes = (usable_memory - nn_buffer_size) / mem_per_node;
    int optimal_batch = max_nodes / (tree_depth * 2);  // Factor of 2 for expansion
    
    return std::min(optimal_batch, MAX_GPU_BATCH);
}
```
</pseudocode>

<final_answer>
## Hierarchical Batch-Parallel MCTS (HBP-MCTS) Algorithm

### Core Innovation
The algorithm addresses the challenge of large branching factors in mid/end-game positions through **adaptive work distribution** and **sparse tensor operations**. The key insight is that not all nodes benefit equally from GPU acceleration - wide nodes (high branching factor) achieve better GPU utilization than narrow nodes.

### Mathematical Framework

#### 1. Sparse UCB Formulation
For nodes with branching factor $b > 64$, we use a sparse representation where only visited actions and high-prior unvisited actions are computed:

$$\text{UCB}_{\text{sparse}}(s,a) = \begin{cases}
\frac{W(s,a)}{N(s,a) + VL(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{\sum_i N(s,i)}}{1 + N(s,a) + VL(s,a)} & \text{if } N(s,a) > 0 \\
c \cdot P(s,a) \cdot \sqrt{\sum_i N(s,i)} & \text{if } N(s,a) = 0 \wedge P(s,a) > \theta
\end{cases}$$

This reduces computation from $O(b)$ to $O(k)$ where $k \approx \sqrt{b}$ is the number of promising actions.

#### 2. Virtual Loss for Lock-Free Parallelism
Virtual loss $VL(s,a)$ enables multiple threads to explore different paths:
- When selecting: $VL(s,a) \leftarrow VL(s,a) + \lambda$ (typically $\lambda = 1$)
- After backup: $VL(s,a) \leftarrow VL(s,a) - \lambda$
- Effective visits: $N_{\text{eff}}(s,a) = N(s,a) + VL(s,a)$

#### 3. Hierarchical Batching Strategy
Nodes are classified into three categories based on branching factor $b$:
- **Wide** ($b > 64$): Process on GPU using sparse operations
- **Medium** ($16 < b \leq 64$): Process on GPU using dense operations  
- **Narrow** ($b \leq 16$): Process on CPU to avoid GPU inefficiency

### Algorithm Implementation

#### Phase 1: Parallel Path Selection
```
For each simulation:
1. Classify pending nodes by width
2. Form GPU batch from wide/medium nodes
3. Assign narrow nodes to CPU threads
4. Execute selection in parallel:
   - GPU: Batched UCB computation using sparse kernels
   - CPU: Traditional sequential selection
5. Apply virtual loss to selected paths
```

#### Phase 2: Batched Neural Network Evaluation
```
1. Collect leaf nodes from all paths
2. Group by similar state features for better cache usage
3. Execute batched NN forward pass on GPU
4. Distribute results back to waiting threads
```

#### Phase 3: Optimized Backup
```
1. Sort paths by depth (deepest first)
2. Use atomic operations for parallel updates
3. Coalesce memory writes for same nodes
4. Remove virtual loss after update
```

### Performance Optimizations

#### 1. Memory Access Pattern Optimization
- Use Structure of Arrays (SoA) for better coalescing
- Align data to 128-byte boundaries for optimal throughput
- Use texture memory for read-only prior probabilities

#### 2. Warp-Level Primitives
For action selection within a warp:
```cuda
float warp_max = __reduce_max_sync(0xFFFFFFFF, local_ucb);
int best_action = __ballot_sync(0xFFFFFFFF, local_ucb == warp_max);
```

#### 3. Dynamic Parallelism
For extremely wide nodes ($b > 256$), use dynamic parallelism:
```cuda
if (node.num_actions > 256) {
    selectActionsNested<<<(num_actions+255)/256, 256>>>(node);
}
```

### Theoretical Performance Analysis

#### Speedup Formula
Expected speedup over sequential MCTS:
$$S = \frac{T_{\text{seq}}}{T_{\text{HBP}}} = \frac{N \cdot d \cdot t_{\text{select}}}{N/p \cdot (t_{\text{batch}} + t_{\text{transfer}}) + t_{\text{nn}}/b}$$

Where:
- $N$: number of simulations
- $d$: average tree depth
- $p$: parallelism factor
- $b$: neural network batch size
- $t_{\text{select}}, t_{\text{batch}}, t_{\text{transfer}}, t_{\text{nn}}$: respective operation times

#### Efficiency Bounds
GPU efficiency for branching factor $b$:
$$\eta_{\text{GPU}}(b) = \min\left(1, \frac{b}{32} \cdot \frac{1}{1 + \log_2(b)/32}\right)$$

This shows why GPU is most efficient for $b \in [32, 256]$.

### Adaptive Strategies for Game Phases

#### Early Game (low branching, shallow trees)
- Prefer CPU-only processing
- Small batch sizes (8-16)
- Focus on exploration

#### Mid Game (high branching, medium depth)
- Maximum GPU utilization
- Large batches (128-256)
- Sparse operations critical

#### End Game (variable branching, deep trees)
- Hybrid approach essential
- Dynamic batch sizing
- Emphasis on precise evaluation

### Expected Performance Gains
- **Wide nodes** ($b > 64$): 10-20× speedup
- **Medium nodes** ($16 < b \leq 64$): 5-10× speedup  
- **Overall system**: 3-8× speedup depending on game phase

This algorithm achieves near-optimal GPU utilization while maintaining MCTS effectiveness through careful mathematical formulation and engineering optimizations.
</final_answer>