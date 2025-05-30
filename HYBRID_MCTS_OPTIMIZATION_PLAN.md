# Hybrid CPU-GPU MCTS Optimization Plan

## Executive Summary

This document outlines a comprehensive plan to transform the current CPU-bound MCTS implementation into a high-performance hybrid CPU-GPU system. Based on analysis of the current bottlenecks and GPU MCTS research, we propose a staged implementation that can achieve 10-50x performance improvements.

## Current Bottlenecks Analysis

1. **Tree Traversal Dominance**: 98.5% of time spent in sequential tree operations
2. **GPU Underutilization**: Small batch sizes (1-15) despite 256 capacity
3. **Memory Inefficiency**: Poor cache locality due to scattered nodes
4. **Atomic Contention**: Heavy atomic operations for updates

## Proposed Hybrid Architecture

### Phase 1: Tensorized Tree Representation

#### Mathematical Foundation

The core insight is to represent the MCTS tree as dense tensors suitable for GPU operations:

**State Tensor**: $S \in \mathbb{R}^{B \times M \times D}$
- B: Batch size (parallel trees)
- M: Maximum nodes per tree
- D: State dimension

**Statistics Tensors**:
- $Q \in \mathbb{R}^{B \times M \times A}$: Action values
- $N \in \mathbb{Z}^{B \times M \times A}$: Visit counts
- $P \in \mathbb{R}^{B \times M \times A}$: Prior probabilities
- $W \in \mathbb{R}^{B \times M \times A}$: Cumulative values

**UCB Computation** (fully parallelized):
$$\text{UCB}[b,m,a] = Q[b,m,a] + c_{\text{puct}} \cdot P[b,m,a] \cdot \frac{\sqrt{\sum_i N[b,m,i]}}{1 + N[b,m,a]}$$

### Phase 2: GPU Kernel Implementation

#### 1. Batch Selection Kernel
```cuda
__global__ void batchSelectPaths(
    float* Q,        // [B, M, A]
    int* N,          // [B, M, A]
    float* P,        // [B, M, A]
    int* paths,      // [B, MAX_DEPTH, 2] (node, action)
    float c_puct,
    int batch_size
);
```

**Algorithm**:
1. Each thread block handles one tree
2. Cooperatively compute UCB scores for all children
3. Use warp shuffle for efficient argmax
4. Apply virtual loss atomically
5. Continue until leaf reached

#### 2. Batch Neural Network Evaluation
```cuda
__global__ void prepareNNBatch(
    int* leaf_nodes,     // [B]
    float* states,       // [B, M, D]
    float* nn_input,     // [B, D]
    int batch_size
);
```

#### 3. Batch Backup Kernel
```cuda
__global__ void batchBackup(
    float* values,       // [B] from NN
    int* paths,         // [B, MAX_DEPTH, 2]
    float* W,           // [B, M, A]
    int* N,             // [B, M, A]
    int batch_size
);
```

### Phase 3: Memory Optimization

#### GPU Memory Layout
```
Unified Memory Region:
├── Tree Statistics (Q, N, P, W)
├── Node Metadata (parent, children indices)
├── State Cache (frequently accessed states)
└── Transposition Table (GPU-accelerated lookup)

Device Memory:
├── NN Weights (persistent)
├── Batch Buffers (double buffered)
└── Temporary Computation Space
```

#### Data Structure Optimizations
1. **SoA Layout**: Separate arrays for Q, N, P instead of AoS
2. **Compressed Indices**: 16-bit indices for nodes < 65K
3. **Sparse Storage**: CSR format for nodes with many children
4. **Aligned Access**: Ensure coalesced memory patterns

### Phase 4: Pipeline Architecture

```
CPU Thread Pool          GPU Execution
┌─────────────┐         ┌──────────────────┐
│ Tree Manager│         │ Selection Kernel │
│   (Host)    │ ──────> │  (UCB + Virtual │
└─────────────┘         │     Loss)        │
                        └──────────────────┘
                                 │
                                 ▼
┌─────────────┐         ┌──────────────────┐
│State Prepare│ <────── │  Leaf Detection  │
│   (Host)    │         │    (Device)      │
└─────────────┘         └──────────────────┘
                                 │
                                 ▼
┌─────────────┐         ┌──────────────────┐
│ Expansion   │         │   NN Inference   │
│   Logic     │ ──────> │   (TorchScript)  │
└─────────────┘         └──────────────────┘
                                 │
                                 ▼
┌─────────────┐         ┌──────────────────┐
│Tree Update  │ <────── │  Backup Kernel   │
│  (Host)     │         │ (Scatter Add)    │
└─────────────┘         └──────────────────┘
```

### Implementation Strategy

#### Stage 1: Basic GPU Tree Operations (Week 1)
- [ ] Implement tensorized tree representation
- [ ] Basic UCB calculation kernel
- [ ] Simple batch selection kernel
- [ ] Integration with existing code

#### Stage 2: Advanced GPU Features (Week 2)
- [ ] Virtual loss mechanism
- [ ] Sparse tensor support for large branching
- [ ] CUDA graph optimization
- [ ] Multi-stream execution

#### Stage 3: Memory Optimization (Week 3)
- [ ] Unified memory implementation
- [ ] GPU-resident transposition table
- [ ] Zero-copy state transfers
- [ ] Memory pool enhancements

#### Stage 4: Full Integration (Week 4)
- [ ] Complete hybrid pipeline
- [ ] Performance tuning
- [ ] Multi-GPU support
- [ ] Production deployment

## Performance Projections

Based on similar implementations and our analysis:

1. **Tree Selection**: 50x speedup (GPU parallel vs CPU sequential)
2. **NN Inference**: 2x improvement (better batching)
3. **Backup Phase**: 20x speedup (parallel updates)
4. **Overall**: 10-30x end-to-end improvement

## Specific Optimizations for Large Branching (Mid/End-game)

### Progressive Widening on GPU
```cuda
__device__ int getExplorationWidth(int visits) {
    return min(MAX_ACTIONS, (int)(C_w * powf(visits, ALPHA_w)));
}
```

### Sparse UCB for Wide Nodes
For nodes with >64 children:
1. Store only visited children in sparse format
2. Compute UCB only for promising actions
3. Use two-stage selection (coarse then fine)

### Adaptive Batch Sizing
```
if (avg_branching_factor > 32) {
    batch_size = min(64, GPU_MEMORY / (avg_branching * state_size));
} else {
    batch_size = 256;  // Standard batch
}
```

## Validation Metrics

1. **Simulations per second**: Target 10x improvement
2. **GPU Utilization**: Target >80% (currently <20%)
3. **Batch Efficiency**: Average batch size >64 (currently ~10)
4. **ELO Preservation**: No degradation in play strength

## Risk Mitigation

1. **Correctness**: Extensive unit tests for each kernel
2. **Compatibility**: Maintain CPU fallback path
3. **Memory**: Graceful degradation for large trees
4. **Debugging**: Comprehensive logging and profiling

## Next Steps

1. Implement basic tensorized tree structure
2. Develop and test UCB calculation kernel
3. Integrate with existing SharedInferenceQueue
4. Benchmark against current implementation
5. Iterate based on profiling results