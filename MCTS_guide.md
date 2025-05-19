# MCTS Implementation Guide

This document provides a comprehensive guide to the Monte Carlo Tree Search (MCTS) implementation in Omoknuni, reflecting the current state of the codebase.

## Overview

Omoknuni's MCTS implementation features:
- **Leaf-parallelization** with centralized batch evaluation
- **Progressive widening** to control tree branching
- **RAVE (Rapid Action Value Estimation)** for improved action value estimation
- **Root parallelization** for running multiple MCTS trees in parallel
- **Transposition tables** for position sharing across the tree
- **Virtual loss** to prevent thread collisions
- **Lock-free concurrent queues** for efficient producer-consumer patterns

## Architecture Components

### 1. MCTSEngine (include/mcts/mcts_engine.h)

The main MCTS controller that orchestrates the search process:

```cpp
class MCTSEngine {
    // Key configuration settings
    MCTSSettings settings_;
    
    // Specialized worker threads
    std::thread result_distributor_worker_;
    
    // Lock-free queues for efficient communication
    moodycamel::ConcurrentQueue<PendingEvaluation> leaf_queue_;
    moodycamel::ConcurrentQueue<BatchInfo> batch_queue_;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>> result_queue_;
    
    // Transposition table for position sharing
    std::unique_ptr<TranspositionTable> transposition_table_;
    
    // Node tracking for lock-free evaluation management
    std::unique_ptr<NodeTracker> node_tracker_;
};
```

### 2. MCTSEvaluator (include/mcts/mcts_evaluator.h)

Handles batched neural network inference:
- Collects leaf nodes until batch_size or timeout
- Performs GPU-accelerated batch inference
- Returns results through promises/futures or external queues

### 3. MCTSNode (include/mcts/mcts_node.h)

Tree nodes with enhanced features:
- Atomic counters for thread-safe updates
- Virtual loss tracking
- RAVE statistics for action value estimation
- Progressive widening support

### 4. MCTSSettings

Current configuration options:
```cpp
struct MCTSSettings {
    int num_simulations = 800;
    int num_threads = 12;
    int batch_size = 256;
    std::chrono::milliseconds batch_timeout = 5ms;
    int max_concurrent_simulations = 512;
    float exploration_constant = 1.4f;
    int virtual_loss = 3;
    
    // Dirichlet noise for exploration
    bool add_dirichlet_noise = true;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    
    // Progressive widening
    bool use_progressive_widening = true;
    float progressive_widening_c = 1.0f;
    float progressive_widening_k = 10.0f;
    
    // Root parallelization
    bool use_root_parallelization = true;
    int num_root_workers = 4;
    
    // RAVE settings
    bool use_rave = true;
    float rave_constant = 3000.0f;
    
    // Transposition table
    bool use_transposition_table = true;
    size_t transposition_table_size_mb = 128;
};
```

## Implementation Details

### Search Flow

1. **Initialization**:
   - Create root node
   - Initialize transposition table if enabled
   - Start evaluator thread

2. **Tree Traversal** (OpenMP parallelized):
   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < num_simulations; i++) {
       executeSingleSimulation(root, thread_local_batch);
   }
   ```

3. **Leaf Node Processing**:
   - Select leaf using PUCT + progressive widening
   - Apply virtual loss
   - Queue leaf for evaluation
   - Batch inference on GPU
   - Backpropagate results

### Progressive Widening

Controls the number of children explored at each node:
```cpp
int max_children = static_cast<int>(
    progressive_widening_c * std::pow(node->getVisitCount(), progressive_widening_k)
);
```

### RAVE (Rapid Action Value Estimation)

Combines MCTS value estimates with all-moves-as-first (AMAF) heuristic:
```cpp
float rave_weight = sqrt(rave_constant / (3 * n + rave_constant));
float combined_value = (1 - rave_weight) * q_value + rave_weight * rave_value;
```

### Root Parallelization

When enabled, runs multiple MCTS trees in parallel:
```cpp
void runRootParallelSearch() {
    std::vector<std::shared_ptr<MCTSNode>> root_nodes(num_root_workers);
    
    #pragma omp parallel for
    for (int worker = 0; worker < num_root_workers; worker++) {
        // Each worker runs independent MCTS
        runSearch(root_nodes[worker]);
    }
    
    // Combine results from all trees
    combineRootResults(root_nodes);
}
```

### Batch Collection Strategy

The evaluator uses an adaptive batching approach:
1. Wait for minimum batch size
2. Continue collecting until batch_size or timeout
3. Process batch on GPU
4. Return results through concurrent queue

## Performance Optimizations

### 1. Lock-Free Queues
Uses moodycamel::ConcurrentQueue for zero-contention communication:
- Leaf submission queue
- Batch collection queue  
- Result distribution queue

### 2. Memory Management
- Node pools for efficient allocation
- GameState pools to reduce cloning overhead
- Transposition table with sharded design

### 3. Thread Configuration
- Worker threads = physical cores
- Separate evaluator thread for GPU inference
- Optional root parallel workers

### 4. Virtual Loss
Prevents thread collisions during selection:
```cpp
node->addVirtualLoss(settings_.virtual_loss);
// ... perform expansion/evaluation ...
node->removeVirtualLoss(settings_.virtual_loss);
```

## Common Issues and Solutions

### Batch Size Stuck at 1
- Increase batch_timeout
- Check queue size thresholds
- Verify evaluator thread is running

### GPU Memory Issues
- Reduce batch_size
- Lower num_filters in neural network
- Monitor with nvidia-smi

### Thread Contention
- Adjust num_threads to match CPU cores
- Enable progressive widening to reduce branching
- Use transposition table to share evaluations

### MCTS Stalling
- Check for deadlocks in queue operations
- Verify virtual loss is properly removed
- Monitor pending_evaluations counter

## Configuration Examples

### Quick Search (for debugging)
```yaml
mcts_num_simulations: 100
mcts_num_threads: 4
mcts_batch_size: 16
mcts_batch_timeout_ms: 10
```

### Performance Optimized
```yaml
mcts_num_simulations: 1600
mcts_num_threads: 12
mcts_batch_size: 256
mcts_batch_timeout_ms: 5
mcts_use_root_parallelization: true
mcts_num_root_workers: 4
```

### Memory Optimized
```yaml
mcts_max_concurrent_simulations: 256
mcts_batch_size: 64
transposition_table_size_mb: 64
```

## Debug Flags

Enable debug output by setting:
```cpp
#define MCTS_DEBUG 1
#define MCTS_VERBOSE 1
```

This provides detailed logging of:
- Batch formation
- Queue operations
- Thread synchronization
- Node expansion statistics

## Future Improvements

1. **Dynamic Batch Sizing**: Adapt batch size based on queue depth
2. **NUMA Awareness**: Pin threads to CPU cores for better cache locality
3. **GPU Stream Optimization**: Overlap computation and memory transfers
4. **Enhanced RAVE**: Implement progressive bias variants
5. **Tree Reuse**: Preserve subtrees between moves