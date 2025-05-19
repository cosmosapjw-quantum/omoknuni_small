# MCTS Architecture Document

This document describes the current architecture of the Monte Carlo Tree Search (MCTS) implementation in Omoknuni.

## Overview

The MCTS implementation in Omoknuni is designed for maximum performance and scalability through:
- **Leaf parallelization** with centralized GPU batch evaluation
- **Lock-free concurrent queues** for efficient thread communication
- **Progressive widening** and **RAVE** for improved search quality
- **Transposition tables** for position caching
- **Root parallelization** for multiple concurrent searches

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCTSEngine                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Worker Threads    │    │        MCTSEvaluator             │   │
│  │   (OpenMP Pool)     │    │  ┌──────────────────────────┐    │   │
│  │ ┌─────────────────┐ │    │  │   Inference Function     │    │   │
│  │ │ Tree Traversal  │ │    │  │   (libtorch/CUDA)       │    │   │
│  │ │ Leaf Selection  ├─┼────┼─►│   Batch Processing      │    │   │
│  │ │ Virtual Loss    │ │    │  │   Result Distribution   │    │   │
│  │ └─────────────────┘ │    │  └──────────────────────────┘    │   │
│  └─────────────────────┘    └──────────────────────────────────┘   │
│            │                              ▲                         │
│            ▼                              │                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Concurrent Queues (Lock-Free)                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  leaf_queue_: PendingEvaluation                              │   │
│  │  batch_queue_: BatchInfo                                     │   │
│  │  result_queue_: <NetworkOutput, PendingEvaluation>           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │  Transposition      │    │        Node Tracker              │   │
│  │  Table              │    │  (Lock-free pending management)  │   │
│  └─────────────────────┘    └──────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### MCTSEngine

The main orchestrator that manages:
- Worker thread pool (OpenMP)
- Queue management
- Transposition table
- Statistics tracking
- Configuration settings

Key methods:
- `search()`: Main entry point for MCTS search
- `runOpenMPSearch()`: Parallel search using OpenMP
- `executeSingleSimulation()`: Single MCTS simulation
- `runRootParallelSearch()`: Multiple independent MCTS trees

### Worker Threads

OpenMP-based thread pool that performs:
- Tree traversal using PUCT selection
- Leaf node identification
- Virtual loss application
- Queue submission of pending evaluations

### MCTSEvaluator

Dedicated thread for neural network evaluation:
- Collects leaf states from queue
- Batches states up to configured size or timeout
- Performs GPU inference
- Distributes results back through queues

### Concurrent Queues

Lock-free queues for thread communication:
- `leaf_queue_`: Worker threads → Evaluator
- `batch_queue_`: Internal batch tracking
- `result_queue_`: Evaluator → Workers

### MCTSNode

Tree nodes with thread-safe operations:
```cpp
class MCTSNode {
    std::atomic<int> visit_count_;
    std::atomic<float> total_value_;
    std::atomic<int> virtual_loss_;
    
    // RAVE statistics
    std::vector<std::atomic<int>> rave_visits_;
    std::vector<std::atomic<float>> rave_values_;
    
    // Progressive widening
    int expanded_children_ = 0;
    std::mutex expansion_mutex_;
};
```

### Transposition Table

Shared position cache:
- Sharded design for reduced contention
- Zobrist hashing for position keys
- Age-based replacement policy

## Search Algorithm

### 1. Selection Phase
```cpp
while (!node->isLeaf()) {
    if (use_progressive_widening) {
        int max_children = c * pow(visits, k);
        if (node->expanded_children < max_children) {
            // Expand new child
        }
    }
    
    node = selectBestChild(node);  // PUCT + RAVE
    node->addVirtualLoss(virtual_loss);
}
```

### 2. Expansion & Evaluation
```cpp
PendingEvaluation eval;
eval.node = leaf_node;
eval.state = cloneGameState(current_state);
eval.path = current_path;

leaf_queue_.enqueue(eval);
```

### 3. Batch Processing
```cpp
// In evaluator thread
while (running) {
    auto batch = collectBatch(batch_size, timeout);
    auto results = neural_network->inference(batch);
    
    for (auto& [result, eval] : zip(results, batch)) {
        result_queue_.enqueue({result, eval});
    }
}
```

### 4. Backpropagation
```cpp
// In result distributor thread
while (running) {
    auto [output, eval] = result_queue_.dequeue();
    
    for (auto& node : eval.path) {
        node->removeVirtualLoss(virtual_loss);
        node->update(output.value);
        
        if (use_rave) {
            updateRAVE(node, eval.action, output.value);
        }
    }
}
```

## Configuration Options

### Basic Settings
```yaml
mcts_num_simulations: 800
mcts_num_threads: 12
mcts_batch_size: 256
mcts_batch_timeout_ms: 5
mcts_virtual_loss: 3
```

### Advanced Features
```yaml
# Progressive Widening
mcts_use_progressive_widening: true
mcts_progressive_widening_c: 1.0
mcts_progressive_widening_k: 10.0

# RAVE
mcts_use_rave: true
mcts_rave_constant: 3000.0

# Root Parallelization
mcts_use_root_parallelization: true
mcts_num_root_workers: 4

# Transposition Table
mcts_use_transposition_table: true
mcts_transposition_table_size_mb: 128
```

## Performance Characteristics

### Throughput
- Single thread: ~1,000 simulations/second
- 12 threads: ~10,000 simulations/second
- With batching: 50-100x speedup on GPU evaluation

### Latency
- Queue operations: < 1 microsecond
- Batch formation: 1-5 milliseconds
- GPU inference: 10-30 milliseconds (256 states)

### Memory Usage
- Node size: ~200 bytes
- Transposition table: Configurable (default 128MB)
- Queue buffers: ~10MB total

## Thread Synchronization

### Lock-Free Operations
- Queue enqueue/dequeue
- Atomic node updates
- Transposition table lookups

### Minimal Locking
- Node expansion (mutex)
- Evaluator startup/shutdown
- Statistics collection

### Virtual Loss
Prevents thread collisions during selection:
```cpp
// Apply virtual loss
node->virtual_loss_ += settings.virtual_loss;

// After evaluation
node->virtual_loss_ -= settings.virtual_loss;
```

## Future Optimizations

1. **Dynamic Batch Sizing**: Adapt batch size based on queue depth
2. **NUMA Optimization**: Pin threads to CPU cores
3. **GPU Streams**: Overlap computation and memory transfers
4. **Tree Reuse**: Maintain subtrees between moves
5. **Neural Network Quantization**: INT8 inference for speed