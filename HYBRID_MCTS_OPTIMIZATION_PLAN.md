# Hybrid MCTS Optimization Plan

## Current Performance Issues
- Move time: 2815ms (99.4% CPU overhead)
- Many tiny batches (1-15 states) despite 120 minimum threshold
- Sequential tree traversal is the bottleneck

## Proposed Hybrid Architecture

### 1. Batch Tree Selection (Immediate Impact)
Instead of sequential traversal, collect multiple paths simultaneously:

```cpp
// Current: Sequential per-thread traversal
for (int sim = 0; sim < num_simulations; sim++) {
    auto leaf = traverseToLeaf();  // Each thread does this independently
    leaf_queue.push(leaf);
}

// Optimized: Batch traversal
struct TraversalBatch {
    std::vector<MCTSNode*> current_nodes;
    std::vector<std::vector<MCTSNode*>> paths;
    std::vector<std::unique_ptr<GameState>> states;
};

// Process level by level
TraversalBatch batch;
batch.current_nodes.resize(batch_size, root);
while (!all_leaves) {
    // Select children for all nodes in parallel
    parallelSelectChildren(batch.current_nodes);
    // Move to next level
    updateBatch(batch);
}
```

### 2. Virtual Tree Traversal
Reduce memory access by traversing "virtual" paths:

```cpp
struct VirtualPath {
    uint32_t node_indices[MAX_DEPTH];
    float path_values[MAX_DEPTH];
    int depth;
};

// Traverse without touching actual nodes until leaf
VirtualPath paths[BATCH_SIZE];
parallelVirtualTraverse(paths);
// Then update real nodes in batch
batchUpdateNodes(paths);
```

### 3. State Pool with Zero-Copy
Eliminate state cloning overhead:

```cpp
class StatePool {
    // Pre-allocated states
    std::vector<GameState> pool;
    std::queue<GameState*> available;
    
    GameState* acquire() {
        auto* state = available.front();
        available.pop();
        return state;
    }
    
    void release(GameState* state) {
        state->reset();
        available.push(state);
    }
};
```

### 4. Pipeline Architecture
Three-stage pipeline with dedicated threads:

```
Stage 1: Tree Traversal (CPU)
- Batch select paths
- Use virtual traversal
- Output: leaf nodes + states

Stage 2: Batch Collection (CPU)
- Aggregate leaves from multiple games
- Enforce minimum batch size
- Output: full batches

Stage 3: NN Inference (GPU)
- Process full batches only
- Use CUDA streams for overlap
- Output: policy + value
```

### 5. Implementation Steps

#### Step 1: Implement Batch Selection
File: `mcts_engine_batch_selection.cpp`
```cpp
void MCTSEngine::batchTraverseToLeaves(
    std::vector<MCTSNode*>& roots,
    std::vector<LeafInfo>& leaves,
    int batch_size) {
    
    // Initialize traversal state
    std::vector<MCTSNode*> current(batch_size);
    std::vector<std::vector<MCTSNode*>> paths(batch_size);
    
    // Traverse level by level
    bool all_terminal = false;
    while (!all_terminal) {
        // Parallel select best children
        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            if (!current[i]->isLeaf()) {
                current[i] = current[i]->selectChild();
                paths[i].push_back(current[i]);
            }
        }
        
        // Check termination
        all_terminal = std::all_of(current.begin(), current.end(),
            [](MCTSNode* n) { return n->isLeaf(); });
    }
    
    // Collect results
    for (int i = 0; i < batch_size; i++) {
        leaves.emplace_back(current[i], paths[i]);
    }
}
```

#### Step 2: State Pool Implementation
File: `game_state_pool.cpp`
```cpp
class GameStatePool {
    static constexpr size_t POOL_SIZE = 10000;
    
    struct PooledState {
        alignas(64) uint8_t data[sizeof(GomokuState)];
        std::atomic<bool> in_use{false};
    };
    
    std::vector<PooledState> pool;
    std::atomic<size_t> next_idx{0};
    
public:
    GameState* acquire() {
        for (size_t attempts = 0; attempts < POOL_SIZE; attempts++) {
            size_t idx = next_idx.fetch_add(1) % POOL_SIZE;
            bool expected = false;
            if (pool[idx].in_use.compare_exchange_strong(expected, true)) {
                return reinterpret_cast<GameState*>(&pool[idx].data);
            }
        }
        // Fallback to allocation
        return new GomokuState();
    }
    
    void release(GameState* state) {
        // Find in pool and mark as free
        auto* pooled = reinterpret_cast<PooledState*>(state);
        pooled->in_use.store(false);
    }
};
```

#### Step 3: Pipeline Thread Coordination
```cpp
class PipelinedMCTSEngine {
    // Stage 1: Traversal threads
    void traversalThread() {
        while (running) {
            TraversalBatch batch;
            fillBatchFromGames(batch);
            batchTraverseToLeaves(batch);
            leaf_queue.enqueue_bulk(batch.leaves);
        }
    }
    
    // Stage 2: Batch collection
    void collectionThread() {
        std::vector<LeafInfo> batch;
        while (running) {
            // Wait for minimum batch size
            while (batch.size() < MIN_BATCH_SIZE) {
                LeafInfo leaf;
                if (leaf_queue.try_dequeue(leaf)) {
                    batch.push_back(leaf);
                }
            }
            
            // Submit to GPU
            gpu_queue.enqueue(std::move(batch));
            batch.clear();
        }
    }
    
    // Stage 3: GPU inference
    void inferenceThread() {
        while (running) {
            std::vector<LeafInfo> batch;
            if (gpu_queue.try_dequeue(batch)) {
                auto results = neural_net->inference(batch);
                distributeResults(batch, results);
            }
        }
    }
};
```

## Expected Performance Gains

1. **Batch Selection**: 5-10x reduction in tree traversal overhead
2. **State Pool**: Eliminate allocation/cloning overhead (~20% speedup)
3. **Pipeline**: Better CPU/GPU overlap, consistent batch sizes
4. **Overall**: 3-5x speedup, reaching 500-800ms per move

## Implementation Priority

1. **Immediate**: Implement batch selection for multiple simulations
2. **Short-term**: Add state pooling to eliminate cloning
3. **Medium-term**: Full pipeline architecture
4. **Long-term**: Virtual traversal optimization