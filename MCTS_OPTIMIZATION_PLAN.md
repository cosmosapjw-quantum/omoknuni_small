# MCTS Performance Optimization Plan

## Problem Analysis
Current performance shows:
- Move time: ~1450ms
- GPU inference time: ~21ms
- GPU utilization: <10% actual (despite good batching)
- CPU time: 98.5% of total

## Root Causes Identified

### 1. Sequential Tree Traversal (Primary Bottleneck)
- Each of 200 simulations traverses tree from root independently
- Each traversal involves:
  - Multiple atomic operations per node
  - UCB calculation for all children
  - Virtual loss updates
  - Memory allocations for game state clones
- With 6 parallel games × 200 simulations = 1200 tree traversals per move

### 2. Excessive Atomic Operations
- Each node access uses 3-4 atomic loads
- Virtual loss uses expensive memory_order_acq_rel
- Value updates use compare_exchange loops

### 3. Poor Memory Access Patterns
- Nodes scattered in memory (shared_ptr indirection)
- No cache prefetching
- Children stored as vector of shared_ptrs

### 4. State Cloning Overhead
- Every selection clones the game state
- Deep copies of board arrays
- No state pooling

## Proposed Solutions

### 1. Batch Tree Selection (Highest Priority)
Instead of 200 independent traversals:
```cpp
// Current: Sequential
for (int i = 0; i < num_simulations; i++) {
    node = selectChild();  // Expensive
    // ... traverse to leaf
}

// Optimized: Batched
vector<Node*> current_level = {root};
while (!current_level.empty()) {
    vector<Node*> next_level;
    // Process all nodes at this level together
    for (auto node : current_level) {
        auto child = selectChildBatched(node);
        next_level.push_back(child);
    }
    current_level = next_level;
}
```

### 2. SIMD UCB Calculation
```cpp
// Use AVX2 to calculate UCB for 8 children at once
__m256 scores = calculateUCB_AVX2(children_stats);
```

### 3. Memory Pool for States
- Pre-allocate game states
- Reuse instead of clone
- Stack allocator for temporary states

### 4. Reduce Atomic Overhead
- Use relaxed memory ordering where safe
- Batch updates to reduce contention
- Consider RCU for read-heavy operations

### 5. Cache-Friendly Node Layout
```cpp
struct MCTSNodeBlock {
    // Store multiple nodes contiguously
    MCTSNodeData nodes[64];
    // Children indices instead of pointers
    uint32_t children[64][MAX_CHILDREN];
};
```

## Implementation Priority

1. **Immediate Fix**: Increase batch collection time and minimum batch size
   - Change batch timeout from 10ms to 20ms
   - Minimum batch size from 80 to 120
   - This will reduce CPU overhead per batch

2. **Short Term**: Implement batch selection
   - Group simulations to traverse tree together
   - Reduces traversals from 1200 to ~10-20

3. **Medium Term**: SIMD optimization and memory pooling
   - Vectorize UCB calculations
   - Implement state pooling

4. **Long Term**: Fundamental architecture changes
   - Cache-friendly node layout
   - Lock-free data structures

## Expected Impact
- Reduce CPU time from 98.5% to ~70%
- Increase GPU utilization from <10% to 30-50%
- Reduce move time from 1450ms to 400-600ms
- Overall 2-3x speedup in self-play