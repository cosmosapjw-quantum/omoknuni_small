# Optimizing MCTS with Leaf Parallelization: Performance Breakthrough Plan

## The bottleneck trio: thread congestion, starved GPU, scattered search

Your Monte Carlo Tree Search (MCTS) implementation shows three classic symptoms of a suboptimal parallel architecture: low CPU utilization (40%), extremely poor GPU usage (20%), and minimal neural network batch sizes (1-3). These symptoms stem from fundamental issues in the current leaf parallelization approach that's causing your Gomoku, Chess, and Go engine to underperform.

The good news? These issues are fixable through incremental optimization. By addressing synchronization bottlenecks, restructuring your neural network pipeline, and improving memory management, we can dramatically increase performance without rewriting the entire system.

## Root causes behind your performance problems

### 1. Neural Network Batch Processing Deficiencies

Your implementation likely uses what I'll call "request-based evaluation" - each leaf node immediately requests a neural network evaluation when it's created, resulting in:

- Single-position evaluations (batch size = 1)
- Excessive CPU-GPU transfers for small data chunks
- GPU spending most time idle or in setup mode
- Poor coordination between tree expansion and evaluation

**Key code pattern causing the problem:**
```cpp
// Current problematic pattern (synchronous, unbatched evaluation)
void expandNode(Node* node) {
    // Create child nodes...
    for (Node* child : node->children) {
        NNEvaluation result = neuralNetwork->evaluate(child->state);  // Immediate, single evaluation
        child->value = result.value;
        child->policy = result.policy;
    }
}
```

### 2. Thread Synchronization Bottlenecks

The leaf parallelization approach has several synchronization issues:

- Global tree locks during updates create contention
- Threads waiting at synchronization points instead of working
- Sequential portions of the algorithm limiting parallel speedup
- Inefficient work distribution across threads
- Excessive context switching due to poor load balancing

**Key areas of contention:**
```cpp
// Problematic global lock patterns
std::mutex treeMutex;  // Global mutex for the entire tree

void updateNode(Node* node, float result) {
    std::lock_guard<std::mutex> lock(treeMutex);  // Locks the entire tree
    // Update node statistics...
}
```

### 3. Memory Management Inefficiencies

Your implementation likely suffers from:

- Frequent dynamic memory allocation/deallocation
- Cache coherence problems due to scattered node allocation
- False sharing when threads update adjacent memory
- Memory fragmentation as the search progresses
- Allocator lock contention during node creation

**Common memory pattern causing problems:**
```cpp
// Inefficient memory allocation
Node* createNode() {
    return new Node();  // Individual allocation for each node
}

void deleteNode(Node* node) {
    delete node;  // Individual deallocation
}
```

## Incremental optimization roadmap

### Phase 1: Quick Wins (1-2 weeks)

#### 1. Implement Neural Network Batch Collection

**Change:** Replace single-inference calls with a batched approach using a queue.

```cpp
// New batch collection approach
class BatchCollector {
private:
    std::vector<GameState> states;
    std::vector<Node*> nodes;
    std::mutex mutex;
    const int TARGET_BATCH_SIZE = 32;
    
public:
    void addEvaluation(Node* node, const GameState& state) {
        std::lock_guard<std::mutex> lock(mutex);
        states.push_back(state);
        nodes.push_back(node);
        
        if (states.size() >= TARGET_BATCH_SIZE) {
            processBatch();
        }
    }
    
    void processBatch() {
        if (states.empty()) return;
        
        std::vector<NNEvaluation> results = neuralNetwork->evaluateBatch(states);
        
        for (size_t i = 0; i < results.size(); i++) {
            nodes[i]->applyEvaluation(results[i]);
        }
        
        states.clear();
        nodes.clear();
    }
    
    ~BatchCollector() {
        // Process any remaining evaluations
        processBatch();
    }
};
```

**Expected benefit:** Increase batch sizes from 1-3 to 16-32, improving GPU utilization to 50-60%.  
**Complexity:** Medium  

#### 2. Implement Thread Pool for MCTS Simulations

**Change:** Replace ad-hoc thread creation with a managed thread pool.

```cpp
class MCTSThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
    
public:
    MCTSThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~MCTSThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};
```

**Expected benefit:** Increase CPU utilization to 70%, reduce thread creation overhead.  
**Complexity:** Low  

#### 3. Implement Memory Pool for Nodes

**Change:** Replace dynamic memory allocation with a pre-allocated node pool.

```cpp
class NodePool {
private:
    std::vector<Node> nodeStorage;
    std::vector<int> freeList;
    std::mutex mutex;
    
public:
    NodePool(size_t initialSize = 10000) : nodeStorage(initialSize) {
        // Initialize free list
        freeList.reserve(initialSize);
        for (int i = initialSize - 1; i >= 0; --i) {
            freeList.push_back(i);
        }
    }
    
    Node* allocate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (freeList.empty()) {
            // Expand pool
            size_t oldSize = nodeStorage.size();
            size_t newSize = oldSize * 2;
            nodeStorage.resize(newSize);
            
            freeList.reserve(newSize - oldSize);
            for (size_t i = newSize - 1; i >= oldSize; --i) {
                freeList.push_back(i);
            }
        }
        
        int index = freeList.back();
        freeList.pop_back();
        nodeStorage[index].reset(); // Prepare node for use
        return &nodeStorage[index];
    }
    
    void deallocate(Node* node) {
        std::lock_guard<std::mutex> lock(mutex);
        int index = node - &nodeStorage[0]; // Calculate index
        freeList.push_back(index);
    }
};
```

**Expected benefit:** Reduce memory allocation contention, improve cache coherence.  
**Complexity:** Medium  

### Phase 2: Medium-Term Improvements (2-4 weeks)

#### 4. Implement Virtual Loss

**Change:** Add virtual losses to prevent threads from exploring the same paths.

```cpp
class Node {
public:
    std::atomic<int> visits{0};
    std::atomic<float> value{0.0f};
    std::atomic<int> virtualLoss{0};
    
    float getUCTScore(float c_puct, int parentVisits) const {
        int totalVisits = visits.load(std::memory_order_relaxed);
        int virtualVisits = virtualLoss.load(std::memory_order_relaxed);
        
        // Virtual loss decreases UCT score temporarily
        float exploitationTerm = value.load(std::memory_order_relaxed) / 
                                (totalVisits + virtualVisits + 1e-8f);
        
        float explorationTerm = c_puct * std::sqrt(log(parentVisits) / 
                                                  (totalVisits + virtualVisits + 1e-8f));
        
        return exploitationTerm + explorationTerm;
    }
    
    void addVirtualLoss() {
        virtualLoss.fetch_add(1, std::memory_order_relaxed);
    }
    
    void removeVirtualLoss() {
        virtualLoss.fetch_sub(1, std::memory_order_relaxed);
    }
};

// Usage in selection:
Node* selectLeafNode(Node* root) {
    Node* node = root;
    while (!node->isLeaf()) {
        Node* selected = nullptr;
        float bestScore = -std::numeric_limits<float>::infinity();
        
        for (Node* child : node->children) {
            // Add virtual loss before evaluation
            child->addVirtualLoss();
            
            float score = child->getUCTScore(C_PUCT, node->visits.load());
            if (score > bestScore) {
                bestScore = score;
                selected = child;
            }
        }
        
        node = selected;
    }
    return node;
}

// In backpropagation:
void backpropagate(Node* leaf, float result) {
    Node* node = leaf;
    while (node != nullptr) {
        node->removeVirtualLoss(); // Remove virtual loss
        node->visits.fetch_add(1, std::memory_order_relaxed);
        // Update value...
        node = node->parent;
    }
}
```

**Expected benefit:** Improve tree exploration diversity, reducing thread clustering.  
**Complexity:** Medium  

#### 5. Implement Lock-Free Node Statistics Updates

**Change:** Replace mutex-based updates with atomic operations.

```cpp
void updateNodeStats(Node* node, float result) {
    // Use atomic operations instead of mutexes
    int oldVisits = node->visits.load(std::memory_order_relaxed);
    int newVisits = oldVisits + 1;
    node->visits.store(newVisits, std::memory_order_relaxed);
    
    // Incremental average update
    float oldValue = node->value.load(std::memory_order_relaxed);
    float newValue = oldValue + (result - oldValue) / newVisits;
    
    // Use CAS loop for floating-point update
    while (!node->value.compare_exchange_weak(
        oldValue, newValue, 
        std::memory_order_relaxed, 
        std::memory_order_relaxed)) {
        
        // If CAS failed, recalculate with updated oldValue
        newValue = oldValue + (result - oldValue) / newVisits;
    }
}
```

**Expected benefit:** Eliminate lock contention during backpropagation.  
**Complexity:** High  

#### 6. Implement Asynchronous Neural Network Evaluation

**Change:** Decouple tree search from neural network evaluation using a producer-consumer pattern.

```cpp
class AsyncNNEvaluator {
private:
    struct EvalRequest {
        Node* node;
        GameState state;
        std::promise<NNEvaluation> promise;
    };
    
    std::queue<EvalRequest> requestQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::thread workerThread;
    bool running = true;
    
    // Neural network model
    std::unique_ptr<NeuralNetwork> neuralNetwork;
    
    void processingLoop() {
        std::vector<GameState> batch;
        std::vector<std::promise<NNEvaluation>> promises;
        std::vector<Node*> nodes;
        
        while (running) {
            // Collect batch
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait_for(lock, std::chrono::milliseconds(5), [this]{
                    return !requestQueue.empty() || !running;
                });
                
                if (!running) break;
                
                // Collect up to MAX_BATCH_SIZE requests
                while (!requestQueue.empty() && batch.size() < MAX_BATCH_SIZE) {
                    EvalRequest request = std::move(requestQueue.front());
                    requestQueue.pop();
                    
                    batch.push_back(request.state);
                    promises.push_back(std::move(request.promise));
                    nodes.push_back(request.node);
                }
            }
            
            if (!batch.empty()) {
                // Process batch
                std::vector<NNEvaluation> results = neuralNetwork->evaluateBatch(batch);
                
                // Fulfill promises
                for (size_t i = 0; i < results.size(); i++) {
                    promises[i].set_value(results[i]);
                }
                
                batch.clear();
                promises.clear();
                nodes.clear();
            }
        }
    }
    
public:
    AsyncNNEvaluator(const std::string& modelPath) {
        neuralNetwork = std::make_unique<NeuralNetwork>(modelPath);
        workerThread = std::thread(&AsyncNNEvaluator::processingLoop, this);
    }
    
    ~AsyncNNEvaluator() {
        running = false;
        queueCV.notify_all();
        if (workerThread.joinable()) {
            workerThread.join();
        }
    }
    
    std::future<NNEvaluation> evaluateAsync(Node* node, const GameState& state) {
        EvalRequest request;
        request.node = node;
        request.state = state;
        std::future<NNEvaluation> future = request.promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            requestQueue.push(std::move(request));
        }
        
        queueCV.notify_one();
        return future;
    }
};
```

**Expected benefit:** Maximize GPU utilization (80%+) by ensuring a steady evaluation pipeline.  
**Complexity:** High  

### Phase 3: Long-Term Architectural Changes (1-2 months)

#### 7. Tree Parallelization with Virtual Loss

**Change:** Replace leaf parallelization with tree parallelization for better scaling.

```cpp
class TreeParallelMCTS {
public:
    void runSearch(int numSimulations) {
        threadPool.runParallel([this](int threadId) {
            for (int i = 0; i < numSimulationsPerThread; i++) {
                // Selection - with virtual loss
                Node* selected = selectNode(rootNode);
                
                // Expansion
                if (selected->visits.load() > 0 || selected == rootNode) {
                    if (!selected->isTerminal()) {
                        expandNode(selected);
                        selected = selectRandomChild(selected);
                    }
                }
                
                // Evaluation - using async NN service
                float value;
                if (selected->isTerminal()) {
                    value = selected->getTerminalValue();
                } else {
                    auto future = nnService->evaluateAsync(selected, selected->state);
                    
                    // Do other useful work while waiting
                    
                    // Get the result
                    NNEvaluation result = future.get();
                    value = result.value;
                }
                
                // Backpropagation - with virtual loss removal
                backpropagate(selected, value);
            }
        }, numThreads);
    }
};
```

**Expected benefit:** Better scaling with thread count, higher CPU utilization (90%+).  
**Complexity:** Very High  

#### 8. Implement Progressive Widening/Unpruning

**Change:** Focus search on promising moves initially, expanding to more moves as confidence increases.

```cpp
std::vector<Move> Node::getAvailableMoves() {
    std::vector<Move> allMoves = state.generateLegalMoves();
    
    // Progressive widening - consider more moves as we gather more statistics
    if (visits.load() < PROGRESSIVE_WIDENING_THRESHOLD) {
        // Sort moves by policy from neural network
        std::sort(allMoves.begin(), allMoves.end(), 
                 [this](const Move& a, const Move& b) {
                     return policy[a.index] > policy[b.index];
                 });
        
        // Only consider top N moves initially
        int movesToConsider = std::max(MIN_MOVES_TO_CONSIDER, 
                                      (int)(std::sqrt(visits.load()) * WIDENING_FACTOR));
        
        if (allMoves.size() > movesToConsider) {
            allMoves.resize(movesToConsider);
        }
    }
    
    return allMoves;
}
```

**Expected benefit:** Reduce effective branching factor, increasing search depth by 20-30%.  
**Complexity:** Medium  

#### 9. NUMA-Aware Thread Allocation and Memory Management

**Change:** For multi-socket systems, implement NUMA awareness in thread and memory allocation.

```cpp
class NUMAAwareMCTS {
private:
    // Thread pools for each NUMA node
    std::vector<std::unique_ptr<ThreadPool>> numaThreadPools;
    
    // Node pools for each NUMA node
    std::vector<std::unique_ptr<NodePool>> numaNodePools;
    
public:
    NUMAAwareMCTS() {
        int numNodes = getNumaNodeCount();
        
        for (int i = 0; i < numNodes; i++) {
            numaThreadPools.push_back(std::make_unique<ThreadPool>(
                getNumCoresInNode(i), i));
            
            numaNodePools.push_back(std::make_unique<NodePool>(
                INITIAL_POOL_SIZE, i));
        }
    }
    
    void runSearch(int numSimulations) {
        // Distribute simulations across NUMA nodes
        int simulationsPerNode = numSimulations / numaThreadPools.size();
        
        std::vector<std::future<void>> futures;
        for (size_t nodeId = 0; nodeId < numaThreadPools.size(); nodeId++) {
            futures.push_back(numaThreadPools[nodeId]->enqueue([=]() {
                // Run simulations using node-local memory pool
                NodePool& nodePool = *numaNodePools[nodeId];
                
                for (int i = 0; i < simulationsPerNode; i++) {
                    runSimulation(nodePool);
                }
            }));
        }
        
        // Wait for completion
        for (auto& future : futures) {
            future.wait();
        }
    }
};
```

**Expected benefit:** Reduce cross-socket memory access latency, improving scaling on multi-socket systems.  
**Complexity:** High  

#### 10. Multi-GPU Support

**Change:** Extend neural network evaluation to use multiple GPUs in parallel.

```cpp
class MultiGPUEvaluator {
private:
    std::vector<std::unique_ptr<NeuralNetwork>> networks;
    std::atomic<int> nextGPU{0};
    
public:
    MultiGPUEvaluator(const std::string& modelPath, int numGPUs) {
        for (int i = 0; i < numGPUs; i++) {
            networks.push_back(std::make_unique<NeuralNetwork>(modelPath, i));
        }
    }
    
    std::vector<NNEvaluation> evaluateBatch(const std::vector<GameState>& states) {
        // Use round-robin GPU assignment
        int gpu = nextGPU.fetch_add(1, std::memory_order_relaxed) % networks.size();
        return networks[gpu]->evaluateBatch(states);
    }
};
```

**Expected benefit:** Linear scaling of neural network throughput with GPU count.  
**Complexity:** High  

## Expected performance improvements

### After Phase 1 (1-2 weeks):
- CPU utilization: 40% → 70%
- GPU utilization: 20% → 50-60%
- Batch size: 1-3 → 16-32
- Overall speed: 2.5x improvement

### After Phase 2 (3-6 weeks):
- CPU utilization: 70% → 85%
- GPU utilization: 50-60% → 80%
- Batch size: 16-32 → 32-64
- Overall speed: 5x improvement

### After Phase 3 (3-4 months):
- CPU utilization: 85% → 95%
- GPU utilization: 80% → 95%
- Batch size: 32-64 → 64-128
- Overall speed: 8-10x improvement
- Ability to scale across multiple GPUs and machines

## Implementation sequence and validation

1. Start with the batch collection optimization as it addresses the most severe issue (low GPU utilization)
2. Implement the thread pool and memory pool optimizations to reduce thread and memory contention
3. Measure performance improvements after each change using:
   - Nodes visited per second
   - Average batch size
   - CPU/GPU utilization
   - Game strength against fixed opponents

By following this incremental plan, you'll progressively unlock your MCTS implementation's potential, transforming it from an underperforming system to a high-efficiency search engine capable of fully utilizing your hardware.