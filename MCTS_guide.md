# MCTS Implementation Optimization Analysis

After reviewing the code, I have identified several areas for optimization and improvement in this AlphaZero-style MCTS implementation. Below is a comprehensive analysis and specific recommendations.

<todo_list>
1. **Batching Mechanism Overhaul**
   - Fix potential deadlock in batch accumulator worker when waiting for minimum batch size
   - Implement adaptive batch sizing based on inference latency and request queue depth
   - Optimize memory layout for tensor batching to reduce CPU→GPU transfer overhead

2. **Thread Coordination Improvements**
   - Redesign synchronization between tree traversal, batch accumulation, and result distribution
   - Implement work stealing for better load balancing between worker threads
   - Add backpressure mechanism to prevent memory explosion with too many pending evaluations

3. **Transposition Table Optimizations**
   - Improve sharding strategy to reduce lock contention
   - Implement a more sophisticated replacement policy based on node quality and search depth
   - Add entry age tracking to prioritize nodes from more recent searches

4. **Node Memory Management**
   - Implement node recycling to reduce allocation overhead
   - Add regular tree pruning to limit memory usage for long-running searches
   - Optimize MCTSNode memory layout to reduce per-node memory footprint

5. **Race Condition & Safety Fixes**
   - Fix potential race condition in node selection when multiple threads select the same node
   - Address possible use-after-move issue in PendingEvaluation handling
   - Ensure proper cleanup of promises during shutdown
   - Fix thread termination sequence in MCTSEngine destructor

6. **Performance Instrumentation**
   - Add detailed performance metrics for batch sizes, inference latency, and thread utilization
   - Implement adaptive timeout mechanism based on queue saturation and batch statistics
   - Create profiling hooks to identify bottlenecks in the search process
</todo_list>

<optimization_scheme>
## Core Architecture Optimization

I recommend a revised architecture focusing on the following components:

### 1. Thread Pool with Task-Based Architecture
Replace the current fixed threading model with a more flexible task-based system:

```cpp
class TaskQueue {
private:
    moodycamel::ConcurrentQueue<std::function<void()>> tasks_;
    std::atomic<bool> shutdown_{false};
    
public:
    // Submit tree traversal tasks when worker threads are available
    void enqueueTraversalTask(std::shared_ptr<MCTSNode> root) {
        tasks_.enqueue([this, root]() { traverseTree(root); });
    }
};
```

### 2. Two-Phase Batch Collection
Implement a two-phase batch collection strategy to ensure efficient GPU utilization:

```cpp
std::vector<EvaluationRequest> collectBatch(size_t target_size) {
    // Phase 1: Fast collection - grab immediately available requests
    std::vector<EvaluationRequest> batch = collectImmediate(target_size);
    
    // If batch is large enough, return immediately
    if (batch.size() >= min_efficient_batch_size_) {
        return batch;
    }
    
    // Phase 2: Wait with timeout for additional requests
    auto deadline = std::chrono::steady_clock::now() + adaptive_timeout_;
    while (std::chrono::steady_clock::now() < deadline && 
           batch.size() < target_size &&
           !shutdown_flag_) {
        // Try to collect more with short polls
        collectImmediate(target_size - batch.size(), batch);
        
        // If we have enough for efficient GPU utilization, break early
        if (batch.size() >= min_efficient_batch_size_) {
            break;
        }
        
        // Brief yield to avoid spinning
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    return batch;
}
```

### 3. Adaptive Batch Parameters
Make batch parameters adaptive based on runtime conditions:

```cpp
void updateBatchParameters() {
    // Adjust based on recent batch statistics
    float avg_time = getAverageBatchLatency().count();
    float avg_size = getAverageBatchSize();
    
    if (avg_size < target_batch_size_ * 0.5f && avg_time < timeout_.count() * 0.3f) {
        // Batches are consistently small and fast, reduce timeout
        timeout_ = std::max(std::chrono::milliseconds(1), 
                          timeout_ - std::chrono::milliseconds(5));
    } else if (avg_size >= target_batch_size_ * 0.9f) {
        // Batches are filling well, slightly increase timeout for better filling
        timeout_ += std::chrono::milliseconds(5);
    }
    
    // Cap the timeout to reasonable bounds
    timeout_ = std::min(timeout_, std::chrono::milliseconds(100));
}
```

### 4. Memory-Efficient Node Design
Optimize the MCTSNode class for memory efficiency:

```cpp
class MCTSNode {
private:
    // Replace individual atomic variables with a single statistics struct
    struct alignas(64) NodeStats {  // Align to cache line
        std::atomic<int> visit_count{0};
        std::atomic<float> value_sum{0.0f};
        std::atomic<int> virtual_loss{0};
    };
    NodeStats stats_;  // Single cache-aligned struct for better locality
    
    // Use a memory-efficient game state representation
    std::unique_ptr<CompactGameState> state_;
    
    // Use vector with reserve to avoid frequent reallocations
    std::vector<std::shared_ptr<MCTSNode>> children_;
    std::vector<int> actions_;
    
    // Use weak_ptr for parent to avoid circular references
    std::weak_ptr<MCTSNode> parent_;
};
```
</optimization_scheme>

<parallelization_improvements>
## Enhanced Parallelization Strategy

The current MCTS implementation has several thread-related issues that can be addressed:

### 1. Leaf Parallelization Improvements

Replace the current leaf parallelization with a more efficient design:

```cpp
void MCTSEngine::traverseTree(std::shared_ptr<MCTSNode> root) {
    // Selection phase with virtual loss to avoid thread collisions
    auto [leaf, path] = selectLeafNode(root);
    
    if (!leaf) return;
    
    // For terminal nodes, process immediately without queueing
    if (leaf->isTerminal()) {
        float value = getTerminalValue(leaf);
        backPropagate(path, value);
        return;
    }
    
    // For non-terminal leaf nodes, expand and queue for evaluation
    if (leaf->isLeaf()) {
        // Use an atomic check-and-set pattern to avoid duplicate expansion
        bool expanded = false;
        if (leaf->tryExpand(&expanded)) {
            // Node was successfully expanded by this thread
            queueForEvaluation(leaf, path);
        } else if (expanded) {
            // Node was already expanded by another thread
            // Re-run selection to find a new leaf
            traverseTree(root);
        } else {
            // Node couldn't be expanded (e.g., no legal moves)
            // Treat as terminal with default value
            backPropagate(path, 0.0f);
        }
    } else {
        // Node is already expanded but not terminal - something is wrong
        // Just backpropagate a default value
        backPropagate(path, 0.0f);
    }
}
```

### 2. Addressing Lock Contention

Minimize lock contention using atomic operations and lock-free data structures:

```cpp
// In MCTSNode::update
void update(float value) {
    // Lockless update using atomic operations
    visit_count_.fetch_add(1, std::memory_order_acq_rel);
    
    // Use atomic floating-point addition if available
    #if defined(__cpp_lib_atomic_float)
        value_sum_.fetch_add(value, std::memory_order_acq_rel);
    #else
        // Otherwise use compare-exchange loop with bounded retries
        float current = value_sum_.load(std::memory_order_acquire);
        float desired;
        int attempts = 0;
        constexpr int MAX_ATTEMPTS = 10;
        
        do {
            desired = current + value;
            if (++attempts > MAX_ATTEMPTS) {
                // After max attempts, use a mutex as fallback
                std::lock_guard<std::mutex> lock(update_mutex_);
                value_sum_.store(value_sum_.load() + value, std::memory_order_release);
                return;
            }
        } while (!value_sum_.compare_exchange_weak(current, desired,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire));
    #endif
}
```

### 3. Non-Blocking Batch Accumulation

Implement a non-blocking batch accumulation strategy to prevent worker starvation:

```cpp
void MCTSEvaluator::batchAccumulatorWorker() {
    std::vector<EvaluationRequest> current_batch;
    current_batch.reserve(max_batch_size_);
    
    while (!shutdown_) {
        // Phase 1: Collect immediately available items without blocking
        size_t initial_size = leaf_queue_.size_approx();
        if (initial_size > 0) {
            // Try to dequeue up to max_batch_size or what's available
            size_t to_dequeue = std::min(initial_size, max_batch_size_ - current_batch.size());
            
            EvaluationRequest temp;
            for (size_t i = 0; i < to_dequeue; i++) {
                if (leaf_queue_.try_dequeue(temp)) {
                    current_batch.push_back(std::move(temp));
                } else {
                    break;
                }
            }
        }
        
        // Phase 2: Determine if we should submit the batch
        bool should_submit = false;
        
        // Submit if batch is full
        if (current_batch.size() >= max_batch_size_) {
            should_submit = true;
        }
        // Submit if batch meets minimum size and we've waited long enough
        else if (current_batch.size() >= min_batch_size_ && 
                (std::chrono::steady_clock::now() - last_submit_time_) > batch_timeout_) {
            should_submit = true;
        }
        // Submit anything on shutdown
        else if (shutdown_ && !current_batch.empty()) {
            should_submit = true;
        }
        
        if (should_submit) {
            // Submit batch and track statistics
            submitBatch(std::move(current_batch));
            current_batch.clear();
            current_batch.reserve(max_batch_size_);
            last_submit_time_ = std::chrono::steady_clock::now();
        } else if (current_batch.empty()) {
            // If batch is empty, sleep briefly to avoid spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            // If batch is partially filled, sleep for a shorter duration
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}
```

### 4. Thread-Safe Node Recycling

Implement a thread-safe node recycling mechanism to reduce allocation overhead:

```cpp
class NodePool {
private:
    moodycamel::ConcurrentQueue<std::unique_ptr<MCTSNode>> recycled_nodes_;
    std::atomic<size_t> created_nodes_{0};
    std::atomic<size_t> recycled_count_{0};
    
public:
    std::shared_ptr<MCTSNode> createNode(std::unique_ptr<core::IGameState> state) {
        // Try to get a recycled node first
        std::unique_ptr<MCTSNode> node;
        if (recycled_nodes_.try_dequeue(node)) {
            // Reset the recycled node with new state
            node->reset(std::move(state));
            recycled_count_.fetch_add(1, std::memory_order_relaxed);
            return std::shared_ptr<MCTSNode>(node.release(), 
                [this](MCTSNode* ptr) { recycleNode(std::unique_ptr<MCTSNode>(ptr)); });
        }
        
        // If no recycled node available, create a new one
        created_nodes_.fetch_add(1, std::memory_order_relaxed);
        return std::shared_ptr<MCTSNode>(new MCTSNode(std::move(state)),
            [this](MCTSNode* ptr) { recycleNode(std::unique_ptr<MCTSNode>(ptr)); });
    }
    
private:
    void recycleNode(std::unique_ptr<MCTSNode> node) {
        // Clean up the node for recycling
        node->cleanup();
        
        // Only keep a reasonable number of recycled nodes
        if (recycled_nodes_.size_approx() < 1000) {
            recycled_nodes_.enqueue(std::move(node));
        }
    }
};
```
</parallelization_improvements>

<gpu_throughput_scenario>
## Maximizing GPU Throughput

To maximize GPU throughput, I propose the following concrete scenario:

### 1. Tensor Preallocation and Batch Compression

```cpp
class BatchProcessor {
private:
    // Preallocated tensors for different common batch sizes
    std::map<size_t, torch::Tensor> preallocated_inputs_;
    
    // Actual processing function
    std::function<std::vector<NetworkOutput>(torch::Tensor)> process_func_;
    
    // Track batch statistics for optimization
    RunningAverage batch_size_stats_{100};
    RunningAverage processing_time_stats_{100};
    
public:
    BatchProcessor(int channels, int height, int width) {
        // Preallocate tensors for common batch sizes
        for (size_t batch_size : {16, 32, 64, 128, 256}) {
            preallocated_inputs_[batch_size] = torch::zeros(
                {static_cast<long>(batch_size), channels, height, width},
                torch::TensorOptions().device(torch::kCUDA).requires_grad(false));
        }
    }
    
    std::vector<NetworkOutput> processBatch(const std::vector<std::unique_ptr<core::IGameState>>& states) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get batch size and find closest preallocated tensor
        size_t batch_size = states.size();
        size_t tensor_size = 0;
        for (auto it = preallocated_inputs_.begin(); it != preallocated_inputs_.end(); ++it) {
            if (it->first >= batch_size) {
                tensor_size = it->first;
                break;
            }
        }
        
        if (tensor_size == 0) {
            // If no suitable preallocated tensor, use the largest one
            tensor_size = preallocated_inputs_.rbegin()->first;
        }
        
        // Get tensor reference without copying
        torch::Tensor& input_tensor = preallocated_inputs_[tensor_size];
        
        // Use pinned memory for efficient CPU->GPU transfer if not already on GPU
        torch::Tensor cpu_tensor;
        if (input_tensor.device().is_cpu()) {
            cpu_tensor = torch::zeros(
                {static_cast<long>(batch_size), channels, height, width},
                torch::TensorOptions().pinned_memory(true));
        }
        
        // Fill the tensor with state data directly to avoid extra copies
        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; i++) {
            auto tensor_view = cpu_tensor.index({static_cast<long>(i)});
            states[i]->fillTensor(tensor_view);
        }
        
        // Transfer to GPU in one operation
        if (input_tensor.device().is_cuda()) {
            input_tensor.index({torch::indexing::Slice(0, batch_size)}).copy_(cpu_tensor);
        }
        
        // Process the batch
        std::vector<NetworkOutput> results = process_func_(
            input_tensor.index({torch::indexing::Slice(0, batch_size)}));
        
        // Track statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
            
        batch_size_stats_.add(batch_size);
        processing_time_stats_.add(duration_ms);
        
        // Periodically optimize preallocated tensor sizes
        optimizePreallocatedSizes();
        
        return results;
    }
    
private:
    void optimizePreallocatedSizes() {
        static int call_count = 0;
        if (++call_count % 100 != 0) return;
        
        // Use statistics to adjust preallocated tensor sizes
        float avg_batch_size = batch_size_stats_.average();
        
        // Update preallocated sizes based on observed patterns
        std::vector<size_t> optimal_sizes = {
            static_cast<size_t>(avg_batch_size * 0.5f),
            static_cast<size_t>(avg_batch_size),
            static_cast<size_t>(avg_batch_size * 1.5f),
            static_cast<size_t>(avg_batch_size * 2.0f)
        };
        
        // Keep sizes within reasonable bounds
        for (auto& size : optimal_sizes) {
            size = std::max(size_t(16), std::min(size_t(512), size));
        }
        
        // Update preallocated tensors
        for (size_t size : optimal_sizes) {
            if (preallocated_inputs_.find(size) == preallocated_inputs_.end()) {
                preallocated_inputs_[size] = torch::zeros(
                    {static_cast<long>(size), channels, height, width},
                    torch::TensorOptions().device(torch::kCUDA).requires_grad(false));
            }
        }
    }
};
```

### 2. Asynchronous GPU Pipeline

Implement an asynchronous pipeline to overlap CPU and GPU operations:

```cpp
class AsyncGpuPipeline {
private:
    // Multiple streams for overlapping operations
    std::vector<torch::cuda::CUDAStream> streams_;
    
    // Multiple buffers for double-buffering
    std::vector<torch::Tensor> input_buffers_;
    std::vector<torch::Tensor> output_buffers_;
    
    // Current buffer index
    std::atomic<int> current_buffer_{0};
    
    // Neural network model
    torch::jit::script::Module model_;
    
public:
    AsyncGpuPipeline(int num_buffers, int batch_size, int channels, int height, int width) 
        : streams_(num_buffers), input_buffers_(num_buffers), output_buffers_(num_buffers) {
        
        // Initialize streams and buffers
        for (int i = 0; i < num_buffers; i++) {
            streams_[i] = torch::cuda::getStreamFromPool();
            
            // Create pinned memory buffers for efficient transfer
            input_buffers_[i] = torch::zeros(
                {batch_size, channels, height, width},
                torch::TensorOptions().pinned_memory(true));
                
            // Create GPU output buffers
            output_buffers_[i] = torch::zeros(
                {batch_size, output_size},
                torch::TensorOptions().device(torch::kCUDA));
        }
    }
    
    std::vector<NetworkOutput> processBatchAsync(
        const std::vector<std::unique_ptr<core::IGameState>>& states) {
        
        // Get current buffer index
        int buffer_idx = current_buffer_.fetch_add(1, std::memory_order_relaxed) % input_buffers_.size();
        
        // Get references to current buffers and stream
        torch::Tensor& input_buffer = input_buffers_[buffer_idx];
        torch::Tensor& output_buffer = output_buffers_[buffer_idx];
        torch::cuda::CUDAStream& stream = streams_[buffer_idx];
        
        // Fill input buffer with state data (CPU operation)
        #pragma omp parallel for
        for (size_t i = 0; i < states.size(); i++) {
            auto tensor_view = input_buffer.index({static_cast<long>(i)});
            states[i]->fillTensor(tensor_view);
        }
        
        // Submit transfer and compute to the stream
        torch::cuda::setCurrentStream(stream);
        
        // Transfer input to GPU
        auto gpu_input = input_buffer.to(torch::kCUDA, true);
        
        // Run model in current stream
        auto outputs = model_.forward({gpu_input}).toTensor();
        
        // Copy results to output buffer
        output_buffer.copy_(outputs, true);
        
        // Synchronize the stream
        stream.synchronize();
        
        // Extract results
        std::vector<NetworkOutput> results;
        results.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); i++) {
            NetworkOutput output;
            // Extract policy and value from output tensor
            auto tensor_view = output_buffer.index({static_cast<long>(i)});
            
            // First value is the scalar value
            output.value = tensor_view[0].item<float>();
            
            // Rest is policy
            output.policy.resize(tensor_view.size(0) - 1);
            for (int j = 1; j < tensor_view.size(0); j++) {
                output.policy[j-1] = tensor_view[j].item<float>();
            }
            
            results.push_back(std::move(output));
        }
        
        return results;
    }
};
```

### 3. Dynamic Batch Sizing Based on GPU Profiling

```cpp
class AdaptiveBatcher {
private:
    // Track GPU utilization and performance
    struct BatchStats {
        float throughput_states_per_second;
        float gpu_utilization_percent;
        std::chrono::milliseconds latency;
    };
    
    std::map<size_t, BatchStats> batch_size_stats_;
    size_t current_batch_size_ = 64;  // Start with reasonable default
    
public:
    size_t getOptimalBatchSize() {
        // Get current GPU utilization
        float current_gpu_util = getGpuUtilization();
        
        if (current_gpu_util < 50.0f) {
            // GPU underutilized, increase batch size
            current_batch_size_ = std::min(current_batch_size_ * 2, 
                                         size_t(512));  // Cap at reasonable maximum
        } else if (current_gpu_util > 95.0f && 
                  batch_size_stats_[current_batch_size_].latency.count() > 50) {
            // GPU saturated with high latency, reduce batch size
            current_batch_size_ = std::max(current_batch_size_ / 2, 
                                         size_t(16));  // Maintain reasonable minimum
        }
        
        return current_batch_size_;
    }
    
    void updateBatchStats(size_t batch_size, std::chrono::milliseconds latency, 
                        float states_per_second, float gpu_util) {
        // Update statistics for this batch size
        batch_size_stats_[batch_size].latency = 
            (batch_size_stats_[batch_size].latency.count() * 0.9f + 
             latency.count() * 0.1f);  // Exponential moving average
             
        batch_size_stats_[batch_size].throughput_states_per_second = 
            batch_size_stats_[batch_size].throughput_states_per_second * 0.9f + 
            states_per_second * 0.1f;
            
        batch_size_stats_[batch_size].gpu_utilization_percent = 
            batch_size_stats_[batch_size].gpu_utilization_percent * 0.9f + 
            gpu_util * 0.1f;
    }
    
private:
    float getGpuUtilization() {
        // Use NVML or other GPU monitoring API to get current utilization
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(0, &device);
        
        nvmlUtilization_t utilization;
        nvmlDeviceGetUtilizationRates(device, &utilization);
        
        return static_cast<float>(utilization.gpu);
    }
};
```

By implementing these optimizations, the MCTS engine will achieve significantly higher GPU throughput and more efficient parallelization. The system will adapt to runtime conditions, maintain optimal batch sizes, and minimize CPU-GPU transfer overhead, resulting in much faster and more efficient tree search.
</gpu_throughput_scenario>

----------
----------

# Monte Carlo Tree Search Implementation Overview

The `omoknuni_small` repository implements a multi-threaded **Monte Carlo Tree Search (MCTS)** engine integrated with a neural network for policy/value evaluations. The MCTS code is primarily in `src/mcts` (with headers in `include/mcts`), and it follows an AlphaZero-style search with **leaf parallelization**, **virtual loss**, a **transposition table**, and a **batched neural network evaluator**. Below we dissect the main components, examine their functionality and potential issues, and then propose optimizations for speed, concurrency, and GPU utilization.

## Main MCTS Components

**MCTSEngine** – The central class orchestrating the search (see `MCTSEngine` in `mcts_engine.h`). It holds MCTS parameters (`MCTSSettings`), search statistics (`MCTSStats`), and manages the search tree and worker threads. Key fields include the root node (`root_`), a `TranspositionTable`, and several thread-synchronization primitives. It sets up **specialized worker threads** for different roles (tree traversal, batch accumulation, result distribution) and coordinates their interaction via lock-free queues and atomic flags. The engine’s `search()` method resets any previous search state, initializes the root node, ensures the evaluator thread is running, then spawns the worker threads to perform the specified number of simulations in parallel. The engine uses `active_simulations_` (an atomic counter) to track how many simulations remain and distributes work among threads accordingly.

**MCTSNode** – Represents a node in the search tree, corresponding to a game state. Each node holds a game state (via `IGameState`), a list of child nodes and corresponding move actions, and statistics such as visit count, total value, prior probability, and virtual loss count. Many of these are atomic for thread-safe updates (e.g. `visit_count_`, `value_sum_`, `virtual_loss_count_`). The node provides methods for **selection** and **expansion**: `selectChild()` implements a PUCT formula (using visit counts, total value, and prior, adjusted by virtual loss) to choose the best child during selection, and `expand()` generates all children by applying each legal move to a cloned state. To prevent race conditions, expansion is guarded by a mutex so that only one thread can expand a leaf node at a time. If expansion succeeds, children are initialized with uniform prior probabilities (which will later be updated by the neural network). Each node also supports **backpropagation** via an `update(float value)` that increments the visit count and adds the outcome value to the value sum atomically, and **virtual loss** methods (`addVirtualLoss()`/`removeVirtualLoss()`) to adjust counters during concurrent simulations.

**MCTSEvaluator** – Encapsulates the neural network inference pipeline (see `MCTSEvaluator` in `mcts_evaluator.h`). It manages a dedicated thread that performs batch evaluations on the GPU. The evaluator is constructed with a function handle to the network’s inference (which takes a vector of game states and returns policy and value outputs) and parameters for maximum batch size and waiting timeout. Internally, it maintains a lock-free queue of evaluation requests (each request bundling a node, a cloned game state, and a promise for the network result). When `start()` is called, the evaluator spawns a worker thread running an `evaluationLoop()`. In **normal mode** (single-threaded MCTS), this loop accumulates requests from the queue until reaching a batch size or timeout, then processes a batch in one GPU forward pass. In the current **multi-threaded integration**, the engine uses **external queues**: the evaluator can be pointed to the engine’s batch and result queues (via `setExternalQueues()`) so that it no longer uses its internal request queue. In this mode, the evaluator’s thread simply pulls ready batches from the external `batch_queue_`, runs the network on the collected states, and pushes the results to the `result_queue_`. This design cleanly decouples GPU inference from the MCTS threads: CPU workers produce batches of states, and the evaluator thread handles all GPU interactions, maximizing throughput by processing states in large batches (default max batch size = 64). The evaluator also tracks metrics like total evaluations, average batch size, and latency for performance monitoring.

**TranspositionTable** – A thread-safe hash table that caches nodes by state hash to detect and reuse duplicate game states (transpositions). It stores entries with a **weak pointer to MCTSNode** (avoiding memory leaks by not owning the node) and associated stats (depth, visits). The table is sharded internally for concurrency (using the parallel-hashmap library). The engine queries the table during selection: after selecting a child node, it checks if the child’s state hash exists in the table. If a different node representing the same state is found, the engine can reuse it instead of exploring a redundant branch. This is meant to save effort in games with transpositions or symmetry. The table is cleared at the start of each new search to avoid using stale pointers.

## Parallelization Scheme (Leaf Parallelization)

The MCTS engine employs **leaf parallelization**, meaning multiple simulations (playouts) are run concurrently by different threads. Rather than one long simulation running at a time, threads collaborate on building the search tree. Key aspects of the parallel design:

* **Thread Roles:** On starting a search, the engine spawns one **batch accumulator** thread, one **result distributor** thread, and N **tree traversal** threads (N = `num_threads` setting, default 4). The tree threads perform the Selection and Expansion phases of MCTS; the batch thread collects un-evaluated leaf nodes; the result thread processes neural network outputs and performs Backpropagation. This specialization avoids a general thread pool and instead dedicates threads to pipeline stages for efficiency.

* **Work Distribution:** The engine uses an atomic counter `active_simulations_` to track how many playouts are left to perform. Initially, this is set to the total simulation count (e.g. 800 by default) and is decremented as threads claim work. Each tree traversal thread loops, checking if simulations remain and if workers are active. When work is available, a thread will **claim a batch of simulations** by atomically decrementing `active_simulations_` in chunks (it takes up to \~max(num\_remaining/num\_threads, 16) tasks, capped at 64). This strategy tries to balance load: threads grab simulations in moderately sized batches, reducing frequent contention on the counter while preventing one thread from taking all work at once. After claiming its batch, a thread repeatedly calls `traverseTree(root_)` for each simulation it took.

* **Selection & Virtual Loss:** Within each `traverseTree()` call, a thread performs the selection phase from the root: following the PUCT formula via `selectChild()` until it reaches a leaf (an unexpanded node) or a terminal state. During this descent, **virtual losses** are used to avoid contention: each time a node is selected, the thread increments that node’s `virtual_loss_count_` (and similarly for the child). This temporarily reduces that node’s computed value in other threads’ selection calculations, discouraging others from selecting the same branch until the first thread finishes. Once a thread finishes a simulation (or abandons it), it will remove those virtual losses during backpropagation. Virtual loss thus acts as a lock-free synchronization mechanism ensuring threads explore diverse parts of the tree instead of dogpiling the current best path.

* **Expansion & Evaluation Queueing:** When a thread reaches a leaf node that is non-terminal, it **expands** it (adding child nodes for all legal moves) and immediately **queues it for neural network evaluation** without waiting for the result. The thread clones the game state of the leaf and packages it into a `PendingEvaluation` object containing the state, the pointer to the leaf node, and the path of nodes from root to leaf. This package is pushed into a lock-free **leaf queue** (`leaf_queue_`). Importantly, the thread does *not* perform backpropagation at this time – since the leaf’s value is not yet known, it defers the backup until after the neural network computes the leaf evaluation. (For terminal leaf nodes, no network eval is needed; the thread computes the outcome value directly and calls `backPropagate` immediately in the same thread.)

* **Batching and GPU Inference:** The **batch accumulator thread** continuously gathers pending evaluations from the leaf queue to form batches. It wakes up frequently and tries to dequeue as many `PendingEvaluation` items as possible up to the target batch size (e.g. 64). It uses adaptive heuristics to decide when to submit a batch to the neural network: for example, it will submit immediately if it has a full batch, but it may also submit smaller batches after short time intervals to avoid long delays (e.g. flush after 100ms if >=16 states, after 500ms if >=8, etc.). These time-based triggers ensure that even if the search is not generating states quickly, the GPU still receives work without stalling for too long. Once a batch is ready, the thread moves the collected states into a `BatchInfo` and pushes it into the `batch_queue_`, then notifies the evaluator thread. (The engine uses a `ConcurrentQueue<BatchInfo>` for this; the moodycamel queue is lock-free, but a condition variable `batch_cv_` is also signaled as a backup notification.)

  The evaluator thread (running inside `MCTSEvaluator`) receives these batches. In “external queue” mode, its `processBatch()` function tries to pop one BatchInfo from the external batch queue each iteration. When a batch is obtained, it extracts all the game states, runs the neural network inference on the entire batch in one go (`inference_fn_` returns a vector of `NetworkOutput`), and then pairs each output with its corresponding `PendingEvaluation` to enqueue into the `result_queue_` for post-processing. This design achieves a high GPU throughput: many leaf states are evaluated simultaneously on the GPU, amortizing the overhead of a forward pass over a large batch. It also means CPU threads aren’t idly waiting for the GPU – they can continue selecting and expanding other parts of the tree while inference is in progress.

* **Result Integration (Backpropagation):** The **result distributor thread** consumes completed evaluations from the `result_queue_` and integrates them back into the search tree. It dequeues results in batches (processing, say, up to 32 results at once). For each `(NetworkOutput, PendingEvaluation)` pair, it locates the associated node and path. It then **updates the tree**: the node’s children get their prior probabilities set from the neural network’s policy vector, and the value is backpropagated up the path. Specifically, it calls the node’s `setPriorProbabilities(policy)` to update all child nodes’ `prior_probability_` fields (protected by the node’s expansion mutex for thread safety), and then calls `backPropagate(path, value)` to propagate the value to each node in the path (alternating the sign of the value for each level since the perspective flips with each move). The backpropagation routine removes the virtual losses and updates visit counts and value sums atomically for each node in the path. Finally, the result thread decrements the global `pending_evaluations_` counter to signal that one more evaluation has completed. (This counter was incremented when the leaf was first queued and again when the batch was submitted, as discussed below.) With backprop now done, that simulation is fully complete.

* **Thread Synchronization:** The threads coordinate using a combination of atomic flags and condition variables. The tree workers wait on a condition variable `cv_` when there are no simulations to process or when the engine temporarily deactivates workers. The engine sets an atomic flag `workers_active_` to false when the search is ending, which causes workers to break out of their loops after finishing current tasks. The batch and result threads run mostly in a loop checking their queues; they also break out when a shutdown flag is set and their queue empties. The overall search operation is monitored by a small thread in `MCTSEngine::runSearch()` that periodically checks if `active_simulations_` is zero *and* no pending evaluations remain; once both are true (meaning all simulations have been expanded and all network results processed), it marks the search as complete. This triggers the cleanup: `workers_active_` is set false and all condition variables are notified to unblock any waiting threads. The engine can then safely join the worker threads or reuse them for another search. (Indeed, the code is written to reuse threads across multiple `search()` calls – threads are only created once and then remain alive but idle between searches.)

In summary, the parallel MCTS works as a pipeline: **Tree threads** produce un-evaluated leaf nodes, the **batch thread** aggregates them, the **GPU thread** evaluates them in batches, and the **result thread** updates the tree. This pipeline allows substantial concurrency while keeping each component focused, and the use of atomic counters and lock-free queues minimizes locking overhead in the hot loops.

## Correctness and Performance Issues

While the implementation is sophisticated, several issues and inefficiencies are present upon close inspection. These include potential correctness bugs (e.g. double-counting, race conditions), suboptimal resource usage (idle threads or GPU underutilization), and areas for improvement in thread synchronization. Below we enumerate the key findings:

* **Pending Evaluation Counting and Throttling:** The `pending_evaluations_` atomic is meant to track how many leaf evaluations are in flight (to avoid memory explosion by limiting concurrent simulations to `max_concurrent_simulations`, default 512). However, the current usage is inconsistent and may **double-count**. When a leaf is first queued by a tree thread, `pending_evaluations_` is incremented. Later, when the batch accumulator actually submits a batch of those leaves to the neural net, it increments `pending_evaluations_` *again by the batch size*. Each result processed then decrements the counter by 1. This means each leaf evaluation effectively adds **2** to the counter but only subtracts 1 when done, causing `pending_evaluations_` to potentially remain overstated. For example, if 100 leaves were queued and all processed, the counter might end up 100 too high. This bug can skew the engine’s logic that checks `pending_evaluations_` against `max_concurrent_simulations`. In the batch thread, there is a loop `while (pending_evaluations_ >= settings_.max_concurrent_simulations) sleep` to throttle expansion. Because of double-counting, the engine might perceive more pending work than actually exists and unnecessarily stall the generation of new simulations. This reduces thread utilization and throughput (threads may pause even though the GPU could accept more work). **Correctness Impact:** The double-count doesn’t corrupt the search results per se, but it can harm performance by starving the pipeline. **Optimization:** To fix this, the counting scheme should be made one-to-one (increment once per leaf and decrement once per completed result). For instance, increment only when a leaf is first queued and **do not increment** again on batch submission (or decrement by batch size accordingly). Ensuring accurate counts will keep the throttling logic working as intended – pausing expansion only when the true number of outstanding evaluations hits the cap.

* **Unutilized Worker Tracking:** Similarly, the engine defines an atomic `num_workers_actively_processing_` intended to track how many workers are busy. In `runSearch()` the code waits until this count becomes zero (meaning previous search’s workers finished) before proceeding. However, we find no code that ever increments or uses `num_workers_actively_processing_` in the worker loops – it stays at 0, causing the wait loop to exit immediately regardless of worker state. This appears to be a leftover from a prior design (“Thread pool (removed)”) and does not reflect actual activity. The result is mostly benign (the engine instead relies on the `active_simulations_` and `pending_evaluations_` checks to know when work is done), but it’s misleading and could mask synchronization issues. **Optimization:** Remove or properly implement `num_workers_actively_processing_`. A correct implementation could increment this when a worker picks up a simulation and decrement when it finishes, then allow `runSearch` to block until truly all threads are idle. However, given the current design, it may be simpler to remove it and trust the existing completion conditions.

* **Transposition Table Integration:** The idea of reusing nodes from the transposition table is sound, but the current implementation can lead to **inconsistent tree structure**. When a transposition is found during selection, the code attempts to **swap in** the existing node: it removes the virtual loss from the newly selected child and assigns the `node` pointer to the transposition node (then adds a virtual loss to that). However, it does *not* update the parent’s child list to replace the new node with the transposition node. In other words, suppose thread A expanded node X and created child Y (with move m), but thread B through a different path found an existing node Z (transposition of Y’s state). Thread B will abandon Y and use Z for further selection, but node X’s `children_` still contains Y as the child for move m. Y becomes an orphaned node: it isn’t used in the search anymore, yet it remains in the tree structure (and will never be updated or backpropagated). This can waste memory and, worse, on future selections of move m from X, the engine might mistakenly consider Y again. If Y has no visits (because B never finished a playout on it) but remains in the children list, `selectChild()` might pick it again in another simulation, even though a superior transposition Z was available. This defeats the purpose of the transposition table and could cause duplicate effort. **Correctness Impact:** This is more of a performance bug than a fatal error – the search might simply explore the same state along two different node objects. But it does risk skewing visit counts and value estimates if one state’s statistics get split across two nodes. **Optimization:** To fix this, when a transposition is detected, the engine should *merge* the nodes properly. One approach is to replace the parent’s child pointer with the transposition node (and possibly delete the newly created node Y). However, care must be taken if multiple parents reference the same transposition (which is a complex scenario). Alternatively, the engine could avoid creating the new node in the first place: look up the transposition table *before* expansion and simply attach the existing node as a child. If attaching is problematic, a simpler workaround is to disable re-expansion of a node that was found as a transposition – i.e. if `transposition_table_->get(hash)` returns a node, do not create a duplicate child at all. The current code path is somewhat convoluted; refactoring the selection/transposition logic to cleanly handle node reuse will improve consistency.

* **Duplicate Neural Evaluations (Race Condition):** There is a subtle race possible in the expansion logic. Multiple threads could arrive at the same leaf node concurrently and both decide to expand and evaluate it. The code locks `expand()` so only one actually creates children, but the other thread, upon finding the node no longer a leaf (children now exist), does not re-select a child. In `traverseTree()`, the check for expansion is `if (!leaf->isTerminal() && leaf->isLeaf())`. If Thread A expands the node and enqueues it for evaluation, Thread B might still pass this check before A’s expansion completes (seeing `isLeaf()==true`), then block on the expansion mutex. Once B acquires the lock, it sees children are now present (expansion already done) and returns from `expand()` immediately. At this point, Thread B continues as if expansion happened (because the code does not re-check `isLeaf` after expansion). It proceeds to clone the state and enqueue a `PendingEvaluation` for the *same node*. Now two evaluations for the same state are in the queue. Both will eventually be processed, and the result distributor will apply both, effectively **backpropagating the same node twice**. This double-counts one simulation’s result and also wastes GPU cycles on a redundant inference. The use of virtual loss doesn’t prevent this scenario because the virtual loss is only removed when backpropagating, and here both threads assumed a simulation was needed. **Correctness Impact:** Slight – the search tree will get an extra visit on that node with (likely) the same value, biasing visit counts. If the neural net is deterministic, both outputs are identical, so the net effect is roughly as if one simulation was counted twice. This could skew the tree statistics (e.g. making that node appear slightly more explored than it truly is). **Optimization:** Introduce a flag or mechanism to mark a node as “evaluation pending” so that only one evaluation is scheduled per leaf. For example, an atomic boolean in MCTSNode could indicate a network eval is in progress. The first thread to queue the node would set this flag; subsequent threads reaching the node would see the flag and *not* queue another eval (they could either skip that simulation or wait for the result). This ensures each node is expanded/evaluated exactly once per search. Combined with virtual loss (to penalize the node until the result comes), this prevents duplicate effort.

* **Thread Utilization and Lock Overheads:** The current design already minimizes locking (using mostly atomic ops and lock-free queues), but there are a few areas to refine:

  * The tree traversal threads use a condition variable `cv_` with a short timeout (10ms) to wait for work. Waking up every 10ms incurs some overhead, though in practice this is minor compared to simulation time. A more efficient approach is possible: use a condition variable *without* a timeout, and notify specific worker threads when work becomes available. For instance, after setting `active_simulations_` in `runSearch`, the engine already calls `cv_.notify_all()` to wake all workers. The additional periodic wake-ups might not be necessary. If spurious sleeps are an issue (observed CPU usage when idle), adjusting this waiting strategy could help: e.g. wait indefinitely on `cv_` until either `active_simulations_ > 0` or `shutdown_` becomes true, thereby eliminating the 10ms polling loop. This requires careful handling to avoid deadlock if a notify is missed, but since the code already signals on key events (work added or shutdown), it can be done.
  * **Expansion Mutex:** Each node’s expansion uses a mutex which ensures thread safety, but it can become a bottleneck if many threads happen to expand nodes on the same branch in quick succession. The impact is usually low (different simulations typically expand different parts of the tree). Still, one could consider reducing this contention by using finer-grained atomic steps: e.g., an atomic flag for “expansion in progress” could prevent double expansion without a full lock. However, because expansion can involve creating many child objects, the mutex is a reasonable choice for simplicity and is held only briefly. There’s no strong evidence of lock contention here unless the game has very shallow depth causing many threads to bang on the root expansion at once (but the root is expanded once with all legal moves upfront, thereafter no contention).
  * **Result Updating:** The result distributor locks each node’s expansion mutex when setting prior probabilities (via `MCTSNode::setPriorProbabilities`). This ensures children list and prior vector are consistent during update. This could serialize updates if many results come in at once for siblings of the same node, but typically network batches contain leaves from various parts of the tree. The critical section is small and unlikely to be a performance issue. We should keep this for correctness (to avoid a race if another thread tries to expand or read the same node concurrently).

* **GPU Throughput and Batching Efficiency:** The batched inference mechanism is one of the strengths of this implementation; however, we should examine if it’s always utilized effectively:

  * The default batch size is 64, but in many cases, the search may not generate 64 parallel leaf evaluations fast enough to fill the batch before the timeout. The code addresses this by using a relatively short batch timeout (50ms) and by flushing smaller batches (≥16 after 100ms, etc.). Still, if the CPU threads are slow (due to heavy game state logic or fewer threads), the GPU might be fed many suboptimal small batches (e.g. 4 or 8 states at a time). This can underutilize the GPU’s parallelism. **Potential Bottleneck:** If the GPU finishes inference much faster than the CPU can generate new states, the GPU will be underutilized (essentially idle waiting for work). One scenario: at game start, the root has many moves; expanding them yields a flurry of leaf evals that fill big batches. But later in the search, many simulations might hit already-expanded nodes (transpositions) or spend time backpropagating, slowing the rate of new leaf generation – the batch thread might then send smaller batches. The evaluator thread currently **polls** the batch queue in a tight 1ms loop, which ensures low latency, but also means it’s constantly spinning. This is somewhat wasteful – it could instead block on a condition variable signaled by `batch_accumulator_worker` when a batch is available (the code actually calls `batch_cv_.notify_one()` on enqueue, but the evaluator doesn’t explicitly wait on it, opting for the polling loop).
  * **Centralized vs. Decentralized Batching:** The current architecture uses a separate batch accumulator thread. An alternative design could let the evaluator thread itself dequeue directly from `leaf_queue_` and form batches. This would remove one queue and thread from the pipeline, reducing context switches and data movement. As it stands, a `PendingEvaluation` goes from `leaf_queue_` -> `current_batch` (vector) -> `BatchInfo` object -> `batch_queue_` -> evaluator thread, then the states are moved again into a new vector for inference. That’s multiple moves/copies of pointers. While each move is not heavy (we’re moving pointers to game states), it’s extra complexity. A single thread could conceivably perform the dequeue and inference steps together. The advantage of the extra batch thread might be that it can continue accumulating the *next* batch while the current batch is being processed on GPU, partially overlapping CPU preparation with GPU compute. However, in practice, preparing a batch of at most 64 states is very fast relative to a GPU forward pass, so this overlap benefit is minimal. **Optimization:** Consider merging the batch accumulator’s functionality into the evaluator thread – i.e., let the evaluator thread pull directly from `leaf_queue_` until either a timeout or batch size is reached, then run the network and immediately process or dispatch results. This would simplify the pipeline (fewer queues, less risk of queue backlog) and potentially reduce latency. The code already has similar logic in non-external mode (using `collectBatch()` on its own request queue), which could be adapted for the external scenario.
  * **Batch Size Adaptation:** The code sets a `min_batch_size_` to 75% of the max (e.g. 48 if max 64) to decide when to stop waiting for more requests. It might be beneficial to allow more dynamic adjustment. For instance, if the GPU is very fast (e.g. using a high-end card on a small model), one could use a larger batch size (128 or more) without hurting per-batch latency much, thus increasing throughput. Conversely, on a slower GPU or if latency is critical, one might prefer smaller batches for quicker responses. Currently, these values are constants or fixed settings. **Optimization:** Expose the batch size and timeout as tunable parameters (they already are in `MCTSSettings`), and potentially implement logic to adjust them on the fly. For example, measure actual average batch latency and if the GPU isn’t near saturated (small batches processed quickly), the engine could increase the batch size in subsequent searches or if pending evaluations consistently back up, it could shorten the timeout to push work out faster. This is a complex optimization, but noting it could guide performance tuning.

* **Memory Usage and Leaks:** Thanks to smart pointers and careful design, outright memory leaks are not obvious in this code. The transposition table using weak\_ptr avoids preventing node cleanup, and the search resets the tree between runs. One area to watch is that large numbers of cloned states could consume memory if `max_concurrent_simulations` is high – but the setting (512) is meant to cap that. Each `PendingEvaluation` holds a unique\_ptr to a game state clone; if 512 are queued simultaneously, that’s 512 game states in memory plus the original tree states. If the game state is large, this is heavy. However, once a batch is processed, those states are moved into the local `states` vector for inference and then destroyed after inference completes. The result distributor no longer needs the state (just node and path), so memory is freed in a timely manner. **Potential leak:** The earlier-mentioned orphaned node in the transposition scenario (Y) would hold a state pointer and children, and if never used or freed until search end, that memory is stuck during the search. Since the entire tree is destroyed at search end, it’s not a permanent leak but could be a temporary waste. Reducing such orphan creation (via proper transposition handling) will also eliminate this minor memory waste.

* **Logging Overhead:** The code is currently instrumented with many `std::cout` debug messages (guarded partially by `MCTS_DEBUG` macros). For example, every worker thread prints start/stop messages, and there are per-second status prints in the search monitor thread. In a performance setting, excessive I/O can slow things down. These should be compiled out or throttled in production builds (likely the intention with the debug macro). Ensuring that the debug logging level is configurable or disabled in release will avoid any unintended slowdowns.

In summary, while the MCTS implementation is robust, addressing the above issues will improve its correctness and efficiency. The most critical are the double-counting of pending evaluations (which can throttle parallelism unnecessarily) and the race leading to duplicate evaluations of the same node. Fixing those will both speed up the search and ensure the statistics (visits, values) are accurate. The transposition integration needs refinement to truly reap the benefits of state reuse. Additionally, some architectural streamlining (merging threads or using proper blocking instead of active waiting) can reduce overhead.

## Optimizations and Recommendations

To enhance the MCTS engine’s performance and thread safety, we propose the following optimization strategies:

**1. Fix Concurrency Bugs and Ensure One Evaluation per Leaf:** As a top priority, implement measures to prevent duplicate evaluations of the same node. Introduce an atomic “in-flight” flag in `MCTSNode` that is checked and set when a leaf is queued for evaluation. If a second thread reaches the same node before the first result returns, it will see the flag and skip queuing another evaluation (it could simply back out and try a different simulation). This change will eliminate the race condition causing double evaluation and double backpropagation for one node. It also implicitly reduces wasted GPU work and keeps the search tree statistics consistent (each simulation contributes exactly once). Alongside this, correct the `pending_evaluations_` accounting so that it truly reflects the number of outstanding evals. This likely means removing the increment before batch enqueue (or the initial increment) and only incrementing once per leaf, then decrementing once per result. With accurate counts, the batch thread’s throttle (`while (pending_evaluations_ >= max_concurrent)`) will function properly to avoid oversubscription without idling too early.

**2. Improve Transposition Handling:** Refactor how the engine uses the transposition table during selection. A recommended approach is: **check the transposition table *before* creating a new child node.** In the expansion phase, for each legal move, look up the resulting state’s hash; if an existing node is found, use that instead of allocating a new `MCTSNode`. You can still create a new child entry in the parent’s children list, but point it to the existing node (and perhaps increase its reference count via shared\_ptr). If the existing node has already been expanded and has its own children, this effectively prunes the new branch and links the tree graphs together. This way, no “orphan” duplicate node is ever created. If modifying the children list like this is tricky, at least modify the current logic to remove the placeholder node when a transposition is found. For example, in `selectLeafNode`, after `node = transposition`, one could pop or mark the previously selected child from its parent’s vector to avoid keeping an unused node around. These adjustments will ensure that each game state in a search corresponds to at most one `MCTSNode`, concentrating visit counts and values correctly and saving memory/cycles.

**3. Streamline Threading and Queues:** Consider collapsing the batch accumulator and evaluator threads into a single entity. The evaluator thread can handle accumulating leaf states into a batch (with the same logic currently in `batchAccumulatorWorker`) and then run the network. This simplifies synchronization – instead of passing data through `leaf_queue_ -> batch_queue_ -> result_queue_`, you would have `leaf_queue_ -> result_queue_` with one thread in between. The sequence could be: evaluator thread waits until either a timeout expires or the batch size is reached in `leaf_queue_`, grabs all available items up to the max, processes them, and directly pushes results to `result_queue_`. This removes one intermediate queue and one context switch. It also avoids the subtlety of double-incrementing pending counts when moving from leaf to batch queue. The code already polls `leaf_queue_` in small bursts (10ms loop); doing so in the evaluator thread is fine. If implementing this, ensure the evaluator thread also respects `max_concurrent_simulations` (i.e. if too many are pending, maybe it waits) – but with accurate pending counts and only one increment per leaf, that control might even be simpler (the evaluator can just limit how many it pulls).

**4. Use Condition Variables for Idle Threads:** Reduce active spinning by leveraging condition variables more fully:

* For the evaluator thread, rather than sleeping 1ms in a loop when no batch is available, use the `batch_cv_`. The batch accumulator (if kept) or tree threads can notify `batch_cv_` when they enqueue the first leaf into an empty queue. The evaluator thread then waits on `batch_cv_` with a timeout equal to the batch timeout. This way, if work arrives immediately, it wakes immediately; if not, it wakes at least at the timeout to process partial batch. This change cuts down unnecessary wake-ups.
* Similarly, for result processing, you could have the evaluator thread signal a `result_cv_` when it enqueues results, and the result distributor thread waits on it instead of polling `result_queue_` continuously. Given the result thread already sleeps only 1ms when idle, the overhead is minor, but it’s an easy win to eliminate even that.
* For tree traversal threads, as noted, you can potentially remove the 10ms periodic wake. They already do `cv_.wait_for(lock, 100ms, predicate)` in a loop. It might be better to use `cv_.wait(lock, predicate)` (no timeout) and rely on `notify_all` calls when new simulations are added or workers reactivated. The code does call `cv_.notify_all()` in all relevant places (`runSearch` when starting and stopping work, also periodically every \~300ms in the wait loop as a failsafe). If we ensure a notification is sent whenever `active_simulations_` transitions from 0 to >0 or `workers_active_` becomes true, then a pure wait (no timeout) is safe. This would eliminate needless wake-ups of sleeping threads.

**5. Tune Batching Parameters:** To maximize GPU utilization, consider making the batch size and timeout adaptive or at least configurable for different hardware. For example, on a powerful GPU, using a batch size of 128 might significantly increase throughput. The current `MCTSSettings` allows setting `batch_size` and `batch_timeout` – ensure these truly propagate (they do in the evaluator construction). We might recommend experimentation with larger batch sizes if the CPU can support it. Another strategy is to increase `num_threads` for CPU if the GPU isn’t saturated – more CPU threads would generate more leaves concurrently, feeding bigger batches. However, raising CPU threads has diminishing returns due to tree contention. A better approach for throughput is to run multiple searches in parallel (e.g., self-play with multiple games at once sharing the same neural network thread). The current design could support that by having multiple MCTSEngine instances share a single evaluator thread/queue, but that would require refactoring (such as a global inference service). As an immediate improvement, focusing on one search at a time, we ensure we are at least hitting the 64 batch size frequently. Profiling could reveal if batches are often smaller than desired – if so, one could increase `max_concurrent_simulations` and `num_threads` to push more parallelism. **Atomic batch assembly:** if merging threads, ensure to still enforce a minimum batch size (the code uses `MIN_BATCH_SIZE = 16` to avoid too-small batches). This threshold could be made dynamic – e.g., if GPU utilization is low, increase MIN\_BATCH\_SIZE to wait for more states.

**6. Efficient Game State Handling:** Cloning game states for every leaf is a cost. If profiling shows this to be a bottleneck, consider optimizing how states are passed to the neural network. If `IGameState` provides a method to extract features (like a tensor or array representation), the engine could maintain a pre-allocated buffer for state features and fill it without cloning the entire object. Cloning might be doing deep copies of game boards; this is memory-intensive. One idea: use a **move semantics** approach where the game state of a leaf node can be std::moved directly into the evaluation request (since after expansion, the leaf’s state won’t be needed until backprop, and even then only read). However, because the node still needs its state for potential future expansions, we can’t steal it entirely. A compromise is to have a lighter representation for the neural input. For example, have `IGameState` implement an interface `encodeTensor()` that writes the state into a tensor (or vector of floats). Then the evaluator can call this on the fly instead of requiring a full object clone. This would cut down memory allocations and possibly speed up state preparation. That said, this is a more involved change requiring modifications to the game state and network input pipeline, and the benefit depends on how heavy a state clone is. Given the current code, the simpler immediate fixes are in concurrency and synchronization; game state optimization is a secondary consideration.

**7. Logging and Debug Controls:** Ensure that all the debug printing is disabled or minimized in performance runs. The numerous `std::cout` calls inside loops (e.g., printing status every second, or even per simulation in some debug modes) can severely degrade performance. Utilizing a proper logging library (the project mentions spdlog in comments) with log levels would be ideal. For now, wrapping these in `#ifdef MCTS_DEBUG` (which is on in the code by default with `#define MCTS_DEBUG 1` at top of `mcts_engine.cpp`) should be switched off (`0`) for normal operation. This will ensure the search threads and others aren’t spending time in I/O, which not only slows them but can also disturb timing (I/O calls release CPU and can affect thread scheduling).

**8. Multi-Game Parallelism (Future Improvement):** While not in the original scope, one architectural recommendation for maximizing GPU usage is to allow the evaluator to batch evaluations from *multiple* MCTS searches at once. In self-play training scenarios, often many games are played in parallel to better utilize the GPU. The current design can be extended such that a single `MCTSEvaluator` serves multiple `MCTSEngine` instances: all their leaf evaluations go into one central queue and get batch-processed together. This would require making `leaf_queue_` a global concurrent queue and tagging requests with which engine they belong to, and then distributing results back. The benefit would be larger batches and higher GPU occupancy, especially when each game’s search might not have enough leaves to fill a batch by itself. This is a more complex change but worth noting for scalability: it’s essentially a *central inference server* idea. The given code already separates inference into its own module, which is a good starting point for this enhancement.

**9. Backpropagation Optimization:** Currently, backpropagation is done in the result thread, sequentially for each result. This is usually fine (backprop is fast, just a few atomic adds per node up the path). However, if the latency of updating the tree becomes a concern (say, if policy vectors are very large or paths are long), one could parallelize backprop or offload some of it. For example, the result thread could push a task to a worker to do the actual backprop if needed. Given typical game depths, this is unlikely to be a bottleneck, so the simplicity of doing it directly is preferable. We mention this only for completeness.

**10. Enhanced Thread Coordination:** Remove any unused synchronization constructs to avoid confusion. For example, if we decide to remove `num_workers_actively_processing_`, also remove its associated condition waits in `runSearch`. Likewise, if the `cv_mutex_destroyed_` flags (there to signal to threads that the engine is tearing down) aren’t needed with proper thread join logic, they can be removed to streamline the code. The engine’s destructor already attempts to join all threads cleanly, so with correct notifications and flags, threads should exit normally. Keeping the codebase clean of vestigial sync logic will make maintenance easier and reduce the chance of overlooked corner cases.

## Prioritized TODO List

1. **Resolve Pending Evaluation Count Bug:** Modify the increment/decrement logic for `pending_evaluations_` so that each evaluation is counted exactly once. Remove the double increment (at leaf queueing and batch submission) and ensure one decrement per completed result. This will fix the premature throttling of simulations and improve parallel utilization.

2. **Prevent Duplicate Leaf Evaluations:** Implement a mechanism (e.g. an atomic flag in `MCTSNode` or a shared data structure) to mark nodes that have an evaluation in progress. Use this to prevent multiple threads from scheduling the same node for neural network inference. This addresses the race condition where two threads expand the same node concurrently and both enqueue it for evaluation. The first thread to mark it should proceed, others should skip or wait.

3. **Improve Transposition Node Merging:** Refactor the selection logic to properly integrate transpositions. Before creating a new child node for a move, check the transposition table. If an existing node is found:

   * Option A: Use it directly as the child (attach to the parent’s children list) instead of creating a new node.
   * Option B: If a new node was already created, replace or remove it and use the existing one for the remainder of the simulation.
   * Ensure that the parent and child pointers are consistent and no duplicate nodes for the same state remain in the tree. Test this on a scenario with known transpositions to verify that visits accumulate in one node.

4. **Unify Batch Accumulation with Evaluation Thread:** Simplify the pipeline by merging the batch accumulator and evaluator roles. Let the evaluator thread fetch states from `leaf_queue_` and form batches internally. Eliminate the intermediate `batch_queue_` if possible. This change will remove one thread and associated context switches, and it naturally solves the double-count issue (since there’s no second queue to “submit” to). It will also make the code easier to follow.

5. **Use Condition Variables to Eliminate Busy-Wait Loops:** Adjust thread waiting logic to be event-driven:

   * Tree threads: rely on `cv_.notify_all()` and remove the fixed 10ms wake-up cycle. They should sleep until signaled that new simulations are available or a shutdown is issued.
   * Evaluator thread: wait on `batch_cv_` when no batch is ready, instead of polling every 1ms. Wake it when batch queue (or leaf queue, if unified) transitions from empty to non-empty.
   * Result thread: similar approach with a `result_cv_` or reuse `batch_cv_` (since results come shortly after batches). Or simply sleep a bit longer when idle, since 1ms polling is extremely frequent.
     These changes reduce CPU usage when threads are idle and improve overall efficiency (important in self-play where many MCTSEngines might be running on one machine).

6. **Tune and Expose Parallelism Parameters:** Re-evaluate `MCTSSettings.num_threads`, `batch_size`, and `batch_timeout` for your deployment scenario. For instance, if the GPU is underutilized, consider increasing `num_threads` (more CPU simulations in parallel) or increasing `batch_size` beyond 64. Monitor the `avg_batch_size` and `avg_batch_latency` stats (already tracked in `MCTSStats`) to guide this tuning. Make sure these settings are easily configurable (perhaps via a config file or CLI) so you can adjust without recompiling. The goal is to find a balance where the GPU is fed large batches consistently, but not so large that it introduces too much latency in the search.

7. **Optimize GameState Cloning (if needed):** Profiling might show that `state->clone()` is consuming significant time (especially if the game state is complex). If so, consider optimizing this by implementing a more direct state-to-network input conversion. In the short term, one micro-optimization is to avoid cloning the state twice for the same node in different threads (which will be solved by item 2 above). In the longer term, one might redesign `NeuralNetwork::inference` to accept a lightweight representation of states (like already-serialized tensors). This is a more involved change but could yield improvements in environments where state cloning is expensive.

8. **Clean Up Synchronization Artifacts:** Remove or fix any unused thread coordination variables. For example, if after the above changes `num_workers_actively_processing_` is still unused, eliminate it to avoid confusion. Similarly, the boolean flags `cv_mutex_destroyed_`, `batch_mutex_destroyed_`, etc., which are set on destructor to break out of waits, may not be needed if threads are correctly woken and joined. Ensure that all threads are properly joined or detached on engine destruction to avoid any potential hangs on program exit.

9. **Logging Level Control:** Switch off the verbose logging in performance runs. Use the `MCTS_DEBUG` flag or a runtime log level to silence debug output. This will remove the overhead of I/O in the search loop and worker threads. If detailed profiling is needed, consider logging aggregated stats rather than per-iteration messages (for example, print one summary at the end of search with total nodes, batches, etc., which is already available in `last_stats_`).

10. **Testing and Validation:** After making these changes, rigorously test the MCTS engine on known scenarios. Verify that search results (e.g., chosen moves and values) remain correct and that performance metrics improve (more simulations per second, larger average batch sizes, etc.). Pay attention to edge cases like game termination (ensure threads exit promptly when a terminal state is input to `search`) and search cancellation (not explicitly present now, but if a stop condition is introduced). Also test transposition-heavy games to ensure the reuse logic works without crashes or incorrect double counting.

By addressing the above, the MCTS implementation will become more **efficient** (through better parallel work distribution and GPU utilization) and more **robust** (through elimination of race conditions and cleaner thread synchronization). The end result should be higher throughput of simulations, allowing deeper searches or more games in the same amount of time, and a stable, deadlock-free execution even under heavy parallel load. The architecture – with a centralized batch evaluator – will be well-suited for scaling, possibly even to multiple concurrent games or multiple GPUs in the future, if needed.

----------
----------

## MCTS Implementation Analysis

Below is a detailed breakdown focusing on correctness, potential improvements, parallelization, and GPU throughput.

### \<todo\_list\>

Here is a list of identified issues and tasks for improvement:

  * **High Priority:**

      * **Review and Simplify `MCTSEngine` Thread Management:**
          * Thoroughly verify the complex shutdown logic in `~MCTSEngine()` for potential race conditions, deadlocks, or resource leaks. The multiple `notify_all()` calls and sleeps suggest potential fragility.
          * Simplify the move constructor and move assignment operator for `MCTSEngine`. Joining threads from the `other` object during a move is complex and can be problematic. Consider alternative approaches for resource transfer if the engine is meant to be movable after being active.
          * Ensure liveness of all worker threads (`treeTraversalWorker`, `batchAccumulatorWorker`, `resultDistributorWorker`) and the main search loop. Verify conditions that could lead to stuck threads, especially interactions involving `active_simulations_`, `pending_evaluations_`, `workers_active_`, and various condition variables.
      * **Clarify `EvaluationRequest` Move Semantics:** Investigate the comment in the `EvaluationRequest` move constructor/assignment regarding not nullifying `other.node`. If `std::shared_ptr other.node` is still needed by other threads post-move from `other`, this indicates a potentially non-standard use of move semantics or a complex sharing model that needs to be robustly handled or clarified.
      * **Ensure Robust Error Propagation:** Worker threads currently catch and log exceptions. Implement a mechanism for critical errors in worker threads to be propagated to the main `MCTSEngine` control flow to allow for more graceful handling or abortion of the search.

  * **Medium Priority:**

      * **`MCTSEvaluator` Queue Logic:**
          * If `MCTSEvaluator` is primarily intended to be used with external queues provided by `MCTSEngine`, consider streamlining or removing its internal `request_queue_` and associated `collectBatch` logic to reduce complexity. The `evaluationLoop` seems to be an alternative or older loop compared to `processBatches`.
      * **Transposition Table (`TT`) Locking and `get` Behavior:**
          * Review the `std::mutex` within `TranspositionEntry`. Using `std::try_to_lock` in `TranspositionTable::get` and treating a failed lock attempt as a miss might increase the miss rate unnecessarily if contention is transient. Evaluate if a brief wait or a different locking strategy could be beneficial, or if `phmap`'s concurrency features can be leveraged more directly for entry updates.
      * **State Cloning Overhead:** Profile the `IGameState::clone()` method. If it proves tobe a significant bottleneck, consider optimizations such as memory pooling for game state objects or copy-on-write semantics if applicable to the game state's structure. State cloning occurs in `MCTSEngine::runSearch` for the root and in `MCTSEngine::traverseTree` for leaf evaluation.
      * **Root Node Initialization and TT Interaction:** The order of TT clearing and root reset in `MCTSEngine::search` and `MCTSEngine::runSearch` needs to be failsafe. Ensure that deleting the old tree (`root_.reset()`) doesn't invalidate `weak_ptr`s in the TT in a way that could cause issues before they are properly cleared or expire. The current approach of clearing TT then resetting the root seems generally correct.

  * **Low Priority:**

      * **`MCTSNode::expansion_mutex_` Scope:** Verify if the `expansion_mutex_` in `MCTSNode::setPriorProbabilities` is optimal. If children are guaranteed to exist and not be structurally modified when priors are set, this lock might be contendable if many threads update priors on recently expanded nodes.
      * **TT `enforceCapacityLimit` Efficiency:** The current sampling and sorting mechanism in `TranspositionTable::enforceCapacityLimit` is reasonable but could be profiled. If it becomes a bottleneck, simpler eviction strategies (e.g., random eviction) could be considered.
      * **Configuration of TT Shards:** The `num_shards_param` in `TranspositionTable` constructor is marked as unused, and `entries_.subcnt()` is removed. Confirm that reliance on `phmap`'s internal parallelism is sufficient and intended.

### \<optimization\_scheme\>

Here's a suggested optimization scheme and fixes for the current code:

1.  **Simplify `MCTSEngine` Thread Lifecycle and State Management:**

      * **Reduce Shutdown Complexity:** Refactor the `MCTSEngine` destructor and move operations. Aim for a clear, sequential shutdown:
        1.  Signal all workers to stop (e.g., set `shutdown_` and `workers_active_` to false).
        2.  Notify all condition variables to wake up any waiting threads.
        3.  Stop the `MCTSEvaluator` first to prevent new work.
        4.  Clear work queues (`leaf_queue_`, `batch_queue_`, `result_queue_`) to allow workers to exit loops dependent on queue contents. Ensure promises in `PendingEvaluation` (if any remain) are fulfilled to avoid client hangs.
        5.  Join all worker threads (`tree_traversal_workers_`, `batch_accumulator_worker_`, `result_distributor_worker_`). Use `std::thread::join()` directly or the existing `join_with_timeout` with sufficient timeout.
      * **Move Operations:** For `MCTSEngine` move constructor/assignment, if the engine is intended to be moved after threads have been active, ensure the `other` engine is fully stopped and its resources (especially threads) are properly released before moving members. A simpler approach might be to disallow moving an active engine or define move only for engines in a pre-start or fully stopped state.

2.  **Refine `MCTSEvaluator` for Clarity:**

      * **Primary External Queue Usage:** If the main operational mode involves `MCTSEngine` providing external queues to `MCTSEvaluator`, make this explicit. The `MCTSEvaluator::processBatches` method seems to correctly handle this path. The `evaluationLoop` and internal `collectBatch` / `processBatch(std::vector<EvaluationRequest>& batch)` might be legacy or for a different use case. If so, clearly document or separate this logic.
      * **Adaptive Batching Centralization:** The `batchAccumulatorWorker` in `MCTSEngine` implements adaptive batching logic (batching based on size and timeout). Ensure this is the sole place for such sophisticated batch timing logic to avoid conflicts if `MCTSEvaluator::collectBatch` has similar goals.

3.  **Strengthen `MCTSNode` Operations:**

      * **`setPriorProbabilities`:** The `expansion_mutex_` is used here. If priors are updated after expansion is complete and children are stable, evaluate if a more fine-grained lock or even atomic updates (if policy structure allows) could reduce contention. However, safety is paramount.

4.  **Transposition Table Enhancements:**

      * **`get` Operation:** Instead of `std::try_to_lock` in `TranspositionTable::get` resulting in an immediate miss, consider a very brief spin/wait if the lock is contended, or profile to see if this is a real issue. The current approach prioritizes avoiding waits over maximizing hit rate under contention.
      * **`store` Operation:** The logic to update existing entries or insert new ones if the existing one is invalid seems largely correct. Ensure that the `TranspositionEntry::mutex` correctly serializes updates to the `node`, `depth`, and `visits` fields.

5.  **Memory and State Management:**

      * **Game State Cloning:** If `IGameState::clone()` is expensive, investigate techniques like memory pools or optimized diff-based cloning if the game rules allow.
      * **`EvaluationRequest` Move Semantics:** Correct the `EvaluationRequest` move constructor/assignment. After moving from `other`, `other.state` should be `nullptr` (as it is `std::unique_ptr`). For `other.node` (a `std::shared_ptr`), standard move behavior means `other.node` remains valid but its pointed-to object is now shared with the new request; `other.node` itself is not nullified by `std::move` but could be explicitly reset if the intent is to fully transfer unique responsibility in this context. The comment about not nullifying it because other threads might need it is confusing and needs review against actual usage patterns.

### \<parallelization\_improvements\>

The current implementation already employs leaf parallelization with dedicated threads for tree traversal, batch accumulation, and result distribution. Here are further recommendations:

1.  **Lock Contention Minimization:**

      * **`MCTSNode::expansion_mutex_`:** This is a key mutex. While necessary to protect the `children_` vector during its modification, high contention here would serialize node expansions. The impact depends on how often multiple threads attempt to expand the *exact same* `MCTSNode` instance simultaneously (after TT hits). If this is rare, it is less of an issue.
      * **`TranspositionTable::TranspositionEntry::mutex`:** As discussed, the per-entry mutex is fine-grained. The main concern is the overhead if entries are numerous and small, and locks are frequently taken. `phmap` itself is designed for concurrent accesses, so the additional mutex should only protect the mutable fields of `TranspositionEntry` if `phmap`'s operations don't cover the required atomicity for multi-field updates.
      * **`MCTSEngine` Condition Variables:** The engine uses several condition variables (`cv_`, `batch_cv_`, `result_cv_`). Ensure predicates are minimal and notifications are precise to avoid spurious wakeups or excessive contention on their associated mutexes.

2.  **Task Granularity and Scheduling:**

      * **`treeTraversalWorker`:** Each worker performs independent tree traversals. This is good. The claiming of a "batch" of simulations by each worker (`claimed = active_simulations_.compare_exchange_weak(...)`) seems to be an attempt to give a chunk of work to each thread, but then it iterates one by one. This is fine as traversals are independent.
      * **`batchAccumulatorWorker`:** This worker plays a crucial role in preparing batches for the GPU. Its adaptive logic (waiting for `OPTIMAL_BATCH_SIZE` or for a timeout with `MIN_BATCH_SIZE`) is important for balancing latency and throughput. Ensure the timeout values are well-tuned.

3.  **Memory Issues and False Sharing:**

      * **`MCTSNode` Atomics:** Members like `visit_count_`, `value_sum_`, `virtual_loss_count_` are atomic. If multiple such atomics from different nodes frequently accessed by the same thread fall into the same cache line, false sharing could occur. This is less likely with `std::shared_ptr<MCTSNode>` allocations (nodes are separate heap objects) but worth keeping in mind for very high-performance scenarios. Padding might be an extreme solution if this becomes a proven issue.
      * **`std::atomic<float>`:** While `std::atomic<float>` operations like `Workspace_add` might not be lock-free on all architectures (potentially falling back to internal locks), the `compare_exchange_strong` used for `value_sum_` in `MCTSNode::update` is a standard way to achieve atomic updates for types not directly supported by `Workspace_add`.

4.  **Using Atomic Variables Effectively:**

      * **Memory Ordering:** The code already uses various memory orders (`std::memory_order_acquire`, `std::memory_order_release`, `std::memory_order_acq_rel`, `std::memory_order_relaxed`). This demonstrates an understanding of the C++ memory model. It is crucial that these are correctly used:
          * `release` operations ensure prior writes are visible to other threads doing an `acquire` on the same atomic.
          * `acquire` operations ensure subsequent reads see writes from other threads that did a `release`.
          * `acq_rel` combines both for operations like `Workspace_add` or `compare_exchange`.
          * `relaxed` can be used for simple counters where synchronization is handled by other means, but must be used cautiously.
      * **Minimize `seq_cst`:** Avoid the default `std::memory_order_seq_cst` for atomics where a weaker (but correct) ordering suffices, as `seq_cst` can be more expensive. The code seems to be explicit about orders, which is good.
      * **Example - `MCTSNode::update`:**
        ```cpp
        // visit_count_ uses fetch_add with acq_rel, which is appropriate.
        visit_count_.fetch_add(1, std::memory_order_acq_rel);

        // value_sum_ uses compare_exchange_strong with acq_rel for success and acquire for failure.
        // This ensures that the read (current) and the conditional write (desired) are properly synchronized.
        float current = value_sum_.load(std::memory_order_acquire); // Acquire for the initial read
        float desired;
        do {
            desired = current + value;
        } while (!value_sum_.compare_exchange_strong(current, desired,
                                                    std::memory_order_acq_rel, // If successful
                                                    std::memory_order_acquire)); // If fails, current is updated, need acquire
        ```
        This pattern for `value_sum_` is robust.

5.  **Deadlocks and Livelocks:**

      * **Circular Waits:** Analyze dependencies between locks and condition variables. For example, ensure no path where Thread A holds Lock L1 and waits for CV1, while Thread B holds Lock L2 (associated with CV1) and tries to acquire L1.
      * **Producer-Consumer Coordination:** The `leaf_queue_`, `batch_queue_`, and `result_queue_` are central.
          * `treeTraversalWorker` produces for `leaf_queue_`.
          * `batchAccumulatorWorker` consumes from `leaf_queue_` and produces for `batch_queue_`.
          * `MCTSEvaluator` (via its thread running `processBatches` in external mode) consumes from `batch_queue_` and produces for `result_queue_`.
          * `resultDistributorWorker` consumes from `result_queue_`.
            Ensure that consumers are always eventually woken up if there's work, and producers do not block indefinitely if queues are momentarily full (though `moodycamel::ConcurrentQueue` handles dynamic sizing).
      * **Shutdown Signal Propagation:** The `shutdown_` flag must reliably propagate to all loops and condition variable waits to ensure threads terminate. The current multi-notify and sleep approach in `~MCTSEngine` hints at the complexity of achieving this reliably.

### \<gpu\_throughput\_scenario\>

To increase GPU throughput for high-speed inference, focusing on fast tensor collection from CPU and efficient transfer to GPU:

**Scenario: Centralized Batched Inference with Pinned Memory and CUDA Streams**

1.  **Data Structures for Tensor Collection (CPU-side):**

      * The `MCTSEngine::batchAccumulatorWorker` is responsible for collecting `PendingEvaluation` objects. Each contains a `std::unique_ptr<core::IGameState> state`.
      * Instead of collecting `IGameState` objects directly into the batch sent to the `MCTSEvaluator`, the `batchAccumulatorWorker` (or a subsequent step before GPU transfer) should convert these game states into their tensor representations (e.g., a flat `std::vector<float>` or a more structured representation suitable for direct `memcpy` to GPU).
      * **Pinned Memory:** Allocate large, reusable CPU buffers for these tensors using pinned (page-locked) memory (e.g., via `cudaHostAlloc` or PyTorch's equivalent for tensors like `tensor.pin_memory()`). This allows for faster DMA transfers to the GPU.

2.  **Tensor Collection in `batchAccumulatorWorker`:**

      * As `PendingEvaluation` items are dequeued from `leaf_queue_`, the worker extracts the `state`.
      * It then calls a function `gameStateToFloatTensor(const core::IGameState& s, float* buffer_offset)` which converts the game state directly into a pre-allocated pinned memory buffer at a specific offset. This function needs to be highly optimized.
      * The worker collects metadata alongside (e.g., pointers to the original `PendingEvaluation` items or their IDs) to associate results back.
      * Once a batch of tensor data is ready in the pinned CPU buffer (either batch size met or timeout), this buffer (or a descriptor of it) is sent to the `MCTSEvaluator` via the `batch_queue_`.

3.  **GPU Transfer and Inference (`MCTSEvaluator`'s Thread):**

      * The `MCTSEvaluator`'s thread receives the batch of tensor data (residing in pinned CPU memory) and its size.
      * **CUDA Streams:** Use multiple CUDA streams for overlapping data transfer and kernel execution.
          * Create a CUDA tensor (e.g., `torch::Tensor`) directly mapping or copying from the pinned CPU buffer to a GPU tensor. If using PyTorch, creating a tensor from pinned memory and then calling `.to(device, non_blocking=true)` initiates an asynchronous transfer.
          * `cudaMemcpyAsync` can be used to transfer data from pinned CPU memory to GPU memory asynchronously on a specific CUDA stream.
          * Launch the neural network inference kernel(s) on the same CUDA stream, or a different one if there are multiple independent parts of the network.
          * Asynchronously copy results (policy and value tensors) back from GPU to pinned CPU memory using `cudaMemcpyAsync` on another stream or the same one after kernel completion.
      * **Double Buffering/Pipelining:**
          * Use at least two sets of pinned CPU buffers and GPU buffers. While the GPU is processing batch N using GPU\_buffer\_N (copied from CPU\_buffer\_N), the `batchAccumulatorWorker` can be filling CPU\_buffer\_N+1, and results from batch N-1 can be copied back from GPU\_buffer\_N-1.
          * The `MCTSEvaluator` would manage a small pool of these buffer sets.

4.  **Result Handling (`MCTSEvaluator` and `resultDistributorWorker`):**

      * Once results are copied back to pinned CPU memory, the `MCTSEvaluator`'s thread parses them.
      * It then packages these results (e.g., `NetworkOutput`) with the corresponding `PendingEvaluation` metadata and sends them to the `result_queue_`.
      * The `MCTSEngine::resultDistributorWorker` consumes these, updates the MCTS nodes, and performs backpropagation.

**Example Flow Snippet (Conceptual):**

  * **`batchAccumulatorWorker`:**

    ```cpp
    // Pseudo-code
    pinned_cpu_tensor_buffer = get_empty_pinned_buffer();
    current_offset = 0;
    batch_metadata.clear();
    while (collecting_batch) {
        PendingEvaluation eval_request = leaf_queue_.dequeue();
        convert_gamestate_to_tensor_inplace(eval_request.state, pinned_cpu_tensor_buffer + current_offset);
        batch_metadata.add(eval_request.node, eval_request.path); // Store original request info
        current_offset += tensor_size_for_one_state;
    }
    BatchInfo gpu_batch;
    gpu_batch.tensor_data_ptr = pinned_cpu_tensor_buffer;
    gpu_batch.num_states = ...;
    gpu_batch.metadata = batch_metadata;
    batch_queue_.enqueue(std::move(gpu_batch));
    ```

  * **`MCTSEvaluator`'s `processBatches` (External Queue Mode):**

    ```cpp
    // Pseudo-code
    BatchInfo gpu_batch = external_batch_queue->dequeue();
    torch::Tensor input_gpu_tensor = get_empty_gpu_tensor_for_stream(streamA);

    // Async copy HtoD
    copy_pinned_cpu_to_gpu_async(gpu_batch.tensor_data_ptr, input_gpu_tensor, streamA);

    // Launch NN inference (kernels) on GPU using input_gpu_tensor
    torch::Tensor policy_gpu_tensor, value_gpu_tensor;
    run_nn_inference_async(input_gpu_tensor, policy_gpu_tensor, value_gpu_tensor, streamA);

    // Async copy DtoH for results to another pinned CPU buffer
    pinned_cpu_result_buffer = get_empty_pinned_result_buffer();
    copy_gpu_to_pinned_cpu_async(policy_gpu_tensor, value_gpu_tensor, pinned_cpu_result_buffer, streamA);

    // Synchronize on the stream to ensure DtoH copy is complete before processing results
    synchronize_stream(streamA); 

    // Process results from pinned_cpu_result_buffer
    for (size_t i = 0; i < gpu_batch.num_states; ++i) {
        NetworkOutput output = extract_output_from_buffer(pinned_cpu_result_buffer, i);
        PendingEvaluation original_eval_info = gpu_batch.metadata.get_info(i);
        external_result_queue->enqueue({std::move(output), std::move(original_eval_info)});
    }
    // Release/recycle buffers
    ```

This approach maximizes GPU utilization by keeping data transfers and kernel executions pipelined and asynchronous, using pinned memory to speed up CPU-GPU data movement. The key is to ensure the game state to tensor conversion on the CPU is also highly efficient.

----------
----------

<todo_list>
1. Fix potential race conditions during node expansion and backpropagation
2. Optimize batch accumulation logic to maximize GPU utilization
3. Reduce lock contention in transposition table through sharding
4. Replace cout statements with lightweight logging in hot paths
5. Implement proper thread shutdown synchronization in destructors
6. Optimize atomic operations with memory ordering constraints
7. Add batch size adaptation based on inference latency metrics
8. Improve virtual loss implementation to reduce cache contention
9. Validate node lifetime management during tree parallel traversal
10. Implement tensor pre-allocation and asynchronous transfer for GPU inference
</todo_list>

<optimization_scheme>
1. Node Management:
- Implement hazard pointers for safe node access
- Use read-write locks for node expansion
- Add batch-friendly node prioritization in selection

2. Batching System:
- Introduce dynamic batch timeout adaptation
- Implement pinned memory buffers for zero-copy GPU transfers
- Add batch size prediction based on historical throughput

3. Transposition Table:
- Split into 64 shards using hash bits
- Use atomic shared_ptr for entry updates
- Implement LRU eviction policy with batch removal

4. Virtual Loss:
- Switch to thread-local virtual loss counters
- Use atomic fetch_add with memory_order_relaxed
- Implement deferred virtual loss recovery

5. Memory Management:
- Pre-allocate node pools with arena allocation
- Use memory-mapped tensors for CPU-GPU transfers
- Implement batch tensor reuse system
</optimization_scheme>

<parallelization_improvements>
1. Thread Synchronization:
- Replace mutexes with atomic_flag for hot paths
- Implement work-stealing queues for tree workers
- Use condition variables with predicate checks

2. Lock-Free Patterns:
- Implement Michael-Scott queue for leaf nodes
- Use atomic shared_ptr for node parent references
- Adopt RCU pattern for transposition table reads

3. Pipeline Optimization:
- Separate inference and backpropagation pipelines
- Implement double buffering for GPU transfers
- Add priority-based batching for critical nodes

4. Contention Reduction:
- Use thread-local random generators
- Implement per-thread transposition cache
- Distribute statistics counters with padding

5. Fault Tolerance:
- Add timeout detection for stalled workers
- Implement batch cancellation mechanism
- Add node validity checks during traversal
</parallelization_improvements>

<gpu_throughput_scenario>
1. Tensor Pre-allocation:
- Reserve contiguous pinned memory buffers for 4 batch sizes (64, 128, 256, 512)
- Maintain circular buffer of pre-converted tensors in NHWC format

2. Asynchronous Pipeline:
- Dedicated collector thread packs states into pre-allocated buffers
- CUDA stream copies completed buffers while next batch prepares
- Inference overlaps with CPU-side batch assembly

3. Batch Prioritization:
- Implement priority queue with freshness scoring
- Process high-value nodes first within batch limits
- Use predictive pre-fetching for likely expansions

4. Throughput Monitoring:
- Track GPU utilization vs batch size curve
- Dynamic batch size adaptation every 100ms
- Fallback to CPU execution for small batches

5. Optimized Transfer:
- Use cudaMemcpyAsync with separate stream
- Batch multiple small tensors into single transfer
- Implement tensor reuse pool with reference counting
</gpu_throughput_scenario>