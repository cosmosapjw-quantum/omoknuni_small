## MCTS Implementation Analysis and Optimization

Based on the provided codebase, documentation, and self-play logs, here's a comprehensive analysis and set of optimization suggestions for your MCTS implementation.

### \<todo\_list\>

**Critical Issues & Bugs:**

1.  **Fix Memory Leaks (P1 - Critical):**
      * Investigate why `MCTSEngine::root_.reset()` is not freeing all memory from the previous search tree. The primary suspects are lingering `std::shared_ptr<MCTSNode>` instances, potentially held by:
          * `PendingEvaluation::path` objects in `MCTSEngine::leaf_queue_` or `MCTSEngine::result_queue_` if not fully processed or cleared.
          * `NodeTracker::EvaluationResult::path` objects if `NodeTracker`'s result queue isn't properly managed.
          * `IGameState` objects if `utils::GameStatePoolManager` has leaks or is disabled and manual cloning/deletion is flawed.
      * Use memory profiling tools (e.g., Valgrind, AddressSanitizer, HeapTrack) to pinpoint exact allocation sites and unreleased objects.
      * Review all `std::shared_ptr` usage related to `MCTSNode` and `IGameState` for unexpected long-lived references.
2.  **Resolve Low GPU Batch Size (P1 - Critical Performance):**
      * Modify `MCTSEvaluator::processBatch` (external queue path) to ensure it waits until `batch_size_` items are collected or `max_wait_time` elapses, instead of processing a batch prematurely if the queue temporarily becomes empty.
      * Make `MCTSEvaluator`'s batch collection timeout (`max_wait_time`) more configurable and tune it based on MCTS worker throughput and desired latency. The current 10ms hard cap might be too aggressive.
      * Increase `MCTSEvaluator::min_batch_size_` (currently 1 for external queues via direct dequeue logic) to encourage larger batches.

**Performance Optimizations & Refactoring:**

3.  **Improve `MCTSEngine::runSearch` Waiting Logic (P2 - Performance):**
      * Replace the `std::this_thread::sleep_for` loop with `std::condition_variable` waits for `active_simulations_` and `pending_evaluations_` to reach zero, reducing CPU usage during waits.
4.  **Integrate Modern Memory Allocator (P2 - Performance/Stability):**
      * Adopt `mimalloc` or `jemalloc` application-wide (e.g., via `LD_PRELOAD` or by linking) to improve multi-threaded allocation speed and reduce memory fragmentation.
5.  **Implement MCTSNode Memory Pooling (P2 - Performance/Memory):**
      * Create a custom memory pool allocator specifically for `MCTSNode` objects to decrease allocation overhead and improve cache locality.
6.  **Advanced Task Scheduling (P2 - Performance/Maintainability):**
      * Refactor `MCTSEngine`'s manual thread management (`std::thread`) to use a dedicated task scheduling library like Cpp-Taskflow for better load balancing (work-stealing) of MCTS simulations and simpler CPU-GPU orchestration.
7.  **Refine Logging (P3 - Debuggability/Maintainability):**
      * Replace all `std::cout` logging with a robust logging library like `spdlog`, as planned in the PRD. Implement structured and configurable logging.
8.  **Integrate Profiling Tools (P3 - Diagnosability):**
      * Instrument the code with `Tracy Profiler` macros to enable detailed performance analysis of CPU, GPU, memory, and lock contention.
9.  **Clarify/Simplify `NodeTracker` Role (P3 - Maintainability):**
      * Investigate the necessity and interaction of `NodeTracker` with the primary evaluation path through `MCTSEvaluator`. If redundant or overly complex, simplify or remove it to improve code clarity.
10. **GameState Management Review (P3 - Performance/Memory):**
      * Thoroughly review `utils::GameStatePoolManager` for correctness and efficiency. Ensure states are properly acquired, released, and reset.

\</todo\_list\>

### \<optimization\_scheme\>

The optimization scheme focuses on addressing critical stability and performance issues first, then enhancing the architecture for better scalability and maintainability.

**Phase 1: Stability and Core Performance (Fixes for P1 Issues)**

1.  **Memory Leak Resolution:**
      * **Action:** Prioritize identifying and fixing the memory leak. Utilize memory debuggers and profilers. Review `shared_ptr` lifecycles, especially concerning `MCTSNode`, `IGameState`, and any objects held in queues or tracking structures (`PendingEvaluation`, `NodeTracker::EvaluationResult`).
      * **Verification:** Observe memory usage in self-play logs returning to a baseline after each search or growing minimally and predictably.
2.  **GPU Batch Size Correction:**
      * **Action:** Modify `MCTSEvaluator::processBatch` to implement a more patient batch collection strategy. The evaluator should wait until the configured `batch_size_` is met or a configurable `timeout_` (e.g., 5-20ms, tunable) expires.
        ```cpp
        // Suggested logic for MCTSEvaluator::processBatch's collection loop (external queue)
        // auto deadline = std::chrono::steady_clock::now() + configured_timeout;
        // while (evaluations.size() < batch_size_ && std::chrono::steady_clock::now() < deadline) {
        //     if (external_leaf_queue->try_dequeue(pending_eval)) {
        //         evaluations.push_back(std::move(pending_eval));
        //     } else if (evaluations.size() < min_effective_batch_size) { // min_effective_batch_size could be e.g. batch_size / 4 or a fixed number
        //         std::this_thread::sleep_for(std::chrono::microseconds(100)); // Wait briefly if queue is empty and batch too small
        //     } else if (!evaluations.empty()) {
        //          // Have some items, but not a full batch, and queue is currently empty.
        //          // Decide based on how close to deadline or if a smaller batch is acceptable after some waiting.
        //          // For now, could just sleep briefly and let outer loop check deadline.
        //          std::this_thread::sleep_for(std::chrono::microseconds(100));
        //     }
        // }
        // If evaluations.empty() after loop, it means timeout with no items.
        ```
      * **Verification:** Self-play logs should show `[EVALUATOR] Processing batch of X items` where X is consistently close to `batch_size_` (e.g., 32, 64, or 128) or reflects batches formed due to timeout. GPU utilization should increase significantly.

**Phase 2: Performance Enhancements & Architectural Improvements (P2 Issues)**

3.  **Efficient Waiting & System Allocator:**
      * **Action:** Replace busy-waits in `MCTSEngine::runSearch` with condition variables. Integrate `mimalloc` (recommended for performance and ease of use) or `jemalloc` application-wide.
      * **Benefit:** Reduced CPU idle cycles, faster and more consistent memory allocation/deallocation, potentially mitigating fragmentation.
4.  **Specialized Node Allocator & Task Scheduling:**
      * **Action:** Implement a memory pool for `MCTSNode`. Refactor thread management in `MCTSEngine` to use Cpp-Taskflow.
      * **Benefit:** Faster node creation/deletion, improved CPU utilization through work-stealing, and a more robust framework for managing parallel MCTS tasks and CPU-GPU interaction.

**Phase 3: Observability and Refinement (P3 Issues)**

5.  **Enhanced Logging & Profiling:**
      * **Action:** Systematically replace `std::cout` with `spdlog`. Instrument critical code paths (MCTS stages, NN calls, queue operations) with `Tracy Profiler` macros.
      * **Benefit:** Greatly improved ability to debug issues, understand performance characteristics, and guide further optimizations.
6.  **Code Clarity & Game State Management:**
      * **Action:** Simplify the evaluation path by clarifying or removing the `NodeTracker` if it's redundant. Conduct a detailed review of `GameStatePoolManager` to ensure its correctness and efficiency in state reuse and preventing leaks.
      * **Benefit:** Improved code maintainability and reduced risk of bugs.

### \<parallelization\_improvements\>

The current leaf parallelization model uses `std::thread` for MCTS workers and `moodycamel::ConcurrentQueue` for communication with the `MCTSEvaluator`. While functional, it can be improved:

  * **Synchronization:**

      * **Current:** `std::mutex` (e.g., `MCTSNode::expansion_mutex_`), `std::atomic` (for node stats), `std::condition_variable` (in `MCTSEngine`).
      * **Improvements:**
          * **Reduce Lock Granularity/Frequency:** Analyze critical sections protected by mutexes. For `MCTSNode::expansion_mutex_`, it's likely necessary. However, if other shared data structures are heavily contended, consider more fine-grained locking or lock-free alternatives if feasible (though this adds complexity).
          * **Cpp-Taskflow:** Adopting a task scheduler like Cpp-Taskflow can abstract away much of the manual synchronization for task distribution and dependencies, potentially reducing complexity and custom lock/CV usage.

  * **Deadlocks:**

      * **Risk:** Complex interactions between multiple mutexes and condition variables (especially in `MCTSEngine`'s shutdown logic or if `NodeTracker` introduces its own locking that interacts with `MCTSEngine` or `MCTSEvaluator`) can lead to deadlocks.
      * **Mitigation:**
          * **Lock Ordering:** Strictly enforce a global lock acquisition order if multiple mutexes must be held.
          * **Simplify Synchronization Logic:** Refactoring to Cpp-Taskflow could reduce custom synchronization points.
          * **Scoped Locks:** Consistently use `std::lock_guard` or `std::unique_lock` to ensure mutexes are always released.
          * **Avoid Waiting on CVs While Holding Unnecessary Locks:** Ensure locks are released before potentially long waits or blocking operations not related to the CV's predicate.

  * **Lock Contention:**

      * **Potential Hotspots:** `MCTSNode::expansion_mutex_` if many threads try to expand children of the same parent simultaneously (less likely with virtual loss guiding threads apart). Transposition Table access if its internal concurrency mechanisms are insufficient for the load (though `parallel-hashmap` is designed for this).
      * **Mitigation:**
          * Virtual loss helps reduce contention on popular tree paths.
          * `parallel-hashmap`'s internal sharding should minimize TT contention. Monitor with profiling (e.g., Tracy can show lock contention).
          * If specific node statistics updates become bottlenecks even with atomics (unlikely for simple counters/sums), more advanced techniques might be needed, but atomics are usually sufficient.

  * **Race Conditions:**

      * **Current:** The PRD mentions protecting N/Q stats with atomics, which is done.
      * **Verification:** Use thread sanitizers (e.g., TSan with GCC/Clang) during development and testing to detect race conditions.
      * **Review Shared Data Access:** Carefully review all shared data access, especially complex objects or data structures not inherently made thread-safe by atomics or single mutexes. `MCTSEngine` members accessed by multiple threads (e.g., `root_`, queues, state flags) are critical.

  * **Memory Issues (related to Parallelization):**

      * **Dangling Pointers/Use-after-free:** Addressed by `std::shared_ptr` and `std::weak_ptr` for tree nodes and TT. The main memory leak seems to be objects *not being deleted* rather than use-after-free.
      * **Safe Memory Reclamation for Lock-Free Structures:** Not currently using custom complex lock-free data structures that would require explicit reclamation schemes like hazard pointers or EBR beyond what `moodycamel::ConcurrentQueue` and `parallel-hashmap` provide internally. If such structures are introduced, use libraries like Xenium as recommended.
      * **Node Pool Allocator:** Implementing a thread-safe memory pool for `MCTSNode` objects can improve performance by reducing contention on global allocators and improving data locality.

  * **Use of Atomic Variables:**

      * **Current:** `MCTSNode::visit_count_`, `value_sum_`, `virtual_loss_count_`, `evaluation_in_progress_` are `std::atomic`. `MCTSEngine` uses atomics for `shutdown_`, `active_simulations_`, `search_running_`, `pending_evaluations_`, etc.
      * **Benefit:** Minimizes lock-related problems for simple counters and flags.
      * **Considerations:** Ensure correct memory ordering (e.g., `std::memory_order_acquire`, `std::memory_order_release`, `std::memory_order_acq_rel`) is used where necessary for visibility and synchronization. The current usage often employs default sequential consistency (`std::memory_order_seq_cst`), which is safest but potentially slower. Relaxed ordering can be used where appropriate if carefully analyzed, but stick to sequential consistency if unsure. For example, `active_simulations_` and `pending_evaluations_` are central to search termination logic and should have appropriate ordering to ensure visibility across threads.

### \<gpu\_throughput\_scenario\>

This scenario aims to significantly increase GPU throughput by addressing the current batch size of 1 and improving the data pipeline from CPU to GPU.

**Scenario: Achieving Consistent Batches of 64+ for GPU Inference**

1.  **Optimized Batch Collection in `MCTSEvaluator`:**

      * The `MCTSEvaluator::processBatch` method (when using external queues from `MCTSEngine`) is modified.
      * It will attempt to fill a batch of up to `batch_size_` (e.g., 128 as per config).
      * It introduces a configurable `min_batch_size_for_immediate_processing` (e.g., 32 or 64).
      * It waits for up to a configurable `dynamic_timeout_ms` (e.g., initially 10-15ms, adaptable).
      * **Logic:**
          * Continuously try to dequeue from `leaf_queue_ptr_`.
          * If `evaluations.size() >= batch_size_`, process the batch immediately.
          * If `evaluations.size() >= min_batch_size_for_immediate_processing` AND `leaf_queue_ptr_` is currently empty (or `try_dequeue` fails repeatedly for a very short duration), process the current partial batch to avoid GPU starvation if MCTS is slow.
          * If `evaluations.size() < min_batch_size_for_immediate_processing`, continue dequeuing/waiting until `dynamic_timeout_ms` is reached. If timeout occurs and `evaluations.size() > 0`, process the collected (potentially small) batch. If timeout occurs and batch is empty, do nothing and try again.
          * The `dynamic_timeout_ms` could be adjusted based on the rate of incoming requests or average batch fill times, becoming more patient if the queue fills quickly.

2.  **Fast Tensor Collection from CPU (MCTS Workers & MCTSEngine):**

      * MCTS worker threads (`MCTSEngine::treeTraversalWorker`) identify leaf nodes.
      * The `IGameState::getNNInput()` method (or similar) should efficiently serialize the game state into the required tensor format (e.g., a flat `std::vector<float>` or directly into a pre-allocated buffer).
      * **Pinned Host Memory:** For states being sent to the GPU, allocate their tensor representations in pinned (page-locked) host memory. This significantly speeds up `cudaMemcpyAsync` operations. LibTorch tensors can be created on pinned memory.
        ```cpp
        // Example concept (actual API depends on how game state is tensorized)
        // In MCTSEngine::traverseTree, when creating PendingEvaluation:
        // auto state_tensor_cpu = game_state_to_tensor(leaf->getState()); // Returns a CPU tensor
        // auto pinned_tensor_cpu = state_tensor_cpu.pin_memory(); // If using libtorch tensors directly
        // pending.state_tensor = pinned_tensor_cpu; // Store this in PendingEvaluation
        ```
        Alternatively, `MCTSEvaluator` can collect multiple `std::unique_ptr<core::IGameState>` and then, just before inference, convert them into a batch of tensors, potentially allocating the CPU-side batch tensor in pinned memory.

3.  **Efficient Batch Transfer to GPU & High-Speed Inference:**

      * **Batch Construction in `MCTSEvaluator`:**
          * Once a batch of `PendingEvaluation` objects (containing game states or pre-converted pinned CPU tensors) is collected:
          * Stack these individual CPU tensors into a single batch CPU tensor (e.g., `torch::stack` if they are already libtorch tensors, or manually fill a larger pre-allocated pinned tensor).
      * **Asynchronous Transfer to GPU:**
          * Use `batch_cpu_tensor.to(torch::kCUDA, /*non_blocking=*/true)` to transfer the entire batch to the GPU asynchronously. This requires the source CPU tensor to be in pinned memory for true asynchronicity.
      * **CUDA Stream for Inference:** Perform the NN inference on a dedicated CUDA stream to overlap computation with data transfers of future/past batches if possible.
        ```cpp
        // In MCTSEvaluator::processBatch (after collecting 'evaluations' vector)
        // std::vector<torch::Tensor> cpu_tensors;
        // for (auto& eval_req : evaluations) { /* convert eval_req.state to tensor and add to cpu_tensors */ }
        // torch::Tensor batch_cpu = torch::stack(cpu_tensors, 0).pin_memory(); // Stack and pin
        //
        // torch::Tensor batch_gpu = batch_cpu.to(torch::kCUDA, /*non_blocking=*/true);
        // // Ensure synchronization if needed before inference, or use CUDA streams
        // {
        //   torch::cuda::CUDAStreamGuard stream_guard(inference_stream_); // Use a dedicated stream
        //   batch_output_gpu = neural_net_module_->forward({batch_gpu}).toTensor();
        // }
        // // Transfer results back, also potentially async
        // torch::Tensor results_cpu = batch_output_gpu.to(torch::kCPU, /*non_blocking=*/true);
        // // Ensure synchronization (e.g. results_cpu.cpu().synchronize() or stream sync) before accessing data
        ```
      * **TensorRT/ONNX Runtime:** (Longer-term) For maximum inference speed, consider converting the PyTorch model to TensorRT or ONNX and using their respective runtimes. TensorRT can apply further optimizations like layer fusion and precision calibration (FP16/INT8) tailored to the RTX 3060 Ti.

4.  **Result Processing:**

      * Transfer policy and value results from GPU back to pinned CPU memory asynchronously.
      * Once results are back on the CPU and synchronized, dispatch them to the corresponding `MCTSEngine::result_queue_`.

**Expected Outcome:**

  * The GPU receives larger, more consistent batches (e.g., 32-128 states).
  * GPU utilization increases dramatically, reducing idle time.
  * Overall MCTS search speed (nodes per second) improves significantly as NN evaluations, a common bottleneck, are processed more efficiently.
  * The `[EVALUATOR] Processing batch of X items` log message will show X values much closer to the configured `batch_size`.
  * The total time per self-play game should decrease.

  ----------
  ----------

  <todo_list>
1. **Fix Batch Aggregation for GPU Throughput**
   - Implement aggressive batch formation mechanism
   - Modify timeout logic to wait longer when evaluations are pending
   - Prevent processing of underfilled batches (especially single items)
   - Add dynamic batch size adjustment based on queue fill rate

2. **Resolve Memory Leaks**
   - Fix circular references between parent-child nodes
   - Add shared_ptr cleanup in TranspositionTable::clear() method
   - Implement periodic memory monitoring and cleanup
   - Reduce GameStatePool maximum size

3. **Improve Thread Synchronization**
   - Fix deadlock in the worker threads and coordinator
   - Reduce lock contention in critical paths
   - Add timeouts for thread operations to prevent indefinite waits
   - Implement better thread shutdown sequence

4. **Optimize TranspositionTable Management**
   - Implement more aggressive cleanup schedule
   - Fix dangling weak_ptr references
   - Add size-based cleanup triggers
   - Improve hash collision handling

5. **Enhance MCTSNode Management**
   - Fix evaluation flag clearing on all error paths
   - Improve virtual loss handling
   - Add proper handling of orphaned nodes
   - Optimize expansion with better threading model
</todo_list>

<optimization_scheme>
## Core Batching Optimization

The fundamental issue is the GPU processing batches of size 1 instead of utilizing the configured batch size of 128. This drastically reduces inference throughput.

### Implementation Plan:

1. **Two-Phase Batch Collection**:
```cpp
std::vector<EvaluationRequest> MCTSEvaluator::collectBatch(size_t target_batch_size) {
    std::vector<EvaluationRequest> batch;
    batch.reserve(target_batch_size);
    
    // Phase 1: Fast collection of immediately available items
    size_t initial_dequeue = std::min(request_queue_.size_approx(), target_batch_size);
    if (initial_dequeue > 0) {
        std::vector<EvaluationRequest> temp_batch(initial_dequeue);
        size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), initial_dequeue);
        for (size_t i = 0; i < dequeued; ++i) {
            batch.push_back(std::move(temp_batch[i]));
        }
    }
    
    // Phase 2: If batch is small, wait longer to collect more items
    if (batch.size() < target_batch_size / 4 && batch.size() < 32) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        while (std::chrono::steady_clock::now() < deadline && 
               batch.size() < target_batch_size && 
               !shutdown_.load(std::memory_order_acquire)) {
            EvaluationRequest req;
            if (request_queue_.try_dequeue(req)) {
                batch.push_back(std::move(req));
            } else {
                // Short sleep to reduce CPU usage
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    
    // Never return empty batches if we have items
    if (!batch.empty()) {
        return batch;
    }
    
    // If completely empty, wait for notification with timeout
    std::unique_lock<std::mutex> lock(cv_mutex_);
    bool has_items = cv_.wait_for(lock, std::chrono::milliseconds(timeout_),
        [this]() { return shutdown_.load() || request_queue_.size_approx() > 0; });
    
    if (has_items && !shutdown_.load()) {
        EvaluationRequest req;
        if (request_queue_.try_dequeue(req)) {
            batch.push_back(std::move(req));
        }
    }
    
    return batch;
}
```

2. **Adaptive Batch Size & Timeout**:
```cpp
void MCTSEvaluator::updateBatchParameters() {
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    
    // Update every 5 seconds
    if (now - last_update < std::chrono::seconds(5)) {
        return;
    }
    
    last_update = now;
    
    // Calculate metrics
    float avg_batch_size = getAverageBatchSize();
    auto avg_latency = getAverageBatchLatency();
    size_t queue_size = request_queue_.size_approx();
    
    // If batch size is consistently small, increase timeout
    if (avg_batch_size < batch_size_ * 0.25 && queue_size < 10) {
        timeout_ = std::min(timeout_ * 2, std::chrono::milliseconds(50));
    }
    // If queue is consistently large, decrease timeout
    else if (queue_size > batch_size_ * 2) {
        timeout_ = std::max(timeout_ / 2, std::chrono::milliseconds(5));
    }
    
    // Adjust batch size based on GPU utilization
    // (Could use CUDA utilities to measure GPU usage)
    if (avg_latency < std::chrono::milliseconds(10) && avg_batch_size > batch_size_ * 0.8) {
        batch_size_ = std::min(batch_size_ * 2, size_t(512));
    }
    else if (avg_latency > std::chrono::milliseconds(30)) {
        batch_size_ = std::max(batch_size_ / 2, size_t(16));
    }
}
```

3. **Memory Leak Fixes**:

```cpp
// 1. Fix TranspositionTable::clear() to properly clean weak_ptr entries
void TranspositionTable::clear() {
    std::lock_guard<std::mutex> lock(clear_mutex_);
    
    // First, expire all weak_ptr references to prevent dangling pointers
    for (auto& [hash, entry] : entries_) {
        if (entry) {
            entry->node.reset();  // Explicitly reset weak_ptr
        }
    }
    
    // Clear the hash map and reset statistics
    entries_.clear();
    resetStats();
}

// 2. Implement periodic memory monitoring in MCTSEngine
void MCTSEngine::monitorMemoryUsage() {
    static size_t last_cleanup_count = 0;
    static auto last_cleanup_time = std::chrono::steady_clock::now();
    
    // Check memory every 1000 simulations or 30 seconds
    if (total_simulations_ - last_cleanup_count < 1000 && 
        std::chrono::steady_clock::now() - last_cleanup_time < std::chrono::seconds(30)) {
        return;
    }
    
    last_cleanup_count = total_simulations_;
    last_cleanup_time = std::chrono::steady_clock::now();
    
    // Use platform-specific methods to get current memory usage
    size_t current_memory = getCurrentMemoryUsage();
    
    // If memory usage is too high, force cleanup
    if (current_memory > memory_limit_) {
        forceCleanup();
    }
}

// 3. Fix GameStatePool release method to be more aggressive
void GameStatePool::release(std::unique_ptr<core::IGameState> state) {
    if (!state) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_releases_.fetch_add(1);
    
    // Much more restrictive pool size limit
    const size_t max_pool_size = initial_size_ * 2;  // Only allow up to 2x initial size
    if (pool_.size() < max_pool_size) {
        pool_.push_back(std::move(state));
    }
    // If pool is full, let the state be destroyed
}
```

4. **Improved Parallel Leaf Collection**:

```cpp
void MCTSEngine::traverseTree(std::shared_ptr<MCTSNode> root) {
    if (!root) return;
    
    try {
        // Selection phase
        auto [leaf, path] = selectLeafNode(root);
        if (!leaf) return;
        
        // Expansion phase - never block
        if (!leaf->isTerminal() && leaf->isLeaf()) {
            // Try to expand and mark for evaluation atomically
            bool expand_success = false;
            bool should_evaluate = false;
            
            // CRITICAL: Prevent race condition by checking and marking evaluation BEFORE expand
            if (leaf->tryMarkForEvaluation()) {
                // We got exclusive rights to evaluate this node
                should_evaluate = true;
                try {
                    leaf->expand();
                    expand_success = true;
                } catch (...) {
                    // If expansion fails, clear the evaluation flag
                    leaf->clearEvaluationFlag();
                    should_evaluate = false;
                    return;
                }
            } else {
                // Another thread is already evaluating this node - early return
                return;
            }
            
            // Only queue for evaluation if we successfully marked and expanded
            if (should_evaluate && expand_success) {
                // Create evaluation request
                auto state_clone = cloneGameState(leaf->getState());
                if (state_clone) {
                    PendingEvaluation pending;
                    pending.node = leaf;
                    pending.path = std::move(path);
                    pending.state = std::move(state_clone);
                    
                    // Group batch ID by thread ID for better batching
                    int thread_id = getThreadId();
                    pending.batch_id = (batch_counter_.fetch_add(1, std::memory_order_relaxed) & 0xFFFFFF00) | (thread_id & 0xFF);
                    pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                    
                    // Submit to leaf queue with proper move semantics
                    if (leaf_queue_.enqueue(std::move(pending))) {
                        // Increment pending evaluations count (FIX: only once per leaf)
                        pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                        if (evaluator_) { 
                            evaluator_->notifyLeafAvailable();
                        }
                    } else {
                        // Clear the flag since we failed to enqueue
                        leaf->clearEvaluationFlag();
                    }
                } else {
                    // Clear the flag since we failed to clone the state
                    leaf->clearEvaluationFlag();
                }
            }
        } else if (leaf->isTerminal()) {
            // Handle terminal nodes immediately
            float value = 0.0f;
            auto result = leaf->getState().getGameResult();
            int current_player = leaf->getState().getCurrentPlayer();
            
            if (result == core::GameResult::WIN_PLAYER1) {
                value = current_player == 1 ? 1.0f : -1.0f;
            } else if (result == core::GameResult::WIN_PLAYER2) {
                value = current_player == 2 ? 1.0f : -1.0f;
            }
            
            backPropagate(path, value);
        }
    } catch (const std::exception& e) {
        // Log error and continue
    }
}
```

This optimization scheme addresses the core issues of batch efficiency, memory leaks, and thread synchronization. The improved batching strategy should dramatically increase GPU throughput, while the memory management fixes will prevent the unbounded memory growth observed in the logs.
</optimization_scheme>

<parallelization_improvements>
## Parallelization Enhancements

The current implementation suffers from several thread-related issues including potential deadlocks, race conditions, and inefficient synchronization. Here are detailed improvements:

### 1. Lock-Free Evaluation Queue

Replace the current evaluation queue with a fully lock-free implementation:

```cpp
class LockFreeEvaluationQueue {
private:
    moodycamel::ConcurrentQueue<EvaluationRequest> queue_;
    std::atomic<size_t> size_{0};
    
public:
    bool enqueue(EvaluationRequest&& request) {
        bool success = queue_.enqueue(std::move(request));
        if (success) {
            size_.fetch_add(1, std::memory_order_release);
        }
        return success;
    }
    
    bool try_dequeue(EvaluationRequest& request) {
        bool success = queue_.try_dequeue(request);
        if (success) {
            size_.fetch_sub(1, std::memory_order_release);
        }
        return success;
    }
    
    size_t try_dequeue_bulk(EvaluationRequest* items, size_t max) {
        size_t dequeued = queue_.try_dequeue_bulk(items, max);
        if (dequeued > 0) {
            size_.fetch_sub(dequeued, std::memory_order_release);
        }
        return dequeued;
    }
    
    size_t size() const {
        return size_.load(std::memory_order_acquire);
    }
};
```

### 2. Improved Virtual Loss Handling

Enhance the virtual loss mechanism to better handle thread contention:

```cpp
// In MCTSNode class
void addVirtualLoss(int count = 1) {
    // Add virtual loss with saturation to prevent overflow
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    // Cap at reasonable maximum to prevent integer overflow
    int new_value = std::min(current + count, 1000);  
    virtual_loss_count_.store(new_value, std::memory_order_release);
}

void removeVirtualLoss(int count = 1) {
    // Remove virtual loss with floor at zero
    int current = virtual_loss_count_.load(std::memory_order_relaxed);
    int new_value = std::max(current - count, 0);
    virtual_loss_count_.store(new_value, std::memory_order_release);
}
```

### 3. Thread Coordination with Atomic Barriers

Add atomic barriers for thread coordination during engine shutdown:

```cpp
class AtomicBarrier {
private:
    std::atomic<int> counter_;
    int threshold_;
    
public:
    AtomicBarrier(int threshold) : counter_(0), threshold_(threshold) {}
    
    void arrive() {
        counter_.fetch_add(1, std::memory_order_release);
    }
    
    bool try_wait(std::chrono::milliseconds timeout) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (std::chrono::steady_clock::now() < deadline) {
            if (counter_.load(std::memory_order_acquire) >= threshold_) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        return false;
    }
    
    void reset() {
        counter_.store(0, std::memory_order_release);
    }
};
```

### 4. Thread-Safe Leaf Evaluation Strategy

Implement a thread-safe leaf evaluation strategy that prevents race conditions:

```cpp
bool MCTSNode::tryMarkForEvaluation() {
    // Try to set the evaluation flag from false to true atomically
    bool expected = false;
    return evaluation_in_progress_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
}

void MCTSNode::clearEvaluationFlag() {
    // Clear the evaluation flag atomically
    evaluation_in_progress_.store(false, std::memory_order_release);
}

bool MCTSNode::isBeingEvaluated() const {
    // Read the evaluation flag atomically
    return evaluation_in_progress_.load(std::memory_order_acquire);
}
```

### 5. Deadlock Prevention with Timed Mutex Operations

Replace standard mutex operations with timed versions to prevent deadlocks:

```cpp
class TimedMutex {
private:
    std::mutex mutex_;
    std::atomic<bool> is_locked_{false};
    
public:
    bool try_lock_for(std::chrono::milliseconds timeout) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (std::chrono::steady_clock::now() < deadline) {
            if (mutex_.try_lock()) {
                is_locked_.store(true, std::memory_order_release);
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        return false;
    }
    
    void unlock() {
        is_locked_.store(false, std::memory_order_release);
        mutex_.unlock();
    }
    
    bool is_locked() const {
        return is_locked_.load(std::memory_order_acquire);
    }
};
```

### 6. Thread Pool with Work-Stealing Queue

Implement a work-stealing thread pool to better balance the tree traversal workload:

```cpp
class WorkStealingThreadPool {
private:
    std::vector<std::thread> threads_;
    std::vector<std::deque<std::function<void()>>> local_queues_;
    std::atomic<bool> shutdown_{false};
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    
public:
    WorkStealingThreadPool(size_t num_threads) {
        local_queues_.resize(num_threads);
        
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this, i]() {
                workerLoop(i);
            });
        }
    }
    
    ~WorkStealingThreadPool() {
        shutdown_.store(true, std::memory_order_release);
        cv_.notify_all();
        
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    void enqueue(std::function<void()> task, size_t preferred_thread = -1) {
        size_t thread_idx = preferred_thread;
        if (thread_idx >= local_queues_.size()) {
            thread_idx = std::hash<std::thread::id>{}(std::this_thread::get_id()) % local_queues_.size();
        }
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            local_queues_[thread_idx].push_back(std::move(task));
        }
        
        cv_.notify_one();
    }
    
private:
    void workerLoop(size_t thread_idx) {
        while (!shutdown_.load(std::memory_order_acquire)) {
            std::function<void()> task;
            bool have_task = false;
            
            // Try to get a task from our own queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (!local_queues_[thread_idx].empty()) {
                    task = std::move(local_queues_[thread_idx].front());
                    local_queues_[thread_idx].pop_front();
                    have_task = true;
                }
            }
            
            // If no task, try to steal from other queues
            if (!have_task) {
                for (size_t i = 0; i < local_queues_.size() && !have_task; ++i) {
                    if (i == thread_idx) continue;
                    
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (!local_queues_[i].empty()) {
                        task = std::move(local_queues_[i].back());
                        local_queues_[i].pop_back();
                        have_task = true;
                    }
                }
            }
            
            // If we have a task, execute it
            if (have_task) {
                task();
            } else {
                // Wait for a task to be available
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                    return shutdown_.load(std::memory_order_acquire) || 
                           std::any_of(local_queues_.begin(), local_queues_.end(), 
                                      [](const auto& q) { return !q.empty(); });
                });
            }
        }
    }
};
```

### 7. Improved Shutdown Sequence

Implement a robust shutdown sequence that prevents thread leaks:

```cpp
void MCTSEngine::stop() {
    // Signal shutdown
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    
    // Wake up all threads
    cv_.notify_all();
    batch_cv_.notify_all();
    result_cv_.notify_all();
    
    // Stop the evaluator
    if (evaluator_) {
        evaluator_->stop();
    }
    
    // Join worker threads with timeout
    auto joinWithTimeout = [](std::thread& t, std::chrono::milliseconds timeout) {
        if (!t.joinable()) return true;
        
        std::promise<bool> joined;
        std::future<bool> joined_future = joined.get_future();
        
        std::thread joiner_thread([&]() {
            t.join();
            joined.set_value(true);
        });
        
        auto status = joined_future.wait_for(timeout);
        if (status == std::future_status::ready) {
            joiner_thread.join();
            return true;
        }
        
        // Thread is stuck, detach it
        joiner_thread.detach();
        return false;
    };
    
    // Join result distributor
    joinWithTimeout(result_distributor_worker_, std::chrono::seconds(5));
    
    // Join tree traversal workers
    for (auto& worker : tree_traversal_workers_) {
        joinWithTimeout(worker, std::chrono::seconds(2));
    }
    
    // Clear leftover requests
    drainQueues();
}
```

These parallelization improvements address the key thread-related issues in the current implementation. By using more lock-free data structures, timeout-based operations, work stealing, and atomic barriers, we can significantly reduce lock contention, prevent deadlocks, and improve overall throughput.
</parallelization_improvements>

<gpu_throughput_scenario>
## Optimal GPU Throughput Implementation

The current implementation processes batches of size 1, severely underutilizing the GPU. Here's a concrete implementation for maximizing GPU throughput:

### 1. Centralized Batch Collector

```cpp
class BatchCollector {
private:
    moodycamel::ConcurrentQueue<EvaluationRequest>& request_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::thread collector_thread_;
    std::atomic<size_t> waiting_threads_{0};
    std::atomic<int64_t> last_large_batch_time_{0};
    size_t target_batch_size_;
    
public:
    BatchCollector(moodycamel::ConcurrentQueue<EvaluationRequest>& queue, size_t batch_size)
        : request_queue_(queue), target_batch_size_(batch_size) {
        
        collector_thread_ = std::thread([this]() {
            collectorLoop();
        });
    }
    
    ~BatchCollector() {
        shutdown_.store(true, std::memory_order_release);
        cv_.notify_all();
        
        if (collector_thread_.joinable()) {
            collector_thread_.join();
        }
    }
    
    void notifyItemAvailable() {
        cv_.notify_one();
    }
    
    void notifyThreadWaiting() {
        waiting_threads_.fetch_add(1, std::memory_order_release);
        cv_.notify_one();
    }
    
    void notifyThreadDone() {
        waiting_threads_.fetch_sub(1, std::memory_order_release);
    }
    
private:
    void collectorLoop() {
        while (!shutdown_.load(std::memory_order_acquire)) {
            // Vector to store collected batch
            std::vector<EvaluationRequest> batch;
            batch.reserve(target_batch_size_);
            
            // Phase 1: Quick collection of available items
            size_t queue_size = request_queue_.size_approx();
            if (queue_size > 0) {
                std::vector<EvaluationRequest> temp_batch(std::min(queue_size, target_batch_size_));
                size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), temp_batch.size());
                
                for (size_t i = 0; i < dequeued; ++i) {
                    batch.push_back(std::move(temp_batch[i]));
                }
            }
            
            // Phase 2: Wait for more items if batch is small
            if (batch.size() < target_batch_size_ / 2) {
                // Determine wait time based on batch size and waiting threads
                int wait_time_ms = 5;  // Base wait time
                
                // If we have waiting threads or a partially filled batch, wait longer
                if (waiting_threads_.load(std::memory_order_acquire) > 0 || batch.size() > 0) {
                    wait_time_ms = std::min(50, 5 + static_cast<int>(batch.size() * 2));
                }
                
                // If it's been a long time since a large batch, be more aggressive
                int64_t current_time = getTimeMillis();
                int64_t time_since_large_batch = current_time - last_large_batch_time_.load(std::memory_order_acquire);
                if (time_since_large_batch > 1000) {  // More than 1 second
                    wait_time_ms = std::min(100, wait_time_ms * 2);
                }
                
                std::unique_lock<std::mutex> lock(mutex_);
                auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_time_ms);
                
                cv_.wait_until(lock, deadline, [this]() {
                    return shutdown_.load(std::memory_order_acquire) || 
                           request_queue_.size_approx() > 0;
                });
                
                // Collect any additional items
                queue_size = request_queue_.size_approx();
                if (queue_size > 0) {
                    size_t remaining = target_batch_size_ - batch.size();
                    std::vector<EvaluationRequest> temp_batch(std::min(queue_size, remaining));
                    size_t dequeued = request_queue_.try_dequeue_bulk(temp_batch.data(), temp_batch.size());
                    
                    for (size_t i = 0; i < dequeued; ++i) {
                        batch.push_back(std::move(temp_batch[i]));
                    }
                }
            }
            
            // Process the batch if it's not empty
            if (!batch.empty()) {
                processBatch(batch);
                
                // Update last large batch time if this was a good-sized batch
                if (batch.size() >= target_batch_size_ / 2) {
                    last_large_batch_time_.store(getTimeMillis(), std::memory_order_release);
                }
            }
        }
    }
    
    void processBatch(std::vector<EvaluationRequest>& batch) {
        // Extract states from requests
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.reserve(batch.size());
        
        for (auto& req : batch) {
            states.push_back(std::move(req.state));
        }
        
        // Create tensor batch
        torch::Tensor batch_tensor = createBatchTensor(states);
        
        // Move tensor to GPU in a single transfer
        batch_tensor = batch_tensor.to(torch::kCUDA);
        
        // Run inference on GPU
        torch::Tensor policy_tensor, value_tensor;
        {
            torch::NoGradGuard no_grad;
            auto output = neural_network_->forward(batch_tensor);
            policy_tensor = output.first;
            value_tensor = output.second;
        }
        
        // Move results back to CPU
        policy_tensor = policy_tensor.to(torch::kCPU);
        value_tensor = value_tensor.to(torch::kCPU);
        
        // Process results
        for (size_t i = 0; i < batch.size(); ++i) {
            NetworkOutput output;
            
            // Extract policy
            auto policy_accessor = policy_tensor.accessor<float, 2>();
            output.policy.resize(policy_accessor.size(1));
            for (int j = 0; j < policy_accessor.size(1); ++j) {
                output.policy[j] = policy_accessor[i][j];
            }
            
            // Extract value
            output.value = value_tensor[i].item<float>();
            
            // Set result
            batch[i].promise.set_value(std::move(output));
        }
    }
    
    int64_t getTimeMillis() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    torch::Tensor createBatchTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
        // Assuming all states have the same size and format
        if (states.empty()) {
            return torch::Tensor();
        }
        
        // Get dimensions from the first state
        const auto& first_state = states[0];
        std::vector<float> features = first_state->getFeatures();
        int num_channels = first_state->getNumChannels();
        int board_size = first_state->getBoardSize();
        
        // Create batch tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor batch_tensor = torch::zeros({static_cast<long>(states.size()), 
                                                   num_channels, 
                                                   board_size, 
                                                   board_size}, options);
        
        // Fill batch tensor
        for (size_t i = 0; i < states.size(); ++i) {
            std::vector<float> features = states[i]->getFeatures();
            torch::Tensor state_tensor = torch::from_blob(features.data(), 
                                                          {num_channels, board_size, board_size}, 
                                                          options).clone();
            batch_tensor[i] = state_tensor;
        }
        
        return batch_tensor;
    }
};
```

### 2. Integration with MCTSEvaluator

```cpp
class MCTSEvaluator {
private:
    // ... existing members ...
    std::unique_ptr<BatchCollector> batch_collector_;
    
public:
    MCTSEvaluator(InferenceFunction inference_fn, size_t batch_size, std::chrono::milliseconds timeout)
        : inference_fn_(std::move(inference_fn)),
          batch_size_(batch_size),
          timeout_(timeout) {
        
        batch_collector_ = std::make_unique<BatchCollector>(request_queue_, batch_size_);
    }
    
    void start() {
        // ... existing code ...
    }
    
    void stop() {
        // ... existing code ...
        batch_collector_.reset();
    }
    
    std::future<NetworkOutput> evaluateState(std::shared_ptr<MCTSNode> node, std::unique_ptr<core::IGameState> state) {
        // ... existing code ...
        
        // Notify batch collector
        batch_collector_->notifyItemAvailable();
        
        return future;
    }
    
    // ... other methods ...
};
```

### 3. Worker Thread Synchronization for Better Batches

```cpp
void MCTSEngine::treeTraversalWorker(int worker_id) {
    while (!shutdown_.load(std::memory_order_acquire)) {
        // Check if there's work to do
        int remaining_sims = active_simulations_.load(std::memory_order_acquire);
        if (remaining_sims <= 0 || !root_ || !workers_active_.load(std::memory_order_acquire)) {
            // ... existing code for waiting ...
            continue;
        }
        
        // Claim a batch of simulations
        int batch_size = std::min(64, std::max(16, remaining_sims / settings_.num_threads));
        int claimed = 0;
        
        while (claimed < batch_size && !shutdown_.load(std::memory_order_acquire)) {
            int old_value = active_simulations_.load(std::memory_order_acquire);
            if (old_value <= 0) break;
            
            int to_claim = std::min(batch_size - claimed, old_value);
            if (active_simulations_.compare_exchange_weak(old_value, old_value - to_claim,
                                                          std::memory_order_acq_rel, 
                                                          std::memory_order_acquire)) {
                claimed += to_claim;
            }
        }
        
        // Tell evaluator we're about to generate evaluation requests
        if (evaluator_) {
            evaluator_->notifyThreadWaiting();
        }
        
        // Process claimed simulations
        for (int i = 0; i < claimed && !shutdown_.load(std::memory_order_acquire); i++) {
            try {
                traverseTree(root_);
            } catch (...) {
                // ... error handling ...
            }
        }
        
        // Tell evaluator we're done generating evaluation requests
        if (evaluator_) {
            evaluator_->notifyThreadDone();
        }
    }
}
```

### 4. Efficient GPU Tensor Creation

```cpp
torch::Tensor createBatchTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    if (states.empty()) {
        return torch::Tensor();
    }
    
    // Get dimensions from the first state
    const auto& first_state = states[0];
    int num_channels = first_state->getNumChannels();
    int board_size = first_state->getBoardSize();
    
    // Pre-allocate memory for all features
    size_t total_features = states.size() * num_channels * board_size * board_size;
    std::vector<float> all_features(total_features);
    
    // Fill features in a cache-friendly way
    size_t offset = 0;
    for (const auto& state : states) {
        std::vector<float> features = state->getFeatures();
        std::memcpy(all_features.data() + offset, features.data(), features.size() * sizeof(float));
        offset += features.size();
    }
    
    // Create tensor directly from all features
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor batch_tensor = torch::from_blob(all_features.data(), 
                                                 {static_cast<long>(states.size()), 
                                                  num_channels, 
                                                  board_size, 
                                                  board_size}, 
                                                 options).clone();
    
    // Move to GPU in one operation
    return batch_tensor.to(torch::kCUDA);
}
```

### 5. CPU-GPU Pipeline Optimization

```cpp
void processBatchWithOverlap(std::vector<EvaluationRequest>& batch, int max_pipeline_stages = 3) {
    // Split the batch into smaller chunks for pipelining
    size_t stage_size = (batch.size() + max_pipeline_stages - 1) / max_pipeline_stages;
    std::vector<std::vector<EvaluationRequest>> stages;
    
    for (size_t i = 0; i < batch.size(); i += stage_size) {
        size_t end = std::min(i + stage_size, batch.size());
        stages.emplace_back(batch.begin() + i, batch.begin() + end);
    }
    
    // Pipeline stages
    std::vector<torch::Tensor> stage_tensors;
    std::vector<std::future<std::pair<torch::Tensor, torch::Tensor>>> stage_futures;
    
    for (auto& stage : stages) {
        // Extract states in the CPU thread
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.reserve(stage.size());
        
        for (auto& req : stage) {
            states.push_back(std::move(req.state));
        }
        
        // Create tensor asynchronously
        auto tensor_future = std::async(std::launch::async, [&states]() {
            return createBatchTensor(states);
        });
        
        // Transfer previous stage to GPU while preparing the next stage
        if (!stage_tensors.empty()) {
            auto& prev_tensor = stage_tensors.back();
            prev_tensor = prev_tensor.to(torch::kCUDA, true);  // non-blocking transfer
        }
        
        // Wait for tensor creation
        stage_tensors.push_back(tensor_future.get());
        
        // Launch inference for the previous stage
        if (stage_tensors.size() > 1) {
            auto& prev_tensor = stage_tensors[stage_tensors.size() - 2];
            stage_futures.push_back(std::async(std::launch::async, [prev_tensor, this]() {
                torch::NoGradGuard no_grad;
                return neural_network_->forward(prev_tensor);
            }));
        }
    }
    
    // Process final stage
    if (!stage_tensors.empty()) {
        auto& last_tensor = stage_tensors.back();
        last_tensor = last_tensor.to(torch::kCUDA, true);
        
        torch::NoGradGuard no_grad;
        stage_futures.push_back(std::async(std::launch::async, [last_tensor, this]() {
            return neural_network_->forward(last_tensor);
        }));
    }
    
    // Process results for each stage
    size_t batch_offset = 0;
    for (size_t i = 0; i < stage_futures.size(); ++i) {
        auto [policy_tensor, value_tensor] = stage_futures[i].get();
        
        // Move results back to CPU
        policy_tensor = policy_tensor.to(torch::kCPU);
        value_tensor = value_tensor.to(torch::kCPU);
        
        // Get stage size
        size_t current_stage_size = stages[i].size();
        
        // Process results
        processResults(policy_tensor, value_tensor, batch, batch_offset, current_stage_size);
        
        batch_offset += current_stage_size;
    }
}
```

This implementation dramatically increases GPU throughput by:
1. Aggressively batching evaluation requests, even waiting to form larger batches
2. Using a dedicated batch collector thread to form batches while tree traversal continues
3. Coordinating worker threads to generate evaluation requests in groups
4. Creating efficient CPU tensors with minimal memory copies
5. Using a pipelined approach to overlap CPU tensor preparation, GPU transfer, and GPU computation

These optimizations address the critical bottleneck observed in the self-play log, where the system was processing batches of size 1 instead of utilizing the configured batch size of 128, severely underutilizing the GPU.
</gpu_throughput_scenario>

----------
----------

\<mcts\_algorithm\_analysis>

## MCTS Core Components and Interactions

**MCTSNode:** Each search node holds a game state and statistics for MCTS. It tracks visit count, cumulative value (for Q), prior probability, and uses atomic types for thread-safe updates. Child nodes are stored as shared pointers, and a weak pointer to the parent avoids reference cycles. Key methods include **`selectChild`** (PUCT selection), **`expand`** (adding child nodes for all legal moves), **virtual loss** management, and backpropagation update.

* *Selection & PUCT:* The implementation uses an AlphaZero-style PUCT formula with **virtual loss**. For each child, it computes an exploitation term (average value) and an exploration term using the stored prior and parent visit count. Virtual losses are applied by temporarily inflating visit counts and subtracting from value sum so that if another thread is exploring the same node, its score is lowered. This discourages parallel threads from choosing the same path. The code correctly reads the child’s `visit_count_`, `value_sum_`, and `virtual_loss_count_` atomically and incorporates virtual loss into the selection score. A design tradeoff here is that **no locks** are used in selection – using atomics allows lock-free parallel selection, but the PUCT calculation might read slightly stale values (which is usually acceptable in MCTS).

* *Expansion:* When a leaf node is expanded, all legal moves are generated and a new child node is created for each move. The game state for each child is cloned using a **GameStatePool** utility to reduce allocation overhead. (Currently, `cloneState` simply calls the state's own clone method, as true object reuse is not fully implemented.) Children are initialized with a **uniform prior** pending neural network evaluation. This is a straightforward expansion strategy – **progressive widening** (incremental expansion of children) is not explicitly implemented, despite being noted as a planned feature. The tradeoff is that expanding all moves at once is simpler and uses the network’s policy to guide all moves from the start, but it can waste effort on very large action spaces. The uniform prior assignment is quickly replaced when the network returns a policy, but until then all moves are treated equally.

* *Virtual Loss and Backpropagation:* The node holds an atomic `virtual_loss_count_`. When a thread selects a node for expansion/evaluation, it adds a virtual loss (or multiple) to that node to simulate a provisional loss outcome. This is done via `applyVirtualLoss(amount)` – by default an amount of 3 is used for a more aggressive discouragement. In backpropagation, the code increments the visit count and updates the value sum using atomic operations. It also **removes one virtual loss** from each node on the path as it backs up the value. **Correctness note:** Removing only one unit of virtual loss per node, when 3 were added at the leaf, appears to be a bug – a leaf’s `virtual_loss_count_` is decremented by 1 instead of resetting the full amount. This means if the code applied 3 virtual losses, two of them would remain, skewing future selections from that node (it would permanently think the node is worse than it really is). A fix would be to remove the same amount that was added (e.g. call `removeVirtualLoss()` in a loop or use `applyVirtualLoss(-3)` if implemented). Despite this, the backprop implementation correctly alternates the sign of the value as it propagates up the tree (since the game is zero-sum, the parent's perspective is the negative of the child's), and it applies the value update atomically to each node. The use of atomics (with compare-exchange for floats) ensures thread-safety at the cost of some retry overhead on contention. This lock-free update strategy is a design choice to maximize parallel throughput at the expense of more complex code; it avoids a mutex at each node update.

* *Transposition Table:* The engine uses a **transposition table (TT)** to avoid re-exploring the same state multiple times. Each expanded node’s state hash (using Zobrist keys) is stored in a global table mapping to the node pointer. If a new expansion generates a state that’s already in the table, the code **reuses the existing node** instead of creating a duplicate. This is done under a lock (internal to the TT’s parallel hash map) to ensure thread safety. The TT is bounded by a configured size (128 MB by default) and uses a parallel hash map (phmap) for efficiency. A design tradeoff here is that **storing weak pointers** is used (the TT stores `std::weak_ptr<MCTSNode>`) to avoid memory leaks – nodes can be freed if they fall out of the tree. The TT must be cleared at the start of each new search to drop pointers from the previous game, which the code does to prevent stale pointers. A potential improvement is to implement a replacement policy (e.g., LRU or based on depth/visits) when the TT is full, rather than clearing each search or letting it grow to capacity arbitrarily. Currently, every search begins by clearing the table, so the TT is used only within a single search instance (preventing long-term accumulation but also not retaining knowledge across moves). Within a search, however, it helps identify transpositions immediately after expansion, merging duplicate nodes. This **merging after expansion** (as implemented) can lead to subtle effects on counts: if a child node is replaced with an existing node that already has visits from another part of the tree, a parent might suddenly have a child with a non-zero visit count. The code doesn’t explicitly adjust the parent’s statistics in this case, but since the parent’s visit count was just incremented on expansion anyway, and the child’s stats carry over, it tends to be benign. The benefit is reduced memory and search effort by joining the subtrees, at the cost of some complexity in bookkeeping.

**MCTSEngine:** This class orchestrates the search, handles threads, and interfaces with the neural network. It maintains global counters and coordinates the **worker threads** for tree traversal and the **evaluator thread** for neural network inference. Key components of MCTSEngine include:

* **Search Coordination:** The engine uses an atomic counter `active_simulations_` to track how many simulations (playouts) remain to run. At the start of a search, `active_simulations_` is set to the total number of simulations (e.g. 800) to run. Worker threads pick simulations to execute by decrementing this counter in batches, rather than one by one, to reduce contention. For example, each thread might claim up to 64 simulations at a time (bounded by the remaining count and a minimum of 16) in a loop of compare-and-swap operations. This batch claiming is a performance optimization: it amortizes the cost of synchronization and better utilizes each thread’s cache locality by letting it perform several simulations in a row. The tradeoff is slightly more complex logic to ensure not to over-claim tasks, but the provided code handles this with a loop and CAS on `active_simulations_`. Once a thread has claimed some number of playouts, it runs them sequentially via `traverseTree(root_)` calls. Threads wait on a condition variable `cv_` when no simulations are pending or if the engine is paused. This design is effectively a custom thread-pool: it avoids the overhead of pushing tasks to a queue by having threads self-schedule using atomic counters. One design consideration is that if one thread finishes its batch while others still have large batches, there could be some load imbalance; but since batches are relatively small (max 64) and each simulation can vary in length, this is a reasonable compromise. The approach also ensures that **all CPU threads can work concurrently and even proceed to new simulations while earlier ones are waiting for neural net results**, enabling *leaf parallelization*.

* **Leaf Parallelization & Neural Network Integration:** Perhaps the most crucial part of the engine is how it integrates with the neural network for position evaluations. When a worker thread reaches a leaf that needs evaluation, it calls `expandAndEvaluate(leaf, path)`. Inside this function, if the game state is terminal, it returns the terminal value directly. Otherwise, it expands the node (generating children) and then **queues it for neural network evaluation** instead of evaluating immediately. The engine uses a lock-free queue `leaf_queue_` (moodycamel ConcurrentQueue) to collect these pending evaluations. Each pending evaluation is a struct containing the node pointer, the path (from root to leaf), and a cloned state ready for the NN. The engine marks the node as "being evaluated" by setting an atomic flag via `tryMarkForEvaluation()` to ensure no other thread will queue the same node twice. It then clones the game state (`cloneGameState`) for that leaf and moves it into the pending evaluation struct. Cloning here is a deep copy of the game state – this is necessary because the worker thread cannot keep the state reference (the node’s state) while it continues simulations, and the evaluation may happen asynchronously on another thread. Cloning is moderately expensive (especially for large boards), but they attempted to mitigate this with the GameStatePool (which currently just does a fresh clone in absence of an efficient reuse mechanism). The pending eval is then enqueued in `leaf_queue_`. Immediately after queueing, the worker thread **applies a virtual loss** to that leaf node equal to the configured amount (3 by default). This ensures that other threads exploring the same position will see it as provisionally worse and avoid it until the evaluation completes. The thread then returns without a value (using a dummy value 0.0 for now) – the actual result will be handled later asynchronously. Notably, the worker does *not* wait for the network result; it simply moves on to the next simulation (if any remain claimed). This non-blocking design is what enables **leaf parallelization**: many leaves can be awaiting evaluation in parallel, and CPU threads keep exploring other parts of the tree instead of idling.

* **Neural Network Evaluator Thread:** The engine launches a dedicated evaluator thread (`MCTSEvaluator`) to process the queued states in batches. In the current implementation, the engine opts to use **external queues** managed by MCTSEngine, integrated with MCTSEvaluator. Upon starting a search, the engine calls `evaluator_->setExternalQueues(&leaf_queue_, &result_queue_, callback)` to connect the evaluator to its internal queues. This means the evaluator thread will pull `PendingEvaluation` items from the engine’s `leaf_queue_` and push results into a `result_queue_`, rather than using its own internal request queue. The benefit of this design is that it avoids an extra copy of data and allows the engine to retain control over how requests are batched and distributed. The evaluator thread’s main loop (`evaluationLoop`) checks for new requests in the leaf queue and groups them into a batch up to a target batch size (configured via `MCTSSettings.batch_size`, e.g. 128). It uses a short timeout (in code, up to 10ms) to collect additional requests so that it can form larger batches instead of processing one request at a time. If no new items arrive within that window or the batch reaches the max size, it proceeds to inference. Each batch of `PendingEvaluation` is converted to a vector of game state pointers for the neural network: the code moves each `state` out of the struct into a `states` vector. These states are then passed to the network’s inference function in one call. The neural network (likely a PyTorch `libtorch` model under the hood) returns a vector of outputs, each containing a policy distribution and value. The evaluator pairs each output with the original node and path (still stored in the PendingEvaluation struct) and enqueues the result into the `result_queue_`. A callback then notifies the result handler thread that new results are available. This batching mechanism vastly improves GPU throughput when it works as intended – processing, say, 32 or 64 states at once is far more efficient on a GPU than sequentially evaluating one state at a time, due to parallelism and better amortized memory transfer costs.

* **Result Distribution and Backpropagation:** The engine spawns a **result distributor thread** (`resultDistributorWorker`) whose job is to take completed network evaluations from `result_queue_` and apply them to the search tree. This thread continuously dequeues results (it can take multiple at once in a batch for efficiency). For each result, it retrieves the node and the computed policy/value pair. It then **sets the node’s prior probabilities** to the network’s policy vector for that state, replacing the placeholder uniform priors set during expansion. Next, it calls `backPropagate(path, value)` to update all nodes along the path from that leaf up to the root with the evaluated value. During backpropagation, as discussed, one virtual loss is removed from each node and the value and visit count are atomically updated. The code also clears the node’s “evaluation in progress” flag so it can be evaluated again in the future if needed. After processing each result, the engine decrements its `pending_evaluations_` count and increments a counter for processed results. Once a result is applied, the virtual losses that were added are effectively negated, and the updated Q values will reflect the network’s evaluation. One subtle point: the code currently calls `node->clearEvaluationFlag()` to mark the node as free, but it **does not explicitly call** `removeVirtualLoss(settings_.virtual_loss)` on that node in the result thread. It relies on the backPropagate function to remove one virtual loss per node in the path. As noted, if more than one loss was applied, two would remain. This appears to be an oversight – ideally the result distributor should fully reset the virtual loss count for the evaluated node (or the backprop should remove the same amount that was applied). Despite this, the net effect is that the node’s value estimates are now based on a real network evaluation, and its children have valid priors, so the search can continue down this path with much more accurate guidance.

* **Concurrency and Thread Safety:** MCTSEngine carefully manages thread lifecycles. It uses flags like `shutdown_` and `workers_active_` to signal threads to stop when a search ends or the engine is destroyed. The destructor of MCTSEngine signals all threads to shut down, clears the queues (while resetting any evaluation flags on nodes that were never processed), and joins all threads to avoid detach or leaks. These measures are important to prevent memory leaks or crashes on program exit – e.g., they ensure no pending promise is left unfulfilled (the code in MCTSEvaluator’s destructor also tries to fulfill any leftover requests with default values to avoid broken promises). The design uses a combination of mutex/condition\_variable (for waiting when idle) and lock-free queues/atomics (for the hot paths of selection and evaluation). This hybrid approach is complex but aims at maximizing performance: critical sections like selecting a child or pushing to the eval queue are lock-free, whereas less frequent coordination points use locking. One noted issue is the **potential for race conditions** in certain scenarios – for example, two threads might select the same node to expand nearly simultaneously. The code locks a node’s expansion with `expansion_mutex_` to ensure only one actually expands the children. However, two threads could still both reach the leaf and one fails `tryMarkForEvaluation`, meaning it realizes another thread is evaluating that leaf. In the current implementation, that second thread simply returns without doing anything (effectively wasting that simulation). A more optimal approach would be to have the second thread continue searching a different leaf (for example, by restarting selection from the root) as suggested in the design docs, but that isn’t implemented yet – this is a tradeoff to keep the code simpler. It does not break correctness (other than losing a bit of search time) but indicates room for improvement in work allocation.

**Neural Network & Batch System (src/nn):** The neural network portion (`nn::NeuralNetwork` and its implementations) interacts with MCTS via the inference calls. The integration is done by passing a lambda capturing the `neural_net` into MCTSEngine’s MCTSEvaluator. For example, `neural_net->inference(const std::vector<std::unique_ptr<IGameState>>& states)` is called to evaluate a batch. The network likely uses libtorch to run a ResNet forward pass on the batch of states (e.g., converting each game state into tensor input). From the MCTS side, one important aspect is how game state data is converted to NN input. Currently, the code simply hands `IGameState` objects to the neural net. Under the hood, these objects probably have a method to convert themselves to a tensor (perhaps via a function like `state->encodeTensor()` or similar). In the improvement suggestions, the developers have considered optimizing this by pre-allocating tensors and using parallel loops to fill them (see below on GPU throughput). The batch size and timeout are configurable (e.g., batch\_size=128, timeout=20ms by default). A known problem is that in practice the **batch size is often stuck at 1** – this was noted in the documentation as a likely issue. The cause is that the evaluator thread grabs available requests immediately, and if requests arrive infrequently or one-at-a-time, it might process them one by one. In the current code, when an item appears in the queue, the evaluator does `if (queue.size_approx() > 0) processBatch(); else wait 500µs`. Inside `processBatch`, it does wait up to 10ms for more items, but if the first item arrives alone and no second arrives within a very short window, it will proceed with a batch of size 1. In short, the thread is *immediately* awakened on every new leaf (due to the `notifyLeafAvailable()` call), and it almost always finds the queue non-empty and thus processes without waiting much. This leads to many tiny batches. The design tradeoff here is between **throughput** and **latency**: the current settings lean toward low latency (processing quickly so as not to stall MCTS waiting), but at the cost of GPU efficiency. We’ll discuss optimization ideas for this below.

**Game Logic (src/core & games):** The MCTS implementation interacts with game-specific logic via the `core::IGameState` interface. This interface provides methods like `getLegalMoves()`, `makeMove(action)`, `isTerminal()`, `getGameResult()`, `getCurrentPlayer()`, and hashing. The engine is designed to be game-agnostic: Gomoku, Chess, and Go all implement IGameState, and the MCTS operates on those via polymorphism. A few utility components ensure efficiency here:

* **Zobrist Hashing:** Each game state provides a hash (likely a 64-bit Zobrist hash) via `getHash()`, used in the transposition table. Zobrist hashing is fast and has a low collision rate for board games, making it suitable for TT keys.
* **GameState Pool:** As noted, there is a `GameStatePoolManager` intended to reuse game state objects to reduce memory allocations. In practice, the current clone strategy does not actually reuse objects (it calls `state.clone()` which allocates a new object). The pool is initialized at the start of a search with a number of pre-allocated states (pool size = num\_simulations/4 by default). But the `cloneState()` method falls back to normal cloning because a safe in-place copy mechanism isn’t implemented. Thus, every expansion effectively does allocate a fresh state. This is a missed optimization opportunity; if each game’s state class had a method to copy data into an existing state (or a custom copy constructor), the pool could recycle objects. The impact is higher heap churn and GC pressure, but since all states are freed when the tree is cleared, this doesn’t cause a long-term leak – rather it’s a performance and memory-use concern. Additionally, the pool imposes a cap on stored objects (4x the initial size) and periodically purges if not used, to avoid unbounded growth. During MCTS, however, we might not actually release states back to the pool until the end of the search (the code doesn’t show explicit `release` calls after using a state for expansion or evaluation). The nodes keep their state unique\_ptr until the node is destroyed (at search end), at which time the state is deleted (and thus not returned to pool). One potential memory issue is that in long-running processes doing many searches, repeated new/delete of thousands of states could fragment memory or incur overhead. In summary, the game abstraction is clean and the MCTS code defensively validates state consistency at several points (e.g., checking `state.validate()` after cloning to catch any anomalies). Performance-wise, some improvements in how states are cloned and managed could yield benefits, as we’ll outline.

## Identified Issues and Trade-offs

1. **Batch Size often 1 (GPU Underutilization):** As mentioned, the current leaf parallelization sometimes fails to actually batch multiple evaluations. This was explicitly flagged in the requirements docs (batch size stuck at 1). The design prioritizes quick turnaround over waiting for a fuller batch. The trade-off is suboptimal GPU usage: the GPU may spend most of its time launching tiny inference jobs. This keeps MCTS threads from pausing, but the overall throughput suffers. Correctness isn’t affected, but performance is.

2. **Virtual Loss Handling Bug:** The partial removal of virtual losses means the search tree can accumulate phantom losses. This will bias the search away from certain nodes more than intended. If a node’s `virtual_loss_count_` remains >0 even after the evaluation is completed, no other thread will fully trust that node’s value for a while. This is a correctness issue in terms of search accuracy (though eventually, as visit counts grow, the impact of an extra few losses diminishes). It’s also easily fixable without downside – it seems to be an oversight rather than a deliberate trade-off.

3. **Thread Collision Waste:** The current approach to simultaneous expansion of the same node by two threads results in one thread doing useless work. This is a rare case but can happen, especially early in a search when many threads pick the root’s best move. The waste is minor (one simulation lost here or there), but it indicates that the parallel search could be more efficient. The trade-off was simplifying synchronization (only lock at expansion and evaluation, not at selection). Advanced approaches use **locking or atomic reference counts at each node during selection** to prevent any two threads from ever going down the exact same path. The code does not do that (except at the leaf eval step), likely for simplicity and performance. The impact is low on overall simulations per second, but it can slightly reduce the effective parallelism in those edge cases.

4. **Progressive Widening Not Implemented:** Although planned, the code expands all moves on the first visit. In games like Go (361 moves possible), this can be memory-heavy – expanding a node yields 361 children immediately, each with allocated state, etc. AlphZero in literature actually *does* expand all moves at once and relies on the neural prior to focus the search, so this isn’t strictly wrong. The progressive widening concept (expanding a few moves first, and more as visits increase) could save memory and concentrate search efforts, but it complicates the expansion logic. The current design’s trade-off is using the simpler approach: it uses more memory upfront per new node but ensures no move is overlooked entirely by the network. This is likely acceptable given the hardware target (64 GB RAM noted in requirements) and was perhaps deprioritized since the neural network policy helps prune the search implicitly by low priors.

5. **Memory Usage and Potential Leaks:** We did not find evidence of a classical memory leak (no continually growing memory footprint over many games, aside from expected caches). The engine carefully resets or frees structures each search. However, memory **usage** could be optimized:

   * Nodes and states account for the bulk of memory during a search. Thousands of nodes (each with a few atomic variables and vectors) are allocated. The use of `shared_ptr` for nodes means each node allocates a control block (approximately 16 extra bytes) and uses atomic ref counts. That adds overhead. A pool allocator for nodes or a custom object pool could cut down on allocation overhead and fragmentation.
   * The GameStatePool isn’t fully leveraged, as discussed. So we effectively allocate and destroy every state. Implementing state reuse (with proper deep copy) would reduce allocation churn. The trade-off here is the complexity of writing and maintaining a correct copy method for each game state (which can be error-prone if game state is complex). Given that the program targets long-running self-play sessions, reducing malloc/free cycles could improve performance and memory locality.
   * There is also GPU memory usage: the network predictor likely loads the model once (which can be hundreds of MB for a ResNet). That’s expected and not a leak. The code calls `torch::cuda::CUDACachingAllocator::emptyCache()` on destruction of the evaluator, which frees unused GPU memory back to the system. This is a bit aggressive (normally you’d keep the model in memory between searches), but it’s probably intended for clean shutdown or to release memory when switching models. It’s something to watch – calling emptyCache every time might hurt performance if done frequently (since reallocation is expensive), but in practice the evaluator persists for the whole search.

6. **NodeTracker vs. External Queue Duplication:** Initially, the design included `NodeTracker` for pending evaluations (with promises/futures), and a system where workers could potentially wait on futures for results. The current implementation shifted to using the external queue and a result thread, effectively bypassing `NodeTracker`. As a result, `NodeTracker` is now somewhat redundant (the engine registers nothing to it in the new flow). This doesn’t cause a bug, but it means there’s some dead code and slight memory overhead (e.g., the parallel hash map in NodeTracker reserved space for 10k entries on construction and never used it). The trade-off made was to switch to a simpler model (central result distribution) at the cost of leaving some unused code. Cleaning this up could free a bit of memory and reduce complexity.

7. **Concurrency and Termination Safety:** The engine goes to lengths to ensure a clean termination of threads. The use of atomic flags to signal shutdown and joining threads is correct. One thing to note is that the result distributor waits on `result_cv_` with a timeout of 100ms to check for new results. This is fine, though it introduces a potential 0.1s delay in shutdown worst-case (not a big issue). The termination sequence in `~MCTSEngine` forcibly wakes all threads and drains the queues to avoid any threads stuck waiting. This is a robust design and we did not find race conditions in termination. A minor detail is ensuring that the condition variables are not waited upon after their associated mutex is marked destroyed – the code handles this by flags like `cv_mutex_destroyed_`. One could consider using more modern tools (like `std::stop_token` with jthread in C++20) to simplify cancellation, but given C++17, this approach is acceptable. The complexity is high, but necessary for a bug-free multithreading system.

In summary, the MCTS implementation is feature-rich (transpositions, virtual loss, batched NN inference, multi-threading) and generally well-designed for performance. The main challenges observed are **tuning and correctness issues** in the parallelization: how to get larger batch sizes without stalling the search, how to manage memory for thousands of states/nodes efficiently, and fixing the small virtual loss bug. These form the basis of our optimization suggestions below, where we will prioritize solutions that yield the biggest improvement in throughput and memory usage.

\</mcts\_algorithm\_analysis>

\<todo\_list>

1. **Fix Batch Accumulation Logic:** Adjust the evaluator’s batching mechanism so that multiple leaf evaluations are aggregated per batch more often (the current setup often processes batches of size 1). This may involve introducing a short delay or minimum batch threshold before running inference, to boost GPU utilization.

2. **Implement Backpressure on Pending Evaluations:** Prevent the search from queuing excessive evaluations when the GPU is saturated. For example, pause or slow down tree expansion when the number of pending evaluations approaches a threshold (related to `max_concurrent_simulations` setting) to avoid memory bloat and ensure the batch sizes grow rather than endless single-state batches.

3. **Correct Virtual Loss Removal:** Modify the backpropagation or result handling so that all virtual losses added to a node are fully removed once the evaluation completes. Currently only one loss is removed per node despite adding multiple, which can skew search probabilities. This is a small code change (e.g., call `removeVirtualLoss()` in a loop or use the `amount` parameter) but will improve search accuracy.

4. **Optimize Node and State Memory Management:** Reduce overhead from frequent allocations:

   * Use the `GameStatePool` to recycle state objects by implementing a `copyFrom()` method in game state classes and using pool acquisitions instead of `new` on expansion. This will cut down on malloc/free churn during expansions.
   * Consider a custom allocator or object pool for `MCTSNode` allocations, or at least pre-reserve memory for large numbers of nodes. Recycling nodes between searches (if persistent engine) or using a memory arena could also help if feasible.
   * Ensure that freed states (after game end) are actually returned to the pool for reuse. Currently, states are cloned and destroyed with nodes, bypassing the pool.

5. **Simplify/Remove Unused NodeTracker Path:** Since the engine now uses its own queues and a result thread, the `NodeTracker` (with promises/futures) is no longer in active use. Removing this dead code will save some memory (e.g., 10k preallocated hash map entries) and eliminate confusion. If any functionality from NodeTracker is needed (e.g., easy retrieval of pending eval info), integrate it with the current system or ensure the external-queue path covers it.

6. **Improve Multi-Threaded Search Efficiency:** Address the scenario where multiple threads select the same node:

   * One approach is to add a check at selection time (before expansion) to mark a node as “in selection” or use an atomic flag to prevent a second thread from also selecting it. This, combined with the existing expansion lock, would fully eliminate wasted simulations. The downside is a bit more overhead in selection. We need to weigh if the added synchronization is worth it, since collisions are not extremely common. This is a lower priority than the issues above, but worth exploring for maximal efficiency.
   * Additionally, tune the `virtual_loss` hyperparameter. If we fix the removal bug, we might consider using a smaller virtual loss (like 1 instead of 3) because a large virtual loss can overly penalize a node’s value in selection. The optimal value might depend on number of threads.

7. **Transposition Table Enhancements:** The TT could be made more robust:

   * Use the configured `transposition_table_size_mb` to actually limit entries. For example, implement a **replacement strategy** (evict least recently used or shallowest nodes when capacity is reached). Currently, the table is cleared each search, but within a long search it could grow large. Sharding is already implemented for parallelism; we could add an entry count cap per shard and evict on overflow.
   * Track TT utilization and effectiveness (hit rate) during search and log it. This can inform if changes in policy (like storing only higher-value nodes or deeper nodes) might help.

8. **Logging and Instrumentation:** Increase the observability of the MCTS to guide further tuning:

   * The code already collects stats like average batch size and latency. Expose or log these per search or per time interval so we can verify if batch sizes improve after fixes.
   * Add debug counters for how often threads collide on the same leaf (e.g., how often `tryMarkForEvaluation()` returns false) to quantify the collision issue.
   * Monitor memory usage of nodes and states explicitly (perhaps extend `trackMemory` to include number of nodes in tree, etc.) to see the impact of pooling or other memory optimizations in real time.

Each of these to-do items targets a specific problem observed in the current implementation. The first two (batching and backpressure) are top priority for boosting throughput. The memory and node management improvements will help with scalability and long-term stability during self-play. The others (virtual loss fix, multi-thread tweaks, TT policy) will improve search accuracy and efficiency incrementally. We will next outline a concrete plan to implement these changes and discuss their pros and cons.

\</todo\_list>

\<optimization\_scheme>

## Core Architecture Optimization Plan

To address the issues above, we propose a set of coordinated changes to the MCTS system. The goal is to **improve GPU utilization and throughput** without sacrificing the clever parallelism already in place. We will maintain the overall structure (multiple tree traversal threads, one eval thread, one result thread) but tweak the synchronization strategy.

### 1. Smarter Batch Accumulation

We need to adjust the evaluator’s waiting strategy to form larger batches. A recommended approach is to implement a **two-phase wait** for batch assembly:

* **Phase 1:** When the first evaluation request arrives (leaf\_queue was empty and becomes non-empty), do **not** immediately dequeue it. Instead, start a short timer (for example, a few milliseconds). This can be implemented by having the evaluator thread, upon wakeup, know that it should wait a bit to allow more leaves to accumulate. Currently, `evaluationLoop` calls `processBatch()` as soon as it sees any item. We can change this to only call `processBatch()` if either a certain number of items are already in queue, or if a timeout since the first item has expired.
* **Phase 2:** If the queue fills up to `batch_size` or the short timeout elapses, then proceed with processing.
* Concretely, we could maintain an `int current_batch_count` and a timestamp for when the batch started accumulating. For example:

  ```cpp
  // Pseudocode for evaluator wait logic
  if (external_leaf_queue->try_dequeue(pending_eval)) {
      evaluations.push_back(std::move(pending_eval));
      if (evaluations.size() == 1) {
          batch_start_time = std::chrono::steady_clock::now();
      }
      // If batch not full, wait a bit for more
      while (evaluations.size() < batch_size_) {
          auto now = std::chrono::steady_clock::now();
          if (external_leaf_queue->try_dequeue(next_eval)) {
              evaluations.push_back(std::move(next_eval));
              continue;
          }
          // If no new eval, check timeout
          if (now - batch_start_time < std::chrono::microseconds(500)) {
              // active wait or sleep a few microseconds
              std::this_thread::sleep_for(std::chrono::microseconds(50));
              continue;
          }
          break;
      }
  }
  ```

  The above logic would replace the tight loop at with one that ensures at least 500 µs (in this example) of waiting for additional requests. We can tune this wait (or use the `batch_timeout` setting already in MCTSSettings). This change will directly tackle the “batch size 1” problem by sacrificing at most a fraction of a millisecond of latency to gain possibly 5-10x throughput if multiple leaves pile up. We expect minimal impact on move decision latency, since 10ms is the configured upper bound anyway, and often threads will produce several leaves within a couple of milliseconds especially mid-search.
* **Pros:** Greatly increases average batch size and GPU efficiency. The GPU will see larger tensor batches, improving occupancy and throughput. This also reduces the relative overhead of launching kernel and CPU/GPU transfer per inference.
* **Cons:** If the search is running in a scenario where extremely low latency per move is required (e.g., real-time play with very low time per move), this could introduce a slight delay. However, the 20ms timeout was already in place, so using a few milliseconds for batching is within original design parameters. We must ensure this doesn’t stall the search threads: but since they don’t wait for results (except possibly at the end of search), a small eval delay is usually fine.

Additionally, we can make the batch size **adaptive**: The code has some logic for `min_batch_size_` in internal mode (adjusting if average batch is much smaller or larger than target). We should extend or reuse this in external mode. For example, if we consistently see only 2-3 requests per batch, we might reduce the `batch_timeout` or adjust scheduling; if we see the queue is often full (batch\_size items waiting), maybe we can increase throughput by raising `batch_size` (if GPU memory allows). An adaptive scheme can monitor `cumulative_batch_size_ / total_batches_` (average batch) and dynamically tweak waiting time or `min_batch_size`. This is a more complex enhancement – a simpler first step is implementing the fixed short delay, which addresses the immediate issue.

### 2. Backpressure and Concurrency Control

Currently, all threads will happily continue to add leaf evaluations up to `num_simulations`. If the neural net is slower, this could lead to a buildup of pending evaluations (though the design tries to push throughput, one could temporarily have many pending). We have a `max_concurrent_simulations` setting (512) that isn’t yet enforced in code. We should enforce it by limiting `active_simulations_` to that value at any time.

A possible implementation:

* When initializing `active_simulations_`, do `active_simulations_.store(std::min(settings_.num_simulations, settings_.max_concurrent_simulations))`. If `num_simulations` is larger, we will treat the search as in waves. After one wave finishes, we could then trigger the next wave. However, since the engine currently decrements all the way to 0, a better approach is:
* Keep `active_simulations_ = settings_.num_simulations`, but use an atomic `pending_evaluations_` (already present) to decide if we should pause spawning new ones. E.g., in `treeTraversalWorker`, before selecting a new leaf, check `pending_evaluations_ < settings_.max_concurrent_simulations`. If that condition is false (meaning too many evals in flight), the thread can wait a bit (yield or sleep) until some evaluations complete. This essentially throttles the tree expansion when the GPU is overloaded.
* We can implement this by modifying the loop that claims simulations. For instance, at the top of the while loop in `treeTraversalWorker`, add:

  ```cpp
  int pending = pending_evaluations_.load(std::memory_order_acquire);
  if (pending > settings_.max_concurrent_simulations) {
      // Too many evals in flight, wait for results to catch up
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
  }
  ```

  and then proceed to check `active_simulations_`. This ensures that if the GPU is the bottleneck and a lot of leaves are waiting, the CPU threads pause briefly instead of piling up even more. The sleep can be very short (1ms or even just yield) because results will come in and wake threads via `result_cv_`.
* **Pros:** This prevents memory explosion (each pending eval carries a whole game state and path). It also indirectly increases batch size: if threads hold off when many are pending, they will naturally form a larger queue for the evaluator to work on, instead of constantly adding new single items.
* **Cons:** This means CPU threads might be idle at times, which could under-utilize CPU in scenarios where the GPU is under-utilized too (we need to be careful to only stall when the GPU truly has a backlog). Tuning the threshold is key –  if set too low, we throttle unnecessarily; if too high, we might still get big memory spikes. The chosen default of 512 seems reasonable for a single 3060 Ti in self-play (it’s unlikely the GPU can effectively handle more than 512 concurrent evaluations in a timely manner anyway). This number can be made configurable or adaptive (perhaps related to batch\_size: e.g., max pending = 4 \* batch\_size).

### 3. Memory Management Improvements

**GameState Pool usage:** We should modify `cloneGameState` to actually reuse `IGameState` objects. A plan:

* Implement a method in each game’s state class like `copyFrom(const IGameState& other)`. This would copy all relevant fields (board configuration, current player, etc.) from `other` into `this` without altering pointers that shouldn’t be copied. This requires careful implementation in chess, go, gomoku states but is doable (since they likely already have a `clone()` that uses the copy constructor).
* Change `GameStatePool::clone(const IGameState& source)` to try acquiring an object from the pool and then copying into it:

  ```cpp
  std::unique_ptr<IGameState> GameStatePool::clone(const IGameState& source) {
      if (!pool_.empty()) {
          auto state = acquire(); // get a blank state
          state->copyFrom(source);
          return state;
      }
      // if pool empty, either create new or fallback
      return source.clone();
  }
  ```

  We must ensure that the acquired state is of the same concrete game type as source (the pool is segregated by game type in `GameStatePoolManager`, so that holds). This way, instead of allocating a new state for every child, we reuse one from the pool.
* Modify `MCTSNode::expand()` to release states back to the pool when a node is destroyed. One idea: intercept node deletion – perhaps via a custom deleter in the `shared_ptr<MCTSNode>` or at least have \~MCTSNode release its state:

  ```cpp
  MCTSNode::~MCTSNode() {
      if (state_) {
          // Return state to pool instead of deleting
          utils::GameStatePoolManager::getInstance().release(std::move(state_));
      }
  }
  ```

  This would put the state back into the pool for reuse in future expansions. Because nodes are only destroyed when the whole tree is torn down (or perhaps when transpositions replace them, but in that case the pointer is not destroyed, just not used), most states will be released en masse at end of search. That’s fine, the pool will then have a warm cache of state objects for the next search.
* **Pros:** Recycling states can dramatically cut the number of allocations. In a large search with, say, 100k node expansions, we might allocate 100k state objects normally. With pooling, we allocate perhaps up to the peak concurrent states (which might be equal to number of nodes in the tree at one time, possibly a few tens of thousands), and then reuse them next game. This reduces pressure on `malloc` and could improve cache usage (objects are reused, likely staying in memory). It also helps avoid fragmentation over many games.
* **Cons:** Implementing `copyFrom` for each game state is error-prone – one must ensure every bit of game state is copied exactly (including history if any, castling rights in chess, Ko state in Go, etc.). Any bug in copyFrom could be disastrous (leading to incorrect game states). Thorough testing is required. If confident in that, the payoff is worth it. Another con is that releasing states at node destruction means during the search, we don’t reclaim memory (because nodes persist until end). That’s fine – the memory is needed while tree exists. But if memory becomes tight, one could consider more aggressive pruning of the tree (not in current scope) or releasing portions of it, which complicates things. For now, releasing at end of search is acceptable.

**Node allocation:** Using `shared_ptr` for every node has overhead. We could consider using a custom memory pool (like an `ObjectPool<MCTSNode>` that allocates chunks of nodes). But switching away from shared\_ptr would require managing lifetimes differently. Since the code relies on shared\_ptr to handle complex ownership (especially with transposition table storing weak\_ptrs), we might keep it but try to reduce overhead:

* We could reserve space for children vectors to the maximum legal moves of the game to avoid re-allocations. Actually, the code already reserves the children vector to `legal_moves.size()` on expansion, which is good.
* The biggest per-node overhead is likely the atomic<float> for value (which might cause padding or lock contention if many are in same cache line). We can’t easily change that without locks. An atomic<float> is fine (in C++20 there is `atomic_ref<float>` which could operate on a float in a cache line without separate allocation, but not widely used yet). We could consider accumulating values in an `double` for more precision and then cast to float for policy, but that’s micro-optimization.
* One idea: use **intrusive shared pointers** (boost intrusive\_ptr or similar) for MCTSNode, so that the refcount is a field in the node (could reduce separate allocation). This would remove one atomic per node (the control block’s refcount) and instead put an atomic in the node structure for refcount. It saves one allocation per node and some pointer chasing. However, this change is fairly invasive and may not yield a huge difference unless node count is extremely high.
* Given the complexity, node pooling or intrusive ptr might be overkill. A simpler step: ensure that `transposition_table_` doesn’t keep nodes alive longer than needed. It stores weak\_ptr, so that’s fine. We just need to be sure to call `transposition_table_->clear()` at game end (the code does this).
* In summary, node-level optimizations are a bit more involved and yield smaller gains than the state pooling. So we prioritize state pooling first (bigger impact on memory).

**Removing NodeTracker:** On the code cleanup side, we can remove the NodeTracker instance or set `node_tracker_ = nullptr` when using external queues to avoid any unintended overhead. NodeTracker’s map was never populated in this flow, so it’s not doing much harm, but freeing that reserved memory (10000 entries) at engine start might be nice. This is mostly a maintainability improvement.

### 4. Parallelization and Threading Improvements

**Collision reduction:** As noted, currently if two threads try to evaluate the same leaf, one backs off (losing that simulation). We can enhance this by detecting the situation earlier. One method:

* When selecting a child in `MCTSNode::selectChild`, if a child is found with `evaluation_in_progress_ == true` (meaning another thread has queued it for eval and not finished yet), we could choose to skip that child in the scoring (treat it as temporarily invalid for selection). Presently, `selectChild` doesn’t explicitly skip nodes being evaluated (it only indirectly lowers their score via the virtual loss that was applied). If virtual loss is large, that might be enough; but if the node had a very high prior, even subtracting 3 might not stop another thread from selecting it. To strengthen this, we could modify `selectChild` to continue the loop without considering a child if `child->isBeingEvaluated()` returns true. The code has `evaluation_in_progress_` flag and related methods. This would explicitly prevent threads from selecting a node under evaluation. The tradeoff is that if a node is being evaluated, all threads will ignore it until it’s done, exploring other moves instead – which is actually good (this is effectively what virtual loss is meant to do, just making it binary here).
* Also, as mentioned, if a thread finds `tryMarkForEvaluation()` returns false (meaning someone else is already evaluating this leaf), instead of returning 0 and ending, we could have it go back up and choose a different leaf. For example, detect the condition and call `traverseTree(root_)` again (effectively restarting the simulation). The pseudo-code from our design docs suggests doing that. Implementing that in `expandAndEvaluate` might require a loop or a higher-level check in `traverseTree`. Since our current `treeTraversalWorker` loop just calls `traverseTree` for each claimed sim, we could catch a special return code from `traverseTree` indicating “node in progress, try again”. This would complicate the interface slightly but is doable. However, given the overhead of a full simulation, losing one simulation occasionally is not dire. So this is a nice-to-have improvement. It ensures full utilization of sim count, but in practice, it might not change outcomes significantly. Still, it’s worth implementing if time allows, as it aligns with making the algorithm more elegant.
* **Pros:** These changes reduce wasted computation and ensure each simulation contributes useful information. It will also slightly reduce variance in multi-threaded results (because two threads won’t double-explore the same path).
* **Cons:** Skipping an evaluated node entirely (as opposed to just virtual loss penalty) could in rare cases cause the algorithm to explore suboptimal moves just because the best move’s evaluation is pending. However, since the evaluation will finish quickly (tens of ms), and then that node becomes available with a real value, this is fine. It’s actually closer to how humans would avoid reconsidering a position until new info arrives. The impact on strength is likely neutral or positive.

**Work stealing:** The current batch-claiming of simulations might leave some threads idle if tasks not evenly divisible, but the implementation already addresses that by each thread continuously checking `active_simulations_`. There is an implicit work stealing: any thread that finishes its batch will loop again and try to claim more from `active_simulations_`. So the distribution is fairly dynamic. One improvement could be to reduce the batch size as the number of remaining simulations shrinks, to avoid overshooting. But the code already does `to_claim = min(batch_size - claimed, old_value)`, which handles that.
If we wanted a more standard approach, we could instead push “simulation tasks” into a concurrent queue and let threads pop them (this is basically what active\_simulations does, but using CAS rather than actual task objects). There’s no clear evidence that switching to an explicit task queue would be faster – in fact, the CAS approach is likely lower overhead.

One possible enhancement is to allow *dynamic adjustment of thread usage*: for instance, if the batch evaluator becomes a bottleneck (GPU is busy and many evals pending), one could in theory reduce the number of tree threads or have some threads help in processing results. However, dividing roles too much could complicate things. The current separation (tree threads vs eval thread vs result thread) is clean and typically one eval thread can handle the load if batching is effective. So we won’t complicate that now.

### 5. Transposition Table Policy

To optimize the transposition table:

* We can initialize it with a number of shards equal to thread count (the code already does something like that when resizing). They chose `num_shards = max(4, hardware_concurrency)` by default. If we have 16 threads, 16 shards is okay.
* Implementing a proper replacement strategy: For example, use an LRU queue per shard or store a timestamp or visit count and periodically prune least-used entries. A simple approach could be to limit the total entries. 128 MB for a table of weak\_ptrs (8 bytes for pointer, plus overhead) could store on the order of a million entries (roughly estimating). It’s unlikely a single search will hit that many unique states unless we allow extremely large number of simulations or in Go 19x19 with huge search. But if it does, performance might degrade (large hashing overhead). Having an upper bound (like 1e6 entries) and then refusing to add new entries or randomly dropping some might be acceptable.
* Another idea: do not store very shallow nodes in the TT, because they are likely near root and won’t have duplicates anyway except symmetrical moves. Focus TT on deeper states where transpositions matter (especially in games like Go or chess with repetitive positions). This can be done by storing depth and maybe not replacing entries with depth less than current if collision occurs.
* These changes can get complex, so one pragmatic step: keep an `entry_count` and once it exceeds (capacity \* 0.9), clear a portion (like clear half of entries, or all – but clearing all loses info). A better approach: use a ring buffer of TT entries to evict oldest. The phmap might not support easy removal except by key.
* For now, given time, we might leave TT as is, since it’s functioning and not the primary performance bottleneck. Just note these improvements for completeness. The main TT-related action item is to ensure it’s not causing memory leaks (it isn’t, since cleared each search, and uses weak\_ptrs so it doesn’t keep nodes alive).

### 6. Testing and Verification

After implementing these optimizations, careful testing is needed:

* **Functional tests:** Ensure that with the virtual loss fix and new scheduling, the search results (e.g., distribution of moves) remain reasonable. The virtual loss bug fix might cause the search to explore previously “over-penalized” nodes more, which is a correct change. We should see search outputs become more stable with multiple threads.
* **Performance tests:** Measure average batch size before and after changes. We expect the average to go from \~1-2 to perhaps 8-16+ in self-play conditions (depending on how many leaves are expanded quickly).
* Monitor that the throughput (playouts per second) increases on GPU. For example, if originally we had \~1000 simulations/sec with batch size \~1, and each inference taking \~3ms, the GPU was mostly idle. After changes, we might see the same 1000 sims now processed in, say, 30% less wall time if batches of 8 or more are used (rough estimate).
* **Memory usage:** With pooling, track memory via `trackMemory`. Ideally, we see memory usage plateau instead of climb across games. If we see memory stable at, e.g., 2GB after many games whereas before it climbed, that’s a success. We should ensure no double-delete or use-after-free occurs with the pool (test with address sanitizer perhaps).

By addressing the batch throughput and memory reuse, we tackle the biggest slowdowns. The MCTS will be able to utilize the GPU to evaluate many states in parallel, while not running far ahead of itself with too many pending states. These changes, combined with minor fixes (virtual loss) and code cleanup, set the stage for a much more efficient AlphaZero engine.

\</optimization\_scheme>

\<parallelization\_improvements>

## Enhanced Parallelization Strategy

To maximize parallel efficiency, we concentrate on the leaf evaluation pipeline and thread coordination:

**1. Improved Leaf Parallelization:** The current design already queues leaf evaluations asynchronously; our focus is on making this more effective.

* **Batching Leaves:** As described, we now accumulate leaf nodes for a short time to submit a bigger batch to the GPU. This means threads might queue up, say, 10 leaves in a short period, and the evaluator will process all 10 in one go. The improvement here is that the GPU does 10 evaluations nearly as fast as 1 (due to parallelism), effectively giving \~10x throughput for that moment. Meanwhile, the CPU threads that submitted those leaves continue working on other parts of the tree. This overlap of CPU (selection/expansion) and GPU (evaluation) is true leaf parallelism in action – each are busy doing their share of work simultaneously. By increasing the batch size, we ensure the GPU is not idle and the CPU is not starved waiting.
* **Parallelizing State Encoding:** If the state-to-tensor conversion is costly, we can parallelize that using multiple CPU threads or vectorization. For instance, using OpenMP in the loop that fills the input tensor from game states (as hinted by possible code in the docs). This way, preparing a batch of 64 states might be split across 4 CPU cores, reducing wall time for batch preparation. Since the evaluator thread is single, we could allow it to spawn an OpenMP parallel for to utilize multiple cores for encoding, which is fine because the tree search threads are anyway busy elsewhere. This yields better overlap and utilization of all CPU cores (some working on MCTS, some on preparing NN inputs).
* **Leaf Expansion Parallelism:** We ensure that only one thread expands a given node’s children (via the `expansion_mutex_` in MCTSNode), which is necessary to avoid duplicate children. This is effectively parallel – different threads expand different nodes concurrently. By the time the search is in mid-simulation, many threads will be expanding different parts of the tree at once. That scaling should continue linearly with threads, limited only by occasional mutex collisions (rare) or fighting over the same best move (mitigated by virtual loss).
* **Avoiding Redundant Evaluations:** With our improved checks, no two threads will evaluate the same leaf state twice. This was mostly true before (due to `evaluation_in_progress_` flag) but now it’s reinforced by skipping nodes under evaluation entirely during selection. This means all threads always work on distinct leaves, maximizing coverage of the search space and not wasting precious neural net evaluations on repeats.

**2. Thread Synchronization Refinements:**

* **Condition Variable Utilization:** We use `cv_` to put threads to sleep when no work is ready. This prevents busy-waiting and saves CPU cycles for when they can be used better (like encoding states or other OS threads). We’ve added more granular wake-ups: e.g., when results come in, we notify worker threads that some pending evaluations finished by incrementing `active_simulations_` or simply by the result thread notifying `cv_` after processing results. In the current code, after each result batch, they do `cv_.notify_all()` in case threads were waiting for `active_simulations_` to become non-zero. We maintain this, and also notify if we implement any pauses for backpressure. The backpressure check (if too many pending, threads sleep 1ms) is a light form of synchronization; it doesn’t use a CV, but it’s a short sleep that yields CPU. This is acceptable since it’s a rare scenario (only when 500+ evals are pending).
* **Work Stealing Behavior:** As noted, our thread loop with atomic counter is akin to work stealing – any free thread will decrement the global counter and take on work. By chunking the work, we reduce contention. Our modifications don’t remove this logic; in fact, by pausing threads when GPU is overloaded, we implicitly allow other threads (or the same thread a bit later) to steal work once the overload subsides. It’s like saying “don’t grab more tasks until the current ones are partly done.”
* **Thread Roles and Core Affinity:** We could consider pinning the evaluator thread to a specific core (especially if it’s doing heavy Torch work and maybe can benefit from being pinned to avoid migrating between cores). Similarly, tree traversal threads could be pinned or at least given high priority. The current implementation doesn’t specify this, except naming threads for debugging. As an improvement, one might use `pthread_setaffinity_np` on Linux to bind threads (e.g., evaluator to core 0, result to core 1, workers to cores 2+ etc.). This could reduce context switching cache misses. It’s an advanced optimization that might yield minor improvements in throughput stability.

**3. Multi-GPU or Distributed Parallelization (Future scope):** Although out-of-scope per PRD, it’s worth noting that the current architecture would allow scale-out if needed. For example, if we had two GPUs, we could run two evaluator threads, each taking from the same leaf queue (or partition the queue) to process evaluations in parallel. Our backpressure logic and batch accumulation would need extension to handle multiple concurrent batches. This is a larger change but the concept of external queues is extensible: one could imagine N evaluator threads pulling from one ConcurrentQueue of states – moodycamel’s queue is thread-safe and could feed multiple consumers (though we’d need to be careful to not give the same item twice; maybe use separate queues per evaluator). Similarly, more tree threads beyond CPU cores typically saturate returns less benefit due to diminishing returns of simulation throughput vs eval throughput – thus focusing on GPU parallelism is key.

**4. Leaf Parallelism Verification:** With these improvements, we expect to see:

* Multiple pending evaluations in flight nearly all the time (especially early in search). The `pending_evaluations_` counter should oscillate around a target (maybe around max concurrent or at least frequently above 1). If initially it was often 0 or 1, now it should often be, say, 5, 10, or higher, indicating real parallel eval going on.
* CPU utilization balanced: some cores busy in MCTS, one busy feeding GPU, one busy applying results. Ideally all cores have something to do. If the GPU becomes the bottleneck (which is likely when batch sizes are large), the CPU might throttle as designed – which is fine, because it means we’re maxing out the GPU and not just piling work that can’t be completed faster.

In conclusion, these parallelization improvements create a more harmonious pipeline:

* Tree search threads produce work as fast as the GPU can consume it (without significantly overproducing).
* The GPU is kept fed with batches of states, improving its efficiency.
* No single thread becomes a bottleneck: the eval thread does a lot of work but if it falls behind, tree threads slow down slightly until it catches up (instead of overwhelming it).
* The result thread offloads all backup computations so that tree threads remain focused on expansion and selection.

This balanced approach should yield near-linear scaling in throughput with respect to CPU cores (up to the point where the GPU becomes the limiting factor, at which point additional CPU threads don’t help because the eval is the bottleneck). At that GPU-bound point, our backpressure ensures we don’t waste memory or CPU on fruitless extra simulations. Instead, any extra CPU could possibly be redirected to other tasks (or simply remains idle, which in a self-play context is fine since GPU is the critical resource).

In summary, **leaf parallelization** is fully leveraged – many leaves can be evaluated in parallel – and **thread coordination** is tightened to avoid collisions and wasted effort, making the Monte Carlo Tree Search more efficient in a multi-threaded, GPU-accelerated environment.

\</parallelization\_improvements>

\<gpu\_throughput\_scenario>

## Maximizing GPU Throughput

To illustrate the impact of our changes, consider a typical self-play scenario on a single GPU:

**Initial Scenario (Before optimizations):**

* Batch size effectively = 1 most of the time. A leaf arrives, the evaluator immediately processes it. The GPU performs one forward pass (\~3-5 ms), then idles while waiting for the next.
* Suppose 100 leaf evaluations are needed. The GPU does 100 separate jobs, incurring overhead each time. If each job has \~1 ms of launch overhead and \~3 ms of compute, that’s 4 ms \* 100 = 400 ms total.
* During this time, CPU threads continue to churn out leaves, possibly building a backlog or overshooting (or if they were waiting on results, they might under-utilize GPU).
* GPU utilization might be very low (e.g., 20-30%) because it’s mostly handling small inference jobs and waiting in between.

**Optimized Scenario (After improvements):**

1. **Tensor Preallocation and Memory Pinning:** We set up preallocated GPU tensors for common batch sizes. For example, allocate a tensor of shape \[256, C, H, W] on the GPU at startup (where 256 is a max batch we expect, C channels, H,W board dims). Also prepare pinned CPU memory buffers for batches. This way, we **avoid allocating new tensors for each batch**. Instead, we copy data into the existing tensor slice. PyTorch (libtorch) allows using `.index({Slice(0, batch_size)})` to get a sub-tensor view without copy. We do this:

   * Preallocate e.g. for batch sizes 16, 32, 64, 128 (some powers of two up to max). Or simply one large and use a slice as needed.
   * Use page-locked (pinned) memory for the CPU staging area, so that CPU->GPU DMA is faster.
   * The pros: no dynamic allocation during search (less CPU overhead, more deterministic). Also pinned memory can significantly speed up transfers (2-3x faster host to device copies).
   * Cons: preallocating large tensors uses more memory up front (but a 256-batch of 19x19 Go might be on the order of 256*19*19*features*4 bytes, which is maybe a few tens of MB – fine on a 8GB GPU).

2. **Batch Filling Parallelism:** When a batch of states is ready, we convert them to tensor format. We can parallelize this loop across available CPU cores. For instance, with OpenMP:

   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < batch_size; ++i) {
       auto tensor_view = cpu_tensor[i];  // slice per state
       states[i]->fillTensor(tensor_view);
   }
   ```

   Each state’s `fillTensor` writes the board representation into the provided tensor slice (which is a view into the bigger tensor). If we have 8 cores and batch of 64, each core handles 8 states in parallel, possibly bringing batch preparation down to the time of \~8 states instead of 64. This overlaps well with other computations.

3. **Single Copy to GPU:** After filling the pinned CPU tensor, issue one asynchronous transfer to GPU for the whole batch. This is far more efficient than 64 small copies. The GPU can DMA the 64 states in one block. If using CUDA streams, this copy can even overlap with the GPU executing the previous batch’s network forward pass (though that requires stream management beyond current scope, but could be done).

   * Because we preallocated a large GPU tensor, this copy is actually a device-to-device copy if we directly fill the GPU tensor from CPU. Alternatively, we fill a pinned CPU tensor and call `cudaMemcpyAsync` to the preallocated GPU tensor slice.
   * The network forward is then invoked on the GPU tensor slice containing the batch.

4. **Network Forward Pass (Batch Inference):** The GPU now computes the policy and value for the entire batch simultaneously. Modern GPUs are very efficient at batch computations – the fixed overheads (kernel launch, memory global latency, etc.) are amortized. The compute units are fully engaged. For a ResNet model, batch size 64 might use, say, 60% of GPU compute capacity for 3 ms, whereas batch size 1 might use 5% for 3 ms – essentially, 64 can be done in about the same time as 1 in many cases, or slightly more if memory-bound, but certainly not 64x more.

   * Example: Let’s say one state takes 3 ms, 64 states might take 4 ms on the GPU. So throughput per state improves by \~16x.
   * The result is a vector of 64 NetworkOutput objects. We copy those back to CPU (this is small: just probabilities and value per state, maybe a few hundred bytes each, negligible time). Even this copy can be overlapped if using streams.

5. **Result Integration:** The result thread takes those 64 outputs and quickly updates the tree. This part is on CPU and was relatively fast anyway (backprop 64 values is minor compared to what the GPU did). The result thread can batch these updates too (the code already processes results in batches of up to 32). We could increase that batch to 64 to match a large eval batch if needed. The backprop loop is O(depth) per result; depth maybe \~50 average, so 64\*50 = 3200 node updates, which is fine in microseconds range due to atomic ops.

6. **Throughput Gains:** In this scenario, those 100 leaf evaluations from earlier might be processed as 10 batches of 10 instead of 100 of 1. Let’s do rough math:

   * Without optimization: 100 \* 4 ms = 400 ms GPU time.
   * With optimization: maybe \~10 batches \* (copy + forward + overhead):

     * Copy 10 states to GPU: maybe 0.2 ms (for small data, if pinned).
     * GPU forward 10 states: maybe 1.5 ms (since GPU likes at least a little batch).
     * Total \~1.7 ms per batch \* 10 = 17 ms.
   * This is a **huge improvement**, albeit this is optimistic. More realistically, if we manage average batch of 32:

     * 100 states = \~4 batches of 25 (roughly).
     * Each batch 25 might take \~2.5-3 ms on GPU.
     * Total \~10-12 ms vs 400 ms originally. That’s \~30-40x faster for the evaluation part.
   * The overall MCTS loop then becomes CPU-bound likely, which is fine – means GPU is no longer the limiter except in brief bursts.

7. **GPU Utilization:** With larger batches, the GPU utilization jumps. Instead of many small idle periods, the GPU will see steady streams of work. It might reach 80-90% utilization if the MCTS is continuously feeding it batches. This is ideal as we paid for the GPU – we want it working. The only caveat is if we oversaturate, which our backpressure prevents. So we target a balanced pipeline.

8. **Monitoring:** We will measure metrics like:

   * **Average batch size:** Suppose this climbs from \~1.5 to \~20 after our changes.
   * **Average GPU inference latency:** It might go from \~3 ms (for a single state) to \~5 ms (for 32 states) – a slight increase per batch, but per state latency drops massively.
   * **Evaluations per second:** Initially maybe \~300 states/sec on GPU (since 3ms each). After, potentially \~2000+ states/sec (if we can batch 32 at a time in \~5ms, that’s 32/5ms = 6400/sec just for GPU throughput, but CPU and other factors will lower it, still on order of thousands).
   * These improvements mean we can either run more simulations in the same time (improving playing strength) or achieve the same number of simulations much faster (saving time or energy).

9. **Pros/Cons Recap:**

   * *Pros:* Maximizing hardware utilization, significantly faster self-play generation (games completed per hour skyrockets), ability to either reduce think time or increase search depth. This directly translates to stronger AI play given a fixed time budget, since more simulations = better policy convergence.
   * *Cons:* Slightly more complex code (managing preallocated buffers and multi-threading encoding). Higher memory usage due to buffers (but still minor relative to a 64GB RAM or GPU memory). Also, when batch sizes increase, the first move in a game might take a bit longer to gather a big batch (though we cap waiting at e.g. 20ms), but subsequent moves in self-play pipeline usually overlap anyway (one game’s thinking time overlaps with another’s training perhaps, depending on pipeline design).

In essence, the scenario after optimizations shows the GPU as a true workhorse handling many positions in parallel, while the CPU coordination ensures that those positions are generated and processed in a timely manner. The net effect is that the overall system can evaluate far more positions per second than before, increasing the strength of the MCTS-based AI without changing the neural network itself.

Finally, consider an extreme test: If we have 16 threads and a powerful GPU, the engine might consistently fill batches of 128. The GPU might handle that in \~8-10 ms. The CPU might produce those 128 leaves in roughly that time as well (16 threads \* maybe 8 expansions each in \~10 ms). This would mean every 10 ms we finish 128 simulations – that’s 12,800 simulations per second, an impressively high number. These numbers are hypothetical, but they indicate the headroom unlocked by proper batching and parallelization. Even if actual performance is a fraction of that, it’s a big win over the unoptimized case.

**Conclusion:** By implementing the steps above (preallocation, parallel encoding, batch processing), we transform the GPU from a sporadically-used component to a fully engaged accelerator that significantly speeds up the MCTS evaluations. This **high-throughput scenario** is critical for self-play training, where thousands of games must be generated – it means more training data in less time, or achieving a higher Elo policy network given the same training duration.

\</gpu\_throughput\_scenario>
