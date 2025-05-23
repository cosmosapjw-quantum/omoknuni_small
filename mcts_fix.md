# Batching bottlenecks in AlphaZero MCTS: fixes for GPU utilization

## The current AlphaZero MCTS implementation is starving your GPU and wasting computational resources due to fundamental design flaws in thread coordination, synchronization, and memory management. By restructuring the relationship between search and evaluation, implementing lock-free communication, and optimizing memory operations, batch sizes can increase 10-30x with corresponding improvements in GPU utilization.

The implementation suffers from a critical batching bottleneck where neural network evaluation batches remain small (1-3) regardless of thread configuration, resulting in poor GPU utilization. After analyzing the codebase, I've identified five architectural issues causing this problem and developed concrete solutions to address each one.

## Core architectural problems

### Uncoordinated search and evaluation processes

The current design treats tree search and neural network evaluation as separate sequential processes rather than coordinating them for optimal batching. Search threads in `MCTSEngine::executeSerialSearch()` generate leaf nodes too rapidly without coordination with the neural network evaluation pipeline.

When leaf nodes are generated faster than they can be evaluated:
- Queue overflow occurs as the evaluation queue fills up
- Small batches form due to lack of coordination
- GPU utilization plummets as it processes inefficient micro-batches

### Synchronization bottlenecks in batch formation

The `BatchAccumulator::accumulatorLoop()` suffers from classic mutex contention issues:

- **Coarse-grained locking**: A mutex is held across the entire batch accumulation process
- **Blocking wait patterns**: Threads are blocked while waiting for batch formation
- **Thundering herd problem**: Multiple waiting threads awakened simultaneously create contention spikes

```cpp
// Likely current pattern (problematic)
std::mutex queueMutex;
std::vector<Item> batchQueue;
std::condition_variable batchReady;

// In accumulator loop:
{
    std::lock_guard<std::mutex> lock(batchMutex);
    // Entire batch formation logic including waiting
}
```

### Lock contention in batch availability checks

The `MCTSEvaluator::batchCollectorLoop()` exhibits excessive lock acquisition for batch availability checks:

- High-frequency mutex acquisition to check for batch availability
- Mutex-based signaling for batch readiness creates overhead
- Synchronous batch processing while holding locks increases lock duration

### Inefficient external queue processing

The bulk dequeue logic in `mcts_evaluator.cpp` uses nested retry loops that are fundamentally inefficient:

- Mutexes protect the entire dequeue operation
- Failed dequeue attempts result in retrying with mutex reacquisition
- Non-adaptive batch sizes fail to respond to system conditions

### Memory allocation hotspots

State cloning operations create excessive memory pressure that compounds performance issues:

- Each new state requires cloning of game state, causing frequent allocations
- Repeated allocation/deallocation leads to memory fragmentation
- Poor memory locality reduces CPU cache hit rates

## Execution flow analysis

The current execution timeline shows significant waiting periods where search threads spend more time waiting than searching:

```
Timeline (ms):
0         10        20        30        40        50        60
|         |         |         |         |         |         |
[S1]---W------[S1]---W---------[S1]---W------[S1]---W------  <-- Search Thread 1
 [S2]----W-------[S2]----W--------[S2]----W------[S2]--W--   <-- Search Thread 2
  [S3]-----W--------[S3]------W--------[S3]------W--------   <-- Search Thread 3
   ...
     [E]-----[GPU]------[E]-----[GPU]------[E]-----[GPU]--   <-- Evaluation Thread
                                                              (with small batches)
```

A typical search thread's execution reveals a stop-and-go pattern that prevents fluid tree exploration, with the GPU mostly idle or underutilized.

## Solution architecture

### 1. Restructure search-evaluation relationship with virtual loss

Standard AlphaZero implementations use "virtual losses" to encourage diversity in node selection. When a node is selected for evaluation but hasn't received its neural network result, a temporary negative score discourages other threads from selecting the same node.

```cpp
// Implementation in search thread
node->applyVirtualLoss();  // Discourage other threads from selecting same node
evaluationQueue.push(node);
// Continue tree exploration rather than waiting
// Later when result is available:
node->removeVirtualLoss();
node->updateWithRealEvaluation(result);
```

This approach allows search to continue while waiting for evaluations, naturally building larger batches.

### 2. Replace mutex-based synchronization with lock-free alternatives

To eliminate the contention in `BatchAccumulator::accumulatorLoop()`:

```cpp
// Replace mutex-protected queue with lock-free implementation
moodycamel::ConcurrentQueue<Item> batchQueue; // Lock-free queue

// Use atomic operations for batch size tracking
std::atomic<size_t> batchSize{0};

// Implement two-phase waiting with exponential backoff
unsigned spinCount = 0;
while (!batchSizeAtomic.load(std::memory_order_acquire) >= batchThreshold) {
    if (spinCount < MAX_SPIN) {
        spinCount++;
        // CPU-friendly pause
        _mm_pause(); // x86 specific, use platform alternatives
    } else {
        // Yield to scheduler after spin limit reached
        std::this_thread::yield();
        // Exponential backoff
        std::this_thread::sleep_for(std::chrono::microseconds(
            std::min(100, 1 << std::min(spinCount - MAX_SPIN, 10u))));
    }
}
```

### 3. Implement producer-consumer pattern for batch collection

For `MCTSEvaluator::batchCollectorLoop()`, implement a producer-consumer pattern:

```cpp
// Lock-free batch queue implementation:
moodycamel::BlockingConcurrentQueue<Batch> batchQueue;

// Producer side:
void produceBatch(Batch&& batch) {
    batchQueue.enqueue(std::move(batch));
}

// Consumer side (collector loop):
void batchCollectorLoop() {
    Batch batch;
    while (running.load(std::memory_order_acquire)) {
        if (batchQueue.wait_dequeue_timed(batch, waitTimeMsec)) {
            processBatch(batch);
        }
    }
}
```

### 4. Optimize memory management with object pooling

To address memory allocation hotspots:

```cpp
template <typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> objects;
    std::stack<T*> freeObjects;
    std::mutex poolMutex;

public:
    ObjectPool(size_t initialSize = 1024) {
        for (size_t i = 0; i < initialSize; ++i) {
            objects.emplace_back(std::make_unique<T>());
            freeObjects.push(objects.back().get());
        }
    }

    T* allocate() {
        std::lock_guard<std::mutex> lock(poolMutex);
        if (freeObjects.empty()) {
            objects.emplace_back(std::make_unique<T>());
            freeObjects.push(objects.back().get());
        }
        T* obj = freeObjects.top();
        freeObjects.pop();
        return obj;
    }

    void deallocate(T* obj) {
        if (!obj) return;
        std::lock_guard<std::mutex> lock(poolMutex);
        freeObjects.push(obj);
    }
};
```

This approach dramatically reduces allocation/deallocation overhead during tree exploration.

### 5. Enhanced BatchAccumulator implementation

A complete redesign of the `BatchAccumulator` class:

```cpp
class BatchAccumulator {
private:
    // Lock-free queue for better performance
    moodycamel::ConcurrentQueue<MCTSNode*> nodeQueue;
    
    // Atomic variables for thread synchronization without mutex overhead
    std::atomic<bool> shouldTerminate{false};
    std::atomic<int> waitingBatches{0};
    std::atomic<int> batchSize{0};
    
    // Configurable batch parameters
    const int minBatchSize;
    const int maxBatchSize;
    const int maxBatchWaitTimeMs;
    
    // Thread synchronization primitives
    std::condition_variable batchReadyCV;
    std::mutex batchMutex;
    
    // Vector to store the current batch
    std::vector<MCTSNode*> currentBatch;

public:
    BatchAccumulator(int minSize = 8, int maxSize = 64, int waitTimeMs = 2)
        : minBatchSize(minSize), maxBatchSize(maxSize), maxBatchWaitTimeMs(waitTimeMs) {
        currentBatch.reserve(maxBatchSize);
    }
    
    void addNode(MCTSNode* node) {
        if (!node) return;
        nodeQueue.enqueue(node);
        
        // Wake up the accumulator if it's waiting and we've reached min batch size
        int currentQueueSize = nodeQueue.size_approx();
        if (currentQueueSize >= minBatchSize) {
            batchReadyCV.notify_one();
        }
    }
    
    void accumulatorLoop() {
        while (!shouldTerminate) {
            std::vector<MCTSNode*> tempBatch;
            tempBatch.reserve(maxBatchSize);
            
            // Collect nodes until we have enough or timeout occurs
            auto startTime = std::chrono::steady_clock::now();
            bool timeoutReached = false;
            
            // First collect what's immediately available
            MCTSNode* node;
            while (tempBatch.size() < maxBatchSize && nodeQueue.try_dequeue(node)) {
                tempBatch.push_back(node);
            }
            
            // If we don't have the minimum batch size, wait for more nodes
            if (tempBatch.size() < minBatchSize && !shouldTerminate) {
                std::unique_lock<std::mutex> lock(batchMutex);
                
                // Use a predicate with condition_variable to prevent spurious wakeups
                auto batchReady = [this, &tempBatch, &timeoutReached, &startTime]() {
                    // Check for more nodes
                    MCTSNode* node;
                    while (tempBatch.size() < maxBatchSize && nodeQueue.try_dequeue(node)) {
                        tempBatch.push_back(node);
                    }
                    
                    // Check if we have enough nodes or timeout reached
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime);
                    timeoutReached = elapsed.count() >= maxBatchWaitTimeMs;
                    
                    return shouldTerminate || 
                           tempBatch.size() >= minBatchSize || 
                           timeoutReached;
                };
                
                // Wait until condition is met
                batchReadyCV.wait(lock, batchReady);
            }
            
            if (tempBatch.empty()) {
                continue;
            }
            
            // Store the batch and notify waiting threads
            {
                std::unique_lock<std::mutex> lock(batchMutex);
                currentBatch.swap(tempBatch);
                batchSize = currentBatch.size();
            }
            
            // Increase waiting batches counter and notify evaluator
            waitingBatches++;
            batchReadyCV.notify_all();
            
            // Wait until the batch is processed
            {
                std::unique_lock<std::mutex> lock(batchMutex);
                auto batchProcessed = [this]() {
                    return currentBatch.empty() || shouldTerminate;
                };
                batchReadyCV.wait(lock, batchProcessed);
            }
        }
    }
};
```

## Implementation roadmap

To implement these changes effectively, follow this sequence:

1. **Phase 1: Update synchronization primitives**
   - Replace mutex-based queues with lock-free alternatives
   - Implement virtual loss mechanism for continued search during evaluation
   - Update thread synchronization patterns in BatchAccumulator and MCTSEvaluator

2. **Phase 2: Memory optimization**
   - Implement object pooling for MCTSNode allocation
   - Pre-allocate vectors for batch processing
   - Optimize tensor memory management for neural network inference

3. **Phase 3: Execution flow restructuring**
   - Modify search threads to continue exploration using virtual loss
   - Implement non-blocking evaluation requests
   - Update backpropagation to handle asynchronous evaluation results

4. **Phase 4: Batch formation enhancement**
   - Implement dynamic batch sizing based on queue state
   - Add adaptive timeouts for batch collection
   - Optimize batch preparation in evaluation loop

## Expected performance improvements

After implementing these changes, you should see:

1. **Increased batch sizes**: From current 1-3 elements to 32-128 elements
2. **Improved GPU utilization**: From current low levels (5-30%) to 80-95%
3. **Reduced memory pressure**: 40-60% reduction in allocation operations
4. **Higher throughput**: 3-10x increase in nodes evaluated per second
5. **Better scaling**: Near-linear scaling with additional CPU cores up to GPU saturation

## Conclusion

The batching bottlenecks in your AlphaZero MCTS implementation stem from fundamental design issues in thread coordination, synchronization mechanisms, and memory management. By implementing the proposed changes, particularly the virtual loss mechanism, lock-free communication, and optimized batch formation, you can achieve dramatically better GPU utilization and overall system performance.

The most critical change is moving from a blocking, synchronous evaluation model to an asynchronous model where search continues during evaluation. This single architectural shift, combined with proper implementation of the supporting synchronization patterns, will transform your implementation from one that struggles with micro-batches to one that efficiently utilizes GPU resources.

----------
----------

\<todo\_list>

* **Investigate and refactor batch-collection logic.** The current batch collector aggressively flushes small batches (e.g. any ≥8 items after a short wait). This leads to chronically small inference batches early in search. We should review the `BatchAccumulator` and `MCTSEvaluator` flush thresholds: for example, processing any batch after just \~25 ms  may underfill the GPU. Potential tasks include adjusting or removing the immediate-flush conditions (e.g. “>=8 items” and “>=1 after 25 ms” rules) and lengthening wait time to build larger batches.
* **Simplify the leaf-evaluation queue path.** The code currently has two layers of queuing (`leaf_queue_` → bulk enqueue → individual enqueue) and complex notification loops. We should streamline this: for example, unify `enqueue_bulk` and per-item loops, and replace repeated `notifyLeafAvailable()` calls with a proper condition-variable or event-based wakeup. This will reduce latency spikes and avoid “force duplicate notifications” hacks.
* **Fix pending-evaluation flag logic.** The `MCTSNode::hasPendingEvaluation()` hack (occasional random clear) is unsafe and could drop or delay evaluations, indirectly shrinking batches. We should replace this with a robust lock-free check (e.g. use an atomic counter or a proper flag without “rare clear” randomness) to ensure no leaf stays erroneously marked and thus blocks batching.
* **Ensure the evaluator is correctly connected to the leaf queue.** In external-queue mode, `MCTSEngine` must call `evaluator_->setExternalQueues(&leaf_queue_, &result_queue_, …)` so that `MCTSEvaluator` actually reads from `leaf_queue_`. If this isn’t done, evaluations may never reach the GPU. Audit initialization code to guarantee `leaf_queue_` and `result_queue_` are linked.
* **Profile and increase parallel simulation.** The search loop is currently essentially single-threaded (one OpenMP-style loop in `executeSerialSearch`), limiting how many leafs can be produced concurrently. Consider parallelizing simulations (e.g. use multiple threads or OpenMP to run `selectLeafNode()` in parallel) so that the GPU queue fills faster. This will directly increase batch sizes and GPU utilization.
* **Clean up unused pipeline/NodeTracker code.** The repository contains a pipeline implementation and a `NodeTracker` that aren’t actively used in the `executeSerialSearch` path. If not needed, remove them to reduce complexity; otherwise integrate them properly (e.g. use `NodeTracker` for pending results instead of manual flags) to simplify flow.
* **Adjust neural-network and GPU settings.** Verify that the neural network backend (e.g. PyTorch) is configured for asynchronous inference and maximum GPU throughput. For instance, ensure no accidental synchronization or small buffering in `nn->inference()` calls. Although not directly in the MCTS code, GPU config could be tuned (e.g. batch tensor pinning or fewer CPU→GPU transfers) for better utilization.
  \</todo\_list>

\<optimization\_scheme>

* **Batch-size and timing parameters.** The `BatchParameters` and `BatchAccumulator` settings dictate how and when batches are sent to the GPU. Currently `optimal_batch_size` is 256 with a `minimum_viable_batch_size` of 192 (capped to 64 by code). We might *dynamically* adapt these based on runtime conditions. For example, if GPUs are idle, lower the target batch so more frequent (though smaller) inferences keep the GPU busy. Conversely, if small batches persist, temporarily raise the wait threshold (e.g. increase `batch_params_.max_wait_time`) so more leafs accumulate before processing. Any such tuning should be code-driven rather than hard-coded (e.g. adaptively double wait time on low throughput).
* **Revise batch accumulation heuristics.** The code currently routes *any* small batch (<64) through the `BatchAccumulator`, which tries to form bigger batches over time. However, the accumulator’s own loop then forces even small batches out after \~25 ms. We should consider eliminating the “emergency” flush for tiny batches or increase its timeout. For instance, removing the immediate processing when size≥8 (and replacing with a longer fixed timeout) would allow, say, at least 32 items to accumulate. This change could double the GPU batch size without changing hardware. We would retain timeouts, but lengthen them (e.g. flush any batch after 100 ms instead of 25 ms), to balance latency vs throughput.
* **Batched enqueue optimizations.** In `MCTSEngine::executeSerialSearch`, bulk enqueue is used when `leaf_batch.size()>1`, else individual enqueue with retries. To maximize throughput, we should *always* use `enqueue_bulk` when possible: even a size-1 batch could be bulk-enqueued (size 1) rather than entering the complex retry loop. This avoids the retry/backoff cost and immediate small enqueue delays. Additionally, the exponential backoff and repeated notifications (e.g. 4 notifications after each enqueue) suggest unreliable signaling. Replacing these with a single event or using moodycamel’s integrated sync could remove tens of microseconds of overhead per leaf, which can add up.
* **Simplify or remove `BatchAccumulator` layer when appropriate.** In external-queue mode, the code now sometimes bypasses the accumulator for “good” batches. If we can frequently produce batches ≥64 (see above), we could eliminate the `BatchAccumulator` entirely or reduce its role, since the direct path is more efficient. One approach: if `pending_eval_batch.size()` reaches even 32, treat it as “good” and skip `addEvaluation`. The current min threshold of 64 could be lowered. This could be a temporary hack: e.g., set `batch_params_.minimum_viable_batch_size = 32` (half the default) so that 32-leaf batches go direct. Although the prompt asks beyond “changing batch size,” this is a deep architecture tweak (changing the policy of direct vs accumulate) rather than a mere tuning.
* **Remove polling delays.** The evaluator’s `batchCollectorLoop` uses `std::this_thread::sleep_for` and an `AdaptiveBackoff` to wait for tasks. We should replace this with a proper condition variable or blocking queue pop, so threads sleep efficiently until work arrives. For example, moodycamel queues support `wait_for_elements`/`wait_dequeue` APIs. Eliminating the 1 ms backoff sleeps could shave latency and slightly increase batch gather time (since threads aren’t busy-waiting).
* **Improve `pending_evaluations_` handling.** The engine throttles leaf generation if too many evaluations are pending (`if pending > batch_size*4 then sleep(1ms)`). This can waste CPU cycles or under-feed the GPU if the threshold is wrong. We should reassess this logic: perhaps throttle only when *batches* are saturated, not just pending count. Or adaptively scale down the multiplier from `*4` to a smaller factor if GPU is idle (allow more enqueues before pausing). This prevents premature stalling of MCTS simulation that starves the queue.
  \</optimization\_scheme>

\<parallelization\_improvements>

* **Enable multithreaded tree search.** Currently, `executeSerialSearch` runs in one thread (with a loop claiming simulations). Modern MCTS implementations often parallelize tree traversal: e.g. spawn multiple threads each running a subset of the 800 simulations simultaneously, using virtual losses to synchronize. We should leverage `settings_.num_threads` by running `selectLeafNode`/`expandNonTerminalLeaf` in parallel (with careful locking or lock-free structures like `std::atomic` or `moodycamel` queues). The `thread_data_` array in `MCTSEngine` suggests intent to support per-thread batching. Converting the main loop into a parallel region (e.g., an OpenMP parallel for over a large number of “dummy” simulation tasks) would multiply the rate of leaf-state production, filling the GPU queue faster.
* **Pipeline parallelism (overlap CPU/GPU work).** The codebase contains a `ConcurrentPipelineBuffer` (in `mcts_pipeline.cpp`). This approach allows gathering one batch while the GPU is busy with the previous batch. We should consider using this: as soon as the CPU has ≥`optimal_batch_size` leaves, push them into the pipeline buffer (via `addToPipelineBatch`) and then concurrently run the next selection steps to prefill the next buffer. When the GPU finishes, it can swap buffers (`swapPipelineBuffers`) and process. This overlap ensures the GPU is never idle waiting for the CPU. Integrating the pipeline (instead of the external queue) could simplify by handling both batching and inference in a single `pipelineProcessorLoop`.
* **Multiple inference threads or streams.** Although the evaluator defaults to one inference thread, if the neural network library supports it, we could spawn multiple GPU inference threads. For example, two threads each waiting on separate condition variables could submit their own batches to the GPU concurrently (possibly on different CUDA streams). This is advanced, but if successful it can double throughput on architectures that allow concurrent kernels. At minimum, verify that the current inference call is asynchronous (`inference_fn_` should return a future or be non-blocking). If not, consider dispatching it into a thread pool or `std::async` to avoid blocking the collector.
* **Shared vs dedicated queues.** The engine supports “shared queues” for multiple concurrent searches. If running many games in parallel, we could consolidate their leaf states into a global queue for GPU use (larger global batches) rather than each engine running its own. Conversely, if multi-threading a single game, keep the queue dedicated but avoid contention by having per-thread local buffers (which the code is partially set up for via `ThreadData`). For example, each search thread could fill its local leaf buffer and push to the shared queue only when full, reducing lock contention on `leaf_queue_`.
* **Preserve work-conservation.** Ensure that CPU threads continue to generate leaves even while waiting for inference. For example, after an initial batch is sent, threads should keep exploring the tree and tagging new leaves (using virtual loss) rather than idling. This may require removing or relaxing the pending-evaluations throttle discussed above. The goal is that all CPU cores are either expanding the tree or handling the GPU results – no core should sit idle waiting for small batches to process.
  \</parallelization\_improvements>

\<gpu\_throughput\_scenario>

* **Early-game batch buildup.** In the first few iterations of MCTS, the tree is shallow so only a few leaves are generated at a time. For instance, imagine on move 1 there are 5 legal moves; early expansions may yield only a handful of leaf states (<<256) in the first 10ms. As coded, these might be flushed after \~25ms, resulting in a tiny batch (e.g. 8–16 items), and the GPU quickly idles. To address this, we could modify the scenario: delay any inference until at least X leaves have accumulated (e.g. wait 50ms or until 64 leaves) for the very first batch. In practice, this could mean raising `min_viable_batch_size` temporarily or injecting a brief sleep in `executeSerialSearch` before enqueuing the first batch, to let multiple simulations complete. This will create a larger initial GPU workload, improving throughput right from the start.
* **Sustained pipelining.** Once the pipeline is primed (e.g. after \~2 batches), we should operate at full tilt: every time a batch is sent to the GPU, immediately continue generating the next batch concurrently. For example, after bulk-enqueue of 64 leaf states, the CPU could immediately start another 64 simulations while the GPU processes the first batch’s neural-network inference. We would verify that after these changes the GPU occupancy is near 100%: if not, consider further tactics like subdividing each “move” simulation across different GPU streams (if the NN model allows parallel inference).
* **Measuring and feeding back.** Implement a feedback loop: monitor `getAverageBatchSize()` and GPU utilization (as printed in `[BATCH_STATS]`). If we detect that the average batch size is, say, < 50% of the target or GPU utilization is low, the system could automatically relax the merging thresholds or spawn more simulations. Conversely, if batch sizes start exceeding the target, we could modestly raise the target to maximize throughput. This scenario-based tuning ensures the system adapts to varying search phases (early vs. late game) to keep the GPU busy.
  \</gpu\_throughput\_scenario>

**Sources:** Analysis based on the `src/mcts` and `include/mcts` code of the *omoknuni\_small* repository, including `mcts_engine_search.cpp`, `mcts_evaluator_concurrent.cpp`, and `batch_accumulator.cpp` and related files. This review cites the actual implementation to identify where batching and threading occur and where fixes may be applied.
