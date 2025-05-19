\<todo\_list\>
Here is a prioritized list of identified issues and tasks for improvement:

1.  **Issue: Suboptimal Batch Assembly and GPU Starvation (Priority: 10/10)**

      * **Description**: The primary cause of low GPU usage and throughput is the formation of small batches. This stems from:
          * `MCTSEngine`: Leaf collection loops in `runSearch` might submit batches too readily (e.g., `MIN_BATCH_SIZE` is 1 for the serial collection path, `MAX_EMPTY_TRIES` is 3, `MAX_BATCH_COLLECTION_TIME` is 5ms).
          * `MCTSEvaluator`: The `processBatch` logic (for external queues) and `collectBatch` (for internal queues) can also process very small batches due to low `MIN_BATCH` thresholds (can be 1) and short `max_wait` (can be 2ms) or `timeout_` periods.
      * **Task**: Overhaul batch accumulation logic in both `MCTSEngine` and `MCTSEvaluator`.
          * Increase minimum batch sizes for submission.
          * Implement more patient batching: allow significantly longer configurable timeouts for batch accumulation, potentially adaptive to the NN inference speed and leaf arrival rate.
          * Consider a strategy where `MCTSEngine` aims to keep a certain number of evaluations "in-flight" to the evaluator.

2.  **Issue: Inefficient `active_simulations_` Management Leading to Low CPU Usage (Priority: 9/10)**

      * **Description**: `active_simulations_` in `MCTSEngine` is decremented per tree traversal attempt. If many traversals hit nodes already being evaluated or take a long time, `active_simulations_` can deplete before enough leaves are generated, leading to CPU threads becoming idle and subsequently starving the GPU.
      * **Task**: Modify `active_simulations_` to represent the number of evaluations to *complete* or *successfully queue*, rather than traversal attempts. Only decrement it when a leaf is actually queued for evaluation or a terminal node's result is fully backpropagated.

3.  **Issue: Potential Bottlenecks in Tree Traversal (Selection/Expansion) (Priority: 8/10)**

      * **Description**: Slowdowns in `MCTSEngine::selectLeafNode` or `MCTSNode::expand` can limit the rate of leaf generation, contributing to low CPU usage and small batches. This includes game state cloning, UCB calculations, and TT interactions.
      * **Task**:
          * Profile `selectLeafNode` (including `MCTSNode::selectChild`) and `MCTSNode::expand` under realistic load.
          * Optimize game-specific `IGameState::clone()`, `getLegalMoves()`, etc.
          * Ensure the OpenMP parallelization in `MCTSNode::selectChild` (threshold `>32` children) is optimal and not causing undue overhead for common case scenarios in Gomoku/Chess.

4.  **Issue: Inconsistent Virtual Loss Application (Priority: 7/10)**

      * **Description**: `MCTSEngine::selectLeafNode` uses `MCTSNode::addVirtualLoss()` which increments by a fixed `+1`. However, `MCTSEngine::expandAndEvaluate` (when `num_threads > 0`) calls `leaf->applyVirtualLoss(settings_.virtual_loss)`, which uses a configurable amount. This inconsistency can affect exploration behavior.
      * **Task**: Standardize virtual loss application. It's generally recommended to use the configurable `settings_.virtual_loss` amount consistently during selection and before queuing for evaluation.

5.  **Issue: Transposition Table Eviction Policy (Priority: 6/10)**

      * **Description**: `TranspositionTable::enforceCapacityLimit` samples a portion of the table, sorts candidates, and then evicts. This happens probabilistically (1% of stores). If the table is large, this sampling and sorting could introduce latency or contention on `clear_mutex_`.
      * **Task**: Evaluate the performance impact of the current eviction policy. Consider simpler/faster policies if it's a bottleneck (e.g., random replacement, or always replace based on depth/visits without extensive sampling/sorting).

6.  **Issue: `MCTSEvaluator` Internal vs. External Queue Logic Complexity (Priority: 5/10)**

      * **Description**: `MCTSEvaluator` has distinct and complex batch collection paths: `evaluationLoop` -\> `processBatch` for external queues, and `processBatches` -\> `collectBatch` -\> `processBatch(std::vector<EvaluationRequest>& batch)` for internal queues. This increases maintenance overhead and potential for inconsistencies.
      * **Task**: Refactor `MCTSEvaluator` to share more common logic for batch accumulation and processing, regardless of whether queues are internal or external.

7.  **Issue: GameState Pool Initialization (Priority: 4/10)**

      * **Description**: `GameStatePoolManager` might be initialized on-demand within `MCTSEngine::cloneGameState` if `game_state_pool_enabled_` is true. This could add slight overhead to the first few clones in a search.
      * **Task**: Ensure `GameStatePoolManager` is explicitly initialized once at the start of `MCTSEngine::runSearch` (or engine construction) if pooling is enabled.

8.  **Issue: Shutdown and Resource Cleanup Complexity (Priority: 4/10)**

      * **Description**: The destructor `MCTSEngine::~MCTSEngine()` has a multi-phase shutdown sequence. Ensuring this is robust and free of deadlocks or race conditions with concurrent operations is critical. For instance, clearing queues while workers might still be trying to access them or nodes needs careful handling.
      * **Task**: Thoroughly review the shutdown logic for all components (`MCTSEngine`, `MCTSEvaluator`), especially focusing on the order of operations, flag signalization, queue draining, and thread joining to prevent hangs or crashes.
        \</todo\_list\>

\<optimization\_scheme\>
Here's a suggested optimization scheme, breaking down the strategy into incremental steps:

**Phase 1: Stabilize Batching and Leaf Generation**

1.  **Modify `active_simulations_` Logic (MCTSEngine)**:

      * Change `active_simulations_` to count *pending evaluations* or *completed simulations* rather than attempts.
      * Decrement `active_simulations_` only when a `PendingEvaluation` is successfully enqueued to `leaf_queue_` or a terminal node is fully processed and backpropagated.
      * **Rationale**: This ensures the search runs for the intended number of useful work units, improving CPU utilization for leaf generation.
      * **Pilot Code (Conceptual for `runSearch` loop):**
        ```cpp
        // Inside the leaf generation loop in MCTSEngine::runSearch
        // Instead of: if (active_simulations_.compare_exchange_weak(... simulations_remaining - to_claim ...))
        // Change to a model where simulations are "claimed" or "dispatched"
        // and only "completed" when a leaf is queued or terminal node processed.

        // At the start of runSearch:
        // std::atomic<int> evaluations_to_initiate = settings_.num_simulations;

        // Inside the parallel leaf generation loop:
        // while (evaluations_to_initiate.load(std::memory_order_acquire) > 0) {
        //    if (/* leaf successfully selected and about to be queued */) {
        //        if (evaluations_to_initiate.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
        //            evaluations_to_initiate.fetch_add(1, std::memory_order_relaxed); // Revert if over-decremented
        //            break; // No more evaluations to initiate
        //        }
        //        // ... enqueue the leaf ...
        //        pending_evaluations_.fetch_add(1, ...);
        //    } else if (/* terminal node processed */) {
        //        if (evaluations_to_initiate.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
        //             evaluations_to_initiate.fetch_add(1, std::memory_order_relaxed);
        //             break;
        //        }
        //    }
        // }
        // The main loop would then wait for pending_evaluations_ to go to zero.
        ```

2.  **Increase Minimum Batch Sizes and Timeouts (MCTSEngine & MCTSEvaluator)**:

      * **MCTSEngine (`runSearch` leaf submission logic):**
          * Increase `MIN_BATCH_SIZE` (currently 1 in the serial path) to a meaningful fraction of `settings_.batch_size` (e.g., `settings_.batch_size / 4` or at least 8-16).
          * Increase `MAX_BATCH_COLLECTION_TIME` (currently 5ms) to something like 10-20ms.
          * Increase `MAX_EMPTY_TRIES` or make its effect less immediate for batch submission.
      * **MCTSEvaluator (`processBatch` for external queues):**
          * Increase `MIN_BATCH` (can be 1) and `max_wait` (can be 2ms). `MIN_BATCH` should be at least `settings_.batch_size / 8` or a fixed moderate number like 16. `max_wait` should be increased to 5-15ms.
      * **MCTSEvaluator (`collectBatch` for internal queues):**
          * `ABSOLUTE_MIN_BATCH` (32) is reasonable. `timeout_` (adaptive) should have a higher floor if it's dropping too low.
      * **Rationale**: Forces larger batches, giving the GPU more work per inference call.

3.  **Implement Adaptive Batch Timeout in `MCTSEvaluator`**:

      * Measure average NN inference time (`avg_inference_time_ms`).
      * Measure average leaf arrival rate (`leaves_per_ms`).
      * Dynamically adjust batch collection timeouts in `MCTSEvaluator` to aim for `settings_.batch_size` leaves, but not significantly exceeding `avg_inference_time_ms * K` (where K is a factor like 1.5-2.0 to allow accumulation).
      * **Rationale**: Tailors batching to the actual performance characteristics of the NN and leaf generation.
      * **Pseudocode (Conceptual for `MCTSEvaluator::collectBatch` or `processBatch`):**
        ```cpp
        // Maintain running averages:
        // float avg_nn_latency_ms = ...;
        // float avg_leaf_arrival_interval_ms = ...;

        // Target time to accumulate a full batch:
        // float target_accumulation_time_ms = settings_.batch_size * avg_leaf_arrival_interval_ms;
        // Max wait:
        // std::chrono::milliseconds dynamic_timeout = std::chrono::milliseconds(
        //    static_cast<long>(std::min(target_accumulation_time_ms, avg_nn_latency_ms * 1.5))
        // );
        // dynamic_timeout = std::max(dynamic_timeout, min_allowable_timeout);
        // dynamic_timeout = std::min(dynamic_timeout, max_allowable_timeout);

        // Use dynamic_timeout in batch_ready_cv_.wait_for or similar wait logic.
        ```

**Phase 2: Optimize Core MCTS Operations**

4.  **Standardize Virtual Loss (MCTSEngine, MCTSNode)**:

      * Modify `MCTSNode::addVirtualLoss()` to take an `int amount` argument.
      * In `MCTSEngine::selectLeafNode`, call `parent_for_selection->addVirtualLoss(settings_.virtual_loss);` and `selected_child->addVirtualLoss(settings_.virtual_loss);`.
      * Ensure `MCTSNode::removeVirtualLoss()` correctly subtracts the same amount or is paired with `applyVirtualLoss(-settings_.virtual_loss)`. The current `removeVirtualLoss` decrements by 1. This also needs to be consistent. It might be better to have `applyVirtualLoss(amount)` and `applyVirtualLoss(-amount)`.
      * **Rationale**: Consistent exploration pressure.
      * **Pilot Code (`MCTSNode.cpp`):**
        ```cpp
        // In MCTSNode.h
        // void addVirtualLoss(int amount); // New or modify existing
        // void removeVirtualLoss(int amount); // New or modify existing

        // In MCTSNode.cpp
        void MCTSNode::addVirtualLoss(int amount) { // Or rename applyVirtualLoss to this
            int current = virtual_loss_count_.load(std::memory_order_relaxed);
            int new_value = std::min(current + amount, 1000); // Max cap
            virtual_loss_count_.store(new_value, std::memory_order_release);
        }

        void MCTSNode::removeVirtualLoss(int amount) {
            int current = virtual_loss_count_.load(std::memory_order_relaxed);
            int new_value = std::max(current - amount, 0); // Min cap
            virtual_loss_count_.store(new_value, std::memory_order_release);
        }
        ```
        Then, in `selectLeafNode`:
        `parent_for_selection->addVirtualLoss(settings_.virtual_loss);`
        And later, when backpropagating or evaluation is done for a node, its virtual loss is removed:
        `node->removeVirtualLoss(settings_.virtual_loss);` (ensure this is done correctly for all paths). Currently, `backPropagate` calls `node->removeVirtualLoss()` which removes 1. This needs to be `node->removeVirtualLoss(settings_.virtual_loss)`.

5.  **Profile and Optimize Hotspots (Selection/Expansion)**:

      * Use a profiler (e.g., VTune, gprof, or custom timers) to identify exact bottlenecks in `selectLeafNode`, `MCTSNode::selectChild`, `MCTSNode::expand`, and game state operations.
      * Optimize based on findings. For example, if `IGameState::clone()` is slow, focus on optimizing its game-specific implementation or the `GameStatePoolManager`.
      * **Rationale**: Reduces CPU time per traversal, increasing leaf generation rate.

**Phase 3: Refinements and Robustness**

6.  **Review Transposition Table Eviction**:

      * If `enforceCapacityLimit` is found to be a bottleneck (due to `clear_mutex_` contention or slow sampling/sorting), consider replacing it with a simpler policy (e.g., random replacement of an entry within a randomly selected shard/bucket) or a policy that doesn't require iterating/sorting large parts of the table.
      * **Rationale**: Ensures TT doesn't become a performance drag.

7.  **Refactor `MCTSEvaluator` Batching**:

      * Aim to unify the batch collection and processing logic for internal and external queues to reduce redundancy and improve maintainability. The core mechanism of collecting up to `batch_size_` items with a timeout can be shared.
      * **Rationale**: Simpler code, easier to debug and tune.

8.  **Explicit GameState Pool Initialization**:

      * In `MCTSEngine::runSearch`, before the main simulation loop, if `game_state_pool_enabled_` is true, explicitly initialize the pool for the current game type if not already done.
      * **Pilot Code (`MCTSEngine::runSearch`):**
        ```cpp
        // At the beginning of runSearch, after root node creation
        if (game_state_pool_enabled_ && !utils::GameStatePoolManager::getInstance().hasPool(state.getGameType())) {
            try {
                size_t pool_size = settings_.num_simulations * 2; // Or some other heuristic
                utils::GameStatePoolManager::getInstance().initializePool(state.getGameType(), pool_size);
            } catch (const std::exception& e) {
                MCTS_LOG_ERROR("Failed to initialize GameState pool: " << e.what());
                // Optionally disable pooling for this search if init fails
            }
        }
        ```

9.  **Thorough Shutdown Review**:

      * Carefully trace the shutdown sequence in `MCTSEngine::~MCTSEngine()` and `MCTSEvaluator::stop()`.
      * Ensure that `shutdown_` flags are checked appropriately in all loops.
      * Verify that queues are drained *after* worker threads are signaled to stop and *before* threads are joined, or that workers handle queue operations gracefully during shutdown.
      * Consider using condition variables more extensively to signal workers to exit cleanly rather than just relying on flags and timeouts for joining.
      * **Rationale**: Prevents deadlocks, crashes, or resource leaks on exit.
        \</optimization\_scheme\>

\<parallelization\_improvements\>
The current implementation uses OpenMP for parallel tree traversals in `MCTSEngine` and `std::thread` for the `MCTSEvaluator` and `resultDistributorWorker`. Leaf parallelization is the core strategy. Here are improvements:

  * **Synchronization**:

      * **Current**: Primarily uses `std::atomic` for node statistics (`visit_count_`, `value_sum_`, etc.) and flags (`is_expanded_`, `evaluation_in_progress_`). `moodycamel::ConcurrentQueue` handles lock-free queueing. `phmap::parallel_flat_hash_map` for TT. Mutexes are present (`evaluator_mutex_`, `clear_mutex_` in TT, `expansion_mutex_` in node but its usage seems limited to TaskFlow variant).
      * **Recommendations**:
        1.  **Virtual Loss Consistency**: As mentioned in `<optimization_scheme>`, ensure `settings_.virtual_loss` is applied and removed consistently. The current mix of `+1` and `settings_.virtual_loss` can lead to uneven exploration.
        2.  **`active_simulations_` Management**: The current approach of decrementing per traversal attempt might cause premature search termination or uneven load. Shifting to decrementing per useful work unit (leaf queued/terminal processed) will better synchronize the simulation count with actual progress.
        3.  **Minimize Critical Sections**: The use of `MCTSEngine::evaluator_mutex_` for `ensureEvaluatorStarted` is acceptable as it's mostly during initialization. `TranspositionTable::clear_mutex_` for `enforceCapacityLimit` could be a point of contention if called very frequently or if the operation is slow; probabilistic calling reduces this risk.

  * **Deadlocks**:

      * **Current**: No obvious deadlocks observed in the primary `MCTSEngine` loop, but complex interactions exist:
          * `MCTSEngine` produces leaves for `MCTSEvaluator`.
          * `MCTSEvaluator` processes batches and sends results to `result_queue_`.
          * `resultDistributorWorker` processes `result_queue_` and updates nodes.
      * **Recommendations**:
        1.  **Shutdown Order**: This is the most critical area for deadlocks. Ensure a clean shutdown:
              * Signal all threads to stop (e.g., set `shutdown_ = true`).
              * Notify all condition variables to wake up waiting threads so they can observe the shutdown flag.
              * Stop the `MCTSEvaluator` first (so it stops accepting new work). This involves draining its internal request queue if necessary.
              * Then, ensure `MCTSEngine`'s leaf generation loops terminate.
              * Drain the `leaf_queue_` and `result_queue_`. It's important that producers stop before consumers are indefinitely waiting or queues are cleared prematurely.
              * Join threads (evaluator, result distributor, etc.).
        2.  **Bounded Queues (Implicit)**: `moodycamel::ConcurrentQueue` is unbounded by default. If memory is a concern and queues grow excessively due to a slow consumer (e.g., slow NN), this could lead to OOM, which isn't a deadlock but a resource exhaustion issue. Consider if capacity limits on queues are needed, though this adds complexity. The current design seems to rely on the system balancing out.

  * **Lock Contention**:

      * **Current**: Minimized by atomics and concurrent data structures.
      * **Recommendations**:
        1.  **TT `enforceCapacityLimit`**: If this becomes an issue (profile\!), the `clear_mutex_` could be a bottleneck. Consider sharding this operation or using a lock-free eviction strategy if feasible (though complex for TTs). The current probabilistic call helps.
        2.  **`MCTSEngine::evaluator_mutex_`**: Used for starting the evaluator. This is likely fine as it's not in the hot path of simulations.
        3.  **Node Expansion**: `MCTSNode::expand` uses `is_expanded_.compare_exchange_strong` for a lock-free initial check. The `expansion_mutex_` is available but primarily seems intended for the `MCTSTaskflowEngine`. The main `MCTSEngine` doesn't appear to use this mutex in its expansion path, relying on the atomic `is_expanded_`. This is good.

  * **Race Conditions**:

      * **Current**: Atomics are used for most shared data in `MCTSNode` and `MCTSEngine` counters. `std::shared_ptr` and `std::weak_ptr` manage memory lifetimes.
      * **Recommendations**:
        1.  **Double-Checked Locking in `ensureEvaluatorStarted`**: The pattern used is correct: check, lock, double-check.
            ```cpp
            // src/mcts/mcts_engine.cpp
            // if (evaluator_started_.load(std::memory_order_acquire)) return true; // First check
            // std::lock_guard<std::mutex> lock(evaluator_mutex_);
            // if (evaluator_started_.load(std::memory_order_relaxed)) return true; // Second check
            // evaluator_->start();
            // evaluator_started_.store(true, std::memory_order_release);
            ```
        2.  **Initialization of `root_` and TT**: `MCTSEngine::search` first clears TT, then resets `root_`, then creates a new `root_`. This order is important to avoid using stale pointers. Storing the new root in the TT afterwards is also correct.
        3.  **Leaf State Cloning**: Ensure `cloneGameState(const core::IGameState& state)` and the underlying `state.clone()` are deeply thread-safe if the `GameStatePoolManager` has internal shared structures that aren't protected. The pool manager itself should handle its own concurrency.

  * **Memory Issues (Leaks, Aliasing, Dangling Pointers)**:

      * **Current**: `std::shared_ptr` and `std::weak_ptr` (for `parent_` in `MCTSNode` and `node` in `TranspositionEntry`) are used, which helps prevent many common memory issues. `MCTSNodePool` is present in the `MCTSTaskflowEngine` but not explicitly used by the main `MCTSEngine`'s `MCTSNode::create` (it uses `new MCTSNode`). *Correction*: `MCTSEngine` does not use `MCTSNodePool` directly, nodes are created with `MCTSNode::create` which calls `new MCTSNode`. The `MCTSTaskflowEngine` uses `MCTSNodePool`. This analysis focuses on `MCTSEngine`.

      * **Recommendations**:

        1.  **`PendingEvaluation` and `EvaluationRequest` State Management**: These structures hold `std::unique_ptr<core::IGameState>`. Ensure proper `std::move` semantics are used when these are passed through queues and processed, to avoid premature deletion or double frees. The provided code for `EvaluationRequest` move constructor/assignment seems careful about not nullifying `other.node` as it's a non-owning `shared_ptr` (actually `std::shared_ptr<MCTSNode> node;`), but `state` is a `std::unique_ptr<core::IGameState> state;` which *is* moved. This looks correct.
        2.  **`TranspositionTable` Node Lifetime**: Storing `std::weak_ptr<MCTSNode>` in `TranspositionEntry` is correct. When retrieving, `entry->node.lock()` attempts to get a `shared_ptr`. If the node has been deleted, this will correctly return `nullptr`.
        3.  **Node Cleanup in `MCTSEngine::~MCTSEngine()`**:
            ```cpp
            // src/mcts/mcts_engine.cpp - MCTSEngine::~MCTSEngine()
            // ...
            // while (leaf_queue_.try_dequeue(temp_eval)) {
            //     if (temp_eval.node) { temp_eval.node->clearEvaluationFlag(); } // Good
            // }
            // ... similar for batch_queue_ and result_queue_
            // ...
            // root_.reset(); // This will trigger recursive deletion of nodes if root_ is the sole owner
            ```
            The clearing of evaluation flags on nodes in queues during destruction is good practice to prevent nodes from being stuck in an "evaluating" state if the engine is destroyed mid-search. The `root_.reset()` will delete the tree. If TT entries point to these nodes via `weak_ptr`, they will correctly expire.

      * **General Parallelism Strategy**:

          * The current OpenMP-based leaf parallelization in `MCTSEngine::runSearch` is a common approach.

          * The distinction between the `omp_in_parallel()` path (serial leaf collection) and the `#pragma omp parallel` path needs careful consideration. If `runSearch` is often called from an existing outer parallel region (e.g., in self-play data generation), the serial collection path becomes critical. It must be efficient enough to feed the `MCTSEvaluator`. Its current loop structure for leaf collection is similar to the parallel one but single-threaded. It might benefit from claiming more simulations at once (e.g., `simulations_to_claim = std::min(settings.batch_size / 2, old_sims);`) to try and fill a batch more quickly.

          * **Synchronization of Batches and Results**: The use of separate `leaf_queue_` and `result_queue_` with the `MCTSEvaluator` is standard. The `resultDistributorWorker` handles backpropagation from `result_queue_`. This decoupling is good for throughput. Ensure the `resultDistributorWorker` is highly efficient and doesn't become a bottleneck. Its use of `#pragma omp parallel for schedule(dynamic, 4)` for processing `result_batch` seems like an over-parallelization if `result_batch` itself isn't very large or if the work per result (backpropagation) isn't substantial. This could add overhead. A simple loop might be faster if batch sizes from the evaluator are moderate.

            ```cpp
            // src/mcts/mcts_engine.cpp - resultDistributorWorker
            // #pragma omp parallel for schedule(dynamic, 4)
            // for (size_t i = 0; i < result_batch.size(); ++i) {
            //     // ... backPropagate ...
            // }
            ```

            Consider removing this inner OpenMP loop if `result_batch` (from `result_queue_.try_dequeue_bulk(result_batch.data(), 64)`) is typically small to medium (e.g., \< 64-128) or if backpropagation is very fast. The overhead of creating parallel regions might outweigh benefits.
            \</parallelization\_improvements\>

\<gpu\_throughput\_scenario\>
Here's a concrete scenario to increase GPU throughput, assuming the NN model benefits from larger batches:

**Goal**: Consistently feed the `MCTSEvaluator` with batches close to `settings_.batch_size` (e.g., 256) and minimize GPU idle time.

**Current State (Hypothetical Bottleneck):**

  * `MCTSEngine` threads generate leaves sporadically.
  * Leaf collection logic in `MCTSEngine::runSearch` submits small partial batches (e.g., 1-32 leaves) to `leaf_queue_` due to short timeouts or aggressive `MIN_BATCH_SIZE` triggers.
  * `MCTSEvaluator::processBatch` (external queue path) picks up these small submissions. Its own `max_wait` (e.g., 2-15ms) is too short to accumulate a large batch from multiple small submissions if they don't arrive rapidly.
  * Result: GPU processes many small batches (e.g., 16, 32, 48) instead of fewer large ones (e.g., 256), leading to underutilization.

**Proposed Scenario & Changes:**

1.  **MCTSEngine: Leaf Accumulation & Submission Refinement**

      * **Objective**: `MCTSEngine`'s OpenMP threads (or the single thread in the `omp_in_parallel()` path) should aim to contribute to larger conceptual batches before notifying the evaluator too frequently.
      * **Change 1: Modify `active_simulations_` logic** as described in `optimization_scheme` (Task 2) to ensure CPUs keep searching for leaves for the full duration of `num_simulations`.
      * **Change 2: Adjust Leaf Batch Submission from `MCTSEngine` (`runSearch`)**:
          * In the loop `while (active_simulations_.load(std::memory_order_acquire) > 0)` within `runSearch`:
              * **Serial Path (`omp_in_parallel()`):**
                  * `MIN_BATCH_SIZE`: Increase from 1 to `settings_.batch_size / 4` (e.g., if batch\_size is 256, min is 64).
                  * `MAX_BATCH_COLLECTION_TIME`: Increase from 5ms to `settings_.batch_timeout / 2` (e.g., if timeout is 20ms, this becomes 10ms). Remove the `consecutive_empty_tries` condition or make it much higher if `MIN_BATCH_SIZE` is larger.
                  * The goal is for this single thread to try harder to make a substantial contribution to the `leaf_queue_`.
              * **OpenMP Parallel Path:**
                  * Each thread has `thread_batch` with `OPTIMAL_THREAD_BATCH` (e.g., 16 or `batch_size / num_threads`).
                  * The condition `thread_batch.size() >= OPTIMAL_THREAD_BATCH || consecutive_empty >= MAX_EMPTY_ATTEMPTS || active_simulations_.load(std::memory_order_acquire) == 0` for submission is reasonable. Ensure `OPTIMAL_THREAD_BATCH` is not too small (e.g., at least 8-16).
          * **Pilot Code Snippet (Conceptual for `MCTSEngine::runSearch` serial path):**
            ```cpp
            // MCTSEngine::runSearch - serial leaf collection part
            const size_t MIN_LEAF_SUBMISSION_SIZE = settings_.batch_size / 4; // e.g., 64
            const auto MAX_LEAF_COLLECTION_TIME = settings_.batch_timeout; // Use full configured timeout, e.g., 20ms

            // ... inside the while(active_simulations_ > 0) loop ...
            // Collection loop:
            // while (leaf_batch.size() < OPTIMAL_BATCH_SIZE && active_simulations_ > 0 &&
            //        (std::chrono::steady_clock::now() - batch_start_time) < MAX_LEAF_COLLECTION_TIME &&
            //        pending_evaluations_.load() < settings_.batch_size * SOME_FACTOR) {
            //    // ... try to find and add leaves to leaf_batch ...
            // }

            // Submission logic:
            // if (!leaf_batch.empty() &&
            //     (leaf_batch.size() >= MIN_LEAF_SUBMISSION_SIZE ||
            //      (std::chrono::steady_clock::now() - batch_start_time) >= MAX_LEAF_COLLECTION_TIME ||
            //      active_simulations_.load() == 0) ) {
            //    // ... enqueue leaf_batch ...
            //    evaluator_->notifyLeafAvailable(); // Notify evaluator
            //    leaf_batch.clear();
            // }
            ```

2.  **MCTSEvaluator: Patient Batch Aggregation**

      * **Objective**: `MCTSEvaluator` should wait longer and more intelligently to form batches close to `settings_.batch_size`.

      * **Change 1: `MCTSEvaluator::processBatch` (external queues path)**:

          * `MIN_BATCH`: Significantly increase. Instead of adapting down to 1, the absolute minimum should be `settings_.batch_size / 8` or a fixed value like 32.
          * `OPTIMAL_BATCH`: Should be `settings_.batch_size`.
          * `max_wait`: This should be the primary adaptive parameter. Start with `settings_.batch_timeout` (e.g., 20ms).
              * If batches are consistently smaller than `OPTIMAL_BATCH` but `max_wait` is reached, and GPU is idle, increase `max_wait` (e.g., up to 50-100ms).
              * If batches are full but NN processing takes much longer than `max_wait`, `max_wait` can be somewhat shorter but still enough to gather a good batch.
          * The "Phase 2: If below minimum, wait briefly for more" logic needs to be more patient.

      * **Change 2: `MCTSEvaluator::notifyLeafAvailable()`**:

          * The current notification batching (`count >= target_batch || time_since_last > 2000us`) might be okay, but ensure `target_batch` for notification is not too large (e.g., `batch_size_ / 4` rather than `/2`). The critical fix for immediate notification for external queues is good.

      * **Pilot Code Snippet (Conceptual for `MCTSEvaluator::processBatch` - external queue path):**

        ```cpp
        // MCTSEvaluator::processBatch - for external queues
        const size_t TARGET_BATCH_SIZE = settings_.batch_size; // e.g., 256
        const size_t MINIMUM_PROCESSING_BATCH_SIZE = std::max(32UL, TARGET_BATCH_SIZE / 8); // e.g., 32
        std::chrono::milliseconds current_max_wait = settings_.batch_timeout; // e.g., 20ms, make this adaptive

        // Phase 1: Bulk dequeue
        evaluations.resize(TARGET_BATCH_SIZE); // Pre-allocate
        size_t dequeued_count = external_leaf_queue->try_dequeue_bulk(evaluations.data(), TARGET_BATCH_SIZE);
        evaluations.resize(dequeued_count);

        // Phase 2: Wait for more if not full and not enough to meet minimum
        if (evaluations.size() < TARGET_BATCH_SIZE && (evaluations.empty() || evaluations.size() < MINIMUM_PROCESSING_BATCH_SIZE)) {
            auto deadline = std::chrono::steady_clock::now() + current_max_wait;
            while (std::chrono::steady_clock::now() < deadline && evaluations.size() < TARGET_BATCH_SIZE &&
                   !shutdown_flag_.load()) {
                PendingEvaluation temp_eval;
                if (external_leaf_queue->try_dequeue(temp_eval)) {
                    evaluations.push_back(std::move(temp_eval));
                } else {
                    // Only yield/sleep if we are still below MINIMUM_PROCESSING_BATCH_SIZE
                    if (evaluations.size() < MINIMUM_PROCESSING_BATCH_SIZE) {
                         std::this_thread::sleep_for(std::chrono::microseconds(500)); // Longer sleep
                    } else {
                         break; // Have enough to process if timeout is near
                    }
                }
            }
        }

        // Only process if we have at least MINIMUM_PROCESSING_BATCH_SIZE or shutting down
        if (evaluations.size() >= MINIMUM_PROCESSING_BATCH_SIZE || (shutdown_flag_.load() && !evaluations.empty())) {
            // ... proceed to inference ...
            // Adapt current_max_wait based on batch fill rate and NN processing time
            // Example: if batch was full and processed quickly, current_max_wait could be slightly reduced.
            // If batch was small and timeout hit, current_max_wait could be increased.
        } else if (!evaluations.empty()) {
            // Not enough to meet MINIMUM_PROCESSING_BATCH_SIZE, and not shutting down.
            // Re-queue items to avoid processing tiny batches.
            // (More complex: or hold them in a temporary buffer in evaluator)
            external_leaf_queue->enqueue_bulk(std::make_move_iterator(evaluations.begin()), evaluations.size());
            evaluations.clear();
            return false; // Did not process a batch
        } else {
            return false; // No items
        }
        ```

3.  **Monitoring and Tuning**:

      * Add detailed logging for:
          * Number of leaves submitted by `MCTSEngine` per batch submission.
          * Actual batch sizes processed by `MCTSEvaluator`.
          * Time spent by `MCTSEvaluator` waiting for leaves.
          * NN inference time per batch.
          * Queue sizes (`leaf_queue_`, `result_queue_`) periodically.
      * Use this data to fine-tune the new timeout values, minimum batch sizes, and the `active_simulations_` logic.

By implementing these changes, the `MCTSEngine` will more effectively utilize CPU resources to generate a steadier and larger stream of leaves. The `MCTSEvaluator` will then be more patient, aggregating these leaves into larger batches that can saturate the GPU, leading to higher throughput and faster MCTS searches.
\</gpu\_throughput\_scenario\>

----------
----------

\<todo\_list>

1. **Instrument and Profile**: Add fine-grained timing logs and counters around critical regions – selection (tree traversal), leaf evaluation batching, neural network inference, and backpropagation – to pinpoint delays. For example, log batch sizes and inference times for each `MCTSEvaluator` batch to see how often we process suboptimal small batches. Monitor CPU thread utilization (time spent active vs. waiting) and GPU utilization per batch. This data will guide further optimizations.
2. **Tune MCTS Settings**: Revisit MCTS configuration defaults to better match available hardware. Ensure `MCTSSettings.num_threads` equals the actual CPU core count (or use `std::thread::hardware_concurrency` if not explicitly set) to avoid oversubscription or underuse. Likewise, adjust `batch_size` and `batch_timeout` to balance throughput vs. latency – e.g. lower `batch_size` from 256 if a single game never approaches that, or slightly increase `batch_timeout` to allow a fuller batch when GPU is underutilized. These simple tweaks can immediately improve CPU/GPU utilization.
3. **Increase Parallel Work**: If feasible for the use-case, run more MCTS searches in parallel. For self-play or simulation, use the existing support for multiple engines/games (the `SelfPlayManager` can spawn several MCTSEngines with a shared evaluator) to aggregate more leaf evaluations per batch. This will raise the average batch size and keep the GPU busier. In a single-game scenario (where parallel games aren’t possible), consider enabling root parallelization (e.g. `num_root_workers=4`) which runs multiple search trees for one move – this can similarly increase concurrent leaf evaluations.
4. **Optimize Leaf Parallelism**: Refine the OpenMP-based leaf selection loop for better efficiency. Currently each thread claims up to 4 simulations at a time and flushes when its local batch is full or work is done. Experiment with larger chunk sizes per thread or dynamic task distribution: e.g. let threads claim more simulations per iteration if many remain, to reduce loop overhead and context switching. Also consider OpenMP tasks or a work-stealing model instead of a fixed `to_claim=4`, to ensure all threads remain busy if some finish early. The goal is to keep CPU threads fully utilized until all simulations are queued.
5. **Reduce Synchronization Overhead**: Identify and alleviate any lock contention or atomic hotspots. For instance, node expansion uses a mutex (per node) – check if this ever bottlenecks (e.g. many threads expanding children of the same node). If so, alternatives like lock-free techniques or more coarse-grained locking (expand a batch of children at once) could help. Similarly, the atomic counters for `pending_evaluations_` and virtual loss could become contended with many threads; monitor their contention rates. If contention is high, batching atomic operations or using thread-local accumulators (with occasional combining) might improve scalability.
6. **Enhance Batching Mechanism**: Ensure the centralized evaluator isn’t processing fragmented mini-batches due to timing misalignment. The design already waits a short time for additional requests, but we can fine-tune this. For example, if GPU is underutilized (low batch size) and latency is not critical, extend the wait window a bit to collect more requests (e.g. increase the 0.5ms or 5ms wait in `evaluationLoop`). Conversely, if latency is important, make sure we aren’t waiting too long when only a few requests are outstanding (perhaps by lowering `min_batch_size` thresholds when the queue is sparse). This tuning, guided by the instrumentation above, will improve throughput without stalling unnecessarily.
7. **Leverage Result Concurrency**: The result distribution stage can be optimized further. In single-engine mode, the separate `resultDistributorWorker` thread processes network outputs in parallel using OpenMP. Check if this thread ever becomes a bottleneck (it likely doesn’t, since backpropagation is lightweight). If it does, we could merge result processing into the evaluator thread (processing results immediately after inference on the same thread) to eliminate context switching – but only if it doesn’t delay starting the next batch. Alternatively, for very large batches, increase the OpenMP parallelism or scheduling for result processing (the code uses dynamic scheduling in chunks of 4; ensure this is appropriate for the typical result batch sizes).
8. **Memory and Data Efficiency**: Profile memory usage and copy costs. Cloning game states for every evaluation can be expensive. The code even includes a `gamestate_pool` utility for recycling state objects – consider re-enabling or improving this to cut down on allocations. Also verify that game state cloning (and destruction) is not holding locks or causing CPU stalls. If it is, optimize the game state representation or copy method (e.g., use move semantics or allocate once and reset state). Reducing this overhead will make leaf expansion faster and threads will spend more time computing and less time waiting on memory.
9. **Validate Thread Safety**: Fix any race conditions discovered. For example, ensure virtual losses and “isBeingEvaluated” flags are correctly applied. In the current code, a thread sets a node’s evaluation-in-progress flag via `tryMarkForEvaluation()` and other threads skip that node – this prevents duplicate evaluations. Confirm that every such flag is cleared at backpropagation, even in edge cases. The result processing already clears the evaluation flag on each node after backprop. Likewise, double-check the atomic updates in backpropagation (visit counts, value sums) for any missed synchronization (they appear to use atomics for these stats). Address any minor bugs (e.g. counting a simulation as completed even if it found a node already under evaluation) to ensure the search doesn’t terminate early or record incorrect counts.

\</todo\_list>

\<optimization\_scheme>
**Step 1: Measure Current Performance** – Use the added instrumentation to gather metrics on CPU utilization (per thread), average batch size, inference latency, and GPU utilization during MCTS. Identify the primary bottleneck. For instance, you might find that the GPU is only 20% utilized with batches of \~32, and CPU threads spend a lot of time waiting on inference.

**Step 2: Increase Parallelism** – Gradually introduce more parallel work to saturate resources. If the metrics show low GPU usage, enable more parallel games or root workers. For example, start 2 self-play games in parallel and observe batch sizes. If GPU usage improves (e.g. batches average 64+ and GPU jumps to >50% utilization), you’re on the right track. Continue scaling (3, 4 games, etc.) until you approach high GPU occupancy. Monitor that CPU threads can handle the extra load – with 2–4 games, the 12 CPU threads will be split among games (or additional threads used, depending on config). Ensure CPU usage remains healthy (near 100% across cores).

**Step 3: Tune Batching Parameters** – Based on the data, adjust `batch_size` and timeouts. If parallel games aren’t an option (or even if they are), you might reduce `batch_size` to, say, 128 so that the system isn’t always waiting to hit 256. The code already sets a dynamic minimum (75% of batch or 64), so with batch\_size=128 the evaluator will aim for \~96 samples or 64 minimum, which may be closer to what one game can produce. Test the effect: if batch\_size=128 yields higher average batch (because previously many batches ended up around the hard floor of 32–64), GPU throughput will increase. Conversely, if using parallel games yields consistently large batches, you might *increase* batch\_size to fully utilize the GPU (just ensure it doesn’t inflate latency too much). Also tune the wait times: e.g., if metrics show the evaluator often times out at 5ms with only half-full batches, consider a slightly longer wait (10–15ms) when plenty of simulations remain, to grab more samples. The goal is to maximize the area under the “GPU busy” curve without adding excessive idle delay.

**Step 4: Optimize CPU Loop** – If profiling shows CPU threads finishing leaf selection very quickly and then idling, try to get more work out of them. Increase the `to_claim` chunk or allow threads to immediately proceed to another iteration without waiting for others. For example, changing `to_claim` from 4 to 8 (or making it adaptive based on how many leaves were actually found last round) could reduce the overhead of synchronizing the parallel loop. Benchmark the impact on throughput (sims/sec). Also, evaluate whether the OpenMP critical section (currently empty) or other sync points are limiting scaling – they likely aren’t, but if they are, remove or minimize them.

**Step 5: Validate Improvements with Metrics** – After each tweak, collect the same metrics to confirm the effect. For instance, after increasing parallel games or adjusting batch size, check that average batch size increased and GPU utilization improved accordingly (e.g., average batch maybe went from 32 to 100, and GPU compute time per batch increased, but overall inference throughput in samples/sec went up). Likewise, ensure the total time per 800 simulations actually drops. It’s possible that bigger batches make each inference slower in absolute time; the key is that they process so many more states at once that the net simulations/sec rises. If you ever see diminishing returns (e.g., going from 2 to 4 parallel games yields no further GPU gain because the GPU was already saturated, while CPU overhead doubles), you’ve hit a limit – back off to the sweet spot.

**Step 6: Memory and Other Micro-optimizations** – Once CPU/GPU utilization is much improved, address secondary bottlenecks. For example, if cloning game states is now a noticeable fraction of CPU time (because everything else sped up), integrate the `GameStatePool` to recycle states and avoid malloc/free overhead. Profile again to ensure this doesn’t introduce contention (it can be implemented with thread-local pools to avoid locks). Also consider caching network inputs: if multiple leaf states repeat (e.g., transpositions), avoid duplicate neural evaluations by using the transposition table to store NN outputs for seen states – the groundwork is there with `TranspositionTable`, so leverage it to skip evaluating identical states twice.

**Step 7: Iterate and Test at Scale** – Apply these optimizations incrementally and test with full-scale runs. It’s important to verify that the search still yields correct results (no regression in algorithm accuracy) after changes. Use unit tests or small scenarios to ensure that multi-threaded search still expands the correct number of nodes and selects reasonable moves. Also, monitor stability in long runs – e.g., no race conditions causing occasional nan values or crashes. By iteratively optimizing and verifying, you will converge on a configuration where both CPU and GPU are maximally utilized, and MCTS runs significantly faster.

\</optimization\_scheme>

\<parallelization\_improvements>
**OpenMP Leaf Parallelism**: The MCTS uses OpenMP to parallelize tree exploration across CPU cores – each thread independently selects leaves to evaluate. This “leaf parallelization” is effective, but we can refine it. One improvement is to use a dynamic task distribution instead of the fixed 4-simulation chunk per thread. For example, leveraging OpenMP’s dynamic scheduling or tasking could let threads grab new simulations as soon as they finish current ones, preventing any core from sitting idle. The current code does a manual chunking with `compare_exchange_weak` on `active_simulations_` – which works, but a task-based approach would simplify load balancing. We could spawn each simulation as an OpenMP task (or use a smaller chunk size) and let the runtime schedule them among threads. This may yield better utilization, especially if some simulations take longer (e.g., a thread that hits many transpositions might finish its 4 slower than others). In testing, ensure the overhead of tasking doesn’t outweigh the benefit; with 800 simulations, tasks are fine-grained but should still be manageable.

**Reduce Critical Sections**: The code enters an empty `#pragma omp critical` at thread start, likely as a placeholder. This can be removed to avoid any implied barrier or serialization (even if empty, it introduces a minor sync point during thread launch). Additionally, review any other `#pragma omp critical` or `omp atomic` usage in hot loops. For instance, the result distribution uses an OpenMP parallel for with atomic operations on `pending_evaluations_` for each result. We could instead accumulate local counters and do one atomic update per batch. Though the overhead here is small, it’s an easy win to minimize atomic ops inside loops.

**Thread Binding and Affinity**: To maximize CPU cache usage and avoid context switching overhead, consider binding threads to cores (affinity). OpenMP usually handles this, but it might help on NUMA systems to ensure all threads of MCTS stick to one socket if possible (to keep memory accesses local). You can use environment settings (`OMP_PROC_BIND`) or programmatically set affinities. This won’t improve algorithmic scalability, but can give a few percent performance boost by reducing cache misses.

**Parallelize Backpropagation (if needed)**: Backpropagating values up the tree is done for each result in the distribution phase. In the single-engine mode, the `resultDistributorWorker` already parallelizes over results using OpenMP. If using the shared-queue mode (multiple engines), currently the main thread processes results in a loop. We could similarly add an OpenMP parallel for when draining the shared result queue to handle dozens of results concurrently. However, be cautious: backprop updates shared ancestor nodes, so multiple threads could write to the same node’s statistics concurrently. In our implementation this is safe because updates use atomic adds to `visit_count_` and `value_sum_`. Still, parallel backprop offers diminishing returns – the updates are fast, and doing them serially might already be a small fraction of time. It’s worth profiling: if result processing (including setting priors and clearing flags) ever becomes a bottleneck when batches are huge, then parallelizing it (with proper synchronization) could help.

**Lock-Free Structures**: We already use lock-free queues (moodycamel) for leaf and result queues. Ensure we maximize their benefit. For example, use the bulk-dequeue/enqueue calls wherever possible (the code does this for batches of leaves and results). One improvement is to avoid waking threads too frequently. The code calls `notifyLeafAvailable()` on the evaluator for every enqueue operation. Instead, we could notify only when transitioning from an empty to non-empty queue (to wake a sleeping evaluator thread), and thereafter let the evaluator thread bulk-process everything. This would cut down on extraneous wake-ups (which involve locking a mutex in the evaluator’s CV). Implementing this might require a slight change: track a flag or use `queue_size_approx()` to decide when to notify. The goal is to reduce synchronization between producer threads and the evaluator thread while not adding latency.

**Transposition Table Concurrency**: If transposition table lookups/insertions are enabled (`use_transposition_table=true`), ensure the underlying data structure scales in parallel. The code uses a parallel hash map (phmap) which is designed for concurrency. Confirm that the number of shards (`num_shards`) is set to a high value (the code sets it to at least `num_threads` shards). This spreads lock contention. In testing, monitor the TT hit rate and ensure that TT operations aren’t slowing down simulations. If they are, one could tune the hash map (e.g., use reserve() to avoid rehashing, or even consider lock-free reads for TT hits). In summary, the existing parallelization is solid – these improvements refine it by balancing load better and trimming unnecessary sync overhead, leading to smoother scaling across threads.

\</parallelization\_improvements>

\<gpu\_throughput\_scenario>
Imagine currently the system runs 800 simulations with an average neural-net batch of only \~32 states. Suppose the neural network (a ResNet) can process 32 states in 8 milliseconds on the GPU. That’s 25 forward passes to handle all 800 evaluations, taking \~200ms, and the GPU is mostly idle between these small bursts. Now, apply our optimizations: we enable 4 parallel games (or 4 root workers) so that the evaluator sees roughly 4× the requests before firing. Now batches come in closer to 128 states. Processing 128 states might take, say, 15ms on the GPU. Even though each batch is slower, we’re doing the work of 4 small batches in one go – effectively achieving \~8.5 states per millisecond vs. 4 states/ms before (throughput more than doubled). In this scenario, the GPU utilization jumps because it’s crunching a larger batch for 15ms straight with fewer idle gaps. The overall time to evaluate 800 states for all games might drop from 200ms to on the order of 100–120ms. This means each game’s MCTS finishes faster. The CPU threads, meanwhile, stay busy preparing the next batch of leaves while the GPU works – thus overlapping computation.

Concretely, let’s say before optimization, GPU utilization was around 20% and CPU utilization 50%. After running 4 games in parallel with tuned parameters, you might see GPU utilization rise to \~80% (processing \~128 at a time efficiently) and CPU utilization to \~90% (threads now have more total work across games). The throughput in terms of simulations per second could increase from \~4k/s to \~8k/s (hypothetical doubling), drastically reducing the per-move computation time. Even in a single-game scenario, using root parallelization and adjusting `batch_timeout` can let the evaluator wait a few extra milliseconds to gather, for example, 64 states instead of 32. If the network processes 64 states in 10ms (instead of 2×8ms for two 32-state batches = 16ms), that’s a 37% speedup for the inference phase. The net effect is that MCTS can either finish the same number of simulations much quicker or afford more simulations in the same time budget, leading to stronger search without lag.

In summary, by batching more work per GPU call and keeping the hardware busy, we move from a regime of underutilization to near-optimal utilization. The GPU’s compute power is fully leveraged – in the improved scenario it spends most of its time evaluating neural networks rather than waiting – and the CPU cores likewise are coordinating continuously without long stalls. This balanced high utilization of CPU/GPU translates directly into faster and more efficient MCTS, which is crucial for both playing strength and training throughput in an AlphaZero-like system.

\</gpu\_throughput\_scenario>