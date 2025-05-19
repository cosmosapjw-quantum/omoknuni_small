\<todo\_list\>

1.  **High Priority: Address Low GPU Throughput & Small Batch Sizes (9.5/10)**
      * Task: Tune `MCTSEvaluator` batching parameters (`max_wait`, `MIN_BATCH`, `OPTIMAL_BATCH` thresholds in `evaluationLoop`/`processBatch` for external queues, and `timeout_` for internal queues). Log actual batch sizes and queue dynamics to guide tuning.
      * Task: Investigate and reduce game state cloning. The path `MCTSEngine::traverseTree` (clones to `shared_ptr`) -\> `leaf_queue_` -\> `MCTSEvaluator::processBatch` (clones `shared_ptr` to `unique_ptr` for `inference_fn_`) involves at least two clones. Modify `PendingEvaluation` to use `std::unique_ptr<core::IGameState>` if `inference_fn_` requires it, or modify `inference_fn_` to accept `const core::IGameState&` or `std::shared_ptr<core::IGameState>` to eliminate the second clone.
      * Task: Profile and optimize leaf generation rate in `MCTSEngine` (e.g., `selectLeafNode` performance, OpenMP scheduling).
2.  **High Priority: Resolve Low CPU Usage / Parallelism Bottlenecks (9/10)**
      * Task: Parallelize `MCTSEngine::resultDistributorWorker`. Currently, it's a single thread processing all NN results and performing backpropagation, which is a major bottleneck.
      * Task: Review OpenMP usage in `MCTSEngine::runSearch`. Ensure effective load balancing and minimize serial sections, especially in the `omp_in_parallel()` path.
      * Task: Analyze and reduce synchronization overhead. Evaluate if polling in `resultDistributorWorker` and `MCTSEvaluator` is optimal or if condition variables (for internal queues) would be better. (MoodyCamel queues are generally fine with polling, but the sleep/yield strategy needs review).
3.  **Medium Priority: Optimize `MCTSNode` Operations (7.5/10)**
      * Task: Simplify `MCTSNode::expand` synchronization. Clarify if `expansion_mutex_` is needed given the atomic `is_expanded_` CAS. If used (e.g., by `MCTSTaskflowEngine`), ensure consistency.
      * Task: Profile OpenMP usage in `MCTSNode::selectChild` and potentially disable it for smaller child counts or replace with a more lightweight parallel approach if C++17 `std::reduce` is available.
4.  **Medium Priority: Transposition Table Efficiency (7/10)**
      * Task: Monitor contention on `TranspositionTable::clear_mutex_` during `enforceCapacityLimit`. If it's an issue, consider more fine-grained or probabilistic eviction strategies that avoid a global lock.
5.  **Low Priority: `MCTSNodePool` Contention (6.5/10)**
      * Task: Profile `MCTSNodePool::pool_mutex_`. If it becomes a bottleneck, consider sharding the free list or using thread-local caches.
6.  **Low Priority: Clarify `NodeTracker` Role (6/10)**
      * Task: Determine if `NodeTracker` is actively and beneficially used by `MCTSEngine`. If it's redundant with the `MCTSEvaluator`'s promise/future mechanism for `MCTSEngine`, consider streamlining.
7.  **Low Priority: Review Engine Shutdown Logic (6/10)**
      * Task: Thoroughly review `MCTSEngine::~MCTSEngine` for potential race conditions or deadlocks during shutdown, especially related to thread joining and queue clearing.
        \</todo\_list\>

\<optimization\_scheme\>
The overall strategy is to identify and alleviate bottlenecks in the MCTS pipeline, focusing on improving parallelism, reducing overhead, and enhancing GPU utilization.

**Step 1: Reduce Game State Cloning Overhead**

  * **Problem:** Cloning game states is expensive and happens multiple times.
    1.  `MCTSEngine::traverseTree` calls `cloneGameState` (returns `shared_ptr`).
    2.  This `shared_ptr` goes into `PendingEvaluation::state`.
    3.  `MCTSEvaluator::processBatch` (external queue path) calls `evaluations[i].state->clone()` to get a `unique_ptr` for `inference_fn_`.
  * **Fix:**
    1.  Modify `PendingEvaluation` struct (in `mcts_engine.h` context) to hold `std::unique_ptr<core::IGameState> state;`.
    2.  In `MCTSEngine::traverseTree` (and other places creating `PendingEvaluation` for the leaf queue), clone directly to `unique_ptr`:
        ```cpp
        // In MCTSEngine::traverseTree, when creating PendingEvaluation
        // auto state_clone_shared = cloneGameState(leaf_state); // Old
        // pending.state = state_clone_shared; // Old

        auto state_clone_unique = leaf_state.clone(); // Assuming IGameState::clone() returns unique_ptr
        if (!state_clone_unique) { /* handle error, clear eval flag */ return; }
        pending.state = std::move(state_clone_unique); // New
        ```
    3.  The `MCTSEvaluator::inference_fn_` is defined as `std::function<std::vector<NetworkOutput>(const std::vector<std::unique_ptr<core::IGameState>>& states)>`.
        If `PendingEvaluation` now holds `unique_ptr`, then in `MCTSEvaluator::processBatch`:
        ```cpp
        // In MCTSEvaluator::processBatch (external queue path)
        // auto unique_clone = evaluations[i].state->clone(); // Old, assuming evaluations[i].state was shared_ptr
        // states.push_back(std::move(unique_clone)); // Old

        // New, if evaluations[i].state is already unique_ptr:
        if (evaluations[i].state) { // Make sure it wasn't moved already or null
            states.push_back(std::move(evaluations[i].state));
            valid_indices.push_back(i);
        }
        ```
    This eliminates one clone if `IGameState::clone()` returns `unique_ptr` and the `PendingEvaluation` struct is updated. If `cloneGameState` was essential for pooling `shared_ptr`s, this change is more complex. A simpler first step is to see if `inference_fn_` can accept `const std::vector<std::shared_ptr<core::IGameState>>&` or `const std::vector<core::IGameState*>&` (with careful lifetime management) to avoid the clone within the evaluator.

**Step 2: Parallelize Result Processing and Backpropagation**

  * **Problem:** `MCTSEngine::resultDistributorWorker` is a single thread.
  * **Fix:** Modify `resultDistributorWorker` to use an OpenMP parallel loop to process dequeued results.
    ```cpp
    // In MCTSEngine::resultDistributorWorker()
    // After dequeuing a batch of results (e.g., result_batch)

    // #pragma omp parallel for schedule(dynamic, 1) // Schedule can be tuned
    // for (size_t i = 0; i < result_batch.size(); ++i) {
    //     if (shutdown_.load(std::memory_order_acquire)) {
    //         continue; // Use 'continue' for omp loop, or check before parallel region
    //     }
    //     auto& output = result_batch[i].first;
    //     auto& eval_request = result_batch[i].second; // This is PendingEvaluation

    //     if (eval_request.node) {
    //         try {
    //             eval_request.node->setPriorProbabilities(output.policy);
    //             backPropagate(eval_request.path, output.value); // backPropagate needs to be thread-safe (uses atomics)
    //             eval_request.node->clearEvaluationFlag();
    //         } catch (const std::exception& e) {
    //             // Log error
    //             if(eval_request.node) eval_request.node->clearEvaluationFlag();
    //         } catch (...) {
    //             // Log error
    //             if(eval_request.node) eval_request.node->clearEvaluationFlag();
    //         }
    //     }
    //     pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
    //     total_results_processed_.fetch_add(1, std::memory_order_relaxed);
    // }
    ```
    *Ensure `backPropagate` and `MCTSNode::update` are fully thread-safe (they appear to be, using atomics).*

**Step 3: Tune `MCTSEvaluator` Batching Logic**

  * **Problem:** Default batching waits/thresholds in `MCTSEvaluator` might be too aggressive, leading to small batches.
  * **Fix:**
    1.  **Logging:** Add detailed logs in `MCTSEvaluator::processBatch` (external queue path) for:
          * `external_leaf_queue->size_approx()` at the start.
          * `evaluations.size()` after initial bulk dequeue.
          * `evaluations.size()` after the wait loop.
          * Time spent in the wait loop.
          * Actual batch size sent to `inference_fn_`.
    2.  **Parameter Adjustment:** Based on logs, adjust:
          * `max_wait` (in `mcts_evaluator.cpp#L260-L273`): Increase this (e.g., from 2-15ms to 5-50ms range) to allow more items to accumulate.
          * `MIN_BATCH` thresholds: Potentially increase the `MIN_BATCH` slightly if `max_wait` is increased, to ensure batches are reasonably full.
          * `settings_.batch_timeout` (for internal queue mode, if used): Increase if batches are consistently timing out small.
    3.  **Adaptive Timeout:** Consider a more sophisticated adaptive timeout that increases if the queue is filling quickly and decreases if it's slow, aiming for a target batch size or submission interval.

**Step 4: Review and Optimize OpenMP Usage in `MCTSEngine`**

  * **Problem:** CPU underutilization might stem from OpenMP scheduling or serial portions.
  * **Fix:**
    1.  Profile the OpenMP regions in `MCTSEngine::runSearch`.
    2.  Experiment with OpenMP `schedule` clauses (e.g., `dynamic`, `guided`) for the main simulation loop if load imbalance is suspected.
    3.  Analyze the `if (omp_in_parallel())` block: The serial leaf collection here might be a bottleneck if this engine is frequently run as a nested OpenMP task. Consider if this path can also be parallelized (e.g., using `omp taskloop`).

**Step 5: Incremental Profiling and Iteration**

  * After each significant change, profile the application (CPU, GPU, MCTS nodes/sec, actual batch sizes) to measure impact and identify new bottlenecks.

\</optimization\_scheme\>

\<parallelization\_improvements\>
Improving parallelization involves addressing synchronization, potential deadlocks, lock contention, race conditions, and memory issues.

**1. Synchronization:**

  * **`MCTSEngine::resultDistributorWorker` Polling:**
      * **Issue:** Polls `result_queue_` using `try_dequeue` with `std::this_thread::yield()` or short sleeps. This can consume CPU cycles unnecessarily if the queue is often empty.
      * **Recommendation:** While MoodyCamel queues are designed for polling, if profiling shows significant CPU time spent in this worker while idle, consider a hybrid approach or slightly longer sleep intervals. However, given `MCTSEvaluator` is the producer, if it pushes results and needs quick processing, polling might be preferred over CV overhead. The main fix is parallelizing its work.
        ```cpp
        // In resultDistributorWorker, when queue is empty:
        // static int empty_count = 0;
        // if (++empty_count < 10) { // Yield for a few cycles
        //     std::this_thread::yield();
        // } else { // Then sleep a bit longer
        //     std::this_thread::sleep_for(std::chrono::microseconds(100)); // Current
        //     // Consider slightly increasing this if CPU usage is high during idle for this thread:
        //     // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        //     empty_count = 0;
        // }
        ```
  * **`MCTSNode::expand` Mutex:**
      * **Issue:** `MCTSNode` has `expansion_mutex_`. `MCTSTaskflowEngine` uses it. `MCTSEngine` does not seem to use it in its expansion path, relying on `is_expanded_` atomic CAS.
      * **Recommendation:**
        1.  Clarify if `expansion_mutex_` is strictly necessary if `is_expanded_` CAS correctly serializes the expansion logic for a single node.
        2.  If `children_` and `actions_` (standard vectors) initialization post-CAS needs protection (it generally doesn't if only one thread proceeds past CAS for a given node), then the mutex is needed.
        3.  If `MCTSEngine` is the primary target, and it's proven that its current expansion logic (CAS only) is safe and sufficient, the mutex might be dead code for it. If `MCTSTaskflowEngine` relies on it, it's fine there. Ensure there's no unsafe interaction if both engines could operate on the same tree (unlikely). The CAS on `is_expanded_` should be sufficient to ensure only one thread initializes `children_` and `actions_`.

**2. Deadlocks:**

  * **General Risk:** Complex interactions between MCTS threads, evaluator thread, and result distributor thread, especially during shutdown or if queues fill up, can risk deadlocks.
  * **Current State:** No obvious deadlocks observed in the provided snippets for the main simulation loop. The shutdown sequence in `MCTSEngine::~MCTSEngine` is complex and is a higher-risk area.
      * `mcts_engine.cpp`: The destructor logic like `cv_mutex_destroyed_.store(true, ...)` is non-standard and indicates potential complexities in ensuring threads terminate correctly without trying to use already destroyed mutexes.
  * **Recommendation:**
      * Simplify shutdown logic. Rely on standard patterns: signal threads to stop, notify all CVs, join threads, then clear queues and release resources.
      * Avoid patterns like `mutex_destroyed_` flags. Instead, ensure locks are released before signaling/joining dependent threads.
      * Thoroughly test shutdown under heavy load.

**3. Lock Contention:**

  * **`MCTSNodePool::pool_mutex_`:**
      * **Issue:** A single mutex protects `free_nodes_` and `memory_blocks_`. High rates of node allocation/deallocation from many threads could cause contention.
      * **Recommendation:** If profiling shows this as a hotspot:
          * Consider using multiple free lists (sharded by thread ID or hash) or a lock-free stack for `free_nodes_`.
          * Thread-local caches for nodes: threads allocate from/deallocate to a small local cache, which syncs with the global pool in batches.
  * **`TranspositionTable::clear_mutex_`:**
      * **Issue:** Used in `enforceCapacityLimit` and `clear`. `enforceCapacityLimit` does some work (sampling, sorting) while holding the lock.
      * **Recommendation:** Minimize work under this lock. For `enforceCapacityLimit`, collect candidates (hashes and visit counts) perhaps with finer-grained read access (if map supports it safely) or release the lock before sorting and then re-acquire to delete. Current probabilistic execution (1% of stores) reduces frequency.

**4. Race Conditions:**

  * **General Check:** Atomics are used for most shared node members, and MoodyCamel queues are lock-free. This reduces common race conditions.
  * **`MCTSNode::expand`:** The primary defense is `is_expanded_.compare_exchange_strong`. This should ensure only one thread performs the actual expansion (populating `children_`, `actions_`). Subsequent reads of `children_` and `actions_` must happen after `is_expanded_` is true and the expansion is complete.
  * **Shared Root in Root Parallelization:**
      * `MCTSEngine::runSearch`: If root parallelization is used, `search_roots` contains copies. The final aggregation `action_visit_counts` and `action_value_sums` are applied to the main `root_`. This aggregation itself is single-threaded after parallel searches complete, which is safe.

**5. Memory Issues:**

  * **Dangling Pointers/Use-After-Free:**
      * `std::weak_ptr` in `MCTSNode::parent_` and `TranspositionEntry::node` is correctly used to break cycles and allow objects to be destroyed.
      * Careful handling of `PendingEvaluation` objects, especially `state` and `node` pointers, is needed as they move between threads and queues. `std::move` is used, which is good.
      * In `MCTSEngine` move constructor/assignment, stopping the `other` engine and joining its threads before moving resources is crucial and seems to be attempted.
  * **Memory Leaks:**
      * `MCTSNodePool` aims to manage node memory. Ensure its `NodeDeleter` is correctly called for all allocated nodes.
      * Promises in `EvaluationRequest` must eventually be satisfied or an exception set to avoid resource leaks if the future is waited upon. The evaluator's stop/destructor logic tries to clear pending requests.
  * **`PendingEvaluation::path` Memory:**
      * **Issue:** Storing `std::vector<std::shared_ptr<MCTSNode>> path` for every pending evaluation can consume significant memory for deep trees and many pending evaluations, and also involves shared\_ptr reference counting overhead.
      * **Recommendation:** If memory becomes an issue due to paths:
          * Store only necessary information for backpropagation (e.g., raw pointers if lifetimes can be guaranteed, or just the leaf and reconstruct path upwards, though less efficient for backprop).
          * For RAVE, if path actions are needed, they could be stored more compactly. Current approach of storing full shared\_ptrs is safest but potentially heavy.

**Improving `MCTSEngine` OpenMP Parallel Leaf Generation:**

  * The current OpenMP loop in `runSearch`:
    ```cpp
    // #pragma omp parallel num_threads(actual_threads)
    // {
    //     // Thread-local batch collection
    //     while (active_simulations_.load(std::memory_order_acquire) > 0) {
    //         // Claim simulations
    //         // Loop to run 'to_claim' simulations
    //         //    selectLeafNode -> tryMarkForEvaluation -> cloneGameState -> enqueue to leaf_queue_
    //     }
    //     // Submit remaining leaves
    // }
    ```
  * **Considerations:**
      * **Load Balancing:** If `selectLeafNode` time varies significantly (e.g., due to tree depth or TT hits/misses), `schedule(dynamic)` or `schedule(guided)` might be better than the implicit static schedule, but could add overhead. The current code doesn't specify a schedule, so it's likely default (implementation-defined, often static).
      * **`active_simulations_` Management:** This atomic variable is decremented by each thread. `compare_exchange_weak` is used to claim chunks of simulations. This is a reasonable approach.
      * **NUMA Effects:** For multi-socket systems, be mindful of memory access patterns. Node pool helps, but game states themselves might be allocated across NUMA nodes. (Advanced optimization).

\</parallelization\_improvements\>

\<gpu\_throughput\_scenario\>
**Scenario: Maximizing GPU Throughput for Gomoku (19x19 board)**

**Objective:** Ensure the GPU is consistently fed large batches of game states for evaluation, minimizing idle time.

**Assumptions:**

  * Gomoku state representation is relatively small. Tensorization is efficient.
  * Target `settings_.num_threads` (MCTS threads) is, e.g., 16-32.
  * Target GPU can handle batches of 256-1024+ efficiently.
  * `MCTSEngine` is used with external queues for `MCTSEvaluator`.

**Steps & Code Snippets:**

1.  **Aggressive Leaf Generation (`MCTSEngine`):**

      * Ensure `MCTSEngine`'s OpenMP threads are efficiently generating leaves. This means `selectLeafNode` must be fast.
      * The number of simulations (`settings_.num_simulations`) per search call should be high enough (e.g., 1600+).
      * **No specific code change here beyond previous CPU optimizations.** The focus is on *rate*.

2.  **Optimized `PendingEvaluation` Enqueueing:**

      * Modify `PendingEvaluation` to use `std::unique_ptr<core::IGameState> state;`.
      * `MCTSEngine::traverseTree` clones state once into this `unique_ptr`.
        ```cpp
        // In MCTSEngine::traverseTree (or equivalent leaf generation point)
        // ... leaf found ...
        if (leaf->tryMarkForEvaluation()) {
            // const core::IGameState& leaf_state = leaf->getState(); // No longer needed if leaf->getState() is efficient
            std::unique_ptr<core::IGameState> state_for_eval;
            try {
                state_for_eval = leaf->getStateMutable().clone(); // Assuming clone creates unique_ptr
            } catch (const std::exception& e) {
                // log error
                leaf->clearEvaluationFlag();
                return; // Or handle error appropriately
            }

            if (!state_for_eval) {
                // log error "Failed to clone state for evaluation"
                leaf->clearEvaluationFlag();
                return;
            }

            PendingEvaluation pending;
            pending.node = leaf;
            pending.path = std::move(path); // path is std::vector<std::shared_ptr<MCTSNode>>
            pending.state = std::move(state_for_eval);
            // ... set batch_id, request_id ...

            if (leaf_queue_.enqueue(std::move(pending))) { // Assuming leaf_queue_ is the external queue
                pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                if (evaluator_) { evaluator_->notifyLeafAvailable(); }
            } else {
                leaf->clearEvaluationFlag(); // Failed to enqueue
            }
        }
        ```

3.  **Enhanced Batch Collection in `MCTSEvaluator::processBatch` (External Queue Path):**

      * **Goal:** Wait longer to form larger batches, but not so long that MCTS stalls.
      * **Parameters (tune these):**
          * `TARGET_BATCH_SIZE = settings_.batch_size` (e.g., 512)
          * `MINIMUM_PROCESSABLE_BATCH = std::max(1, settings_.batch_size / 8)` (e.g., 64)
          * `MAX_WAIT_MS = 20` (milliseconds) - Max time to wait to fill a batch.
          * `SHORT_WAIT_MS = 2` (milliseconds) - Short polling interval.

    <!-- end list -->

    ```cpp
    // In MCTSEvaluator::processBatch (external queue variant)
    // ...
    // std::vector<PendingEvaluation> evaluations; // Holds unique_ptr<IGameState>
    // evaluations.reserve(TARGET_BATCH_SIZE);
    // auto* external_leaf_queue = static_cast<moodycamel::ConcurrentQueue<PendingEvaluation>*>(leaf_queue_ptr_);

    // auto batch_collection_start_time = std::chrono::steady_clock::now();

    // // 1. Initial fast bulk dequeue
    // size_t initial_dequeued = external_leaf_queue->try_dequeue_bulk(
    //     std::back_inserter(evaluations), // Use back_inserter if evaluations is empty
    //     TARGET_BATCH_SIZE);

    // // 2. Conditional wait loop if not full and not enough for min processing
    // while (evaluations.size() < TARGET_BATCH_SIZE &&
    //        (evaluations.size() < MINIMUM_PROCESSABLE_BATCH ||
    //         std::chrono::duration_cast<std::chrono::milliseconds>(
    //             std::chrono::steady_clock::now() - batch_collection_start_time) < std::chrono::milliseconds(MAX_WAIT_MS)
    //        ) &&
    //        !shutdown_flag_.load(std::memory_order_acquire)) {

    //     PendingEvaluation temp_eval;
    //     if (external_leaf_queue->try_dequeue(temp_eval)) {
    //         evaluations.push_back(std::move(temp_eval));
    //     } else {
    //         // Queue is empty, wait briefly
    //         std::this_thread::sleep_for(std::chrono::milliseconds(SHORT_WAIT_MS));
    //         // Break if minimum is met and some time has passed to avoid over-waiting
    //         if (evaluations.size() >= MINIMUM_PROCESSABLE_BATCH &&
    //             std::chrono::duration_cast<std::chrono::milliseconds>(
    //                 std::chrono::steady_clock::now() - batch_collection_start_time).count() > MAX_WAIT_MS / 2) {
    //             break;
    //         }
    //     }
    // }

    // if (evaluations.empty() || evaluations.size() < MINIMUM_PROCESSABLE_BATCH_UNLESS_TIMEOUT_OR_SHUTDOWN) {
    //     // If still too few after waiting (unless it's a final batch during shutdown or timeout forces it)
    //     // Option 1: Put them back (if not shutting down and MINIMUM_PROCESSABLE_BATCH is strict)
    //     // Option 2: Process if any (if MINIMUM_PROCESSABLE_BATCH is a soft guide and latency is critical)
    //     // For this scenario, let's assume we process if evaluations.size() >= 1 after the wait.
    //     if (evaluations.empty()) return false;
    // }

    // // ... existing logic to extract states (now already unique_ptr) and call inference_fn_ ...
    // std::vector<std::unique_ptr<core::IGameState>> states_for_nn;
    // states_for_nn.reserve(evaluations.size());
    // std::vector<PendingEvaluation> original_evals_for_results; // To pair with NN outputs
    // original_evals_for_results.reserve(evaluations.size());

    // for (auto& eval_req : evaluations) {
    //     if (eval_req.state) { // Ensure state is valid
    //         states_for_nn.push_back(std::move(eval_req.state));
    //         // Move other necessary parts of eval_req (node, path) to original_evals_for_results
    //         // This example simplifies; you'd actually emplace_back a struct or pair.
    //         PendingEvaluation result_context;
    //         result_context.node = eval_req.node;
    //         result_context.path = std::move(eval_req.path);
    //         // any other IDs
    //         original_evals_for_results.push_back(std::move(result_context));
    //     }
    // }
    // evaluations.clear(); // Original evaluations vector is now drained

    // if (states_for_nn.empty()) return false;

    // std::vector<NetworkOutput> results = inference_fn_(states_for_nn);

    // // Enqueue results to external_result_queue_, pairing NetworkOutput with original_evals_for_results
    // // ...
    ```

4.  **Parallel Result Processing (`MCTSEngine`):**

      * As described in `<optimization_scheme>` Step 2, use OpenMP in `resultDistributorWorker`.

5.  **Monitoring and Feedback:**

      * Continuously log:
          * Actual batch sizes sent to GPU.
          * `leaf_queue_` size over time.
          * `result_queue_` size over time.
          * Time spent in each MCTS phase (selection, expansion, queuing, backpropagation).
          * GPU utilization.
          * MCTS nodes per second.
      * This data will guide further tuning of `MAX_WAIT_MS`, `MINIMUM_PROCESSABLE_BATCH`, MCTS thread count, and `settings_.batch_size`. If leaf generation is still too slow, CPU-side optimizations (TT, node operations, selection logic) become more critical. If leaf queue grows unbounded, the evaluator or GPU is the bottleneck. If result queue grows, the `resultDistributorWorker` is the bottleneck.

This scenario aims to create a smoother, higher-volume flow of states to the GPU by reducing cloning overhead, allowing more time for batches to fill, and parallelizing result processing to prevent backpressure. The key is balancing batch size with evaluation latency.
\</gpu\_throughput\_scenario\>

----------
----------

**1. Priority Issues & Tasks (Checklist)**

1. **Overly Serial Tree Expansion (Progressive Widening):** The current progressive widening parameters (`kpw=10`) and implementation cause extremely narrow expansion – e.g. at node visit count 2, the formula allows up to `1*2^10 = 1024` children, but the node is locked after adding only 1 child. This essentially serializes expansion (one child per node for many simulations), bottlenecking parallel threads on the same path. *Task:* Dramatically reduce `kpw` (e.g. 0.5 or 1.0) or disable progressive widening until properly implemented. Also refactor expansion so multiple children can be added incrementally (see pseudocode below).
2. **Insufficient Concurrency & Virtual Loss Contention:** Because of the above, many threads repeatedly select the same branch, adding virtual losses and often finding the leaf “already being evaluated” by another thread. Those simulations then do no useful work. This yields poor CPU utilization and wasted simulations. *Task:* After fixing (1), ensure threads can explore distinct branches – e.g. allow expansion of a new child when a node with unexplored moves has high visit count or high virtual loss on existing children. This reduces contention and lets leaf parallelism actually spread out.
3. **Small GPU Batch Sizes:** Currently the average neural-network batch is very small (often single digits), so the GPU is underutilized. The batcher starts processing after \~5–10ms or 25% of `batch_size` filled, which with only a few worker threads often yields tiny batches. *Task:* Increase effective batch sizes via higher concurrency or tuning the batch wait. For example, allow up to 20–30ms latency if the GPU isn’t saturated, or raise `num_threads` and parallel games to feed more simultaneous evaluations (see GPU scenario below). Also remove the hard cap of 10ms on batch timeout in the code to honor the configuration’s 20ms if larger batches are desired.
4. **Excessive Logging & Debug Overhead:** Debug logging is enabled by default (`MCTS_DEBUG` is 1) and prints many messages on each simulation. E.g. each search prints step-by-step validation, and the result distributor logs every batch. This **severely** slows down multithreaded execution (console I/O is a huge bottleneck). *Task:* Disable or drastically reduce logging in performance runs. Use a compile-time flag or runtime setting to toggle debug prints. Ensure `MCTS_DEBUG` is 0 in production builds.
5. **Threading Model Oversubscription & Sync:** The OpenMP worker threads plus an extra result-distributor thread can oversubscribe CPU cores (e.g. 4 OMP threads + 1 result thread on a 4-core CPU). The result thread may starve or context-switch frequently. *Task:* If using `N` CPU threads for MCTS, consider setting OMP threads to `N-1` and dedicating 1 core to the result thread (or use `std::thread` for workers to control affinity). Also consider restoring a condition-variable notify for results instead of busy-wait polling to reduce spin-wait overhead when idle.
6. **Atomic Hot-Spots in Backpropagation:** Backprop updates use atomic fetch-and-add in a loop. In the current design, backpropagation happens in a single thread (the result distributor), so these atomics are mostly uncontended – but they still incur overhead. *Task:* Simplify `MCTSNode::update()` by using relaxed ordering or non-atomic updates under the result thread’s protection (no other thread writes to `visit_count_` or `value_sum_` at that moment). At minimum, use `fetch_add` for the value sum instead of a CAS loop since only one writer exists. This will slightly boost throughput.
7. **Node Pool Utilization:** The custom `MCTSNodePool` is not fully leveraged during expansion – e.g. children are created with `MCTSNode::create()` (using `new`) rather than the pool. *Task:* Integrate the node pool in `expand()`: pass a pool reference and use it to allocate new nodes, or modify `MCTSNode::create` to use the pool. This will cut allocation overhead and improve cache locality (as noted in docs). Also ensure the game-state cloning uses the pool manager (it appears to, via `GameStatePoolManager::cloneState`).
8. **Transposition Table Thread-Safety:** If the transposition table is enabled, the current approach to replace children via `updateChildReference` is not thread-safe (comment notes rely on external sync). While this likely isn’t executed in the OpenMP version (TT entries are just inserted and looked up), it’s worth reviewing. *Task:* If TT usage is causing locks or complexity, consider disabling it in multi-thread runs or refactoring it with fine-grained locks per shard. Prioritize getting leaf parallelism efficient before re-introducing TT, since a poorly synchronized TT can serialize large parts of the search.

**2. Optimization Plan (Algorithm, Code, Config, Infrastructure)**

**a. Algorithmic Adjustments** – *Expand the search breadth and reduce contention:*

* **Progressive Widening Fix:** Use a gentler formula or none at all. For example, use `kpw = 0.5` (sqrt) or `1.0` so that new moves are introduced much sooner. Better yet, implement incremental expansion: instead of marking a node fully expanded after one call, allow multiple expansions as visits increase. For instance:

  ```cpp
  // Pseudocode for incremental child expansion during selection:
  if (use_progressive_widening) {
      int maxChildren = int(c_pw * std::pow(node->getVisitCount(), k_pw));
      maxChildren = std::max(maxChildren, 1);
      if (node->getNumExpandedChildren() < std::min(maxChildren, node->getTotalLegalMoves())) {
          std::lock_guard<std::mutex> lock(node->getExpansionMutex());
          if (node->getNumExpandedChildren() < maxChildren) {
              node->expandOneChild();  // new method: expand one additional child
          }
      }
  }
  child = node->selectChild(...); // then proceed with normal selection
  ```

  Here, `expandOneChild()` would create the next child (perhaps using stored prior probabilities to pick the move with highest policy that isn’t expanded). This ensures new branches appear gradually without a massive one-time expansion. The `expansion_mutex_` (already in `MCTSNode`) would serialize child creation, but only when a new child is allowed – far better than all threads piling on one child. This change will let multiple threads explore different children of the same node concurrently (after a few visits), greatly improving CPU utilization and tree growth.
* **Virtual Loss Tuning:** Continue to use virtual loss to discourage threads from dogpiling one node, but ensure it’s applied appropriately. The code currently adds 1 per descent; with the default `virtual_loss=3`, you could call `applyVirtualLoss(3)` once at the leaf instead of 3 separate adds – but this is minor. The main point is that once more children are expanded, virtual loss will effectively push threads to different branches instead of all sticking to one.

**b. C++ Implementation Optimizations** – *Make better use of concurrency and memory:*

* **Batching Evaluations:** Leverage the existing `moodycamel` lock-free queues to batch operations where possible. The evaluator already dequeues up to 64 states in one go and enqueues results in bulk. We should similarly batch enqueue from workers if multiple evaluations become ready at once. For example, a worker thread could accumulate local `PendingEvaluation` objects and push them via one `enqueue_bulk` when it hits a threshold or when it finishes its simulation chunk. Although each simulation typically produces only one evaluation, if root parallelism is on or threads finish simulations nearly simultaneously, a bulk push could reduce contention. This would use the existing `thread_data_` structure (currently unused). Pseudo-change:

  ```cpp
  // In traverseTree or after expansion:
  auto &batch = thread_data_[omp_get_thread_num()].local_batch;
  batch.push_back(std::move(pendingEval));
  if (batch.size() >= BATCH_SUBMISSION_INTERVAL) {
      leaf_queue_.enqueue_bulk(batch.begin(), batch.size());
      pending_evaluations_.fetch_add(batch.size(), ...);
      batch.clear();
      evaluator_->notifyLeafAvailable();
  }
  ```

  This way, instead of N threads each doing a tiny atomic enqueue, we amortize the overhead. This is an optional micro-optimization – the bigger wins come from improved parallel expansion and scheduling.
* **Reduce Atomic Overheads in Backprop:** As noted, only the result thread updates node stats, so we can safely relax some atomics. For example, `visit_count_` can be updated with `relaxed` ordering (since other threads only need eventually consistent reads), and `value_sum_` can use a simpler atomic add. Ensuring no other thread writes these fields (which holds true in the current design) means we’re not sacrificing correctness, just avoiding unnecessary memory fences. This change will slightly improve throughput during backprop. Similarly, RAVE updates can be done with relaxed order (they are ancillary stats).
* **Logging Guards:** Wrap debug outputs in conditionals or use a compiled-out macro. For instance, change `MCTS_LOG_DEBUG(...)` to only log when a verbose flag is set. The current always-on debug logging (printing to `std::cout`/`cerr`) must be disabled for real runs. This one change can turn 0% CPU idle into 100% CPU utilization for the worker threads, as printing was likely the biggest blocker. In practice, after turning off logging, you should see a significant jump in simulations per second.

**c. Configuration & Tuning** – *Adjust settings to balance throughput vs latency:*

* **Increase `num_threads` and Simulation Count:** If the hardware allows, raise `MCTSSettings.num_threads` (e.g. 8 or 16) to generate more parallel playouts. More threads → more leaf evaluations in flight → larger batches. Monitor that CPU can still keep up and that threads aren’t all contending on one lock (after the above fixes). If the tree search still isn’t saturating the GPU, you can also increase `num_simulations` to, say, 1600 so threads have more total work to do per move. This gives the batcher more opportunities to fill up.
* **Batch Size and Timeout:** The default batch size (128) is reasonable for a modern GPU, but the effective timeout was capped to 10ms in code. Depending on your latency tolerance, you can allow a bit more waiting to gather larger batches. For example, setting `batch_timeout = 20–30ms` (and removing that 10ms cap) could let late-arriving evaluations join a batch, increasing average batch size. This should be tuned: too high a timeout adds move latency without much gain if the threads aren’t producing enough concurrent requests. Start with \~20ms and measure average batch size (the engine tracks this in `last_stats_.avg_batch_size`). The goal is to approach the batch\_size (128) under heavy load; if you consistently only get e.g. 8 or 16, you have headroom to either increase threads or wait a bit longer.
* **Progressive Widening Toggle:** Make it easy to disable progressive widening via config. In games like Gomoku or small boards, the action space isn’t huge, so you might not need PW at all – expanding all moves at once is fine and ensures full policy utilization. Only very large games (Go) truly need PW. Providing a boolean to turn it off can help with A/B testing performance. If off, `expand()` would simply generate all legal children with one call (which our node pool can handle) and use network priors immediately for all of them. This could massively improve parallel exploration at the cost of more memory.
* **Thread Binding:** For consistency, consider pinning the OpenMP threads and the result thread to specific cores (using OS APIs or OMP env vars). This avoids context switches. For example, on Linux you might use `OMP_PLACES=cores` and `OMP_PROC_BIND=close` for OpenMP, and set the result thread on a different core via `pthread_setaffinity_np` (since you already name it "ResultDist"). This is an advanced tweak – not critical, but can squeeze out a bit more performance by reducing scheduler overhead.

**d. Infrastructure & Scaling** – *Beyond single-engine optimization:*

* **Parallel Games / Self-Play:** The ultimate way to drive up GPU usage is to run multiple searches in parallel. The repository already hints at a `SelfPlayManager` that creates multiple MCTSEngines and a shared evaluator thread. By running, say, 8 games concurrently (with 8 engines feeding into one inference queue), you multiply the inference load and can achieve near-max batch sizes consistently. In this setup, each engine could use fewer threads (to avoid thrashing the CPU) since aggregate concurrency comes from many games. The shared evaluator concept is sound – ensure that all engines use the **same** `moodycamel` queues for leaves/results and a single evaluator thread so the GPU sees one combined stream of states. This might involve refactoring MCTSEngine to accept external queue references (similar to `setExternalQueues` usage). The payoff is significant: with 8 parallel games, if each game on average produces, say, 4–8 evaluations in a short window, the combined batch can be 32–64, keeping the GPU far busier than batches of 4.
* **Multi-GPU Support:** If training or self-play is run on a multi-GPU server, you can scale by assigning different MCTS engines to different devices. For example, instantiate two `MCTSEvaluator` threads, each with its own CUDA device, and route half the games to each. This requires the neural network inference function to be aware of device IDs or for you to spawn separate processes/threads bound to specific GPUs. The code currently uses LibTorch CUDA globally, so to use multiple GPUs you’d either set the GPU in the `inference_fn_` (e.g. move data to a particular `torch::Device`) or run separate instances of the program per GPU. This is more of a future scaling consideration; the immediate step is to optimize single-GPU throughput first.

**3. Threading Model Review & Concurrency Fixes**
The current threading model uses OpenMP for simulation workers and a dedicated std::thread for result distribution. This generally works but had a few bottlenecks:

* **Leaf Parallelism Implementation:** The good news is that leaf parallelism *is* implemented (multiple simulations proceed without waiting for evaluations). However, due to the single-child expansion issue, threads often converged on the same leaf, leading to effectively serial behavior. By fixing that (as above), the leaf parallelism will truly allow MCTS to expand multiple different leaves concurrently – the core idea needed for high throughput. Each thread picks a simulation via the atomic counter (`active_simulations_`) and traverses the tree independently. We must ensure that when different threads hit the *same* node: (a) only one does the expansion/evaluation (handled by `tryMarkForEvaluation()` flag), and (b) others either find alternative moves or gracefully back off. The `tryMarkForEvaluation` is working (it uses an atomic flag to prevent duplicate evals), but our improvement is to give those other threads something useful to do – e.g. expand another child as shown above, or if no alternative, backtrack and try a different path (not currently implemented, but an idea).
* **Synchronization & Race Conditions:** With the current lock-free design, we need to be vigilant about a few things:

  * **Virtual Loss Accounting:** Ensure that every `addVirtualLoss()` has a matching `removeVirtualLoss()`. In the result thread’s backprop, they do remove the virtual losses for each node in the path (via `backPropagate` calling `node->removeVirtualLoss()` in a loop). This prevents net win evaluations from being permanently skewed. If a thread abandons a simulation because another thread is evaluating that leaf, that thread should ideally remove any virtual losses it added. Currently, if `tryMarkForEvaluation` fails, the code path simply ends the simulation without backpropagation – meaning it does not explicitly remove the virtual losses it added on the way down. This is a minor bug: those losses will eventually be removed when the *other* thread’s result comes back and is backpropagated, but it means one simulation didn’t increment visit counts. *Fix:* If `tryMarkForEvaluation` returns false (another thread is on it), iterate through the `path` and call `removeVirtualLoss()` for each node so that those temporary losses aren’t counted twice. This will make the simulation a no-op (which is fine) and keep tree stats consistent.
  * **Result Distribution vs Search Completion:** The engine waits up to 5 seconds for all pending evaluations after simulations finish. If a deadlock or slow inference occurs beyond this, it breaks out, but note that it doesn’t join the result thread – it just sets `workers_active_=false` and leaves the thread to exit on its own. To avoid any race on engine destruction, it would be cleaner to signal and join that thread explicitly. *Fix:* In `MCTSEngine::search()` after collecting results, set `shutdown_=true`, notify the evaluator (in case it’s waiting), and join the `result_distributor_worker_` thread. This ensures no dangling threads. The engine’s destructor should also call `safelyStopEvaluator()` and join any worker threads.
  * **Transposition Table Coordination:** As mentioned, if using the TT, guard any structure that might be accessed by multiple threads. E.g., inserting into TT can use per-shard mutexes. Also, ensure that when a node is added from TT (i.e. a duplicate state found), only one thread links it into the tree. Given the complexity and relatively lower priority of TT in early performance tuning, it’s acceptable to keep TT off until the rest is stable.

**4. GPU Throughput Scenario**
*Imagine we deploy the above optimizations and run self-play with multiple parallel games:* Suppose we run 8 MCTS engines in parallel, each with 4 worker threads, all using a single shared GPU evaluator. In a short time window (say 10ms), each engine’s workers might collectively request around \~5 evaluations on average (some engines will have more, some less, depending on where they are in the search). The centralized `leaf_queue_` could then accumulate \~40 pending states. The MCTSEvaluator thread will wait a few milliseconds for more states and likely hit the 25% batch threshold (in this case 25% of 128 is 32). It will form a batch of, e.g., 40 states and send them through the neural network in one go. The GPU, which might be underutilized running single states before, now sees a batch of 40 – this achieves much higher occupancy and throughput (the matrix-multiplication units on a GPU like to work on larger matrices). The result is that the GPU can amortize the overhead of launching a kernel over 40 inputs instead of 1, and the per-state inference time drops significantly. As we increase the search complexity (more threads or more games), we could approach batches of \~128, at which point the GPU is near fully saturated for that inference cycle. The system can then **scale** further by shortening the wait timeout (to avoid too-large batches queuing up) or by adding another GPU. For instance, with 16 parallel games, you might deliberately split the load: 8 games feeding one GPU and 8 feeding another. Each GPU’s batch size stays around 50–100, and both GPUs are utilized. The infrastructure should evolve to support this by allowing multiple evaluator threads (each with its own device and queues) and assigning engines to device “shards.”

In summary, the plan is to **first unlock parallelism** (fix tree expansion and thread contention), **then tune batching** (threads, timeout, parallel games) to push the GPU to higher throughput, all while eliminating unnecessary overhead in the C++ code. By tackling the above issues in order of priority, you’ll move from an MCTS that was effectively doing sequential expansions with tiny GPU bursts to a truly scalable MCTS engine that keeps all CPU cores and the GPU busy. The end result should be a dramatic increase in simulations/sec and much larger average batch sizes (e.g. 1–4 → 32–128), leading to better hardware utilization and faster searches. The code-level changes – particularly around progressive widening, locking, and batching – will directly address the core inefficiencies, setting the stage for scaling up to chess, Go, and beyond with strong performance.

----------
----------

# Turbocharged MCTS: Unlocking performance in parallel tree search

Monte Carlo Tree Search (MCTS) is a powerful algorithm for decision-making in complex domains like board games, but achieving high performance requires careful optimization. Based on my analysis of the MCTS implementation in the src/mcts code, I've identified several key bottlenecks and optimization opportunities that can dramatically improve performance, particularly for Gomoku, Chess, and Go.

## The core bottlenecks: why your GPU is idle

The primary issues causing low CPU/GPU usage, poor batch sizes, and low MCTS speed are:

1. **Inefficient leaf parallelization** preventing optimal batch formation
2. **Synchronization contention** in the tree traversal phase
3. **Suboptimal virtual loss implementation** causing thread collisions
4. **GPU starvation** from inconsistent batch creation
5. **Transposition table conflicts** leading to cache inefficiency

Let's dive into each component and explore concrete solutions.

## Leaf selection and evaluation pipeline

### Current implementation issues

The flow from leaf selection to evaluation to result distribution suffers from several inefficiencies:

- **Sequential tree traversal** creates a bottleneck before GPU evaluation
- **Inconsistent batch formation** leads to underutilized GPU compute
- **Thread contention** during node selection causes CPU stalls
- **Poor workload distribution** between selection and evaluation phases

When multiple threads independently traverse the tree to select leaf nodes, they often collide on popular paths, particularly in games like Gomoku with strategic hot spots. This creates contention and prevents efficient batch formation.

### Optimization strategy: Hybrid selection approach

```cpp
// Implement a two-phase selection process
void selectLeavesForBatch(Node* root, int batchSize, std::vector<Node*>& leaves) {
    ThreadPool pool(num_threads);
    std::mutex leavesMutex;
    std::atomic<int> leafCount{0};
    
    // Phase 1: Multiple threads independently select first ~40% of batch
    for (int i = 0; i < num_threads; i++) {
        pool.enqueue([&, i]() {
            int threadTargetLeaves = batchSize * 0.4 / num_threads;
            std::vector<Node*> threadLeaves;
            
            while (threadLeaves.size() < threadTargetLeaves && leafCount.load() < batchSize * 0.4) {
                Node* leaf = selectLeafWithVirtualLoss(root);
                if (leaf) {
                    threadLeaves.push_back(leaf);
                    leafCount++;
                }
            }
            
            // Add thread-local leaves to global batch
            {
                std::lock_guard<std::mutex> lock(leavesMutex);
                leaves.insert(leaves.end(), threadLeaves.begin(), threadLeaves.end());
            }
        });
    }
    pool.wait();
    
    // Phase 2: Use coordinated selection for remaining ~60% to maximize diversity
    coordinatedLeafSelection(root, batchSize - leaves.size(), leaves);
}

// Coordinated selection to maximize tree coverage
void coordinatedLeafSelection(Node* root, int targetCount, std::vector<Node*>& leaves) {
    std::queue<Node*> expansionQueue;
    expansionQueue.push(root);
    
    while (!expansionQueue.empty() && leaves.size() < targetCount) {
        Node* current = expansionQueue.front();
        expansionQueue.pop();
        
        // If node is terminal or has no unexpanded children, skip
        if (current->isTerminal() || current->isFullyExpanded()) {
            continue;
        }
        
        // Select unexpanded child or perform rollout
        if (auto leaf = current->selectUnexpandedChild()) {
            leaves.push_back(leaf);
        } else {
            // Queue all promising children for BFS-style expansion
            for (auto& child : current->getChildren()) {
                if (child->getUCBScore() > THRESHOLD) {
                    expansionQueue.push(child);
                }
            }
        }
    }
}
```

This hybrid approach maximizes batch diversity while reducing contention. The first phase allows threads to independently select leaves, while the second phase ensures we get a complete batch with good tree coverage.

## Thread synchronization and contention points

### Current implementation issues

The implementation suffers from several synchronization bottlenecks:

- **Coarse-grained locking** during tree traversal
- **Lock contention** on frequently visited nodes near the root
- **Busy-waiting** while forming evaluation batches
- **Sequential backpropagation** limiting throughput

These issues are particularly problematic in games like Go with large branching factors, where multiple threads end up competing for access to the same popular nodes.

### Optimization strategy: Lock-free traversal with path-specific locking

```cpp
// Lock-free tree traversal with path-specific locking only for updates
Node* selectLeafNode(Node* root) {
    std::vector<Node*> path;
    Node* current = root;
    
    // Selection phase - no locks during traversal
    while (!current->isLeaf()) {
        path.push_back(current);
        
        // Apply virtual loss atomically without locks
        current->applyVirtualLossAtomic();
        
        // Select best child using lock-free reads
        current = current->selectBestChildLockFree();
    }
    
    // Lock only the selected leaf for expansion
    if (!current->isTerminal() && current->needsExpansion()) {
        std::lock_guard<std::mutex> lock(current->mutex);
        // Check if another thread expanded it while we were acquiring the lock
        if (current->needsExpansion()) {
            current->expand();
        }
    }
    
    return current;
}

// Lock-free backpropagation with batched updates
void backpropagate(std::vector<Node*>& path, float result) {
    // Update nodes bottom-up
    for (int i = path.size() - 1; i >= 0; i--) {
        Node* node = path[i];
        
        // Use atomic operations for visit count
        node->visits.fetch_add(1, std::memory_order_relaxed);
        
        // Remove virtual loss
        node->removeVirtualLossAtomic();
        
        // Use a separate value mutex for updating aggregated values
        // Can be optimized further with CAS operations or local aggregation
        {
            std::lock_guard<std::mutex> lock(node->valueMutex);
            node->value += (result - node->value) / node->visits.load();
        }
    }
}
```

This approach minimizes synchronization overhead by:
1. Using atomic operations for most counters
2. Eliminating locks during tree traversal
3. Using fine-grained locks only when necessary
4. Using a combination of atomic visit counts and protected value updates

## Virtual loss implementation

### Current implementation issues

The virtual loss mechanism shows several weaknesses:

- **Insufficient divergence** leading to thread collisions
- **Inconsistent application/removal** causing statistical bias
- **Static virtual loss values** not adapting to game state
- **Missing thread collision detection** resulting in redundant work

### Optimization strategy: Enhanced adaptive virtual loss

```cpp
// Enhanced adaptive virtual loss
class AdaptiveVirtualLoss {
private:
    std::atomic<int> collisionCount{0};
    float baseVirtualLoss = 1.0f;
    float adaptiveFactor = 1.0f;
    
public:
    // Apply dynamic virtual loss based on collision statistics
    void applyTo(Node* node) {
        float dynamicLoss = baseVirtualLoss * adaptiveFactor;
        
        // Check if other threads are already exploring this node
        if (node->virtualLossCount.fetch_add(1, std::memory_order_relaxed) > 0) {
            // Collision detected, increase for next time
            collisionCount.fetch_add(1, std::memory_order_relaxed);
            if (collisionCount.load() % 10 == 0) {
                // Increase virtual loss when consistent collisions occur
                adaptiveFactor = std::min(adaptiveFactor * 1.2f, 5.0f);
            }
            
            // Apply progressively stronger virtual loss at deeper nodes
            dynamicLoss *= (1.0f + 0.2f * node->depth);
        }
        
        // Atomically apply the virtual loss
        node->virtualLoss.fetch_add(dynamicLoss, std::memory_order_relaxed);
    }
    
    // Remove virtual loss and return collision status
    bool removeFrom(Node* node) {
        bool wasCollision = node->virtualLossCount.load() > 1;
        node->virtualLossCount.fetch_sub(1, std::memory_order_relaxed);
        
        // Remove the correct amount of virtual loss
        float dynamicLoss = baseVirtualLoss * adaptiveFactor;
        if (wasCollision) {
            dynamicLoss *= (1.0f + 0.2f * node->depth);
        }
        
        node->virtualLoss.fetch_sub(dynamicLoss, std::memory_order_relaxed);
        
        // Occasionally decrease adaptive factor when collisions are rare
        if (collisionCount.load() % 20 == 0 && !wasCollision) {
            adaptiveFactor = std::max(adaptiveFactor * 0.9f, 1.0f);
        }
        
        return wasCollision;
    }
};
```

This implementation:
1. Dynamically adjusts virtual loss magnitude based on collision frequency
2. Applies stronger virtual loss to deeper nodes where exploration investment is higher
3. Tracks collision statistics to fine-tune the divergence factor
4. Uses atomic operations for thread safety without locks

## Neural network evaluation batching

### Current implementation issues

The neural network evaluation batching mechanism shows several weaknesses:

- **Inconsistent batch sizes** causing GPU underutilization
- **Synchronous evaluation** blocking CPU threads
- **Long wait times** for batch completion
- **Unbalanced workload** between CPU and GPU
- **Poor memory transfer patterns** between CPU and GPU

These issues are particularly evident in Go and Chess, where evaluation complexity is higher and batch size consistency matters more.

### Optimization strategy: Predictive dual-buffer batching

```cpp
// Predictive dual-buffer batching system
class NeuralNetworkBatcher {
private:
    // Dual buffer system to overlap evaluation and batch formation
    struct BatchBuffer {
        std::vector<GameState> states;
        std::vector<std::promise<NetworkOutput>> promises;
        std::atomic<bool> ready{false};
        std::atomic<bool> processing{false};
    };
    
    BatchBuffer buffers[2];
    std::atomic<int> activeBuffer{0};
    std::atomic<int> pendingStates{0};
    int targetBatchSize;
    float dynamicTimeoutMs;
    std::thread workerThread;
    std::atomic<bool> running{true};
    
    // Adaptive timing statistics
    float avgEvalTimeMs = 10.0f;
    float avgBatchFormTimeMs = 5.0f;
    std::mutex timingMutex;
    
public:
    NeuralNetworkBatcher(int targetSize) : targetBatchSize(targetSize), dynamicTimeoutMs(5.0f) {
        // Initialize buffers
        buffers[0].states.reserve(targetBatchSize);
        buffers[1].states.reserve(targetBatchSize);
        
        // Start worker thread
        workerThread = std::thread([this]() { processBatches(); });
    }
    
    // Submit a state for evaluation
    std::future<NetworkOutput> submitEvaluation(const GameState& state) {
        std::promise<NetworkOutput> promise;
        std::future<NetworkOutput> future = promise.get_future();
        
        int currentBuffer = activeBuffer.load();
        
        {
            std::lock_guard<std::mutex> lock(buffers[currentBuffer].mutex);
            buffers[currentBuffer].states.push_back(state);
            buffers[currentBuffer].promises.push_back(std::move(promise));
        }
        
        int pending = pendingStates.fetch_add(1) + 1;
        
        // If we've reached target batch size, signal ready immediately
        if (pending >= targetBatchSize) {
            triggerBatchProcessing();
        }
        
        return future;
    }
    
private:
    // Worker thread that processes batches
    void processBatches() {
        while (running.load()) {
            // Check if current batch is ready or timeout has occurred
            bool shouldProcess = false;
            int current = activeBuffer.load();
            
            {
                std::lock_guard<std::mutex> lock(buffers[current].mutex);
                int pending = pendingStates.load();
                
                // Process if we have a full batch or timeout occurred with sufficient states
                if (pending >= targetBatchSize || 
                    (pending > targetBatchSize / 4 && 
                     checkDynamicTimeout())) {
                    shouldProcess = true;
                }
            }
            
            if (shouldProcess) {
                processBatch();
            } else {
                // Sleep briefly to avoid busy waiting
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    
    // Process the current batch
    void processBatch() {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Swap buffers
        int current = activeBuffer.exchange(1 - activeBuffer.load());
        pendingStates.store(0);
        
        // Mark as processing
        buffers[current].processing.store(true);
        
        // Get local copies of the batch data
        std::vector<GameState> states;
        std::vector<std::promise<NetworkOutput>> promises;
        
        {
            std::lock_guard<std::mutex> lock(buffers[current].mutex);
            states = std::move(buffers[current].states);
            promises = std::move(buffers[current].promises);
            buffers[current].states.reserve(targetBatchSize);
            buffers[current].promises.reserve(targetBatchSize);
        }
        
        // Perform the actual neural network evaluation
        auto evalStartTime = std::chrono::high_resolution_clock::now();
        std::vector<NetworkOutput> results = evaluateOnGPU(states);
        auto evalEndTime = std::chrono::high_resolution_clock::now();
        
        // Update timing statistics
        {
            std::lock_guard<std::mutex> lock(timingMutex);
            auto batchFormTime = std::chrono::duration<float, std::milli>(
                evalStartTime - startTime).count();
            auto evalTime = std::chrono::duration<float, std::milli>(
                evalEndTime - evalStartTime).count();
            
            // Exponential moving average
            avgBatchFormTimeMs = avgBatchFormTimeMs * 0.95f + batchFormTime * 0.05f;
            avgEvalTimeMs = avgEvalTimeMs * 0.95f + evalTime * 0.05f;
            
            // Adjust dynamic timeout based on timing statistics
            updateDynamicTimeout();
        }
        
        // Fulfill promises with results
        for (size_t i = 0; i < promises.size(); i++) {
            promises[i].set_value(results[i]);
        }
        
        // Mark as no longer processing
        buffers[current].processing.store(false);
    }
    
    // Check if dynamic timeout has occurred
    bool checkDynamicTimeout() {
        static auto lastBatchTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        
        float elapsedMs = std::chrono::duration<float, std::milli>(
            now - lastBatchTime).count();
        
        return elapsedMs > dynamicTimeoutMs;
    }
    
    // Update dynamic timeout based on timing statistics
    void updateDynamicTimeout() {
        // Set timeout to be slightly longer than GPU eval time
        // but not too long to avoid waiting unnecessarily
        dynamicTimeoutMs = avgEvalTimeMs * 0.8f;
        
        // Cap minimum and maximum values
        dynamicTimeoutMs = std::min(std::max(dynamicTimeoutMs, 1.0f), 50.0f);
    }
};
```

This predictive dual-buffer implementation:
1. Uses two separate buffers to overlap batch formation and evaluation
2. Dynamically adjusts timeout based on actual evaluation timing
3. Maintains consistent batch sizes to maximize GPU utilization
4. Adapts to changing conditions during gameplay
5. Predicts optimal batch timing based on historical performance

## Transposition table implementation

### Current implementation issues

The transposition table shows several limitations:

- **Hash collisions** leading to incorrect state evaluations
- **Lock contention** during concurrent access
- **Poor cache locality** causing CPU cache misses
- **Inefficient replacement policies** leading to premature eviction
- **Memory fragmentation** impacting overall performance

### Optimization strategy: High-performance concurrent transposition table

```cpp
// Cache-efficient, lock-free transposition table
template <typename StateInfo>
class ConcurrentTranspositionTable {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Ensure each entry fits in a single cache line
    struct alignas(CACHE_LINE_SIZE) TableEntry {
        std::atomic<uint64_t> key{0};
        std::atomic<uint32_t> lock{0};
        StateInfo info;
        uint16_t depth{0};
        uint8_t generation{0};
        uint8_t bound{0};
    };
    
    std::vector<TableEntry> entries;
    size_t sizeMask;
    std::atomic<uint8_t> currentGeneration{1};
    
public:
    ConcurrentTranspositionTable(size_t sizeInMB) {
        // Calculate number of entries that fit in the given size
        size_t numEntries = (sizeInMB * 1024 * 1024) / sizeof(TableEntry);
        
        // Round down to power of 2 for efficient indexing
        size_t powerOf2 = 1;
        while (powerOf2 * 2 <= numEntries) powerOf2 *= 2;
        entries.resize(powerOf2);
        sizeMask = powerOf2 - 1;
    }
    
    // Find an entry without locking
    bool probe(uint64_t key, StateInfo& result) {
        size_t index = key & sizeMask;
        size_t probeCount = 0;
        
        // Linear probing with age-based replacement
        while (probeCount < 4) {  // Limit probe depth
            TableEntry& entry = entries[index];
            uint64_t storedKey = entry.key.load(std::memory_order_relaxed);
            
            if (storedKey == key) {
                // Entry found, try to acquire read lock
                while (true) {
                    uint32_t lockValue = entry.lock.load(std::memory_order_relaxed);
                    if (lockValue > 0) {
                        // Entry is being written, skip
                        std::this_thread::yield();
                        continue;
                    }
                    
                    // Copy the data quickly while not locked
                    result = entry.info;
                    
                    // Verify key hasn't changed
                    if (entry.key.load(std::memory_order_acquire) == key) {
                        return true;
                    }
                    
                    // Key changed during read, retry
                    break;
                }
            }
            
            if (storedKey == 0) {
                // Empty slot, entry doesn't exist
                return false;
            }
            
            // Collision, try next slot
            index = (index + 1) & sizeMask;
            probeCount++;
        }
        
        return false;
    }
    
    // Store an entry with optimistic locking
    void store(uint64_t key, const StateInfo& info, uint8_t depth, uint8_t bound) {
        size_t index = key & sizeMask;
        size_t bestIndex = index;
        int bestScore = -1;
        
        // Find best slot for replacement in a small window
        for (size_t i = 0; i < 4; i++) {
            size_t idx = (index + i) & sizeMask;
            TableEntry& entry = entries[idx];
            
            // Score this slot for replacement
            int score = scoreForReplacement(entry, key, depth);
            if (score > bestScore) {
                bestScore = score;
                bestIndex = idx;
            }
            
            // Exact match for key is best
            if (entry.key.load(std::memory_order_relaxed) == key) {
                bestIndex = idx;
                break;
            }
        }
        
        // Try to acquire the lock on the chosen slot
        TableEntry& entry = entries[bestIndex];
        uint32_t expected = 0;
        
        if (!entry.lock.compare_exchange_strong(expected, 1, 
                                              std::memory_order_acquire)) {
            // Another thread is updating this entry, skip
            return;
        }
        
        // We have the lock, update the entry
        entry.key.store(key, std::memory_order_relaxed);
        entry.info = info;
        entry.depth = depth;
        entry.bound = bound;
        entry.generation = currentGeneration.load(std::memory_order_relaxed);
        
        // Release the lock
        entry.lock.store(0, std::memory_order_release);
    }
    
    // Age the table by incrementing generation
    void incrementAge() {
        uint8_t current = currentGeneration.load(std::memory_order_relaxed);
        currentGeneration.store((current + 1) % 256, std::memory_order_relaxed);
    }
    
private:
    // Score a slot for replacement
    int scoreForReplacement(const TableEntry& entry, uint64_t key, uint8_t depth) {
        uint64_t entryKey = entry.key.load(std::memory_order_relaxed);
        
        // Empty slot is perfect
        if (entryKey == 0) return 100;
        
        // Same key is very good
        if (entryKey == key) return 90;
        
        // Prefer replacing old entries
        uint8_t gen = entry.generation;
        uint8_t currentGen = currentGeneration.load(std::memory_order_relaxed);
        int ageDelta = (currentGen + 256 - gen) % 256;
        
        // Prefer replacing shallow entries with deeper searches
        int depthBonus = (depth > entry.depth) ? 20 : 0;
        
        return depthBonus + std::min(ageDelta * 2, 50);
    }
};
```

This implementation:
1. Aligns entries to cache lines to minimize false sharing
2. Uses optimistic concurrency control to reduce locking overhead
3. Implements a smart age-based replacement policy
4. Limits probe depth to maintain consistent performance
5. Uses atomic operations for thread safety without global locks

## Root parallelization implementation

### Current implementation issues

The root parallelization approach has several shortcomings:

- **Redundant exploration** of the same lines across threads
- **Inefficient result aggregation** when combining tree statistics
- **Inconsistent search depth** across parallel trees
- **Poor work distribution** leading to some threads doing redundant work

These issues particularly impact Chess, where tactical precision is important.

### Optimization strategy: Tree sharing with speculative split points

```cpp
// Improved root parallelization with tree sharing
class SpeculativeRootParallelization {
private:
    Node* sharedRoot;
    int numThreads;
    std::vector<Node*> splitPoints;
    std::atomic<int> completedThreads{0};
    
public:
    SpeculativeRootParallelization(Node* root, int threads) 
        : sharedRoot(root), numThreads(threads) {}
    
    void runSearch(int totalSimulations) {
        // First phase: Build initial tree with a single thread
        int warmupSimulations = totalSimulations / 20; // 5% for warmup
        runSimulations(sharedRoot, warmupSimulations);
        
        // Identify promising split points for workers
        identifySplitPoints();
        
        // Second phase: Distribute remaining simulations across threads
        int simulationsPerThread = (totalSimulations - warmupSimulations) / numThreads;
        
        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; i++) {
            threads.push_back(std::thread([this, i, simulationsPerThread]() {
                // Each thread searches from its assigned split point
                Node* startPoint = (i < splitPoints.size()) ? 
                    splitPoints[i] : sharedRoot;
                
                // If split point is terminal or already fully expanded,
                // fall back to searching from root
                if (startPoint->isTerminal() || startPoint->isFullyExpanded()) {
                    startPoint = sharedRoot;
                }
                
                runSimulations(startPoint, simulationsPerThread);
                completedThreads.fetch_add(1);
            }));
        }
        
        // Wait for threads to complete
        for (auto& t : threads) {
            t.join();
        }
    }
    
private:
    // Identify diverse high-potential split points
    void identifySplitPoints() {
        splitPoints.clear();
        std::vector<Node*> candidates;
        std::queue<Node*> queue;
        queue.push(sharedRoot);
        
        // Breadth-first search to find candidate split points
        while (!queue.empty() && candidates.size() < numThreads * 3) {
            Node* current = queue.front();
            queue.pop();
            
            // Skip terminal nodes
            if (current->isTerminal()) continue;
            
            // High-quality nodes with enough visits are good candidates
            if (current != sharedRoot && 
                current->visits > 50 && 
                current->value > 0.4) {
                candidates.push_back(current);
            }
            
            // Add children to queue
            for (Node* child : current->getChildren()) {
                queue.push(child);
            }
        }
        
        // Select diverse split points using a greedy approach
        while (splitPoints.size() < numThreads && !candidates.empty()) {
            // Find candidate most different from existing split points
            auto bestIt = candidates.begin();
            float bestDiversity = 0;
            
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                float diversity = calculateDiversity(*it, splitPoints);
                if (diversity > bestDiversity) {
                    bestDiversity = diversity;
                    bestIt = it;
                }
            }
            
            // Add the most diverse candidate to split points
            splitPoints.push_back(*bestIt);
            candidates.erase(bestIt);
        }
    }
    
    // Calculate how different a node is from existing split points
    float calculateDiversity(Node* node, const std::vector<Node*>& existingPoints) {
        if (existingPoints.empty()) return 1.0f;
        
        float minSimilarity = 1.0f;
        for (Node* existing : existingPoints) {
            float similarity = calculateSimilarity(node, existing);
            minSimilarity = std::min(minSimilarity, similarity);
        }
        
        return 1.0f - minSimilarity;
    }
    
    // Calculate similarity between two nodes based on shared ancestry
    float calculateSimilarity(Node* a, Node* b) {
        // Trace paths from root to each node
        std::vector<Node*> pathA = getPathFromRoot(a);
        std::vector<Node*> pathB = getPathFromRoot(b);
        
        // Count common ancestors
        int commonPrefix = 0;
        while (commonPrefix < pathA.size() && 
               commonPrefix < pathB.size() && 
               pathA[commonPrefix] == pathB[commonPrefix]) {
            commonPrefix++;
        }
        
        // Similarity is ratio of common path to average path length
        return static_cast<float>(commonPrefix) / 
               ((pathA.size() + pathB.size()) / 2.0f);
    }
    
    // Get path from root to node
    std::vector<Node*> getPathFromRoot(Node* node) {
        std::vector<Node*> path;
        while (node != nullptr) {
            path.insert(path.begin(), node);
            node = node->parent;
        }
        return path;
    }
};
```

This approach:
1. Uses a warm-up phase to build an initial shared tree
2. Identifies diverse high-quality split points for workers
3. Distributes work to maximize exploration diversity
4. Shares the tree structure to eliminate redundant computation
5. Adapts dynamically based on the developing search tree

## RAVE implementation

### Current implementation issues

The RAVE (Rapid Action Value Estimation) implementation shows inefficiencies:

- **Excessive memory usage** for storing RAVE statistics
- **Redundant updates** during backpropagation
- **Static RAVE weighting** that doesn't adapt to game phase
- **Inefficient statistical aggregation** during updates

These issues are particularly important for Go, where RAVE significantly improves performance.

### Optimization strategy: Memory-efficient adaptive RAVE

```cpp
// Memory-efficient adaptive RAVE implementation
class AdaptiveRAVE {
private:
    // Compact storage for RAVE values
    struct RAVEStats {
        // Use 16-bit integers to save memory
        uint16_t visits = 0;
        int16_t value = 0;  // Fixed-point representation: value/10000
        
        // Convert to/from floating point
        void setValue(float v) {
            value = static_cast<int16_t>(v * 10000);
        }
        
        float getValue() const {
            return value / 10000.0f;
        }
    };
    
    // Mapping from action to RAVE statistics
    std::unordered_map<Action, RAVEStats, ActionHash> raveMap;
    
    // Game phase detection for adaptive weighting
    int movesPlayed;
    int estimatedGameLength;
    float adaptiveK;
    
public:
    AdaptiveRAVE(int gameLength) 
        : movesPlayed(0), estimatedGameLength(gameLength) {
        // Initialize adaptive RAVE parameter
        updateAdaptiveK();
    }
    
    // Update adaptive RAVE parameter based on game phase
    void updateAdaptiveK() {
        float gameProgress = static_cast<float>(movesPlayed) / estimatedGameLength;
        
        // RAVE is more useful early in the game, less useful later
        if (gameProgress < 0.3f) {
            adaptiveK = 1000.0f;  // Strong RAVE influence early
        } else if (gameProgress < 0.7f) {
            adaptiveK = 100.0f;   // Moderate influence mid-game
        } else {
            adaptiveK = 10.0f;    // Weak influence late game
        }
    }
    
    // Calculate UCT score with RAVE
    float calculateScore(const Action& action, float uctScore, 
                        int nodeVisits, int actionVisits) {
        auto it = raveMap.find(action);
        if (it == raveMap.end() || it->second.visits == 0) {
            return uctScore;  // No RAVE data, use UCT
        }
        
        // Get RAVE statistics
        int raveVisits = it->second.visits;
        float raveValue = it->second.getValue();
        
        // Calculate RAVE weight (β)
        float beta = raveVisits / (raveVisits + actionVisits + adaptiveK * 
                                  actionVisits * raveVisits);
        
        // Weighted combination of UCT and RAVE
        return (1.0f - beta) * uctScore + beta * raveValue;
    }
    
    // Update RAVE statistics during backpropagation
    void update(const std::vector<Action>& actionsInSimulation, float result) {
        // Update for all actions in the simulation
        for (const Action& action : actionsInSimulation) {
            auto it = raveMap.find(action);
            if (it == raveMap.end()) {
                // Create new entry
                RAVEStats stats;
                stats.visits = 1;
                stats.setValue(result);
                raveMap[action] = stats;
            } else {
                // Update existing entry with incremental average
                RAVEStats& stats = it->second;
                int newVisits = stats.visits + 1;
                float oldValue = stats.getValue();
                float newValue = oldValue + (result - oldValue) / newVisits;
                
                stats.visits = std::min(newVisits, 65535);  // Cap to prevent overflow
                stats.setValue(newValue);
            }
        }
    }
    
    // Record game progression
    void notifyMovePlayed() {
        movesPlayed++;
        updateAdaptiveK();
    }
    
    // Prune RAVE table to save memory
    void prune() {
        // Remove entries with few visits or old entries
        auto it = raveMap.begin();
        while (it != raveMap.end()) {
            if (it->second.visits < 5) {
                it = raveMap.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

This implementation:
1. Uses compact representation for RAVE statistics to reduce memory usage
2. Adapts RAVE influence based on game progress
3. Efficiently updates statistics with incremental averaging
4. Prunes the RAVE table to maintain memory efficiency
5. Handles overflow conditions gracefully

## Progressive widening implementation

### Current implementation issues

The progressive widening implementation shows limitations:

- **Fixed widening parameters** not adapting to game characteristics
- **Inefficient action selection** during expansion
- **Unnecessary expansion** of low-potential nodes
- **Suboptimal balance** between exploration and exploitation

These issues impact games with large branching factors like Go and Gomoku.

### Optimization strategy: Adaptive progressive widening with policy guidance

```cpp
// Adaptive progressive widening with policy guidance
class AdaptiveProgressiveWidening {
private:
    float baseK;
    float baseAlpha;
    
    // Dynamic adjustment factors
    float kMultiplier = 1.0f;
    float alphaMultiplier = 1.0f;
    
    // Game state tracking for adaptation
    int currentPly = 0;
    int estimatedGameLength;
    std::atomic<int> totalNodesExpanded{0};
    std::atomic<int> expansionCycles{0};
    
public:
    AdaptiveProgressiveWidening(float k, float alpha, int gameLength) 
        : baseK(k), baseAlpha(alpha), estimatedGameLength(gameLength) {}
    
    // Calculate max children based on current parameters
    int calculateMaxChildren(int visits) {
        float k = baseK * kMultiplier;
        float alpha = baseAlpha * alphaMultiplier;
        
        return static_cast<int>(std::ceil(k * std::pow(visits, alpha)));
    }
    
    // Update parameters based on observed tree growth
    void updateParameters() {
        expansionCycles.fetch_add(1);
        
        // Only adjust periodically
        if (expansionCycles.load() % 100 == 0) {
            int expanded = totalNodesExpanded.load();
            int cycles = expansionCycles.load();
            
            float expansionRate = static_cast<float>(expanded) / cycles;
            
            // Game phase-based adjustment
            float gameProgress = static_cast<float>(currentPly) / estimatedGameLength;
            
            // Early game: wider tree
            if (gameProgress < 0.25f) {
                // If expansion rate is too low, increase width
                if (expansionRate < 0.5f) {
                    kMultiplier = std::min(kMultiplier * 1.05f, 2.0f);
                } 
                // If expansion rate is too high, decrease width
                else if (expansionRate > 1.5f) {
                    kMultiplier = std::max(kMultiplier * 0.95f, 0.5f);
                }
            } 
            // Mid game: balanced exploration
            else if (gameProgress < 0.75f) {
                // Gradually normalize parameters
                kMultiplier = kMultiplier * 0.99f + 1.0f * 0.01f;
                alphaMultiplier = alphaMultiplier * 0.99f + 1.0f * 0.01f;
            } 
            // Late game: deeper search
            else {
                // Increase alpha for more selective growth
                alphaMultiplier = std::min(alphaMultiplier * 1.02f, 1.5f);
                // Decrease k to focus on promising paths
                kMultiplier = std::max(kMultiplier * 0.98f, 0.7f);
            }
        }
    }
    
    // Decide if a node should be expanded based on policy
    bool shouldExpandNode(Node* node, const std::vector<float>& policyValues) {
        int maxChildren = calculateMaxChildren(node->visits);
        
        // If we haven't reached the limit, expand
        if (node->children.size() < maxChildren) {
            return true;
        }
        
        // Check if there's a high-value move worth exploring
        for (size_t i = 0; i < policyValues.size(); i++) {
            if (!node->hasChildForAction(i) && 
                policyValues[i] > 0.05f && // Only consider significant moves
                node->children.size() < maxChildren * 1.5f) { // Allow some flexibility
                return true;
            }
        }
        
        return false;
    }
    
    // Prioritize actions for expansion based on policy
    std::vector<int> prioritizeActionsForExpansion(
            const std::vector<float>& policyValues,
            const std::unordered_set<int>& existingActions) {
        
        // Create pairs of (action index, policy value)
        std::vector<std::pair<int, float>> actionScores;
        for (size_t i = 0; i < policyValues.size(); i++) {
            if (existingActions.count(i) == 0 && policyValues[i] > 0.0f) {
                actionScores.push_back({i, policyValues[i]});
            }
        }
        
        // Sort by policy value (descending)
        std::sort(actionScores.begin(), actionScores.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        // Extract action indices
        std::vector<int> prioritizedActions;
        for (const auto& pair : actionScores) {
            prioritizedActions.push_back(pair.first);
        }
        
        return prioritizedActions;
    }
    
    // Update game state
    void advancePly() {
        currentPly++;
    }
    
    // Notify when a node is expanded
    void notifyNodeExpanded() {
        totalNodesExpanded.fetch_add(1);
    }
};
```

This implementation:
1. Dynamically adjusts widening parameters based on game phase
2. Adapts to observed tree growth patterns
3. Uses policy information to guide which actions to explore
4. Balances exploration/exploitation throughout the game
5. Preserves search focus while allowing for discovery of unexpected moves

## Conclusion: Integrated optimization strategy

To significantly improve MCTS performance for Gomoku, Chess, and Go, implement these optimizations in stages:

1. **First optimization phase**: Focus on the batching mechanism and virtual loss implementation to immediately improve GPU utilization and reduce thread collisions.

2. **Second optimization phase**: Implement the high-performance transposition table and the improved leaf selection approach to reduce memory contention and improve batch formation.

3. **Third optimization phase**: Add the adaptive RAVE and progressive widening implementations to improve search quality, particularly for Go and Gomoku.

4. **Fine-tuning phase**: Adjust parameters specifically for each game type:
   - For Gomoku: Emphasize RAVE and progressive widening with higher kMultiplier
   - For Chess: Focus on virtual loss and transposition table efficiency
   - For Go: Prioritize all optimizations, especially batching and RAVE

These optimizations address the core bottlenecks in the existing implementation while preserving the MCTS algorithm's correctness. The result will be significantly improved CPU/GPU utilization, larger and more consistent batch sizes, and faster, stronger MCTS performance across all target games.