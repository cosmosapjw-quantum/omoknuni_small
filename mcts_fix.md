# MCTS Batch Accumulation Issue Analysis

I've analyzed the logs from your self-play system and identified why batch accumulation isn't working. The issue appears to be that **no leaf nodes are being submitted to the evaluation queue**, despite the batch accumulator being properly initialized and running.

## Root Cause Analysis

The primary issues appear to be:

1. **Search Not Starting**: The MCTS engines are initialized, but the actual search process isn't running or is blocked.

2. **Disconnection Between Components**: While the batch accumulators are running (as shown by many log messages), no items are being submitted for evaluation.

3. **Configuration Mismatch**: The logs show root parallelization is disabled (`"Using single root worker per game (root parallelization disabled - this is a workaround note in code)"`) despite the config trying to enable it.

## Specific Code Issues

Looking at the MCTS code, I've identified these likely causes:

### 1. Search Not Being Triggered or Executing

```cpp
// In MCTSEngine::runSearch
while (completed_simulations < num_simulations && search_running_) {
    // Submit a batch of simulations
    submitSimulationBatch(current_batch);
    
    // Wait for batch completion
    executor_.run(taskflow_).wait();
    
    completed_simulations += current_batch;
}
```

The most likely issue is that the engine's `runSearch` method isn't being called by the self-play manager, or it's being called but the internal taskflow execution isn't working properly.

### 2. Leaf Selection/Evaluation Problem

In `MCTSEngine::treeTraversalTask` or `MCTSEngine::executeSingleSimulation`, leaves should be selected and added to the queue:

```cpp
// Queue for neural network evaluation
if (current->needsEvaluation()) {
    current->markEvaluationInProgress();
    
    PendingEvaluation pending;
    pending.state = current->getState().clone();
    pending.node = current;
    pending.path = std::move(path);
    
    leaf_queue_->enqueue(std::move(pending));
    pending_evaluations_++;
}
```

This code isn't executing or is failing silently because no items are being added to the queue.

### 3. Root Parallelization Issue

The logs show a mismatch in root parallelization configuration:
- Config: `mcts_use_root_parallelization: 'true'`
- Actual: `"Using single root worker per game (root parallelization disabled..."`

This is significant because the comment mentions it's "CRITICAL for batch formation".

## Recommended Fixes

1. **Add Debug Logging in Key Methods**:
   ```cpp
   // In SelfPlayManager's method that should trigger search
   std::cout << "Starting MCTS search for game " << game_id << std::endl;
   engine->runSearch(...);
   
   // In MCTSEngine::runSearch at the beginning
   std::cout << "MCTSEngine::runSearch - Starting with simulations=" << num_simulations << std::endl;
   
   // In MCTSEngine::submitSimulationBatch
   std::cout << "Submitting batch of " << batch_size << " simulations" << std::endl;
   
   // In treeTraversalTask when finding a leaf
   std::cout << "Found leaf node, needs evaluation: " << current->needsEvaluation() << std::endl;
   
   // When adding to queue
   std::cout << "Adding leaf to queue: " << (queue_success ? "success" : "FAILED") << std::endl;
   ```

2. **Fix the Shared Queue Integration**:
   
   The critical issue may be how thread-local batches are added to the shared queue. Look for code like this:

   ```cpp
   // After collecting thread-local batches
   if (!thread_local_batch.empty()) {
       // This part might be missing or failing
       shared_leaf_queue_->enqueue_bulk(
           std::make_move_iterator(thread_local_batch.begin()),
           thread_local_batch.size()
       );
       
       // Notify after adding
       if (external_queue_notify_fn_) {
           external_queue_notify_fn_();
       }
   }
   ```

3. **Fix Root Parallelization Configuration**:
   
   There appears to be a mismatch between configuration and actual behavior. Find where this is handled:

   ```cpp
   // Look for code like this in SelfPlayManager
   bool use_root_parallelization = config.getBool("mcts_use_root_parallelization", false);
   // Check if this value is being overridden somewhere
   ```

4. **Check for Deadlocks in the Task Execution**:
   
   The Taskflow executor might be misconfigured. Look at the executor initialization:

   ```cpp
   executor_(settings.num_threads)
   ```
   
   Ensure the number of threads is appropriate for your system.

5. **Verify Game Start Logic**:

   The most basic issue might be that games aren't properly starting. Add logging:
   
   ```cpp
   // In SelfPlayManager's game loop
   std::cout << "Starting game " << game_id << " with engine " << engine_id << std::endl;
   ```

## Most Critical Fix

The most likely fix based on the logs is to correct the issue where leaf nodes aren't being added to the shared queue. The key place to focus is:

```cpp
// In whichever method submits thread-local batches to the shared queue
if (!thread_local_batch.empty()) {
    std::cout << "Submitting " << thread_local_batch.size() << " leaves to shared queue" << std::endl;
    
    // Ensure this code is properly adding to the shared queue
    bool success = shared_leaf_queue_->enqueue_bulk(
        std::make_move_iterator(thread_local_batch.begin()),
        thread_local_batch.size()
    );
    
    std::cout << "Shared queue submission " << (success ? "succeeded" : "FAILED") << std::endl;
    
    // Make sure the notification is called
    if (external_queue_notify_fn_) {
        external_queue_notify_fn_();
    }
    
    thread_local_batch.clear();
}
```

This would help confirm whether leaf nodes are being discovered and properly added to the queue that the batch accumulator is monitoring.

-----------
-----------

# MCTS Engine Startup (runSearch)

Self-play uses `engine.search(state)`, which calls `MCTSEngine::search` and in turn `runSearch`.  Logs around `engine.search(*game)` confirm this: the manager prints **“Starting MCTS search…”** and **“Completed MCTS search…”** for each move.  Inside `search()`, the code prints **“Starting MCTS search with num\_simulations=…”**.  Confirm that this log appears when games begin.  If it does not, the search may not be running at all.

* *Check*: Ensure `num_simulations` is positive (default 800) and that `runSearch` is invoked. If `num_simulations` were set to 0 in **config.yaml**, the loop would never run. Add a log at the top of `runSearch()` to verify entry.
* *Fix*: If no **“Starting MCTS search…”** log appears, the issue may lie before search (e.g. game is terminal or engine not called). Adding early prints in `selfplay::generateGame` and `MCTSEngine::search/runSearch` can catch this.

# Leaf Selection and Expansion

Within `runSearch`, `executeSerialSearch()` drives the MCTS loop.  The core step is `selectLeafNode(root)`, which should find an unexpanded leaf, expand it, and return it for evaluation.  In our debug code, `selectLeafNode` has two key parts: it expands the root if needed, then traverses. When it finds a **non-terminal leaf**, it expands it and **returns it for immediate evaluation** (the “expanded leaf for direct evaluation” branch).

* *Check*: Instrument `selectLeafNode` to log when it returns `nullptr` (meaning it hit a node with pending evaluation) vs. a valid leaf.  The existing logs only cover expansion and returning leaves.  If `selectLeafNode` repeatedly returns `nullptr`, no leaves will be sent to the evaluator.
* *Fix*: If a node keeps having `hasPendingEvaluation()` true, clear its flag to break deadlock. The code already resets flags occasionally, but you may need to force `leaf->clearEvaluationFlag()` on repeated failures. Also verify that `expandNonTerminalLeaf(leaf)` actually adds children (if expansion fails, leaf remains leaf but should still be marked for eval).

# Submitting Leaves to the Evaluator

When a valid leaf is returned, the code does `safelyMarkNodeForEvaluation(leaf)` and creates a `PendingEvaluation`.  For *shared-queue* mode (used in self-play), the code checks `if (use_shared_queues_ && evaluator_->getBatchAccumulator())` and then **directly adds** the evaluation to the `BatchAccumulator`.  Otherwise, it pushes to a local `leaf_batch`.  Finally, it submits batches of `PendingEvaluation` to the queue.

* *Check*: Verify the `BatchAccumulator` was started by `SelfPlayManager`. In the manager we see:

  ```cpp
  auto* batch_acc = shared_evaluator_->getBatchAccumulator();
  if (batch_acc) { batch_acc->start(); }
  shared_evaluator_->start();
  ```

  . If `batch_acc->start()` wasn’t called, leafs stuck there. Also ensure `external_queue_notify_fn_` is set and invoked – the code calls the notify function after enqueues.
* *Fix*: If no leaves are reaching the evaluator, check that `use_shared_queues_` is true (it is set by `engine->setSharedExternalQueues(&shared_leaf_queue_, &shared_result_queue_, notify_fn)`).  Confirm that the evaluator’s external queues are configured (we see in logs: **“setExternalQueues - External queues configured successfully”**).  You might add logs after `addEvaluation` or `enqueue` to see if these methods are actually called.

# Self-Play Manager and MCTS Configuration

`SelfPlayManager` initializes the shared evaluator and creates multiple `MCTSEngine` instances.  It explicitly sets **shared queues** on each engine:

```cpp
engine->setSharedExternalQueues(&shared_leaf_queue_, &shared_result_queue_, notify_fn);
```

.  It also prints the number of simulations and root-parallelization settings on construction.  Note: the code’s comment claims root parallelization is “disabled”, but it never actually sets `settings_.mcts_settings.use_root_parallelization = false`. In fact, pipeline code enables root parallelization and sets `num_root_workers`.  This mismatch could create unexpected behavior (multiple roots per engine).

* *Check*: Ensure `engines_.size()` matches `settings_.num_parallel_games` and that each thread picks a valid engine (generateGames uses `engines_[thread_id % engines_.size()]`).  If `num_parallel_games` > number of engines, some threads may idle.
* *Fix*: Align settings so that one engine per thread (root parallelization false) or properly use multiple roots. The code suggests disabling root parallelization to avoid “cloning issues”, so you could set `settings_.mcts_settings.use_root_parallelization = false` and `num_root_workers = 1` before creating engines to match the intent.

# Neural Network Evaluator Wiring

The neural net is provided to both the shared evaluator and the engines.  In the manager, a `MCTSEvaluator` is constructed with a lambda calling `neural_net_->inference(states)`, and it’s configured to use the shared queues.  Each `MCTSEngine` is also constructed with the same network and given these shared queues.

* *Check*: Confirm the shared evaluator is running (its thread pool started) and that it’s consuming from `shared_leaf_queue_`. You should see its log **“setExternalQueues - External queues configured successfully”** and then activity in its batch collector. Also inspect if `MCTSEngine::executeSerialSearch` is actually calling `evaluator_->getBatchAccumulator()->addEvaluation(...)`. If this path is taken, you should see **“Directly adding leaf to BatchAccumulator”** logs.
* *Fix*: If no activity, consider switching off the `BatchAccumulator` path to force use of the queue: temporarily disable the `if (use_shared_queues_ && ...) { add to accumulator; continue; }` block and let the code push onto `shared_leaf_queue_` via `enqueue_bulk`. This will test the external-queue pipeline more directly.

# Configuration and Environment

Inspect **`config.yaml`** and **`run.sh`** for issues.  The `AlphaZeroPipelineConfig` defaults enable multiple games and set `mcts_num_simulations=800` and `mcts_num_threads=8`.  In `run.sh`, `OMP_NUM_THREADS=20` is set, but the code uses `num_parallel_games` (default 8) with OpenMP to parallelize games. Mismatched thread counts can cause confusion.

* *Check*: Ensure `self_play_num_parallel_games` (config) is not zero.  If it’s lower than `engines_.size()`, some engines won’t be used. Also verify `mcts_num_threads` – although not explicitly used in the engine (only for transposition table sizing), setting it to 0 or a huge number might break assumptions.
* *Fix*: Use realistic thread settings. For example, if you set `self_play_num_parallel_games = 4`, and you have 8 cores, set `OMP_NUM_THREADS=4` to match. In general, try aligning `num_parallel_games`, OpenMP threads, and actual hardware.

# Summary of Next Steps

1. **Add Debug Logging** at critical points:

   * At the start of `runSearch()` and `executeSerialSearch()` (to confirm entry).
   * Inside `selectLeafNode()` when it returns `nullptr` vs non-null.
   * After enqueuing leaves (`enqueue` or `addEvaluation`) and after evaluator notifications.
2. **Trace the First Loop Iteration**: Check if the first pass of the while-loop in `executeSerialSearch` actually finds any leaves. The existing logs on empty tries should help; if you see “consecutive tries without finding leaves,” that pinpoints the failure.
3. **Configuration Audit**: Validate `config.yaml` values for MCTS (simulations, threads, games). Try a minimal config (one game, small threads) to isolate the issue.
4. **Test Without Shared Queues**: As a diagnostic, temporarily disable `setSharedExternalQueues` so each `MCTSEngine` uses its own evaluator (non-shared mode). If leaves then accumulate, the issue is in the shared-queue wiring.
5. **Fix Code Issues**: If you find a specific bug (e.g. flags never cleared, loops exiting prematurely), implement the straightforward fix (e.g. remove the “SUPER CRITICAL FIX” line that forces 100 simulations, or better manage the flag counters).

By following these steps and using the code references above, you should be able to identify the earliest break in the control flow (likely in `selectLeafNode` or the loop logic) and correct it so that leaf evaluations are properly submitted. Each of the cited code blocks highlights how the pieces are intended to work, so comparing runtime logs to these paths will reveal where things diverge.
