Here’s a step-by-step implementation plan that tackles your core challenge—getting a true batched, leaf-parallel evaluation—before you integrate it into the full MCTS engine.

1. Set up a Minimal Stand-Alone Batching Prototype
  • Create a small C++ project that only has:
    – `moodycamel::ConcurrentQueue<EvaluationRequest>`
    – A `LightweightSemaphore` (or condition variable) to signal the evaluator thread
    – A dummy “neural network” function that takes a `std::vector<Tensor>` and returns fake outputs
  • Goal: verify the queue+semaphore + batching logic works in isolation.

2. Define the EvaluationRequest API
  • struct EvaluationRequest {
      Tensor input;                     // single‐state tensor
      std::promise<EvalResult> promise; // for the worker to wait on
    };
  • Ensure that each request carries its own `std::promise` so results can be routed back.

3. Implement the Producer Side (Worker Threads)
  • In `submit_request(state)`:
    1. Build a `Tensor` from the game state.
    2. Push an `EvaluationRequest{tensor, promise}` into the concurrent queue.
    3. Signal the semaphore (or cv) to wake the evaluator.
    4. Return `future = promise.get_future()` so the caller can wait.
  • Test: spawn N worker threads each submitting requests in a loop; verify nothing deadlocks.

4. Implement the Consumer Side (NN Evaluator Thread)
  • Loop forever:
     1. Wait on semaphore until at least one request is available or a timeout elapses.
     2. Dequeue up to `max_batch_size` requests into a local vector.
     3. If vector is empty (timeout with no requests), continue.
     4. Extract tensors, call `batched_input = torch::stack(inputs)`.
     5. Call `outputs = model.forward(batched_input)`.
     6. For each output, fulfill the corresponding `std::promise`.
  • Key: enforce that the consumer thread only calls forward once per batch.

5. Verify Batching Behavior in Isolation
  • Write unit tests that:
    – Submit 1 request and ensure the evaluator waits only up to timeout, then processes a batch of size 1.
    – Submit M requests (M < max_batch_size) simultaneously and ensure they all go through in one batch.
    – Submit > max_batch_size requests and verify they’re split into correct batch sizes.
  • Log batch sizes and timestamps to confirm batching logic.

6. Integrate with Single-Threaded MCTS
  • Replace the dummy NN with the real libtorch model.
  • Inside MCTS leaf expansion, call your `submit_request` + `future.get()`.
  • Run a few single‐threaded games to confirm correctness remains intact.

7. Add Multi‐Threaded MCTS (Without Leaf Parallelization)
  • Spawn multiple MCTS workers that race through the tree but synchronously call `submit_request`/`get()`.
  • Confirm stability: no data races on the tree statistics.
  • Use mutexes or atomics in the tree backup phase.

8. Enable Leaf Parallelization
  • Now your workers will no longer block on `get()` immediately—instead they will enqueue a request and continue exploring other leaves.
  • After they’ve generated enough leaves (or at expansion), they wait on their stored futures.
  • Test that workers can overlap tree traversal with inference, and that batch sizes increase.

9. Logging & Monitoring (spdlog)
  • Log per‐batch: size, queue length, inference time.
  • Log per‐worker: submission time vs. response time.
  • Periodically dump stats so you can trace stalls or suboptimal batch sizes.

10. Performance Profiling & Tuning
  • Profile CPU usage (threads), GPU utilization, and latency distribution.
  • Tune:
    – `max_batch_size` vs. `evaluator_timeout`
    – Number of MCTS threads
    – Libtorch session options (e.g., `enable_mem_pattern`)
  • Identify contention points (queue hot spots, model forward time).

11. Stress Test & Scale-Up
  • Ramp up to target thread count (e.g., 8–16 workers).
  • Play thousands of self-play games to ensure long-term stability.
  • Monitor for memory leaks or deadlocks.

12. Evaluate ONNX Runtime (Optional Optimization)
  • Export your libtorch model to ONNX.
  • Build an ORT session with CUDA EP and dynamic batch enabled.
  • Plug ORT into your evaluator thread; rerun profiling to compare throughput.

13. Finalize & Integrate into Full Engine
  • Merge this batching subsystem back into the main MCTS codebase.
  • Add comprehensive unit and integration tests around the queue, semaphore, and inference.
  • Ensure build scripts (CMake) include all dependencies (`moodycamel::ConcurrentQueue`, libtorch/ORT, spdlog).

By following these steps—starting from a tiny prototype, expanding into MCTS integration, and then optimizing—you’ll guarantee that your batch size actually grows beyond 1, that GPU inference is efficient, and that your leaf-parallel MCTS runs both correctly and at top speed.