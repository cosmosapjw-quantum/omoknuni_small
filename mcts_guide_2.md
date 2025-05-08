# Implementing a Leaf-Parallel MCTS Engine in C++

We will build a production-quality Monte Carlo Tree Search (MCTS) engine in C++, suitable for games like Gomoku, Chess, and Go.  Our design follows an **AlphaZero**-style architecture: multiple CPU threads run MCTS simultaneously, pausing at leaf nodes to collect board states for batched neural-net evaluation on a GPU.  Key features include virtual loss, progressive widening, a central inference queue (`moodycamel::ConcurrentQueue`), and promise/future mechanisms for thread synchronization.  We will use standard C++ threading (`std::thread`, `std::mutex`, `std::atomic`) so it runs on both Linux and Windows (64-bit).  Throughout, we use **spdlog** for logging and **ThreadSanitizer (TSAN)** for debugging data races.

![A human player considering a chess move as an analogy for MCTS decision-makingoaicite:2](65†embed_image)
*MCTS is widely used in perfect-information games (Chess, Go, etc.).  We combine it with neural-network evaluation (AlphaZero style) to guide search by policy/value predictions.*

## Monte Carlo Tree Search (MCTS) Core

MCTS repeatedly performs four steps: **Selection**, **Expansion**, **Simulation (Rollout)**, and **Backpropagation**.  In selection, starting at the root node we descend by choosing child nodes that maximize the UCT score:

```cpp
double uct_value = node->meanValue() + 
    C * sqrt(log(parent->visits()) / (1 + node->visits()));
```

Typically `C≈√2`.  We select the child with highest UCT until we reach a leaf (unexpanded) node.  In **Expansion**, we add one or more children for unexplored moves.  In **Simulation**, we perform a (random or heuristic) playout from the new leaf to a terminal state to get a result (win/loss/draw).  Finally, in **Backpropagation**, we propagate the outcome up the tree: for each node on the path, we increment its visit count and add the simulation reward to its total value.

```cpp
struct Node {
    std::vector<Node*> children;
    Node* parent;
    std::atomic<int> visits{0};
    std::atomic<double> value{0.0};
    std::mutex mtx;  // only if using mutex protection
    // ... (game-specific state, prior probabilities, etc.)
};

// UCT selection (single-threaded example)
Node* select_uct(Node* root, double C) {
    Node* node = root;
    while (!node->children.empty()) {
        double best_score = -1e9;
        Node* best_child = nullptr;
        for (Node* c : node->children) {
            // UCT = Q/N + C*sqrt(log(N_parent)/N)
            double q = c->value.load();
            int n = c->visits.load();
            double score = (n > 0 ? q/n : 0.0) +
                           C * sqrt(log(node->visits.load() + 1) / (n + 1));
            if (score > best_score) {
                best_score = score;
                best_child = c;
            }
        }
        node = best_child;
    }
    return node;
}
```

We can use **progressive widening** to handle games with very large action sets (like Go).  Progressive widening limits the number of children expanded at a node based on its visit count, gradually adding moves as more simulations occur.  For example, only expand a new child if

```cpp
node->children.size() < sqrt(node->visits())
```

or using a threshold like `k * visits^α`.  This keeps the branching factor manageable in complex games.

## Multi-Threaded Playouts: Virtual Loss and Progressive Widening

To exploit multiple CPU cores, we run **tree-parallel MCTS**: many threads share the same search tree.  Each thread independently performs Select/Expand/Simulate/Backup cycles.  However, without coordination, threads often duplicate work and conflict.  To mitigate this, we use **virtual loss**: when a thread traverses a node during selection, it temporarily “loses” that node (reducing its UCT score) so other threads are discouraged from selecting the same path.  Concretely, when a thread visits a node, it increments a *virtual loss* counter (or subtracts from its value) and later removes that loss in backpropagation.

```cpp
// Example of virtual loss during selection (pseudo-code)
Node* select_with_virtual_loss(Node* root, int virtual_loss) {
    Node* node = root;
    node->visits.fetch_add(virtual_loss);
    while (!node->children.empty()) {
        Node* best = nullptr; double bestScore = -1e9;
        for (Node* c : node->children) {
            // Compute UCT using (value - virtualLoss) and (visits + virtualLoss)
            double q = c->value.load() - c->virtualLoss;
            int n = c->visits.load() + virtual_loss;
            double score = (n>0 ? q/n : 0) + 
                           C * sqrt(log(node->visits.load()) / (n+1));
            if (score > bestScore) {
                bestScore = score; best = c;
            }
        }
        node = best;
        node->visits.fetch_add(virtual_loss);
        node->virtualLoss += virtual_loss;
    }
    return node;
}
```

When the thread eventually performs a simulation and backs up, it removes the virtual loss: it decrements the `visits` and `virtualLoss` that it earlier added.  This ensures “tentative” losses don’t permanently affect statistics.  Virtual loss encourages diversity: “it will inspire other threads to traverse different paths”. Without it, threads might repeatedly explore the same high-UCT branch, wasting CPU.

While threads choose nodes in parallel, we must protect shared node updates.  We can use **atomic variables** for counters (as above) or **fine-grained mutexes** on each node.  For instance, updating `visits` and `value` atomically is thread-safe:

```cpp
node->visits.fetch_add(1, std::memory_order_relaxed);
node->value.fetch_add(reward, std::memory_order_relaxed);
```

Alternatively, use a small mutex per node:

```cpp
std::lock_guard<std::mutex> lock(node->mtx);
node->visits++;
node->value += reward;
```

Either way, ensure **backup** is thread-safe.  The PRD explicitly notes: “Must protect shared N/Q statistics with atomics or fine-grained mutexes”.

We also apply progressive widening in this threaded context: each node tracks how many children it has.  If the node’s visit count is low, we allow only a few children and add more as visits grow.  For example:

```cpp
if (node->children.size() < alpha * pow(node->visits.load(), beta)) {
    // Expand a new child
    node->children.push_back(new Node(...));
}
```

This way, threads won’t explode the tree when a node is rarely visited.

## Leaf-Parallelization with moodycamel::ConcurrentQueue

In leaf-parallel MCTS we offload neural-net evaluations to a centralized queue, improving GPU throughput.  Each thread runs selection until it reaches a leaf that needs evaluation.  Instead of immediately running a (slower) rollout or inference, the thread packages the leaf state and submits it to a **lock-free queue**. We use **moodycamel::ConcurrentQueue** – a high-performance multi-producer, multi-consumer queue – to collect leaf states across threads.

```cpp
#include <concurrentqueue.h>
struct EvalRequest {
    GameState state;                       // the board state to evaluate
    std::promise<EvalResult> promise;      // will receive network output
};
moodycamel::ConcurrentQueue<EvalRequest*> evalQueue;
```

When a thread hits an unexpanded leaf, it performs expansion *partially* (adding children if needed) and then stops to evaluate the node via NN.  It creates an `EvalRequest` with the current `GameState` and a `std::promise<EvalResult>`. The thread then **pushes** this request into the queue:

```cpp
EvalRequest* req = new EvalRequest{state, std::promise<EvalResult>()};
std::future<EvalResult> fut = req->promise.get_future();
evalQueue.enqueue(req);
```

Immediately after enqueuing, the thread **blocks on the future** (`fut.get()`), pausing the search until the inference result returns.  Meanwhile, other threads can enqueue their leaf states.

On the other side, a dedicated **inference thread** (or threads) pulls requests from `evalQueue` to form batches.  It repeatedly tries to dequeue up to `batch_size` states, or waits up to a timeout if the queue is not full.  This “wait-for-batch” logic is crucial: the PRD warns that otherwise “batch size stuck at 1” can occur (the evaluator immediately drains the queue).  In practice we use a loop with `try_dequeue` and `std::this_thread::sleep_for` until either enough requests are collected or the batch-timeout elapses.

```cpp
std::vector<EvalRequest*> batch;
auto start = std::chrono::steady_clock::now();
while (batch.size() < batch_size) {
    EvalRequest* req = nullptr;
    if (evalQueue.try_dequeue(req)) {
        batch.push_back(req);
    } else {
        // If timed out, proceed with whatever we have
        if (std::chrono::steady_clock::now() - start >= timeout) break;
        // Sleep briefly to reduce busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
```

This central queue and batching approach decouples the tree threads from the GPU.  It also naturally provides **leaf parallelism**: multiple threads can reach leaves at roughly the same time and have their states evaluated in one GPU batch, while the tree itself remains single (shared).

## Centralized GPU Inference via LibTorch

Our GPU evaluator is based on LibTorch (the C++ PyTorch API).  We load a TorchScript model with policy and value heads (e.g. a ResNet) on the GPU.  The inference thread repeatedly forms batches of preprocessed input tensors and calls the model's `forward()`.  For example:

```cpp
torch::jit::script::Module model = torch::jit::load("model.pt");
model.to(device);  // e.g., CUDA

while (!shutdown) {
    // Collect batch (see above)
    std::vector<torch::Tensor> inputs;
    for (auto* req : batch) {
        inputs.push_back(preprocess(req->state));  // see next section
    }
    torch::Tensor batchInput = torch::stack(inputs).to(device);
    auto output = model.forward({batchInput}).toTuple();
    auto policies = output->elements()[0].toTensor().cpu();
    auto values   = output->elements()[1].toTensor().cpu();
    // Assign results back to threads via promises
    for (size_t i = 0; i < batch.size(); i++) {
        EvalResult result;
        result.policy = policies[i];
        result.value  = values[i].item<float>();
        batch[i]->promise.set_value(result);
    }
}
```

We ensure **batch size and timeout** are configurable (e.g. from YAML or command-line) so users can tune throughput vs. latency.  The PRD explicitly warns that without a timeout, the batch might never fill, causing single-request batches.  By tuning these parameters, we balance GPU utilization vs. search responsiveness.

The **preprocessing** step must produce identical tensor shapes for every request in a batch.  For board games, a common approach is to encode the game state as a fixed-size float tensor (e.g. channels for player stones and history).  We must **validate** that all states in a batch share the same dimensions (especially important if different games or board sizes are supported).

Once results return, each worker thread unblocks (via `future.get()`) and resumes the MCTS cycle: it uses the returned policy and value to set the prior for the leaf’s children and to backpropagate the value.  Often the leaf node is assigned those prior probabilities (for policy) and the network value, and backpropagation updates all ancestors’ statistics with the value.

## Promise/Future for Thread Synchronization

We use `std::promise`/`std::future` to deliver inference results back to worker threads.  Each `EvalRequest` holds a promise.  After launching inference, the inference thread sets `promise.set_value(result)` for each request.  The original worker uses `future.get()`, which blocks until `set_value()` is called. This elegant mechanism avoids complex locks or condition variables. For example:

```cpp
// Worker thread code when reaching a leaf needing NN eval:
EvalRequest* req = new EvalRequest{state, std::promise<EvalResult>()};
std::future<EvalResult> fut = req->promise.get_future();
evalQueue.enqueue(req);
EvalResult res = fut.get();  // blocks here until inference thread fulfills
// Now use res.policy, res.value to continue MCTS...
```

This pattern cleanly decouples tree search threads from inference.  It also integrates well with MoodyCamel’s queue: tasks can carry their promise, and the inference loop fulfills them in FIFO order.  Because `std::future` blocks by default, we usually detach threads or manage them so they can resume when data arrives.

## Thread-Safe Backup and Tree Updates

After a simulation (which might be a random rollout if no neural value) finishes, each worker backpropagates the result up the tree.  It must update `visits` and `value` in all nodes along the path. As noted, these updates must be atomic or protected by locks.  A typical backup looks like:

```cpp
for (Node* node : path) {
    // Either atomic ops:
    node->visits.fetch_add(1, std::memory_order_relaxed);
    node->value.fetch_add(reward, std::memory_order_relaxed);
    // Or with a mutex:
    // std::lock_guard<std::mutex> lock(node->mtx);
    // node->visits++;
    // node->value += reward;
}
```

Using `std::atomic<int>` and `std::atomic<double>` (C++20) is lock-free on many platforms.  Alternatively, the PRD suggests “a small per-node mutex for value updates”.  Whichever method is used, consistency is vital: concurrent increments must not corrupt the statistics.  (Note: if using `double`, beware that atomic doubles may not be lock-free on all systems; a mutex is safer if maximum portability is needed.)

In our code, we combine atomic counters for visits and a spinlock for values, which is lightweight if contention is low.  After backprop, the nodes’ UCT scores reflect the new statistics for all threads to see.

## Input Preprocessing and Tensor Batching

Each game’s state must be converted to a tensor input for the neural network.  We assume a modular game layer: each `GameState` can produce a fixed-size array (e.g. an `std::vector<float>`) representing the board features (stones, player to move, etc.).  It’s crucial that **all games use the same tensor layout** or that the engine resets tensors between games, so that batching works.  A typical preprocessing might look like:

```cpp
torch::Tensor preprocess(const GameState& s) {
    // Example for a 8x8 board with 2 channels:
    auto tensor = torch::zeros({channels, board_size, board_size});
    for (int i = 0; i < board_size; i++) {
        for (int j = 0; j < board_size; j++) {
            if (s.isMyStone(i,j))  tensor[0][i][j] = 1.0;
            if (s.isOppStone(i,j)) tensor[1][i][j] = 1.0;
        }
    }
    // e.g., add additional planes for history or to-move
    return tensor;
}
```

In a batch, we simply stack these tensors: `torch::stack(inputs)`.  Ensure that `GameState` always produces the **same tensor shape**; otherwise inference will throw shape-mismatch errors.  The PRD explicitly warns to “validate shapes before stacking”.  If multiple game types are supported, one could pad tensors or reject incompatible games at batch time.

## Queue Contention and Thread Management

The MooneyCamel queue is lock-free and highly optimized, but under heavy multi-thread load it can still suffer cache contention.  To mitigate this, the PRD suggests tuning the number of worker threads relative to CPU cores (e.g. don’t launch dozens of threads if cores are few) and even sharding into multiple queues if needed.  In practice, one could use one queue per NUMA node or per CPU socket to reduce cross-core traffic.

In our basic design, we use one global queue for simplicity.  We minimize contention by having the inference thread dequeue in **bulk** (moodycamel supports `try_dequeue_bulk`) and sleeping briefly when the queue is empty.  Worker threads only block on their own futures and re-enter selection when resumed, keeping the CPU cores busy.  If profiling shows a queue bottleneck, options include using `try_dequeue_from_producer` or a pool of queues.

## Configurability and Cross-Platform Threads

All key parameters (number of threads, batch size, batch timeout) are configurable at runtime (e.g. via a YAML or CLI).  For example, we might allow:

```yaml
mcts:
  threads: 8
  virtual_loss: 1
  progressive_widening:
    alpha: 1.5
    beta: 0.5
gpu:
  batch_size: 32
  timeout_ms: 5
```

This way users can adjust the parallelism for their hardware.  The code uses `std::thread`, `std::mutex`, and `std::atomic` which are cross-platform (supported on Linux and Windows out of the box).  The PRD notes that Windows might treat condition variables or semaphores slightly differently, but in practice `std::thread`/`std::mutex` are implemented on top of Win32 threads, so no special #ifdefs are needed.  (We should, however, test on Windows to ensure e.g. timed waits behave as expected.)

## Logging and Debugging

For observability, we integrate **spdlog** – a fast C++ logging library.  In each thread and the inference loop, we log key events (e.g. “node expanded”, “batch sent”, “result received”).  Example setup:

```cpp
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

auto logger = spdlog::stdout_color_mt("mcts");
logger->set_level(spdlog::level::info);

logger->info("Starting MCTS with {} threads", numThreads);
```

Since spdlog’s multi-threaded loggers (`_mt`) are thread-safe, we can safely log from any thread.  We also use its **backtrace** feature to store recent logs in a ring buffer and dump them on errors, which aids debugging.

To catch concurrency bugs, we strongly recommend running under **ThreadSanitizer (TSAN)** during development.  TSAN is available in Clang/GCC (`-fsanitize=thread`) and will flag data races in our tree updates or promise handling.  As the PRD suggests, enabling TSAN early in testing can expose subtle bugs that only occur under parallel load. (The trade-off is a \~5–15× slowdown, so TSAN is used in debug builds, not in production runs.)

Finally, build the project with maximum warnings and consider sanitizers (`-fsanitize=address,thread`) during integration testing.  Use `spdlog`’s logging to record batch sizes and latencies.  The PRD’s mitigation steps include profiling tools (Nsight, Linux perf) and incremental testing to isolate bottlenecks.

## Summary

In summary, our C++ MCTS engine consists of:

* **MCTS core**: Node class, UCT selection, expansion, rollout, backprop.
* **Parallel threads**: Use `std::thread`, apply virtual loss during selection, optional progressive widening.
* **Leaf queue**: A `moodycamel::ConcurrentQueue<EvalRequest*>` that collects NN evaluation requests.  Workers push `EvalRequest` with a promise; an inference thread batches requests to LibTorch (GPU).
* **LibTorch inference**: A TorchScript model on GPU, invoked on batches of size ≤ `batch_size` or after `timeout_ms`.  Output (policy, value) is fed back via promises.
* **Thread-safe backup**: Node visits/values updated atomically or under mutex to avoid races.
* **Configurable**: Number of threads, batch size, timeout etc. are not hard-coded.
* **Logging/Debugging**: Use spdlog for runtime logs and TSAN for finding data races.

This design meets the PRD goals: it’s platform-agnostic (Linux/Windows), uses leaf-parallel MCTS with a shared tree, employs batched GPU inference, and is robust to concurrency issues.  Following these steps and code patterns should yield a stable, high-performance MCTS engine for games like Gomoku, Chess, and Go.

**Sources:** Core MCTS concepts and parallelization strategies (PRD); implementation tips (spdlog, TSAN). Each code snippet and recommendation is informed by these references.
