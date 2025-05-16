# Batch Evaluation with moodycamel::ConcurrentQueue

This skeleton demonstrates how to implement **leaf parallelization** with a centralized, batched neural-network (NN) evaluator. It collects requests up to a configurable batch size or until a timeout elapses, then performs a single inference for the entire batch.

## Key Components

- **EvaluationRequest**: encapsulates the game-state tensor and a `std::promise` to return results.  
- **EvaluationResult**: holds the NN’s policy and value outputs.  
- **moodycamel::ConcurrentQueue**: lock-free queue for passing requests from MCTS workers to the evaluator thread.  
- **BatchEvaluator**: thread that aggregates requests, runs batched inference, and fulfills promises.  
- **MCTSWorker**: example worker thread that enqueues states and waits on futures.  

---

```cpp
#include <torch/torch.h>
#include <moodycamel/ConcurrentQueue.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>

//------------------------------------------------------------------------------
// 1. Data Structures
//------------------------------------------------------------------------------

struct EvaluationResult {
    torch::Tensor policy;  // e.g., [batch_size, action_space]
    torch::Tensor value;   // e.g., [batch_size, 1]
};

struct EvaluationRequest {
    torch::Tensor state;
    std::promise<EvaluationResult> promise;
};

using ConcurrentQueue = moodycamel::ConcurrentQueue<EvaluationRequest>;

//------------------------------------------------------------------------------
// 2. BatchEvaluator: collects requests, batches them, runs NN, fulfills promises
//------------------------------------------------------------------------------

class BatchEvaluator {
public:
    BatchEvaluator(ConcurrentQueue& queue,
                   std::shared_ptr<torch::nn::Module> model,
                   size_t batch_size,
                   int timeout_ms)
      : queue_(queue)
      , model_(std::move(model))
      , batch_size_(batch_size)
      , timeout_ms_(timeout_ms)
      , running_(true)
    {}

    // Thread entry point
    void run() {
        while (running_) {
            std::vector<EvaluationRequest> batch;
            batch.reserve(batch_size_);

            // 1) Wait for at least one request or timeout
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait_for(lock,
                    std::chrono::milliseconds(timeout_ms_),
                    [&]{ return queue_.size_approx() > 0 || !running_; });
            }

            if (!running_) break;

            // 2) Dequeue up to batch_size_ requests
            queue_.try_dequeue_bulk(batch, batch_size_);
            if (batch.empty()) continue; // nothing to do

            // 3) Prepare batched input
            std::vector<torch::Tensor> inputs;
            inputs.reserve(batch.size());
            for (auto& req : batch) {
                inputs.push_back(req.state);
            }
            torch::Tensor batched_input = torch::stack(inputs);

            // 4) Inference on GPU (no grad)
            torch::NoGradGuard no_grad;
            auto output = model_->forward(batched_input).toTuple();
            torch::Tensor policy_out = output->elements()[0].toTensor();
            torch::Tensor value_out  = output->elements()[1].toTensor();

            // 5) Fulfill promises
            for (size_t i = 0; i < batch.size(); ++i) {
                EvaluationResult result{
                    policy_out[i].detach(),
                    value_out[i].detach()
                };
                batch[i].promise.set_value(std::move(result));
            }
        }
    }

    void stop() {
        running_ = false;
        cond_var_.notify_all();
    }

    // Called by producers after enqueue
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        cond_var_.notify_one();
    }

private:
    ConcurrentQueue& queue_;
    std::shared_ptr<torch::nn::Module> model_;
    size_t batch_size_;
    int timeout_ms_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};

//------------------------------------------------------------------------------
// 3. Example MCTS worker thread
//------------------------------------------------------------------------------

void MCTSWorker(int id,
                ConcurrentQueue& queue,
                BatchEvaluator& evaluator,
                int num_requests) {
    for (int i = 0; i < num_requests; ++i) {
        // 1) Create a dummy state tensor (e.g., 3x19x19 for Go)
        torch::Tensor state = torch::rand({3, 19, 19});

        // 2) Prepare promise/future
        std::promise<EvaluationResult> prom;
        auto fut = prom.get_future();

        // 3) Enqueue request
        EvaluationRequest req{state, std::move(prom)};
        queue.enqueue(std::move(req));
        evaluator.notify();

        // 4) Wait for result
        EvaluationResult res = fut.get();
        std::cout << "Worker " << id
                  << " got policy size: " << res.policy.sizes()
                  << ", value: " << res.value.item<float>() << "\n";

        // 5) Update MCTS tree with res.policy and res.value...
    }
}

//------------------------------------------------------------------------------
// 4. Main: set up queue, evaluator thread, and workers
//------------------------------------------------------------------------------

int main() {
    // 1) Concurrent queue
    ConcurrentQueue queue;

    // 2) Dummy model (replace with your DDW-RandWire-ResNet)
    auto model = std::make_shared<torch::nn::Sequential>();
    // ... load weights, to(model->device()) ...

    // 3) BatchEvaluator config
    size_t batch_size = 8;
    int timeout_ms  = 5;
    BatchEvaluator evaluator(queue, model, batch_size, timeout_ms);

    // 4) Start evaluator thread
    std::thread eval_thread(&BatchEvaluator::run, &evaluator);

    // 5) Spawn MCTS workers
    const int num_workers = 4;
    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(MCTSWorker, i, std::ref(queue), std::ref(evaluator), 16);
    }

    // 6) Join workers
    for (auto& w : workers) w.join();

    // 7) Stop evaluator
    evaluator.stop();
    eval_thread.join();

    return 0;
}
```

---

## How It Works

1. **MCTSWorker** creates a `std::promise`/`std::future` pair, wraps the game-state tensor in an `EvaluationRequest`, and enqueues it.  It then calls `evaluator.notify()` and waits on the future.  
2. **BatchEvaluator** sits in a loop, waiting on a condition variable for either:
   - The queue to become non-empty, or  
   - A timeout (`timeout_ms`)  
   This ensures it collects multiple requests before proceeding.  
3. After waking, it uses `try_dequeue_bulk` to pull up to `batch_size` requests at once.  If none are available, it loops again.  
4. It stacks all state tensors into a single batched tensor, runs a single forward pass (no grad), then slices the policy/value outputs and fulfills each promise.  
5. **MCTSWorker** unblocks, retrieves the policy & value, and updates the tree.  

### Benefits & Security Considerations

- **Batching** maximizes GPU utilization and reduces per-request overhead.  
- **Lock-free queue + condition variable** enforces **defense in depth**: the queue is non-blocking, and the CV prevents busy-waiting.  
- **Promise/Future** securely transfers results to the correct thread, avoiding race conditions.  
- **Timeout** avoids starvation when traffic is low.  
- **std::atomic** and proper locking ensure thread safety without exposing sensitive data.  

This pattern can be incrementally integrated into your MCTS core. Adjust tensor shapes, model loading, and tree updates as needed.