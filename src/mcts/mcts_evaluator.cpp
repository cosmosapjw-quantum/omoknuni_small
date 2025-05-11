// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>

namespace alphazero {
namespace mcts {

MCTSEvaluator::MCTSEvaluator(InferenceFunction inference_fn, 
                           size_t batch_size, 
                           std::chrono::milliseconds timeout)
    : inference_fn_(std::move(inference_fn)),
      batch_size_(batch_size),
      timeout_(timeout),
      shutdown_flag_(false),
      total_batches_(0),
      total_evaluations_(0),
      cumulative_batch_size_(0),
      cumulative_batch_time_ms_(0) {
}

MCTSEvaluator::~MCTSEvaluator() {
    stop();
}

void MCTSEvaluator::start() {
    if (worker_thread_.joinable()) {
        return; // Already started
    }
    
    shutdown_flag_ = false;
    worker_thread_ = std::thread(&MCTSEvaluator::processBatches, this);
}

void MCTSEvaluator::stop() {
    if (!worker_thread_.joinable()) {
        return; // Already stopped
    }
    
    shutdown_flag_ = true;
    cv_.notify_all(); // Wake up worker if it's waiting
    worker_thread_.join();
}

std::future<NetworkOutput> MCTSEvaluator::evaluateState(MCTSNode* node, std::unique_ptr<core::IGameState> state) {
    EvaluationRequest request(node, std::move(state));
    std::future<NetworkOutput> future = request.promise.get_future();
    
    request_queue_.enqueue(std::move(request));
    
    // Notify worker thread that a new request is available
    cv_.notify_one();
    
    return future;
}

size_t MCTSEvaluator::getQueueSize() const {
    return request_queue_.size_approx();
}

float MCTSEvaluator::getAverageBatchSize() const {
    size_t batches = total_batches_.load();
    if (batches == 0) return 0.0f;
    return static_cast<float>(cumulative_batch_size_.load()) / batches;
}

std::chrono::milliseconds MCTSEvaluator::getAverageBatchLatency() const {
    size_t batches = total_batches_.load();
    if (batches == 0) return std::chrono::milliseconds(0);
    return std::chrono::milliseconds(cumulative_batch_time_ms_.load() / batches);
}

size_t MCTSEvaluator::getTotalEvaluations() const {
    return total_evaluations_.load();
}

void MCTSEvaluator::processBatches() {
    while (!shutdown_flag_) {
        auto batch = collectBatch();
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch() {
    std::vector<EvaluationRequest> batch;
    batch.reserve(batch_size_);
    
    // Start with a timestamp for timeout calculation
    auto start_time = std::chrono::steady_clock::now();
    
    // Try to collect up to batch_size_ requests
    EvaluationRequest request(nullptr, nullptr);
    
    while (batch.size() < batch_size_) {
        // If the queue is empty, wait on the condition variable
        if (request_queue_.size_approx() == 0) {
            // Use condition variable to wait with timeout
            std::unique_lock<std::mutex> lock(cv_mutex_);
            if (cv_.wait_for(lock, timeout_, [this] { 
                    return request_queue_.size_approx() > 0 || shutdown_flag_; 
                })) {
                // Condition met (either a request is available or shutdown)
                if (shutdown_flag_) break;
            } else {
                // Timeout occurred
                break;
            }
        }
        
        // Try to dequeue a request
        if (request_queue_.try_dequeue(request)) {
            batch.push_back(std::move(request));
        }
        
        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > timeout_ && batch.size() > 0) {
            break;
        }
    }
    
    return batch;
}

void MCTSEvaluator::processBatch(std::vector<EvaluationRequest>& batch) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Prepare states for inference
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());
    
    for (auto& request : batch) {
        states.push_back(std::move(request.state));
    }
    
    // Run batch inference
    std::vector<NetworkOutput> outputs;
    try {
        outputs = inference_fn_(states);
    } catch (const std::exception& e) {
        // Handle inference error
        std::cerr << "Neural network inference error: " << e.what() << std::endl;
        
        // Provide default outputs
        outputs.resize(batch.size());
        for (auto& output : outputs) {
            output.value = 0.0f;
            output.policy.clear();
        }
    }
    
    // Distribute results to requesters via promises
    for (size_t i = 0; i < batch.size(); ++i) {
        if (i < outputs.size()) {
            batch[i].promise.set_value(std::move(outputs[i]));
        } else {
            // Provide default output if mismatch occurs
            NetworkOutput default_output;
            default_output.value = 0.0f;
            batch[i].promise.set_value(std::move(default_output));
        }
    }
    
    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(elapsed_ms, std::memory_order_relaxed);
}

} // namespace mcts
} // namespace alphazero