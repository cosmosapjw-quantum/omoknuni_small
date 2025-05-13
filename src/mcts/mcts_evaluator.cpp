// src/mcts/mcts_evaluator.cpp
#include "mcts/mcts_evaluator.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>
#include "utils/debug_monitor.h"
#include "utils/memory_debug.h"

// Use shortened namespace for debug functions
namespace ad = alphazero::debug;

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
    DEBUG_THREAD_STATUS("evaluator_thread_start", "Starting neural network evaluator thread");

    // Performance stats tracking with minimal variables
    size_t total_batch_count = 0;
    size_t total_batch_size = 0;

    while (!shutdown_flag_) {
        // Collect a batch of states - without detailed status updates
        DEBUG_THREAD_STATUS("evaluator_collecting", "");
        auto batch = collectBatch();

        if (!batch.empty()) {
            // Process the batch
            DEBUG_THREAD_STATUS("evaluator_processing_batch", "Size: " + std::to_string(batch.size()));
            processBatch(batch);

            // Record simple batch statistics without verbose output
            total_batch_count++;
            total_batch_size += batch.size();

            // Only log basic stats occasionally to reduce overhead
            if (total_batch_count % 50 == 0) {
                float avg_batch_size = total_batch_size / static_cast<float>(total_batch_count);
                debug::SystemMonitor::instance().recordResourceUsage("AvgBatchSize", avg_batch_size);
            }
        } else {
            // To avoid tight loops when queue is empty
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    DEBUG_THREAD_STATUS("evaluator_thread_exit", "Neural network evaluator thread exiting");
}

std::vector<EvaluationRequest> MCTSEvaluator::collectBatch() {
    // Use timing without verbose logging
    debug::ScopedTimer timer("MCTSEvaluator::collectBatch");

    std::vector<EvaluationRequest> batch;
    batch.reserve(batch_size_);

    // Start with a timestamp for timeout calculation
    auto start_time = std::chrono::steady_clock::now();

    // Try to collect up to batch_size_ requests
    EvaluationRequest request(nullptr, nullptr);

    // First quick pass to collect available items
    for (size_t i = 0; i < batch_size_; ++i) {
        if (request_queue_.try_dequeue(request)) {
            batch.push_back(std::move(request));
        } else {
            break; // No more available immediately
        }
    }

    // If we already have a full batch, return it
    if (batch.size() >= batch_size_) {
        // Minimal logging
        DEBUG_THREAD_STATUS("evaluator_batch_full", "Collected full batch immediately");
        return batch;
    }

    // If we have some items but not a full batch, wait with proper timeout
    if (batch.size() > 0) {
        // We have some items - wait for more with a strict timeout
        auto wait_end = start_time + timeout_;

        while (batch.size() < batch_size_) {
            if (std::chrono::steady_clock::now() >= wait_end) {
                break; // Timeout expired
            }

            if (request_queue_.try_dequeue(request)) {
                batch.push_back(std::move(request));
            } else {
                // Short sleep to avoid tight loop
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } else if (request_queue_.size_approx() == 0) {
        // Empty queue - use condition variable with proper timeout
        std::unique_lock<std::mutex> lock(cv_mutex_);

        // Wait for notification with timeout
        bool notified = cv_.wait_for(lock, timeout_, [this] {
            return request_queue_.size_approx() > 0 || shutdown_flag_;
        });

        // Check for shutdown
        if (shutdown_flag_) {
            return batch;
        }

        // Try one more time after wait
        while (batch.size() < batch_size_) {
            if (request_queue_.try_dequeue(request)) {
                batch.push_back(std::move(request));
            } else {
                break;
            }
        }
    }

    // Record metrics without excessive logging
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Only record metrics, no console output
    if (batch.size() > 0) {
        float batch_efficiency = static_cast<float>(batch.size()) / batch_size_ * 100.0f;
        debug::SystemMonitor::instance().recordResourceUsage("BatchEfficiency", batch_efficiency);
        debug::SystemMonitor::instance().recordTiming("BatchCollectionTime", collection_time);
    }

    return batch;
}

void MCTSEvaluator::processBatch(std::vector<EvaluationRequest>& batch) {
    debug::ScopedTimer timer("MCTSEvaluator::processBatch");
    DEBUG_THREAD_STATUS("evaluator_inference_start", "Starting neural network inference for batch of " + std::to_string(batch.size()));

    auto start_time = std::chrono::steady_clock::now();

    // Prepare states for inference - minimize logging
    std::vector<std::unique_ptr<core::IGameState>> states;
    states.reserve(batch.size());

    {
        debug::ScopedTimer prep_timer("MCTSEvaluator::prepareStates");

        for (auto& request : batch) {
            if (!request.state) {
                std::cerr << "ERROR: Null game state in batch request!" << std::endl;
                continue;
            }
            states.push_back(std::move(request.state));
        }
    }

    // Run batch inference with minimal logging
    std::vector<NetworkOutput> outputs;
    {
        debug::ScopedTimer nn_timer("MCTSEvaluator::runInference");

        try {
            outputs = inference_fn_(states);

            if (outputs.size() != batch.size()) {
                std::cerr << "Warning: neural network returned " << outputs.size()
                          << " outputs for " << batch.size() << " inputs!" << std::endl;
            }

        } catch (const std::bad_alloc& e) {
            // Special handling for memory errors
            std::cerr << "MEMORY ERROR during neural network inference: " << e.what() << std::endl;

            // Provide default outputs
            outputs.resize(batch.size());
            for (auto& output : outputs) {
                output.value = 0.0f;
                output.policy.clear(); // Use minimal memory
            }
        } catch (const std::exception& e) {
            // Handle other inference errors
            std::cerr << "Neural network inference error: " << e.what() << std::endl;

            // Provide default outputs
            outputs.resize(batch.size());
            for (auto& output : outputs) {
                output.value = 0.0f;
                output.policy.clear();
            }
        }
    }

    // Distribute results to requesters via promises
    {
        debug::ScopedTimer dist_timer("MCTSEvaluator::distributeResults");

        for (size_t i = 0; i < batch.size(); ++i) {
            try {
                if (i < outputs.size()) {
                    batch[i].promise.set_value(std::move(outputs[i]));
                } else {
                    // Provide default output if mismatch occurs
                    NetworkOutput default_output;
                    default_output.value = 0.0f;
                    batch[i].promise.set_value(std::move(default_output));
                }
            } catch (const std::exception& e) {
                std::cerr << "Error setting promise result: " << e.what() << std::endl;
            }
        }
    }

    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Record only essential metrics
    debug::SystemMonitor::instance().recordTiming("BatchProcessingTime", elapsed_ms);
    debug::SystemMonitor::instance().recordResourceUsage("BatchSize", batch.size());

    // Calculate per-state processing time
    float ms_per_state = batch.size() > 0 ? elapsed_ms / static_cast<float>(batch.size()) : 0.0f;
    debug::SystemMonitor::instance().recordTiming("MsPerState", ms_per_state);

    // Update counters
    int batch_id = total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
    cumulative_batch_time_ms_.fetch_add(elapsed_ms, std::memory_order_relaxed);

    DEBUG_THREAD_STATUS("evaluator_inference_complete",
                       "Finished neural network inference for batch of " + std::to_string(batch.size()));
}

} // namespace mcts
} // namespace alphazero