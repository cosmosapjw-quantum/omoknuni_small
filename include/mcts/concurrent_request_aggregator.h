#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <array>
#include "moodycamel/concurrentqueue.h"
#include "core/igamestate.h"
#include "mcts/evaluation_types.h"
#include "nn/neural_network.h"

namespace alphazero {
namespace mcts {

/**
 * CONCURRENT REQUEST AGGREGATOR
 * 
 * This class solves the fundamental batching problem:
 * - Multiple OpenMP threads submit individual evaluateBatch(1 state) calls
 * - This aggregator collects these individual calls into larger batches
 * - Uses lock-free atomic coordination to enable true concurrent operation
 * - Maximizes GPU utilization through optimal batch formation
 */
class ConcurrentRequestAggregator {
public:
    struct AggregatorConfig {
        size_t target_batch_size = 32;           // Target batch size for GPU optimization
        size_t max_batch_size = 64;              // Maximum batch size limit
        std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(25);  // Wait time for batch formation
        std::chrono::milliseconds max_wait_time = std::chrono::milliseconds(50);  // Maximum wait before processing
        size_t num_aggregator_threads = 2;       // Number of aggregator worker threads
    };

    struct PendingRequest {
        std::vector<std::unique_ptr<core::IGameState>> states;
        std::promise<std::vector<NetworkOutput>> promise;
        std::chrono::steady_clock::time_point submit_time;
        uint64_t request_id;
        
        PendingRequest() = default;
        PendingRequest(std::vector<std::unique_ptr<core::IGameState>>&& s, uint64_t id) 
            : states(std::move(s)), submit_time(std::chrono::steady_clock::now()), request_id(id) {}
    };

    static AggregatorConfig default_config() {
        return AggregatorConfig{};
    }

    ConcurrentRequestAggregator(std::shared_ptr<nn::NeuralNetwork> neural_net, 
                                const AggregatorConfig& config = default_config());
    ~ConcurrentRequestAggregator();

    // Main interface: Submit requests for batched evaluation
    std::vector<NetworkOutput> evaluateBatch(std::vector<std::unique_ptr<core::IGameState>> states);

    // Control methods
    void start();
    void stop();

    struct AggregatorStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> total_batches_processed{0};
        std::atomic<uint64_t> total_states_evaluated{0};
        std::atomic<uint64_t> requests_dropped{0};
        std::atomic<double> average_batch_size{0.0};
        std::atomic<double> average_wait_time_ms{0.0};
    };

    void getStats(AggregatorStats& stats_out) const {
        stats_out.total_requests.store(stats_.total_requests.load());
        stats_out.total_batches_processed.store(stats_.total_batches_processed.load());
        stats_out.total_states_evaluated.store(stats_.total_states_evaluated.load());
        stats_out.requests_dropped.store(stats_.requests_dropped.load());
        stats_out.average_batch_size.store(stats_.average_batch_size.load());
        stats_out.average_wait_time_ms.store(stats_.average_wait_time_ms.load());
    }

private:
    // Core components
    std::shared_ptr<nn::NeuralNetwork> neural_network_;
    AggregatorConfig config_;
    AggregatorStats stats_;

    // Atomic coordination for lock-free operation
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{1};
    std::atomic<size_t> pending_request_count_{0};
    std::atomic<size_t> active_batch_count_{0};

    // Lock-free concurrent queue for request submission
    moodycamel::ConcurrentQueue<PendingRequest> request_queue_;

    // Worker threads for batch processing
    std::vector<std::thread> aggregator_threads_;

    // Internal methods
    void aggregatorWorkerLoop();
    void processBatch(std::vector<PendingRequest>& batch);
    bool shouldProcessBatch(const std::vector<PendingRequest>& batch) const;
    size_t collectRequests(std::vector<PendingRequest>& batch);
};

} // namespace mcts
} // namespace alphazero