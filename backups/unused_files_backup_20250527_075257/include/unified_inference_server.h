#ifndef ALPHAZERO_MCTS_UNIFIED_INFERENCE_SERVER_H
#define ALPHAZERO_MCTS_UNIFIED_INFERENCE_SERVER_H

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <unordered_map>
#include "mcts/concurrent_request_aggregator.h"
#include "mcts/adaptive_batch_sizer.h"
#include "mcts/dynamic_batch_adjuster.h"

#include "mcts/evaluation_types.h"
#include "mcts/mcts_node.h"
#include "core/igamestate.h"
#include "core/export_macros.h"
#include "nn/neural_network.h"

#include <moodycamel/concurrentqueue.h>

namespace alphazero {
namespace mcts {

// Forward declarations
class CUDAStreamOptimizer;

/**
 * @brief Unified Inference Server - Central neural network processing hub
 * 
 * This class implements a dedicated inference server that eliminates the deadlock
 * patterns and fragmented batch formation of the previous architecture. It uses:
 * 
 * 1. Single-threaded inference coordination to eliminate deadlocks
 * 2. Aggressive batch accumulation with smart timeout management
 * 3. Virtual loss integration to prevent thread collisions
 * 4. Streamlined memory management with object pooling
 * 5. Lock-free producer-consumer queues for maximum throughput
 */
class ALPHAZERO_API UnifiedInferenceServer {
public:
    // Request structure for neural network inference
    struct InferenceRequest {
        uint64_t request_id;
        std::shared_ptr<MCTSNode> node;
        std::shared_ptr<core::IGameState> state;
        std::vector<std::shared_ptr<MCTSNode>> path;  // For backpropagation
        std::promise<NetworkOutput> result_promise;
        std::chrono::steady_clock::time_point submitted_time;
        int virtual_loss_applied = 0;  // Track virtual loss for this request
        
        // Move constructor
        InferenceRequest(InferenceRequest&& other) noexcept
            : request_id(other.request_id),
              node(std::move(other.node)),
              state(std::move(other.state)),
              path(std::move(other.path)),
              result_promise(std::move(other.result_promise)),
              submitted_time(other.submitted_time),
              virtual_loss_applied(other.virtual_loss_applied) {}
        
        // Move assignment
        InferenceRequest& operator=(InferenceRequest&& other) noexcept {
            if (this != &other) {
                request_id = other.request_id;
                node = std::move(other.node);
                state = std::move(other.state);
                path = std::move(other.path);
                result_promise = std::move(other.result_promise);
                submitted_time = other.submitted_time;
                virtual_loss_applied = other.virtual_loss_applied;
            }
            return *this;
        }
        
        // Delete copy operations
        InferenceRequest(const InferenceRequest&) = delete;
        InferenceRequest& operator=(const InferenceRequest&) = delete;
        
        // Default constructor for queue operations
        InferenceRequest() = default;
    };

    // Configuration for the inference server
    struct ServerConfig {
        size_t target_batch_size;           // Target batch size for optimal GPU utilization
        size_t min_batch_size;               // Minimum batch size to process
        size_t max_batch_size;             // Maximum batch size limit
        
        std::chrono::milliseconds max_batch_wait;  // Max wait for batch formation
        std::chrono::milliseconds min_batch_wait;   // Min wait before processing small batches
        
        int virtual_loss_value;              // Virtual loss value to apply during batching
        size_t max_pending_requests;       // Maximum pending requests before backpressure
        
        size_t num_worker_threads;           // Number of inference worker threads
        bool enable_request_coalescing;   // Enable request coalescing for efficiency
        bool enable_priority_processing;  // Enable priority-based request processing
        
        // Constructor with default values optimized for RTX 3060 Ti and parallel MCTS
        ServerConfig() 
            : target_batch_size(32)  // Increased for RTX 3060 Ti optimal throughput
            , min_batch_size(8)      // Increased to match parallel thread count
            , max_batch_size(64)     // Increased for better GPU utilization
            , max_batch_wait(std::chrono::milliseconds(50))  // CRITICAL: Increased from 10ms to 50ms for better batching
            , min_batch_wait(std::chrono::milliseconds(20))  // CRITICAL: Increased from 2ms to 20ms for accumulation
            , virtual_loss_value(3)
            , max_pending_requests(256)  // Increased for higher parallelism
            , num_worker_threads(2)      // Increased for parallel processing
            , enable_request_coalescing(true)
            , enable_priority_processing(true)
        {}
    };

    // Statistics for monitoring
    struct ServerStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> total_batches{0};
        std::atomic<uint64_t> total_evaluations{0};
        std::atomic<uint64_t> dropped_requests{0};
        
        std::atomic<size_t> current_queue_size{0};
        std::atomic<size_t> peak_queue_size{0};
        
        std::atomic<uint64_t> cumulative_batch_size{0};
        std::atomic<uint64_t> cumulative_batch_time_ms{0};
        
        std::atomic<uint64_t> virtual_loss_applications{0};
        std::atomic<uint64_t> virtual_loss_reversals{0};
        
        float getAverageBatchSize() const {
            uint64_t batches = total_batches.load();
            return batches > 0 ? static_cast<float>(cumulative_batch_size.load()) / batches : 0.0f;
        }
        
        // Default constructor
        ServerStats() = default;
        
        // Delete copy constructor and assignment operator (atomic members are not copyable)
        ServerStats(const ServerStats&) = delete;
        ServerStats& operator=(const ServerStats&) = delete;
        
        // Move operations are also problematic with atomics, so we'll implement manual copying
        ServerStats(ServerStats&&) = delete;
        ServerStats& operator=(ServerStats&&) = delete;
        
        float getAverageBatchLatency() const {
            uint64_t batches = total_batches.load();
            return batches > 0 ? static_cast<float>(cumulative_batch_time_ms.load()) / batches : 0.0f;
        }
    };

private:
    // Core components
    std::shared_ptr<nn::NeuralNetwork> neural_network_;
    std::unique_ptr<ConcurrentRequestAggregator> request_aggregator_;
    std::unique_ptr<::mcts::AdaptiveBatchSizer> adaptive_batch_sizer_;
    std::unique_ptr<CUDAStreamOptimizer> cuda_stream_optimizer_;
    ServerConfig config_;
    ServerStats stats_;
    
    // Control flags
    std::atomic<bool> shutdown_flag_{false};
    std::atomic<bool> server_running_{false};
    
    // Request processing
    std::atomic<uint64_t> next_request_id_{1};
    moodycamel::ConcurrentQueue<InferenceRequest> request_queue_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::thread batch_coordinator_thread_;
    
    // LOCK-FREE ATOMIC COORDINATION: Enable true concurrent operation
    std::atomic<size_t> active_request_count_{0};      // Track requests in flight
    std::atomic<size_t> completed_batch_count_{0};     // Track completed batches
    std::atomic<bool> batch_processing_active_{false}; // Signal batch processing state
    std::atomic<size_t> current_batch_size_{0};        // Current batch size being processed
    
    // Reduced synchronization (only for critical sections)
    mutable std::mutex stats_mutex_;
    std::condition_variable server_cv_;
    mutable std::mutex server_mutex_;
    
    // Enhanced batch formation state with atomic coordination
    struct BatchState {
        std::vector<InferenceRequest> pending_requests;
        std::chrono::steady_clock::time_point batch_start_time;
        std::atomic<bool> batch_forming{false};        // Atomic flag for coordination
        std::atomic<size_t> target_size{0};            // Target batch size
        
        void reset() {
            pending_requests.clear();
            batch_start_time = std::chrono::steady_clock::now();
            batch_forming = false;
        }
    };
    
    BatchState current_batch_;
    mutable std::mutex batch_mutex_;

    // Core processing methods
    void batchCoordinatorLoop();
    void inferenceWorkerLoop();
    
    // Batch management
    bool shouldProcessBatch(const BatchState& batch) const;
    std::vector<InferenceRequest> extractBatchForProcessing();
    void processInferenceBatch(std::vector<InferenceRequest>& batch);
    
    // Virtual loss management
    void applyVirtualLoss(InferenceRequest& request);
    void revertVirtualLoss(const InferenceRequest& request);
    
    // Utility methods
    void updateQueueStats();
    void cleanupExpiredRequests();

public:
    /**
     * @brief Construct a new Unified Inference Server
     * 
     * @param neural_network Shared pointer to the neural network
     * @param config Server configuration (optional)
     */
    explicit UnifiedInferenceServer(std::shared_ptr<nn::NeuralNetwork> neural_network,
                                   const ServerConfig& config = ServerConfig());
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~UnifiedInferenceServer();
    
    // Server lifecycle management
    void start();
    void stop();
    bool isRunning() const { return server_running_.load(std::memory_order_acquire); }
    
    /**
     * @brief Submit an inference request
     * 
     * @param node The MCTS node requiring evaluation
     * @param state The game state to evaluate
     * @param path The path from root to this node (for backpropagation)
     * @return Future containing the network output
     */
    std::future<NetworkOutput> submitRequest(std::shared_ptr<MCTSNode> node,
                                           std::shared_ptr<core::IGameState> state,
                                           std::vector<std::shared_ptr<MCTSNode>> path = {});
    
    /**
     * @brief Submit multiple requests in bulk
     * 
     * @param requests Vector of request data
     * @return Vector of futures for the results
     */
    std::vector<std::future<NetworkOutput>> submitBulkRequests(
        const std::vector<std::tuple<std::shared_ptr<MCTSNode>, 
                                   std::shared_ptr<core::IGameState>, 
                                   std::vector<std::shared_ptr<MCTSNode>>>>& requests);
    
    // Configuration and monitoring
    void updateConfig(const ServerConfig& config);
    ServerConfig getConfig() const;
    // Simple stats structure that can be copied (non-atomic)
    struct ServerStatsSnapshot {
        uint64_t total_requests;
        uint64_t total_batches;
        uint64_t total_evaluations;
        uint64_t dropped_requests;
        size_t current_queue_size;
        size_t peak_queue_size;
        uint64_t cumulative_batch_size;
        uint64_t cumulative_batch_time_ms;
        uint64_t virtual_loss_applications;
        uint64_t virtual_loss_reversals;
        
        float getAverageBatchSize() const {
            return total_batches > 0 ? static_cast<float>(cumulative_batch_size) / total_batches : 0.0f;
        }
        
        float getAverageBatchLatency() const {
            return total_batches > 0 ? static_cast<float>(cumulative_batch_time_ms) / total_batches : 0.0f;
        }
    };
    
    ServerStatsSnapshot getStats() const;
    
    // Utility methods
    void clearPendingRequests();
    size_t getPendingRequestCount() const;
    void forceProcessPendingBatch();
    
    /**
     * @brief Evaluate a batch of game states directly
     * 
     * @param states Vector of game states to evaluate
     * @return Vector of neural network outputs
     */
    std::vector<NetworkOutput> evaluateBatch(const std::vector<std::unique_ptr<core::IGameState>>& states);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_UNIFIED_INFERENCE_SERVER_H