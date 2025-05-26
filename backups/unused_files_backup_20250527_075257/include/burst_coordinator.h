#ifndef ALPHAZERO_MCTS_BURST_COORDINATOR_H
#define ALPHAZERO_MCTS_BURST_COORDINATOR_H

#include <vector>
#include <memory>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "core/export_macros.h"
#include "mcts_node.h"
#include "evaluation_types.h"
#include "unified_inference_server.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Coordinated burst-mode leaf collection and evaluation
 * 
 * This coordinator implements a two-phase approach:
 * 1. Burst Collection Phase: Multiple threads simultaneously collect evaluation candidates
 * 2. Bulk Evaluation Phase: Submit all candidates together for optimal batching
 */
class ALPHAZERO_API BurstCoordinator {
public:
    // Priority levels for requests
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2,
        Critical = 3
    };
    
    struct BurstRequest {
        std::shared_ptr<MCTSNode> leaf;
        std::unique_ptr<core::IGameState> state;
        std::vector<std::shared_ptr<MCTSNode>> path;
        NetworkOutput result; // Direct result storage
        bool result_ready = false;
        
        // Added missing fields needed by other components
        std::shared_ptr<MCTSNode> node; // Node to update with the result
        Priority priority = Priority::Normal; // Priority for processing
        std::function<void(const NetworkOutput&)> callback; // Optional callback for async processing
        
        // Default constructor for vector.resize()
        BurstRequest() = default;
        
        BurstRequest(std::shared_ptr<MCTSNode> l, 
                    std::unique_ptr<core::IGameState> s,
                    std::vector<std::shared_ptr<MCTSNode>> p) 
            : leaf(std::move(l)), state(std::move(s)), path(std::move(p)), node(leaf) {}
        
        // Allow move operations
        BurstRequest(BurstRequest&&) = default;
        BurstRequest& operator=(BurstRequest&&) = default;
        
        // Disable copy operations
        BurstRequest(const BurstRequest&) = delete;
        BurstRequest& operator=(const BurstRequest&) = delete;
    };
    
    struct BurstConfig {
        size_t target_burst_size;
        size_t min_burst_size;
        std::chrono::milliseconds collection_timeout; // Short collection window
        std::chrono::milliseconds evaluation_timeout; // Longer evaluation timeout
        size_t max_parallel_threads;
        
        // Constructor with default values
        BurstConfig() 
            : target_burst_size(16)
            , min_burst_size(4)
            , collection_timeout(5)
            , evaluation_timeout(50)
            , max_parallel_threads(4)
        {}
    };

private:
    std::shared_ptr<UnifiedInferenceServer> inference_server_;
    BurstConfig config_;
    
    // Lock-free burst collection state
    std::vector<BurstRequest> current_burst_;
    std::atomic<size_t> burst_size_{0};  // Lock-free size tracking
    mutable std::mutex burst_finalization_mutex_;  // Only for finalization, made mutable for const methods
    std::condition_variable burst_ready_;
    std::atomic<bool> collection_active_{false};
    std::atomic<int> active_collectors_{0};
    
    // Timing coordination
    std::chrono::steady_clock::time_point burst_start_time_;
    std::atomic<bool> shutdown_{false};
    
    // Adaptive collection tracking
    std::atomic<int> consecutive_empty_collections_{0};
    std::atomic<size_t> successful_collections_{0};
    std::atomic<double> avg_efficiency_{0.0};
    static constexpr int MAX_EMPTY_COLLECTIONS = 8;  // OPTIMIZED: Increased from 2 to 8 for better persistence

public:
    explicit BurstCoordinator(std::shared_ptr<UnifiedInferenceServer> server, 
                             const BurstConfig& config = BurstConfig{})
        : inference_server_(std::move(server)), config_(config) {}
    
    /**
     * @brief Start a coordinated burst collection phase (simplified)
     * @param simulations_needed Number of simulations to perform
     * @param search_roots Vector of root nodes for parallel search
     * @return Vector of completed results
     */
    std::vector<NetworkOutput> startBurstCollection(
        int simulations_needed,
        const std::vector<std::shared_ptr<MCTSNode>>& search_roots);
    
    
    /**
     * @brief Process collected burst when ready
     */
    void processBurstWhenReady();
    
    /**
     * @brief Reset consecutive empty collections counter for new search
     */
    void resetEmptyCollectionCounter();
    
    /**
     * @brief Get burst statistics
     */
    struct BurstStats {
        size_t total_bursts = 0;
        size_t total_evaluations = 0;
        double average_burst_size = 0.0;
        std::chrono::microseconds average_collection_time{0};
        std::chrono::milliseconds average_evaluation_time{0};
    };
    
    BurstStats getStats() const;
    
    // Config management
    const BurstConfig& getConfig() const { return config_; }
    void updateConfig(const BurstConfig& config) { config_ = config; }
    
    // Check if all requests are complete
    bool allRequestsComplete() const { return !collection_active_.load(); }
    
    // Coordinated collection and evaluation (added for updated architecture)
    std::vector<NetworkOutput> collectAndEvaluate(
        const std::vector<BurstRequest>& requests, 
        size_t target_count);
        
    /**
     * @brief Submit a batch of requests for evaluation
     * 
     * @param requests Vector of burst requests to evaluate
     */
    void submitBurst(std::vector<BurstRequest>&& requests);
    
    // Efficiency statistics for optimization
    struct EfficiencyStats {
        double average_collection_efficiency = 0.0;
        double target_utilization_rate = 0.0;
        double average_burst_size = 0.0;
    };
    
    EfficiencyStats getEfficiencyStats() const {
        EfficiencyStats stats;
        stats.average_collection_efficiency = avg_efficiency_.load();
        stats.target_utilization_rate = successful_collections_.load() > 0 ? 1.0 : 0.0;
        stats.average_burst_size = successful_collections_.load() > 0 ? 
            static_cast<double>(burst_size_.load()) / successful_collections_.load() : 0.0;
        return stats;
    }
    
    void shutdown();

private:
    bool shouldProcessBurst() const;
    void submitBurstForEvaluation();
    void waitForCollectionComplete();
    
    // Helper method for leaf selection during burst collection
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>> 
    selectLeafForBurstEvaluation(std::shared_ptr<MCTSNode> root);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_BURST_COORDINATOR_H