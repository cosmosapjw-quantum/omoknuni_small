#ifndef ALPHAZERO_MCTS_OPTIMIZED_EXTERNAL_QUEUE_PROCESSOR_H
#define ALPHAZERO_MCTS_OPTIMIZED_EXTERNAL_QUEUE_PROCESSOR_H

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include "core/export_macros.h"
#include "moodycamel/concurrentqueue.h"
#include "nn/neural_network.h"
#include "mcts/evaluation_types.h"

namespace alphazero {
namespace mcts {

// Forward declarations
struct PendingEvaluation;

struct ALPHAZERO_API OptimizedQueueConfig {
    // Target batch size for optimal performance
    size_t batch_size = 256;
    
    // Minimum viable batch size for processing
    size_t min_batch_size = 64;
    
    // Maximum time to wait for batch formation
    std::chrono::milliseconds max_wait_time = std::chrono::milliseconds(10);
};

struct ALPHAZERO_API OptimizedQueueStats {
    // Total batches processed
    size_t total_batches_processed = 0;
    
    // Total evaluations processed
    size_t total_evaluations_processed = 0;
    
    // Average batch size
    float average_batch_size = 0.0f;
};

/**
 * @brief Optimized external queue processor for efficient neural network inference
 * 
 * Collects and processes batches from an external queue, optimizing
 * batch size and processing timing for maximum throughput.
 */
class ALPHAZERO_API OptimizedExternalQueueProcessor {
public:
    /**
     * @brief Constructor with configuration and neural network
     * 
     * @param config Queue processor configuration
     * @param neural_network Neural network to use for inference
     */
    OptimizedExternalQueueProcessor(
        const OptimizedQueueConfig& config,
        std::shared_ptr<nn::NeuralNetwork> neural_network);
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~OptimizedExternalQueueProcessor();
    
    /**
     * @brief Set the input and output queues
     * 
     * @param input_queue Queue for incoming evaluation requests
     * @param output_queue Queue for outgoing evaluation results
     */
    void setQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* input_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* output_queue);
    
    /**
     * @brief Get the processor statistics
     * 
     * @return Current processor statistics
     */
    OptimizedQueueStats getStats() const;
    
    /**
     * @brief Shutdown the processor and join threads
     */
    void shutdown();
    
private:
    // Configuration
    OptimizedQueueConfig config_;
    
    // Neural network
    std::shared_ptr<nn::NeuralNetwork> neural_network_;
    
    // Thread control
    std::atomic<bool> shutdown_;
    std::thread processing_thread_;
    
    // Statistics
    std::atomic<size_t> total_batches_processed_{0};
    std::atomic<size_t> total_evaluations_processed_{0};
    
    // Queues
    moodycamel::ConcurrentQueue<PendingEvaluation>* input_queue_ = nullptr;
    moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* output_queue_ = nullptr;
    
    // Worker methods
    void processingLoop();
    void collectBatch(std::vector<PendingEvaluation>& batch);
    void processBatch(const std::vector<PendingEvaluation>& batch);
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_OPTIMIZED_EXTERNAL_QUEUE_PROCESSOR_H