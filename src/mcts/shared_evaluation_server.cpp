#include "mcts/shared_evaluation_server.h"
#include "utils/logger.h"
#include <chrono>
#include <algorithm>

namespace alphazero {
namespace mcts {

// Global instance
std::unique_ptr<SharedEvaluationServer> GlobalEvaluationServer::instance_;
std::mutex GlobalEvaluationServer::mutex_;

SharedEvaluationServer::SharedEvaluationServer(
    std::shared_ptr<nn::NeuralNetwork> network, 
    const Config& config)
    : config_(config), network_(network) {
    LOG_MCTS_INFO("Initializing SharedEvaluationServer with:");
    LOG_MCTS_INFO("  Max batch size: {}", config.max_batch_size);
    LOG_MCTS_INFO("  Min batch size: {}", config.min_batch_size);
    LOG_MCTS_INFO("  Batch timeout: {} ms", config.batch_timeout_ms);
    LOG_MCTS_INFO("  Worker threads: {}", config.num_worker_threads);
}

SharedEvaluationServer::~SharedEvaluationServer() {
    stop();
}

void SharedEvaluationServer::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }
    
    // Start worker threads
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&SharedEvaluationServer::workerLoop, this);
    }
    
    LOG_MCTS_INFO("SharedEvaluationServer started with {} workers", 
                  config_.num_worker_threads);
}

void SharedEvaluationServer::stop() {
    if (!running_.exchange(false)) {
        return;  // Already stopped
    }
    
    // Wake up all workers
    queue_cv_.notify_all();
    
    // Wait for workers to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    LOG_MCTS_INFO("SharedEvaluationServer stopped");
}

std::future<std::pair<std::vector<float>, float>> 
SharedEvaluationServer::evaluate(
    std::unique_ptr<core::IGameState> state,
    int game_id,
    int priority) {
    
    EvaluationRequest request;
    request.state = std::move(state);
    request.game_id = game_id;
    request.priority = priority;
    
    auto future = request.promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(request));
    }
    queue_cv_.notify_one();
    
    return future;
}

void SharedEvaluationServer::workerLoop() {
    while (running_) {
        std::vector<EvaluationRequest> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for requests or timeout
            auto timeout = std::chrono::milliseconds(
                static_cast<int>(config_.batch_timeout_ms));
            
            queue_cv_.wait_for(lock, timeout, [this] {
                return !request_queue_.empty() || !running_;
            });
            
            if (!running_ && request_queue_.empty()) {
                break;
            }
            
            // Collect batch
            size_t batch_size = 0;
            auto start_time = std::chrono::steady_clock::now();
            
            while (!request_queue_.empty() && batch_size < config_.max_batch_size) {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
                batch_size++;
                
                // Check if we have enough for min batch
                if (batch_size >= config_.min_batch_size) {
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    if (elapsed > std::chrono::milliseconds(
                            static_cast<int>(config_.batch_timeout_ms))) {
                        break;
                    }
                }
            }
        }
        
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
}

void SharedEvaluationServer::processBatch(std::vector<EvaluationRequest>& batch) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Convert states to network input format
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.reserve(batch.size());
        
        for (auto& request : batch) {
            states.push_back(std::move(request.state));
        }
        
        // Run neural network inference
        auto outputs = network_->inference(states);
        
        // Return results
        for (size_t i = 0; i < batch.size(); ++i) {
            try {
                batch[i].promise.set_value({outputs[i].policy, outputs[i].value});
            } catch (const std::exception& e) {
                LOG_MCTS_ERROR("Failed to set promise: {}", e.what());
            }
        }
        
        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        total_evaluations_.fetch_add(batch.size());
        total_batches_.fetch_add(1);
        total_batch_size_.fetch_add(batch.size());
        total_inference_time_ms_.fetch_add(static_cast<size_t>(inference_ms));
        
        // Log batch statistics periodically
        if (total_batches_ % 100 == 0) {
            auto stats = getStats();
            LOG_MCTS_INFO("Evaluation Server Stats: {} evals, {} batches, "
                         "avg batch: {:.1f}, avg time: {:.2f}ms",
                         stats.total_evaluations, stats.total_batches,
                         stats.avg_batch_size, stats.avg_inference_time_ms);
        }
        
    } catch (const std::exception& e) {
        LOG_MCTS_ERROR("Batch processing failed: {}", e.what());
        // Set error for all promises
        for (auto& request : batch) {
            try {
                request.promise.set_exception(std::current_exception());
            } catch (...) {
                // Ignore if promise already set
            }
        }
    }
}

SharedEvaluationServer::Stats SharedEvaluationServer::getStats() const {
    Stats stats;
    stats.total_evaluations = total_evaluations_.load();
    stats.total_batches = total_batches_.load();
    
    if (stats.total_batches > 0) {
        stats.avg_batch_size = static_cast<double>(total_batch_size_.load()) / 
                              static_cast<double>(stats.total_batches);
        stats.avg_inference_time_ms = static_cast<double>(total_inference_time_ms_.load()) / 
                                     static_cast<double>(stats.total_batches);
    } else {
        stats.avg_batch_size = 0.0;
        stats.avg_inference_time_ms = 0.0;
    }
    
    {
        std::lock_guard<std::mutex> lock(
            const_cast<std::mutex&>(queue_mutex_));
        stats.pending_requests = request_queue_.size();
    }
    
    stats.avg_wait_time_ms = 0.0;  // TODO: Implement wait time tracking
    
    return stats;
}

// Global server implementation
void GlobalEvaluationServer::initialize(
    std::shared_ptr<nn::NeuralNetwork> network,
    const SharedEvaluationServer::Config& config) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) {
        LOG_MCTS_WARN("GlobalEvaluationServer already initialized");
        return;
    }
    
    instance_ = std::make_unique<SharedEvaluationServer>(network, config);
    instance_->start();
    LOG_MCTS_INFO("GlobalEvaluationServer initialized and started");
}

SharedEvaluationServer* GlobalEvaluationServer::get() {
    std::lock_guard<std::mutex> lock(mutex_);
    return instance_.get();
}

void GlobalEvaluationServer::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) {
        instance_->stop();
        instance_.reset();
        LOG_MCTS_INFO("GlobalEvaluationServer shutdown");
    }
}

}  // namespace mcts
}  // namespace alphazero