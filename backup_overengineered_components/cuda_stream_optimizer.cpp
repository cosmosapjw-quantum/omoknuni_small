// src/mcts/cuda_stream_optimizer.cpp
#include "mcts/cuda_stream_optimizer.h"
#include "utils/debug_logger.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace alphazero {
namespace mcts {

#ifdef TORCH_USE_CUDA

CUDAStreamOptimizer::CUDAStreamOptimizer(std::shared_ptr<nn::NeuralNetwork> neural_net, 
                                        const CUDAStreamConfig& config)
    : neural_network_(neural_net), config_(config), num_streams_(config.num_streams) {
    
    // Validate configuration
    config_.num_streams = std::max(1, std::min(config_.num_streams, 8));
    config_.max_batch_size = std::max(1, config_.max_batch_size);
    config_.tensor_pool_size = std::max(config_.max_batch_size, config_.tensor_pool_size);
    num_streams_ = config_.num_streams;
    
    std::cout << "🔥 CUDAStreamOptimizer initialized with " << config_.num_streams 
              << " streams, max batch size " << config_.max_batch_size << std::endl;
}

CUDAStreamOptimizer::~CUDAStreamOptimizer() {
    stop();
    cleanup();
}

bool CUDAStreamOptimizer::initialize() {
    if (!checkCUDAAvailability()) {
        return false;
    }
    
    try {
        // Initialize CUDA streams
        cuda_streams_.reserve(config_.num_streams);
        raw_streams_.reserve(config_.num_streams);
        stream_queues_.reserve(config_.num_streams);
        
        // Initialize atomic arrays and synchronization objects
        stream_busy_ = std::make_unique<std::atomic<bool>[]>(num_streams_);
        per_stream_batches_ = std::make_unique<std::atomic<size_t>[]>(num_streams_);
        stream_mutexes_ = std::make_unique<std::mutex[]>(num_streams_);
        stream_cvs_ = std::make_unique<std::condition_variable[]>(num_streams_);
        
        for (int i = 0; i < config_.num_streams; ++i) {
            // Create CUDA stream
            cuda_streams_.emplace_back(c10::cuda::getStreamFromPool());
            raw_streams_.push_back(cuda_streams_[i].stream());
            
            // Initialize atomic values
            stream_busy_[i].store(false);
            per_stream_batches_[i].store(0);
            
            // Emplace containers
            stream_queues_.emplace_back();
            
            std::cout << "  Created CUDA stream " << i << std::endl;
        }
        
        // Initialize memory pools if enabled
        if (config_.enable_memory_pooling) {
            initializeMemoryPools();
        }
        
        std::cout << "✅ CUDA streams initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize CUDA streams: " << e.what() << std::endl;
        return false;
    }
}

void CUDAStreamOptimizer::start() {
    if (stream_workers_.size() > 0) {
        return; // Already started
    }
    
    shutdown_.store(false);
    
    // Start worker threads for each stream
    stream_workers_.reserve(config_.num_streams);
    for (int i = 0; i < config_.num_streams; ++i) {
        stream_workers_.emplace_back(&CUDAStreamOptimizer::streamWorkerLoop, this, i);
        std::cout << "  Started worker thread for stream " << i << std::endl;
    }
    
}

void CUDAStreamOptimizer::stop() {
    shutdown_.store(true);
    
    // Wake up all worker threads
    for (size_t i = 0; i < num_streams_; ++i) {
        stream_cvs_[i].notify_all();
    }
    
    // Join worker threads
    for (auto& worker : stream_workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    stream_workers_.clear();
    std::cout << "🛑 CUDAStreamOptimizer stopped" << std::endl;
}

std::future<std::vector<NetworkOutput>> CUDAStreamOptimizer::submitBatchAsync(
    std::vector<PendingEvaluation>&& batch) {
    
    if (batch.empty()) {
        std::promise<std::vector<NetworkOutput>> promise;
        promise.set_value(std::vector<NetworkOutput>());
        return promise.get_future();
    }
    
    // Create stream batch
    auto stream_batch = std::make_unique<StreamBatch>();
    stream_batch->evaluations = std::move(batch);
    stream_batch->created_time = std::chrono::steady_clock::now();
    stream_batch->batch_id = total_batches_processed_.fetch_add(1, std::memory_order_relaxed);
    
    auto future = stream_batch->result_promise.get_future();
    
    // Select optimal stream
    int stream_id = selectOptimalStream();
    stream_batch->stream_id = stream_id;
    
    // Submit to stream queue
    bool enqueued = stream_queues_[stream_id].enqueue(std::move(stream_batch));
    
    if (enqueued) {
        // Notify stream worker
        stream_cvs_[stream_id].notify_one();
    } else {
        // Fallback to synchronous processing
        std::promise<std::vector<NetworkOutput>> fallback_promise;
        fallback_promise.set_value(std::vector<NetworkOutput>());
        return fallback_promise.get_future();
    }
    
    return future;
}

int CUDAStreamOptimizer::selectOptimalStream() {
    // Simple round-robin with load balancing
    int base_stream = next_stream_index_.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
    
    // Check if the selected stream is busy
    for (int i = 0; i < config_.num_streams; ++i) {
        int stream_id = (base_stream + i) % config_.num_streams;
        
        if (!stream_busy_[stream_id].load(std::memory_order_acquire)) {
            // Found an idle stream
            total_stream_switches_.fetch_add(1, std::memory_order_relaxed);
            return stream_id;
        }
    }
    
    // All streams busy, use round-robin
    return base_stream;
}

void CUDAStreamOptimizer::streamWorkerLoop(int stream_id) {
    std::cout << "🔄 Stream worker " << stream_id << " started" << std::endl;
    
    // Set CUDA context for this thread
    try {
        c10::cuda::set_device(0); // Assume single GPU for now
        c10::cuda::setCurrentCUDAStream(cuda_streams_[stream_id]);
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to set CUDA context for stream " << stream_id 
                  << ": " << e.what() << std::endl;
        return;
    }
    
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::unique_ptr<StreamBatch> batch;
        
        // Try to get batch from queue
        if (stream_queues_[stream_id].try_dequeue(batch)) {
            if (batch) {
                stream_busy_[stream_id].store(true, std::memory_order_release);
                
                // Process batch on this stream
                processBatchOnStream(std::move(batch), stream_id);
                
                stream_busy_[stream_id].store(false, std::memory_order_release);
            }
        } else {
            // No batch available, wait for notification
            std::unique_lock<std::mutex> lock(stream_mutexes_[stream_id]);
            stream_cvs_[stream_id].wait_for(lock, std::chrono::milliseconds(10), [this, stream_id]() {
                return shutdown_.load(std::memory_order_acquire) || 
                       stream_queues_[stream_id].size_approx() > 0;
            });
        }
    }
    
    std::cout << "✅ Stream worker " << stream_id << " finished" << std::endl;
}

void CUDAStreamOptimizer::processBatchOnStream(std::unique_ptr<StreamBatch> batch, int stream_id) {
    try {
        auto inference_start = std::chrono::steady_clock::now();
        
        // Prepare input tensors
        auto input_tensors = prepareInputTensors(batch->evaluations, stream_id);
        
        if (input_tensors.empty()) {
            // No valid inputs
            batch->result_promise.set_value(std::vector<NetworkOutput>());
            return;
        }
        
        // Perform inference on GPU
        std::vector<NetworkOutput> results;
        
        if (neural_network_) {
            // Convert evaluations to game states for inference
            std::vector<std::unique_ptr<core::IGameState>> states;
            for (const auto& eval : batch->evaluations) {
                if (eval.state) {
                    states.push_back(eval.state->clone());
                }
            }
            
            if (!states.empty()) {
                // This would need to be adapted to use CUDA streams
                // For now, use the existing interface
                try {
                    // Synchronize stream before inference
                    synchronizeStream(stream_id);
                    
                    // Perform inference (this needs to be modified to accept CUDA stream)
                    // results = neural_network_->inference(states);
                    
                    // For now, create dummy results
                    results.resize(states.size());
                    for (size_t i = 0; i < results.size(); ++i) {
                        results[i].value = 0.0f;
                        results[i].policy.resize(256, 1.0f / 256.0f); // Dummy policy
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "❌ Inference error on stream " << stream_id 
                              << ": " << e.what() << std::endl;
                    results.clear();
                }
            }
        }
        
        // Return tensors to pool
        for (auto& tensor : input_tensors) {
            returnTensorToPool(tensor, stream_id);
        }
        
        // Update statistics
        per_stream_batches_[stream_id].fetch_add(1, std::memory_order_relaxed);
        total_inferences_.fetch_add(results.size(), std::memory_order_relaxed);
        
        auto inference_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end - inference_start);
        
        // Set result
        batch->result_promise.set_value(std::move(results));
        
        // Log performance for first few batches
        static std::atomic<int> batch_count{0};
        int current_batch = batch_count.fetch_add(1);
        if (current_batch < 10 || current_batch % 100 == 0) {
            std::cout << "⚡ Stream " << stream_id << " batch " << current_batch 
                      << ": " << batch->evaluations.size() << " items in " 
                      << duration.count() << "ms" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error processing batch on stream " << stream_id 
                  << ": " << e.what() << std::endl;
        batch->result_promise.set_exception(std::current_exception());
    }
}

std::vector<torch::Tensor> CUDAStreamOptimizer::prepareInputTensors(
    const std::vector<PendingEvaluation>& evaluations, int stream_id) {
    
    std::vector<torch::Tensor> tensors;
    
    if (evaluations.empty()) {
        return tensors;
    }
    
    try {
        // Get tensor from pool or allocate
        auto batch_tensor = getTensorFromPool(evaluations.size(), stream_id);
        
        if (batch_tensor.defined()) {
            tensors.push_back(batch_tensor);
        }
        
        // Here you would fill the tensor with actual game state data
        // This depends on the specific neural network input format
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error preparing tensors for stream " << stream_id 
                  << ": " << e.what() << std::endl;
    }
    
    return tensors;
}

torch::Tensor CUDAStreamOptimizer::getTensorFromPool(size_t batch_size, int stream_id) {
    if (!config_.enable_memory_pooling || stream_id >= tensor_pools_.size()) {
        // Allocate new tensor
        try {
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCUDA, 0);
            
            return torch::zeros({static_cast<long>(batch_size), 256}, options); // Example dimensions
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to allocate tensor: " << e.what() << std::endl;
            return torch::Tensor();
        }
    }
    
    auto& pool = tensor_pools_[stream_id];
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    if (!pool->available_tensors.empty()) {
        auto tensor = pool->available_tensors.back();
        pool->available_tensors.pop_back();
        
        // Resize if necessary
        if (tensor.size(0) != static_cast<long>(batch_size)) {
            tensor = tensor.slice(0, 0, batch_size);
        }
        
        return tensor;
    }
    
    // Pool empty, allocate new tensor
    try {
        auto options = torch::TensorOptions()
            .dtype(pool->dtype)
            .device(pool->device);
        
        return torch::zeros({static_cast<long>(batch_size), 256}, options);
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to allocate pooled tensor: " << e.what() << std::endl;
        return torch::Tensor();
    }
}

void CUDAStreamOptimizer::returnTensorToPool(torch::Tensor tensor, int stream_id) {
    if (!config_.enable_memory_pooling || !tensor.defined() || 
        stream_id >= tensor_pools_.size()) {
        return;
    }
    
    auto& pool = tensor_pools_[stream_id];
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    if (pool->available_tensors.size() < config_.tensor_pool_size) {
        pool->available_tensors.push_back(tensor);
    }
    // If pool is full, let tensor be deallocated
}

void CUDAStreamOptimizer::synchronizeStream(int stream_id) {
    if (stream_id >= 0 && stream_id < cuda_streams_.size()) {
        cuda_streams_[stream_id].synchronize();
    }
}

bool CUDAStreamOptimizer::checkCUDAAvailability() {
    if (!torch::cuda::is_available()) {
        std::cerr << "❌ CUDA not available" << std::endl;
        return false;
    }
    
    if (torch::cuda::device_count() == 0) {
        std::cerr << "❌ No CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "✅ CUDA available with " << torch::cuda::device_count() << " devices" << std::endl;
    return true;
}

void CUDAStreamOptimizer::initializeMemoryPools() {
    tensor_pools_.clear();
    tensor_pools_.reserve(config_.num_streams);
    
    for (int i = 0; i < config_.num_streams; ++i) {
        auto pool = std::make_unique<TensorPool>(
            config_.tensor_pool_size,
            torch::kFloat32,
            torch::Device(torch::kCUDA, 0)
        );
        
        // Pre-allocate some tensors
        for (size_t j = 0; j < 8; ++j) {
            try {
                auto options = torch::TensorOptions()
                    .dtype(pool->dtype)
                    .device(pool->device);
                
                auto tensor = torch::zeros({config_.max_batch_size, 256}, options);
                pool->available_tensors.push_back(tensor);
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Failed to pre-allocate tensor for pool " << i 
                          << ": " << e.what() << std::endl;
                break;
            }
        }
        
        tensor_pools_.push_back(std::move(pool));
        std::cout << "  Initialized tensor pool " << i << " with " 
                  << tensor_pools_[i]->available_tensors.size() << " pre-allocated tensors" << std::endl;
    }
}

void CUDAStreamOptimizer::cleanup() {
    // Synchronize all streams before cleanup
    for (int i = 0; i < cuda_streams_.size(); ++i) {
        try {
            synchronizeStream(i);
        } catch (const std::exception& e) {
            std::cerr << "⚠️  Error synchronizing stream " << i 
                      << " during cleanup: " << e.what() << std::endl;
        }
    }
    
    // Clear resources
    tensor_pools_.clear();
    cuda_streams_.clear();
    raw_streams_.clear();
    
    std::cout << "🧹 CUDA resources cleaned up" << std::endl;
}

std::vector<float> CUDAStreamOptimizer::getStreamUtilization() const {
    std::vector<float> utilization(config_.num_streams);
    
    for (int i = 0; i < config_.num_streams; ++i) {
        utilization[i] = stream_busy_[i].load(std::memory_order_acquire) ? 100.0f : 0.0f;
    }
    
    return utilization;
}

CUDAStreamOptimizer::StreamStats CUDAStreamOptimizer::getStatistics() const {
    StreamStats stats;
    
    stats.total_batches = total_batches_processed_.load();
    stats.total_inferences = total_inferences_.load();
    stats.stream_switches = total_stream_switches_.load();
    
    stats.per_stream_batches.resize(config_.num_streams);
    size_t total_stream_batches = 0;
    for (int i = 0; i < config_.num_streams; ++i) {
        stats.per_stream_batches[i] = per_stream_batches_[i].load();
        total_stream_batches += stats.per_stream_batches[i];
    }
    
    stats.avg_batch_size = total_stream_batches > 0 ? 
        static_cast<float>(stats.total_inferences) / total_stream_batches : 0.0f;
    
    // Calculate average GPU utilization
    auto utilization = getStreamUtilization();
    stats.total_gpu_utilization = 0.0f;
    for (float util : utilization) {
        stats.total_gpu_utilization += util;
    }
    stats.total_gpu_utilization /= config_.num_streams;
    
    return stats;
}

void CUDAStreamOptimizer::configureDynamicStreams(bool enable, float threshold) {
    // This would implement dynamic stream allocation based on load
    // For now, just log the configuration
}

#endif // TORCH_USE_CUDA

} // namespace mcts
} // namespace alphazero