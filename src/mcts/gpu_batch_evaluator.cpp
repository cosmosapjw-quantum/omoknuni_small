#include "mcts/gpu_batch_evaluator.h"
#include "utils/logger.h"
#include "utils/profiler.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

// Forward declarations of CUDA kernel launcher functions
extern "C" {
    void launchUCBKernel(
        const float* Q,
        const int* N,
        const float* P,
        const int* N_total,
        float* UCB,
        float c_puct,
        int batch_size,
        int num_nodes,
        int num_actions,
        cudaStream_t stream
    );
    
    void launchPathSelectionKernel(
        const float* UCB,
        const bool* children_mask,
        int* selected_paths,
        int* virtual_loss,
        int batch_size,
        int num_nodes,
        int num_actions,
        int max_depth,
        cudaStream_t stream
    );
    
    void launchBackupKernel(
        float* W,
        int* N,
        int* virtual_loss,
        const float* values,
        const int* paths,
        int batch_size,
        int num_nodes,
        int num_actions,
        int max_depth,
        cudaStream_t stream
    );
}

namespace alphazero {
namespace mcts {

// Static member definition
std::unique_ptr<GPUBatchEvaluator> GlobalGPUBatchEvaluator::instance_ = nullptr;

GPUBatchEvaluator::GPUBatchEvaluator(
    std::shared_ptr<nn::NeuralNetwork> neural_net,
    const Config& config)
    : config_(config),
      neural_net_(neural_net),
      request_queue_(config.max_batch_size * 4),
      producer_token_(request_queue_),
      consumer_token_(request_queue_) {
    
    // Initialize GPU optimizer with custom config
    nn::GPUOptimizer::Config gpu_config;
    gpu_config.max_batch_size = config.max_batch_size;
    gpu_config.num_streams = config.num_cuda_streams;
    gpu_config.enable_cuda_graphs = config.enable_cuda_graphs;
    gpu_config.enable_tensor_cores = config.enable_tensor_cores;
    gpu_config.pre_allocate = true;
    gpu_config.tensor_cache_size = 4;
    
    gpu_optimizer_ = std::make_unique<nn::GPUOptimizer>(gpu_config);
    
    // Initialize CUDA streams
    cuda_streams_.resize(config.num_cuda_streams);
    for (size_t i = 0; i < config.num_cuda_streams; ++i) {
        cudaStreamCreate(&cuda_streams_[i]);
    }
    
    // Pre-allocate tree tensors
    preallocated_trees_ = std::make_unique<TreeTensors>(
        initializeTreeTensors(config.max_batch_size)
    );
    
    // LOG_SYSTEM_INFO("GPU Batch Evaluator initialized with:");
    // LOG_SYSTEM_INFO("  - Min batch size: {}", config.min_batch_size);
    // LOG_SYSTEM_INFO("  - Max batch size: {}", config.max_batch_size);
    // LOG_SYSTEM_INFO("  - Batch timeout: {} us", config.batch_timeout_us);
    // LOG_SYSTEM_INFO("  - CUDA graphs: {}", config.enable_cuda_graphs);
    // LOG_SYSTEM_INFO("  - Adaptive batching: {}", config.enable_adaptive_batching);
}

GPUBatchEvaluator::~GPUBatchEvaluator() {
    stop();
    
    // Clean up CUDA streams
    for (auto stream : cuda_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
}

void GPUBatchEvaluator::start() {
    if (!running_) {
        running_ = true;
        processing_thread_ = std::thread(&GPUBatchEvaluator::processingLoop, this);
    }
}

void GPUBatchEvaluator::stop() {
    if (running_) {
        running_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        
        // LOG_SYSTEM_INFO("GPU Batch Evaluator stopped. Stats:");
        // LOG_SYSTEM_INFO("  - Total batches: {}", stats_.total_batches.load());
        // LOG_SYSTEM_INFO("  - Total states: {}", stats_.total_states.load());
        // LOG_SYSTEM_INFO("  - Avg batch size: {:.2f}", stats_.avg_batch_size.load());
        // LOG_SYSTEM_INFO("  - GPU utilization: {:.2f}%", stats_.gpu_utilization.load());
        // LOG_SYSTEM_INFO("  - CUDA graph hits: {}", stats_.cuda_graph_hits.load());
        // LOG_SYSTEM_INFO("  - Tensorized operations: {}", stats_.tensorized_operations.load());
    }
}

std::future<std::vector<NetworkOutput>> GPUBatchEvaluator::submitBatch(
    std::vector<std::unique_ptr<core::IGameState>> states,
    const std::vector<int32_t>& node_ids) {
    
    BatchRequest request;
    request.states = std::move(states);
    request.node_ids = node_ids;
    request.timestamp = std::chrono::steady_clock::now();
    
    auto future = request.promise.get_future();
    
    // Use lock-free enqueue
    request_queue_.enqueue(producer_token_, std::move(request));
    
    return future;
}

void GPUBatchEvaluator::processingLoop() {
    std::vector<BatchRequest> batch;
    batch.reserve(config_.max_batch_size);
    
    // Dequeue buffer for bulk operations
    std::vector<BatchRequest> dequeue_buffer(config_.max_batch_size);
    
    while (running_) {
        batch.clear();
        int total_states = 0;
        
        auto batch_start_time = std::chrono::steady_clock::now();
        auto timeout_time = batch_start_time + 
                          std::chrono::microseconds(config_.batch_timeout_us);
        
        // Collect requests into batch
        while (running_ && total_states < static_cast<int>(config_.max_batch_size)) {
            // Try bulk dequeue for efficiency
            size_t dequeued = request_queue_.try_dequeue_bulk(
                consumer_token_, 
                dequeue_buffer.begin(), 
                std::min(config_.max_batch_size - total_states, dequeue_buffer.size())
            );
            
            if (dequeued > 0) {
                for (size_t i = 0; i < dequeued; ++i) {
                    int request_size = dequeue_buffer[i].states.size();
                    
                    // Check if this would exceed batch size
                    if (total_states + request_size > static_cast<int>(config_.max_batch_size)) {
                        // Put back remaining requests
                        for (size_t j = i; j < dequeued; ++j) {
                            request_queue_.enqueue(std::move(dequeue_buffer[j]));
                        }
                        break;
                    }
                    
                    total_states += request_size;
                    batch.push_back(std::move(dequeue_buffer[i]));
                }
            }
            
            // Check if we should process
            auto now = std::chrono::steady_clock::now();
            
            if (config_.enable_adaptive_batching) {
                int optimal_size = computeOptimalBatchSize();
                if (shouldProcessBatch(total_states, optimal_size, batch_start_time)) {
                    break;
                }
            } else {
                // Fixed batching strategy
                if (total_states >= static_cast<int>(config_.min_batch_size) || 
                    (!batch.empty() && now >= timeout_time)) {
                    break;
                }
            }
            
            // Small sleep to reduce CPU usage
            if (dequeued == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        // Process collected batch
        if (!batch.empty() && running_) {
            processBatchGPU(batch);
        }
    }
}

void GPUBatchEvaluator::processBatchGPU(std::vector<BatchRequest>& batch) {
    PROFILE_SCOPE_N("GPUBatchEvaluator::processBatchGPU");
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Prepare all states for GPU processing
    std::vector<std::unique_ptr<core::IGameState>> all_states;
    std::vector<int> request_boundaries;
    
    int total_states = 0;
    for (auto& request : batch) {
        request_boundaries.push_back(total_states);
        total_states += request.states.size();
        for (auto& state : request.states) {
            all_states.push_back(std::move(state));
        }
    }
    
    try {
        // Get current CUDA stream
        // cudaStream_t stream = cuda_streams_[
        //     current_stream_.fetch_add(1) % config_.num_cuda_streams
        // ];  // TODO: Pass stream to GPU operations
        
        // Prepare states batch using GPU optimizer (with double-buffering)
        torch::Tensor state_tensors = gpu_optimizer_->prepareStatesBatch(
            all_states, false  // Asynchronous transfer
        );
        
        // Run neural network inference
        std::vector<NetworkOutput> results;
        
        // Check if we can use CUDA graphs
        if (config_.enable_cuda_graphs && total_states == config_.max_batch_size) {
            // Use CUDA graph for fixed-size batches
            std::string graph_id = "batch_inference_" + std::to_string(config_.max_batch_size);
            
            if (gpu_optimizer_->isCudaGraphAvailable(graph_id)) {
                // Execute captured graph
                torch::Tensor output = gpu_optimizer_->executeCudaGraph(graph_id, state_tensors);
                stats_.cuda_graph_hits++;
                
                // Convert output to NetworkOutput format
                // TODO: Implement conversion based on your NetworkOutput structure
            } else {
                // Capture graph on first execution
                // For now, disable CUDA graph capture as it requires tensor-based forward method
                // TODO: Implement tensor-based forward method in NeuralNetwork interface
                results = neural_net_->inference(all_states);
            }
        }
        
        // Fallback to regular inference if CUDA graphs not used
        if (results.empty()) {
            results = neural_net_->inference(all_states);
        }
        
        // Distribute results back to requests
        int result_idx = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            int request_size = (i + 1 < batch.size()) 
                ? request_boundaries[i + 1] - request_boundaries[i]
                : total_states - request_boundaries[i];
            
            std::vector<NetworkOutput> request_results(
                results.begin() + result_idx,
                results.begin() + result_idx + request_size
            );
            
            batch[i].promise.set_value(std::move(request_results));
            result_idx += request_size;
        }
        
        // Update statistics
        auto end_time = std::chrono::steady_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        stats_.total_batches++;
        stats_.total_states += total_states;
        
        // Update average batch size
        double current_avg = stats_.avg_batch_size.load();
        double new_avg = (current_avg * (stats_.total_batches - 1) + total_states) 
                        / stats_.total_batches;
        stats_.avg_batch_size = new_avg;
        
        // Estimate GPU utilization
        double batch_fullness = static_cast<double>(total_states) / config_.max_batch_size;
        double time_efficiency = std::min(1.0, 
            static_cast<double>(config_.batch_timeout_us) / duration_us);
        stats_.gpu_utilization = batch_fullness * time_efficiency * 100.0;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("GPU Batch Evaluator error: {}", e.what());
        
        // Set error for all requests
        for (auto& request : batch) {
            request.promise.set_exception(std::current_exception());
        }
    }
}

int GPUBatchEvaluator::computeOptimalBatchSize() {
    // Adaptive batch sizing based on GPU utilization
    double current_utilization = stats_.gpu_utilization.load();
    int current_optimal = current_optimal_batch_.load();
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_adjustment = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_adjustment_).count();
    
    // Adjust every 100ms
    if (time_since_adjustment > 100) {
        if (current_utilization < config_.gpu_utilization_target * 100 - 10) {
            // Increase batch size to improve utilization
            current_optimal = std::min(
                static_cast<int>(config_.max_batch_size),
                static_cast<int>(current_optimal * 1.1)
            );
        } else if (current_utilization > config_.gpu_utilization_target * 100 + 10) {
            // Decrease batch size if we're over-utilizing (causing delays)
            current_optimal = std::max(
                static_cast<int>(config_.min_batch_size),
                static_cast<int>(current_optimal * 0.9)
            );
        }
        
        current_optimal_batch_ = current_optimal;
        last_adjustment_ = now;
    }
    
    return current_optimal;
}

bool GPUBatchEvaluator::shouldProcessBatch(
    int current_size, 
    int optimal_size,
    std::chrono::steady_clock::time_point first_timestamp) {
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - first_timestamp).count();
    
    // Process if we've reached optimal size
    if (current_size >= optimal_size) {
        return true;
    }
    
    // Process if we've hit timeout
    if (elapsed_us >= static_cast<long>(config_.batch_timeout_us)) {
        return true;
    }
    
    // Adaptive timeout based on batch fullness
    float fullness = static_cast<float>(current_size) / optimal_size;
    long adaptive_timeout = config_.batch_timeout_us * (1.0f - fullness * 0.5f);
    
    return elapsed_us >= adaptive_timeout;
}

GPUBatchEvaluator::TreeTensors GPUBatchEvaluator::initializeTreeTensors(int batch_size) {
    TreeTensors tensors;
    
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    
    // Initialize statistics tensors
    tensors.Q_tensor = torch::zeros({batch_size, 
                                   static_cast<long>(config_.max_nodes_per_tree), 
                                   static_cast<long>(config_.max_actions)}, 
                                  float_options);
    
    tensors.N_tensor = torch::zeros({batch_size, 
                                   static_cast<long>(config_.max_nodes_per_tree), 
                                   static_cast<long>(config_.max_actions)}, 
                                  int_options);
    
    tensors.P_tensor = torch::zeros({batch_size, 
                                   static_cast<long>(config_.max_nodes_per_tree), 
                                   static_cast<long>(config_.max_actions)}, 
                                  float_options);
    
    tensors.W_tensor = torch::zeros({batch_size, 
                                   static_cast<long>(config_.max_nodes_per_tree), 
                                   static_cast<long>(config_.max_actions)}, 
                                  float_options);
    
    tensors.virtual_losses = torch::zeros({batch_size, 
                                         static_cast<long>(config_.max_nodes_per_tree), 
                                         static_cast<long>(config_.max_actions)}, 
                                        int_options);
    
    // Initialize structure tensors
    tensors.parent_indices = torch::full({batch_size, 
                                        static_cast<long>(config_.max_nodes_per_tree)}, 
                                       -1, int_options);
    
    tensors.children_mask = torch::zeros({batch_size, 
                                        static_cast<long>(config_.max_nodes_per_tree), 
                                        static_cast<long>(config_.max_actions)}, 
                                       torch::kBool);
    
    tensors.node_visits = torch::zeros({batch_size, 
                                      static_cast<long>(config_.max_nodes_per_tree)}, 
                                     int_options);
    
    // Initialize result tensors
    tensors.ucb_scores = torch::zeros({batch_size, 
                                     static_cast<long>(config_.max_nodes_per_tree), 
                                     static_cast<long>(config_.max_actions)}, 
                                    float_options);
    
    tensors.selected_actions = torch::zeros({batch_size, 100}, int_options);  // Max depth 100
    tensors.selected_paths = torch::zeros({batch_size, 100, 2}, int_options);
    
    return tensors;
}

void GPUBatchEvaluator::batchSelectPaths(
    const TreeTensors& tree_tensors,
    torch::Tensor& selected_paths,
    int batch_size,
    float c_puct) {
    
    PROFILE_SCOPE_N("GPUBatchEvaluator::batchSelectPaths");
    
    // Get current CUDA stream
    cudaStream_t stream = cuda_streams_[
        current_stream_.fetch_add(1) % config_.num_cuda_streams
    ];
    
    int num_nodes = tree_tensors.Q_tensor.size(1);
    int num_actions = tree_tensors.Q_tensor.size(2);
    int max_depth = selected_paths.size(1);
    
    // Launch UCB calculation kernel
    ::launchUCBKernel(
        tree_tensors.Q_tensor.data_ptr<float>(),
        tree_tensors.N_tensor.data_ptr<int>(),
        tree_tensors.P_tensor.data_ptr<float>(),
        tree_tensors.node_visits.data_ptr<int>(),
        const_cast<TreeTensors&>(tree_tensors).ucb_scores.data_ptr<float>(),
        c_puct,
        batch_size,
        num_nodes,
        num_actions,
        stream
    );
    
    // Launch path selection kernel
    ::launchPathSelectionKernel(
        tree_tensors.ucb_scores.data_ptr<float>(),
        tree_tensors.children_mask.data_ptr<bool>(),
        selected_paths.data_ptr<int>(),
        tree_tensors.virtual_losses.data_ptr<int>(),
        batch_size,
        num_nodes,
        num_actions,
        max_depth,
        stream
    );
    
    // Synchronize if needed
    cudaStreamSynchronize(stream);
    
    stats_.tensorized_operations++;
}

void GPUBatchEvaluator::batchUpdateStatistics(
    TreeTensors& tree_tensors,
    const torch::Tensor& values,
    const torch::Tensor& paths,
    int batch_size) {
    
    PROFILE_SCOPE_N("GPUBatchEvaluator::batchUpdateStatistics");
    
    // Get current CUDA stream
    cudaStream_t stream = cuda_streams_[
        current_stream_.fetch_add(1) % config_.num_cuda_streams
    ];
    
    int num_nodes = tree_tensors.W_tensor.size(1);
    int num_actions = tree_tensors.W_tensor.size(2);
    int max_depth = paths.size(1);
    
    // Launch backup kernel
    ::launchBackupKernel(
        tree_tensors.W_tensor.data_ptr<float>(),
        tree_tensors.N_tensor.data_ptr<int>(),
        tree_tensors.virtual_losses.data_ptr<int>(),
        values.data_ptr<float>(),
        paths.data_ptr<int>(),
        batch_size,
        num_nodes,
        num_actions,
        max_depth,
        stream
    );
    
    // Update Q values from W and N
    tree_tensors.Q_tensor = tree_tensors.W_tensor / 
        tree_tensors.N_tensor.to(torch::kFloat32).clamp_min(1.0);
    
    stats_.tensorized_operations++;
}

} // namespace mcts
} // namespace alphazero