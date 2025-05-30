#include "mcts/cuda_graph_capture.h"
#include "utils/logger.h"
#include <chrono>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace alphazero {
namespace mcts {

CUDAGraphCapture::CUDAGraphCapture(const Config& config) 
    : config_(config), cuda_graph_(nullptr), graph_exec_(nullptr) {
    initializeStreams();
}

CUDAGraphCapture::~CUDAGraphCapture() {
    clearGraph();
    cleanupStreams();
}

void CUDAGraphCapture::initializeStreams() {
    cudaStreamCreate(&capture_stream_);
    cudaStreamCreate(&exec_stream_);
}

void CUDAGraphCapture::cleanupStreams() {
    if (capture_stream_) {
        cudaStreamDestroy(capture_stream_);
        capture_stream_ = nullptr;
    }
    if (exec_stream_) {
        cudaStreamDestroy(exec_stream_);
        exec_stream_ = nullptr;
    }
}

bool CUDAGraphCapture::captureInference(
    torch::jit::script::Module& model,
    const torch::Tensor& sample_input,
    size_t batch_size) {
    
    if (!config_.enable_graph_capture) {
        return false;
    }
    
    if (!validateBatchSize(batch_size)) {
        LOG_SYSTEM_ERROR("Invalid batch size for graph capture: {}", batch_size);
        return false;
    }
    
    // Clear any existing graph
    clearGraph();
    
    try {
        // Prepare input tensor with correct batch size
        auto input_shape = sample_input.sizes().vec();
        input_shape[0] = batch_size;
        graph_input_buffer_ = torch::zeros(input_shape, sample_input.options());
        
        // Warm up the model
        // LOG_SYSTEM_INFO("Warming up model for graph capture (batch_size={})", batch_size);
        for (int i = 0; i < config_.capture_warmup_iterations; ++i) {
            auto outputs = model.forward({graph_input_buffer_}).toTuple();
            graph_policy_output_ = outputs->elements()[0].toTensor();
            graph_value_output_ = outputs->elements()[1].toTensor();
        }
        
        // Synchronize before capture
        cudaStreamSynchronize(capture_stream_);
        
        // Begin graph capture
        // LOG_SYSTEM_INFO("Beginning CUDA graph capture for batch size {}", batch_size);
        cudaGraphCreate(&cuda_graph_, 0);
        
        if (config_.use_stream_capture) {
            // Stream capture mode
            cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal);
            
            // Set stream for PyTorch operations
            c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(capture_stream_, 0));
            
            // Execute model in capture mode
            auto outputs = model.forward({graph_input_buffer_}).toTuple();
            graph_policy_output_ = outputs->elements()[0].toTensor();
            graph_value_output_ = outputs->elements()[1].toTensor();
            
            // End capture
            cudaStreamEndCapture(capture_stream_, &cuda_graph_);
        } else {
            // Manual graph construction (fallback)
            LOG_SYSTEM_WARN("Manual graph construction not implemented, falling back to regular execution");
            return false;
        }
        
        // Create executable graph
        cudaGraphInstantiate(&graph_exec_, cuda_graph_, nullptr, nullptr, 0);
        
        captured_batch_size_ = batch_size;
        graph_captured_.store(true);
        
        // LOG_SYSTEM_INFO("Successfully captured CUDA graph for batch size {}", batch_size);
        return true;
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to capture CUDA graph: {}", e.what());
        clearGraph();
        return false;
    }
}

std::pair<torch::Tensor, torch::Tensor> CUDAGraphCapture::executeGraph(
    const torch::Tensor& input,
    size_t batch_size) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    total_executions_.fetch_add(1);
    
    try {
        // Check if we can use the captured graph
        if (graph_captured_.load() && 
            batch_size == captured_batch_size_ && 
            !graph_executing_.exchange(true)) {
            
            // Copy input to graph buffer
            graph_input_buffer_.copy_(input);
            
            // Execute graph
            cudaGraphLaunch(graph_exec_, exec_stream_);
            cudaStreamSynchronize(exec_stream_);
            
            graph_executing_.store(false);
            graph_hits_.fetch_add(1);
            
            // Record execution time
            auto end_time = std::chrono::high_resolution_clock::now();
            double exec_time_ms = std::chrono::duration<double, std::milli>(
                end_time - start_time).count();
            recordGraphExecution(exec_time_ms, true);
            
            return {graph_policy_output_.clone(), graph_value_output_.clone()};
            
        } else {
            // Fallback to regular execution
            graph_misses_.fetch_add(1);
            
            // This should be handled by the caller with actual model execution
            LOG_SYSTEM_WARN("Graph execution not available for batch size {}, falling back", batch_size);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double exec_time_ms = std::chrono::duration<double, std::milli>(
                end_time - start_time).count();
            recordGraphExecution(exec_time_ms, false);
            
            // Return empty tensors to indicate fallback needed
            return {torch::Tensor(), torch::Tensor()};
        }
        
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Graph execution failed: {}", e.what());
        graph_executing_.store(false);
        return {torch::Tensor(), torch::Tensor()};
    }
}

void CUDAGraphCapture::clearGraph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
        cuda_graph_ = nullptr;
    }
    graph_captured_.store(false);
    captured_batch_size_ = 0;
}

bool CUDAGraphCapture::validateBatchSize(size_t batch_size) const {
    return batch_size > 0 && batch_size <= config_.max_batch_size;
}

void CUDAGraphCapture::recordGraphExecution(double exec_time_ms, bool used_graph) {
    if (used_graph) {
        double current = total_graph_time_ms_.load();
        while (!total_graph_time_ms_.compare_exchange_weak(current, current + exec_time_ms));
    } else {
        double current = total_regular_time_ms_.load();
        while (!total_regular_time_ms_.compare_exchange_weak(current, current + exec_time_ms));
    }
}

CUDAGraphCapture::GraphStats CUDAGraphCapture::getStats() const {
    GraphStats stats;
    stats.total_executions = total_executions_.load();
    stats.graph_hits = graph_hits_.load();
    stats.graph_misses = graph_misses_.load();
    
    if (stats.graph_hits > 0) {
        stats.avg_graph_exec_time_ms = total_graph_time_ms_.load() / stats.graph_hits;
    } else {
        stats.avg_graph_exec_time_ms = 0.0;
    }
    
    if (stats.graph_misses > 0) {
        stats.avg_regular_exec_time_ms = total_regular_time_ms_.load() / stats.graph_misses;
    } else {
        stats.avg_regular_exec_time_ms = 0.0;
    }
    
    if (stats.avg_regular_exec_time_ms > 0) {
        stats.speedup_ratio = stats.avg_regular_exec_time_ms / stats.avg_graph_exec_time_ms;
    } else {
        stats.speedup_ratio = 1.0f;
    }
    
    return stats;
}

bool CUDAGraphCapture::updateGraphInput(const torch::Tensor& new_input, size_t batch_size) {
    if (!graph_captured_.load() || batch_size != captured_batch_size_) {
        return false;
    }
    
    // For now, we just copy the input to the buffer
    // In the future, we could update graph nodes directly
    graph_input_buffer_.copy_(new_input);
    return true;
}

// MultiGraphManager implementation

MultiGraphManager::MultiGraphManager(const Config& config)
    : config_(config), model_(nullptr) {
}

MultiGraphManager::~MultiGraphManager() {
    graphs_.clear();
}

void MultiGraphManager::initialize(torch::jit::script::Module& model) {
    model_ = &model;
    
    // Pre-capture graphs for configured batch sizes
    if (config_.batch_sizes.empty()) {
        return;
    }
    
    // Create sample input
    auto sample_input = torch::randn({1, 119, 15, 15}, torch::kCUDA);
    
    std::lock_guard<std::mutex> lock(graphs_mutex_);
    for (size_t batch_size : config_.batch_sizes) {
        auto graph_config = CUDAGraphCapture::Config();
        graph_config.max_batch_size = batch_size;
        
        auto graph = std::make_unique<CUDAGraphCapture>(graph_config);
        if (graph->captureInference(model, sample_input, batch_size)) {
            graphs_[batch_size] = std::move(graph);
            lru_list_.push_back(batch_size);
            lru_map_[batch_size] = std::prev(lru_list_.end());
            // LOG_SYSTEM_INFO("Pre-captured graph for batch size {}", batch_size);
        }
    }
}

CUDAGraphCapture* MultiGraphManager::getGraphForBatchSize(size_t batch_size) {
    std::lock_guard<std::mutex> lock(graphs_mutex_);
    
    auto it = graphs_.find(batch_size);
    if (it != graphs_.end()) {
        updateLRU(batch_size);
        return it->second.get();
    }
    
    // Auto-capture if enabled
    if (config_.auto_capture && model_) {
        // Check if we need to evict
        if (graphs_.size() >= config_.max_graphs) {
            evictLRUGraph();
        }
        
        // Create new graph
        auto graph_config = CUDAGraphCapture::Config();
        graph_config.max_batch_size = batch_size;
        
        auto graph = std::make_unique<CUDAGraphCapture>(graph_config);
        auto sample_input = torch::randn({1, 119, 15, 15}, torch::kCUDA);
        
        if (graph->captureInference(*model_, sample_input, batch_size)) {
            auto* graph_ptr = graph.get();
            graphs_[batch_size] = std::move(graph);
            lru_list_.push_back(batch_size);
            lru_map_[batch_size] = std::prev(lru_list_.end());
            // LOG_SYSTEM_INFO("Auto-captured graph for batch size {}", batch_size);
            return graph_ptr;
        }
    }
    
    return nullptr;
}

std::pair<torch::Tensor, torch::Tensor> MultiGraphManager::execute(
    torch::jit::script::Module& model,
    const torch::Tensor& input,
    size_t batch_size) {
    
    auto* graph = getGraphForBatchSize(batch_size);
    if (graph) {
        auto result = graph->executeGraph(input, batch_size);
        if (result.first.defined() && result.second.defined()) {
            return result;
        }
    }
    
    // Fallback to regular execution
    auto outputs = model.forward({input}).toTuple();
    return {outputs->elements()[0].toTensor(), 
            outputs->elements()[1].toTensor()};
}

void MultiGraphManager::evictLRUGraph() {
    if (lru_list_.empty()) {
        return;
    }
    
    size_t evict_size = lru_list_.front();
    lru_list_.pop_front();
    lru_map_.erase(evict_size);
    graphs_.erase(evict_size);
    
    // LOG_SYSTEM_INFO("Evicted graph for batch size {} (LRU)", evict_size);
}

void MultiGraphManager::updateLRU(size_t batch_size) {
    auto it = lru_map_.find(batch_size);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
        lru_list_.push_back(batch_size);
        it->second = std::prev(lru_list_.end());
    }
}

MultiGraphManager::CombinedStats MultiGraphManager::getStats() const {
    CombinedStats stats;
    stats.total_graphs = graphs_.size();
    stats.active_graphs = 0;
    
    double total_speedup = 0.0;
    size_t speedup_count = 0;
    
    for (const auto& [batch_size, graph] : graphs_) {
        auto graph_stats = graph->getStats();
        stats.graph_stats.emplace_back(batch_size, graph_stats);
        
        if (graph_stats.total_executions > 0) {
            stats.active_graphs++;
        }
        
        if (graph_stats.speedup_ratio > 1.0f) {
            total_speedup += graph_stats.speedup_ratio;
            speedup_count++;
        }
    }
    
    stats.overall_speedup = speedup_count > 0 ? 
        static_cast<float>(total_speedup / speedup_count) : 1.0f;
    
    return stats;
}

}  // namespace mcts
}  // namespace alphazero