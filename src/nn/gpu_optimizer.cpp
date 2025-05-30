#include "nn/gpu_optimizer.h"
#include "utils/logger.h"
#include "utils/profiler.h"
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>

namespace alphazero {
namespace nn {

GPUOptimizer::GPUOptimizer(const Config& config) : config_(config) {
    if (!torch::cuda::is_available()) {
        LOG_SYSTEM_WARN("GPU not available, GPU optimizer will run in CPU mode");
        return;
    }
    
    initializeCUDAStreams();
    if (config_.pre_allocate) {
        allocatePinnedMemory();
        preallocateTensors(config_.max_batch_size);
    }
    
    if (config_.enable_persistent_kernels) {
        setupPersistentKernels();
    }
    
    // LOG_SYSTEM_INFO("GPU Optimizer initialized with {} streams, max batch {}", 
    //                config_.num_streams, config_.max_batch_size);
    // LOG_SYSTEM_INFO("GPU Optimizations: CUDA Graphs={}, Persistent Kernels={}, TorchScript={}", 
    //                config_.enable_cuda_graphs, config_.enable_persistent_kernels, config_.enable_torch_script);
}

GPUOptimizer::~GPUOptimizer() {
    cleanupResources();
}

void GPUOptimizer::initializeCUDAStreams() {
    if (!torch::cuda::is_available()) return;
    
    cuda_streams_.resize(config_.num_streams);
    for (size_t i = 0; i < config_.num_streams; ++i) {
        cudaError_t error = cudaStreamCreate(&cuda_streams_[i]);
        if (error != cudaSuccess) {
            LOG_SYSTEM_ERROR("Failed to create CUDA stream {}: {}", i, cudaGetErrorString(error));
        }
    }
}

void GPUOptimizer::allocatePinnedMemory() {
    PROFILE_SCOPE_N("GPUOptimizer::allocatePinnedMemory");
    
    if (!torch::cuda::is_available() || !config_.use_pinned_memory) return;
    
    tensor_cache_ = std::make_unique<TensorCache>();
    
    // Pre-allocate pinned memory tensors
    for (size_t i = 0; i < config_.tensor_cache_size; ++i) {
        // Create CPU tensor with pinned memory
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(true);
            
        // Allocate for max batch size
        auto pinned_tensor = torch::empty(
            {static_cast<long>(config_.max_batch_size), 
             static_cast<long>(config_.num_channels),
             static_cast<long>(config_.board_height), 
             static_cast<long>(config_.board_width)}, 
            options);
            
        tensor_cache_->cpu_pinned_tensors.push_back(pinned_tensor);
        
        // Pre-allocate corresponding GPU tensor
        auto gpu_tensor = torch::empty(
            {static_cast<long>(config_.max_batch_size), 
             static_cast<long>(config_.num_channels),
             static_cast<long>(config_.board_height), 
             static_cast<long>(config_.board_width)}, 
            torch::device(torch::kCUDA));
            
        tensor_cache_->gpu_tensors.push_back(gpu_tensor);
    }
    
    // Create buffer pairs for double-buffering
    // Use 3 buffer pairs for optimal pipelining (triple-buffering)
    const size_t num_buffer_pairs = 3;
    tensor_cache_->buffer_pairs.resize(num_buffer_pairs);
    
    for (size_t i = 0; i < num_buffer_pairs; ++i) {
        BufferPair& buffer = tensor_cache_->buffer_pairs[i];
        
        // Create pinned memory CPU tensor for this buffer
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(true);
            
        buffer.cpu_tensor = torch::empty(
            {static_cast<long>(config_.max_batch_size), 
             static_cast<long>(config_.num_channels),
             static_cast<long>(config_.board_height), 
             static_cast<long>(config_.board_width)}, 
            options);
            
        // Create GPU tensor for this buffer
        buffer.gpu_tensor = torch::empty_like(buffer.cpu_tensor, torch::device(torch::kCUDA));
        
        // Create CUDA events for synchronization
        cudaEventCreate(&buffer.copy_done);
        cudaEventCreate(&buffer.compute_done);
        
        // Initialize in_use flag
        buffer.in_use.store(false, std::memory_order_release);
    }
    
    // LOG_SYSTEM_INFO("Allocated {} pinned memory tensors and {} buffer pairs of size {}x{}x{}x{}", 
    //                config_.tensor_cache_size, num_buffer_pairs, config_.max_batch_size, 
    //                config_.num_channels, config_.board_height, config_.board_width);
}

torch::Tensor GPUOptimizer::prepareStatesBatch(const std::vector<std::unique_ptr<core::IGameState>>& states, 
                                              bool synchronize) {
    PROFILE_SCOPE_N("GPUOptimizer::prepareStatesBatch");
    
    if (states.empty()) {
        return torch::empty({0, static_cast<long>(config_.num_channels), 
                            static_cast<long>(config_.board_height), 
                            static_cast<long>(config_.board_width)});
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t batch_size = states.size();
    const size_t channels = config_.num_channels;
    const size_t height = config_.board_height;
    const size_t width = config_.board_width;
    
    // Check if we should use double-buffering
    bool use_double_buffer = !synchronize && tensor_cache_ && 
                           !tensor_cache_->buffer_pairs.empty() && 
                           batch_size <= config_.max_batch_size;
    
    // Choose between double-buffering and regular tensor allocation
    torch::Tensor cpu_tensor;
    torch::Tensor gpu_tensor;
    cudaStream_t stream = getCurrentStream();
    
    if (use_double_buffer) {
        // Double-buffering path - get a buffer pair
        BufferPair& buffer_pair = getNextBufferPair(batch_size);
        
        if (batch_size < config_.max_batch_size) {
            // Use slices for smaller batches
            cpu_tensor = buffer_pair.cpu_tensor.slice(0, 0, batch_size);
            gpu_tensor = buffer_pair.gpu_tensor.slice(0, 0, batch_size);
        } else {
            // Use full tensor for max batch size
            cpu_tensor = buffer_pair.cpu_tensor;
            gpu_tensor = buffer_pair.gpu_tensor;
        }
    } else if (config_.use_pinned_memory && tensor_cache_ && 
              batch_size <= config_.max_batch_size) {
        // Regular path with pre-allocated tensors (fallback)
        size_t tensor_idx = tensor_cache_->next_tensor_.fetch_add(1) % config_.tensor_cache_size;
        cpu_tensor = tensor_cache_->cpu_pinned_tensors[tensor_idx].slice(0, 0, batch_size);
        gpu_tensor = tensor_cache_->gpu_tensors[tensor_idx].slice(0, 0, batch_size);
    } else {
        // Allocate new tensors (rarely used fallback)
        auto cpu_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(config_.use_pinned_memory && torch::cuda::is_available());
            
        cpu_tensor = torch::zeros({static_cast<long>(batch_size), 
                                  static_cast<long>(channels),
                                  static_cast<long>(height), 
                                  static_cast<long>(width)}, 
                                 cpu_options);
                                 
        if (torch::cuda::is_available()) {
            gpu_tensor = torch::empty_like(cpu_tensor, torch::device(torch::kCUDA));
        }
    }
    
    // Convert states to tensor in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < states.size(); ++i) {
        if (states[i]) {
            stateToTensor(*states[i], cpu_tensor, i, channels, height, width);
        }
    }
    
    // Transfer to GPU if available
    if (torch::cuda::is_available()) {
        // Non-blocking copy from pinned memory to GPU
        gpu_tensor.copy_(cpu_tensor, /*non_blocking=*/true);
        
        if (use_double_buffer) {
            // Find the buffer pair we're using
            for (auto& buffer : tensor_cache_->buffer_pairs) {
                if (gpu_tensor.data_ptr() == buffer.gpu_tensor.data_ptr() ||
                    (batch_size < config_.max_batch_size && 
                     gpu_tensor.data_ptr() == buffer.gpu_tensor.slice(0, 0, batch_size).data_ptr())) {
                    
                    // Record event to signal when the copy is complete
                    cudaEventRecord(buffer.copy_done, stream);
                    
                    // Only wait when synchronize is true
                    if (synchronize) {
                        cudaEventSynchronize(buffer.copy_done);
                        buffer.in_use.store(false, std::memory_order_release);
                    }
                    
                    break;
                }
            }
        } else if (synchronize) {
            // Traditional synchronization
            cudaStreamSynchronize(stream);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        transfer_count_.fetch_add(1);
        total_transfer_time_us_.fetch_add(duration.count());
        
        return gpu_tensor;
    }
    
    return cpu_tensor;
}

void GPUOptimizer::stateToTensor(const core::IGameState& state, torch::Tensor& output,
                                size_t batch_idx, size_t channels, size_t height, size_t width) {
    // Convert game state to tensor representation
    // This is game-specific and should be overridden or specialized per game
    
    // Example implementation for board games
    std::string game_type = core::gameTypeToString(state.getGameType());
    if (game_type == "GOMOKU" || game_type == "GO" || game_type == "CHESS") {
        // Get tensor representation from the state
        try {
            // Use enhanced representation if channels > 3
            std::vector<std::vector<std::vector<float>>> tensor_data;
            if (channels > 3) {
                // Use enhanced representation for more complex neural networks
                tensor_data = state.getEnhancedTensorRepresentation();
            } else {
                // Use basic representation for simple networks
                tensor_data = state.getTensorRepresentation();
            }
            
            // Convert vector data to torch tensor
            torch::Tensor state_tensor = torch::zeros({static_cast<long>(channels), 
                                                     static_cast<long>(height), 
                                                     static_cast<long>(width)});
            
            // Copy data from vector representation
            for (size_t c = 0; c < channels && c < tensor_data.size(); ++c) {
                for (size_t h = 0; h < height && h < tensor_data[c].size(); ++h) {
                    for (size_t w = 0; w < width && w < tensor_data[c][h].size(); ++w) {
                        state_tensor[c][h][w] = tensor_data[c][h][w];
                    }
                }
            }
            
            // Copy to the output tensor at the correct batch index
            output[batch_idx].copy_(state_tensor);
        } catch (...) {
            // On error, fill with zeros
            output[batch_idx].zero_();
        }
    } else {
        // Default implementation - fill with zeros
        output[batch_idx].zero_();
    }
}

torch::Tensor GPUOptimizer::getPreallocatedTensor(size_t batch_size, size_t height, 
                                                 size_t width, size_t channels) {
    if (!tensor_cache_ || batch_size > config_.max_batch_size) {
        // Return empty tensor if no cache or batch too large
        return torch::empty({static_cast<long>(batch_size), 
                           static_cast<long>(channels),
                           static_cast<long>(height), 
                           static_cast<long>(width)}, 
                          torch::device(torch::kCUDA));
    }
    
    size_t tensor_idx = tensor_cache_->next_tensor_.fetch_add(1) % config_.tensor_cache_size;
    return tensor_cache_->gpu_tensors[tensor_idx].slice(0, 0, batch_size);
}

cudaStream_t GPUOptimizer::getCurrentStream() {
    if (cuda_streams_.empty()) {
        return cudaStreamDefault;
    }
    
    size_t stream_idx = current_stream_idx_.fetch_add(1) % cuda_streams_.size();
    return cuda_streams_[stream_idx];
}

void GPUOptimizer::synchronizeStreams() {
    for (auto stream : cuda_streams_) {
        cudaStreamSynchronize(stream);
    }
}

GPUOptimizer::MemoryStats GPUOptimizer::getMemoryStats() const {
    MemoryStats stats;
    
    if (torch::cuda::is_available()) {
        // For PyTorch 2.x, use the newer API
        auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        stats.allocated_gpu_memory = device_stats.allocated_bytes[0].current;
        stats.peak_gpu_memory = device_stats.allocated_bytes[0].peak;
    }
    
    stats.allocated_pinned_memory = 0;
    if (tensor_cache_) {
        for (const auto& tensor : tensor_cache_->cpu_pinned_tensors) {
            stats.allocated_pinned_memory += tensor.numel() * sizeof(float);
        }
    }
    
    stats.transfer_count = transfer_count_.load();
    if (stats.transfer_count > 0) {
        stats.avg_transfer_time = std::chrono::microseconds(
            total_transfer_time_us_.load() / stats.transfer_count);
    }
    
    return stats;
}

void GPUOptimizer::preallocateTensors(size_t batch_size) {
    if (!torch::cuda::is_available()) return;
    
    // This method is called during initialization
    // The actual allocation is done in allocatePinnedMemory()
    // LOG_SYSTEM_INFO("Pre-allocated tensors for batch size {}", batch_size);
}

GPUOptimizer::BufferPair& GPUOptimizer::getNextBufferPair(size_t batch_size) {
    if (!tensor_cache_ || tensor_cache_->buffer_pairs.empty() || batch_size > config_.max_batch_size) {
        // No buffer pairs or batch too large - fallback to default buffer
        static BufferPair dummy_buffer;
        return dummy_buffer;
    }
    
    // Try to find an available buffer pair
    for (int attempt = 0; attempt < 3; ++attempt) {
        size_t next_idx = tensor_cache_->next_buffer_.fetch_add(1) % tensor_cache_->buffer_pairs.size();
        auto& buffer = tensor_cache_->buffer_pairs[next_idx];
        
        // Try to acquire the buffer
        bool expected = false;
        if (buffer.in_use.compare_exchange_strong(expected, true)) {
            return buffer;
        }
        
        // If all buffers are in use, we'll return the last one we tried
        if (attempt == 2) {
            // Wait for this buffer to be ready
            cudaEventSynchronize(buffer.compute_done);
            buffer.in_use.store(true, std::memory_order_release);
            return buffer;
        }
    }
    
    // Fallback - should never reach here
    return tensor_cache_->buffer_pairs[0];
}

void GPUOptimizer::setupPersistentKernels() {
    if (!torch::cuda::is_available()) return;
    
    // Enable persistent L2 cache for better data locality
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.persistingL2CacheMaxSize > 0) {
        // Set L2 cache persistence for frequently accessed data
        size_t persistL2Size = prop.persistingL2CacheMaxSize;
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistL2Size);
        
        // LOG_SYSTEM_INFO("Enabled persistent L2 cache: {} MB", 
        //                persistL2Size / (1024 * 1024));
    }
    
    // Enable tensor core usage for supported operations
    if (config_.enable_tensor_cores && prop.major >= 7) {
        at::globalContext().setAllowTF32CuBLAS(true);
        at::globalContext().setAllowTF32CuDNN(true);
        // LOG_SYSTEM_INFO("Enabled TensorCore acceleration for compute capability {}.{}", 
        //                prop.major, prop.minor);
    }
}

bool GPUOptimizer::captureCudaGraph(
    const std::string& graph_id,
    std::function<torch::Tensor()> forward_fn,
    const torch::Tensor& example_input
) {
    if (!config_.enable_cuda_graphs || !torch::cuda::is_available()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
    
    // Check if graph already exists
    auto it = cuda_graphs_.find(graph_id);
    if (it != cuda_graphs_.end() && it->second.is_valid) {
        return true;
    }
    
    // Create new graph handle
    CudaGraphHandle& handle = cuda_graphs_[graph_id];
    handle.input_shape = example_input.sizes().vec();
    
    // Warmup phase
    if (handle.warmup_count < config_.cuda_graph_warmup_steps) {
        handle.warmup_count++;
        // LOG_SYSTEM_DEBUG("CUDA Graph warmup {}/{} for {}", 
        //                 handle.warmup_count, config_.cuda_graph_warmup_steps, graph_id);
        return false;
    }
    
    // Capture the graph
    cudaStream_t capture_stream = getCurrentStream();
    
    // Begin capture
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    
    try {
        // Execute the forward function
        torch::NoGradGuard no_grad;
        auto output = forward_fn();
        
        // Ensure all operations are captured
        cudaStreamSynchronize(capture_stream);
        
        // End capture
        cudaStreamEndCapture(capture_stream, &handle.graph);
        
        // Create executable graph
        cudaGraphInstantiate(&handle.exec, handle.graph, nullptr, nullptr, 0);
        
        handle.is_valid = true;
        // LOG_SYSTEM_INFO("Successfully captured CUDA graph for {}", graph_id);
        
        return true;
    } catch (const std::exception& e) {
        LOG_SYSTEM_ERROR("Failed to capture CUDA graph for {}: {}", graph_id, e.what());
        cudaStreamEndCapture(capture_stream, &handle.graph);
        return false;
    }
}

torch::Tensor GPUOptimizer::executeCudaGraph(
    const std::string& graph_id,
    const torch::Tensor& input
) {
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
    
    auto it = cuda_graphs_.find(graph_id);
    if (it == cuda_graphs_.end() || !it->second.is_valid) {
        cuda_graph_misses_.fetch_add(1);
        throw std::runtime_error("CUDA graph not found: " + graph_id);
    }
    
    // Verify input shape matches
    auto& handle = it->second;
    if (input.sizes().vec() != handle.input_shape) {
        cuda_graph_misses_.fetch_add(1);
        throw std::runtime_error("Input shape mismatch for CUDA graph");
    }
    
    // Execute the graph
    cudaGraphLaunch(handle.exec, getCurrentStream());
    cuda_graph_hits_.fetch_add(1);
    
    // Return placeholder - actual output handling would need to be implemented
    return input;
}

bool GPUOptimizer::isCudaGraphAvailable(const std::string& graph_id) const {
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
    auto it = cuda_graphs_.find(graph_id);
    return it != cuda_graphs_.end() && it->second.is_valid;
}

torch::jit::Module GPUOptimizer::optimizeWithTorchScript(
    torch::nn::Module& model,
    const std::vector<int64_t>& example_input_shape,
    bool optimize_for_inference
) {
    if (!config_.enable_torch_script) {
        // Return empty module when disabled
        return torch::jit::Module();
    }
    
    // IMPORTANT: torch::jit::trace is NOT available in C++/libtorch
    // Models must be traced/scripted in Python and then loaded in C++
    // This function is kept for API compatibility but returns empty module
    
    LOG_SYSTEM_WARN("Direct TorchScript conversion is not available in libtorch C++. "
                    "Models must be traced/scripted in Python first, then loaded via torch::jit::load()");
    
    // For future implementation:
    // 1. Pre-trace models in Python using torch.jit.trace or torch.jit.script
    // 2. Save them as .pt files
    // 3. Load them here using torch::jit::load("model.pt")
    // 4. Apply optimizations like torch::jit::optimize_for_inference
    
    return torch::jit::Module();
}

torch::jit::Module GPUOptimizer::loadTorchScriptModel(
    const std::string& model_path,
    bool optimize_for_inference,
    torch::Device device
) {
    try {
        // Load the traced/scripted model
        torch::jit::Module model = torch::jit::load(model_path);
        
        // Move to specified device
        model.to(device);
        
        // Set to eval mode
        model.eval();
        
        if (optimize_for_inference && torch::jit::optimize_for_inference) {
            // Apply inference optimizations
            model = torch::jit::optimize_for_inference(model);
            // LOG_SYSTEM_INFO("Applied TorchScript inference optimizations to model from: {}", model_path);
        }
        
        // Cache the model
        {
            std::lock_guard<std::mutex> lock(torch_script_mutex_);
            torch_script_models_[model_path] = model;
        }
        
        // LOG_SYSTEM_INFO("Successfully loaded TorchScript model from: {}", model_path);
        return model;
        
    } catch (const c10::Error& e) {
        LOG_SYSTEM_ERROR("Failed to load TorchScript model from {}: {}", model_path, e.what());
        throw std::runtime_error("Failed to load TorchScript model: " + std::string(e.what()));
    }
}

std::unique_ptr<GPUOptimizer::DynamicBatchAccumulator> 
GPUOptimizer::createBatchAccumulator(int optimal_size, int max_size) {
    return std::make_unique<DynamicBatchAccumulator>(this, optimal_size, max_size);
}

// DynamicBatchAccumulator implementation
GPUOptimizer::DynamicBatchAccumulator::DynamicBatchAccumulator(
    GPUOptimizer* optimizer, int optimal_size, int max_size
) : optimizer_(optimizer), optimal_size_(optimal_size), max_size_(max_size),
    current_target_size_(optimal_size) {
    pending_inputs_.reserve(max_size);
    request_ids_.reserve(max_size);
}

void GPUOptimizer::DynamicBatchAccumulator::addInput(
    torch::Tensor input, size_t request_id
) {
    if (pending_inputs_.empty()) {
        first_input_time_ = std::chrono::steady_clock::now();
    }
    
    pending_inputs_.push_back(input);
    request_ids_.push_back(request_id);
}

bool GPUOptimizer::DynamicBatchAccumulator::shouldProcess() const {
    if (pending_inputs_.empty()) {
        return false;
    }
    
    // Process if we've reached target size
    if (pending_inputs_.size() >= current_target_size_) {
        return true;
    }
    
    // Process if max size reached
    if (pending_inputs_.size() >= max_size_) {
        return true;
    }
    
    // Process if timeout reached
    auto elapsed = std::chrono::steady_clock::now() - first_input_time_;
    if (elapsed >= MAX_WAIT_TIME) {
        return true;
    }
    
    return false;
}

std::pair<torch::Tensor, std::vector<size_t>> 
GPUOptimizer::DynamicBatchAccumulator::extractBatch() {
    if (pending_inputs_.empty()) {
        return {torch::Tensor(), std::vector<size_t>()};
    }
    
    // Stack inputs into batch
    torch::Tensor batch = torch::stack(pending_inputs_);
    std::vector<size_t> ids = request_ids_;
    
    reset();
    
    return {batch, ids};
}

void GPUOptimizer::DynamicBatchAccumulator::reset() {
    pending_inputs_.clear();
    request_ids_.clear();
}

void GPUOptimizer::DynamicBatchAccumulator::updateOptimalSize(
    int queue_depth, float gpu_utilization
) {
    // Adaptive sizing based on queue pressure
    if (queue_depth > 1000) {
        // High pressure - increase batch size
        current_target_size_ = std::min(max_size_, optimal_size_ * 2);
    } else if (queue_depth > 500) {
        // Moderate pressure
        current_target_size_ = std::min(max_size_, (optimal_size_ * 3) / 2);
    } else if (queue_depth < 100) {
        // Low pressure - prioritize latency
        current_target_size_ = std::max(optimal_size_ / 2, 16);
    } else {
        // Normal operation
        current_target_size_ = optimal_size_;
    }
    
    // Adjust based on GPU utilization
    if (gpu_utilization < 50.0f) {
        // GPU underutilized - increase batch size
        current_target_size_ = std::min(max_size_, (current_target_size_ * 3) / 2);
    } else if (gpu_utilization > 90.0f) {
        // GPU saturated - reduce batch size
        current_target_size_ = std::max(16, (current_target_size_ * 2) / 3);
    }
}

void GPUOptimizer::cleanupResources() {
    // Clean up CUDA graphs
    {
        std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
        for (auto& [id, handle] : cuda_graphs_) {
            if (handle.graph) cudaGraphDestroy(handle.graph);
            if (handle.exec) cudaGraphExecDestroy(handle.exec);
        }
        cuda_graphs_.clear();
    }
    
    // Destroy CUDA streams
    for (auto stream : cuda_streams_) {
        cudaStreamDestroy(stream);
    }
    cuda_streams_.clear();
    
    // Clean up buffer pair events and clear tensor cache
    if (tensor_cache_) {
        // The ~BufferPair() destructor now handles event destruction.
        tensor_cache_->buffer_pairs.clear();
        tensor_cache_->gpu_tensors.clear();
        tensor_cache_->cpu_pinned_tensors.clear();
    }
    
    // Force GPU memory cleanup
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

// Global instance
GPUOptimizer& getGlobalGPUOptimizer() {
    static GPUOptimizer global_optimizer;
    return global_optimizer;
}

} // namespace nn
} // namespace alphazero