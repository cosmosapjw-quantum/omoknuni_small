#include "mcts/gpu_virtual_loss.h"
#include "utils/logger.h"
#include <algorithm>
#include <cmath>

namespace alphazero {
namespace mcts {

// Forward declarations of CUDA kernels
extern "C" {
    void launchApplyVirtualLossKernelCuda(
        const int* paths,
        int* virtual_loss_tensor,
        const int* path_lengths,
        int batch_size,
        int max_depth,
        int num_nodes,
        int num_actions,
        int virtual_loss_value,
        cudaStream_t stream
    );
    
    void launchRemoveVirtualLossKernelCuda(
        const int* paths,
        int* virtual_loss_tensor,
        const int* path_lengths,
        int batch_size,
        int max_depth,
        int num_nodes,
        int num_actions,
        cudaStream_t stream
    );
    
    void launchApplyVirtualLossToUCBKernel(
        float* Q_tensor,
        const int* virtual_loss_tensor,
        float virtual_loss_weight,
        int total_elements,
        cudaStream_t stream
    );
}

GPUVirtualLoss::GPUVirtualLoss(const Config& config) : config_(config) {
    if (config.enable_adaptive && torch::cuda::is_available()) {
        // Pre-compute adaptive factors for different visit counts
        adaptive_vl_factors_ = torch::zeros({1000}, torch::kFloat32).cuda();
        
        auto factors = adaptive_vl_factors_.accessor<float, 1>();
        for (int i = 0; i < 1000; ++i) {
            // Logarithmic scaling to encourage exploration of less visited nodes
            float factor = 1.0f + config.adaptive_factor * std::log1p(i);
            factors[i] = std::min(factor, static_cast<float>(config.max_virtual_loss) / config.base_virtual_loss);
        }
    }
    
    // LOG_SYSTEM_INFO("GPU Virtual Loss initialized with:");
    // LOG_SYSTEM_INFO("  - Base virtual loss: {}", config.base_virtual_loss);
    // LOG_SYSTEM_INFO("  - Adaptive: {}", config.enable_adaptive);
    // LOG_SYSTEM_INFO("  - Max virtual loss: {}", config.max_virtual_loss);
}

GPUVirtualLoss::~GPUVirtualLoss() {
    // LOG_SYSTEM_INFO("GPU Virtual Loss stats:");
    // LOG_SYSTEM_INFO("  - Total applications: {}", stats_.total_applications.load());
    // LOG_SYSTEM_INFO("  - Total removals: {}", stats_.total_removals.load());
    // LOG_SYSTEM_INFO("  - Max concurrent VL: {}", stats_.max_concurrent_vl.load());
}

void GPUVirtualLoss::batchApplyVirtualLoss(
    const torch::Tensor& paths,
    torch::Tensor& virtual_loss_tensor,
    const std::vector<int>& path_lengths,
    cudaStream_t stream) {
    
    if (!config_.use_gpu_atomics) {
        // CPU fallback
        auto paths_cpu = paths.cpu();
        auto vl_cpu = virtual_loss_tensor.cpu();
        
        auto paths_accessor = paths_cpu.accessor<int32_t, 3>();
        auto vl_accessor = vl_cpu.accessor<int32_t, 3>();
        
        int batch_size = paths.size(0);
        int max_depth = paths.size(1);  // Used in kernel call
        
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < path_lengths[b]; ++d) {
                int node_idx = paths_accessor[b][d][0];
                int action_idx = paths_accessor[b][d][1];
                
                if (node_idx >= 0 && action_idx >= 0) {
                    vl_accessor[b][node_idx][action_idx] += config_.base_virtual_loss;
                }
            }
        }
        
        virtual_loss_tensor.copy_(vl_cpu);
    } else {
        // GPU kernel
        int batch_size = paths.size(0);
        int max_depth = paths.size(1);  // Used in kernel call
        int num_nodes = virtual_loss_tensor.size(1);
        int num_actions = virtual_loss_tensor.size(2);
        
        // Convert path lengths to GPU tensor
        auto path_lengths_tensor = torch::tensor(path_lengths, torch::kInt32).cuda();
        
        launchApplyVirtualLossKernelCuda(
            paths.data_ptr<int>(),
            virtual_loss_tensor.data_ptr<int>(),
            path_lengths_tensor.data_ptr<int>(),
            batch_size,
            max_depth,
            num_nodes,
            num_actions,
            config_.base_virtual_loss,
            stream
        );
        
        stats_.total_applications += batch_size;
    }
}

void GPUVirtualLoss::batchRemoveVirtualLoss(
    const torch::Tensor& paths,
    torch::Tensor& virtual_loss_tensor,
    const std::vector<int>& path_lengths,
    cudaStream_t stream) {
    
    if (!config_.use_gpu_atomics) {
        // CPU fallback
        auto paths_cpu = paths.cpu();
        auto vl_cpu = virtual_loss_tensor.cpu();
        
        auto paths_accessor = paths_cpu.accessor<int32_t, 3>();
        auto vl_accessor = vl_cpu.accessor<int32_t, 3>();
        
        int batch_size = paths.size(0);
        int max_depth = paths.size(1);  // Used in kernel call
        
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < path_lengths[b]; ++d) {
                int node_idx = paths_accessor[b][d][0];
                int action_idx = paths_accessor[b][d][1];
                
                if (node_idx >= 0 && action_idx >= 0) {
                    vl_accessor[b][node_idx][action_idx] = 
                        std::max(0, vl_accessor[b][node_idx][action_idx] - config_.base_virtual_loss);
                }
            }
        }
        
        virtual_loss_tensor.copy_(vl_cpu);
    } else {
        // GPU kernel
        int batch_size = paths.size(0);
        int max_depth = paths.size(1);  // Used in kernel call
        int num_nodes = virtual_loss_tensor.size(1);
        int num_actions = virtual_loss_tensor.size(2);
        
        // Convert path lengths to GPU tensor
        auto path_lengths_tensor = torch::tensor(path_lengths, torch::kInt32).cuda();
        
        launchRemoveVirtualLossKernelCuda(
            paths.data_ptr<int>(),
            virtual_loss_tensor.data_ptr<int>(),
            path_lengths_tensor.data_ptr<int>(),
            batch_size,
            max_depth,
            num_nodes,
            num_actions,
            stream
        );
        
        stats_.total_removals += batch_size;
    }
}

int GPUVirtualLoss::computeAdaptiveVirtualLoss(
    int node_visits,
    int parent_visits,
    float prior_probability) const {
    
    if (!config_.enable_adaptive) {
        return config_.base_virtual_loss;
    }
    
    // Base virtual loss
    int vl = config_.base_virtual_loss;
    
    // Increase virtual loss for frequently visited nodes
    if (node_visits > 0 && parent_visits > 0) {
        float visit_ratio = static_cast<float>(node_visits) / parent_visits;
        
        // If this node is visited more than expected by prior, increase VL
        if (visit_ratio > prior_probability * 2.0f) {
            vl = static_cast<int>(vl * config_.adaptive_factor);
        }
    }
    
    // Increase virtual loss based on absolute visit count
    if (node_visits < 1000 && adaptive_vl_factors_.defined()) {
        auto factors = adaptive_vl_factors_.accessor<float, 1>();
        vl = static_cast<int>(vl * factors[node_visits]);
    } else if (node_visits >= 1000) {
        // For very visited nodes, use maximum virtual loss
        vl = config_.max_virtual_loss;
    }
    
    return std::min(vl, config_.max_virtual_loss);
}

void GPUVirtualLoss::applyVirtualLossToUCB(
    torch::Tensor& Q_tensor,
    const torch::Tensor& virtual_loss_tensor,
    float virtual_loss_weight) {
    
    if (!torch::cuda::is_available()) {
        // CPU version
        Q_tensor -= virtual_loss_tensor.to(torch::kFloat32) * virtual_loss_weight;
    } else {
        // GPU kernel
        int total_elements = Q_tensor.numel();
        
        launchApplyVirtualLossToUCBKernel(
            Q_tensor.data_ptr<float>(),
            virtual_loss_tensor.data_ptr<int>(),
            virtual_loss_weight,
            total_elements,
            nullptr  // Default stream
        );
    }
}

// HybridVirtualLossTracker implementation

HybridVirtualLossTracker::HybridVirtualLossTracker(size_t max_nodes) 
    : max_nodes_(max_nodes) {
    virtual_losses_ = std::make_unique<std::atomic<int>[]>(max_nodes);
    for (size_t i = 0; i < max_nodes; ++i) {
        virtual_losses_[i].store(0);
    }
}

void HybridVirtualLossTracker::applyVirtualLoss(size_t node_id, int amount) {
    if (node_id < max_nodes_) {
        virtual_losses_[node_id].fetch_add(amount, std::memory_order_relaxed);
        total_applications_.fetch_add(1, std::memory_order_relaxed);
    }
}

void HybridVirtualLossTracker::removeVirtualLoss(size_t node_id, int amount) {
    if (node_id < max_nodes_) {
        int current = virtual_losses_[node_id].fetch_sub(amount, std::memory_order_relaxed);
        if (current < amount) {
            virtual_losses_[node_id].store(0, std::memory_order_relaxed);
        }
        total_removals_.fetch_add(1, std::memory_order_relaxed);
    }
}

int HybridVirtualLossTracker::getVirtualLoss(size_t node_id) const {
    if (node_id < max_nodes_) {
        return virtual_losses_[node_id].load(std::memory_order_relaxed);
    }
    return 0;
}

void HybridVirtualLossTracker::syncToGPU(torch::Tensor& gpu_virtual_loss_tensor) {
    // Prepare CPU tensor
    auto cpu_tensor = torch::zeros_like(gpu_virtual_loss_tensor, torch::kCPU);
    // auto accessor = cpu_tensor.accessor<int32_t, 3>();  // Unused in sync operation
    
    // Copy virtual losses to tensor
    // Note: This assumes a specific mapping between node IDs and tensor indices
    // In practice, you'd need a proper mapping mechanism
    
    gpu_virtual_loss_tensor.copy_(cpu_tensor);
}

void HybridVirtualLossTracker::syncFromGPU(const torch::Tensor& gpu_virtual_loss_tensor) {
    auto cpu_tensor = gpu_virtual_loss_tensor.cpu();
    // auto accessor = cpu_tensor.accessor<int32_t, 3>();  // Unused in sync operation
    
    // Update atomic counters from tensor
    // Note: This assumes a specific mapping between tensor indices and node IDs
}

void HybridVirtualLossTracker::clear() {
    for (size_t i = 0; i < max_nodes_; ++i) {
        virtual_losses_[i].store(0, std::memory_order_relaxed);
    }
}

} // namespace mcts
} // namespace alphazero