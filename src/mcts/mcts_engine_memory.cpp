#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
// #include "mcts/advanced_memory_pool.h" // Removed
#include "utils/debug_monitor.h"
#include "utils/gpu_memory_manager.h"
#include "utils/logger.h"
#include <iostream>
#include <queue>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

void MCTSEngine::cleanupTree() {
    if (!root_) {
        return;
    }
    
    // SPDLOG_DEBUG("MCTSEngine: Starting tree cleanup");
    
    // Store the current tree statistics before cleanup
    if (root_) {
        // Count nodes if we need the statistic
        // size_t nodes_before = countTreeNodes(root_);
    }
    
    // Clear all children of the root node recursively
    // This is the main source of memory accumulation
    root_->getChildren().clear();
    
    // Clear transposition table if used
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    // Reset root node statistics but keep it for potential reuse
    root_ = nullptr;
    
    // CRITICAL FIX: Always compact node pool to release unused memory
    if (node_pool_) {
        node_pool_->compact();
    }
    
    // Force GPU memory cleanup - enhanced version
#ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        // Empty all GPU caches
        c10::cuda::CUDACachingAllocator::emptyCache();
        
        // Force synchronization to ensure cleanup completes
        torch::cuda::synchronize();
        
        // Additional cleanup for GPU memory pools
        if (gpu_memory_pool_) {
            gpu_memory_pool_->trim(0.5f);  // Keep only 50% of allocated memory
        }
    }
#endif
    
    // SPDLOG_DEBUG("MCTSEngine: Tree cleanup completed");
}

void MCTSEngine::cleanupPendingEvaluations() {
    // SPDLOG_DEBUG("MCTSEngine: Cleaning up pending evaluations");
    
    // Reset pending evaluation counter
    pending_evaluations_.store(0);
    
    // UnifiedInferenceServer was removed in simplification
    
    // BurstCoordinator was removed in simplification
    
    // Reset batch counters
    batch_counter_.store(0);
    total_leaves_generated_.store(0);
    total_results_processed_.store(0);
}

void MCTSEngine::resetForNewSearch() {
    // SPDLOG_DEBUG("MCTSEngine: Resetting for new search");
    
    // Clean up the tree
    cleanupTree();
    
    // Clean up pending evaluations
    cleanupPendingEvaluations();
    
    // Reset search flags
    search_running_.store(false);
    active_simulations_.store(0);
    shutdown_.store(false);
    
    // Clear statistics
    last_stats_ = MCTSStats();
    
    // Memory pool removed
    /*
    if (memory_pool_) {
        // Get current pool stats before clearing
        auto pool_stats = memory_pool_->getStats();
        // SPDLOG_DEBUG("MCTSEngine: Memory pool stats - nodes in use: {}, nodes available: {}, "
        //              "states in use: {}, states available: {}",
        //              pool_stats.nodes_in_use, pool_stats.nodes_available,
        //              pool_stats.states_in_use, pool_stats.states_available);*/
        
        // Note: We don't clear the pool itself as memory may be reused
        // but we should ensure no allocations are marked as in use
    //}
    
    // CRITICAL FIX: Always compact node pool during reset
    if (node_pool_) {
        node_pool_->compact();
    }
    
    // Additional GPU memory cleanup - enhanced version
#ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        // Empty all GPU caches
        c10::cuda::CUDACachingAllocator::emptyCache();
        
        // Force synchronization
        torch::cuda::synchronize();
        
        // Additional cleanup for CUDA caches
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
        
        // Get GPU memory stats if available
        try {
            // PyTorch 2.0+ changed the API
            size_t gpu_allocated = 0;
            size_t gpu_reserved = 0;
            try {
                auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                gpu_allocated = stats.allocated_bytes[0].current;
                gpu_reserved = stats.reserved_bytes[0].current;
                // SPDLOG_DEBUG("MCTSEngine: GPU memory - allocated: {:.2f}MB, reserved: {:.2f}MB",
                //              gpu_allocated / (1024.0 * 1024.0), gpu_reserved / (1024.0 * 1024.0));
            } catch (...) {
                // Fallback for older versions or errors
                gpu_reserved = 0;
            }
            
            // If GPU memory usage is high, force more aggressive cleanup
            const size_t GPU_MEMORY_THRESHOLD_MB = 4000;  // 4GB threshold
            if (gpu_reserved > GPU_MEMORY_THRESHOLD_MB * 1024 * 1024) {
                SPDLOG_INFO("MCTSEngine: High GPU memory usage detected, forcing aggressive cleanup");
                c10::cuda::CUDACachingAllocator::emptyCache();
                if (gpu_memory_pool_) {
                    gpu_memory_pool_->trim(0.3f);  // Keep only 30% during high usage
                }
            }
        } catch (...) {
            // Ignore errors getting GPU stats
        }
    }
#endif
    
    // SPDLOG_DEBUG("MCTSEngine: Reset completed");
}

} // namespace mcts
} // namespace alphazero