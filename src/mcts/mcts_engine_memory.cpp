#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/advanced_memory_pool.h"
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
    
    SPDLOG_DEBUG("MCTSEngine: Starting tree cleanup");
    
    // Store the current tree statistics before cleanup
    size_t nodes_before = 0;
    if (root_) {
        // Count nodes if we need the statistic
        // nodes_before = countTreeNodes(root_);
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
    
    // Force GPU memory cleanup
#ifdef TORCH_CUDA_AVAILABLE
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
    
    SPDLOG_DEBUG("MCTSEngine: Tree cleanup completed");
}

void MCTSEngine::cleanupPendingEvaluations() {
    SPDLOG_DEBUG("MCTSEngine: Cleaning up pending evaluations");
    
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
    SPDLOG_DEBUG("MCTSEngine: Resetting for new search");
    
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
    
    // Clear memory pools if they exist
    if (memory_pool_) {
        // Get current pool stats before clearing
        auto pool_stats = memory_pool_->getStats();
        SPDLOG_DEBUG("MCTSEngine: Memory pool stats - nodes in use: {}, nodes available: {}, "
                     "states in use: {}, states available: {}",
                     pool_stats.nodes_in_use, pool_stats.nodes_available,
                     pool_stats.states_in_use, pool_stats.states_available);
        
        // Note: We don't clear the pool itself as memory may be reused
        // but we should ensure no allocations are marked as in use
    }
    
    // Additional GPU memory cleanup
#ifdef TORCH_CUDA_AVAILABLE
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
        
        // Get GPU memory stats if available
        try {
            size_t gpu_allocated = c10::cuda::CUDACachingAllocator::currentMemoryAllocated(0);
            size_t gpu_reserved = c10::cuda::CUDACachingAllocator::currentMemoryCached(0);
            SPDLOG_DEBUG("MCTSEngine: GPU memory - allocated: {:.2f}MB, reserved: {:.2f}MB",
                         gpu_allocated / (1024.0 * 1024.0), gpu_reserved / (1024.0 * 1024.0));
        } catch (...) {
            // Ignore errors getting GPU stats
        }
    }
#endif
    
    SPDLOG_DEBUG("MCTSEngine: Reset completed");
}

} // namespace mcts
} // namespace alphazero