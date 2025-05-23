#include "mcts/mcts_engine.h"
#include "mcts/memory_pressure_monitor.h"
#include "utils/gamestate_pool.h"
#include "utils/logger.h"
#include <iostream>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

void MCTSEngine::handleMemoryPressure(MemoryPressureMonitor::PressureLevel level) {
    switch (level) {
        case MemoryPressureMonitor::PressureLevel::Warning:
            LOG_SYSTEM_WARN("Memory pressure WARNING: Starting preventive cleanup");
            
            // Clear game state pools
            utils::GameStatePoolManager::getInstance().clearAllPools();
            
            // Compact node pool if possible
            if (node_pool_ && node_pool_->shouldCompact()) {
                node_pool_->compact();
            }
            
            // Clear GPU cache
            #ifdef WITH_TORCH
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            #endif
            break;
            
        case MemoryPressureMonitor::PressureLevel::Critical:
            LOG_SYSTEM_ERROR("Memory pressure CRITICAL: Aggressive cleanup initiated");
            
            // Clear all pools immediately
            utils::GameStatePoolManager::getInstance().clearAllPools();
            
            // Force node pool compaction
            if (node_pool_) {
                node_pool_->compact();
            }
            
            // Clear transposition table
            if (transposition_table_) {
                transposition_table_->clear();
            }
            
            // Force GPU cleanup
            #ifdef WITH_TORCH
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            #endif
            
            // Force garbage collection
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            break;
            
        case MemoryPressureMonitor::PressureLevel::Emergency:
            LOG_SYSTEM_ERROR("Memory pressure EMERGENCY: Stopping search");
            
            // Emergency shutdown
            shutdown_.store(true);
            search_running_.store(false);
            
            // Clear everything possible
            cleanupTree();
            resetForNewSearch();
            
            // Clear all memory
            utils::GameStatePoolManager::getInstance().clearAllPools();
            if (node_pool_) {
                node_pool_->clear();
            }
            if (transposition_table_) {
                transposition_table_->clear();
            }
            
            #ifdef WITH_TORCH
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            #endif
            break;
            
        default:
            // Normal level - no action needed
            break;
    }
}

} // namespace mcts
} // namespace alphazero