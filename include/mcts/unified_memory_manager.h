#ifndef ALPHAZERO_UNIFIED_MEMORY_MANAGER_H
#define ALPHAZERO_UNIFIED_MEMORY_MANAGER_H

#include <memory>
#include <atomic>
#include "core/igamestate.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Unified memory manager for MCTS game state management
 * 
 * Replaces complex memory pools with a simple, reliable approach
 * that eliminates memory fragmentation and reduces overhead.
 * 
 * Key benefits:
 * - No pool fragmentation
 * - Automatic cleanup via smart pointers
 * - Simple allocation tracking
 * - Minimal overhead
 * - Thread-safe without locks
 */
class UnifiedMemoryManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return UnifiedMemoryManager& The global memory manager instance
     */
    static UnifiedMemoryManager& getInstance();
    
    /**
     * @brief Clone a game state with memory tracking
     * 
     * @param state The state to clone
     * @return std::shared_ptr<core::IGameState> Cloned state with automatic cleanup
     */
    std::shared_ptr<core::IGameState> cloneGameState(const core::IGameState& state);
    
    /**
     * @brief Perform memory cleanup
     */
    void cleanup();
    
    /**
     * @brief Get current memory usage estimate
     * 
     * @return size_t Current memory usage in bytes
     */
    size_t getCurrentMemoryUsage() const;
    
    /**
     * @brief Get peak memory usage
     * 
     * @return size_t Peak memory usage in bytes
     */
    size_t getPeakMemoryUsage() const;
    
    /**
     * @brief Get total number of allocations
     * 
     * @return size_t Total allocations performed
     */
    size_t getTotalAllocations() const;
    
    /**
     * @brief Get total number of deallocations
     * 
     * @return size_t Total deallocations performed
     */
    size_t getTotalDeallocations() const;
    
    /**
     * @brief Get memory management efficiency
     * 
     * @return float Ratio of deallocations to allocations (closer to 1.0 is better)
     */
    float getMemoryEfficiency() const;

private:
    UnifiedMemoryManager();
    ~UnifiedMemoryManager();
    
    // Delete copy operations
    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
    
    // Statistics tracking
    std::atomic<size_t> total_allocations_;
    std::atomic<size_t> total_deallocations_;
    std::atomic<size_t> peak_memory_usage_;
    std::atomic<size_t> current_memory_usage_;
    
    // Helper methods
    size_t estimateStateSize(const core::IGameState& state) const;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_UNIFIED_MEMORY_MANAGER_H