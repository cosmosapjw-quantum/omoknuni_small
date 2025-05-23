#include "mcts/unified_memory_manager.h"
#include <iostream>
#include <algorithm>

namespace alphazero {
namespace mcts {

UnifiedMemoryManager& UnifiedMemoryManager::getInstance() {
    static UnifiedMemoryManager instance;
    return instance;
}

UnifiedMemoryManager::UnifiedMemoryManager() 
    : total_allocations_(0)
    , total_deallocations_(0)
    , peak_memory_usage_(0)
    , current_memory_usage_(0) {
    
    std::cout << "UnifiedMemoryManager: Initialized unified memory management" << std::endl;
}

UnifiedMemoryManager::~UnifiedMemoryManager() {
    cleanup();
    
    std::cout << "UnifiedMemoryManager: Shutdown complete. "
              << "Total allocations: " << total_allocations_.load()
              << ", deallocations: " << total_deallocations_.load()
              << ", peak usage: " << (peak_memory_usage_.load() / 1048576) << " MB" << std::endl;
}

std::shared_ptr<core::IGameState> UnifiedMemoryManager::cloneGameState(const core::IGameState& state) {
    try {
        // Simple direct cloning - avoid pool complexity
        auto cloned = state.clone();
        if (cloned) {
            // Track memory usage
            size_t state_size = estimateStateSize(state);
            current_memory_usage_.fetch_add(state_size, std::memory_order_relaxed);
            total_allocations_.fetch_add(1, std::memory_order_relaxed);
            
            // Update peak if necessary
            size_t current = current_memory_usage_.load(std::memory_order_relaxed);
            size_t peak = peak_memory_usage_.load(std::memory_order_relaxed);
            while (current > peak && !peak_memory_usage_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                peak = peak_memory_usage_.load(std::memory_order_relaxed);
            }
            
            // Wrap in shared_ptr with custom deleter for tracking
            return std::shared_ptr<core::IGameState>(
                cloned.release(),
                [this, state_size](core::IGameState* ptr) {
                    delete ptr;
                    current_memory_usage_.fetch_sub(state_size, std::memory_order_relaxed);
                    total_deallocations_.fetch_add(1, std::memory_order_relaxed);
                }
            );
        }
    } catch (const std::exception& e) {
        std::cerr << "UnifiedMemoryManager: Error cloning state: " << e.what() << std::endl;
    }
    
    return nullptr;
}

void UnifiedMemoryManager::cleanup() {
    // Force cleanup can't do much with simple approach
    // Just reset statistics
    std::cout << "UnifiedMemoryManager: Performing cleanup. Current usage: " 
              << (current_memory_usage_.load() / 1048576) << " MB" << std::endl;
    
    // Memory cleanup is handled by smart pointer destructors
    // This is much more reliable than manual pool management
}

size_t UnifiedMemoryManager::getCurrentMemoryUsage() const {
    return current_memory_usage_.load(std::memory_order_relaxed);
}

size_t UnifiedMemoryManager::getPeakMemoryUsage() const {
    return peak_memory_usage_.load(std::memory_order_relaxed);
}

size_t UnifiedMemoryManager::getTotalAllocations() const {
    return total_allocations_.load(std::memory_order_relaxed);
}

size_t UnifiedMemoryManager::getTotalDeallocations() const {
    return total_deallocations_.load(std::memory_order_relaxed);
}

float UnifiedMemoryManager::getMemoryEfficiency() const {
    size_t allocs = total_allocations_.load(std::memory_order_relaxed);
    size_t deallocs = total_deallocations_.load(std::memory_order_relaxed);
    
    if (allocs == 0) return 1.0f;
    
    return static_cast<float>(deallocs) / allocs;
}

size_t UnifiedMemoryManager::estimateStateSize(const core::IGameState& state) const {
    // Simple heuristic based on action space size and board complexity
    try {
        int action_space = state.getActionSpaceSize();
        
        // Base size estimate
        size_t base_size = 512; // 512 bytes base
        
        // Scale with action space
        size_t action_overhead = action_space * 8; // 8 bytes per action
        
        // Add board representation (rough estimate)
        size_t board_size = action_space * 4; // 4 bytes per position
        
        return base_size + action_overhead + board_size;
    } catch (...) {
        // Fallback estimate
        return 1024; // 1KB default
    }
}

} // namespace mcts
} // namespace alphazero