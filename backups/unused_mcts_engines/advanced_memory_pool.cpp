#include "mcts/advanced_memory_pool.h"
#include "mcts/mcts_node.h"
#include "mcts/mcts_object_pool.h"
#include "core/game_export.h"
#include <algorithm>

namespace alphazero {
namespace mcts {

AdvancedMemoryPool::AdvancedMemoryPool(const AdvancedMemoryPoolConfig& config) {
    // Create node pool with REDUCED settings for memory efficiency
    // MCTSNode is typically around 256 bytes with 16-byte alignment
    size_t reduced_initial = std::min(config.initial_size, size_t(512));  // Max 512 initial
    size_t reduced_max = std::min(config.max_pool_size, size_t(2048));    // Max 2048 total
    node_pool_ = std::make_unique<alphazero::mcts::LockFreeObjectPool<MCTSNode>>(
        reduced_initial, reduced_max, 256, 16);
    
    // Store configuration for dynamic adjustments
    config_ = config;
}

AdvancedMemoryPool::AdvancedMemoryPool(size_t initial_node_capacity, 
                                       size_t initial_state_capacity) {
    // Create node pool with REDUCED conservative size estimates
    // MCTSNode is typically around 256 bytes with 16-byte alignment
    size_t reduced_initial = std::min(initial_node_capacity, size_t(512));  // Max 512 initial
    node_pool_ = std::make_unique<alphazero::mcts::LockFreeObjectPool<MCTSNode>>(
        reduced_initial, 1024, 256, 16);  // Reduced max from 2048 to 1024
    
    // Set default configuration
    config_.initial_size = initial_node_capacity;
    config_.max_pool_size = 2048;
    config_.growth_factor = 1.5;
}

AdvancedMemoryPool::~AdvancedMemoryPool() = default;

MCTSNode* AdvancedMemoryPool::acquireNode() {
    // MEMORY FIX: Add bounds checking to prevent unbounded growth
    size_t current = current_usage_.load(std::memory_order_relaxed);
    const size_t MAX_NODES = 5000; // CONSERVATIVE: Reduced from 100000 to prevent memory stacking
    
    if (current >= MAX_NODES) {
        // Return nullptr to signal memory pressure - caller should handle gracefully
        std::cout << "[MEMORY_POOL] Node allocation limit reached: " << current << "/" << MAX_NODES << std::endl;
        return nullptr;
    }
    
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    current = current_usage_.fetch_add(1, std::memory_order_relaxed) + 1;
    
    size_t peak = peak_usage_.load(std::memory_order_relaxed);
    while (current > peak && !peak_usage_.compare_exchange_weak(
        peak, current, std::memory_order_relaxed)) {}
    
    return node_pool_->acquire();
}

void AdvancedMemoryPool::releaseNode(MCTSNode* node) {
    if (node) {
        total_deallocations_.fetch_add(1, std::memory_order_relaxed);
        current_usage_.fetch_sub(1, std::memory_order_relaxed);
        node_pool_->release(node);
    }
}

std::unique_ptr<alphazero::core::IGameState> AdvancedMemoryPool::acquireGameState(const std::string& game_type) {
    GameStatePool* pool = getOrCreateStatePool(game_type);
    
    std::lock_guard<std::mutex> lock(pool->mutex);
    
    if (!pool->available_states.empty()) {
        auto state = std::move(pool->available_states.back());
        pool->available_states.pop_back();
        pool->in_use_count.fetch_add(1, std::memory_order_relaxed);
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        return state;
    }
    
    pool->in_use_count.fetch_add(1, std::memory_order_relaxed);
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    return createGameState(game_type);
}

void AdvancedMemoryPool::releaseGameState(std::unique_ptr<alphazero::core::IGameState> state) {
    if (!state) return;
    
    std::string game_type = alphazero::core::gameTypeToString(state->getGameType());
    GameStatePool* pool = getOrCreateStatePool(game_type);
    
    std::lock_guard<std::mutex> lock(pool->mutex);
    
    const size_t MAX_POOL_SIZE = 1000;
    if (pool->available_states.size() < MAX_POOL_SIZE) {
        pool->available_states.push_back(std::move(state));
    }
    
    pool->in_use_count.fetch_sub(1, std::memory_order_relaxed);
    total_deallocations_.fetch_add(1, std::memory_order_relaxed);
}

void AdvancedMemoryPool::preallocateStates(const std::string& game_type, size_t count) {
    GameStatePool* pool = getOrCreateStatePool(game_type);
    
    std::lock_guard<std::mutex> lock(pool->mutex);
    
    pool->available_states.reserve(pool->available_states.size() + count);
    
    for (size_t i = 0; i < count; ++i) {
        auto state = createGameState(game_type);
        if (state) {
            pool->available_states.push_back(std::move(state));
        }
    }
}

AdvancedMemoryPool::PoolStats AdvancedMemoryPool::getStats() const {
    PoolStats stats;
    
    auto node_stats = node_pool_->getStats();
    stats.nodes_in_use = node_stats.total_allocations - node_stats.pool_hits; // In use = allocated - returned
    stats.nodes_available = node_stats.current_pool_size;
    
    stats.states_in_use = 0;
    stats.states_available = 0;
    
    std::lock_guard<std::mutex> pools_lock(state_pools_mutex_);
    for (const auto& [game_type, pool] : state_pools_) {
        std::lock_guard<std::mutex> pool_lock(pool->mutex);
        stats.states_in_use += pool->in_use_count.load(std::memory_order_relaxed);
        stats.states_available += pool->available_states.size();
    }
    
    stats.total_allocations = total_allocations_.load(std::memory_order_relaxed);
    stats.total_deallocations = total_deallocations_.load(std::memory_order_relaxed);
    stats.peak_usage = peak_usage_.load(std::memory_order_relaxed);
    
    return stats;
}

void AdvancedMemoryPool::resetStats() {
    total_allocations_.store(0, std::memory_order_relaxed);
    total_deallocations_.store(0, std::memory_order_relaxed);
    peak_usage_.store(0, std::memory_order_relaxed);
    current_usage_.store(0, std::memory_order_relaxed);
    // Note: LockFreeObjectPool doesn't have resetStats method
}

std::shared_ptr<alphazero::core::IGameState> AdvancedMemoryPool::allocateGameState(
    const alphazero::core::IGameState& source_state) {
    
    // Get the game type from the source state
    alphazero::core::GameType game_type = source_state.getGameType();
    std::string game_type_str = alphazero::core::gameTypeToString(game_type);
    
    // Try to get a state from the pool
    auto unique_state = acquireGameState(game_type_str);
    
    if (!unique_state) {
        // If no state available in pool, create a fresh one by cloning directly
        return source_state.clone();
    }
    
    // Copy the state from source to our pooled state
    unique_state->copyFrom(source_state);
    
    // Create a shared_ptr with custom deleter to return the state to the pool
    auto self = this;
    return std::shared_ptr<alphazero::core::IGameState>(
        unique_state.release(),
        [self, game_type_str](alphazero::core::IGameState* state) {
            if (self && state) {
                self->releaseGameState(std::unique_ptr<alphazero::core::IGameState>(state));
            }
        }
    );
}

std::unique_ptr<alphazero::core::IGameState> AdvancedMemoryPool::createGameState(const std::string& game_type) {
    // Convert string to GameType enum and create game instance
    if (game_type == "gomoku") {
        return alphazero::core::GameFactory::createGame(alphazero::core::GameType::GOMOKU);
    } else if (game_type == "chess") {
        return alphazero::core::GameFactory::createGame(alphazero::core::GameType::CHESS);
    } else if (game_type == "go") {
        return alphazero::core::GameFactory::createGame(alphazero::core::GameType::GO);
    } else {
        // Default to gomoku if unknown type
        return alphazero::core::GameFactory::createGame(alphazero::core::GameType::GOMOKU);
    }
}

AdvancedMemoryPool::GameStatePool* AdvancedMemoryPool::getOrCreateStatePool(const std::string& game_type) {
    std::lock_guard<std::mutex> lock(state_pools_mutex_);
    
    auto it = state_pools_.find(game_type);
    if (it != state_pools_.end()) {
        return it->second.get();
    }
    
    auto pool = std::make_unique<GameStatePool>();
    GameStatePool* pool_ptr = pool.get();
    state_pools_[game_type] = std::move(pool);
    return pool_ptr;
}

} // namespace mcts
} // namespace alphazero