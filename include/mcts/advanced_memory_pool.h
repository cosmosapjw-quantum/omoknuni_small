#pragma once

#include "core/igamestate.h"
#include "mcts_object_pool.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace alphazero {
namespace core {
    class IGameState;
}
}

namespace alphazero {
namespace mcts {

class MCTSNode;

// Configuration for the advanced memory pool
struct AdvancedMemoryPoolConfig {
    size_t initial_size = 10000;        // Initial number of objects to preallocate
    size_t growth_factor = 1.5;         // Factor by which to grow the pool when needed
    size_t max_pool_size = 1000000;     // Maximum size the pool can grow to
    bool enable_stats = true;           // Enable collection of usage statistics
};

class AdvancedMemoryPool {
public:
    AdvancedMemoryPool(const AdvancedMemoryPoolConfig& config = AdvancedMemoryPoolConfig());
    
    // Legacy constructor for backward compatibility
    AdvancedMemoryPool(size_t initial_node_capacity = 10000,
                      size_t initial_state_capacity = 5000);
    ~AdvancedMemoryPool();

    MCTSNode* acquireNode();
    void releaseNode(MCTSNode* node);
    
    std::unique_ptr<alphazero::core::IGameState> acquireGameState(const std::string& game_type);
    void releaseGameState(std::unique_ptr<alphazero::core::IGameState> state);
    
    // Create a shared_ptr to a game state from another state (for cloning)
    std::shared_ptr<alphazero::core::IGameState> allocateGameState(const alphazero::core::IGameState& source_state);
    
    void preallocateStates(const std::string& game_type, size_t count);
    
    struct PoolStats {
        size_t nodes_in_use;
        size_t nodes_available;
        size_t states_in_use;
        size_t states_available;
        size_t total_allocations;
        size_t total_deallocations;
        size_t peak_usage;
    };
    
    PoolStats getStats() const;
    void resetStats();
    
private:
    std::unique_ptr<alphazero::mcts::LockFreeObjectPool<MCTSNode>> node_pool_;
    AdvancedMemoryPoolConfig config_;
    
    struct GameStatePool {
        std::vector<std::unique_ptr<alphazero::core::IGameState>> available_states;
        mutable std::mutex mutex;
        std::atomic<size_t> in_use_count{0};
    };
    
    mutable std::mutex state_pools_mutex_;
    std::unordered_map<std::string, std::unique_ptr<GameStatePool>> state_pools_;
    
    mutable std::atomic<size_t> total_allocations_{0};
    mutable std::atomic<size_t> total_deallocations_{0};
    mutable std::atomic<size_t> peak_usage_{0};
    mutable std::atomic<size_t> current_usage_{0};
    
    std::unique_ptr<alphazero::core::IGameState> createGameState(const std::string& game_type);
    GameStatePool* getOrCreateStatePool(const std::string& game_type);
};

class PooledGameStateWrapper {
public:
    PooledGameStateWrapper(std::unique_ptr<alphazero::core::IGameState> state, AdvancedMemoryPool* pool)
        : state_(std::move(state)), pool_(pool) {}
    
    ~PooledGameStateWrapper() {
        if (state_ && pool_) {
            pool_->releaseGameState(std::move(state_));
        }
    }
    
    alphazero::core::IGameState* get() const { return state_.get(); }
    alphazero::core::IGameState* operator->() const { return state_.get(); }
    alphazero::core::IGameState& operator*() const { return *state_; }
    
    PooledGameStateWrapper(const PooledGameStateWrapper&) = delete;
    PooledGameStateWrapper& operator=(const PooledGameStateWrapper&) = delete;
    
    PooledGameStateWrapper(PooledGameStateWrapper&& other) noexcept
        : state_(std::move(other.state_)), pool_(other.pool_) {
        other.pool_ = nullptr;
    }
    
    PooledGameStateWrapper& operator=(PooledGameStateWrapper&& other) noexcept {
        if (this != &other) {
            if (state_ && pool_) {
                pool_->releaseGameState(std::move(state_));
            }
            state_ = std::move(other.state_);
            pool_ = other.pool_;
            other.pool_ = nullptr;
        }
        return *this;
    }

private:
    std::unique_ptr<alphazero::core::IGameState> state_;
    AdvancedMemoryPool* pool_;
};

} // namespace mcts
} // namespace alphazero