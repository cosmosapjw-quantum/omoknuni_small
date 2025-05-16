#include "utils/gamestate_pool.h"
#include "games/go/go_state.h"
#include "games/chess/chess_state.h"
#include "games/gomoku/gomoku_state.h"
#include <iostream>
#include <chrono>

namespace alphazero {
namespace utils {

GameStatePool::GameStatePool(size_t initial_size, 
                           std::function<std::unique_ptr<core::IGameState>()> factory)
    : factory_(factory), initial_size_(initial_size),
      last_cleanup_time_(std::chrono::steady_clock::now()) {
    
    // Pre-allocate initial pool
    for (size_t i = 0; i < initial_size; ++i) {
        pool_.push_back(factory_());
        total_allocated_.fetch_add(1);
    }
}

GameStatePool::~GameStatePool() {
    // Pool will be automatically cleaned up
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.clear();
}

std::unique_ptr<core::IGameState> GameStatePool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_acquires_.fetch_add(1);
    
    // Perform periodic cleanup
    if (total_acquires_ % 1000 == 0) {
        progressiveCleanup();
    }
    
    if (!pool_.empty()) {
        auto state = std::move(pool_.front());
        pool_.pop_front();
        pool_hits_.fetch_add(1);
        return state;
    }
    
    // Pool is empty, create a new object
    pool_misses_.fetch_add(1);
    total_allocated_.fetch_add(1);
    return factory_();
}

void GameStatePool::release(std::unique_ptr<core::IGameState> state) {
    if (!state) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_releases_.fetch_add(1);
    
    // FIX: Limit pool size to prevent unbounded growth
    const size_t max_pool_size = initial_size_ * 4;  // Allow up to 4x initial size
    if (pool_.size() < max_pool_size) {
        pool_.push_back(std::move(state));
    }
    // If pool is full, let the state be destroyed
}

std::unique_ptr<core::IGameState> GameStatePool::clone(const core::IGameState& source) {
    // For now, just use regular cloning since we can't safely reuse objects
    // without a proper assignment operator or copyFrom method
    return source.clone();
}

GameStatePool::Stats GameStatePool::getStats() const {
    Stats stats;
    stats.pool_size = pool_.size();
    stats.total_allocated = total_allocated_.load();
    stats.hits = pool_hits_.load();
    stats.misses = pool_misses_.load();
    return stats;
}

// GameStatePoolManager implementation
GameStatePoolManager& GameStatePoolManager::getInstance() {
    static GameStatePoolManager instance;
    return instance;
}

void GameStatePoolManager::initializePool(core::GameType game_type, size_t pool_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto factory = [game_type]() -> std::unique_ptr<core::IGameState> {
        switch (game_type) {
            case core::GameType::GO:
                return std::make_unique<games::go::GoState>();
            case core::GameType::CHESS:
                return std::make_unique<games::chess::ChessState>();
            case core::GameType::GOMOKU:
                return std::make_unique<games::gomoku::GomokuState>();
            default:
                throw std::runtime_error("Unknown game type for pool creation");
        }
    };
    
    pools_[static_cast<int>(game_type)] = 
        std::make_unique<GameStatePool>(pool_size, factory);
}

std::unique_ptr<core::IGameState> GameStatePoolManager::cloneState(const core::IGameState& state) {
    auto game_type = state.getGameType();
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pools_.find(static_cast<int>(game_type));
    
    if (it != pools_.end() && it->second) {
        return it->second->clone(state);
    }
    
    // No pool available, use regular cloning
    return state.clone();
}

bool GameStatePoolManager::hasPool(core::GameType game_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pools_.find(static_cast<int>(game_type)) != pools_.end();
}

void GameStatePoolManager::clearAllPools() {
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.clear();
}

void GameStatePool::trimToSize(size_t target_size) {
    // No need to lock here, already called from locked context
    while (pool_.size() > target_size && !pool_.empty()) {
        pool_.pop_back();
    }
}

void GameStatePool::progressiveCleanup() {
    // Already under lock when called
    auto now = std::chrono::steady_clock::now();
    
    // Only cleanup once per minute
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_time_).count() < 60) {
        return;
    }
    
    last_cleanup_time_ = now;
    
    // Calculate usage metrics
    uint64_t acquires = total_acquires_.load();
    uint64_t releases = total_releases_.load();
    uint64_t hits = pool_hits_.load();
    uint64_t misses = pool_misses_.load();
    
    if (acquires == 0) return;
    
    // Calculate hit rate
    double hit_rate = static_cast<double>(hits) / (hits + misses);
    
    // If hit rate is low, reduce pool size as it's not being used effectively
    if (hit_rate < 0.3 && pool_.size() > initial_size_) {
        trimToSize(initial_size_);
    }
    // If pool is mostly full and hit rate is high, allow slight growth
    else if (hit_rate > 0.8 && pool_.size() < initial_size_ * 2) {
        // Allow pool to grow naturally
    }
    // If pool is too large and not being used much, trim it
    else if (pool_.size() > initial_size_ * 3) {
        trimToSize(initial_size_ * 2);
    }
}

} // namespace utils
} // namespace alphazero