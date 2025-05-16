#ifndef ALPHAZERO_UTILS_GAMESTATE_POOL_H
#define ALPHAZERO_UTILS_GAMESTATE_POOL_H

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <functional>
#include <unordered_map>
#include "core/igamestate.h"
#include "core/export_macros.h"

namespace alphazero {
namespace utils {

/**
 * @brief Simple memory pool for GameState objects to reduce allocation overhead
 */
class ALPHAZERO_API GameStatePool {
public:
    /**
     * @brief Constructor
     * @param initial_size Number of objects to pre-allocate
     * @param factory Function to create new GameState objects
     */
    GameStatePool(size_t initial_size, 
                  std::function<std::unique_ptr<core::IGameState>()> factory);
    
    /**
     * @brief Destructor
     */
    ~GameStatePool();
    
    /**
     * @brief Acquire a GameState from the pool
     * @return Unique pointer to a GameState object
     */
    std::unique_ptr<core::IGameState> acquire();
    
    /**
     * @brief Release a GameState back to the pool
     * @param state The state to release (will be moved)
     */
    void release(std::unique_ptr<core::IGameState> state);
    
    /**
     * @brief Clone a state using an object from the pool
     * @param source The state to clone
     * @return Cloned state using pooled object
     */
    std::unique_ptr<core::IGameState> clone(const core::IGameState& source);
    
    /**
     * @brief Get pool statistics
     */
    struct Stats {
        size_t pool_size;
        size_t total_allocated;
        uint64_t hits;
        uint64_t misses;
    };
    
    Stats getStats() const;
    
    /**
     * @brief Trim pool to target size (removes excess objects)
     * @param target_size The target size for the pool
     */
    void trimToSize(size_t target_size);
    
    /**
     * @brief Perform progressive cleanup based on usage patterns
     */
    void progressiveCleanup();
    
private:
    mutable std::mutex mutex_;
    std::deque<std::unique_ptr<core::IGameState>> pool_;
    std::function<std::unique_ptr<core::IGameState>()> factory_;
    size_t initial_size_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<uint64_t> pool_hits_{0};
    std::atomic<uint64_t> pool_misses_{0};
    
    // Additional stats for progressive cleanup
    std::atomic<uint64_t> total_releases_{0};
    std::atomic<uint64_t> total_acquires_{0};
    std::chrono::steady_clock::time_point last_cleanup_time_;
};

/**
 * @brief Global pool manager for different game types
 */
class ALPHAZERO_API GameStatePoolManager {
public:
    static GameStatePoolManager& getInstance();
    
    /**
     * @brief Initialize pool for a specific game type
     * @param game_type The game type
     * @param pool_size Size of the pool
     */
    void initializePool(core::GameType game_type, size_t pool_size = 1024);
    
    /**
     * @brief Clone a state using the appropriate pool
     * @param state The state to clone
     * @return Cloned state
     */
    std::unique_ptr<core::IGameState> cloneState(const core::IGameState& state);
    
    /**
     * @brief Check if a pool exists for a game type
     * @param game_type The game type
     * @return true if pool exists
     */
    bool hasPool(core::GameType game_type) const;
    
    /**
     * @brief Clear all pools to free memory
     */
    void clearAllPools();
    
private:
    GameStatePoolManager() = default;
    ~GameStatePoolManager() = default;
    GameStatePoolManager(const GameStatePoolManager&) = delete;
    GameStatePoolManager& operator=(const GameStatePoolManager&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<GameStatePool>> pools_;
};

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_UTILS_GAMESTATE_POOL_H