// include/mcts/transposition_table.h
#ifndef ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H
#define ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <utility>
#include <cstddef>
#include "core/export_macros.h"
#include "parallel_hashmap/phmap.h"

namespace alphazero {
namespace mcts {

class MCTSNode;

/**
 * @brief Entry in the transposition table
 */
struct ALPHAZERO_API TranspositionEntry {
    // The MCTS node for this position (weak_ptr to avoid memory leaks)
    std::weak_ptr<MCTSNode> node;
    
    // The hash key of the position
    uint64_t hash;
    
    // The depth of the search when this entry was stored
    int depth;
    
    // The number of visits at the time of storage
    int visits;
    
    // Check if the node is still valid
    bool isValid() const { return !node.expired(); }
    
    // Constructor
    TranspositionEntry(std::weak_ptr<MCTSNode> n, uint64_t h, int d, int v);
};

/**
 * @brief Transposition table for MCTS
 * 
 * A thread-safe table that allows the MCTS algorithm to recognize and
 * reuse previously searched positions, significantly improving efficiency.
 * This implementation aggressively uses the parallel-hashmap library for high-performance
 * concurrent access.
 */
class ALPHAZERO_API TranspositionTable {
public:
    /**
     * @brief Constructor
     * 
     * @param size_mb Size of the table in megabytes (approximate)
     * @param num_shards Number of internal shards for parallelism (0 for auto)
     */
    explicit TranspositionTable(size_t size_mb = 128, size_t /*num_shards*/ = 0);
    
    /**
     * @brief Get a node from the table
     * 
     * Thread-safe method to retrieve a node by its hash.
     * 
     * @param hash Position hash
     * @return Shared pointer to the node, or nullptr if not found
     */
    std::shared_ptr<MCTSNode> get(uint64_t hash);
    
    /**
     * @brief Store a node in the table
     * 
     * Thread-safe method to store a node in the table.
     * Uses a replacement policy that favors nodes with more visits.
     * 
     * @param hash Position hash
     * @param node The node to store (weak_ptr)
     * @param depth The current search depth
     */
    void store(uint64_t hash, std::weak_ptr<MCTSNode> node, int depth);
    
    /**
     * @brief Clear the table
     * 
     * Removes all entries from the table.
     */
    void clear();
    
    /**
     * @brief Get the number of entries in the table
     * 
     * @return Number of entries
     */
    size_t size() const;
    
    /**
     * @brief Get the capacity of the table
     * 
     * @return Maximum number of entries (approximate)
     */
    size_t capacity() const;
    
    /**
     * @brief Get the hit rate
     * 
     * @return Hit rate (0.0 to 1.0)
     */
    float hitRate() const;
    
    /**
     * @brief Reset hit statistics
     */
    void resetStats();
    
private:
    // Entry type for our hash map
    using EntryPtr = std::shared_ptr<TranspositionEntry>;
    
    // Thread-safe parallel hash map with built-in sharding
    // We use shared_ptr<TranspositionEntry> to maintain thread safety when erasing
    using HashMapType = phmap::parallel_flat_hash_map<uint64_t, EntryPtr>;
    
    // Main storage for transposition entries
    HashMapType entries_;
    
    // Thread-local cache to improve lookup performance
    static constexpr int MAX_THREADS = 64;
    struct ThreadLocalCache {
        // Recent lookups cache with limited size (unordered_map for O(1) lookups)
        std::unordered_map<uint64_t, std::weak_ptr<MCTSNode>> recent_lookups;
        
        // Statistics for this thread's cache
        size_t hits = 0;
        size_t misses = 0;
        
        // Padding to avoid false sharing between threads
        alignas(64) char padding[64];
    };
    std::array<ThreadLocalCache, MAX_THREADS> thread_caches_;
    
    // Target capacity (approximate)
    size_t capacity_;
    
    // Atomic counters for hit statistics
    std::atomic<size_t> hits_;
    std::atomic<size_t> misses_;
    
    // Mutex for clear operations (rare, so one global mutex is fine)
    mutable std::mutex clear_mutex_;
    
    /**
     * @brief Enforce the capacity limit
     * 
     * Removes the least valuable entries if the table exceeds capacity.
     * Uses a sampling approach for efficiency.
     */
    void enforceCapacityLimit();
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H