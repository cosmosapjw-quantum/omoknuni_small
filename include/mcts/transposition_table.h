// include/mcts/transposition_table.h
#ifndef ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H
#define ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "core/export_macros.h"

namespace alphazero {
namespace mcts {

class MCTSNode;

/**
 * @brief Entry in the transposition table
 */
struct ALPHAZERO_API TranspositionEntry {
    // The MCTS node for this position
    std::atomic<MCTSNode*> node;
    
    // The hash key of the position
    uint64_t hash;
    
    // The depth of the search when this entry was stored
    int depth;
    
    // The number of visits at the time of storage
    int visits;
    
    // A lock for thread-safe access with timeout capability
    std::timed_mutex lock;
    
    // Constructor
    TranspositionEntry(MCTSNode* n, uint64_t h, int d, int v);
};

/**
 * @brief Transposition table for MCTS
 * 
 * A thread-safe table that allows the MCTS algorithm to recognize and
 * reuse previously searched positions, significantly improving efficiency.
 */
class ALPHAZERO_API TranspositionTable {
public:
    /**
     * @brief Constructor
     * 
     * @param size_mb Size of the table in megabytes
     */
    explicit TranspositionTable(size_t size_mb = 128);
    
    /**
     * @brief Destructor - safely cleans up resources
     */
    ~TranspositionTable();
    
    /**
     * @brief Get a node from the table
     * 
     * @param hash Position hash
     * @return Pointer to the node, or nullptr if not found
     */
    MCTSNode* get(uint64_t hash);
    
    /**
     * @brief Store a node in the table
     * 
     * @param hash Position hash
     * @param node The node to store
     * @param depth The current search depth
     */
    void store(uint64_t hash, MCTSNode* node, int depth);
    
    /**
     * @brief Clear the table
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
     * @return Maximum number of entries
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
    // Table storage - unique_ptrs to entries
    std::vector<std::unique_ptr<TranspositionEntry>> entries_;
    
    // Hash to entry index mapping
    std::unordered_map<uint64_t, size_t> index_map_;
    
    // Size of the table (capacity in number of entries)
    size_t capacity_;
    
    // Hit statistics - atomic counters for thread safety
    std::atomic<size_t> hits_;
    std::atomic<size_t> misses_;
    
    // Locks for thread-safety - one per shard
    std::vector<std::unique_ptr<std::timed_mutex>> table_locks_;
    
    // Number of locks (shards) - higher number reduces contention
    static constexpr size_t NUM_LOCKS = 1024;
    
    // Flag to track if the table is being destroyed
    std::atomic<bool> is_shutdown_{false};
    
    // Get lock for a hash
    std::timed_mutex& getLock(uint64_t hash) {
        return *table_locks_[hash % NUM_LOCKS];
    }
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_TRANSPOSITION_TABLE_H