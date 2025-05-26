// include/mcts/phmap_transposition_table.h
#ifndef ALPHAZERO_PHMAP_TRANSPOSITION_TABLE_H
#define ALPHAZERO_PHMAP_TRANSPOSITION_TABLE_H

#include <memory>
#include <atomic>
#include <vector>
#include <cstdint>
#include <chrono>
#include "core/export_macros.h"
#include "mcts/mcts_node.h"
#include "parallel_hashmap/phmap.h"

namespace alphazero {
namespace mcts {

/**
 * @brief High-performance Transposition Table using parallel-hashmap
 * 
 * This implementation leverages phmap's parallel hash map for:
 * - Lock-free concurrent access with internal sharding
 * - Better cache locality and memory efficiency
 * - Automatic load balancing across threads
 * - Built-in memory management
 */
class ALPHAZERO_API PHMapTranspositionTable {
public:
    /**
     * @brief Configuration for the transposition table
     */
    struct Config {
        size_t size_mb;              // Memory limit in MB
        size_t num_shards;           // Number of shards (0 for auto)
        bool enable_compression;     // Enable node compression
        bool enable_stats;           // Enable detailed statistics
        float replacement_threshold; // Replacement score threshold
        
        Config() : 
            size_mb(1024),
            num_shards(0),
            enable_compression(true),
            enable_stats(true),
            replacement_threshold(0.1f) {}
    };

    /**
     * @brief Compressed entry for memory efficiency
     */
    struct Entry {
        // Core data
        uint64_t hash;               // Full hash for verification
        std::weak_ptr<MCTSNode> node; // Weak reference to avoid cycles
        
        // Compressed statistics
        uint32_t visits;             // Visit count
        float value;                 // Node value
        uint16_t depth;              // Search depth
        uint16_t age;                // Generation counter
        
        // For compression
        uint32_t best_action;        // Best action index
        float prior;                 // Prior probability
        
        // Access tracking
        std::atomic<uint32_t> last_access{0};
        
        // Calculate replacement score
        float getReplacementScore(uint32_t current_time) const {
            float recency = 1.0f / (1.0f + (current_time - last_access.load()));
            float importance = visits * (1.0f + value) * (1.0f + depth);
            return recency * importance;
        }
    };

    explicit PHMapTranspositionTable(const Config& config = Config());
    ~PHMapTranspositionTable();

    /**
     * @brief Store a node in the table
     */
    void store(uint64_t hash, std::shared_ptr<MCTSNode> node, int depth);
    
    /**
     * @brief Lookup a node by hash
     */
    std::shared_ptr<MCTSNode> lookup(uint64_t hash, int min_depth = 0);
    
    /**
     * @brief Prefetch entries for better cache performance
     */
    void prefetch(const std::vector<uint64_t>& hashes);
    
    /**
     * @brief Clear all entries
     */
    void clear();
    
    /**
     * @brief Get current size
     */
    size_t size() const { return entries_.size(); }
    
    /**
     * @brief Get statistics
     */
    struct Stats {
        size_t total_lookups;
        size_t successful_lookups;
        size_t total_stores;
        size_t replacements;
        float hit_rate;
        size_t memory_usage_mb;
    };
    Stats getStats() const;
    /**
     * @brief Force memory limit enforcement
     */
    void enforceMemoryLimit();
    
private:
    // Use parallel_flat_hash_map with custom allocator and sharding
    using EntryPtr = std::shared_ptr<Entry>;
    using TableType = phmap::parallel_flat_hash_map<
        uint64_t,                    // Key type
        EntryPtr,                    // Value type
        phmap::priv::hash_default_hash<uint64_t>, // Hash function
        phmap::priv::hash_default_eq<uint64_t>,   // Equality
        std::allocator<std::pair<const uint64_t, EntryPtr>>, // Allocator
        4,                          // Submap count log2 (16 submaps)
        std::mutex                  // Mutex type
    >;
    
    TableType entries_;
    Config config_;
    
    // Statistics
    std::atomic<size_t> total_lookups_{0};
    std::atomic<size_t> successful_lookups_{0};
    std::atomic<size_t> total_stores_{0};
    std::atomic<size_t> replacements_{0};
    
    // Access counter for LRU
    std::atomic<uint32_t> access_counter_{0};
    
    // Memory management
    size_t max_entries_;
    std::atomic<size_t> current_memory_usage_{0};
    
    // Helper methods
    EntryPtr createEntry(uint64_t hash, std::shared_ptr<MCTSNode> node, int depth);
    size_t estimateEntrySize(const Entry& entry) const;
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_PHMAP_TRANSPOSITION_TABLE_H