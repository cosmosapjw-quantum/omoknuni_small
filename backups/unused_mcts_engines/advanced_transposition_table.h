// include/mcts/advanced_transposition_table.h
#ifndef ALPHAZERO_ADVANCED_TRANSPOSITION_TABLE_H
#define ALPHAZERO_ADVANCED_TRANSPOSITION_TABLE_H

#include <memory>
#include <atomic>
#include <vector>
#include <cstdint>
#include <chrono>
#include <functional>
#include "core/export_macros.h"
#include "mcts/mcts_node.h"
#include "parallel_hashmap/phmap.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Advanced Transposition Table with production-ready features
 * 
 * This implementation provides:
 * - Lock-free operations for read-heavy workloads
 * - Cuckoo hashing with multiple hash functions
 * - Zobrist key verification to prevent collisions
 * - Age-based replacement policy
 * - Depth-preferred replacement
 * - Memory-bounded operation with LRU eviction
 * - Compression for stored nodes
 * - Prefetching for better cache performance
 */
class ALPHAZERO_API AdvancedTranspositionTable {
public:
    /**
     * @brief Configuration for the transposition table
     */
    struct Config {
        size_t size_mb;
        size_t num_buckets;
        int num_hash_functions;
        bool enable_compression;
        bool enable_prefetch;
        bool verify_hash;
        float load_factor;
        int max_age;
        
        // Replacement policy weights
        float depth_weight;
        float visits_weight;
        float age_weight;
        
        // Constructor with defaults
        Config() : 
            size_mb(1024),
            num_buckets(0),
            num_hash_functions(4),
            enable_compression(true),
            enable_prefetch(true),
            verify_hash(true),
            load_factor(0.75f),
            max_age(10),
            depth_weight(0.3f),
            visits_weight(0.4f),
            age_weight(0.3f) {}
    };

    /**
     * @brief Compressed node entry for storage efficiency
     */
    struct CompressedEntry {
        uint64_t hash_verification;  // Full hash for collision detection
        uint32_t compressed_visits;  // Log-scale compressed visits
        uint16_t depth;             // Search depth
        uint16_t age;               // Generation counter
        float value;                // Node value
        uint32_t best_action;       // Best action index
        
        // Compressed statistics
        uint16_t compressed_q;      // Compressed Q-value (fixed point)
        uint16_t compressed_prior;  // Compressed prior (fixed point)
        
        // Weak reference to full node (may be null)
        std::weak_ptr<MCTSNode> full_node;
        
        // Timestamp for LRU
        std::atomic<uint32_t> last_access_time{0};
        
        // Lock bit for atomic updates
        std::atomic<uint8_t> lock{0};
    };

    explicit AdvancedTranspositionTable(const Config& config = Config());
    ~AdvancedTranspositionTable();

    /**
     * @brief Store or update an entry
     * 
     * @param hash Primary hash key
     * @param node Node to store
     * @param depth Current search depth
     * @return true if stored, false if rejected
     */
    bool store(uint64_t hash, std::shared_ptr<MCTSNode> node, int depth);

    /**
     * @brief Lookup an entry
     * 
     * @param hash Primary hash key
     * @param min_depth Minimum required depth
     * @return Node if found and meets criteria, nullptr otherwise
     */
    std::shared_ptr<MCTSNode> lookup(uint64_t hash, int min_depth = 0);

    /**
     * @brief Prefetch entries for better cache performance
     * 
     * @param hashes Array of hashes to prefetch
     * @param count Number of hashes
     */
    void prefetch(const uint64_t* hashes, size_t count);

    /**
     * @brief Clear all entries
     */
    void clear();

    /**
     * @brief Age all entries (call between games)
     */
    void ageEntries();

    /**
     * @brief Get table statistics
     */
    struct Stats {
        size_t total_entries;
        size_t compressed_entries;
        size_t hash_collisions;
        size_t successful_lookups;
        size_t failed_lookups;
        size_t evictions;
        float hit_rate;
        float avg_depth;
        size_t memory_used_mb;
        float load_factor;
    };
    Stats getStats() const;

    /**
     * @brief Resize table if needed
     */
    void resize(size_t new_size_mb);

private:
    // Multi-table structure for cuckoo hashing
    struct Table {
        struct Bucket {
            std::atomic<CompressedEntry*> entry{nullptr};
            alignas(64) char padding[64 - sizeof(std::atomic<CompressedEntry*>)];
        };
        
        std::unique_ptr<Bucket[]> buckets;
        size_t num_buckets;
        size_t mask;
        
        explicit Table(size_t size);
    };

    Config config_;
    std::vector<std::unique_ptr<Table>> tables_;
    
    // Memory pool for entries
    struct EntryPool {
        std::vector<std::unique_ptr<CompressedEntry[]>> blocks;
        std::vector<CompressedEntry*> free_list;
        std::mutex mutex;
        size_t block_size = 1024;
        size_t total_allocated = 0;
        
        CompressedEntry* allocate();
        void deallocate(CompressedEntry* entry);
    };
    EntryPool entry_pool_;

    // Hash functions for cuckoo hashing
    std::vector<std::function<size_t(uint64_t)>> hash_functions_;
    
    // Statistics
    mutable std::atomic<size_t> total_entries_{0};
    mutable std::atomic<size_t> compressed_entries_{0};
    mutable std::atomic<size_t> hash_collisions_{0};
    mutable std::atomic<size_t> successful_lookups_{0};
    mutable std::atomic<size_t> failed_lookups_{0};
    mutable std::atomic<size_t> evictions_{0};
    
    // Generation counter for aging
    std::atomic<uint16_t> current_generation_{0};
    
    // Timing for LRU
    std::atomic<uint32_t> access_counter_{0};

    // Helper methods
    void initializeHashFunctions();
    CompressedEntry* findEntry(uint64_t hash);
    bool insertEntry(CompressedEntry* entry, uint64_t hash);
    bool cuckooInsert(CompressedEntry* entry, uint64_t hash, int max_kicks = 500);
    void compressNode(const MCTSNode& node, CompressedEntry* entry);
    std::shared_ptr<MCTSNode> decompressNode(const CompressedEntry* entry);
    float calculateReplacementScore(const CompressedEntry* entry) const;
    bool shouldReplace(const CompressedEntry* existing, const CompressedEntry* candidate) const;
    void evictEntry(CompressedEntry* entry);
    
    // Lock-free helpers
    bool tryLockEntry(CompressedEntry* entry);
    void unlockEntry(CompressedEntry* entry);
    
    // Compression helpers
    static uint32_t compressVisits(int visits);
    static int decompressVisits(uint32_t compressed);
    static uint16_t compressFloat(float value, float min, float max);
    static float decompressFloat(uint16_t compressed, float min, float max);
};

/**
 * @brief Global transposition table instance
 */
class ALPHAZERO_API TranspositionTableManager {
public:
    static AdvancedTranspositionTable& getInstance() {
        static AdvancedTranspositionTable instance;
        return instance;
    }
    
    static void initialize(const AdvancedTranspositionTable::Config& config) {
        static std::once_flag init_flag;
        std::call_once(init_flag, [&config]() {
            getInstance().~AdvancedTranspositionTable();
            new (&getInstance()) AdvancedTranspositionTable(config);
        });
    }
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_ADVANCED_TRANSPOSITION_TABLE_H