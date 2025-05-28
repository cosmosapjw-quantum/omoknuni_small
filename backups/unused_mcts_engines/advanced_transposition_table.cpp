// src/mcts/advanced_transposition_table.cpp
#include "mcts/advanced_transposition_table.h"
#include "utils/logger.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <immintrin.h> // For prefetch intrinsics

namespace alphazero {
namespace mcts {

// Table implementation
AdvancedTranspositionTable::Table::Table(size_t size) {
    // Round up to power of 2 for efficient masking
    size_t n = 1;
    while (n < size) n <<= 1;
    
    num_buckets = n;
    mask = n - 1;
    buckets = std::make_unique<Bucket[]>(num_buckets);
}

// Entry pool implementation
AdvancedTranspositionTable::CompressedEntry* AdvancedTranspositionTable::EntryPool::allocate() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (free_list.empty()) {
        // Allocate new block
        auto block = std::make_unique<CompressedEntry[]>(block_size);
        
        // Add all entries to free list
        for (size_t i = 0; i < block_size; ++i) {
            free_list.push_back(&block[i]);
        }
        
        blocks.push_back(std::move(block));
        total_allocated += block_size;
    }
    
    CompressedEntry* entry = free_list.back();
    free_list.pop_back();
    
    // Initialize entry
    new (entry) CompressedEntry();
    
    return entry;
}

void AdvancedTranspositionTable::EntryPool::deallocate(CompressedEntry* entry) {
    if (!entry) return;
    
    // Reset entry
    entry->~CompressedEntry();
    
    std::lock_guard<std::mutex> lock(mutex);
    free_list.push_back(entry);
}

// Main class implementation
AdvancedTranspositionTable::AdvancedTranspositionTable(const Config& config) 
    : config_(config) {
    
    // Calculate number of buckets based on memory limit
    size_t entry_size = sizeof(CompressedEntry) + sizeof(Table::Bucket);
    size_t total_entries = (config.size_mb * 1024 * 1024) / entry_size;
    size_t entries_per_table = total_entries / config.num_hash_functions;
    
    // Initialize tables
    tables_.reserve(config.num_hash_functions);
    for (int i = 0; i < config.num_hash_functions; ++i) {
        tables_.push_back(std::make_unique<Table>(entries_per_table));
    }
    
    // Initialize hash functions
    initializeHashFunctions();
    
    LOG_MCTS_INFO("Advanced Transposition Table initialized: {} MB, {} tables, {} total buckets",
                  config.size_mb, tables_.size(), total_entries);
}

AdvancedTranspositionTable::~AdvancedTranspositionTable() {
    clear();
}

void AdvancedTranspositionTable::initializeHashFunctions() {
    // Use different mixing constants for each hash function
    const uint64_t mixers[] = {
        0x9e3779b97f4a7c15ULL,  // Golden ratio
        0x517cc1b727220a95ULL,  // Random prime
        0x8764538926473829ULL,  // Another prime
        0x2545f4914f6cdd1dULL,  // And another
        0x6c74e8b8e9a584e1ULL,
        0x829f3c28f92a8b65ULL
    };
    
    hash_functions_.clear();
    for (int i = 0; i < config_.num_hash_functions; ++i) {
        hash_functions_.push_back([i, mixer = mixers[i]](uint64_t hash) -> size_t {
            // MurmurHash3-style mixing
            hash ^= hash >> 33;
            hash *= mixer;
            hash ^= hash >> 33;
            hash *= 0xc4ceb9fe1a85ec53ULL;
            hash ^= hash >> 33;
            
            // Additional mixing based on table index
            hash += i * 0x9ddfea08eb382d69ULL;
            hash ^= (hash << 21) | (hash >> 43);
            
            return static_cast<size_t>(hash);
        });
    }
}

bool AdvancedTranspositionTable::store(
    uint64_t hash, 
    std::shared_ptr<MCTSNode> node, 
    int depth
) {
    if (!node) return false;
    
    // Check if entry already exists
    CompressedEntry* existing = findEntry(hash);
    if (existing) {
        // Try to update existing entry
        if (!tryLockEntry(existing)) {
            return false; // Skip if locked
        }
        
        // Check if we should replace
        auto temp_entry = entry_pool_.allocate();
        compressNode(*node, temp_entry);
        temp_entry->depth = depth;
        temp_entry->age = 0;
        temp_entry->hash_verification = hash;
        
        bool should_replace = shouldReplace(existing, temp_entry);
        
        if (should_replace) {
            // Copy new data to existing entry
            existing->compressed_visits = temp_entry->compressed_visits;
            existing->depth = temp_entry->depth;
            existing->value = temp_entry->value;
            existing->best_action = temp_entry->best_action;
            existing->compressed_q = temp_entry->compressed_q;
            existing->compressed_prior = temp_entry->compressed_prior;
            existing->full_node = node;
            existing->age = 0;
            existing->last_access_time.store(access_counter_.fetch_add(1));
        }
        
        entry_pool_.deallocate(temp_entry);
        unlockEntry(existing);
        
        return should_replace;
    }
    
    // Create new entry
    CompressedEntry* new_entry = entry_pool_.allocate();
    compressNode(*node, new_entry);
    new_entry->depth = depth;
    new_entry->age = 0;
    new_entry->hash_verification = hash;
    new_entry->full_node = node;
    new_entry->last_access_time.store(access_counter_.fetch_add(1));
    
    // Try to insert
    bool inserted = insertEntry(new_entry, hash);
    
    if (!inserted) {
        entry_pool_.deallocate(new_entry);
        return false;
    }
    
    total_entries_.fetch_add(1);
    if (!node) {
        compressed_entries_.fetch_add(1);
    }
    
    return true;
}

std::shared_ptr<MCTSNode> AdvancedTranspositionTable::lookup(
    uint64_t hash, 
    int min_depth
) {
    CompressedEntry* entry = findEntry(hash);
    
    if (!entry) {
        failed_lookups_.fetch_add(1);
        return nullptr;
    }
    
    // Verify hash to prevent collisions
    if (config_.verify_hash && entry->hash_verification != hash) {
        hash_collisions_.fetch_add(1);
        failed_lookups_.fetch_add(1);
        return nullptr;
    }
    
    // Check depth requirement
    if (entry->depth < min_depth) {
        failed_lookups_.fetch_add(1);
        return nullptr;
    }
    
    // Update access time
    entry->last_access_time.store(access_counter_.fetch_add(1));
    
    // Try to get full node first
    auto full_node = entry->full_node.lock();
    if (full_node) {
        successful_lookups_.fetch_add(1);
        return full_node;
    }
    
    // Decompress if needed
    if (config_.enable_compression) {
        auto decompressed = decompressNode(entry);
        if (decompressed) {
            successful_lookups_.fetch_add(1);
            return decompressed;
        }
    }
    
    failed_lookups_.fetch_add(1);
    return nullptr;
}

void AdvancedTranspositionTable::prefetch(const uint64_t* hashes, size_t count) {
    if (!config_.enable_prefetch) return;
    
    for (size_t i = 0; i < count && i < 8; ++i) {  // Limit prefetch
        uint64_t hash = hashes[i];
        
        // Prefetch from each table
        for (size_t t = 0; t < tables_.size(); ++t) {
            size_t index = hash_functions_[t](hash) & tables_[t]->mask;
            void* addr = &tables_[t]->buckets[index];
            
            #ifdef __builtin_prefetch
            __builtin_prefetch(addr, 0, 1); // Read, low temporal locality
            #elif defined(_mm_prefetch)
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
            #endif
        }
    }
}

AdvancedTranspositionTable::CompressedEntry* AdvancedTranspositionTable::findEntry(uint64_t hash) {
    // Check each table
    for (size_t i = 0; i < tables_.size(); ++i) {
        size_t index = hash_functions_[i](hash) & tables_[i]->mask;
        CompressedEntry* entry = tables_[i]->buckets[index].entry.load(std::memory_order_acquire);
        
        if (entry && entry->hash_verification == hash) {
            return entry;
        }
    }
    
    return nullptr;
}

bool AdvancedTranspositionTable::insertEntry(CompressedEntry* entry, uint64_t hash) {
    // Try simple insertion first
    for (size_t i = 0; i < tables_.size(); ++i) {
        size_t index = hash_functions_[i](hash) & tables_[i]->mask;
        CompressedEntry* expected = nullptr;
        
        if (tables_[i]->buckets[index].entry.compare_exchange_strong(
                expected, entry, std::memory_order_release)) {
            return true;
        }
    }
    
    // Use cuckoo hashing if simple insertion fails
    return cuckooInsert(entry, hash);
}

bool AdvancedTranspositionTable::cuckooInsert(
    CompressedEntry* entry, 
    uint64_t hash, 
    int max_kicks
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> table_dist(0, tables_.size() - 1);
    
    CompressedEntry* current = entry;
    uint64_t current_hash = hash;
    
    for (int kick = 0; kick < max_kicks; ++kick) {
        // Choose random table
        int table_idx = table_dist(gen);
        size_t index = hash_functions_[table_idx](current_hash) & tables_[table_idx]->mask;
        
        // Swap with existing entry
        CompressedEntry* displaced = tables_[table_idx]->buckets[index].entry.exchange(
            current, std::memory_order_acq_rel);
        
        if (!displaced) {
            // Successfully inserted
            return true;
        }
        
        // Check if we should evict the displaced entry
        if (shouldReplace(displaced, current)) {
            // Keep current, evict displaced
            evictEntry(displaced);
            return true;
        }
        
        // Continue with displaced entry
        current = displaced;
        current_hash = displaced->hash_verification;
    }
    
    // Failed to insert - evict the entry we're trying to insert
    evictEntry(current);
    return false;
}

void AdvancedTranspositionTable::compressNode(
    const MCTSNode& node, 
    CompressedEntry* entry
) {
    entry->compressed_visits = compressVisits(node.getVisitCount());
    entry->value = node.getValue();
    
    // Get best action (simplified - would need actual implementation)
    entry->best_action = 0; // TODO: Get actual best action from node
    
    // Compress Q-value and prior (simplified)
    entry->compressed_q = compressFloat(node.getValue(), -1.0f, 1.0f);
    entry->compressed_prior = compressFloat(0.5f, 0.0f, 1.0f); // TODO: Get actual prior
}

std::shared_ptr<MCTSNode> AdvancedTranspositionTable::decompressNode(
    const CompressedEntry* entry
) {
    // Note: This is a simplified decompression
    // In practice, you'd need more information to fully reconstruct the node
    LOG_MCTS_WARN("Node decompression not fully implemented");
    return nullptr;
}

float AdvancedTranspositionTable::calculateReplacementScore(
    const CompressedEntry* entry
) const {
    if (!entry) return 0.0f;
    
    // Calculate weighted score
    float score = 0.0f;
    
    // Depth component (higher is better)
    score += config_.depth_weight * (entry->depth / 100.0f);
    
    // Visits component (higher is better, log scale)
    int visits = decompressVisits(entry->compressed_visits);
    score += config_.visits_weight * std::log1p(visits) / 10.0f;
    
    // Age component (lower is better)
    float age_penalty = entry->age / static_cast<float>(config_.max_age);
    score -= config_.age_weight * age_penalty;
    
    return score;
}

bool AdvancedTranspositionTable::shouldReplace(
    const CompressedEntry* existing, 
    const CompressedEntry* candidate
) const {
    float existing_score = calculateReplacementScore(existing);
    float candidate_score = calculateReplacementScore(candidate);
    
    return candidate_score > existing_score;
}

void AdvancedTranspositionTable::evictEntry(CompressedEntry* entry) {
    if (entry) {
        evictions_.fetch_add(1);
        entry_pool_.deallocate(entry);
    }
}

bool AdvancedTranspositionTable::tryLockEntry(CompressedEntry* entry) {
    uint8_t expected = 0;
    return entry->lock.compare_exchange_strong(
        expected, 1, std::memory_order_acquire);
}

void AdvancedTranspositionTable::unlockEntry(CompressedEntry* entry) {
    entry->lock.store(0, std::memory_order_release);
}

uint32_t AdvancedTranspositionTable::compressVisits(int visits) {
    // Log scale compression
    if (visits <= 0) return 0;
    return static_cast<uint32_t>(std::log2(visits + 1) * 1000);
}

int AdvancedTranspositionTable::decompressVisits(uint32_t compressed) {
    if (compressed == 0) return 0;
    return static_cast<int>(std::exp2(compressed / 1000.0) - 1);
}

uint16_t AdvancedTranspositionTable::compressFloat(float value, float min, float max) {
    // Normalize to [0, 1]
    float normalized = (value - min) / (max - min);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    
    // Convert to 16-bit fixed point
    return static_cast<uint16_t>(normalized * 65535);
}

float AdvancedTranspositionTable::decompressFloat(uint16_t compressed, float min, float max) {
    float normalized = compressed / 65535.0f;
    return min + normalized * (max - min);
}

void AdvancedTranspositionTable::clear() {
    // Clear all tables
    for (auto& table : tables_) {
        for (size_t i = 0; i < table->num_buckets; ++i) {
            CompressedEntry* entry = table->buckets[i].entry.exchange(
                nullptr, std::memory_order_acquire);
            if (entry) {
                entry_pool_.deallocate(entry);
            }
        }
    }
    
    // Reset statistics
    total_entries_.store(0);
    compressed_entries_.store(0);
    hash_collisions_.store(0);
    successful_lookups_.store(0);
    failed_lookups_.store(0);
    evictions_.store(0);
}

void AdvancedTranspositionTable::ageEntries() {
    current_generation_.fetch_add(1);
    
    // Age all entries
    for (auto& table : tables_) {
        for (size_t i = 0; i < table->num_buckets; ++i) {
            CompressedEntry* entry = table->buckets[i].entry.load(std::memory_order_acquire);
            if (entry) {
                entry->age++;
                
                // Evict if too old
                if (entry->age > config_.max_age) {
                    if (table->buckets[i].entry.compare_exchange_strong(
                            entry, nullptr, std::memory_order_release)) {
                        evictEntry(entry);
                    }
                }
            }
        }
    }
}

AdvancedTranspositionTable::Stats AdvancedTranspositionTable::getStats() const {
    Stats stats;
    
    stats.total_entries = total_entries_.load();
    stats.compressed_entries = compressed_entries_.load();
    stats.hash_collisions = hash_collisions_.load();
    stats.successful_lookups = successful_lookups_.load();
    stats.failed_lookups = failed_lookups_.load();
    stats.evictions = evictions_.load();
    
    size_t total_lookups = stats.successful_lookups + stats.failed_lookups;
    if (total_lookups > 0) {
        stats.hit_rate = static_cast<float>(stats.successful_lookups) / total_lookups;
    } else {
        stats.hit_rate = 0.0f;
    }
    
    // Calculate average depth and memory usage
    size_t total_depth = 0;
    size_t entry_count = 0;
    
    for (const auto& table : tables_) {
        for (size_t i = 0; i < table->num_buckets; ++i) {
            CompressedEntry* entry = table->buckets[i].entry.load(std::memory_order_relaxed);
            if (entry) {
                total_depth += entry->depth;
                entry_count++;
            }
        }
    }
    
    if (entry_count > 0) {
        stats.avg_depth = static_cast<float>(total_depth) / entry_count;
        stats.load_factor = static_cast<float>(entry_count) / 
                           (tables_.size() * tables_[0]->num_buckets);
    } else {
        stats.avg_depth = 0.0f;
        stats.load_factor = 0.0f;
    }
    
    // Memory usage
    stats.memory_used_mb = (entry_pool_.total_allocated * sizeof(CompressedEntry)) / (1024 * 1024);
    for (const auto& table : tables_) {
        stats.memory_used_mb += (table->num_buckets * sizeof(Table::Bucket)) / (1024 * 1024);
    }
    
    return stats;
}

void AdvancedTranspositionTable::resize(size_t new_size_mb) {
    LOG_MCTS_INFO("Resizing transposition table to {} MB", new_size_mb);
    
    // Save current configuration
    Config new_config = config_;
    new_config.size_mb = new_size_mb;
    
    // Create new instance
    AdvancedTranspositionTable new_table(new_config);
    
    // Copy entries (simplified - in practice would be more sophisticated)
    // ...
    
    // Swap
    std::swap(config_, new_table.config_);
    std::swap(tables_, new_table.tables_);
    std::swap(hash_functions_, new_table.hash_functions_);
}

} // namespace mcts
} // namespace alphazero