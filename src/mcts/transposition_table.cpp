// src/mcts/transposition_table.cpp
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include "utils/memory_debug.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <sstream>

// Namespace alias for memory debug functions
namespace ad = alphazero::debug;

namespace alphazero {
namespace mcts {

// Debug helper for transposition table
void debugTranspositionTableStats(const TranspositionTable* table, const std::string& operation) {
    static auto last_print_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();

    // Only print every second to avoid spamming logs
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count() > 1000) {
        float hit_rate = table->hitRate();
        size_t size = table->size();
        size_t capacity = table->capacity();
        float fill_ratio = static_cast<float>(size) / static_cast<float>(capacity);

        std::stringstream ss;
        ss << "[TT] " << operation
           << " - Size: " << size << "/" << capacity
           << " (" << (fill_ratio * 100.0f) << "%) Hit rate: "
           << (hit_rate * 100.0f) << "%";

        std::cout << ss.str() << std::endl;
        last_print_time = now;
    }
}

TranspositionEntry::TranspositionEntry(MCTSNode* n, uint64_t h, int d, int v)
    : node(n), hash(h), depth(d), visits(v) {
}

TranspositionTable::TranspositionTable(size_t size_mb)
    : hits_(0), misses_(0) {
    // Calculate number of entries based on memory size
    // Assuming approximately 80 bytes per entry (including overhead)
    capacity_ = (size_mb * 1024 * 1024) / 80;
    
    // Initialize locks
    table_locks_.reserve(NUM_LOCKS);
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        table_locks_.emplace_back(std::make_unique<std::mutex>());
    }
    
    // Reserve space for entries
    entries_.reserve(capacity_);
    index_map_.reserve(capacity_);
    
    misses_.store(0, std::memory_order_relaxed);
}

MCTSNode* TranspositionTable::get(uint64_t hash) {
    // Take memory snapshot at start of lookup
    ad::takeMemorySnapshot("TT_Get_Start_" + std::to_string(hash % 1000));

    // Get lock for this hash
    auto& lock = getLock(hash);
    std::lock_guard<std::mutex> guard(lock);

    // Look up hash in index map
    auto it = index_map_.find(hash);
    if (it != index_map_.end()) {
        size_t index = it->second;
        if (index < entries_.size()) {
            auto& entry = entries_[index];

            // Check if hash matches and entry is valid
            if (entry && entry->hash == hash && entry->node) {
                hits_.fetch_add(1, std::memory_order_relaxed);

                // Debug output
                debugTranspositionTableStats(this, "HIT");

                // Take memory snapshot after successful lookup
                ad::takeMemorySnapshot("TT_Get_Hit_" + std::to_string(hash % 1000));

                return entry->node;
            }
        }
    }

    misses_.fetch_add(1, std::memory_order_relaxed);

    // Debug output
    debugTranspositionTableStats(this, "MISS");

    // Take memory snapshot after missed lookup
    ad::takeMemorySnapshot("TT_Get_Miss_" + std::to_string(hash % 1000));

    return nullptr;
}

void TranspositionTable::store(uint64_t hash, MCTSNode* node, int depth) {
    if (!node) {
        return;
    }

    // Take memory snapshot at start of store operation
    ad::takeMemorySnapshot("TT_Store_Start_" + std::to_string(hash % 1000));

    // Get lock for this hash
    auto& lock = getLock(hash);
    std::lock_guard<std::mutex> guard(lock);

    // Check if hash already exists
    auto it = index_map_.find(hash);
    if (it != index_map_.end()) {
        size_t index = it->second;
        if (index < entries_.size()) {
            auto& entry = entries_[index];

            // Update existing entry if it's for the same hash
            if (entry && entry->hash == hash) {
                // Only replace if new node has more visits or higher depth
                if (node->getVisitCount() > entry->visits || depth > entry->depth) {
                    std::lock_guard<std::mutex> entry_guard(entry->lock);

                    // Debug output
                    std::cout << "[TT] Updating existing entry: Hash=" << hash
                              << ", Old visits=" << entry->visits
                              << ", New visits=" << node->getVisitCount()
                              << ", Old depth=" << entry->depth
                              << ", New depth=" << depth << std::endl;

                    entry->node = node;
                    entry->depth = depth;
                    entry->visits = node->getVisitCount();
                }

                // Debug stats
                debugTranspositionTableStats(this, "UPDATE");

                // Take memory snapshot after updating
                ad::takeMemorySnapshot("TT_Store_Update_" + std::to_string(hash % 1000));

                return;
            }
        }
    }

    // Add new entry
    if (entries_.size() < capacity_) {
        // Still have room, just add
        std::cout << "[TT] Adding new entry: Hash=" << hash
                  << ", Visits=" << node->getVisitCount()
                  << ", Depth=" << depth
                  << ", Table size=" << entries_.size() << "/" << capacity_ << std::endl;

        auto entry = std::make_unique<TranspositionEntry>(node, hash, depth, node->getVisitCount());

        // Store index
        size_t index = entries_.size();
        index_map_[hash] = index;

        // Add entry
        entries_.push_back(std::move(entry));

        // Debug stats
        debugTranspositionTableStats(this, "ADD");

        // Take memory snapshot after adding
        ad::takeMemorySnapshot("TT_Store_Add_" + std::to_string(hash % 1000));
    } else {
        // Table is full, replace an entry
        // Find the entry with the least visits
        size_t min_visits_index = 0;
        int min_visits = std::numeric_limits<int>::max();

        // Sample a small set of entries to find a replacement candidate
        // This is more efficient than scanning the entire table
        const size_t SAMPLE_SIZE = 10;

        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            size_t index = (hash + i) % entries_.size();
            auto& entry = entries_[index];

            if (entry) {
                std::lock_guard<std::mutex> entry_guard(entry->lock);
                if (entry->visits < min_visits) {
                    min_visits = entry->visits;
                    min_visits_index = index;
                }
            } else {
                // Found an empty slot, use it
                min_visits_index = index;
                break;
            }
        }

        // Debug output for replacement
        std::cout << "[TT] Replacing entry: Hash=" << hash
                  << ", Visits=" << node->getVisitCount()
                  << ", Depth=" << depth
                  << ", Replaced visits=" << min_visits << std::endl;

        // Replace the entry
        auto& entry = entries_[min_visits_index];

        // Remove old hash from index map if it exists
        if (entry) {
            uint64_t old_hash = entry->hash;
            auto old_it = index_map_.find(old_hash);
            if (old_it != index_map_.end() && old_it->second == min_visits_index) {
                index_map_.erase(old_it);
            }
        }

        // Create new entry
        entry = std::make_unique<TranspositionEntry>(node, hash, depth, node->getVisitCount());

        // Update index map
        index_map_[hash] = min_visits_index;

        // Debug stats
        debugTranspositionTableStats(this, "REPLACE");

        // Take memory snapshot after replacement
        ad::takeMemorySnapshot("TT_Store_Replace_" + std::to_string(hash % 1000));
    }
}

void TranspositionTable::clear() {
    std::cout << "[TT] Clearing transposition table with " << entries_.size()
              << " entries (" << index_map_.size() << " indexed positions)" << std::endl;

    // Take memory snapshot before clearing
    ad::takeMemorySnapshot("TT_Clear_Before");

    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        std::lock_guard<std::mutex> guard(*table_locks_[i]);

        // Clear only the entries in this shard
        size_t cleared = 0;
        for (size_t j = i; j < entries_.size(); j += NUM_LOCKS) {
            entries_[j].reset();
            cleared++;
        }

        std::cout << "[TT] Cleared shard " << i << ": " << cleared << " entries" << std::endl;
    }

    // Clear index map
    size_t index_map_size = index_map_.size();
    index_map_.clear();

    // Reset statistics
    resetStats();

    std::cout << "[TT] Table cleared: removed " << index_map_size << " index entries" << std::endl;

    // Take memory snapshot after clearing
    ad::takeMemorySnapshot("TT_Clear_After");
}

size_t TranspositionTable::size() const {
    return index_map_.size();
}

size_t TranspositionTable::capacity() const {
    return capacity_;
}

float TranspositionTable::hitRate() const {
    size_t hits = hits_.load(std::memory_order_relaxed);
    size_t misses = misses_.load(std::memory_order_relaxed);
    size_t total = hits + misses;
    
    if (total == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(hits) / static_cast<float>(total);
}

void TranspositionTable::resetStats() {
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
}

} // namespace mcts
} // namespace alphazero