// src/mcts/transposition_table.cpp
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>

namespace alphazero {
namespace mcts {

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
                return entry->node;
            }
        }
    }
    
    misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void TranspositionTable::store(uint64_t hash, MCTSNode* node, int depth) {
    if (!node) {
        return;
    }
    
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
                    entry->node = node;
                    entry->depth = depth;
                    entry->visits = node->getVisitCount();
                }
                return;
            }
        }
    }
    
    // Add new entry
    if (entries_.size() < capacity_) {
        // Still have room, just add
        auto entry = std::make_unique<TranspositionEntry>(node, hash, depth, node->getVisitCount());
        
        // Store index
        size_t index = entries_.size();
        index_map_[hash] = index;
        
        // Add entry
        entries_.push_back(std::move(entry));
    } else {
        // Table is full, replace an entry
        // Find the entry with the least visits
        size_t min_visits_index = 0;
        int min_visits = std::numeric_limits<int>::max();
        
        // Sample a small set of entries to find a replacement candidate
        // This is more efficient than scanning the entire table
        const size_t SAMPLE_SIZE = 10;
        std::vector<size_t> samples;
        
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
    }
}

void TranspositionTable::clear() {
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        std::lock_guard<std::mutex> guard(*table_locks_[i]);
        
        // Clear only the entries in this shard
        for (size_t j = i; j < entries_.size(); j += NUM_LOCKS) {
            entries_[j].reset();
        }
    }
    
    // Clear index map
    index_map_.clear();
    
    // Reset statistics
    resetStats();
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