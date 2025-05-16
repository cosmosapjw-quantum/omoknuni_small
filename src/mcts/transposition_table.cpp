// src/mcts/transposition_table.cpp
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <vector>
#include <limits>
#include <thread>

namespace alphazero {
namespace mcts {

TranspositionEntry::TranspositionEntry(std::weak_ptr<MCTSNode> n, uint64_t h, int d, int v)
    : node(n), hash(h), depth(d), visits(v) {
}

TranspositionTable::TranspositionTable(size_t size_mb, size_t /*num_shards_param*/)
    : hits_(0), misses_(0) {
    // Calculate approximate number of entries based on memory size
    // Assuming approximately 80 bytes per entry including overhead
    capacity_ = (size_mb * 1024 * 1024) / 80;
    
    // Use aggressive number of shards for maximum parallelism
    size_t num_shards = std::max(size_t(16), size_t(std::thread::hardware_concurrency() * 2));
    
    // parallel_flat_hash_map automatically handles sharding internally
    // Just reserve the capacity
    entries_.reserve(capacity_);
}

std::shared_ptr<MCTSNode> TranspositionTable::get(uint64_t hash) {
    try {
        // Use the parallel_hashmap's find which is already thread-safe
        auto it = entries_.find(hash);
        
        if (it != entries_.end() && it->second) {
            // Access the EntryPtr directly - it's thread-safe
            auto entry = it->second;
            
            if (entry && entry->isValid()) {
                // Try to lock the weak_ptr to get a shared_ptr
                auto node_ptr = entry->node.lock();
                if (node_ptr) {
                    // Found a valid entry
                    hits_.fetch_add(1, std::memory_order_relaxed);
                    return node_ptr;
                }
            }
        }
    } catch (...) {
        // Any exception during lookup means we failed to find a valid entry
    }
    
    // Entry not found or invalid
    misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void TranspositionTable::store(uint64_t hash, std::weak_ptr<MCTSNode> node, int depth) {
    if (node.expired()) {
        return;  // Don't store expired nodes
    }
    
    // Safely get the visit count of the new node once to avoid repeated access
    int visit_count = 0;
    try {
        if (auto locked_node = node.lock()) {
            visit_count = locked_node->getVisitCount();
            
            // FIX: Additional validation to prevent storing nodes that are being cleaned up
            if (!locked_node->getState().validate()) {
                return; // Don't store nodes with invalid state
            }
        } else {
            return; // Node expired before we could read it
        }
    } catch (...) {
        // If we can't even get the visit count of the new node, don't store it
        return;
    }
    
    // Create a new entry
    auto new_entry = std::make_shared<TranspositionEntry>(node, hash, depth, visit_count);
    
    // Try to insert or update the entry
    entries_.insert_or_assign(hash, new_entry);
    
    // Periodically check and enforce capacity limits
    // We don't do this on every insertion for performance reasons
    static thread_local std::mt19937 rng(std::random_device{}());
    if (std::uniform_int_distribution<>(0, 999)(rng) < 10) {  // ~1% probability
        try {
            enforceCapacityLimit();
        } catch (...) {
            // Silently ignore errors in capacity enforcement
            // This is a non-critical operation
        }
    }
}

void TranspositionTable::clear() {
    // Lock to prevent concurrent access during clear
    std::lock_guard<std::mutex> lock(clear_mutex_);
    
    // First, expire all weak_ptr references to prevent dangling pointers
    try {
        // Iterate through all entries and reset weak_ptr references  
        for (auto& [hash, entry] : entries_) {
            if (entry) {
                entry->node.reset();  // Explicitly reset weak_ptr
            }
        }
    } catch (...) {
        // Continue with cleanup even if some entries can't be accessed
    }
    
    // Clear the hash map and reset statistics
    entries_.clear();
    resetStats();
}

size_t TranspositionTable::size() const {
    return entries_.size();
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

void TranspositionTable::enforceCapacityLimit() {
    // Take a lock to prevent concurrent capacity enforcement
    std::lock_guard<std::mutex> guard(clear_mutex_);
    
    // Early exit if we're under capacity
    if (entries_.size() <= capacity_) {
        return;
    }
    
    // Determine number of entries to remove
    size_t to_remove = entries_.size() - capacity_;
    
    // Use a probabilistic approach for efficiency - only remove some entries
    // but with a higher probability when we're far over capacity
    size_t actual_to_remove = std::min(to_remove, entries_.size() / 10);  // Max 10% at once
    
    if (actual_to_remove < 10) {
        // Too few to remove, not worth the effort now
        return;
    }
    
    // We'll sample entries and remove those with the lowest visit counts
    static thread_local std::mt19937 rng(std::random_device{}());
    
    // Collect candidates for removal via sampling
    std::vector<std::pair<uint64_t, int>> candidates;
    const size_t sample_size = std::min(actual_to_remove * 3, entries_.size() / 5);
    candidates.reserve(sample_size);
    
    // Thread-safe sampling approach - only look at the visit counts, which are atomic
    size_t sample_count = 0;
    for (auto it = entries_.begin(); it != entries_.end() && sample_count < sample_size; ++it, ++sample_count) {
        if (it->second) {
            try {
                // Try to read the visit count atomically without taking a lock
                int visit_count = it->second->visits;
                candidates.emplace_back(it->first, visit_count);
            } catch (...) {
                // If the entry is deleted while we're looking at it, just skip it
                continue;
            }
        }
    }
    
    // Sort by visit count (ascending)
    std::sort(candidates.begin(), candidates.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Only keep the candidates we need to remove
    if (candidates.size() > actual_to_remove) {
        candidates.resize(actual_to_remove);
    }
    
    // Remove selected entries - we already have the lock from the lock_guard at the top
    for (const auto& [hash, _] : candidates) {
        // Safely remove the entry
        entries_.erase(hash);
    }
    
    // Log capacity enforcement if we're removing a significant number
    if (actual_to_remove > 1000) {
        // Commented out: TranspositionTable removed entries statistics
    }
}

} // namespace mcts
} // namespace alphazero