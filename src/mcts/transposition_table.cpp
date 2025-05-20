// src/mcts/transposition_table.cpp
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <vector>
#include <limits>
#include <thread>
#include <omp.h>

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
    // Get thread ID for thread-local cache
    int thread_id = omp_get_thread_num() % MAX_THREADS;
    auto& thread_cache = thread_caches_[thread_id];
    
    try {
        // First check the thread-local cache for better performance
        auto cache_it = thread_cache.recent_lookups.find(hash);
        if (cache_it != thread_cache.recent_lookups.end()) {
            // Try to lock the weak_ptr to get a shared_ptr
            auto node_ptr = cache_it->second.lock();
            if (node_ptr) {
                // Cache hit - found in thread-local cache
                thread_cache.hits++;
                hits_.fetch_add(1, std::memory_order_relaxed);
                return node_ptr;
            } else {
                // Weak pointer expired, remove from cache
                thread_cache.recent_lookups.erase(cache_it);
            }
        }
        
        // Cache miss, check the main transposition table
        auto it = entries_.find(hash);
        
        if (it != entries_.end() && it->second) {
            // Access the EntryPtr directly - it's thread-safe
            auto entry = it->second;
            
            if (entry && entry->isValid()) {
                // Try to lock the weak_ptr to get a shared_ptr
                auto node_ptr = entry->node.lock();
                if (node_ptr) {
                    // Found a valid entry in main table
                    
                    // Add to thread-local cache for future lookups
                    // Limit cache size to prevent excessive memory usage
                    const size_t MAX_CACHE_SIZE = 128;
                    if (thread_cache.recent_lookups.size() >= MAX_CACHE_SIZE) {
                        // Simple cache management: clear when full
                        thread_cache.recent_lookups.clear();
                    }
                    thread_cache.recent_lookups[hash] = node_ptr;
                    
                    hits_.fetch_add(1, std::memory_order_relaxed);
                    return node_ptr;
                }
            }
        }
    } catch (...) {
        // Any exception during lookup means we failed to find a valid entry
    }
    
    // Entry not found or invalid in both cache and main table
    thread_cache.misses++;
    misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void TranspositionTable::store(uint64_t hash, std::weak_ptr<MCTSNode> node, int depth) {
    if (node.expired()) {
        return;  // Don't store expired nodes
    }
    
    // Get thread ID for thread-local cache
    int thread_id = omp_get_thread_num() % MAX_THREADS;
    auto& thread_cache = thread_caches_[thread_id];
    
    // Get a shared_ptr to the node
    std::shared_ptr<MCTSNode> locked_node;
    int visit_count = 0;
    
    try {
        locked_node = node.lock();
        if (!locked_node) {
            return; // Node expired before we could read it
        }
        
        visit_count = locked_node->getVisitCount();
        
        // FIX: Additional validation to prevent storing nodes that are being cleaned up
        try {
            if (!locked_node->getState().validate()) {
                return; // Don't store nodes with invalid state
            }
        } catch (...) {
            // If validation throws an exception, don't store the node
            return;
        }
    } catch (...) {
        // If we can't even get the visit count of the new node, don't store it
        return;
    }
    
    // Update thread-local cache with this node
    const size_t MAX_CACHE_SIZE = 128;
    if (thread_cache.recent_lookups.size() >= MAX_CACHE_SIZE) {
        // Simple cache management: clear when full
        thread_cache.recent_lookups.clear();
    }
    thread_cache.recent_lookups[hash] = locked_node;
    
    // Create a new entry for the main table
    auto new_entry = std::make_shared<TranspositionEntry>(node, hash, depth, visit_count);
    
    // Try to insert or update the entry in the main table
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
    
    // Clear the thread-local caches
    for (auto& cache : thread_caches_) {
        cache.recent_lookups.clear();
        cache.hits = 0;
        cache.misses = 0;
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
    // Combine global and thread-local statistics
    size_t global_hits = hits_.load(std::memory_order_relaxed);
    size_t global_misses = misses_.load(std::memory_order_relaxed);
    
    // Add thread-local cache statistics
    size_t total_hits = global_hits;
    size_t total_misses = global_misses;
    
    for (const auto& cache : thread_caches_) {
        total_hits += cache.hits;
        total_misses += cache.misses;
    }
    
    size_t total = total_hits + total_misses;
    
    if (total == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(total_hits) / static_cast<float>(total);
}

void TranspositionTable::resetStats() {
    // Reset global statistics
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
    
    // Reset thread-local cache statistics
    for (auto& cache : thread_caches_) {
        cache.hits = 0;
        cache.misses = 0;
    }
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