// src/mcts/transposition_table.cpp
#include "mcts/transposition_table.h"
#include "mcts/mcts_node.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <limits>

namespace alphazero {
namespace mcts {

TranspositionEntry::TranspositionEntry(MCTSNode* n, uint64_t h, int d, int v)
    : hash(h), depth(d), visits(v) {
    node.store(n, std::memory_order_release);
}

TranspositionTable::TranspositionTable(size_t size_mb)
    : hits_(0), misses_(0), is_shutdown_(false) {
    try {
        // Calculate number of entries based on memory size
        // Assuming approximately 80 bytes per entry (including overhead)
        capacity_ = (size_mb * 1024 * 1024) / 80;
        
        // Make sure capacity is reasonable to prevent allocation issues
        if (capacity_ < 1000) {
            capacity_ = 1000; // Minimum size for any useful transposition table
        }
        
        // Initialize locks with proper exception handling
        table_locks_.reserve(NUM_LOCKS);
        for (size_t i = 0; i < NUM_LOCKS; ++i) {
            auto lock = std::make_unique<std::timed_mutex>();
            if (!lock) {
                throw std::runtime_error("Failed to allocate mutex for transposition table");
            }
            table_locks_.emplace_back(std::move(lock));
        }
        
        // Reserve space for entries
        try {
            entries_.reserve(capacity_);
            index_map_.reserve(capacity_);
        } catch (const std::exception& e) {
            // If we can't allocate the full capacity, try a smaller size
            std::cerr << "Warning: Could not allocate full transposition table: " << e.what() << std::endl;
            std::cerr << "Trying with reduced capacity" << std::endl;
            
            capacity_ = capacity_ / 2;
            entries_.reserve(capacity_);
            index_map_.reserve(capacity_);
        }
        
        // Initialize atomic counters with proper memory ordering
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        is_shutdown_.store(false, std::memory_order_seq_cst);
        
        // Memory barrier to ensure everything is properly initialized
        std::atomic_thread_fence(std::memory_order_seq_cst);
    } catch (const std::exception& e) {
        // Log any initialization errors
        std::cerr << "Error during transposition table initialization: " << e.what() << std::endl;
        throw; // Re-throw to notify caller
    }
}

TranspositionTable::~TranspositionTable() {
    // First mark table as shutting down to prevent new accesses
    // Use memory_order_seq_cst to ensure this flag is immediately visible to all threads
    is_shutdown_.store(true, std::memory_order_seq_cst);
    
    // Give threads a chance to see the shutdown flag and exit any operations in progress
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    try {
        // Process entries in small batches to avoid long-running operations
        const size_t BATCH_SIZE = 64;
        
        // First pass: null out all node pointers in a thread-safe way
        for (size_t start = 0; start < entries_.size(); start += BATCH_SIZE) {
            size_t end = std::min(start + BATCH_SIZE, entries_.size());
            
            for (size_t i = start; i < end; ++i) {
                if (i < entries_.size() && entries_[i]) {  // Bounds and null check
                    try {
                        // Try to lock without blocking
                        if (entries_[i]->lock.try_lock()) {
                            entries_[i]->node.store(nullptr, std::memory_order_release);  // Clear the node reference
                            entries_[i]->lock.unlock();
                        }
                    } catch (...) {
                        // Ignore errors during shutdown
                    }
                }
            }
            
            // Small yield to avoid blocking other threads
            std::this_thread::yield();
        }
        
        // Memory barrier to ensure node pointer nullification is visible
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        // Second pass: try to acquire all locks before clearing the data structures
        {
            // Try to acquire a set of locks, but don't wait too long for any single lock
            std::vector<std::unique_lock<std::timed_mutex>> locks;
            
            for (size_t i = 0; i < NUM_LOCKS; ++i) {
                if (i % 32 == 0) {
                    // Periodically yield to avoid excessive CPU usage
                    std::this_thread::yield();
                }
                
                try {
                    // Try to lock without blocking
                    if (table_locks_[i]->try_lock()) {
                        locks.emplace_back(*table_locks_[i], std::adopt_lock);
                    }
                } catch (...) {
                    // Ignore lock acquisition failures during shutdown
                }
            }
            
            // Clear the index map under whatever locks we managed to acquire
            index_map_.clear();
            
            // Make a copy of entries_ that we'll clear separately
            auto entries_copy = std::move(entries_);
            
            // Clear the entries vector while holding whatever locks we acquired
            entries_.clear();
            
            // Now entries_copy will be destroyed after locks are released
        }
        
        // Final memory barrier to ensure all cleanup is visible
        std::atomic_thread_fence(std::memory_order_seq_cst);
    } catch (const std::exception& e) {
        // Log but continue with destruction
        std::cerr << "Exception during TranspositionTable destruction: " << e.what() << std::endl;
    } catch (...) {
        // Log unknown exceptions
        std::cerr << "Unknown exception during TranspositionTable destruction" << std::endl;
    }
    
    // Ensure all internal data is cleared to prevent any use-after-free
    table_locks_.clear();
}

MCTSNode* TranspositionTable::get(uint64_t hash) {
    // Early check if the table is being shut down
    if (is_shutdown_.load(std::memory_order_acquire)) {
        return nullptr;
    }
    
    // Get lock for this hash
    auto& lock = getLock(hash);
    
    // Use try_lock to avoid deadlocks in high-contention scenarios
    std::unique_lock<std::timed_mutex> guard(lock, std::try_to_lock);
    if (!guard.owns_lock()) {
        // Couldn't get the lock without blocking - rather than wait,
        // treat this as a miss and return quickly
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // Double-check shutdown after acquiring lock
    if (is_shutdown_.load(std::memory_order_acquire)) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // Look up hash in index map with bounds checking
    auto it = index_map_.find(hash);
    if (it == index_map_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    size_t index = it->second;
    if (index >= entries_.size()) {
        // Index out of bounds - invalid mapping
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    auto& entry = entries_[index];
    // Check if entry is valid
    if (!entry) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    // Check if hash matches and entry has a node
    if (entry->hash != hash || !entry->node.load(std::memory_order_acquire)) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    // Now we need to check if the node itself is still valid
    // Use a regular lock with timeout instead of try_lock to reduce race conditions
    std::unique_lock<std::timed_mutex> entry_guard(entry->lock, std::defer_lock);
    if (!entry_guard.try_lock_for(std::chrono::milliseconds(5))) {
        // Entry is locked by another thread - count as a miss and return quickly
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    // Recheck node pointer after acquiring lock (it might have been set to nullptr)
    if (!entry->node.load(std::memory_order_acquire)) {
        // Node has been removed, count as a miss
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    // Store the node pointer locally before validation
    MCTSNode* result_node = entry->node.load(std::memory_order_acquire);
    
    // Additional safety check - verify we can access node state
    try {
        // Use a memory barrier to ensure we have a consistent view of the node data
        std::atomic_thread_fence(std::memory_order_acquire);
        
        // Get node state and verify hash matches
        const auto& state = result_node->getState();
        if (state.getHash() != hash) {
            // Hash mismatch - something is wrong with this node
            misses_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        
        // Node is valid, count as a hit and return it
        hits_.fetch_add(1, std::memory_order_relaxed);
        
        // Use a memory barrier to ensure the node data is properly synchronized
        std::atomic_thread_fence(std::memory_order_release);
        
        return result_node;
    } catch (const std::exception& e) {
        // Log detailed exception information
        std::cerr << "Exception validating node in get(): " << e.what() << std::endl;
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    } catch (...) {
        // Exception during node validation, count as a miss
        std::cerr << "Unknown exception validating node in get()" << std::endl;
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
}

void TranspositionTable::store(uint64_t hash, MCTSNode* node, int depth) {
    // Validate arguments
    if (!node || is_shutdown_.load(std::memory_order_acquire)) {
        return;
    }
    
    // Additional validation for node - make sure it's still valid
    try {
        const auto& state = node->getState();
        // Double-check that the hash matches what we expect
        if (state.getHash() != hash) {
            std::cerr << "Warning: Hash mismatch in store operation" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        // If we can't access the state, the node is invalid
        std::cerr << "Error validating node: " << e.what() << std::endl;
        return;
    } catch (...) {
        // If we can't access the state, the node is invalid
        std::cerr << "Unknown error validating node" << std::endl;
        return;
    }

    // Get the visit count safely before entering critical section
    int node_visits = 0;
    try {
        node_visits = node->getVisitCount();
    } catch (const std::exception& e) {
        std::cerr << "Error getting visit count: " << e.what() << std::endl;
        node_visits = 0; // Default to 0 if we can't access visit count
    } catch (...) {
        node_visits = 0; // Default to 0 for any other error
    }

    // Memory barrier to ensure state visibility before critical section
    std::atomic_thread_fence(std::memory_order_acq_rel);

    // Get lock for this hash
    auto& lock = getLock(hash);
    std::unique_lock<std::timed_mutex> guard(lock);

    // Double-check that table is not being shut down after acquiring lock
    if (is_shutdown_.load(std::memory_order_acquire)) {
        return;
    }

    // Check if hash already exists
    auto it = index_map_.find(hash);
    if (it != index_map_.end()) {
        size_t index = it->second;
        if (index < entries_.size()) {
            auto& entry = entries_[index];

            // Update existing entry if it's for the same hash
            if (entry && entry->hash == hash) {
                // Entry is valid since we checked above, proceed with update
                
                // Acquire entry lock for thread-safety
                std::unique_lock<std::timed_mutex> entry_guard(entry->lock, std::try_to_lock);
                if (!entry_guard.owns_lock()) {
                    // Could not acquire lock - might be in use by another thread
                    // Rather than blocking, we'll just skip this update
                    return;
                }
                
                // We need to update the node pointer to ensure the transposition table
                // has references to the most up-to-date nodes with the most information
                if (node_visits > entry->visits || depth > entry->depth) {
                    // Double-check that we're not shutting down
                    if (!is_shutdown_.load(std::memory_order_acquire)) {
                        // Update the entry
                        entry->node.store(node, std::memory_order_release);
                        entry->depth = depth;
                        entry->visits = node_visits;
                        
                        // Memory barrier to ensure all updates are visible
                        std::atomic_thread_fence(std::memory_order_release);
                    }
                }
                return;
            }
        }
    }

    // Add new entry
    if (entries_.size() < capacity_) {
        // We're adding a new entry - prepare it outside the critical section
        auto new_entry = std::make_unique<TranspositionEntry>(node, hash, depth, node_visits);
        
        // Now add it to the table with proper synchronization
        size_t index = entries_.size();
        
        // We need to be careful about race conditions when adding to the vector
        // First, update the index map (this will remain valid even if other threads add entries)
        index_map_[hash] = index;
        
        // Add the entry to the table
        entries_.push_back(std::move(new_entry));
        
        // Memory barrier to ensure visibility
        std::atomic_thread_fence(std::memory_order_release);
    } else {
        // Table is full, replace an entry using the improved sampling approach
        // We'll use a more robust approach to select entries for replacement
        
        // Sample a small set of entries to find a replacement candidate
        // This is more efficient than scanning the entire table and reduces contention
        const size_t SAMPLE_SIZE = 16; // Increased sample size for better distribution
        
        // Create a thread-local random generator
        static thread_local std::mt19937 rng(
            static_cast<unsigned int>(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ 
                std::chrono::high_resolution_clock::now().time_since_epoch().count()
            )
        );
        std::uniform_int_distribution<size_t> dist(0, entries_.size() > 0 ? entries_.size() - 1 : 0);
        
        size_t min_visits_index = 0;
        int min_visits = std::numeric_limits<int>::max();
        bool found_valid_entry = false;

        // Try to find a low-visit entry across multiple samples
        for (size_t attempt = 0; attempt < 3 && !found_valid_entry; ++attempt) {
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                // Use a better distribution for sampling with our proper random generator
                size_t index = (hash + i * 97 + dist(rng)) % entries_.size();
                
                // Check if entry exists before accessing it
                if (index >= entries_.size()) {
                    continue;
                }
                
                auto& entry = entries_[index];
                if (!entry) {
                    // Found an empty slot, use it immediately
                    min_visits_index = index;
                    found_valid_entry = true;
                    break;
                }
                
                // Try to lock the entry with a timeout instead of no blocking
                std::unique_lock<std::timed_mutex> entry_lock(entry->lock, std::defer_lock);
                if (entry_lock.try_lock_for(std::chrono::milliseconds(5))) {
                    // We got the lock, now check if this is a good candidate
                    // Note: We keep ownership through entry_lock
                    
                    // For safety, check again that the entry is still valid
                    if (entry && entry->visits < min_visits) {
                        min_visits = entry->visits;
                        min_visits_index = index;
                        found_valid_entry = true;
                    }
                }
                // If we couldn't get the lock, just skip this entry
            }
        }
        
        // If we didn't find a good entry, use a fallback
        if (!found_valid_entry) {
            // Use a simple hash-based selection as fallback
            min_visits_index = hash % entries_.size();
        }

        // Make sure index is valid
        if (min_visits_index >= entries_.size()) {
            min_visits_index = hash % entries_.size();
        }

        // Replace the entry - but first we need to lock it
        if (min_visits_index < entries_.size()) {
            auto& entry_ptr = entries_[min_visits_index];
            
            // Prepare the new entry
            auto new_entry = std::make_unique<TranspositionEntry>(node, hash, depth, node_visits);
            
            // Now handle the replacement with proper locking
            if (entry_ptr) {
                // Lock the entry we're going to replace
                std::unique_lock<std::timed_mutex> entry_guard(entry_ptr->lock, std::try_to_lock);
                if (!entry_guard.owns_lock()) {
                    // Couldn't lock it - it might be in use by another thread
                    // We'll create a new entry instead of replacing it
                    if (entries_.size() < capacity_ + 100) { // Allow a small buffer over capacity
                        // Add as new instead of replacing
                        size_t new_index = entries_.size();
                        entries_.push_back(std::move(new_entry));
                        index_map_[hash] = new_index;
                        return;
                    } else {
                        // We're really out of space, skip this store
                        return;
                    }
                }
                
                // If we got here, we have the lock and can safely update
                
                // Save old hash for cleanup
                uint64_t old_hash = entry_ptr->hash;
                
                // First set the node to nullptr to break any references 
                // (note: this is safe because we have the entry lock)
                entry_ptr->node = nullptr;
                
                // Remove the old hash from index map if it exists and points to this entry
                auto old_it = index_map_.find(old_hash);
                if (old_it != index_map_.end() && old_it->second == min_visits_index) {
                    index_map_.erase(old_it);
                }
            }
            
            // Install the new entry (either replacing null or the entry we just cleaned up)
            entries_[min_visits_index] = std::move(new_entry);
            
            // Update the index map
            index_map_[hash] = min_visits_index;
            
            // Memory barrier to ensure visibility
            std::atomic_thread_fence(std::memory_order_release);
        }
    }
}

void TranspositionTable::clear() {
    // Check if table is being shut down
    if (is_shutdown_.load(std::memory_order_acquire)) {
        return; // Don't attempt to clear a table that's being destroyed
    }
    
    try {
        // A simpler and more reliable approach for clearing the table
        // This completely replaces the entries vector and index map
        
        // First set all node pointers to nullptr to break any cycles
        {
            // Process in small chunks to avoid holding locks for too long
            const size_t CHUNK_SIZE = 100;
            
            for (size_t start = 0; start < entries_.size(); start += CHUNK_SIZE) {
                // Process a chunk of entries
                size_t end = std::min(start + CHUNK_SIZE, entries_.size());
                
                for (size_t i = start; i < end; ++i) {
                    if (i < entries_.size() && entries_[i]) {  // Safety check
                        try {
                            std::lock_guard<std::timed_mutex> entry_guard(entries_[i]->lock);
                            entries_[i]->node.store(nullptr, std::memory_order_release);  // Break the reference
                        } catch (...) {
                            // Ignore lock acquisition errors
                        }
                    }
                }
                
                // Small yield to give other threads a chance
                std::this_thread::yield();
            }
        }
        
        // Now create new empty structures
        std::vector<std::unique_ptr<TranspositionEntry>> new_entries;
        std::unordered_map<uint64_t, size_t> new_index_map;
        
        // Reserve capacity to avoid reallocation
        new_entries.reserve(capacity_);
        new_index_map.reserve(capacity_);
        
        // Acquire all locks before swapping data structures
        {
            // Acquire all locks in order to prevent deadlock
            std::vector<std::unique_lock<std::timed_mutex>> all_locks;
            all_locks.reserve(NUM_LOCKS);
            
            for (size_t i = 0; i < NUM_LOCKS; ++i) {
                try {
                    all_locks.emplace_back(*table_locks_[i]);
                } catch (...) {
                    // If we can't acquire a lock, just continue
                    // The clear operation might be incomplete, but that's better than deadlock
                }
            }
            
            // Swap with the new empty structures
            entries_.swap(new_entries);
            index_map_.swap(new_index_map);
        }
        
        // The old entries will be cleaned up when new_entries goes out of scope
        
        // Reset statistics
        resetStats();
    } catch (const std::exception& e) {
        std::cerr << "Error during transposition table clear: " << e.what() << std::endl;
    }
}

size_t TranspositionTable::size() const {
    return index_map_.size();
}

size_t TranspositionTable::capacity() const {
    return capacity_;
}

float TranspositionTable::hitRate() const {
    // Use memory_order_relaxed for statistics - they don't need strict ordering
    // This is more efficient and we only care about approximate values for stats
    size_t hits = hits_.load(std::memory_order_relaxed);
    size_t misses = misses_.load(std::memory_order_relaxed);
    size_t total = hits + misses;
    
    if (total == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(hits) / static_cast<float>(total);
}

void TranspositionTable::resetStats() {
    // Use sequential consistency here to ensure stats are properly reset
    // in a concurrent environment - this method is called rarely
    hits_.store(0, std::memory_order_seq_cst);
    misses_.store(0, std::memory_order_seq_cst);
}

} // namespace mcts
} // namespace alphazero