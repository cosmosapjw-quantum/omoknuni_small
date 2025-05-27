// src/mcts/phmap_transposition_table.cpp
#include "mcts/phmap_transposition_table.h"
#include "utils/logger.h"
#include "utils/progress_bar.h"
#include <algorithm>
#include <vector>

namespace alphazero {
namespace mcts {

PHMapTranspositionTable::PHMapTranspositionTable(const Config& config) 
    : config_(config) {
    
    // Calculate maximum entries based on memory limit
    // Estimate ~100 bytes per entry including overhead
    max_entries_ = (config.size_mb * 1024 * 1024) / 100;
    
    // Configure the hash map
    size_t num_shards = config.num_shards;
    if (num_shards == 0) {
        // Auto-determine based on hardware
        num_shards = std::max(size_t(16), size_t(std::thread::hardware_concurrency() * 2));
    }
    
    // Reserve capacity to avoid rehashing
    entries_.reserve(max_entries_);
    
    auto& progress_manager = alphazero::utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        LOG_MCTS_INFO("PHMap Transposition Table initialized: {} MB, {} max entries, {} shards",
                      config.size_mb, max_entries_, entries_.subcnt());
    }
}

PHMapTranspositionTable::~PHMapTranspositionTable() {
    clear();
}

void PHMapTranspositionTable::store(uint64_t hash, std::shared_ptr<MCTSNode> node, int depth) {
    if (!node) return;
    
    total_stores_.fetch_add(1, std::memory_order_relaxed);
    
    // Create new entry
    auto new_entry = createEntry(hash, node, depth);
    if (!new_entry) return;
    
    // Check if we need to replace an existing entry
    auto it = entries_.find(hash);
    if (it != entries_.end()) {
        // Existing entry found - check replacement policy
        auto& existing = it->second;
        uint32_t current_time = access_counter_.fetch_add(1);
        
        float existing_score = existing->getReplacementScore(current_time);
        float new_score = new_entry->getReplacementScore(current_time);
        
        // Replace if new entry is significantly better
        if (new_score > existing_score * (1.0f + config_.replacement_threshold)) {
            entries_[hash] = new_entry;
            replacements_.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        // New entry - check memory limit first
        if (entries_.size() >= max_entries_) {
            enforceMemoryLimit();
        }
        
        // Insert new entry
        entries_[hash] = new_entry;
    }
}

std::shared_ptr<MCTSNode> PHMapTranspositionTable::lookup(uint64_t hash, int min_depth) {
    total_lookups_.fetch_add(1, std::memory_order_relaxed);
    
    auto it = entries_.find(hash);
    if (it == entries_.end()) {
        return nullptr;
    }
    
    auto& entry = it->second;
    if (!entry || entry->hash != hash) {
        return nullptr;  // Hash collision check
    }
    
    // Check depth requirement
    if (entry->depth < min_depth) {
        return nullptr;
    }
    
    // Update access time
    entry->last_access.store(access_counter_.fetch_add(1), std::memory_order_relaxed);
    
    // Try to get the node
    auto node = entry->node.lock();
    if (node) {
        successful_lookups_.fetch_add(1, std::memory_order_relaxed);
        return node;
    }
    
    // Node has been deleted - remove this entry
    entries_.erase(it);
    return nullptr;
}

void PHMapTranspositionTable::prefetch(const std::vector<uint64_t>& hashes) {
    // PHMap doesn't provide direct prefetch, but we can trigger cache warming
    // by doing lightweight lookups
    for (size_t i = 0; i < std::min(hashes.size(), size_t(8)); ++i) {
        entries_.count(hashes[i]);  // This will load the bucket into cache
    }
}

void PHMapTranspositionTable::clear() {
    entries_.clear();
    total_lookups_ = 0;
    successful_lookups_ = 0;
    total_stores_ = 0;
    replacements_ = 0;
    access_counter_ = 0;
}

PHMapTranspositionTable::Stats PHMapTranspositionTable::getStats() const {
    Stats stats;
    stats.total_lookups = total_lookups_.load();
    stats.successful_lookups = successful_lookups_.load();
    stats.total_stores = total_stores_.load();
    stats.replacements = replacements_.load();
    
    if (stats.total_lookups > 0) {
        stats.hit_rate = static_cast<float>(stats.successful_lookups) / stats.total_lookups;
    } else {
        stats.hit_rate = 0.0f;
    }
    
    // Estimate memory usage
    stats.memory_usage_mb = (entries_.size() * 100) / (1024 * 1024);
    
    return stats;
}

void PHMapTranspositionTable::enforceMemoryLimit() {
    // CRITICAL FIX: More aggressive cleanup and handle expired weak_ptrs
    uint32_t current_time = access_counter_.load();
    [[maybe_unused]] size_t initial_size = entries_.size();
    
    // First pass: remove entries with expired nodes
    std::vector<uint64_t> expired_entries;
    for (const auto& [hash, entry] : entries_) {
        if (!entry || entry->node.expired()) {
            expired_entries.push_back(hash);
        }
    }
    
    // Remove all expired entries
    for (uint64_t hash : expired_entries) {
        entries_.erase(hash);
    }
    
    // If still over limit, remove by score
    if (entries_.size() > max_entries_ * 0.75) {
        std::vector<std::pair<uint64_t, float>> entries_with_scores;
        entries_with_scores.reserve(entries_.size());
        
        // Collect entries and scores
        for (const auto& [hash, entry] : entries_) {
            if (entry) {
                float score = entry->getReplacementScore(current_time);
                entries_with_scores.emplace_back(hash, score);
            }
        }
        
        // Sort by score (lowest first)
        std::partial_sort(entries_with_scores.begin(),
                         entries_with_scores.begin() + entries_with_scores.size() / 2,
                         entries_with_scores.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Remove bottom 50% to free more memory
        size_t remove_count = entries_with_scores.size() / 2;
        for (size_t i = 0; i < remove_count; ++i) {
            entries_.erase(entries_with_scores[i].first);
        }
    }
    
    LOG_MCTS_DEBUG("Transposition table cleanup: {} -> {} entries", initial_size, entries_.size());
}

PHMapTranspositionTable::EntryPtr PHMapTranspositionTable::createEntry(
    uint64_t hash, 
    std::shared_ptr<MCTSNode> node, 
    int depth
) {
    if (!node) return nullptr;
    
    auto entry = std::make_shared<Entry>();
    entry->hash = hash;
    entry->node = node;
    entry->visits = node->getVisitCount();
    entry->value = node->getValue();
    entry->depth = static_cast<uint16_t>(depth);
    entry->age = 0;  // Could be generation counter
    
    // Get best action if available
    auto children = node->getChildren();
    if (!children.empty()) {
        // Find best child by visit count
        auto best_it = std::max_element(children.begin(), children.end(),
            [](const auto& a, const auto& b) {
                return a->getVisitCount() < b->getVisitCount();
            });
        
        // Find the action that leads to this best child
        auto actions = node->getActions();
        if (best_it != children.end() && !actions.empty()) {
            size_t best_idx = std::distance(children.begin(), best_it);
            if (best_idx < actions.size()) {
                entry->best_action = actions[best_idx];
                entry->prior = (*best_it)->getPrior();
            } else {
                entry->best_action = 0;
                entry->prior = 0.0f;
            }
        } else {
            entry->best_action = 0;
            entry->prior = 0.0f;
        }
    } else {
        entry->best_action = 0;
        entry->prior = 0.0f;
    }
    
    entry->last_access.store(access_counter_.fetch_add(1), std::memory_order_relaxed);
    
    return entry;
}

size_t PHMapTranspositionTable::estimateEntrySize(const Entry& entry) const {
    return sizeof(Entry) + sizeof(EntryPtr) + 32; // Overhead estimate
}

} // namespace mcts
} // namespace alphazero