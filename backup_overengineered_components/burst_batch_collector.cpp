#include "mcts/burst_batch_collector.h"
#include "mcts/mcts_node.h"
#include "core/igamestate.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace alphazero {
namespace mcts {

// PendingEvaluation is now available from mcts_engine.h

BurstBatchCollector::BurstBatchCollector(size_t target_batch_size, std::chrono::milliseconds burst_timeout)
    : target_batch_size_(target_batch_size)
    , burst_timeout_(burst_timeout)
    , shutdown_(false)
    , collection_active_(false)
    , total_batches_collected_(0)
    , total_items_collected_(0) {
    
    // Reserve space for efficient batch collection
    current_batch_.reserve(target_batch_size_ * 2); // Extra space for burst collection
    
    std::cout << "BurstBatchCollector: Initialized with target batch size " << target_batch_size 
              << " and burst timeout " << burst_timeout.count() << "ms" << std::endl;
}

BurstBatchCollector::~BurstBatchCollector() {
    shutdown();
}

void BurstBatchCollector::start() {
    if (collection_active_.load(std::memory_order_acquire)) {
        return; // Already started
    }
    
    shutdown_.store(false, std::memory_order_release);
    collection_active_.store(true, std::memory_order_release);
    
    // Start the burst collection thread
    collection_thread_ = std::thread(&BurstBatchCollector::burstCollectionLoop, this);
    
    std::cout << "BurstBatchCollector: Started burst collection thread" << std::endl;
}

void BurstBatchCollector::shutdown() {
    if (!collection_active_.load(std::memory_order_acquire)) {
        return; // Already stopped
    }
    
    shutdown_.store(true, std::memory_order_release);
    collection_active_.store(false, std::memory_order_release);
    
    // Wake up the collection thread
    collection_cv_.notify_all();
    
    // Wait for thread to finish
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
    
    std::cout << "BurstBatchCollector: Shutdown complete. Collected " 
              << total_batches_collected_.load() << " batches with "
              << total_items_collected_.load() << " total items" << std::endl;
}

void BurstBatchCollector::submitEvaluation(PendingEvaluation eval) {
    if (shutdown_.load(std::memory_order_acquire)) {
        return;
    }
    
    // Quick validation
    if (!eval.node || !eval.state) {
        return;
    }
    
    // Enqueue evaluation for burst collection
    bool enqueued = pending_queue_.enqueue(std::move(eval));
    if (enqueued) {
        // Notify collection thread that work is available
        collection_cv_.notify_one();
    }
}

std::vector<PendingEvaluation> BurstBatchCollector::collectBatch() {
    std::unique_lock<std::mutex> lock(batch_mutex_);
    
    // Wait for a complete batch or timeout
    ready_cv_.wait(lock, [this] {
        return !completed_batches_.empty() || shutdown_.load(std::memory_order_acquire);
    });
    
    if (!completed_batches_.empty()) {
        auto batch = std::move(completed_batches_.front());
        completed_batches_.pop();
        return batch;
    }
    
    return std::vector<PendingEvaluation>();
}

bool BurstBatchCollector::hasPendingBatch() const {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    return !completed_batches_.empty();
}

size_t BurstBatchCollector::getPendingCount() const {
    return pending_queue_.size_approx();
}

float BurstBatchCollector::getAverageBatchSize() const {
    size_t batches = total_batches_collected_.load(std::memory_order_acquire);
    if (batches == 0) return 0.0f;
    
    return static_cast<float>(total_items_collected_.load(std::memory_order_acquire)) / batches;
}

void BurstBatchCollector::burstCollectionLoop() {
    std::cout << "BurstBatchCollector: Burst collection loop started" << std::endl;
    
    while (!shutdown_.load(std::memory_order_acquire)) {
        try {
            // Phase 1: Rapid burst collection - collect as many items as possible quickly
            current_batch_.clear();
            auto burst_start = std::chrono::steady_clock::now();
            
            // Aggressive initial collection
            collectBurstItems(target_batch_size_);
            
            // Phase 2: Strategic wait for more items if we're below target
            if (current_batch_.size() < target_batch_size_ && 
                !shutdown_.load(std::memory_order_acquire)) {
                
                auto wait_deadline = burst_start + burst_timeout_;
                
                while (std::chrono::steady_clock::now() < wait_deadline &&
                       current_batch_.size() < target_batch_size_ &&
                       !shutdown_.load(std::memory_order_acquire)) {
                    
                    // Collect more items with shorter bursts
                    size_t before = current_batch_.size();
                    collectBurstItems(target_batch_size_ - current_batch_.size());
                    
                    // If no progress, wait briefly
                    if (current_batch_.size() == before) {
                        std::unique_lock<std::mutex> lock(collection_mutex_);
                        collection_cv_.wait_for(lock, std::chrono::milliseconds(2));
                    }
                }
            }
            
            // Phase 3: Process collected batch
            if (!current_batch_.empty()) {
                // Final validation and cleanup
                validateAndCleanBatch();
                
                if (!current_batch_.empty()) {
                    // Submit completed batch
                    {
                        std::lock_guard<std::mutex> lock(batch_mutex_);
                        completed_batches_.push(std::move(current_batch_));
                    }
                    
                    // Update statistics
                    total_batches_collected_.fetch_add(1, std::memory_order_relaxed);
                    total_items_collected_.fetch_add(current_batch_.size(), std::memory_order_relaxed);
                    
                    // Notify waiting threads
                    ready_cv_.notify_all();
                    
                    // Log batch completion
                    if (total_batches_collected_.load() % 10 == 0) {
                        std::cout << "BurstBatchCollector: Completed batch #" 
                                  << total_batches_collected_.load() 
                                  << " with " << current_batch_.size() << " items"
                                  << " (avg: " << getAverageBatchSize() << ")" << std::endl;
                    }
                }
            } else {
                // No items collected, wait for work
                std::unique_lock<std::mutex> lock(collection_mutex_);
                collection_cv_.wait_for(lock, std::chrono::milliseconds(10));
            }
            
        } catch (const std::exception& e) {
            std::cerr << "BurstBatchCollector: Exception in collection loop: " << e.what() << std::endl;
        }
    }
    
    std::cout << "BurstBatchCollector: Burst collection loop terminated" << std::endl;
}

void BurstBatchCollector::collectBurstItems(size_t max_items) {
    const size_t BURST_SIZE = std::min(max_items, size_t(128)); // Collect up to 128 items per burst
    
    // Try bulk dequeue first for efficiency
    if (pending_queue_.size_approx() > 0) {
        std::vector<PendingEvaluation> burst_batch(BURST_SIZE);
        size_t dequeued = pending_queue_.try_dequeue_bulk(burst_batch.data(), BURST_SIZE);
        
        // Process dequeued items
        for (size_t i = 0; i < dequeued && current_batch_.size() < target_batch_size_; ++i) {
            if (burst_batch[i].node && burst_batch[i].state) {
                current_batch_.push_back(std::move(burst_batch[i]));
            }
        }
    }
    
    // Fill remaining space with individual dequeues
    while (current_batch_.size() < max_items && 
           current_batch_.size() < target_batch_size_) {
        
        PendingEvaluation eval;
        if (pending_queue_.try_dequeue(eval)) {
            if (eval.node && eval.state) {
                current_batch_.push_back(std::move(eval));
            }
        } else {
            break; // No more items available
        }
    }
}

void BurstBatchCollector::validateAndCleanBatch() {
    // Remove any invalid items from the batch
    size_t valid_count = 0;
    
    for (size_t i = 0; i < current_batch_.size(); ++i) {
        // Check if item is still valid
        bool is_valid = current_batch_[i].node && 
                       current_batch_[i].state &&
                       !current_batch_[i].node->hasPendingEvaluation();
        
        if (is_valid) {
            // Verify state is still valid
            try {
                is_valid = current_batch_[i].state->validate();
            } catch (...) {
                is_valid = false;
            }
        }
        
        if (is_valid) {
            if (i != valid_count) {
                current_batch_[valid_count] = std::move(current_batch_[i]);
            }
            valid_count++;
        } else {
            // Clear evaluation flag for invalid items
            if (current_batch_[i].node) {
                current_batch_[i].node->clearEvaluationFlag();
            }
        }
    }
    
    current_batch_.resize(valid_count);
}

} // namespace mcts
} // namespace alphazero