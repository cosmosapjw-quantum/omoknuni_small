// src/mcts/batch_barrier.cpp
#include "mcts/batch_barrier.h"
#include <algorithm>

namespace alphazero {
namespace mcts {

BatchBarrier::BatchBarrier(size_t target_batch_size, std::chrono::milliseconds timeout)
    : target_batch_size_(target_batch_size)
    , default_timeout_(timeout) {
    waiting_threads_.reserve(target_batch_size);
}

size_t BatchBarrier::arrive_and_wait(size_t thread_id) {
    return wait_internal(thread_id, default_timeout_);
}

size_t BatchBarrier::try_arrive_and_wait(size_t thread_id, std::chrono::milliseconds custom_timeout) {
    return wait_internal(thread_id, custom_timeout);
}

size_t BatchBarrier::wait_internal(size_t thread_id, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Record arrival time
    auto arrival_time = std::chrono::steady_clock::now();
    
    // Get current generation
    size_t my_generation = generation_.load(std::memory_order_relaxed);
    
    // Add to waiting threads
    size_t my_position = waiting_threads_.size();
    waiting_threads_.push_back(thread_id);
    size_t count = waiting_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
    
    // First thread sets the batch start time
    if (count == 1) {
        batch_start_time_ = arrival_time;
    }
    
    // Check if we've reached target batch size
    if (count >= target_batch_size_) {
        // This thread triggers the batch
        stats_.total_batches.fetch_add(1, std::memory_order_relaxed);
        stats_.full_batches.fetch_add(1, std::memory_order_relaxed);
        
        // Calculate wait time
        auto wait_duration = std::chrono::steady_clock::now() - batch_start_time_;
        stats_.total_wait_time_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(wait_duration).count(),
            std::memory_order_relaxed);
        
        // Call callback if set
        if (batch_callback_) {
            batch_callback_(waiting_threads_);
        }
        
        // Reset for next batch
        waiting_threads_.clear();
        waiting_count_.store(0, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_acq_rel);
        force_release_ = false;
        
        // Wake all waiting threads
        cv_.notify_all();
        
        return my_position;
    }
    
    // Wait for batch completion or timeout
    bool timed_out = !cv_.wait_for(lock, timeout, [this, my_generation] {
        return generation_.load(std::memory_order_acquire) != my_generation || force_release_;
    });
    
    if (timed_out && generation_.load(std::memory_order_acquire) == my_generation) {
        // This thread timed out - check if we should trigger partial batch
        if (waiting_count_.load(std::memory_order_acquire) == waiting_threads_.size()) {
            // We're the last waiting thread - trigger partial batch
            stats_.total_batches.fetch_add(1, std::memory_order_relaxed);
            stats_.partial_batches.fetch_add(1, std::memory_order_relaxed);
            stats_.timeouts.fetch_add(1, std::memory_order_relaxed);
            
            // Calculate wait time
            auto wait_duration = std::chrono::steady_clock::now() - batch_start_time_;
            stats_.total_wait_time_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(wait_duration).count(),
                std::memory_order_relaxed);
            
            // Call callback if set
            if (batch_callback_) {
                batch_callback_(waiting_threads_);
            }
            
            // Reset for next batch
            waiting_threads_.clear();
            waiting_count_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            force_release_ = false;
            
            // Wake any other waiting threads
            cv_.notify_all();
        }
        
        return size_t(-1);  // Indicate timeout
    }
    
    return my_position;
}

void BatchBarrier::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_threads_.clear();
    waiting_count_.store(0, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_acq_rel);
    force_release_ = false;
    cv_.notify_all();
}

void BatchBarrier::force_release() {
    std::lock_guard<std::mutex> lock(mutex_);
    force_release_ = true;
    cv_.notify_all();
}

// ThreadLocalBatchCoordinator implementation
thread_local std::unique_ptr<ThreadLocalBatchCoordinator::ThreadState> 
    ThreadLocalBatchCoordinator::thread_state_;

std::atomic<size_t> ThreadLocalBatchCoordinator::next_thread_id_{0};

ThreadLocalBatchCoordinator::ThreadState& ThreadLocalBatchCoordinator::getThreadState() {
    if (!thread_state_) {
        size_t id = next_thread_id_.fetch_add(1, std::memory_order_relaxed);
        thread_state_ = std::make_unique<ThreadState>(id);
    }
    return *thread_state_;
}

void ThreadLocalBatchCoordinator::registerEvaluation(void* state_ptr, size_t move_count) {
    auto& state = getThreadState();
    state.pending_evaluations.emplace_back(state_ptr, move_count);
}

void ThreadLocalBatchCoordinator::clearPendingEvaluations() {
    auto& state = getThreadState();
    state.pending_evaluations.clear();
}

size_t ThreadLocalBatchCoordinator::getTotalPendingEvaluations() {
    // This would need a more complex implementation to aggregate across threads
    // For now, return local thread count
    if (thread_state_) {
        return thread_state_->pending_evaluations.size();
    }
    return 0;
}

} // namespace mcts
} // namespace alphazero