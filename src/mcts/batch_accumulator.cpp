#include "mcts/batch_accumulator.h"
#include "utils/debug_logger.h"
#include <algorithm>
#include <iostream>

namespace alphazero {
namespace mcts {

BatchAccumulator::BatchAccumulator(size_t target_batch_size, 
                                 size_t min_viable_batch_size,
                                 std::chrono::milliseconds max_wait_time)
    : target_batch_size_(target_batch_size),
      min_viable_batch_size_(min_viable_batch_size > 0 ? min_viable_batch_size : target_batch_size * 3 / 4),
      max_wait_time_(max_wait_time),
      batch_start_time_(std::chrono::steady_clock::now()) {
    
    // Ensure sensible defaults and constraints
    if (target_batch_size_ < 8) {
        target_batch_size_ = 8;
    }
    
    // CRITICAL FIX: Use much smaller minimum batch size to avoid stalling
    // Even a single item is acceptable for processing
    if (min_viable_batch_size_ < 1) {
        min_viable_batch_size_ = 1;
    } else {
        // Limit to a reasonable fraction of target to ensure processing happens
        min_viable_batch_size_ = std::min(min_viable_batch_size_, target_batch_size_ / 4);
    }
    
    // CRITICAL FIX: Use a much shorter timeout to prevent stalling
    if (max_wait_time_ < std::chrono::milliseconds(1)) {
        max_wait_time_ = std::chrono::milliseconds(1);
    } else {
        // Limit maximum timeout to prevent excessive waiting
        max_wait_time_ = std::min(max_wait_time_, std::chrono::milliseconds(10));
    }
    
    std::cout << "BatchAccumulator::Constructor - Created with target_size=" << target_batch_size_
              << ", min_viable=" << min_viable_batch_size_
              << ", max_wait=" << max_wait_time_.count() << "ms" << std::endl;
    
    // Pre-allocate batch to avoid reallocations
    current_batch_.reserve(target_batch_size_ * 2);
}

BatchAccumulator::~BatchAccumulator() {
    stop();
}

void BatchAccumulator::start() {
    std::cout << "====== DEBUG: BatchAccumulator::start CALLED ======" << std::endl;
    std::cout << "BEFORE: this=" << static_cast<void*>(this)
              << ", target_batch_size_=" << target_batch_size_
              << ", min_viable_batch_size_=" << min_viable_batch_size_
              << ", max_wait_time_=" << max_wait_time_.count() << "ms"
              << ", thread_joinable=" << (accumulator_thread_.joinable() ? "yes" : "no")
              << ", shutdown_flag=" << (shutdown_.load(std::memory_order_acquire) ? "true" : "false")
              << ", current_batch_size=" << current_batch_.size()
              << ", completed_batches_size=" << completed_batches_.size_approx()
              << std::endl;
    
    // Verify parameters are sane before starting
    bool params_valid = true;
    
    if (target_batch_size_ < 8) {
        std::cout << "ERROR: target_batch_size_ is too small (" << target_batch_size_ << "), should be at least 8" << std::endl;
        params_valid = false;
    }
    
    if (min_viable_batch_size_ < 1) {
        std::cout << "ERROR: min_viable_batch_size_ is too small (" << min_viable_batch_size_ << "), should be at least 1" << std::endl;
        params_valid = false;
    }
    
    if (max_wait_time_.count() < 1) {
        std::cout << "ERROR: max_wait_time_ is too small (" << max_wait_time_.count() << "ms), should be at least 1ms" << std::endl;
        params_valid = false;
    }
    
    if (!params_valid) {
        std::cout << "FIXING: Adjusting invalid parameters to sane defaults" << std::endl;
        if (target_batch_size_ < 8) target_batch_size_ = 128;
        if (min_viable_batch_size_ < 1) min_viable_batch_size_ = target_batch_size_ * 3 / 4;
        if (max_wait_time_.count() < 1) max_wait_time_ = std::chrono::milliseconds(50);
        
        std::cout << "ADJUSTED PARAMETERS: target_batch_size_=" << target_batch_size_
                  << ", min_viable_batch_size_=" << min_viable_batch_size_
                  << ", max_wait_time_=" << max_wait_time_.count() << "ms" << std::endl;
    }
    
    // Don't start twice
    if (accumulator_thread_.joinable()) {
        std::cout << "NOTICE: BatchAccumulator::start - Thread already running (id=" 
                  << accumulator_thread_.get_id() << "), not starting again" << std::endl;
                  
        // Force update shutdown flag to make sure it's correct
        if (shutdown_.load(std::memory_order_acquire)) {
            std::cout << "WARNING: Thread joinable but shutdown flag is true! Resetting to false." << std::endl;
            shutdown_.store(false, std::memory_order_release);
        }
        
        // Check if the thread is actually running by testing if it's responsive
        // Might need a more sophisticated technique in the future
        std::cout << "CHECKING: Testing if thread is responsive..." << std::endl;
        
        // Clear any completed batches as a test of thread responsiveness
        std::vector<PendingEvaluation> test_batch;
        size_t cleared_batches = 0;
        while (completed_batches_.try_dequeue(test_batch)) {
            cleared_batches++;
        }
        
        if (cleared_batches > 0) {
            std::cout << "INFO: Thread appears active, cleared " << cleared_batches << " pending batches" << std::endl;
        } else {
            std::cout << "INFO: No pending batches found, but thread is joinable" << std::endl;
        }
        
        return;
    }
    
    // Clear any existing batches before starting
    std::vector<PendingEvaluation> dummy_batch;
    size_t cleared_count = 0;
    while (completed_batches_.try_dequeue(dummy_batch)) {
        cleared_count++;
    }
    
    if (cleared_count > 0) {
        std::cout << "NOTICE: Cleared " << cleared_count << " stale batches from the queue" << std::endl;
    }
    
    // Reset the current batch
    std::lock_guard<std::mutex> lock(accumulator_mutex_);
    if (!current_batch_.empty()) {
        std::cout << "NOTICE: Clearing current batch with " << current_batch_.size() << " pending items" << std::endl;
        current_batch_.clear();
    }
    
    // Reset shutdown flag
    shutdown_.store(false, std::memory_order_release);
    
    // Start accumulator thread
    try {
        std::cout << "STARTING: Creating accumulator thread..." << std::endl;
        accumulator_thread_ = std::thread(&BatchAccumulator::accumulatorLoop, this);
        
        // Get thread ID for debugging
        auto thread_id = accumulator_thread_.get_id();
        std::cout << "SUCCESS: Started accumulator thread with ID " << thread_id
                  << " for batch size " << target_batch_size_ 
                  << ", min viable: " << min_viable_batch_size_ 
                  << ", max wait: " << max_wait_time_.count() << "ms" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Failed to start thread: " << e.what() << std::endl;
        shutdown_.store(true, std::memory_order_release);
        return;
    } catch (...) {
        std::cout << "ERROR: Unknown exception when starting thread" << std::endl;
        shutdown_.store(true, std::memory_order_release);
        return;
    }
    
    // Add a delay to ensure thread actually starts
    std::cout << "WAITING: Giving thread time to initialize..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Increased from 10ms to 20ms
    
    // Check if thread is actually running after the delay
    if (!accumulator_thread_.joinable()) {
        std::cout << "ERROR: Thread not joinable after delay! Thread failed to start properly." << std::endl;
    } else {
        // Test if thread is responsive by notifying the condition variable
        std::cout << "NOTIFYING: Sending initial notification to thread..." << std::endl;
        batch_ready_ = true;
        cv_.notify_all();
        
        // Reset batch_ready_ flag after notification
        batch_ready_ = false;
    }
    
    // Log final state
    std::cout << "AFTER: this=" << static_cast<void*>(this)
              << ", thread_id=" << accumulator_thread_.get_id()
              << ", thread_joinable=" << (accumulator_thread_.joinable() ? "yes" : "no")
              << ", shutdown_flag=" << (shutdown_.load(std::memory_order_acquire) ? "true" : "false")
              << std::endl;
    std::cout << "====== DEBUG: BatchAccumulator::start COMPLETED ======" << std::endl;
}

void BatchAccumulator::stop() {
    // Signal thread to stop
    shutdown_.store(true, std::memory_order_release);
    
    // Wake up accumulator thread
    {
        std::lock_guard<std::mutex> lock(accumulator_mutex_);
        batch_ready_ = true;
        cv_.notify_all();
    }
    
    // Wait for thread to finish
    if (accumulator_thread_.joinable()) {
        accumulator_thread_.join();
    }
}

void BatchAccumulator::addEvaluation(PendingEvaluation&& eval) {
    static int items_added = 0;
    items_added++;
    
    // CRITICAL DEBUG: More frequent logging to help diagnose batch issues
    // Always print for initial items to track the flow
    if (items_added <= 50 || items_added % 20 == 0) {
        std::cout << "BatchAccumulator::addEvaluation - Item added #" << items_added 
                 << ", current size=" << current_batch_.size() 
                 << ", target=" << target_batch_size_ 
                 << ", running=" << (isRunning() ? "yes" : "no")
                 << ", state addr=" << (eval.state ? eval.state.get() : nullptr)
                 << ", state valid=" << (eval.state && eval.state->validate() ? "yes" : "no") 
                 << ", node addr=" << (eval.node ? eval.node.get() : nullptr) << std::endl;
        
        utils::debug_logger().logBatchAccumulator(
            "Item added #" + std::to_string(items_added),
            current_batch_.size(),
            target_batch_size_
        );
    }
    
    // CRITICAL FIX: Ensure accumulator is running
    if (!isRunning()) {
        std::cout << "⚠️ CRITICAL: BatchAccumulator::addEvaluation - Accumulator is not running! Starting..." << std::endl;
        start();
        
        // Verify that the thread actually started
        if (!accumulator_thread_.joinable()) {
            std::cout << "ERROR: BatchAccumulator::addEvaluation - Failed to start accumulator thread!" << std::endl;
            // Try one more time with a small delay
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            start();
            if (!accumulator_thread_.joinable()) {
                std::cout << "CRITICAL ERROR: BatchAccumulator::addEvaluation - Failed to start accumulator thread after retry!" << std::endl;
                // If we still can't start the thread, we're in trouble
                // Try to notify anyway, but this probably won't work
                cv_.notify_all();
                return;
            } else {
                std::cout << "BatchAccumulator::addEvaluation - Successfully started accumulator thread on retry" << std::endl;
            }
        } else {
            std::cout << "BatchAccumulator::addEvaluation - Successfully started accumulator thread" << std::endl;
        }
    }
    
    // CRITICAL FIX: Validate the state before adding to batch
    if (!eval.state) {
        std::cout << "WARNING: BatchAccumulator::addEvaluation - Null state in evaluation, skipping" << std::endl;
        return;
    }
    
    // Additional validation
    bool state_valid = false;
    try {
        state_valid = eval.state->validate();
    } catch (...) {
        state_valid = false;
    }
    
    if (!state_valid) {
        std::cout << "WARNING: BatchAccumulator::addEvaluation - Invalid state in evaluation, skipping" << std::endl;
        return;
    }
    
    // After validation, lock the mutex and add to the batch
    {
        std::lock_guard<std::mutex> lock(accumulator_mutex_);
        
        // CRITICAL FIX: Add a timestamp to this batch if it's the first item
        if (current_batch_.empty()) {
            batch_start_time_ = std::chrono::steady_clock::now();
        }
        
        // Add to current batch
        current_batch_.push_back(std::move(eval));
        
        // Increment total count
        total_evaluations_.fetch_add(1, std::memory_order_relaxed);
        
        // Check if batch is ready
        bool batch_is_ready = current_batch_.size() >= target_batch_size_;
        batch_ready_ = batch_is_ready;
        
        // Log when a batch is ready
        if (batch_is_ready) {
            std::cout << "✅ BatchAccumulator::addEvaluation - Batch ready with " 
                     << current_batch_.size() << " items (target: " << target_batch_size_ << ")" << std::endl;
            
            utils::debug_logger().logBatchAccumulator(
                "Batch ready with " + std::to_string(current_batch_.size()) + " items",
                current_batch_.size(), 
                target_batch_size_
            );
        }
    }
    
    // CRITICAL FIX: Always notify after adding an evaluation
    // This is critical to prevent waiting indefinitely for a batch to fill
    // Do this outside the lock to avoid deadlocks
    cv_.notify_all(); // Using notify_all instead of notify_one for reliability
}

bool BatchAccumulator::getCompletedBatch(std::vector<PendingEvaluation>& batch) {
    return completed_batches_.try_dequeue(batch);
}

// Helper function for batch prioritization
BatchAccumulator::BatchPriority BatchAccumulator::calculateBatchPriority(
        const std::vector<PendingEvaluation>& evals, 
        std::chrono::steady_clock::time_point created_time) const {
    
    // Age-based prioritization - older batches get higher priority
    auto current_time = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - created_time).count();
    
    // Size-based priority - batches closer to target size get higher priority
    float size_ratio = static_cast<float>(evals.size()) / target_batch_size_;
    
    // Combine factors for final priority
    if (age > max_wait_time_.count() * 0.8) {
        // Very old batches - highest priority to prevent starvation
        return BatchPriority::High;
    } else if (size_ratio > 0.9 || (age > max_wait_time_.count() * 0.5 && size_ratio > 0.7)) {
        // Large batches or moderately old batches with decent size
        return BatchPriority::High;
    } else if (size_ratio > 0.6 || age > max_wait_time_.count() * 0.3) {
        // Medium-sized batches or slightly aged batches
        return BatchPriority::Normal;
    } else {
        // Small, fresh batches
        return BatchPriority::Low;
    }
}

void BatchAccumulator::accumulatorLoop() {
    std::cout << "BatchAccumulator::accumulatorLoop - Thread started, target_size=" 
             << target_batch_size_ << ", min_viable=" << min_viable_batch_size_ 
             << ", wait_time=" << max_wait_time_.count() << "ms" << std::endl;
             
    // Track statistics for logging
    size_t loop_counter = 0;
    size_t total_batches_created = 0;
    size_t total_items_processed = 0;
    
    // Remember last batch processing time to avoid getting stuck
    auto last_batch_time = std::chrono::steady_clock::now();
    const std::chrono::milliseconds force_process_delay(100); // Reduced from 500ms to 100ms - Force process if no batch for 100ms
    
    // CRITICAL FIX: Add debug counter for consecutive empty iterations
    int consecutive_empty_iterations = 0;
    // After this many empty iterations, try more aggressive measures
    const int MAX_CONSECUTIVE_EMPTY = 10;
    
    while (!shutdown_.load(std::memory_order_acquire)) {
        loop_counter++;
        std::vector<PendingEvaluation> batch_to_submit;
        
        {
            std::unique_lock<std::mutex> lock(accumulator_mutex_);
            
            // Debug logging every 100 iterations or for first few
            if (loop_counter <= 10 || loop_counter % 100 == 0) {
                std::cout << "BatchAccumulator::accumulatorLoop - [Iteration " << loop_counter 
                         << "] current_batch_size=" << current_batch_.size() 
                         << ", total_batches=" << total_batches_created
                         << ", total_items=" << total_items_processed 
                         << ", consecutive_empty=" << consecutive_empty_iterations << std::endl;
            }
            
            // CRITICAL FIX: Diagnose why no items are being added
            if (current_batch_.empty()) {
                consecutive_empty_iterations++;
                
                // After several empty iterations, print debugging info
                if (consecutive_empty_iterations == MAX_CONSECUTIVE_EMPTY) {
                    std::cout << "WARNING: BatchAccumulator::accumulatorLoop - " 
                             << consecutive_empty_iterations 
                             << " consecutive empty iterations. This may indicate a problem with the leaf queue." 
                             << std::endl;
                }
                
                // Every MAX_CONSECUTIVE_EMPTY iterations, print more detailed diagnostics
                if (consecutive_empty_iterations % MAX_CONSECUTIVE_EMPTY == 0) {
                    std::cout << "DIAGNOSTIC: BatchAccumulator loop " << loop_counter 
                             << ", no items being added to batch for " << consecutive_empty_iterations 
                             << " consecutive iterations." << std::endl;
                }
            } else {
                // Reset counter when we have items
                consecutive_empty_iterations = 0;
            }
            
            // CRITICAL FIX: Force processing if we haven't processed a batch in a while
            // and have at least one item, or if we have enough items
            bool force_batch_processing = false;
            
            if (!current_batch_.empty()) {
                auto time_since_last_batch = std::chrono::steady_clock::now() - last_batch_time;
                
                // Force process if:
                // 1. It's been too long since last batch and we have at least one item
                // 2. We have at least min_viable_batch_size_ items
                if (time_since_last_batch > force_process_delay || 
                    current_batch_.size() >= min_viable_batch_size_) {
                    
                    std::cout << "BatchAccumulator::accumulatorLoop - FORCING BATCH PROCESSING after " 
                              << std::chrono::duration_cast<std::chrono::milliseconds>(time_since_last_batch).count() 
                              << "ms with " << current_batch_.size() << " items (min viable: " 
                              << min_viable_batch_size_ << ")" << std::endl;
                    force_batch_processing = true;
                }
            }
            
            // Phase 1: Wait for optimal batch size or timeout with a shorter wait
            auto start_wait = std::chrono::steady_clock::now();
            bool timeout_occurred = false;
            
            // CRITICAL FIX: Use an extremely short timeout to avoid getting stuck
            auto actual_wait_time = std::chrono::milliseconds(1); // Even faster 1ms timeout
            
            auto wait_result = cv_.wait_for(lock, actual_wait_time, [this, &start_wait, &timeout_occurred, force_batch_processing]() {
                // Check if batch is ready, shutdown requested, or force processing
                if (batch_ready_ || shutdown_.load(std::memory_order_acquire) || force_batch_processing) {
                    std::cout << "BatchAccumulator::accumulatorLoop - Predicate returned true: batch_ready=" 
                              << (batch_ready_ ? "yes" : "no") 
                              << ", shutdown=" << (shutdown_.load(std::memory_order_acquire) ? "yes" : "no")
                              << ", force_processing=" << (force_batch_processing ? "yes" : "no")
                              << ", batch_size=" << current_batch_.size() << std::endl;
                    return true;
                }
                
                // CRITICAL FIX: Process ANY batch after 1ms
                // This ensures we don't get stuck waiting for more items that never arrive
                auto elapsed = std::chrono::steady_clock::now() - start_wait;
                
                // CRITICAL FIX: Always process at least one item immediately
                // This is critical to prevent stalls when the item flow is slow
                if (!current_batch_.empty()) { 
                    std::cout << "BatchAccumulator::accumulatorLoop - Processing non-empty batch immediately: "
                              << current_batch_.size() << " items" << std::endl;
                    timeout_occurred = true;
                    return true;
                }
                
                return false;
            });
            
            // Log wait result for debugging
            if (loop_counter <= 20 || loop_counter % 100 == 0) {
                std::string reason;
                if (batch_ready_) reason = "batch_ready";
                else if (timeout_occurred) reason = "timeout";
                else if (force_batch_processing) reason = "forced";
                else if (shutdown_.load(std::memory_order_acquire)) reason = "shutdown";
                else reason = "unknown";
                
                std::cout << "BatchAccumulator::accumulatorLoop - Wait completed: " 
                          << (wait_result ? "true" : "false") 
                          << ", reason=" << reason 
                          << ", current_size=" << current_batch_.size() << std::endl;
            }
            
            // Phase 2: Decide whether to submit batch
            if (shutdown_.load(std::memory_order_acquire)) {
                // Process any remaining items before shutdown
                if (!current_batch_.empty()) {
                    batch_to_submit = std::move(current_batch_);
                    current_batch_.clear();
                    std::cout << "BatchAccumulator::accumulatorLoop - Processing final batch of " 
                             << batch_to_submit.size() << " items during shutdown" << std::endl;
                }
            } else if (batch_ready_ || timeout_occurred || force_batch_processing) {
                // Either batch_ready_ is true (optimal batch size reached)
                // or timeout_occurred is true (waited long enough with viable batch)
                // or force_batch_processing is true (waited too long since last batch)
                if (!current_batch_.empty()) {
                    batch_to_submit = std::move(current_batch_);
                    current_batch_.clear();
                    batch_ready_ = false;
                    
                    // Remember last batch processing time
                    last_batch_time = std::chrono::steady_clock::now();
                    
                    // Always log batch creation now for better diagnostics
                    std::cout << "✅ BatchAccumulator::accumulatorLoop - Created batch with " << batch_to_submit.size() 
                              << " items (target: " << target_batch_size_ 
                              << ", timeout: " << (timeout_occurred ? "yes" : "no")
                              << ", forced: " << (force_batch_processing ? "yes" : "no") << ")" << std::endl;
                    
                    // Collect metrics
                    if (timeout_occurred || force_batch_processing) {
                        batch_timeouts_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        optimal_batches_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    std::cout << "BatchAccumulator::accumulatorLoop - Batch ready or timeout but no items in current batch" << std::endl;
                }
            }
            
            // Reset batch start time for next batch
            batch_start_time_ = std::chrono::steady_clock::now();
        }
        
        // Process any prioritized batches - use lock-free queue outside the lock
        PrioritizedBatch prioritized_batch;
        if (batch_to_submit.empty()) {
            // Check if we have any prioritized batches
            if (priority_queue_.try_dequeue(prioritized_batch)) {
                // Update priority based on current age
                auto updated_priority = calculateBatchPriority(
                    prioritized_batch.evals, 
                    prioritized_batch.created_time
                );
                
                // Accept any batch if we haven't processed one in a while
                auto time_since_last_batch = std::chrono::steady_clock::now() - last_batch_time;
                bool accept_any_batch = time_since_last_batch > force_process_delay;
                
                // Submit high priority or any batch if we're in accept_any_batch mode
                if (updated_priority == BatchPriority::High || accept_any_batch) {
                    batch_to_submit = std::move(prioritized_batch.evals);
                    std::cout << "BatchAccumulator::accumulatorLoop - Using prioritized batch with " 
                             << batch_to_submit.size() << " items (priority: " 
                             << (updated_priority == BatchPriority::High ? "high" : "any")
                             << ", accept_any_batch: " << (accept_any_batch ? "yes" : "no") << ")" << std::endl;
                    
                    // Update last batch time
                    last_batch_time = std::chrono::steady_clock::now();
                } else {
                    // Re-queue with updated priority
                    priority_queue_.enqueue(std::move(prioritized_batch), updated_priority);
                }
            }
        }
        
        // Submit the batch outside the lock if we have one
        if (!batch_to_submit.empty()) {
            // Do quick validation to count valid items
            size_t valid_count = 0;
            size_t null_state_count = 0;
            
            for (const auto& eval : batch_to_submit) {
                if (eval.state) {
                    valid_count++;
                } else {
                    null_state_count++;
                }
            }
            
            if (null_state_count > 0) {
                std::cout << "WARNING: BatchAccumulator::accumulatorLoop - " 
                         << null_state_count << " out of " << batch_to_submit.size() 
                         << " items have null states" << std::endl;
            }
            
            // Update metrics
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_evaluations_.fetch_add(batch_to_submit.size(), std::memory_order_relaxed);
            cumulative_batch_size_.fetch_add(batch_to_submit.size(), std::memory_order_relaxed);
            
            // Update local counters for debug logging
            total_batches_created++;
            total_items_processed += batch_to_submit.size();
            
            // Submit batch to queue
            bool enqueued = completed_batches_.enqueue(std::move(batch_to_submit));
            
            if (!enqueued) {
                std::cout << "ERROR: BatchAccumulator::accumulatorLoop - Failed to enqueue batch to completed_batches_" << std::endl;
            } else {
                std::cout << "✅ BatchAccumulator::accumulatorLoop - Successfully enqueued batch of " 
                         << valid_count << " valid items" << std::endl;
            }
        } else {
            // CRITICAL FIX: If we've seen many consecutive empty iterations, try more aggressive waiting strategy
            if (consecutive_empty_iterations > MAX_CONSECUTIVE_EMPTY) {
                // After many failed attempts, try to sleep longer to allow items to be added
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                // If no batch to submit, use adaptive sleep to avoid busy waiting
                auto queue_size_approx = completed_batches_.size_approx();
                
                if (queue_size_approx > 5) {
                    // If queue has many pending batches, sleep longer to allow consumer to catch up
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } else {
                    // Normal sleep when no batch ready - very short to ensure responsiveness
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                }
            }
        }
    }
    
    std::cout << "BatchAccumulator::accumulatorLoop - Thread exiting, processed " 
             << total_batches_created << " batches with " 
             << total_items_processed << " total items" << std::endl;
}

std::tuple<float, size_t, size_t, size_t> BatchAccumulator::getStats() const {
    // Calculate average batch size
    size_t batches = total_batches_.load(std::memory_order_relaxed);
    float avg_batch_size = 0.0f;
    if (batches > 0) {
        avg_batch_size = static_cast<float>(cumulative_batch_size_.load(std::memory_order_relaxed)) / batches;
    }
    
    return std::make_tuple(
        avg_batch_size,
        batches,
        batch_timeouts_.load(std::memory_order_relaxed),
        optimal_batches_.load(std::memory_order_relaxed)
    );
}

void BatchAccumulator::updateParameters(size_t target_size, 
                                      size_t min_viable_size,
                                      std::chrono::milliseconds max_wait) {
    std::lock_guard<std::mutex> lock(accumulator_mutex_);
    
    target_batch_size_ = target_size;
    min_viable_batch_size_ = min_viable_size > 0 ? min_viable_size : target_size * 3 / 4;
    max_wait_time_ = max_wait;
    
    // Resize current batch if needed
    current_batch_.reserve(target_batch_size_ * 2);
}

void BatchAccumulator::reset() {
    // Use a lock to protect access to current_batch_
    {
        std::lock_guard<std::mutex> lock(accumulator_mutex_);
        
        // Clear current batch
        if (!current_batch_.empty()) {
            std::cout << "BatchAccumulator::reset - Clearing current batch with " 
                     << current_batch_.size() << " items" << std::endl;
            current_batch_.clear();
        }
        
        // Reset batch ready flag
        batch_ready_ = false;
        
        // Reset batch start time
        batch_start_time_ = std::chrono::steady_clock::now();
    }
    
    // Clear completed batches queue
    std::vector<PendingEvaluation> dummy_batch;
    size_t cleared_count = 0;
    while (completed_batches_.try_dequeue(dummy_batch)) {
        cleared_count++;
    }
    
    if (cleared_count > 0) {
        std::cout << "BatchAccumulator::reset - Cleared " << cleared_count 
                 << " batches from completed queue" << std::endl;
    }
    
    // Clear priority queue
    PrioritizedBatch dummy_prioritized;
    size_t cleared_priority_count = 0;
    while (priority_queue_.try_dequeue(dummy_prioritized)) {
        cleared_priority_count++;
    }
    
    if (cleared_priority_count > 0) {
        std::cout << "BatchAccumulator::reset - Cleared " << cleared_priority_count 
                 << " batches from priority queue" << std::endl;
    }
    
    // Notify any waiting threads
    cv_.notify_all();
    
    std::cout << "BatchAccumulator::reset - All pending batches cleared" << std::endl;
}

} // namespace mcts
} // namespace alphazero