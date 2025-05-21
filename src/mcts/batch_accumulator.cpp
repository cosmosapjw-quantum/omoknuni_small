#include "mcts/batch_accumulator.h"
#include "utils/debug_logger.h"
#include <algorithm>
#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision

namespace alphazero {
namespace mcts {

BatchAccumulator::BatchAccumulator(size_t target_batch_size, 
                                 size_t min_viable_batch_size,
                                 std::chrono::milliseconds max_wait_time)
    : target_batch_size_(target_batch_size),
      min_viable_batch_size_(min_viable_batch_size), // Use the provided min_viable_batch_size
      max_wait_time_(max_wait_time), // Use the provided max_wait_time
      batch_start_time_(std::chrono::steady_clock::now()) {
    
    // Ensure sensible defaults and constraints
    if (target_batch_size_ < 8) {
        target_batch_size_ = 8;
    }
    
    // CRITICAL FIX: Make min_viable_batch_size MUCH lower to allow smaller batches to be processed
    // This is the key fix for the zero batch issue - we need to allow very small batches to form
    if (min_viable_batch_size_ < 1 || min_viable_batch_size_ > target_batch_size_) {
        // Set to just 1 item minimum - process any non-empty batch to avoid stalling
        min_viable_batch_size_ = 1;
    } else {
        // Even if caller provided a value, ensure it's not too high - cap at 25% now (was 50%)
        size_t max_min_viable = std::max(size_t(1), target_batch_size_ / 4);
        if (min_viable_batch_size_ > max_min_viable) {
            min_viable_batch_size_ = max_min_viable;
        }
    }
    
    // CRITICAL FIX: Ensure max_wait_time is not too short or too long
    if (max_wait_time_.count() < 1) {
        max_wait_time_ = std::chrono::milliseconds(50); // Default 50ms timeout
    } else if (max_wait_time_.count() > 500) {
        // Also cap the maximum wait time to avoid stalled batches
        max_wait_time_ = std::chrono::milliseconds(500);
    }
    
    
    // Pre-allocate batch to avoid reallocations
    current_batch_.reserve(target_batch_size_ * 2);
}

BatchAccumulator::~BatchAccumulator() {
    stop();
}

void BatchAccumulator::start() {
    
    // Verify parameters are sane before starting
    bool params_valid = true;
    
    if (target_batch_size_ < 8) {
        params_valid = false;
    }
    
    if (min_viable_batch_size_ < 1) {
        params_valid = false;
    }
    
    if (max_wait_time_.count() < 1) {
        params_valid = false;
    }
    
    if (!params_valid) {
        if (target_batch_size_ < 8) target_batch_size_ = 128;
        if (min_viable_batch_size_ < 1) min_viable_batch_size_ = target_batch_size_ * 3 / 4;
        if (max_wait_time_.count() < 1) max_wait_time_ = std::chrono::milliseconds(50);
    }
    
    // Don't start twice
    if (accumulator_thread_.joinable()) {
        // Force update shutdown flag to make sure it's correct
        if (shutdown_.load(std::memory_order_acquire)) {
            shutdown_.store(false, std::memory_order_release);
        }
        
        // Clear any completed batches as a test of thread responsiveness
        std::vector<PendingEvaluation> test_batch;
        while (completed_batches_.try_dequeue(test_batch)) {}
        
        return;
    }
    
    // Clear any existing batches before starting
    std::vector<PendingEvaluation> dummy_batch;
    while (completed_batches_.try_dequeue(dummy_batch)) {}
    
    // Reset the current batch
    std::lock_guard<std::mutex> lock(accumulator_mutex_);
    if (!current_batch_.empty()) {
        current_batch_.clear();
    }
    
    // Reset shutdown flag
    shutdown_.store(false, std::memory_order_release);
    
    // Start accumulator thread
    try {
        accumulator_thread_ = std::thread(&BatchAccumulator::accumulatorLoop, this);
    } catch (const std::exception& e) {
        shutdown_.store(true, std::memory_order_release);
        return;
    } catch (...) {
        shutdown_.store(true, std::memory_order_release);
        return;
    }
    
    // Add a delay to ensure thread actually starts
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    
    // Test if thread is responsive by notifying the condition variable
    if (accumulator_thread_.joinable()) {
        batch_ready_ = true;
        cv_.notify_all();
        batch_ready_ = false;
    }
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

void BatchAccumulator::submitDirectBatch(std::vector<PendingEvaluation>&& batch) {
    if (batch.empty()) {
        return;
    }
    
    // Update statistics
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evaluations_.fetch_add(batch.size(), std::memory_order_relaxed);
    cumulative_batch_size_.fetch_add(batch.size(), std::memory_order_relaxed);
    
    if (batch.size() >= target_batch_size_ * 3 / 4) {
        optimal_batches_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Submit directly to completed queue, bypassing accumulator
    completed_batches_.enqueue(std::move(batch));
    
    // Notify waiting inference worker
    cv_.notify_all();
}

void BatchAccumulator::addEvaluation(PendingEvaluation&& eval) {
    
    
    // CRITICAL FIX: Ensure accumulator is running
    if (!isRunning()) {
        start();
        
        // Verify that the thread actually started
        if (!accumulator_thread_.joinable()) {
            // Try one more time with a small delay
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            start();
            if (!accumulator_thread_.joinable()) {
                // If we still can't start the thread, we're in trouble
                cv_.notify_all();
                return;
            }
        }
    }
    
    // CRITICAL FIX: Validate the state before adding to batch
    if (!eval.state) {
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
        return;
    }
    
    // CRITICAL DEBUG: Validate the node
    if (!eval.node) {
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
        
        // FIXED: Proper batch accumulation with reasonable timing
        // Balance between efficiency (larger batches) and responsiveness
        
        bool batch_at_target = current_batch_.size() >= target_batch_size_;
        bool batch_has_items = !current_batch_.empty();
        
        bool time_threshold_reached = false;
        
        if (batch_has_items) {
            auto current_time = std::chrono::steady_clock::now();
            auto batch_age = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - batch_start_time_).count();
            
            // CRITICAL FIX: More patient batch accumulation for better GPU utilization
            bool waited_briefly = batch_age > 50;   // 50ms wait for decent batches
            bool waited_normally = batch_age > 100; // 100ms wait for smaller batches  
            bool waited_long = batch_age > 200;     // 200ms max wait for any batch
            
            // Prioritize larger batches for GPU efficiency
            if (current_batch_.size() >= target_batch_size_) {
                // Full batch - process immediately
                time_threshold_reached = true;
            } else if (current_batch_.size() >= target_batch_size_ * 3 / 4 && waited_briefly) {
                // 75% full batch - process after brief wait (48 items for 64 target)
                time_threshold_reached = true;
            } else if (current_batch_.size() >= target_batch_size_ / 2 && waited_normally) {
                // 50% full batch - process after normal wait (32 items for 64 target)
                time_threshold_reached = true;
            } else if (current_batch_.size() >= target_batch_size_ / 4 && waited_long) {
                // 25% full batch - process after longer wait (16 items for 64 target)
                time_threshold_reached = true;
            } else if (batch_has_items && waited_long) {
                // Any batch - emergency processing after max wait (only after 200ms)
                time_threshold_reached = true;
            }
        }
        
        // CRITICAL FIX: Ultra-aggressive batch ready flag for immediate processing
        // Process batches as soon as we have any reasonable amount
        batch_ready_ = batch_at_target || time_threshold_reached || 
                      (current_batch_.size() >= 8) ||  // Process immediately when we have 8+ items
                      (batch_has_items && consecutive_empty_iterations_.load(std::memory_order_relaxed) > 2); // Much faster emergency processing
        
    }
    
    // Always notify the condition variable after adding an item
    // IMPORTANT: Do this outside the lock to avoid deadlocks
    cv_.notify_all();  // Prefer notify_all to ensure all waiting threads are notified
    
    // Only send additional notifications for larger batches or when batch is ready
    if (batch_ready_ || current_batch_.size() % 10 == 0) {
        // Small delay to ensure the notification is received
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        cv_.notify_all();
    }
    
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
             
    // Track statistics for logging
    size_t loop_counter = 0;
    size_t total_batches_created = 0;
    size_t total_items_processed = 0;
    
    // Remember last batch processing time to avoid getting stuck
    auto last_batch_time = std::chrono::steady_clock::now();
    // BALANCED: Reasonable force process delay for good batch formation  
    // This ensures batches are processed regularly while allowing accumulation
    const std::chrono::milliseconds force_process_delay(30); // Reasonable 30ms for batch accumulation
    
    // CRITICAL FIX: Add debug counter for consecutive empty iterations
    // After this many empty iterations, try more aggressive measures
    const int MAX_CONSECUTIVE_EMPTY = 10;
    
    consecutive_empty_iterations_.store(0, std::memory_order_relaxed);
    
    // CRITICAL DEBUG: Add counters for detailed statistics
    static std::atomic<int> total_items_added_to_accumulator{0};
    static std::atomic<int> total_batches_submitted{0};
    
    // Track last successful batch submission to detect deadlocks
    auto last_successful_batch = std::chrono::steady_clock::now();
    
    while (!shutdown_.load(std::memory_order_acquire)) {
        loop_counter++;
        std::vector<PendingEvaluation> batch_to_submit;
        
        {
            std::unique_lock<std::mutex> lock(accumulator_mutex_);
            
            if (current_batch_.empty()) {
                // Increment the atomic counter
                consecutive_empty_iterations_.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Reset counter when we have items
                consecutive_empty_iterations_.store(0, std::memory_order_relaxed);
            }
            
            // CRITICAL FIX: Force processing if we haven't processed a batch in a while
            // and have at least one item, or if we have enough items
            bool force_batch_processing = false;
            
            if (!current_batch_.empty()) {
                auto time_since_last_batch = std::chrono::steady_clock::now() - last_batch_time;
                
                // CRITICAL FIX: Immediate aggressive processing for speed
                // Prioritize processing speed over waiting for larger batches
                if (current_batch_.size() >= target_batch_size_ / 2) {
                    // Half batch (32+ items) - process immediately
                    force_batch_processing = true;
                } else if (current_batch_.size() >= target_batch_size_ / 4 && time_since_last_batch > std::chrono::milliseconds(5)) {
                    // Quarter batch (16+ items) - process after just 5ms
                    force_batch_processing = true;
                } else if (current_batch_.size() >= 8 && time_since_last_batch > std::chrono::milliseconds(10)) {
                    // Small decent batch (8+ items) - process after 10ms
                    force_batch_processing = true;
                } else if (current_batch_.size() >= 1 && time_since_last_batch > std::chrono::milliseconds(25)) {
                    // Any batch - emergency processing after just 25ms
                    force_batch_processing = true;
                }
            }
            
            // Phase 1: Wait for optimal batch size or timeout with a MUCH shorter wait
            auto start_wait = std::chrono::steady_clock::now();
            bool timeout_occurred = false;
            
            // NUCLEAR OPTION: Zero-latency immediate processing
            // Don't wait at all - process immediately if we have any reasonable batch
            bool process_immediately = !current_batch_.empty() && 
                                     (current_batch_.size() >= 8 || force_batch_processing || batch_ready_);
            
            if (!process_immediately) {
                // Only wait very briefly if we don't have a decent batch yet
                auto actual_wait_time = std::chrono::milliseconds(1); 
                cv_.wait_for(lock, actual_wait_time, [this, &start_wait, &timeout_occurred, force_batch_processing]() {
                // Get the current state for decision making
                bool is_shutdown = shutdown_.load(std::memory_order_acquire);
                bool is_batch_ready = batch_ready_;
                bool has_items = !current_batch_.empty();
                bool has_enough_items = has_items && (current_batch_.size() >= min_viable_batch_size_);
                
                // Check if we've been waiting with items for too long (time-based processing)
                bool time_to_process = false;
                if (has_items) {
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        current_time - batch_start_time_).count();
                    
                    // CRITICAL FIX: Ultra-aggressive immediate processing for speed
                    // Prioritize processing speed over perfect batch sizes
                    if (current_batch_.size() >= target_batch_size_) {
                        // Full batch - process immediately
                        time_to_process = true;
                        timeout_occurred = true;
                    } else if (current_batch_.size() >= target_batch_size_ / 2) {
                        // Half batch (32+ items) - process immediately without delay
                        time_to_process = true;
                        timeout_occurred = true;
                    } else if (current_batch_.size() >= target_batch_size_ / 4 && elapsed > 5) {
                        // Quarter batch (16+ items) - process after just 5ms
                        time_to_process = true;
                        timeout_occurred = true;
                    } else if (current_batch_.size() >= 8 && elapsed > 10) {
                        // Small decent batch (8+ items) - process after 10ms
                        time_to_process = true;
                        timeout_occurred = true;
                    } else if (has_items && elapsed > 20) {
                        // Any batch - emergency processing after just 20ms
                        time_to_process = true;
                        timeout_occurred = true;
                    }
                }
                
                // Check if we should take emergency action to prevent deadlock
                bool emergency_action = false;
                int current_empty_iterations = consecutive_empty_iterations_.load(std::memory_order_relaxed);
                if (current_empty_iterations > 20) {
                    // After many empty iterations, start processing batches more aggressively
                    if (has_items || current_empty_iterations % 30 == 0) {
                        emergency_action = true;
                    }
                }
                
                // Determine if we should process now based on all the conditions
                bool should_process = is_shutdown || is_batch_ready || has_enough_items || 
                                     time_to_process || force_batch_processing || emergency_action;
                
                return should_process;
                });
            }
            
            // Phase 2: Decide whether to submit batch - immediate processing for decent batches
            bool should_process_batch = process_immediately || 
                                      force_batch_processing || 
                                      !current_batch_.empty();
            if (shutdown_.load(std::memory_order_acquire) || should_process_batch) {
                // Process any remaining items before shutdown
                if (!current_batch_.empty()) {
                    batch_to_submit = std::move(current_batch_);
                    current_batch_.clear();
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
                    
                    // CRITICAL DEBUG: Increment all-time counter
                    total_batches_submitted.fetch_add(1, std::memory_order_relaxed);
                    
                    // Update last successful batch time
                    last_successful_batch = std::chrono::steady_clock::now();
                    
                    // ENHANCED DEBUG: Log batch submission for tracking
                    if (total_batches_submitted.load() <= 20 || total_batches_submitted.load() % 10 == 0) {
                        std::cout << "🚀 [BATCH_SUBMITTED] #" << total_batches_submitted.load() 
                                  << " | Size: " << batch_to_submit.size() 
                                  << " | Type: " << (timeout_occurred ? "timeout" : (force_batch_processing ? "forced" : "optimal"))
                                  << " | Target: " << target_batch_size_ << std::endl;
                    }
                    
                    // Collect metrics
                    if (timeout_occurred || force_batch_processing) {
                        batch_timeouts_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        optimal_batches_.fetch_add(1, std::memory_order_relaxed);
                    }
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
            
            // Update metrics
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_evaluations_.fetch_add(batch_to_submit.size(), std::memory_order_relaxed);
            cumulative_batch_size_.fetch_add(batch_to_submit.size(), std::memory_order_relaxed);
            
            // Update local counters for debug logging
            total_batches_created++;
            total_items_processed += batch_to_submit.size();
            
            // CRITICAL DEBUG: Track items added to accumulator all-time
            total_items_added_to_accumulator.fetch_add(batch_to_submit.size(), std::memory_order_relaxed);
            
            // Submit batch to queue - this is the critical fix
            bool enqueued = completed_batches_.enqueue(std::move(batch_to_submit));
            
            if (enqueued) {
                // Reset consecutive empty counter after successful submission
                consecutive_empty_iterations_.store(0, std::memory_order_relaxed);
                
                // CRITICAL FIX: Ensure that consumers are notified about the new batch
                // This is essential for the inference workers to wake up
                cv_.notify_all();
            } else {
                std::cerr << "ERROR: Failed to enqueue batch to completed_batches_ queue!" << std::endl;
            }
        } else {
            // CRITICAL FIX: Check for potential deadlock conditions
            auto time_since_last_batch = std::chrono::steady_clock::now() - last_successful_batch;
            
            // If we've seen many consecutive empty iterations, try more aggressive waiting strategy
            if (consecutive_empty_iterations_.load(std::memory_order_relaxed) > MAX_CONSECUTIVE_EMPTY) {
                // After many failed attempts, try to sleep longer to allow items to be added
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                
                // DEADLOCK DETECTION: If we haven't processed a batch in a long time, force action
                if (time_since_last_batch > std::chrono::milliseconds(200)) {
                    // Emergency action: try to force process any pending items
                    std::lock_guard<std::mutex> emergency_lock(accumulator_mutex_);
                    if (!current_batch_.empty()) {
                        // Force submit whatever we have to prevent deadlock
                        batch_to_submit = std::move(current_batch_);
                        current_batch_.clear();
                        batch_ready_ = false;
                        last_batch_time = std::chrono::steady_clock::now();
                        last_successful_batch = last_batch_time;
                        
                        std::cout << "⚠️ EMERGENCY BATCH SUBMISSION: Size " << batch_to_submit.size() 
                                  << " (preventing deadlock)" << std::endl;
                    }
                }
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
            
            // Process any emergency batch created above
            if (!batch_to_submit.empty()) {
                bool emergency_enqueued = completed_batches_.enqueue(std::move(batch_to_submit));
                if (emergency_enqueued) {
                    cv_.notify_all();
                    total_batches_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    
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
            current_batch_.clear();
        }
        
        // Reset batch ready flag
        batch_ready_ = false;
        
        // Reset batch start time
        batch_start_time_ = std::chrono::steady_clock::now();
    }
    
    // Clear completed batches queue
    std::vector<PendingEvaluation> dummy_batch;
    while (completed_batches_.try_dequeue(dummy_batch)) {}
    
    // Clear priority queue
    PrioritizedBatch dummy_prioritized;
    while (priority_queue_.try_dequeue(dummy_prioritized)) {}
    
    // Notify any waiting threads
    cv_.notify_all();
    
}

} // namespace mcts
} // namespace alphazero