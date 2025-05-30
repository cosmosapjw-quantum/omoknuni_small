#include "mcts/shared_inference_queue.h"
#include "games/gomoku/gomoku_state.h"
#include "utils/logger.h"
#include <iostream>
#include <algorithm>

namespace alphazero {
namespace mcts {

// Static member definition
std::unique_ptr<SharedInferenceQueue> GlobalInferenceQueue::instance_ = nullptr;

SharedInferenceQueue::SharedInferenceQueue(
    std::shared_ptr<nn::NeuralNetwork> neural_net,
    int max_batch_size,
    int batch_timeout_ms)
    : neural_net_(neural_net),
      max_batch_size_(max_batch_size),
      batch_timeout_ms_(std::min(batch_timeout_ms, 10)),  // Cap timeout at 10ms for responsiveness
      request_queue_(1024)  // Initial capacity - removed consumer_token_ initialization
      {
    
    // Check if GPU attack/defense is available
#ifdef WITH_TORCH
    use_gpu_attack_defense_ = games::gomoku::GomokuState::isGPUEnabled();
#endif
}

SharedInferenceQueue::~SharedInferenceQueue() {
    stop();
}

void SharedInferenceQueue::start() {
    if (!running_) {
        running_ = true;
        processing_thread_ = std::thread(&SharedInferenceQueue::processingLoop, this);
    }
}

void SharedInferenceQueue::stop() {
    if (running_) {
        running_ = false;
        notify_cv_.notify_all();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        std::cout << "SharedInferenceQueue: Stopped. Stats - Total requests: " << stats_.total_requests
                  << ", Total states: " << stats_.total_states
                  << ", Total batches: " << stats_.total_batches
                  << ", Avg batch size: " << stats_.average_batch_size << std::endl;
    }
}

std::future<std::vector<mcts::NetworkOutput>> SharedInferenceQueue::submitBatch(
    std::vector<std::unique_ptr<core::IGameState>> states) {
    
    // LOG_SYSTEM_INFO("SharedInferenceQueue::submitBatch - Submitting {} states", states.size());
    
    InferenceRequest request;
    request.states = std::move(states);
    request.timestamp = std::chrono::steady_clock::now();
    
    auto future = request.promise.get_future();
    
    // Use tokenless enqueue for multi-producer scenario
    bool enqueued = request_queue_.enqueue(std::move(request));
    if (!enqueued) {
        LOG_SYSTEM_ERROR("SharedInferenceQueue::submitBatch - Failed to enqueue request!");
        throw std::runtime_error("Failed to enqueue inference request");
    }
    
    stats_.total_requests++;
    int pending = pending_requests_.fetch_add(1) + 1;
    
    // LOG_SYSTEM_INFO("SharedInferenceQueue::submitBatch - Request enqueued successfully, pending={}, queue_size_approx={}", 
    //                 pending, request_queue_.size_approx());
    
    // Notify processing thread if needed
    if (pending_requests_.load() == 1) {
        // LOG_SYSTEM_INFO("SharedInferenceQueue::submitBatch - Notifying processing thread");
        std::lock_guard<std::mutex> lock(notify_mutex_);
        notify_cv_.notify_one();
    }
    
    return future;
}

void SharedInferenceQueue::processingLoop() {
    // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Started with max_batch_size={}, timeout={}ms", 
    //                 max_batch_size_, batch_timeout_ms_);
    
    std::vector<InferenceRequest> batch;
    batch.reserve(max_batch_size_);
    
    // Debug: Track queue statistics
    int successful_batches = 0;
    int empty_loops = 0;
    auto last_log_time = std::chrono::steady_clock::now();
    
    while (running_) {
        batch.clear();
        int total_states = 0;
        
        auto batch_start_time = std::chrono::steady_clock::now();
        auto timeout_time = batch_start_time + std::chrono::milliseconds(batch_timeout_ms_);
        
        // Log statistics every second instead of cycle count
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 1) {
            // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Stats: successful_batches={}, empty_loops={}, pending={}, queue_size_approx={}", 
            //                successful_batches, empty_loops, pending_requests_.load(), request_queue_.size_approx());
            last_log_time = now;
            empty_loops = 0;  // Reset counter
        }
        
        // Try to collect requests
        int loop_iterations = 0;
        const int MAX_ITERATIONS = 1000;  // Failsafe to prevent infinite loops
        
        // Debug: Log when we start trying to dequeue
        static int dequeue_attempts = 0;
        dequeue_attempts++;
        if (dequeue_attempts <= 10 || dequeue_attempts % 1000 == 0) {
            // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Starting dequeue attempt #{}, pending={}, queue_size={}", 
            //                dequeue_attempts, pending_requests_.load(), request_queue_.size_approx());
        }
        
        // Debug: Check if we enter the inner loop
        if (loop_iterations == 0 && dequeue_attempts <= 10) {
            // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Entering inner loop: running={}, total_states={}, max_batch_size={}", 
            //                running_, total_states, max_batch_size_);
        }
        
        while (running_ && total_states < max_batch_size_ && loop_iterations++ < MAX_ITERATIONS) {
            // Use tokenless dequeue
            InferenceRequest request;
            bool dequeued = request_queue_.try_dequeue(request);
            
            // Debug every 1000 attempts
            static int total_attempts = 0;
            total_attempts++;
            if (total_attempts % 1000 == 0) {
                // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Dequeue attempt #{}: dequeued={}, pending={}, queue_size={}", 
                //                total_attempts, dequeued, pending_requests_.load(), request_queue_.size_approx());
            }
            
            if (dequeued) {
                int request_size = request.states.size();
                
                // Debug: Log if we get empty requests
                if (request_size == 0) {
                    LOG_SYSTEM_ERROR("SharedInferenceQueue::processingLoop - Empty request detected!");
                    pending_requests_.fetch_sub(1);
                    continue;  // Skip empty requests
                }
                
                // Check if this would exceed batch size
                if (total_states + request_size > max_batch_size_) {
                    // Put back the request
                    request_queue_.enqueue(std::move(request));
                    break;  // Process what we have
                }
                
                total_states += request_size;
                batch.push_back(std::move(request));
                pending_requests_.fetch_sub(1);
                
                // Log first few batches for debugging
                static int total_requests_processed = 0;
                total_requests_processed++;
                if (total_requests_processed <= 10) {
                    // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Added request #{} with {} states, batch now has {} requests with {} total states", 
                    //                 total_requests_processed, request_size, batch.size(), total_states);
                }
            }
            
            // Check if we should process or wait
            auto now = std::chrono::steady_clock::now();
            
            // Balanced batching for responsiveness and GPU utilization
            bool should_process = false;
            
            if (total_states >= max_batch_size_) {
                should_process = true;  // Batch is full
            } else if (!batch.empty() && now >= timeout_time) {
                should_process = true;  // Timeout reached - process what we have
            } else if (total_states >= 32 && now >= batch_start_time + std::chrono::milliseconds(5)) {
                should_process = true;  // Reasonable batch size after 5ms
            } else if (total_states >= 16 && now >= batch_start_time + std::chrono::milliseconds(10)) {
                should_process = true;  // Smaller batch after 10ms
            } else if (!batch.empty() && now >= batch_start_time + std::chrono::milliseconds(20)) {
                should_process = true;  // Process any batch after 20ms max wait
            }
            
            if (should_process) {
                // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Breaking to process batch (total_states={}, batch_size={}, elapsed={}ms)", 
                //                 total_states, batch.size(), 
                //                 std::chrono::duration_cast<std::chrono::milliseconds>(now - batch_start_time).count());
                break;
            } else if (batch.empty() || (!dequeued && pending_requests_.load() == 0)) {
                // No work available, wait for notification
                std::unique_lock<std::mutex> lock(notify_mutex_);
                if (pending_requests_.load() == 0 && batch.empty()) {
                    notify_cv_.wait_for(lock, std::chrono::milliseconds(1), 
                        [this] { return pending_requests_.load() > 0 || !running_; });
                }
                
                // Reset timeout after waiting
                if (batch.empty()) {
                    batch_start_time = std::chrono::steady_clock::now();
                    timeout_time = batch_start_time + std::chrono::milliseconds(batch_timeout_ms_);
                }
            } else if (!dequeued) {
                // No items dequeued - check various conditions
                auto pending = pending_requests_.load();
                auto queue_size = request_queue_.size_approx();
                
                if (pending > 0 || queue_size > 0) {
                    // We have items but can't dequeue them
                    static int consecutive_failures = 0;
                    consecutive_failures++;
                    
                    if (consecutive_failures == 1 || consecutive_failures % 1000 == 0) {
                        LOG_SYSTEM_ERROR("SharedInferenceQueue::processingLoop - Cannot dequeue! pending={}, queue_size={}, failures={}, batch_size={}", 
                                       pending, queue_size, consecutive_failures, batch.size());
                        
                        // Try without consumer token as a fallback
                        InferenceRequest fallback_request;
                        if (request_queue_.try_dequeue(fallback_request)) {
                            // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Fallback dequeue succeeded!");
                            // Process this request
                            request = std::move(fallback_request);
                            dequeued = true;
                            consecutive_failures = 0;
                            // Jump back to the dequeued handling code
                            if (dequeued) {
                                int request_size = request.states.size();
                                if (request_size == 0) {
                                    LOG_SYSTEM_ERROR("SharedInferenceQueue::processingLoop - Empty request from fallback!");
                                    pending_requests_.fetch_sub(1);
                                    continue;
                                }
                                
                                if (total_states + request_size > max_batch_size_) {
                                    request_queue_.enqueue(std::move(request));
                                    // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Re-enqueued request, would exceed batch size");
                                    break;
                                }
                                
                                batch.push_back(std::move(request));
                                total_states += request_size;
                                pending_requests_.fetch_sub(1);
                                continue;
                            }
                        }
                    }
                    
                    // Try yielding to give enqueuers a chance
                    std::this_thread::yield();
                    
                    if (consecutive_failures > 10000 && batch.empty()) {
                        // After many failures with no batch, break to avoid infinite loop
                        LOG_SYSTEM_ERROR("SharedInferenceQueue::processingLoop - Breaking after {} failures", consecutive_failures);
                        break;
                    }
                } else {
                    // No pending requests, break to wait
                    break;
                }
            }
        }
        
        // Check if we hit the iteration limit
        if (loop_iterations >= MAX_ITERATIONS) {
            static int iteration_limit_count = 0;
            iteration_limit_count++;
            if (iteration_limit_count <= 10 || iteration_limit_count % 10000 == 0) {
                // LOG_SYSTEM_ERROR("SharedInferenceQueue::processingLoop - Hit iteration limit! batch_size={}, total_states={}, count={}", 
                //                 batch.size(), total_states, iteration_limit_count);
            }
            // Force process whatever we have
            if (!batch.empty()) {
                // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Force processing batch due to iteration limit");
            }
        }
        
        // Process collected batch
        if (!batch.empty() && running_) {
            // LOG_SYSTEM_INFO("SharedInferenceQueue::processingLoop - Processing batch with {} requests containing {} states", 
            //                 batch.size(), total_states);
            processBatch(batch);
            successful_batches++;
        } else if (running_ && pending_requests_.load() == 0) {
            // No work available, wait for notification to avoid busy-waiting
            std::unique_lock<std::mutex> lock(notify_mutex_);
            notify_cv_.wait_for(lock, std::chrono::milliseconds(10), 
                [this] { return pending_requests_.load() > 0 || !running_; });
        } else {
            // We have pending requests but couldn't dequeue them - this is the bug!
            // Add a small sleep to avoid spinning
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void SharedInferenceQueue::processBatch(std::vector<InferenceRequest>& batch) {
    auto start_time = std::chrono::steady_clock::now();
    
    // DEBUG: Log batch processing
    static int debug_batch_count = 0;
    debug_batch_count++;
    
    // Combine all states into a single batch
    std::vector<std::unique_ptr<core::IGameState>> all_states;
    std::vector<int> request_boundaries;  // Track where each request starts
    
    int total_states = 0;
    for (auto& request : batch) {
        request_boundaries.push_back(total_states);
        total_states += request.states.size();
        for (auto& state : request.states) {
            all_states.push_back(std::move(state));
        }
    }
    
    
    try {
        // Special handling for Gomoku with GPU attack/defense
        std::vector<mcts::NetworkOutput> results;
        
#ifdef WITH_TORCH
        if (use_gpu_attack_defense_ && !all_states.empty() && 
            all_states[0]->getGameType() == core::GameType::GOMOKU) {
            
            // Convert to GomokuState pointers for GPU batch processing
            std::vector<const games::gomoku::GomokuState*> gomoku_states;
            gomoku_states.reserve(all_states.size());
            
            for (const auto& state : all_states) {
                auto* gomoku_state = dynamic_cast<const games::gomoku::GomokuState*>(state.get());
                if (gomoku_state) {
                    gomoku_states.push_back(gomoku_state);
                }
            }
            
            if (gomoku_states.size() == all_states.size()) {
                // All states are Gomoku, use GPU batch processing
                auto tensor_batch = games::gomoku::GomokuState::computeEnhancedTensorBatch(gomoku_states);
                
                // Convert tensor batch to format expected by neural network
                std::vector<std::unique_ptr<core::IGameState>> enhanced_states;
                // ... conversion logic ...
                
                // Run neural network inference
                results = neural_net_->inference(all_states);
            }
        }
#endif
        
        // Fallback to regular inference if not using GPU attack/defense
        if (results.empty()) {
            results = neural_net_->inference(all_states);
        }
        
        // Distribute results back to requests
        int result_idx = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            int request_size = (i + 1 < batch.size()) 
                ? request_boundaries[i + 1] - request_boundaries[i]
                : total_states - request_boundaries[i];
            
            std::vector<mcts::NetworkOutput> request_results(
                results.begin() + result_idx,
                results.begin() + result_idx + request_size
            );
            
            batch[i].promise.set_value(std::move(request_results));
            result_idx += request_size;
        }
        
        // Update statistics
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        stats_.total_batches++;
        stats_.total_states += total_states;
        stats_.total_inference_time_ms += duration_ms;
        
        double current_avg = stats_.average_batch_size.load();
        double new_avg = (current_avg * (stats_.total_batches - 1) + total_states) 
                        / stats_.total_batches;
        stats_.average_batch_size = new_avg;
        
        // Estimate GPU utilization based on batch fullness and frequency
        double batch_fullness = static_cast<double>(total_states) / max_batch_size_;
        double time_efficiency = std::min(1.0, static_cast<double>(duration_ms) / batch_timeout_ms_);
        stats_.gpu_utilization = batch_fullness * time_efficiency * 100.0;  // Percentage
        
        
    } catch (const std::exception& e) {
        std::cerr << "SharedInferenceQueue: Error processing batch: " << e.what() << std::endl;
        
        // Set error for all requests
        for (auto& request : batch) {
            request.promise.set_exception(std::current_exception());
        }
    }
}

} // namespace mcts
} // namespace alphazero