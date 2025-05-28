#include "mcts/shared_inference_queue.h"
#include "games/gomoku/gomoku_state.h"
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
      request_queue_(1024),  // Initial capacity
      producer_token_(request_queue_),
      consumer_token_(request_queue_) {
    
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
        std::cout << "SharedInferenceQueue: Started with max_batch_size=" << max_batch_size_ 
                  << ", timeout=" << batch_timeout_ms_ << "ms" << std::endl;
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
    
    InferenceRequest request;
    request.states = std::move(states);
    request.timestamp = std::chrono::steady_clock::now();
    
    auto future = request.promise.get_future();
    
    // Use lock-free enqueue with producer token
    request_queue_.enqueue(producer_token_, std::move(request));
    stats_.total_requests++;
    pending_requests_.fetch_add(1);
    
    // Notify processing thread if needed
    if (pending_requests_.load() == 1) {
        std::lock_guard<std::mutex> lock(notify_mutex_);
        notify_cv_.notify_one();
    }
    
    return future;
}

void SharedInferenceQueue::processingLoop() {
    std::vector<InferenceRequest> batch;
    batch.reserve(max_batch_size_);
    
    // Bulk dequeue buffer
    std::vector<InferenceRequest> dequeue_buffer(max_batch_size_);
    
    // Debug logging
    // (removed unused variables)
    
    while (running_) {
        batch.clear();
        int total_states = 0;
        
        auto batch_start_time = std::chrono::steady_clock::now();
        auto timeout_time = batch_start_time + std::chrono::milliseconds(batch_timeout_ms_);
        
        // Try to collect requests
        while (running_ && total_states < max_batch_size_) {
            // First, try bulk dequeue for efficiency
            size_t dequeued = request_queue_.try_dequeue_bulk(
                consumer_token_, 
                dequeue_buffer.begin(), 
                std::min(static_cast<size_t>(max_batch_size_ - total_states), dequeue_buffer.size())
            );
            
            if (dequeued > 0) {
                // Add dequeued requests to batch
                for (size_t i = 0; i < dequeued; ++i) {
                    int request_size = dequeue_buffer[i].states.size();
                    
                    // Check if this would exceed batch size
                    if (total_states + request_size > max_batch_size_) {
                        // Put back remaining requests
                        for (size_t j = i; j < dequeued; ++j) {
                            request_queue_.enqueue(std::move(dequeue_buffer[j]));
                            pending_requests_.fetch_add(1);
                        }
                        break;
                    }
                    
                    total_states += request_size;
                    batch.push_back(std::move(dequeue_buffer[i]));
                    pending_requests_.fetch_sub(1);
                }
            }
            
            // Check if we should process or wait
            auto now = std::chrono::steady_clock::now();
            
            // AGGRESSIVE BATCHING: Wait longer to collect full batches for better GPU utilization
            bool should_process = false;
            
            if (total_states >= max_batch_size_) {
                should_process = true;  // Batch is full
            } else if (total_states >= max_batch_size_ * 9 / 10 && now >= batch_start_time + std::chrono::milliseconds(2)) {
                should_process = true;  // 90% full and 2ms passed - almost full, process soon
            } else if (total_states >= max_batch_size_ * 3 / 4 && 
                       now >= batch_start_time + std::chrono::milliseconds(batch_timeout_ms_ * 3 / 4)) {
                should_process = true;  // 75% full and 75% timeout - good balance
            } else if (!batch.empty() && now >= timeout_time) {
                should_process = true;  // Timeout reached - but we have a longer timeout (5ms)
            }
            
            // OPTIMIZED: Dynamic minimum batch size based on time elapsed
            // Start with high requirement but reduce as time passes
            int min_batch_requirement = max_batch_size_ * 3 / 4;  // 75% = 96 states for 128 batch
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - batch_start_time).count();
            
            // Reduce requirement as time passes to avoid stalls
            if (elapsed > 3) min_batch_requirement = max_batch_size_ / 2;  // 50% after 3ms
            if (elapsed > 5) min_batch_requirement = max_batch_size_ / 4;  // 25% after 5ms
            
            if (should_process && total_states < min_batch_requirement && now < timeout_time) {
                should_process = false;  // Wait for better batch utilization
            }
            
            if (should_process) {
                break;
            } else if (batch.empty() || (dequeued == 0 && pending_requests_.load() == 0)) {
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
            } else if (dequeued == 0) {
                // No items dequeued but some pending, longer sleep to reduce CPU usage
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        // Process collected batch
        if (!batch.empty() && running_) {
            processBatch(batch);
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
    
    // DEBUG: Log batch info every 10th batch
    if (debug_batch_count % 10 == 0) {
        std::cout << "[SharedInferenceQueue] Batch #" << debug_batch_count 
                  << ": requests=" << batch.size() 
                  << ", total_states=" << total_states 
                  << ", max_batch=" << max_batch_size_ << std::endl;
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
        
        // Log batch processing
        if (debug_batch_count % 10 == 0 || total_states != max_batch_size_) {
            std::cout << "[DEBUG SharedInferenceQueue] Batch " << debug_batch_count 
                      << ": processed " << total_states << " states"
                      << " (target: " << max_batch_size_ << ") in " 
                      << duration_ms << "ms" << std::endl;
        }
        
        if (stats_.total_batches % 100 == 0) {
            std::cout << "SharedInferenceQueue: Processed batch " << stats_.total_batches
                      << " with " << total_states << " states in " << duration_ms << "ms"
                      << " (avg batch size: " << stats_.average_batch_size 
                      << ", GPU util estimate: " << stats_.gpu_utilization << "%)" << std::endl;
        }
        
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