#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include "mcts/phmap_transposition_table.h"
#include "utils/gamestate_pool.h"
#include "utils/shutdown_manager.h"
#include "utils/progress_bar.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <vector>
#include <future>
#include <moodycamel/concurrentqueue.h>
#include "mcts/shared_inference_queue.h"

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

void MCTSEngine::executeTaskflowSearch(MCTSNode* root, int num_simulations) {
    
    if (!root) {
        std::cerr << "ERROR: Root node is null!" << std::endl;
        return;
    }
    
    if (!direct_inference_fn_) {
        std::cerr << "ERROR: No inference function available!" << std::endl;
        return;
    }
    
    
    // Expand root if needed
    if (!root->isExpanded() && !root->isTerminal()) {
        root->expand(settings_.use_progressive_widening,
                    settings_.progressive_widening_c,
                    settings_.progressive_widening_k);
    }
    
    auto search_start = std::chrono::steady_clock::now();
    
    // CRITICAL FIX: Initialize memory manager for 64GB system
    auto& memory_manager = AggressiveMemoryManager::getInstance();
    AggressiveMemoryManager::Config config;
    config.warning_threshold_gb = 32.0;   // 50% of 64GB RAM
    config.critical_threshold_gb = 40.0; // 62.5% of RAM
    config.emergency_threshold_gb = 48.0; // 75% of RAM - plenty of headroom
    memory_manager.setConfig(config);
    
    // Performance metrics
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> shutdown(false);
    std::atomic<long> total_inference_time_us(0);
    
    // Lock-free queues with limited capacity to prevent memory explosion
    struct BatchItem {
        MCTSNode* node;
        std::unique_ptr<core::IGameState> state;
        std::vector<MCTSNode*> path;
    };
    
    struct EvalResult {
        MCTSNode* node;
        float value;
        std::vector<float> policy;
        std::vector<MCTSNode*> path;
    };
    
    const int queue_capacity = 1024;  // Larger queue for high-performance system
    moodycamel::ConcurrentQueue<BatchItem> leaf_queue(queue_capacity);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(queue_capacity);
    
    // CRITICAL: Track pending evaluations to prevent runaway leaf generation
    std::atomic<int> pending_evaluations(0);
    const int max_pending = settings_.batch_size * 4;  // Allow more pending for 8 streams
    
    std::vector<std::thread> threads;
    
    // CRITICAL FIX: Optimize thread count for Ryzen 9 5900X (24 threads)
    int actual_threads = std::min(settings_.num_threads, 12);  // Use up to 12 threads for leaf collection
    
    // Leaf collection threads
    for (int worker_id = 0; worker_id < actual_threads; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            std::mt19937 thread_rng(std::random_device{}() + worker_id);
            int local_collected = 0;
            int cleanup_counter = 0;
            
            while (!shutdown.load() && !utils::isShutdownRequested() && simulations_completed.load() < num_simulations) {
                // PIPELINE OPTIMIZATION: Allow slight oversubscription for better throughput
                int dynamic_max_pending = max_pending;
                if (simulations_completed.load() > num_simulations / 2) {
                    // Allow more pending in second half for pipeline efficiency
                    dynamic_max_pending = max_pending * 3 / 2;
                }
                
                while (pending_evaluations.load() > dynamic_max_pending && !shutdown.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2)); // Reduce CPU usage
                }
                
                // CRITICAL FIX: Less frequent memory cleanup with 64GB RAM
                if (++cleanup_counter % 200 == 0) {  // Even less frequent with ample RAM
                    auto pressure = memory_manager.getMemoryPressure();
                    if (pressure >= AggressiveMemoryManager::PressureLevel::WARNING) {
                        memory_manager.forceCleanup(pressure);
                    }
                    
                    // More aggressive cleanup on worker 0
                    if (worker_id == 0 && cleanup_counter % 100 == 0) {
                        #ifdef WITH_TORCH
                        if (torch::cuda::is_available()) {
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        }
                        #endif
                    }
                }
                
                // Tree traversal
                MCTSNode* current = root;
                std::vector<MCTSNode*> path;
                
                // Debug output - show worker activity
                static std::atomic<int> worker_attempts(0);
                int attempt_num = worker_attempts.fetch_add(1);
                if (attempt_num % 50 == 0) {
                    // Worker attempt logged
                }
                
                auto state = root->getState().clone();
                
                // Apply virtual loss to root
                current->applyVirtualLoss(settings_.virtual_loss);
                path.push_back(current);
                
                // Selection with timeout detection
                int selection_iterations = 0;
                const int MAX_SELECTION_ITERATIONS = 1000;  // Prevent infinite loops
                
                while (current && !current->isLeaf() && !state->isTerminal()) {
                    if (++selection_iterations > MAX_SELECTION_ITERATIONS) {
                        std::cerr << "[ERROR] Worker " << worker_id << " stuck in selection after " 
                                  << MAX_SELECTION_ITERATIONS << " iterations" << std::endl;
                        // Revert virtual loss and skip this simulation
                        for (auto it = path.rbegin(); it != path.rend(); ++it) {
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                        }
                        break;
                    }
                    
                    auto next = current->selectBestChildUCB(settings_.exploration_constant, thread_rng);
                    if (!next) {
                        // Revert virtual loss if selection fails
                        for (auto it = path.rbegin(); it != path.rend(); ++it) {
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                        }
                        break;
                    }
                    
                    next->applyVirtualLoss(settings_.virtual_loss);
                    path.push_back(next.get());
                    state->makeMove(next->getAction());
                    current = next.get();
                }
                
                if (!current) continue;
                
                // Handle terminal nodes
                if (state->isTerminal()) {
                    // Calculate value from perspective of root player
                    float value = 0.0f;
                    auto result = state->getGameResult();
                    int root_player = root->getState().getCurrentPlayer();
                    
                    if (result == core::GameResult::WIN_PLAYER1) {
                        value = (root_player == 1) ? 1.0f : -1.0f;
                    } else if (result == core::GameResult::WIN_PLAYER2) {
                        value = (root_player == 2) ? 1.0f : -1.0f;
                    }
                    
                    // Adjust for player perspective
                    if (path.size() % 2 == 0) {
                        value = -value;
                    }
                    
                    // Immediate backpropagation
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        (*it)->update(value);
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                        value = -value;
                    }
                    
                    simulations_completed.fetch_add(1);
                    continue;
                }
                
                // Expansion
                if (current->getVisitCount() > 0 && !current->isExpanded()) {
                    current->expand(settings_.use_progressive_widening,
                                  settings_.progressive_widening_c,
                                  settings_.progressive_widening_k);
                }
                
                // Add to evaluation queue with backpressure
                BatchItem item;
                item.node = current;
                item.state = std::move(state);
                item.path = std::move(path);
                
                // Try to enqueue, but don't block forever
                bool enqueued = false;
                for (int retry = 0; retry < 50 && !enqueued && !shutdown.load(); ++retry) {
                    enqueued = leaf_queue.enqueue(std::move(item));
                    if (!enqueued) {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                }
                
                if (enqueued) {
                    pending_evaluations.fetch_add(1);
                    leaves_collected.fetch_add(1);
                    local_collected++;
                } else {
                    // Queue full - revert virtual loss and skip this simulation
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                    }
                }
                
                if (leaves_collected.load() % 10 == 0) {
                    // Leaves collected
                }
            }
        });
    }
    
    // Batch processing thread - OPTIMIZED with continuous batching for high-end hardware
    threads.emplace_back([&]() {
        
        // CRITICAL FIX: Multiple concurrent inference streams for RTX 3060 Ti + Ryzen 9 5900X
        struct InferenceStream {
            std::vector<BatchItem> batch;
            std::future<std::vector<NetworkOutput>> future;
            bool pending = false;
            std::chrono::steady_clock::time_point start_time;
            int stream_id;
        };
        
        const int NUM_STREAMS = 8;  // RTX 3060 Ti can handle multiple concurrent streams efficiently
        std::vector<InferenceStream> streams(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams[i].batch.reserve(settings_.batch_size);
            streams[i].stream_id = i;
        }
        
        
        auto last_batch_time = std::chrono::steady_clock::now();
        const auto max_batch_wait = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        // CRITICAL FIX: Adaptive batch sizing based on remaining simulations and memory
        auto get_adaptive_batch_size = [&]() -> size_t {
            int remaining = num_simulations - simulations_completed.load();
            size_t base_size = settings_.batch_size;
            
            // Keep large batches for GPU efficiency
            size_t target_size;
            if (remaining >= static_cast<int>(base_size)) {
                target_size = base_size;  // Full size when plenty of work
            } else if (remaining >= 64) {
                target_size = 64;  // Maintain good GPU utilization
            } else if (remaining >= 32) {
                target_size = 32;  // Minimum efficient batch
            } else {
                target_size = std::max(1, remaining);  // Final cleanup
            }
            
            // Also consider memory pressure
            auto level = memory_manager.getMemoryPressure();
            switch (level) {
                case AggressiveMemoryManager::PressureLevel::WARNING:
                    target_size = target_size * 3 / 4;
                    break;
                case AggressiveMemoryManager::PressureLevel::CRITICAL:
                    target_size = target_size / 2;
                    break;
                case AggressiveMemoryManager::PressureLevel::EMERGENCY:
                    target_size = std::max(size_t(16), target_size / 4);
                    break;
                default:
                    break;
            }
            
            return target_size;
        };
        
        while ((!shutdown.load() && !utils::isShutdownRequested()) || leaf_queue.size_approx() > 0) {
            
            // STEP 1: Check and process completed inference streams
            for (auto& stream : streams) {
                if (stream.pending && stream.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    auto nn_results = stream.future.get();
                    stream.pending = false;
                    
                    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - stream.start_time).count();
                    
                    // CRITICAL FIX: Track neural network statistics
                    batches_processed.fetch_add(1);
                    total_batch_size.fetch_add(stream.batch.size());
                    total_inference_time_us.fetch_add(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - stream.start_time
                        ).count()
                    );
                    
                    // Process results immediately
                    int processed = 0;
                    for (size_t i = 0; i < stream.batch.size() && i < nn_results.size(); ++i) {
                        EvalResult result;
                        result.node = stream.batch[i].node;
                        result.value = nn_results[i].value;
                        result.policy = nn_results[i].policy;
                        result.path = std::move(stream.batch[i].path);
                        
                        result_queue.enqueue(std::move(result));
                        processed++;
                    }
                    
                    pending_evaluations.fetch_sub(processed);
                    stream.batch.clear();
                }
            }
            
            // STEP 2: Find an available stream for new batch
            InferenceStream* available_stream = nullptr;
            for (auto& stream : streams) {
                if (!stream.pending) {
                    available_stream = &stream;
                    break;
                }
            }
            
            // STEP 3: If we have an available stream, try to fill it
            if (available_stream) {
                available_stream->batch.clear();
                
                // DIRECT BATCHING: Collect exactly what we need
                size_t target_batch = get_adaptive_batch_size();
                
                // Don't over-collect
                int remaining_sims = num_simulations - simulations_completed.load();
                if (remaining_sims <= 0) break;
                target_batch = std::min(target_batch, static_cast<size_t>(remaining_sims));
                
                size_t collected = leaf_queue.try_dequeue_bulk(
                    std::back_inserter(available_stream->batch), 
                    target_batch
                );
                
                if (collected > 0) {
                    // Process any collected items, no minimum required
                    
                    // STEP 4: Process the batch immediately - no waiting!
                    auto now = std::chrono::steady_clock::now();
                    
                    // Separate terminal and non-terminal
                    std::vector<BatchItem> non_terminal_batch;
                    std::vector<EvalResult> terminal_results;
                    
                    for (auto& item : available_stream->batch) {
                        if (item.node->isTerminal()) {
                            EvalResult result;
                            result.node = item.node;
                            result.value = 0.0f;  // Terminal value already handled
                            result.path = std::move(item.path);
                            terminal_results.push_back(std::move(result));
                        } else {
                            non_terminal_batch.push_back(std::move(item));
                        }
                    }
                    
                    // Process terminal results immediately
                    for (auto& result : terminal_results) {
                        result_queue.enqueue(std::move(result));
                    }
                    pending_evaluations.fetch_sub(terminal_results.size());
                    
                    // Launch inference for non-terminal states
                    if (!non_terminal_batch.empty()) {
                        std::vector<std::unique_ptr<core::IGameState>> state_batch;
                        state_batch.reserve(non_terminal_batch.size());
                        
                        for (auto& item : non_terminal_batch) {
                            state_batch.push_back(std::move(item.state));
                        }
                        
                        // Replace batch with non-terminal items only
                        available_stream->batch = std::move(non_terminal_batch);
                        
                        // Launch inference asynchronously
                        // Launching inference
                        
                        available_stream->start_time = now;
                        available_stream->future = std::async(std::launch::async, 
                            [this, state_batch = std::move(state_batch)]() mutable {
                                // ALWAYS use SharedInferenceQueue for proper batching
                                if (GlobalInferenceQueue::isInitialized()) {
                                    try {
                                        // DEBUG: Log SharedInferenceQueue usage
                                        static std::atomic<int> queue_submit_count(0);
                                        int count = ++queue_submit_count;
                                        if (count % 100 == 0) {
                                            std::cout << "[TaskflowSearch] SharedInferenceQueue submit #" << count 
                                                      << ", batch_size=" << state_batch.size() << std::endl;
                                        }
                                        
                                        auto future = GlobalInferenceQueue::getInstance().submitBatch(std::move(state_batch));
                                        return future.get();
                                    } catch (const std::exception& e) {
                                        std::cerr << "[TaskflowSearch] SharedInferenceQueue failed: " << e.what() 
                                                  << ", falling back to direct inference" << std::endl;
                                    }
                                } else {
                                    // DEBUG: Log when queue is not initialized
                                    static bool logged_not_init = false;
                                    if (!logged_not_init) {
                                        std::cerr << "[TaskflowSearch] SharedInferenceQueue not initialized, using direct inference" << std::endl;
                                        logged_not_init = true;
                                    }
                                }
                                // Fallback to direct inference only if queue not available
                                return direct_inference_fn_(state_batch);
                            });
                        available_stream->pending = true;
                    } else {
                        // All were terminal, clear the batch
                        available_stream->batch.clear();
                    }
                    
                    last_batch_time = now;
                }
            } else {
                // All streams busy - yield briefly
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        // CRITICAL FIX: Process any remaining pending inference streams before exiting
        for (auto& stream : streams) {
            if (stream.pending && stream.future.valid()) {
                auto nn_results = stream.future.get();
                stream.pending = false;
                
                // Process final results
                int processed = 0;
                for (size_t i = 0; i < stream.batch.size() && i < nn_results.size(); ++i) {
                    EvalResult result;
                    result.node = stream.batch[i].node;
                    result.value = nn_results[i].value;
                    result.policy = nn_results[i].policy;
                    result.path = std::move(stream.batch[i].path);
                    
                    result_queue.enqueue(std::move(result));
                    processed++;
                }
                
                pending_evaluations.fetch_sub(processed);
                // Final batch processed
            }
        }
    });
    
    // Backpropagation threads - OPTIMIZED for Ryzen 9 5900X
    for (int i = 0; i < 4; ++i) {  // More backprop threads for 24-thread CPU
        threads.emplace_back([&]() {
            std::vector<EvalResult> results;
            results.reserve(32);
            
            while ((!shutdown.load() && !utils::isShutdownRequested()) || result_queue.size_approx() > 0) {
                results.clear();
                
                // Bulk dequeue for efficiency - larger batches for powerful CPU
                size_t dequeued = result_queue.try_dequeue_bulk(
                    std::back_inserter(results), 64
                );
                
                if (dequeued > 0) {
                    for (auto& result : results) {
                        // Set policy
                        if (!result.policy.empty() && result.node && result.node->isLeaf()) {
                            result.node->setPriorProbabilities(result.policy);
                        }
                        
                        // Backpropagate
                        float value = result.value;
                        for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
                            if (*it) {
                                (*it)->update(value);
                                (*it)->revertVirtualLoss(settings_.virtual_loss);
                                value = -value;
                            }
                        }
                        
                        simulations_completed.fetch_add(1);
                    }
                } else {
                    // Very short yield to reduce CPU usage
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Monitor and memory management thread
    threads.emplace_back([&]() {
        auto start_time = std::chrono::steady_clock::now();
        auto last_report = std::chrono::steady_clock::now();
        auto last_cleanup = std::chrono::steady_clock::now();
        auto last_node_pool_compact = std::chrono::steady_clock::now();
        
        while (!utils::isShutdownRequested() && simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Less frequent monitoring to reduce overhead
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start_time).count();
            
            // CRITICAL FIX: Aggressive memory cleanup every 2 seconds
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup).count() >= 2) {
                auto pressure = memory_manager.getMemoryPressure();
                if (pressure >= AggressiveMemoryManager::PressureLevel::WARNING) {
                    memory_manager.forceCleanup(pressure);
                }
                
                // Force GPU cleanup if memory pressure
                if (pressure >= AggressiveMemoryManager::PressureLevel::WARNING) {
                    #ifdef WITH_TORCH
                    if (torch::cuda::is_available()) {
                        torch::cuda::synchronize();
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
                    #endif
                    
                    // Clear game state pools
                    if (game_state_pool_enabled_) {
                        utils::GameStatePoolManager::getInstance().clearAllPools();
                    }
                    
                    // CRITICAL FIX: Clean transposition table under pressure
                    if (transposition_table_ && transposition_table_->size() > 50000) {
                        dynamic_cast<PHMapTranspositionTable*>(transposition_table_.get())->enforceMemoryLimit();
                    }
                }
                
                last_cleanup = now;
            }
            
            // CRITICAL FIX: Compact node pool periodically
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_node_pool_compact).count() >= 5) {
                if (node_pool_ && node_pool_->shouldCompact()) {
                    node_pool_->compact();
                }
                last_node_pool_compact = now;
            }
            
            if (elapsed_ms > 0) {
                int sims = simulations_completed.load();
                int batches = batches_processed.load();
                float throughput = 1000.0f * sims / elapsed_ms;
                float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
                
                // GPU utilization estimate
                float gpu_util = 0;
                if (batches > 0) {
                    float avg_inference_us = float(total_inference_time_us.load()) / batches;
                    float inference_duty_cycle = avg_inference_us / (elapsed_ms * 1000.0f / batches);
                    gpu_util = std::min(100.0f, inference_duty_cycle * 100.0f);
                }
                
                // Log performance and memory status only if verbose
                auto& progress_manager = alphazero::utils::SelfPlayProgressManager::getInstance();
                if (progress_manager.isVerboseLoggingEnabled() && 
                    std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count() >= 2) {
                    std::cout << "[TASKFLOW] Sims: " << sims << "/" << num_simulations 
                              << " | Throughput: " << std::fixed << std::setprecision(1) << throughput << " sims/s"
                              << " | Avg batch: " << std::fixed << std::setprecision(1) << avg_batch
                              << " | GPU util: " << std::fixed << std::setprecision(1) << gpu_util << "%"
                              << " | Memory: " << std::fixed << std::setprecision(1) 
                              << memory_manager.getCurrentMemoryUsageGB() << "GB"
                              << " | Pressure: " << static_cast<int>(memory_manager.getMemoryPressure())
                              << std::endl;
                    last_report = now;
                }
            }
        }
    });
    
    // Wait for completion with timeout
    auto wait_start = std::chrono::steady_clock::now();
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - wait_start).count();
        if (elapsed > 30) {
            std::cerr << "[ERROR] Search timeout after 30 seconds. Completed: " 
                      << simulations_completed.load() << "/" << num_simulations << std::endl;
            break;
        }
        
        // Only print progress every 50 simulations or every 5 seconds
        int current_sims = simulations_completed.load();
        static std::chrono::steady_clock::time_point last_print_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto time_since_print = std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count();
        
        if (current_sims >= 50 && (current_sims % 50 == 0 || time_since_print >= 5)) {
            // Progress logged
            last_print_time = now;
        }
    }
    
    // Shutdown
    shutdown.store(true);
    
    // Give threads time to process final batches
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // CRITICAL FIX: Update search statistics before cleanup
    last_stats_.total_evaluations = simulations_completed.load();
    last_stats_.total_batches_processed = batches_processed.load();
    if (batches_processed.load() > 0) {
        last_stats_.avg_batch_size = static_cast<float>(total_batch_size.load()) / batches_processed.load();
        last_stats_.avg_batch_latency = std::chrono::milliseconds(
            total_inference_time_us.load() / (1000 * batches_processed.load())
        );
    }
    
    // Final cleanup
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    #endif
    
    // Clear game state pools
    if (game_state_pool_enabled_) {
        utils::GameStatePoolManager::getInstance().clearAllPools();
    }
    
    // Force memory manager cleanup
    memory_manager.forceCleanup(AggressiveMemoryManager::PressureLevel::EMERGENCY);
    
}

} // namespace mcts
} // namespace alphazero
