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
    
    // CRITICAL FIX: Initialize aggressive memory manager with lower thresholds
    auto& memory_manager = AggressiveMemoryManager::getInstance();
    AggressiveMemoryManager::Config config;
    config.warning_threshold_gb = 8.0;   // Much lower threshold
    config.critical_threshold_gb = 12.0; // Aggressive cleanup at 12GB
    config.emergency_threshold_gb = 16.0; // Emergency at 16GB
    memory_manager.setConfig(config);
    
    // Performance metrics
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> shutdown(false);
    std::atomic<long> total_inference_time_us(0);
    
    // Lock-free queues with larger capacity
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
    
    const int queue_capacity = 4096;
    moodycamel::ConcurrentQueue<BatchItem> leaf_queue(queue_capacity);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(queue_capacity);
    
    std::vector<std::thread> threads;
    
    // Leaf collection threads
    for (int worker_id = 0; worker_id < settings_.num_threads; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            std::mt19937 thread_rng(std::random_device{}() + worker_id);
            int local_collected = 0;
            int cleanup_counter = 0;
            
            while (!shutdown.load() && !utils::isShutdownRequested() && simulations_completed.load() < num_simulations) {
                // CRITICAL FIX: Periodic memory cleanup every 50 iterations
                if (++cleanup_counter % 50 == 0) {
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
                
                auto state = root->getState().clone();
                
                // Apply virtual loss to root
                current->applyVirtualLoss(settings_.virtual_loss);
                path.push_back(current);
                
                // Selection
                while (current && !current->isLeaf() && !state->isTerminal()) {
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
                
                // Add to evaluation queue
                BatchItem item;
                item.node = current;
                item.state = std::move(state);
                item.path = std::move(path);
                
                leaf_queue.enqueue(std::move(item));
                leaves_collected.fetch_add(1);
                local_collected++;
            }
        });
    }
    
    // Batch processing thread - OPTIMIZED with double buffering
    threads.emplace_back([&]() {
        // CRITICAL FIX: Double buffering for CPU/GPU pipelining
        std::vector<BatchItem> batch_a, batch_b;
        batch_a.reserve(settings_.batch_size);
        batch_b.reserve(settings_.batch_size);
        
        std::vector<BatchItem>* current_batch = &batch_a;
        std::vector<BatchItem>* processing_batch = &batch_b;
        
        std::future<std::vector<NetworkOutput>> inference_future;
        bool inference_pending = false;
        
        auto last_batch_time = std::chrono::steady_clock::now();
        const auto min_batch_wait = std::chrono::milliseconds(2);  // Reduced wait
        const auto max_batch_wait = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        // CRITICAL FIX: Memory-aware batch sizing
        auto get_adaptive_batch_size = [&]() -> size_t {
            size_t base_size = settings_.batch_size;
            auto level = memory_manager.getMemoryPressure();
            switch (level) {
                case AggressiveMemoryManager::PressureLevel::WARNING:
                    return base_size * 3 / 4;
                case AggressiveMemoryManager::PressureLevel::CRITICAL:
                    return base_size / 2;
                case AggressiveMemoryManager::PressureLevel::EMERGENCY:
                    return std::max(size_t(32), base_size / 4);
                default:
                    return base_size;
            }
        };
        
        while ((!shutdown.load() && !utils::isShutdownRequested()) || leaf_queue.size_approx() > 0) {
            current_batch->clear();
            
            // Try to collect a batch with adaptive sizing
            size_t target_batch = get_adaptive_batch_size();
            size_t collected = leaf_queue.try_dequeue_bulk(
                std::back_inserter(*current_batch), 
                target_batch
            );
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time);
            
            // Dynamic batching logic
            bool should_process = false;
            
            if (collected >= target_batch) {
                // Full batch - process immediately
                should_process = true;
            } else if (collected > 0) {
                // Partial batch - use adaptive timeout
                if (collected >= target_batch * 0.75 && elapsed >= min_batch_wait) {
                    // Good batch size and minimum time elapsed
                    should_process = true;
                } else if (elapsed >= max_batch_wait) {
                    // Maximum timeout reached
                    should_process = true;
                } else {
                    // Try to collect more items with very short wait
                    std::this_thread::yield();
                    size_t additional = leaf_queue.try_dequeue_bulk(
                        std::back_inserter(*current_batch), 
                        target_batch - collected
                    );
                    collected += additional;
                    
                    // Re-evaluate
                    if (collected >= target_batch * 0.5 || 
                        std::chrono::steady_clock::now() - last_batch_time >= max_batch_wait) {
                        should_process = true;
                    }
                }
            } else if (elapsed < std::chrono::milliseconds(1)) {
                // No items and very recent batch - yield to avoid busy wait
                std::this_thread::yield();
                continue;
            }
            
            // CRITICAL FIX: Process previous inference results while collecting new batch
            if (inference_pending && inference_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                auto nn_results = inference_future.get();
                inference_pending = false;
                
                // Process results from previous batch
                for (size_t i = 0; i < processing_batch->size() && i < nn_results.size(); ++i) {
                    EvalResult result;
                    result.node = (*processing_batch)[i].node;
                    result.value = nn_results[i].value;
                    result.policy = nn_results[i].policy;
                    result.path = std::move((*processing_batch)[i].path);
                    
                    result_queue.enqueue(std::move(result));
                }
                
                processing_batch->clear();
            }
            
            if (should_process && !current_batch->empty()) {
                auto eval_start = std::chrono::steady_clock::now();
                
                // Separate terminal and non-terminal
                std::vector<BatchItem> non_terminal_batch;
                std::vector<EvalResult> terminal_results;
                
                for (auto& item : *current_batch) {
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
                
                // Process non-terminal states
                if (!non_terminal_batch.empty()) {
                    std::vector<std::unique_ptr<core::IGameState>> state_batch;
                    state_batch.reserve(non_terminal_batch.size());
                    
                    for (auto& item : non_terminal_batch) {
                        state_batch.push_back(std::move(item.state));
                    }
                    
                    // CRITICAL FIX: Double buffering - swap buffers and launch inference
                    std::swap(current_batch, processing_batch);
                    
                    // Wait for previous inference if still running
                    if (inference_pending && inference_future.valid()) {
                        auto nn_results = inference_future.get();
                        inference_pending = false;
                        
                        // This shouldn't happen with proper pipelining, but handle it
                        for (size_t i = 0; i < processing_batch->size() && i < nn_results.size(); ++i) {
                            EvalResult result;
                            result.node = (*processing_batch)[i].node;
                            result.value = nn_results[i].value;
                            result.policy = nn_results[i].policy;
                            result.path = std::move((*processing_batch)[i].path);
                            
                            result_queue.enqueue(std::move(result));
                        }
                    }
                    
                    // Launch new inference asynchronously
                    inference_future = std::async(std::launch::async, [this, state_batch = std::move(state_batch)]() {
                        // CRITICAL FIX: Add memory tracking to inference
                        MemoryTracker tracker("NN_Inference", state_batch.size() * sizeof(float) * 1024);
                        return direct_inference_fn_(state_batch);
                    });
                    inference_pending = true;
                }
                
                // Enqueue terminal results
                for (auto& result : terminal_results) {
                    result_queue.enqueue(std::move(result));
                }
                
                auto eval_end = std::chrono::steady_clock::now();
                auto eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    eval_end - eval_start).count();
                total_inference_time_us.fetch_add(eval_us);
                
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(current_batch->size());
                
                
                last_batch_time = now;
            }
        }
    });
    
    // Backpropagation threads - OPTIMIZED
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&]() {
            std::vector<EvalResult> results;
            results.reserve(32);
            
            while ((!shutdown.load() && !utils::isShutdownRequested()) || result_queue.size_approx() > 0) {
                results.clear();
                
                // Bulk dequeue for efficiency
                size_t dequeued = result_queue.try_dequeue_bulk(
                    std::back_inserter(results), 32
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
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // More frequent monitoring
            
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
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
    
    // CRITICAL FIX: Final aggressive cleanup
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
    
    // Force memory manager cleanup
    memory_manager.forceCleanup();
    
    // Final stats
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - search_start);
    
    float final_throughput = duration.count() > 0 ? 
        1000.0f * num_simulations / duration.count() : 0;
    
    // Only print final stats if verbose logging is enabled
    auto& progress_manager = alphazero::utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        std::cout << "[TASKFLOW] Search completed in " << duration.count() << "ms"
                  << " | Throughput: " << std::fixed << std::setprecision(1) << final_throughput << " sims/s"
                  << " | Final memory: " << std::fixed << std::setprecision(1) 
                  << memory_manager.getCurrentMemoryUsageGB() << "GB"
                  << std::endl;
    }
}

} // namespace mcts
} // namespace alphazero