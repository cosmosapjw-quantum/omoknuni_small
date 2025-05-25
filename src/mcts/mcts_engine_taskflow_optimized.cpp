#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <moodycamel/concurrentqueue.h>

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
    
    // Defer memory manager access to avoid contention during initialization
    // The memory manager will be accessed later in the monitoring thread
    
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
            
            while (!shutdown.load() && simulations_completed.load() < num_simulations) {
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
                    int current_player = state->getCurrentPlayer();
                    
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
    
    // Batch processing thread - OPTIMIZED
    threads.emplace_back([&]() {
        std::vector<BatchItem> batch;
        batch.reserve(settings_.batch_size);
        
        auto last_batch_time = std::chrono::steady_clock::now();
        const auto min_batch_wait = std::chrono::milliseconds(5);  // Minimum wait
        const auto max_batch_wait = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        while (!shutdown.load() || leaf_queue.size_approx() > 0) {
            batch.clear();
            
            // Try to collect a full batch quickly
            size_t collected = leaf_queue.try_dequeue_bulk(
                std::back_inserter(batch), 
                settings_.batch_size
            );
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time);
            
            // Dynamic batching logic
            bool should_process = false;
            
            if (collected >= settings_.batch_size) {
                // Full batch - process immediately
                should_process = true;
            } else if (collected > 0) {
                // Partial batch - use adaptive timeout
                if (collected >= settings_.batch_size * 0.75 && elapsed >= min_batch_wait) {
                    // Good batch size and minimum time elapsed
                    should_process = true;
                } else if (elapsed >= max_batch_wait) {
                    // Maximum timeout reached
                    should_process = true;
                } else {
                    // Try to collect more items with very short wait
                    std::this_thread::yield();
                    size_t additional = leaf_queue.try_dequeue_bulk(
                        std::back_inserter(batch), 
                        settings_.batch_size - collected
                    );
                    collected += additional;
                    
                    // Re-evaluate
                    if (collected >= settings_.batch_size * 0.5 || 
                        std::chrono::steady_clock::now() - last_batch_time >= max_batch_wait) {
                        should_process = true;
                    }
                }
            } else if (elapsed < std::chrono::milliseconds(1)) {
                // No items and very recent batch - yield to avoid busy wait
                std::this_thread::yield();
                continue;
            }
            
            if (should_process && !batch.empty()) {
                auto eval_start = std::chrono::steady_clock::now();
                
                // Separate terminal and non-terminal
                std::vector<BatchItem> non_terminal_batch;
                std::vector<EvalResult> terminal_results;
                
                for (auto& item : batch) {
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
                    
                    // Neural network inference
                    auto nn_results = direct_inference_fn_(state_batch);
                    
                    // Create results
                    for (size_t i = 0; i < non_terminal_batch.size() && i < nn_results.size(); ++i) {
                        EvalResult result;
                        result.node = non_terminal_batch[i].node;
                        result.value = nn_results[i].value;
                        result.policy = nn_results[i].policy;
                        result.path = std::move(non_terminal_batch[i].path);
                        
                        result_queue.enqueue(std::move(result));
                    }
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
                total_batch_size.fetch_add(batch.size());
                
                
                last_batch_time = now;
            }
        }
    });
    
    // Backpropagation threads - OPTIMIZED
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&]() {
            std::vector<EvalResult> results;
            results.reserve(32);
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
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
    
    // Monitor thread
    threads.emplace_back([&]() {
        auto start_time = std::chrono::steady_clock::now();
        auto last_report = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start_time).count();
            
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
                
            }
            
            last_report = now;
        }
    });
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Shutdown
    shutdown.store(true);
    
    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Final stats
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - search_start);
    
    float final_throughput = duration.count() > 0 ? 
        1000.0f * num_simulations / duration.count() : 0;
    
}

} // namespace mcts
} // namespace alphazero