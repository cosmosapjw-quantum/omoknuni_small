// src/mcts/mcts_engine_continuous_pipeline.cpp
// Continuous pipeline implementation with zero GPU idle time

#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include <moodycamel/concurrentqueue.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>

namespace alphazero {
namespace mcts {

void MCTSEngine::executeTaskflowSearch(MCTSNode* root, int num_simulations) {
    auto search_start = std::chrono::steady_clock::now();
    std::cout << "🚀 CONTINUOUS PIPELINE: " << num_simulations << " simulations, "
              << "batch_size=" << settings_.batch_size << ", threads=" << settings_.num_threads << std::endl;
    
    auto& memory_manager = AggressiveMemoryManager::getInstance();
    std::cout << "Initial memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
    
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
    
    // Request structures
    struct LeafEvalRequest {
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
    
    // Double buffering for continuous GPU utilization
    struct BatchBuffer {
        std::vector<LeafEvalRequest> items;
        std::vector<std::unique_ptr<core::IGameState>> states;
        std::atomic<bool> ready{false};
        std::atomic<bool> processing{false};
    };
    
    BatchBuffer buffers[2];  // Double buffering
    std::atomic<int> write_buffer{0};
    
    // Shared queues
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(4096);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(4096);
    
    // Statistics
    std::atomic<int> simulations_started{0};
    std::atomic<int> simulations_completed{0};
    std::atomic<int> batches_processed{0};
    std::atomic<int> total_batch_size{0};
    std::atomic<bool> shutdown{false};
    std::atomic<int64_t> total_gpu_time_us{0};
    std::atomic<int64_t> total_collect_time_us{0};
    
    std::vector<std::thread> threads;
    threads.reserve(settings_.num_threads + 4);
    
    // Selection/Expansion threads
    for (int i = 0; i < settings_.num_threads; ++i) {
        threads.emplace_back([&, i]() {
            std::mt19937 gen(std::random_device{}() + i);
            
            while (simulations_started.load() < num_simulations && !shutdown.load()) {
                int sim_id = simulations_started.fetch_add(1);
                if (sim_id >= num_simulations) break;
                
                // Selection phase
                std::vector<MCTSNode*> path;
                MCTSNode* current = root;
                auto state = root->getState().clone();
                
                // Apply virtual loss immediately
                current->applyVirtualLoss(settings_.virtual_loss);
                path.push_back(current);
                
                // Tree traversal with virtual loss
                while (!current->isLeaf() && !current->isTerminal()) {
                    auto child = current->selectBestChildUCB(settings_.exploration_constant, gen);
                    if (!child) break;
                    
                    child->applyVirtualLoss(settings_.virtual_loss);
                    path.push_back(child.get());
                    state->makeMove(child->getAction());
                    current = child.get();
                }
                
                if (!current || current->isTerminal()) {
                    // Terminal node - immediate backprop
                    float value = 0.0f;
                    if (current && state->isTerminal()) {
                        // Get reward based on game result and current player
                        auto result = state->getGameResult();
                        int player = state->getCurrentPlayer();
                        if (result == core::GameResult::WIN_PLAYER1) {
                            value = (player == 1) ? 1.0f : -1.0f;
                        } else if (result == core::GameResult::WIN_PLAYER2) {
                            value = (player == 2) ? 1.0f : -1.0f;
                        } else {
                            value = 0.0f; // Draw or ongoing
                        }
                    }
                    
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        if (*it) {
                            (*it)->update(value);
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                            value = -value;
                        }
                    }
                    
                    simulations_completed.fetch_add(1);
                    continue;
                }
                
                // Expansion if needed
                if (current->getVisitCount() > 0 && !current->isExpanded()) {
                    current->expand(settings_.use_progressive_widening,
                                  settings_.progressive_widening_c,
                                  settings_.progressive_widening_k);
                }
                
                // Queue leaf for evaluation
                LeafEvalRequest request;
                request.node = current;
                request.path = std::move(path);
                request.state = std::move(state);
                
                leaf_queue.enqueue(std::move(request));
            }
        });
    }
    
    // Batch collection thread - fills buffers continuously
    threads.emplace_back([&]() {
        std::vector<LeafEvalRequest> temp_buffer;
        temp_buffer.reserve(settings_.batch_size);
        
        while (!shutdown.load()) {
            auto collect_start = std::chrono::steady_clock::now();
            int wb = write_buffer.load();
            auto& buffer = buffers[wb];
            
            // Skip if buffer is already being processed
            if (buffer.processing.load()) {
                std::this_thread::yield();
                continue;
            }
            
            // Clear buffer
            buffer.items.clear();
            buffer.states.clear();
            buffer.ready = false;
            
            // Collect items into buffer
            temp_buffer.clear();
            size_t collected = leaf_queue.try_dequeue_bulk(
                std::back_inserter(temp_buffer),
                settings_.batch_size
            );
            
            auto deadline = std::chrono::steady_clock::now() + 
                           std::chrono::milliseconds(settings_.batch_timeout.count() / 2);
            
            while (collected < settings_.batch_size && 
                   std::chrono::steady_clock::now() < deadline) {
                
                size_t additional = leaf_queue.try_dequeue_bulk(
                    std::back_inserter(temp_buffer),
                    settings_.batch_size - collected
                );
                
                collected += additional;
                
                if (collected < settings_.batch_size && additional == 0) {
                    std::this_thread::yield();  // No sleep, just yield
                }
            }
            
            if (collected > 0) {
                // Move items to buffer
                buffer.items = std::move(temp_buffer);
                
                // Prepare states for GPU
                buffer.states.reserve(collected);
                for (auto& item : buffer.items) {
                    buffer.states.push_back(std::move(item.state));
                }
                
                // Mark buffer ready and swap
                buffer.ready = true;
                write_buffer.store(1 - wb);
                
                auto collect_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - collect_start).count();
                total_collect_time_us.fetch_add(collect_time);
            }
        }
    });
    
    // GPU inference thread - processes buffers continuously
    threads.emplace_back([&]() {
        while (!shutdown.load()) {
            // Check both buffers for ready data
            for (int b = 0; b < 2; ++b) {
                auto& buffer = buffers[b];
                
                if (buffer.ready.load() && !buffer.processing.load()) {
                    buffer.processing = true;
                    
                    auto gpu_start = std::chrono::steady_clock::now();
                    
                    // Run GPU inference
                    auto nn_results = direct_inference_fn_(buffer.states);
                    
                    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - gpu_start).count();
                    total_gpu_time_us.fetch_add(gpu_time);
                    
                    // Queue results
                    for (size_t i = 0; i < buffer.items.size() && i < nn_results.size(); ++i) {
                        EvalResult result;
                        result.node = buffer.items[i].node;
                        result.value = nn_results[i].value;
                        result.policy = nn_results[i].policy;
                        result.path = std::move(buffer.items[i].path);
                        
                        result_queue.enqueue(std::move(result));
                    }
                    
                    batches_processed.fetch_add(1);
                    total_batch_size.fetch_add(buffer.items.size());
                    
                    if (batches_processed.load() % 5 == 0) {
                        std::cout << "✅ Batch " << batches_processed.load() 
                                  << ": " << buffer.items.size() << " states in " 
                                  << (gpu_time / 1000.0) << "ms" << std::endl;
                    }
                    
                    // Clear buffer
                    buffer.items.clear();
                    buffer.states.clear();
                    buffer.ready = false;
                    buffer.processing = false;
                }
            }
            
            // Yield if no work
            std::this_thread::yield();
        }
    });
    
    // Backpropagation threads
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
                    std::this_thread::yield();  // No sleep, just yield
                }
            }
        });
    }
    
    // Monitor thread
    threads.emplace_back([&]() {
        auto start_time = std::chrono::steady_clock::now();
        
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
                
                // GPU utilization based on actual GPU time
                float gpu_util = 0;
                if (batches > 0 && elapsed_ms > 0) {
                    float total_gpu_ms = total_gpu_time_us.load() / 1000.0f;
                    gpu_util = std::min(100.0f, (total_gpu_ms / elapsed_ms) * 100.0f);
                }
                
                std::cout << "\n📊 Progress: " << sims << "/" << num_simulations 
                          << " (" << (100.0f * sims / num_simulations) << "%)"
                          << " | Throughput: " << throughput << " sims/sec";
                
                if (throughput >= 200) {
                    std::cout << " 🚀";
                } else if (throughput >= 150) {
                    std::cout << " ✅";
                } else {
                    std::cout << " ⚠️";
                }
                
                std::cout << "\n   Batches: " << batches << " (avg " << avg_batch << " states)"
                          << " | GPU Util: " << gpu_util << "%"
                          << " | Memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB"
                          << " | Leaf Queue: " << leaf_queue.size_approx()
                          << " | Result Queue: " << result_queue.size_approx()
                          << std::endl;
            }
        }
    });
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    
    std::cout << "\n✅ Search completed in " << duration.count() << "ms" << std::endl;
    std::cout << "  Final throughput: " << final_throughput << " sims/sec ";
    
    if (final_throughput >= 200) {
        std::cout << "🚀 EXCELLENT PERFORMANCE!" << std::endl;
    } else if (final_throughput >= 150) {
        std::cout << "✅ TARGET ACHIEVED!" << std::endl;
    } else {
        std::cout << "⚠️ Below target" << std::endl;
    }
}

} // namespace mcts
} // namespace alphazero