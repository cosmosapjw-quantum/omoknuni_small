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
    std::cout << "🚀 CORRECT LEAF PARALLELIZATION: " << num_simulations 
              << " simulations, batch_size=" << settings_.batch_size 
              << ", threads=" << settings_.num_threads << std::endl;
    
    if (!root) {
        std::cerr << "ERROR: Root node is null!" << std::endl;
        return;
    }
    
    // The neural network should be accessed through direct_inference_fn_ which is set in constructor
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
    AggressiveMemoryManager& memory_manager = AggressiveMemoryManager::getInstance();
    
    std::cout << "Initial memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
    
    // Performance metrics
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> shutdown(false);
    std::atomic<long> total_inference_time_us(0);
    
    // Lock-free queues
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
    
    const int queue_capacity = 2048;
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(queue_capacity);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(queue_capacity);
    
    std::vector<std::thread> threads;
    
    // Leaf collection threads
    for (int worker_id = 0; worker_id < settings_.num_threads; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            std::mt19937 thread_rng(std::random_device{}() + worker_id);
            int local_collected = 0;
            
            while (!shutdown.load() && simulations_completed.load() < num_simulations) {
                // Tree traversal
                std::shared_ptr<MCTSNode> current(root, [](MCTSNode*){});
                std::vector<std::shared_ptr<MCTSNode>> path;
                
                auto state = root->getState().clone();
                
                // Selection
                while (current && !current->isLeaf() && !state->isTerminal()) {
                    path.push_back(current);
                    current->applyVirtualLoss(settings_.virtual_loss);
                    
                    auto next = current->selectBestChildUCB(settings_.exploration_constant, thread_rng);
                    if (!next) {
                        for (auto it = path.rbegin(); it != path.rend(); ++it) {
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                        }
                        break;
                    }
                    
                    current = next;
                    state->makeMove(current->getAction());
                }
                
                if (!current) continue;
                
                path.push_back(current);
                
                // Expansion
                if (!state->isTerminal() && current->getVisitCount() > 0 && !current->isExpanded()) {
                    current->expand(settings_.use_progressive_widening,
                                  settings_.progressive_widening_c,
                                  settings_.progressive_widening_k);
                    
                    auto& children = current->getChildren();
                    if (!children.empty()) {
                        current->applyVirtualLoss(settings_.virtual_loss);
                        current = children[0];
                        path.push_back(current);
                        state = current->getState().clone();
                    }
                }
                
                // Create request
                LeafEvalRequest request;
                request.node = current.get();
                request.state = std::move(state);
                for (const auto& node : path) {
                    request.path.push_back(node.get());
                }
                
                if (leaf_queue.enqueue(std::move(request))) {
                    leaves_collected.fetch_add(1);
                    local_collected++;
                }
            }
        });
    }
    
    // SINGLE INFERENCE THREAD - Using direct_inference_fn_ for batch processing
    threads.emplace_back([&]() {
        std::vector<LeafEvalRequest> batch;
        batch.reserve(settings_.batch_size);
        auto last_batch_time = std::chrono::steady_clock::now();
        
        while (!shutdown.load() || leaf_queue.size_approx() > 0) {
            // Collect batch
            LeafEvalRequest request;
            while (batch.size() < static_cast<size_t>(settings_.batch_size) && 
                   leaf_queue.try_dequeue(request)) {
                batch.push_back(std::move(request));
            }
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time).count();
            
            bool should_process = !batch.empty() && (
                batch.size() >= static_cast<size_t>(settings_.batch_size) ||
                elapsed_ms >= settings_.batch_timeout.count() ||
                (shutdown.load() && leaf_queue.size_approx() == 0)
            );
            
            if (should_process) {
                auto eval_start = std::chrono::steady_clock::now();
                
                // Separate terminal and non-terminal states
                std::vector<LeafEvalRequest> non_terminal_batch;
                std::vector<EvalResult> terminal_results;
                
                for (auto& req : batch) {
                    if (req.state->isTerminal()) {
                        // Handle terminal directly
                        EvalResult result;
                        result.node = req.node;
                        result.path = std::move(req.path);
                        
                        auto game_result = req.state->getGameResult();
                        int current_player = req.state->getCurrentPlayer();
                        
                        if (game_result == core::GameResult::WIN_PLAYER1) {
                            result.value = (current_player == 1) ? 1.0f : -1.0f;
                        } else if (game_result == core::GameResult::WIN_PLAYER2) {
                            result.value = (current_player == 2) ? 1.0f : -1.0f;
                        } else {
                            result.value = 0.0f;
                        }
                        
                        terminal_results.push_back(std::move(result));
                    } else {
                        non_terminal_batch.push_back(std::move(req));
                    }
                }
                
                // Process non-terminal states with neural network
                if (!non_terminal_batch.empty()) {
                    // Prepare states for batch inference
                    std::vector<std::unique_ptr<core::IGameState>> state_batch;
                    state_batch.reserve(non_terminal_batch.size());
                    
                    for (auto& req : non_terminal_batch) {
                        state_batch.push_back(std::move(req.state));
                    }
                    
                    // Call the inference function
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
                
                if (batches_processed.load() % 5 == 0) {
                    std::cout << "✅ Batch " << batches_processed.load() 
                              << ": " << batch.size() << " states in " 
                              << (eval_us / 1000.0) << "ms" << std::endl;
                }
                
                batch.clear();
                last_batch_time = now;
            } else if (batch.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });
    
    // Backpropagation threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&]() {
            int local_processed = 0;
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
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
                    local_processed++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
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
                
                std::cout << "\n📊 Progress: " << sims << "/" << num_simulations 
                          << " (" << (100.0f * sims / num_simulations) << "%)"
                          << " | Throughput: " << throughput << " sims/sec";
                
                if (throughput >= 70) {
                    std::cout << " ✅";
                } else if (throughput >= 50) {
                    std::cout << " ⚠️";
                } else {
                    std::cout << " ❌";
                }
                
                std::cout << "\n   Batches: " << batches << " (avg " << avg_batch << " states)"
                          << " | Est. GPU: " << gpu_util << "%"
                          << " | Memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB"
                          << " | Leaf Queue: " << leaf_queue.size_approx()
                          << " | Result Queue: " << result_queue.size_approx()
                          << std::endl;
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
    
    std::cout << "\n✅ Search completed in " << duration.count() << "ms" << std::endl;
    std::cout << "  Final throughput: " << final_throughput << " sims/sec ";
    
    if (final_throughput >= 70) {
        std::cout << "🎉 TARGET ACHIEVED!" << std::endl;
    } else {
        std::cout << "❌ Below target (70+ sims/sec)" << std::endl;
        std::cout << "  Consider optimizing:" << std::endl;
        std::cout << "  - Reduce batch timeout (current: " << settings_.batch_timeout.count() << "ms)" << std::endl;
        std::cout << "  - Reduce virtual loss (current: " << settings_.virtual_loss << ")" << std::endl;
        std::cout << "  - Check GPU load from other processes" << std::endl;
    }
}

} // namespace mcts
} // namespace alphazero