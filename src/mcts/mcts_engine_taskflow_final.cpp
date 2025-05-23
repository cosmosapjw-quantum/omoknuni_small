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
    std::cout << "🚀 FINAL OPTIMIZED LEAF PARALLELIZATION: " << num_simulations 
              << " simulations, batch_size=" << settings_.batch_size 
              << ", threads=" << settings_.num_threads << std::endl;
    
    if (!root) {
        std::cerr << "ERROR: Root node is null!" << std::endl;
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
    std::atomic<long> total_inference_time_ms(0);
    
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
    
    // SINGLE INFERENCE THREAD - Using direct_inference_fn_
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
                
                // Process each state individually using direct_inference_fn_
                for (auto& req : batch) {
                    EvalResult result;
                    result.node = req.node;
                    result.path = std::move(req.path);
                    
                    if (req.state->isTerminal()) {
                        // Terminal state - evaluate directly
                        auto game_result = req.state->getGameResult();
                        int current_player = req.state->getCurrentPlayer();
                        
                        if (game_result == core::GameResult::WIN_PLAYER1) {
                            result.value = (current_player == 1) ? 1.0f : -1.0f;
                        } else if (game_result == core::GameResult::WIN_PLAYER2) {
                            result.value = (current_player == 2) ? 1.0f : -1.0f;
                        } else {
                            result.value = 0.0f;
                        }
                    } else {
                        // Non-terminal - use neural network
                        if (direct_inference_fn_) {
                            // Use direct inference function
                            auto [value, policy] = direct_inference_fn_(*req.state);
                            result.value = value;
                            result.policy = std::move(policy);
                        } else if (neural_network_) {
                            // Use neural network directly
                            std::vector<std::unique_ptr<core::IGameState>> single_state;
                            single_state.push_back(std::move(req.state));
                            auto nn_results = neural_network_->inference(single_state);
                            if (!nn_results.empty()) {
                                result.value = nn_results[0].value;
                                result.policy = nn_results[0].policy;
                            }
                        } else {
                            // Fallback - random evaluation
                            result.value = (std::rand() % 201 - 100) / 100.0f;
                            int action_space = req.node->getState().getActionSpaceSize();
                            result.policy.resize(action_space, 1.0f / action_space);
                        }
                    }
                    
                    result_queue.enqueue(std::move(result));
                }
                
                auto eval_end = std::chrono::steady_clock::now();
                auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    eval_end - eval_start).count();
                total_inference_time_ms.fetch_add(eval_ms);
                
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                
                if (batches_processed.load() % 10 == 0) {
                    std::cout << "✅ Batch " << batches_processed.load() 
                              << ": " << batch.size() << " states in " 
                              << eval_ms << "ms" << std::endl;
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
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            
            if (elapsed_sec > 0) {
                int sims = simulations_completed.load();
                int batches = batches_processed.load();
                float throughput = float(sims) / elapsed_sec;
                float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
                
                // GPU utilization estimate
                float gpu_util = 0;
                if (batches > 0 && elapsed_sec > 0) {
                    float avg_inference_ms = float(total_inference_time_ms.load()) / batches;
                    float batches_per_sec = float(batches) / elapsed_sec;
                    gpu_util = std::min(100.0f, (avg_inference_ms * batches_per_sec) / 10.0f);
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
    }
}

} // namespace mcts
} // namespace alphazero