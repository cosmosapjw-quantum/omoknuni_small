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
    std::cout << "🚀 OPTIMIZED LEAF PARALLELIZATION: " << num_simulations 
              << " simulations, batch_size=" << settings_.batch_size 
              << ", threads=" << settings_.num_threads << std::endl;
    
    // Ensure root is properly initialized
    if (!root) {
        std::cerr << "ERROR: Root node is null!" << std::endl;
        return;
    }
    
    // Check if we have a neural network
    if (!neural_network_) {
        std::cerr << "ERROR: Neural network is null!" << std::endl;
        return;
    }
    
    // Expand root if needed
    if (!root->isExpanded() && !root->isTerminal()) {
        std::cout << "Expanding root node..." << std::endl;
        root->expand(settings_.use_progressive_widening,
                    settings_.progressive_widening_c,
                    settings_.progressive_widening_k);
        std::cout << "Root expanded with " << root->getChildren().size() << " children" << std::endl;
    }
    
    auto search_start = std::chrono::steady_clock::now();
    AggressiveMemoryManager& memory_manager = AggressiveMemoryManager::getInstance();
    
    std::cout << "Initial memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
    
    // Atomic counters
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> shutdown(false);
    
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
    
    const int queue_capacity = 1024;
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(queue_capacity);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(queue_capacity);
    
    std::cout << "Starting threads..." << std::endl;
    std::vector<std::thread> threads;
    
    // Leaf collection threads
    for (int worker_id = 0; worker_id < settings_.num_threads; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            std::cout << "Collector " << worker_id << " started" << std::endl;
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
                
                // Queue leaf
                LeafEvalRequest request;
                request.node = current.get();
                request.state = std::move(state);
                for (const auto& node : path) {
                    request.path.push_back(node.get());
                }
                
                if (leaf_queue.enqueue(std::move(request))) {
                    local_collected++;
                    leaves_collected.fetch_add(1);
                    
                    if (local_collected % 10 == 0) {
                        std::cout << "Collector " << worker_id << ": " << local_collected << " leaves" << std::endl;
                    }
                }
            }
            
            std::cout << "Collector " << worker_id << " finished: " << local_collected << " leaves" << std::endl;
        });
    }
    
    // Inference thread - FIXED to handle NN inference properly
    threads.emplace_back([&]() {
        std::cout << "Inference thread started" << std::endl;
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
                std::cout << "Processing batch of " << batch.size() << " states..." << std::endl;
                
                try {
                    // Process terminal states first
                    std::vector<size_t> non_terminal_indices;
                    std::vector<EvalResult> terminal_results;
                    
                    for (size_t i = 0; i < batch.size(); ++i) {
                        if (batch[i].state->isTerminal()) {
                            // Handle terminal state directly
                            EvalResult result;
                            result.node = batch[i].node;
                            result.path = std::move(batch[i].path);
                            
                            auto game_result = batch[i].state->getGameResult();
                            int current_player = batch[i].state->getCurrentPlayer();
                            
                            if (game_result == core::GameResult::WIN_PLAYER1) {
                                result.value = (current_player == 1) ? 1.0f : -1.0f;
                            } else if (game_result == core::GameResult::WIN_PLAYER2) {
                                result.value = (current_player == 2) ? 1.0f : -1.0f;
                            } else {
                                result.value = 0.0f;  // Draw
                            }
                            
                            terminal_results.push_back(std::move(result));
                        } else {
                            non_terminal_indices.push_back(i);
                        }
                    }
                    
                    // Process non-terminal states with NN
                    if (!non_terminal_indices.empty()) {
                        // Prepare states for NN
                        std::vector<std::unique_ptr<core::IGameState>> state_batch;
                        state_batch.reserve(non_terminal_indices.size());
                        
                        for (size_t idx : non_terminal_indices) {
                            state_batch.push_back(std::move(batch[idx].state));
                        }
                        
                        // Neural network inference
                        auto eval_start = std::chrono::steady_clock::now();
                        auto nn_results = neural_network_->inference(state_batch);
                        auto eval_end = std::chrono::steady_clock::now();
                        
                        auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            eval_end - eval_start).count();
                        
                        std::cout << "✅ NN inference: " << state_batch.size() 
                                  << " states in " << eval_ms << "ms" << std::endl;
                        
                        // Create results
                        for (size_t i = 0; i < non_terminal_indices.size(); ++i) {
                            size_t batch_idx = non_terminal_indices[i];
                            EvalResult result;
                            result.node = batch[batch_idx].node;
                            result.value = nn_results[i].value;
                            result.policy = nn_results[i].policy;
                            result.path = std::move(batch[batch_idx].path);
                            
                            result_queue.enqueue(std::move(result));
                        }
                    }
                    
                    // Enqueue terminal results
                    for (auto& result : terminal_results) {
                        result_queue.enqueue(std::move(result));
                    }
                    
                    batches_processed.fetch_add(1);
                    total_batch_size.fetch_add(batch.size());
                    
                } catch (const std::exception& e) {
                    std::cerr << "ERROR in inference: " << e.what() << std::endl;
                }
                
                batch.clear();
                last_batch_time = now;
            } else if (batch.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        std::cout << "Inference thread finished" << std::endl;
    });
    
    // Backpropagation threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&, i]() {
            std::cout << "Backprop " << i << " started" << std::endl;
            int local_processed = 0;
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
                    // Set policy if needed
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
                    
                    if (local_processed % 50 == 0) {
                        std::cout << "Backprop " << i << ": " << simulations_completed.load() 
                                  << " simulations" << std::endl;
                    }
                }
            }
            
            std::cout << "Backprop " << i << " finished: " << local_processed << std::endl;
        });
    }
    
    // Monitor thread
    threads.emplace_back([&]() {
        auto start_time = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            
            int sims = simulations_completed.load();
            float throughput = elapsed_sec > 0 ? float(sims) / elapsed_sec : 0;
            
            std::cout << "\n📊 Progress: " << sims << "/" << num_simulations 
                      << " | Throughput: " << throughput << " sims/sec"
                      << " | Queues: " << leaf_queue.size_approx() << "/" << result_queue.size_approx()
                      << " | Memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
        }
    });
    
    std::cout << "All threads started. Waiting for completion..." << std::endl;
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Shutdown
    std::cout << "\nShutting down..." << std::endl;
    shutdown.store(true);
    
    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Final stats
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - search_start);
    
    std::cout << "\n✅ Search completed in " << duration.count() << "ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f * num_simulations / duration.count()) << " sims/sec" << std::endl;
}

} // namespace mcts
} // namespace alphazero