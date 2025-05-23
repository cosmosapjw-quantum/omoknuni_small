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
    std::atomic<bool> collection_active(true);
    std::atomic<bool> inference_active(true);
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
    
    const int queue_capacity = 1024;  // Larger capacity
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
            
            while (!shutdown.load()) {
                // Check if we need more simulations
                if (simulations_completed.load() >= num_simulations) {
                    break;
                }
                
                // Tree traversal
                std::shared_ptr<MCTSNode> current(root, [](MCTSNode*){});  // Non-owning
                std::vector<std::shared_ptr<MCTSNode>> path;
                path.reserve(100);
                
                // Clone state
                auto state = root->getState().clone();
                if (!state) {
                    std::cerr << "Failed to clone root state!" << std::endl;
                    continue;
                }
                
                // Selection phase
                while (current && !current->isLeaf() && !state->isTerminal()) {
                    path.push_back(current);
                    current->applyVirtualLoss(settings_.virtual_loss);
                    
                    auto next = current->selectBestChildUCB(settings_.exploration_constant, thread_rng);
                    if (!next) {
                        // No valid child - revert and retry
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
                
                // Expansion phase
                if (!state->isTerminal() && current->getVisitCount() > 0 && !current->isExpanded()) {
                    current->expand(settings_.use_progressive_widening,
                                  settings_.progressive_widening_c,
                                  settings_.progressive_widening_k);
                    
                    auto& children = current->getChildren();
                    if (!children.empty()) {
                        current->applyVirtualLoss(settings_.virtual_loss);
                        current = children[0];  // Select first child
                        path.push_back(current);
                        state = current->getState().clone();
                    }
                }
                
                // Create request
                LeafEvalRequest request;
                request.node = current.get();
                request.state = std::move(state);
                request.path.reserve(path.size());
                for (const auto& node : path) {
                    request.path.push_back(node.get());
                }
                
                // Enqueue
                if (leaf_queue.enqueue(std::move(request))) {
                    local_collected++;
                    leaves_collected.fetch_add(1);
                    
                    if (local_collected % 10 == 0) {
                        std::cout << "Collector " << worker_id << ": " << local_collected << " leaves" << std::endl;
                    }
                } else {
                    // Queue full - revert
                    for (auto node : request.path) {
                        node->revertVirtualLoss(settings_.virtual_loss);
                    }
                }
                
                // Yield to prevent spinning
                if (local_collected % 5 == 0) {
                    std::this_thread::yield();
                }
            }
            
            std::cout << "Collector " << worker_id << " finished: " << local_collected << " leaves" << std::endl;
        });
    }
    
    // Inference thread
    threads.emplace_back([&]() {
        std::cout << "Inference thread started" << std::endl;
        std::vector<LeafEvalRequest> batch;
        batch.reserve(settings_.batch_size);
        auto last_batch_time = std::chrono::steady_clock::now();
        int total_processed = 0;
        
        while (!shutdown.load() || leaf_queue.size_approx() > 0) {
            // Collect batch
            LeafEvalRequest request;
            while (batch.size() < static_cast<size_t>(settings_.batch_size) && 
                   leaf_queue.try_dequeue(request)) {
                batch.push_back(std::move(request));
            }
            
            // Check timing
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time).count();
            
            bool should_process = !batch.empty() && (
                batch.size() >= static_cast<size_t>(settings_.batch_size) ||
                elapsed_ms >= settings_.batch_timeout.count() ||
                (shutdown.load() && leaf_queue.size_approx() == 0)
            );
            
            if (should_process) {
                // Prepare states
                std::vector<std::unique_ptr<core::IGameState>> state_batch;
                state_batch.reserve(batch.size());
                for (auto& req : batch) {
                    state_batch.push_back(std::move(req.state));
                }
                
                // Neural network inference
                auto eval_start = std::chrono::steady_clock::now();
                
                std::cout << "Processing batch of " << batch.size() << " states..." << std::endl;
                auto results = neural_network_->inference(state_batch);
                
                auto eval_end = std::chrono::steady_clock::now();
                auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    eval_end - eval_start).count();
                
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                total_processed += batch.size();
                
                std::cout << "✅ Batch " << batches_processed.load() 
                          << ": " << batch.size() << " states in " 
                          << eval_ms << "ms (" << (batch.size() * 1000.0 / (eval_ms + 1)) 
                          << " states/sec)" << std::endl;
                
                // Queue results
                for (size_t i = 0; i < batch.size(); ++i) {
                    EvalResult result;
                    result.node = batch[i].node;
                    result.value = results[i].value;
                    result.policy = results[i].policy;
                    result.path = std::move(batch[i].path);
                    
                    if (!result_queue.enqueue(std::move(result))) {
                        std::cerr << "Failed to enqueue result!" << std::endl;
                    }
                }
                
                batch.clear();
                last_batch_time = now;
            } else if (batch.empty() && leaf_queue.size_approx() == 0) {
                // No work - brief sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        std::cout << "Inference thread finished: " << total_processed << " states processed" << std::endl;
    });
    
    // Backpropagation threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&, i]() {
            std::cout << "Backprop " << i << " started" << std::endl;
            int local_processed = 0;
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
                    // Set policy if leaf
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
                                  << "/" << num_simulations << " simulations" << std::endl;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            std::cout << "Backprop " << i << " finished: " << local_processed << " results" << std::endl;
        });
    }
    
    // Progress monitor
    threads.emplace_back([&]() {
        std::cout << "Monitor thread started" << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            
            int sims = simulations_completed.load();
            int collected = leaves_collected.load();
            int batches = batches_processed.load();
            float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
            float throughput = elapsed_sec > 0 ? float(sims) / elapsed_sec : 0;
            
            std::cout << "\n📊 Progress: " << sims << "/" << num_simulations 
                      << " | Collected: " << collected 
                      << " | Batches: " << batches << " (avg " << avg_batch << ")"
                      << " | Queues: " << leaf_queue.size_approx() << "/" << result_queue.size_approx()
                      << " | Throughput: " << throughput << " sims/sec"
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
    
    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Final stats
    auto search_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - search_start);
    
    std::cout << "\n✅ Search completed:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "  Simulations: " << simulations_completed.load() << std::endl;
    std::cout << "  Throughput: " << (duration.count() > 0 ? 
        1000.0f * simulations_completed.load() / duration.count() : 0) << " sims/sec" << std::endl;
}

} // namespace mcts
} // namespace alphazero