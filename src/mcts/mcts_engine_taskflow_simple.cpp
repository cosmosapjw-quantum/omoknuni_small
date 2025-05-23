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
    
    auto search_start = std::chrono::steady_clock::now();
    AggressiveMemoryManager& memory_manager = AggressiveMemoryManager::getInstance();
    
    // Print initial memory state
    std::cout << "Initial memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
    
    // Atomic counters
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> collection_active(true);
    std::atomic<bool> inference_active(true);
    std::atomic<bool> shutdown(false);
    
    // Performance tracking
    std::atomic<int> gpu_inferences(0);
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
    
    const int queue_capacity = settings_.num_threads * 32;
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(queue_capacity);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(queue_capacity);
    
    // Create threads directly (simpler than taskflow)
    std::vector<std::thread> threads;
    
    // Leaf collection threads
    for (int worker_id = 0; worker_id < settings_.num_threads; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            std::mt19937 thread_rng(std::random_device{}() + worker_id);
            int local_collected = 0;
            
            while (!shutdown.load() && simulations_completed.load() < num_simulations) {
                // Tree traversal - use shared_ptr for consistency
                std::shared_ptr<MCTSNode> current(root, [](MCTSNode*){});  // Non-owning
                std::vector<std::shared_ptr<MCTSNode>> path;
                path.reserve(50);
                
                auto state = root->getState().clone();
                
                // Selection phase with virtual loss
                while (!current->isLeaf() && !state->isTerminal()) {
                    path.push_back(current);
                    current->applyVirtualLoss(settings_.virtual_loss);
                    
                    auto next = current->selectBestChildUCB(settings_.exploration_constant, thread_rng);
                    if (!next) {
                        // Revert virtual losses if selection failed
                        for (auto it = path.rbegin(); it != path.rend(); ++it) {
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                        }
                        current = nullptr;
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
                        path.push_back(current);
                        current = children[0];
                        state = current->getState().clone();
                    }
                }
                
                // Create request
                LeafEvalRequest request;
                request.node = current.get();
                request.state = std::move(state);
                
                // Convert path to raw pointers
                request.path.reserve(path.size());
                for (const auto& node : path) {
                    request.path.push_back(node.get());
                }
                
                // Try to enqueue
                if (leaf_queue.enqueue(std::move(request))) {
                    local_collected++;
                    leaves_collected.fetch_add(1);
                } else {
                    // Queue full - revert virtual losses
                    for (auto node : request.path) {
                        node->revertVirtualLoss(settings_.virtual_loss);
                    }
                }
                
                // Yield occasionally
                if (local_collected % 5 == 0) {
                    std::this_thread::yield();
                }
            }
            
            std::cout << "Collector " << worker_id << " finished: " 
                      << local_collected << " leaves" << std::endl;
        });
    }
    
    // Single inference thread
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
            
            // Check if should process
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time).count();
            
            bool should_process = !batch.empty() && (
                batch.size() >= static_cast<size_t>(settings_.batch_size) ||
                elapsed_ms >= settings_.batch_timeout.count() ||
                shutdown.load()
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
                auto results = neural_network_->inference(state_batch);
                auto eval_end = std::chrono::steady_clock::now();
                
                auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    eval_end - eval_start).count();
                total_inference_time_ms.fetch_add(eval_ms);
                gpu_inferences.fetch_add(1);
                
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                
                std::cout << "✅ Batch " << batches_processed.load() 
                          << ": " << batch.size() << " states in " 
                          << eval_ms << "ms" << std::endl;
                
                // Queue results
                for (size_t i = 0; i < batch.size(); ++i) {
                    EvalResult result;
                    result.node = batch[i].node;
                    result.value = results[i].value;
                    result.policy = results[i].policy;
                    result.path = std::move(batch[i].path);
                    
                    result_queue.enqueue(std::move(result));
                }
                
                batch.clear();
                last_batch_time = now;
            } else if (batch.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::cout << "Inference thread finished: " 
                  << batches_processed.load() << " batches" << std::endl;
    });
    
    // Backpropagation threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&, i]() {
            int local_processed = 0;
            
            while (!shutdown.load() || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
                    // Set policy
                    if (!result.policy.empty() && result.node) {
                        result.node->setPriorProbabilities(result.policy);
                    }
                    
                    // Backpropagate
                    float value = result.value;
                    for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
                        (*it)->update(value);
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                        value = -value;
                    }
                    
                    simulations_completed.fetch_add(1);
                    local_processed++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            std::cout << "Backprop " << i << " finished: " 
                      << local_processed << " results" << std::endl;
        });
    }
    
    // Progress monitor thread
    threads.emplace_back([&]() {
        auto start_time = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            
            int sims = simulations_completed.load();
            int collected = leaves_collected.load();
            int batches = batches_processed.load();
            float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
            float throughput = elapsed_sec > 0 ? float(sims) / elapsed_sec : 0;
            
            // GPU utilization estimate
            float gpu_util = 0;
            if (gpu_inferences.load() > 0 && elapsed_sec > 0) {
                float avg_inference_ms = float(total_inference_time_ms.load()) / gpu_inferences.load();
                float inferences_per_sec = float(gpu_inferences.load()) / elapsed_sec;
                gpu_util = (avg_inference_ms * inferences_per_sec) / 10.0f;
            }
            
            std::cout << "\n📊 METRICS: " << sims << "/" << num_simulations 
                      << " | Collected: " << collected 
                      << " | Batches: " << batches << " (avg " << avg_batch << ")"
                      << " | Queues: " << leaf_queue.size_approx() << "/" << result_queue.size_approx()
                      << "\n   Throughput: " << throughput << " sims/sec"
                      << " | GPU: ~" << std::min(100.0f, gpu_util) << "%"
                      << " | Memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB";
            
            if (throughput < 50 && sims > 50) {
                std::cout << " ⚠️ LOW";
            }
            std::cout << std::endl;
        }
    });
    
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
    std::cout << "  Batches: " << batches_processed.load() << std::endl;
    std::cout << "  Avg batch size: " << (batches_processed > 0 ? 
        float(total_batch_size.load()) / batches_processed.load() : 0) << std::endl;
    std::cout << "  Throughput: " << (duration.count() > 0 ? 
        1000.0f * simulations_completed.load() / duration.count() : 0) << " sims/sec" << std::endl;
    std::cout << "  Final memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
}

} // namespace mcts
} // namespace alphazero