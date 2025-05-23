#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cmath>
#include <moodycamel/concurrentqueue.h>

namespace alphazero {
namespace mcts {

void MCTSEngine::executeTaskflowSearch(MCTSNode* root, int num_simulations) {
    std::cout << "🚀 TASKFLOW LEAF PARALLELIZATION: " << num_simulations 
              << " simulations, batch_size=" << settings_.batch_size 
              << ", threads=" << settings_.num_threads << std::endl;
    
    auto search_start = std::chrono::steady_clock::now();
    AggressiveMemoryManager& memory_manager = AggressiveMemoryManager::getInstance();
    
    // Print initial memory state
    std::cout << "Initial memory: " << memory_manager.getMemoryReport() << std::endl;
    
    // Atomic counters
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    
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
    
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(settings_.num_threads * 32);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(settings_.num_threads * 32);
    
    // Control flags
    std::atomic<bool> collection_active(true);
    std::atomic<bool> inference_active(true);
    
    // Create taskflow executor with specified number of threads
    tf::Executor executor(settings_.num_threads);
    tf::Taskflow taskflow;
    
    // No need for pre-allocated RNGs - each thread will create its own
    
    // Task 1: Leaf collection using taskflow parallel_for
    auto leaf_collection = taskflow.emplace([&](tf::Subflow& subflow) {
        const int batch_target = settings_.batch_size;
        const int collectors_per_batch = 4;  // Multiple collectors per batch for efficiency
        
        while (collection_active.load() && simulations_completed.load() < num_simulations) {
            // Launch parallel collectors for this batch
            subflow.for_each_index(0, collectors_per_batch, 1, [&](int collector_id) {
                // Create thread-local RNG
                std::mt19937 thread_rng(std::random_device{}() + collector_id);
                
                int local_collected = 0;
                const int max_per_collector = batch_target / collectors_per_batch + 1;
                
                while (local_collected < max_per_collector && 
                       collection_active.load() && 
                       simulations_completed.load() < num_simulations) {
                    
                    // Tree traversal - use shared_ptr for consistency with node methods
                    std::shared_ptr<MCTSNode> current(root, [](MCTSNode*){});  // Non-owning shared_ptr
                    std::vector<std::shared_ptr<MCTSNode>> path;
                    path.reserve(50);
                    
                    auto state = root->getState().clone();
                    // Track memory allocation (estimate size)
                    TRACK_MEMORY_ALLOC("GameStateClone", sizeof(*state) + 1024);
                    
                    // Selection phase with virtual loss
                    while (!current->isLeaf() && !state->isTerminal()) {
                        path.push_back(current);
                        
                        // Apply virtual loss
                        current->applyVirtualLoss(settings_.virtual_loss);
                        
                        // Select best child
                        current = current->selectBestChildUCB(settings_.exploration_constant, thread_rng);
                        if (!current) break;
                        
                        // Make move
                        int move = current->getAction();
                        state->makeMove(move);
                    }
                    
                    if (!current) {
                        // Revert virtual losses if selection failed
                        for (auto it = path.rbegin(); it != path.rend(); ++it) {
                            (*it)->revertVirtualLoss(settings_.virtual_loss);
                        }
                        continue;
                    }
                    
                    path.push_back(current);
                    
                    // Expansion phase
                    if (!state->isTerminal() && current->getVisitCount() > 0 && !current->isExpanded()) {
                        // Expand node using its built-in method
                        current->expand(settings_.use_progressive_widening,
                                      settings_.progressive_widening_c,
                                      settings_.progressive_widening_k);
                        
                        // Select first child if expansion was successful
                        auto& children = current->getChildren();
                        if (!children.empty()) {
                            current->applyVirtualLoss(settings_.virtual_loss);
                            path.push_back(current);
                            current = children[0];  // Select first child as leaf
                            state = current->getState().clone();
                        }
                    }
                    
                    // Queue for evaluation
                    LeafEvalRequest request;
                    request.node = current.get();  // Convert to raw pointer for storage
                    request.state = std::move(state);
                    // Convert path to raw pointers
                    std::vector<MCTSNode*> raw_path;
                    raw_path.reserve(path.size());
                    for (const auto& node : path) {
                        raw_path.push_back(node.get());
                    }
                    request.path = std::move(raw_path);
                    
                    if (!leaf_queue.enqueue(std::move(request))) {
                        std::cerr << "Failed to enqueue leaf!" << std::endl;
                        // Revert virtual losses
                        for (auto node : request.path) {
                            node->revertVirtualLoss(settings_.virtual_loss);
                        }
                    } else {
                        local_collected++;
                        leaves_collected.fetch_add(1);
                    }
                }
            });
            
            subflow.join();  // Wait for this batch of collectors
        }
    }).name("leaf_collection");
    
    // Task 2: Batch inference processor
    auto inference_processor = taskflow.emplace([&]() {
        auto last_batch_time = std::chrono::steady_clock::now();
        std::vector<LeafEvalRequest> batch;
        batch.reserve(settings_.batch_size);
        
        while (inference_active.load() || leaf_queue.size_approx() > 0) {
            LeafEvalRequest request;
            
            // Try to fill a batch
            while (batch.size() < static_cast<size_t>(settings_.batch_size) && 
                   leaf_queue.try_dequeue(request)) {
                batch.push_back(std::move(request));
            }
            
            // Check if we should process the batch
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time).count();
            
            bool should_process = !batch.empty() && (
                batch.size() >= static_cast<size_t>(settings_.batch_size) ||
                elapsed >= settings_.batch_timeout.count() ||
                !inference_active.load()
            );
            
            if (should_process) {
                // Prepare states for evaluation
                std::vector<std::unique_ptr<core::IGameState>> state_batch;
                state_batch.reserve(batch.size());
                for (auto& req : batch) {
                    state_batch.push_back(std::move(req.state));
                }
                
                // Neural network inference
                auto eval_start = std::chrono::steady_clock::now();
                // Call neural network inference with unique_ptr vector
                auto results = neural_network_->inference(state_batch);
                
                std::vector<float> values;
                std::vector<std::vector<float>> policies;
                values.reserve(results.size());
                policies.reserve(results.size());
                
                for (const auto& result : results) {
                    values.push_back(result.value);
                    policies.push_back(result.policy);
                }
                
                // Move states back to requests for cleanup
                for (size_t i = 0; i < batch.size(); ++i) {
                    batch[i].state = std::move(state_batch[i]);
                }
                auto eval_end = std::chrono::steady_clock::now();
                
                auto eval_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    eval_end - eval_start).count() / 1000.0;
                
                // Update statistics
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                
                std::cout << "✅ Batch " << batches_processed.load() 
                          << ": " << batch.size() << " states in " 
                          << eval_duration << "ms (" 
                          << (batch.size() * 1000.0 / eval_duration) 
                          << " states/sec)" << std::endl;
                
                // Queue results for backpropagation
                for (size_t i = 0; i < batch.size(); ++i) {
                    EvalResult result;
                    result.node = batch[i].node;
                    result.value = values[i];
                    result.policy = policies[i];
                    result.path = std::move(batch[i].path);
                    
                    if (!result_queue.enqueue(std::move(result))) {
                        std::cerr << "Failed to enqueue result!" << std::endl;
                    }
                    
                    // Free game state memory
                    // Track memory deallocation (estimate)
                    TRACK_MEMORY_FREE("GameStateClone", sizeof(*batch[i].state) + 1024);
                }
                
                batch.clear();
                last_batch_time = now;
                
                // Memory cleanup every 10 batches
                if (batches_processed.load() % 10 == 0) {
                    memory_manager.forceCleanup();
                }
            } else if (batch.empty()) {
                // No work available, brief sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }).name("inference_processor");
    
    // Task 3: Backpropagation workers using taskflow
    auto backprop_workers = taskflow.emplace([&](tf::Subflow& subflow) {
        const int num_backprop_workers = 2;
        
        subflow.for_each_index(0, num_backprop_workers, 1, [&](int worker_id) {
            int local_processed = 0;
            
            while (inference_active.load() || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
                    // Update node values
                    float value = result.value;
                    
                    // Set policy for leaf node if needed
                    if (result.node->isLeaf() && !result.policy.empty()) {
                        // Set policy for unexpanded leaf node
                        if (!result.policy.empty()) {
                            result.node->setPriorProbabilities(result.policy);
                        }
                    }
                    
                    // Backpropagate value through path
                    for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
                        (*it)->update(value);
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                        value = -value;  // Flip for opponent
                    }
                    
                    simulations_completed.fetch_add(1);
                    local_processed++;
                    
                    if (local_processed % 10 == 0) {
                        std::cout << "✓ Backprop worker " << worker_id 
                                  << " completed " << simulations_completed.load() 
                                  << " simulations" << std::endl;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            std::cout << "Backprop worker " << worker_id 
                      << " finished with " << local_processed 
                      << " results processed" << std::endl;
        });
    }).name("backprop_workers");
    
    // Task 4: Progress monitor
    auto progress_monitor = taskflow.emplace([&]() {
        auto last_report = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            auto now = std::chrono::steady_clock::now();
            int sims = simulations_completed.load();
            int collected = leaves_collected.load();
            int batches = batches_processed.load();
            float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
            
            std::cout << "Progress: " << sims << "/" << num_simulations 
                      << " simulations | " << collected << " collected | "
                      << batches << " batches (avg: " << avg_batch << ") | "
                      << "Queue sizes: " << leaf_queue.size_approx() << " leaves, "
                      << result_queue.size_approx() << " results | "
                      << "Memory: " << memory_manager.getMemoryReport() << std::endl;
            
            last_report = now;
        }
    }).name("progress_monitor");
    
    // Set up task dependencies
    leaf_collection.precede(inference_processor);
    inference_processor.precede(backprop_workers);
    
    // Run the taskflow
    auto future = executor.run(taskflow);
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Shutdown sequence
    std::cout << "Initiating taskflow shutdown..." << std::endl;
    collection_active.store(false);
    
    // Wait for queues to drain
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    inference_active.store(false);
    
    // Wait for taskflow to complete
    future.wait();
    
    // Final statistics
    auto search_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - search_start);
    
    std::cout << "\n✅ TASKFLOW search completed:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "  Simulations: " << simulations_completed.load() << std::endl;
    std::cout << "  Batches: " << batches_processed.load() << std::endl;
    std::cout << "  Avg batch size: " << (batches_processed > 0 ? 
        float(total_batch_size.load()) / batches_processed.load() : 0) << std::endl;
    std::cout << "  Throughput: " << (duration.count() > 0 ? 
        1000.0f * simulations_completed.load() / duration.count() : 0) << " sims/sec" << std::endl;
    
    // Final memory report
    std::cout << "\nFinal memory: " << memory_manager.getMemoryReport() << std::endl;
}

} // namespace mcts
} // namespace alphazero