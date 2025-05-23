#include "mcts/mcts_engine.h"
#include "mcts/aggressive_memory_manager.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cmath>
#include <moodycamel/concurrentqueue.h>
#include <thread>

// For CPU monitoring
#ifdef __linux__
#include <sys/times.h>
#include <unistd.h>
#endif

namespace alphazero {
namespace mcts {

// CPU usage tracker
class CPUMonitor {
private:
    clock_t lastCPU, lastSysCPU, lastUserCPU;
    int numProcessors;
    
public:
    CPUMonitor() {
        #ifdef __linux__
        FILE* file = fopen("/proc/stat", "r");
        fscanf(file, "cpu %ld %ld %ld %ld", &lastCPU, &lastUserCPU, &lastSysCPU, &lastCPU);
        fclose(file);
        numProcessors = sysconf(_SC_NPROCESSORS_ONLN);
        #endif
    }
    
    double getCurrentCPUUsage() {
        #ifdef __linux__
        double percent = 0.0;
        FILE* file = fopen("/proc/stat", "r");
        clock_t nowCPU, nowUserCPU, nowSysCPU, nowIdle;
        fscanf(file, "cpu %ld %ld %ld %ld", &nowCPU, &nowUserCPU, &nowSysCPU, &nowIdle);
        fclose(file);
        
        double total = (nowCPU - lastCPU) + (nowUserCPU - lastUserCPU) + (nowSysCPU - lastSysCPU);
        percent = total;
        percent /= (total + nowIdle - lastCPU);
        percent *= 100;
        
        lastCPU = nowIdle;
        lastUserCPU = nowUserCPU;
        lastSysCPU = nowSysCPU;
        
        return percent;
        #else
        return 0.0;
        #endif
    }
};

void MCTSEngine::executeTaskflowSearch(MCTSNode* root, int num_simulations) {
    std::cout << "🚀 TASKFLOW LEAF PARALLELIZATION: " << num_simulations 
              << " simulations, batch_size=" << settings_.batch_size 
              << ", threads=" << settings_.num_threads << std::endl;
    
    auto search_start = std::chrono::steady_clock::now();
    AggressiveMemoryManager& memory_manager = AggressiveMemoryManager::getInstance();
    CPUMonitor cpu_monitor;
    
    // Print initial memory state
    std::cout << "Initial memory: " << memory_manager.getMemoryReport() << std::endl;
    
    // Atomic counters
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<int> total_batch_size(0);
    std::atomic<bool> collection_active(true);
    std::atomic<bool> inference_active(true);
    
    // Performance tracking
    std::atomic<int> gpu_inferences(0);
    std::atomic<long> total_inference_time_us(0);
    std::atomic<long> total_collection_time_us(0);
    
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
    
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(settings_.num_threads * 64);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(settings_.num_threads * 64);
    
    // Create taskflow executor with specified number of threads
    tf::Executor executor(settings_.num_threads + 3); // Extra threads for inference and monitoring
    tf::Taskflow taskflow;
    
    // Task 1: Leaf collectors (multiple parallel workers)
    for (int worker_id = 0; worker_id < settings_.num_threads; ++worker_id) {
        taskflow.emplace([&, worker_id]() {
            std::mt19937 thread_rng(std::random_device{}() + worker_id);
            int local_collected = 0;
            auto collection_start = std::chrono::steady_clock::now();
            
            while (collection_active.load() && simulations_completed.load() < num_simulations) {
                // Tree traversal - use shared_ptr for consistency with node methods
                std::shared_ptr<MCTSNode> current(root, [](MCTSNode*){});  // Non-owning shared_ptr
                std::vector<std::shared_ptr<MCTSNode>> path;
                path.reserve(50);
                
                auto state = root->getState().clone();
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
                
                // Brief yield to prevent CPU spinning
                if (local_collected % 10 == 0) {
                    std::this_thread::yield();
                }
            }
            
            auto collection_end = std::chrono::steady_clock::now();
            auto collection_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                collection_end - collection_start).count();
            total_collection_time_us.fetch_add(collection_duration);
            
            std::cout << "Collector " << worker_id << " finished with " 
                      << local_collected << " leaves collected" << std::endl;
        }).name("collector_" + std::to_string(worker_id));
    }
    
    // Task 2: Batch inference processor (single thread)
    taskflow.emplace([&]() {
        std::vector<LeafEvalRequest> batch;
        batch.reserve(settings_.batch_size);
        auto last_batch_time = std::chrono::steady_clock::now();
        
        while (inference_active.load() || leaf_queue.size_approx() > 0) {
            // Try to fill a batch
            LeafEvalRequest request;
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
                (!inference_active.load() && leaf_queue.size_approx() == 0)
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
                auto results = neural_network_->inference(state_batch);
                auto eval_end = std::chrono::steady_clock::now();
                
                auto eval_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    eval_end - eval_start).count();
                total_inference_time_us.fetch_add(eval_duration);
                gpu_inferences.fetch_add(1);
                
                // Update statistics
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                
                std::cout << "✅ Batch " << batches_processed.load() 
                          << ": " << batch.size() << " states in " 
                          << (eval_duration / 1000.0) << "ms ("
                          << (batch.size() * 1000000.0 / eval_duration) 
                          << " states/sec)" << std::endl;
                
                // Queue results for backpropagation
                for (size_t i = 0; i < batch.size(); ++i) {
                    EvalResult result;
                    result.node = batch[i].node;
                    result.value = results[i].value;
                    result.policy = results[i].policy;
                    result.path = std::move(batch[i].path);
                    
                    if (!result_queue.enqueue(std::move(result))) {
                        std::cerr << "Failed to enqueue result!" << std::endl;
                    }
                    
                    // Track memory deallocation
                    TRACK_MEMORY_FREE("GameStateClone", sizeof(*state_batch[i]) + 1024);
                }
                
                batch.clear();
                last_batch_time = now;
                
                // Memory cleanup every 10 batches
                if (batches_processed.load() % 10 == 0) {
                    memory_manager.forceCleanup();
                }
            } else if (batch.empty() && leaf_queue.size_approx() == 0) {
                // No work available, brief sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::cout << "Inference processor finished with " 
                  << batches_processed.load() << " batches processed" << std::endl;
    }).name("inference_processor");
    
    // Task 3: Backpropagation workers
    for (int worker_id = 0; worker_id < 2; ++worker_id) {
        taskflow.emplace([&, worker_id]() {
            int local_processed = 0;
            
            while (simulations_completed.load() < num_simulations || result_queue.size_approx() > 0) {
                EvalResult result;
                if (result_queue.try_dequeue(result)) {
                    // Update node values
                    float value = result.value;
                    
                    // Set policy for unexpanded leaf node
                    if (!result.policy.empty()) {
                        result.node->setPriorProbabilities(result.policy);
                    }
                    
                    // Backpropagate value through path
                    for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
                        (*it)->update(value);
                        (*it)->revertVirtualLoss(settings_.virtual_loss);
                        value = -value;  // Flip for opponent
                    }
                    
                    simulations_completed.fetch_add(1);
                    local_processed++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            std::cout << "Backprop worker " << worker_id 
                      << " finished with " << local_processed 
                      << " results processed" << std::endl;
        }).name("backprop_" + std::to_string(worker_id));
    }
    
    // Task 4: Progress monitor with CPU/GPU monitoring
    taskflow.emplace([&]() {
        auto last_report = std::chrono::steady_clock::now();
        auto start_time = std::chrono::steady_clock::now();
        
        while (simulations_completed.load() < num_simulations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            auto now = std::chrono::steady_clock::now();
            int sims = simulations_completed.load();
            int collected = leaves_collected.load();
            int batches = batches_processed.load();
            float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
            
            // Calculate throughput
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            float sim_throughput = elapsed_sec > 0 ? float(sims) / elapsed_sec : 0;
            
            // Calculate GPU utilization
            float gpu_util = 0.0f;
            if (gpu_inferences.load() > 0 && elapsed_sec > 0) {
                float avg_inference_ms = total_inference_time_us.load() / 1000.0f / gpu_inferences.load();
                float inferences_per_sec = gpu_inferences.load() / float(elapsed_sec);
                gpu_util = (avg_inference_ms * inferences_per_sec) / 10.0f; // Approximate %
            }
            
            // Get CPU usage
            double cpu_usage = cpu_monitor.getCurrentCPUUsage();
            
            std::cout << "\n📊 PERFORMANCE METRICS:" << std::endl;
            std::cout << "  Progress: " << sims << "/" << num_simulations 
                      << " simulations | " << collected << " collected | "
                      << batches << " batches (avg: " << avg_batch << ")" << std::endl;
            std::cout << "  Queues: " << leaf_queue.size_approx() << " leaves, "
                      << result_queue.size_approx() << " results" << std::endl;
            std::cout << "  Throughput: " << sim_throughput << " sims/sec | "
                      << "CPU: " << cpu_usage << "% | "
                      << "GPU: ~" << gpu_util << "% utilization" << std::endl;
            std::cout << "  Memory: " << memory_manager.getCurrentMemoryUsageGB() << " GB" << std::endl;
            
            // Warn if throughput is low
            if (sim_throughput < 50 && sims > 50) {
                std::cout << "  ⚠️  LOW THROUGHPUT! Target: 70+ sims/sec" << std::endl;
            }
            
            last_report = now;
        }
    }).name("progress_monitor");
    
    // NO DEPENDENCIES - all tasks run concurrently!
    // This is key to achieving high throughput
    
    // Run the taskflow
    auto future = executor.run(taskflow);
    
    // Wait for completion
    while (simulations_completed.load() < num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Shutdown sequence
    std::cout << "\nInitiating taskflow shutdown..." << std::endl;
    collection_active.store(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    inference_active.store(false);
    
    // Wait for taskflow to complete
    executor.wait_for_all();
    
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
    
    // GPU efficiency
    if (gpu_inferences.load() > 0) {
        float avg_inference_ms = total_inference_time_us.load() / 1000.0f / gpu_inferences.load();
        std::cout << "  GPU avg inference: " << avg_inference_ms << "ms per batch" << std::endl;
    }
    
    std::cout << "\nFinal memory: " << memory_manager.getMemoryReport() << std::endl;
}

} // namespace mcts
} // namespace alphazero