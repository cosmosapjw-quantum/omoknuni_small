// src/mcts/mcts_engine_true_parallel_search.cpp
// TRUE PARALLEL MCTS SEARCH IMPLEMENTATION WITH PROPER LEAF PARALLELIZATION

#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/aggressive_memory_manager.h"
#include "utils/debug_logger.h"
#include <moodycamel/concurrentqueue.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <future>

namespace alphazero {
namespace mcts {

// Structure for a leaf that needs evaluation
struct LeafEvalRequest {
    std::shared_ptr<MCTSNode> node;
    std::unique_ptr<core::IGameState> state;
    std::vector<std::shared_ptr<MCTSNode>> path;
    int thread_id;
    std::chrono::steady_clock::time_point collection_time;
};

// Structure for evaluation result
struct EvalResult {
    std::shared_ptr<MCTSNode> node;
    std::vector<std::shared_ptr<MCTSNode>> path;
    NetworkOutput output;
};

// Helper function to count nodes in subtree
int countNodes(MCTSNode* node) {
    if (!node) return 0;
    
    int count = 1; // Count this node
    auto& children = node->getChildren();
    for (auto& child : children) {
        if (child) {
            count += countNodes(child.get());
        }
    }
    return count;
}

// Helper function to prune low-visit branches
int pruneTree(MCTSNode* node, int min_visits) {
    if (!node) return 0;
    
    auto& children = node->getChildren();
    if (children.empty()) return 0;
    
    int total_removed = 0;
    
    // First, recursively prune children
    for (auto& child : children) {
        if (child) {
            total_removed += pruneTree(child.get(), min_visits);
        }
    }
    
    // Then remove children with low visits
    auto initial_size = children.size();
    children.erase(
        std::remove_if(children.begin(), children.end(),
            [min_visits, &total_removed](const std::shared_ptr<MCTSNode>& child) {
                if (child && child->getVisitCount() < min_visits) {
                    // Count all nodes in the subtree being removed
                    total_removed += countNodes(child.get());
                    return true;
                }
                return false;
            }),
        children.end()
    );
    
    return total_removed;
}

// Helper function to clear all virtual losses (for deadlock recovery)
void clearAllVirtualLosses(MCTSNode* node) {
    if (!node) return;
    
    // Clear any virtual losses on this node
    int vl = node->getVirtualLoss();
    if (vl > 0) {
        node->removeVirtualLoss(vl);
    }
    
    // Recursively clear children
    auto& children = node->getChildren();
    for (auto& child : children) {
        if (child) {
            clearAllVirtualLosses(child.get());
        }
    }
}

void MCTSEngine::executeTrueParallelSearch(MCTSNode* root, std::unique_ptr<core::IGameState> root_state) {
    auto search_start = std::chrono::steady_clock::now();
    
    std::cout << "🚀🚀 TRUE PARALLEL SEARCH: " 
              << settings_.num_simulations << " simulations, "
              << "batch_size=" << settings_.batch_size << ", "
              << "threads=" << settings_.num_threads << std::endl;
    
    // Initialize aggressive memory management
    auto& memory_manager = AggressiveMemoryManager::getInstance();
    
    // Track initial memory
    std::cout << "Initial memory: " << memory_manager.getMemoryReport() << std::endl;
    
    // Core components for true parallelization with bounded queues
    const size_t max_queue_size = settings_.batch_size * 2;  // Tight bound to force batching
    moodycamel::ConcurrentQueue<LeafEvalRequest> leaf_queue(max_queue_size);
    moodycamel::ConcurrentQueue<EvalResult> result_queue(max_queue_size);
    
    // Track queue memory
    TRACK_MEMORY_ALLOC("LeafQueue", max_queue_size * sizeof(LeafEvalRequest));
    TRACK_MEMORY_ALLOC("ResultQueue", max_queue_size * sizeof(EvalResult));
    
    // Atomic counters
    std::atomic<int> simulations_completed(0);
    std::atomic<int> leaves_collected(0);
    std::atomic<int> batches_processed(0);
    std::atomic<bool> collection_active(true);
    std::atomic<bool> inference_active(true);
    
    // Statistics
    std::atomic<int> total_batch_size(0);
    std::atomic<int> max_queue_size_seen(0);
    
    // Tree pruning control - memory-based dynamic pruning
    std::atomic<int> nodes_created(0);
    std::atomic<int> current_node_count(1);  // Start with root
    std::atomic<int> last_prune_count(0);
    std::atomic<int> prune_threshold(1);     // Minimum visits to keep a node
    
    // Dynamic pruning based on memory pressure
    const int prune_check_interval = 25;     // Check memory every 25 simulations
    const size_t bytes_per_node = sizeof(MCTSNode) + 256; // Node + children vector estimate
    
    // Memory thresholds for different pruning aggressiveness
    const size_t memory_warning_gb = 8;      // Start gentle pruning at 8GB
    const size_t memory_critical_gb = 16;    // Aggressive pruning at 16GB
    const size_t memory_emergency_gb = 24;   // Emergency pruning at 24GB
    
    std::cout << "Dynamic memory-based tree pruning enabled:" << std::endl;
    std::cout << "  Warning threshold: " << memory_warning_gb << "GB (prune 1-visit nodes)" << std::endl;
    std::cout << "  Critical threshold: " << memory_critical_gb << "GB (prune 2-visit nodes)" << std::endl;
    std::cout << "  Emergency threshold: " << memory_emergency_gb << "GB (prune 5-visit nodes)" << std::endl;
    
    // Deadlock detection
    std::atomic<std::chrono::steady_clock::time_point> last_progress_time(std::chrono::steady_clock::now());
    std::atomic<int> stuck_threads(0);
    
    // PHASE 1: Launch collection workers that CONTINUOUSLY collect leaves
    std::vector<std::thread> collectors;
    collectors.reserve(settings_.num_threads);
    
    auto collector_fn = [&](int thread_id) {
        std::mt19937 thread_rng(std::random_device{}() + thread_id);
        int local_collected = 0;
        auto last_collection_time = std::chrono::steady_clock::now();
        
        while (collection_active.load() && simulations_completed.load() < settings_.num_simulations) {
            // Debug log every 10th iteration
            static thread_local int iteration_count = 0;
            iteration_count++;
            bool debug_this_iter = (iteration_count % 10 == 0);
            
            if (debug_this_iter && thread_id == 0) {
                std::cout << "Thread " << thread_id << " iter " << iteration_count 
                          << ": sims=" << simulations_completed.load() 
                          << ", collected=" << leaves_collected.load() << std::endl;
            }
            
            // Tree traversal to find leaf
            std::shared_ptr<MCTSNode> current = std::shared_ptr<MCTSNode>(root, [](MCTSNode*){});
            std::vector<std::shared_ptr<MCTSNode>> path;
            
            // Selection phase with virtual loss
            while (current && !current->isTerminal() && current->isExpanded()) {
                // Apply virtual loss BEFORE selection to prevent collision
                // Use smaller virtual loss to prevent saturation
                current->applyVirtualLoss(std::min(settings_.virtual_loss, 5));
                path.push_back(current);
                
                // Select best child
                auto children = current->getChildren();
                if (children.empty()) break;
                
                float best_uct = -std::numeric_limits<float>::infinity();
                std::shared_ptr<MCTSNode> best_child = nullptr;
                
                for (auto& child : children) {
                    // Calculate UCT inline
                    int child_visits = child->getVisitCount();
                    int parent_visits = current->getVisitCount();
                    float child_value = child->getValue();
                    float prior = child->getPriorProbability();
                    
                    float exploitation = child_visits > 0 ? child_value : 0.0f;
                    float exploration_param = settings_.exploration_constant * 
                        std::sqrt(static_cast<float>(parent_visits));
                    float exploration = prior * exploration_param / (1 + child_visits);
                    float uct = exploitation + exploration;
                    
                    if (uct > best_uct) {
                        best_uct = uct;
                        best_child = child;
                    }
                }
                
                current = best_child;
            }
            
            // Expansion phase if needed
            if (current && !current->isTerminal() && !current->isExpanded()) {
                auto state_clone = current->getState().clone();
                if (state_clone) {
                    auto legal_moves = state_clone->getLegalMoves();
                    if (!legal_moves.empty()) {
                        // Expand node with progressive widening to limit memory usage
                        current->expand(settings_.use_progressive_widening, 
                                      settings_.progressive_widening_c,
                                      settings_.progressive_widening_k);
                        
                        // Track new nodes created
                        auto children = current->getChildren();
                        current_node_count.fetch_add(children.size());
                        
                        // Select first child as leaf
                        if (!children.empty()) {
                            current->applyVirtualLoss(std::min(settings_.virtual_loss, 5));
                            path.push_back(current);
                            current = children[0];
                        }
                    }
                }
            }
            
            // If we have a valid leaf, add to queue
            if (current && !current->isTerminal()) {
                LeafEvalRequest request;
                request.node = current;
                request.state = current->getState().clone();
                request.path = std::move(path);
                request.thread_id = thread_id;
                request.collection_time = std::chrono::steady_clock::now();
                
                // Track memory for the request
                TRACK_MEMORY_ALLOC("LeafRequest", sizeof(LeafEvalRequest) + 
                                   (request.path.size() * sizeof(std::shared_ptr<MCTSNode>)));
                
                // Implement backpressure - wait if queue is getting full
                int queue_size = leaf_queue.size_approx();
                if (queue_size >= max_queue_size * 0.8) {  // 80% full
                    // Revert virtual loss and wait
                    for (auto& node : path) {
                        node->removeVirtualLoss(std::min(settings_.virtual_loss, 5));
                    }
                    TRACK_MEMORY_FREE("LeafRequest", sizeof(LeafEvalRequest) + 
                                      (path.size() * sizeof(std::shared_ptr<MCTSNode>)));
                    
                    // Wait for queue to drain
                    while (leaf_queue.size_approx() >= max_queue_size * 0.5 && 
                           collection_active.load()) {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    continue;  // Retry with a new leaf
                }
                
                // Try to enqueue with retries
                bool enqueued = false;
                for (int retry = 0; retry < 3 && !enqueued; ++retry) {
                    if (leaf_queue.enqueue(std::move(request))) {
                        leaves_collected.fetch_add(1);
                        local_collected++;
                        enqueued = true;
                        
                        // Update max queue size for monitoring
                        int current_max = max_queue_size_seen.load();
                        while (queue_size > current_max && 
                               !max_queue_size_seen.compare_exchange_weak(current_max, queue_size)) {}
                    } else {
                        // Brief wait before retry
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                }
                
                if (!enqueued) {
                    // Failed to enqueue - revert virtual loss
                    for (auto& node : path) {
                        node->removeVirtualLoss(std::min(settings_.virtual_loss, 5));
                    }
                    TRACK_MEMORY_FREE("LeafRequest", sizeof(LeafEvalRequest) + 
                                      (path.size() * sizeof(std::shared_ptr<MCTSNode>)));
                }
            } else if (!path.empty()) {
                // Terminal node or expansion failed - revert virtual loss
                for (auto& node : path) {
                    node->removeVirtualLoss(std::min(settings_.virtual_loss, 5));
                }
            }
            
            // NO WAITING - immediately continue to next collection
            std::this_thread::yield();
            
            // Deadlock detection - check if stuck
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_collection_time).count() > 5) {
                stuck_threads.fetch_add(1);
                std::cout << "⚠️ Collector " << thread_id << " stuck for 5+ seconds. Virtual losses may be saturated." << std::endl;
                
                // Try to recover by clearing some virtual losses
                if (current) {
                    current->removeVirtualLoss(settings_.virtual_loss * 10);  // Remove extra to unstick
                }
                last_collection_time = now;
            }
            
            // Update progress timestamp on successful collection
            if (local_collected > 0) {
                last_progress_time.store(now);
            }
        }
        
        std::cout << "Collector " << thread_id << " finished with " 
                  << local_collected << " leaves collected" << std::endl;
    };
    
    // PHASE 2: Launch inference processor that batches and evaluates
    std::thread inference_processor([&]() {
        std::vector<LeafEvalRequest> batch;
        batch.reserve(settings_.batch_size);
        
        // auto last_batch_time = std::chrono::steady_clock::now();  // Currently unused
        const auto batch_timeout = std::chrono::milliseconds(settings_.batch_timeout.count());
        
        while (inference_active.load()) {
            // Collect batch from queue
            batch.clear();
            
            // Try bulk dequeue first
            std::vector<LeafEvalRequest> bulk_buffer(settings_.batch_size);
            size_t dequeued = leaf_queue.try_dequeue_bulk(bulk_buffer.begin(), settings_.batch_size);
            
            for (size_t i = 0; i < dequeued; ++i) {
                batch.push_back(std::move(bulk_buffer[i]));
            }
            
            // Continue collecting until batch is ready or timeout
            auto batch_start = std::chrono::steady_clock::now();
            const auto min_batch_wait = std::chrono::milliseconds(50);  // Increase to 50ms for better batching
            
            while (batch.size() < settings_.batch_size) {
                auto elapsed = std::chrono::steady_clock::now() - batch_start;
                
                // Only process partial batch if:
                // 1. Timeout exceeded AND we waited minimum time AND have minimum batch
                // 2. No more simulations to run
                const size_t min_batch_size = 8;  // Don't process less than 8
                bool should_process_partial = 
                    (elapsed > batch_timeout && elapsed > min_batch_wait && batch.size() >= min_batch_size) ||
                    (simulations_completed.load() >= settings_.num_simulations && !batch.empty());
                
                if (should_process_partial) {
                    break;
                }
                
                LeafEvalRequest request;
                if (leaf_queue.try_dequeue(request)) {
                    batch.push_back(std::move(request));
                } else {
                    // Wait more aggressively for full batches
                    if (batch.size() < settings_.batch_size / 2) {
                        // Less than half full - wait longer
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    } else {
                        // More than half full - shorter wait  
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                }
            }
            
            // Process batch if we have items
            if (!batch.empty()) {
                // Track batch memory
                size_t batch_memory = batch.size() * sizeof(LeafEvalRequest);
                TRACK_MEMORY_ALLOC("InferenceBatch", batch_memory);
                
                // Prepare states for neural network
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.reserve(batch.size());
                
                for (auto& request : batch) {
                    states.push_back(std::move(request.state));
                    // Free the request memory after moving state
                    TRACK_MEMORY_FREE("LeafRequest", sizeof(LeafEvalRequest) + 
                                      (request.path.size() * sizeof(std::shared_ptr<MCTSNode>)));
                }
                
                // SINGLE GPU INFERENCE CALL FOR ENTIRE BATCH
                auto nn_start = std::chrono::steady_clock::now();
                std::vector<NetworkOutput> outputs = direct_inference_fn_(states);
                auto nn_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - nn_start);
                
                // Create results
                for (size_t i = 0; i < batch.size() && i < outputs.size(); ++i) {
                    EvalResult result;
                    result.node = batch[i].node;
                    result.path = std::move(batch[i].path);
                    result.output = outputs[i];
                    
                    result_queue.enqueue(std::move(result));
                }
                
                // Update statistics
                batches_processed.fetch_add(1);
                total_batch_size.fetch_add(batch.size());
                
                // Free batch memory
                TRACK_MEMORY_FREE("InferenceBatch", batch_memory);
                
                std::cout << "✅ Batch " << batches_processed.load() 
                          << ": " << batch.size() << " states in " 
                          << nn_duration.count() / 1000.0 << "ms ("
                          << (batch.size() * 1000000.0 / nn_duration.count()) 
                          << " states/sec)" << std::endl;
                
                // Check memory pressure periodically
                if (batches_processed.load() % 10 == 0) {
                    auto pressure = memory_manager.getMemoryPressure();
                    if (pressure >= AggressiveMemoryManager::PressureLevel::WARNING) {
                        std::cout << "Memory pressure detected: " << static_cast<int>(pressure) << std::endl;
                        memory_manager.forceCleanup(pressure);
                    }
                }
            }
            
            // Check if we should stop
            if (simulations_completed.load() >= settings_.num_simulations && 
                leaf_queue.size_approx() == 0) {
                break;
            }
        }
        
        std::cout << "Inference processor finished with " 
                  << batches_processed.load() << " batches processed" << std::endl;
    });
    
    // PHASE 3: Launch backpropagation workers
    std::vector<std::thread> backprop_workers;
    const int num_backprop_workers = 2;  // Fewer than collectors
    
    auto backprop_fn = [&]() {
        int local_processed = 0;
        
        while (simulations_completed.load() < settings_.num_simulations || 
               result_queue.size_approx() > 0) {
            
            EvalResult result;
            if (result_queue.try_dequeue(result)) {
                // Apply policy to node
                if (result.node && !result.output.policy.empty()) {
                    result.node->setPriorProbabilities(result.output.policy);
                }
                
                // Backpropagate value
                float value = result.output.value;
                for (auto& node : result.path) {
                    // Remove same amount of virtual loss as applied
                    node->removeVirtualLoss(std::min(settings_.virtual_loss, 5));
                    node->update(value);
                    value = -value;  // Flip for opponent
                }
                
                simulations_completed.fetch_add(1);
                local_processed++;
                
                // Log every 10th completion
                if (simulations_completed.load() % 10 == 0) {
                    std::cout << "✓ Backprop completed " << simulations_completed.load() 
                              << " simulations (queue: " << result_queue.size_approx() << ")" << std::endl;
                }
                
                // Dynamic memory-based tree pruning
                int sims = simulations_completed.load();
                int last_prune = last_prune_count.load();
                if (sims - last_prune >= prune_check_interval && 
                    last_prune_count.compare_exchange_strong(last_prune, sims)) {
                    
                    // Check memory usage and adjust pruning threshold
                    auto& mem_mgr = AggressiveMemoryManager::getInstance();
                    double current_usage_gb = mem_mgr.getCurrentMemoryUsageGB();
                    int nodes_before = current_node_count.load();
                    
                    // Determine pruning threshold based on memory
                    int min_visits = 1;
                    AggressiveMemoryManager::PressureLevel pressure_level = AggressiveMemoryManager::PressureLevel::NORMAL;
                    
                    if (current_usage_gb >= memory_emergency_gb) {
                        min_visits = 5;  // Very aggressive pruning
                        pressure_level = AggressiveMemoryManager::PressureLevel::EMERGENCY;
                        std::cout << "🚨 EMERGENCY memory pruning (>" << memory_emergency_gb << "GB): removing nodes with <" << min_visits << " visits" << std::endl;
                    } else if (current_usage_gb >= memory_critical_gb) {
                        min_visits = 2;  // Aggressive pruning
                        pressure_level = AggressiveMemoryManager::PressureLevel::CRITICAL;
                        std::cout << "⚠️ Critical memory pruning (>" << memory_critical_gb << "GB): removing nodes with <" << min_visits << " visits" << std::endl;
                    } else if (current_usage_gb >= memory_warning_gb) {
                        min_visits = 1;  // Gentle pruning
                        pressure_level = AggressiveMemoryManager::PressureLevel::WARNING;
                        std::cout << "💡 Warning memory pruning (>" << memory_warning_gb << "GB): removing nodes with <" << min_visits << " visits" << std::endl;
                    }
                    
                    // Prune tree if needed
                    if (current_usage_gb >= memory_warning_gb) {
                        int nodes_removed = pruneTree(root, min_visits);
                        current_node_count.fetch_sub(nodes_removed);
                        int nodes_after = current_node_count.load();
                        
                        std::cout << "🌳 Tree pruned at " << sims << " simulations: " 
                                  << nodes_removed << " nodes removed (from " << nodes_before 
                                  << " to " << nodes_after << "), memory: " 
                                  << current_usage_gb << "GB" << std::endl;
                        
                        // Force memory cleanup
                        mem_mgr.forceCleanup(pressure_level);
                    }
                    
                    prune_threshold.store(min_visits);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        std::cout << "Backprop worker finished with " 
                  << local_processed << " results processed" << std::endl;
    };
    
    // Launch workers
    for (int i = 0; i < settings_.num_threads; ++i) {
        collectors.emplace_back(collector_fn, i);
    }
    
    for (int i = 0; i < num_backprop_workers; ++i) {
        backprop_workers.emplace_back(backprop_fn);
    }
    
    // Add a shutdown flag for clean termination
    std::atomic<bool> shutdown_requested(false);
    
    // Launch deadlock monitor thread
    std::thread deadlock_monitor([&]() {
        auto last_sim_count = 0;
        auto stuck_count = 0;
        
        while (!shutdown_requested.load() && simulations_completed.load() < settings_.num_simulations) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            int current_sims = simulations_completed.load();
            if (current_sims == last_sim_count && current_sims < settings_.num_simulations) {
                stuck_count++;
                std::cout << "\n🚨 DEADLOCK DETECTED! No progress for " << (stuck_count * 5) 
                          << " seconds at " << current_sims << " simulations" << std::endl;
                std::cout << "  Stuck threads: " << stuck_threads.load() << std::endl;
                std::cout << "  Leaf queue size: " << leaf_queue.size_approx() << std::endl;
                std::cout << "  Result queue size: " << result_queue.size_approx() << std::endl;
                std::cout << "  Node count: " << current_node_count.load() << std::endl;
                
                // Attempt recovery after 10 seconds
                if (stuck_count >= 2) {
                    std::cout << "🔧 Attempting deadlock recovery..." << std::endl;
                    
                    // Clear virtual losses from root to help unstick
                    if (root) {
                        clearAllVirtualLosses(root);
                    }
                    
                    // Reset stuck thread counter
                    stuck_threads.store(0);
                    
                    // Print more diagnostic info
                    std::cout << "🔍 Diagnostic info:" << std::endl;
                    std::cout << "  Total threads: " << settings_.num_threads << std::endl;
                    std::cout << "  Collectors running: " << collectors.size() << std::endl;
                    std::cout << "  Simulations completed: " << simulations_completed.load() << std::endl;
                    std::cout << "  Leaves collected: " << leaves_collected.load() << std::endl;
                    std::cout << "  Current node count: " << current_node_count.load() << std::endl;
                    
                    // Check if we're stuck at exactly 30
                    if (current_sims == 30) {
                        std::cout << "⚠️ STUCK AT EXACTLY 30! This suggests a hardcoded limit." << std::endl;
                    }
                    
                    // If still stuck after 20 seconds, break out
                    if (stuck_count >= 4) {
                        std::cout << "❌ Unable to recover from deadlock. Terminating search." << std::endl;
                        collection_active.store(false);
                        inference_active.store(false);
                        shutdown_requested.store(true);
                        break;
                    }
                }
            } else {
                stuck_count = 0;  // Reset if progress made
                last_sim_count = current_sims;
            }
        }
        
        std::cout << "Deadlock monitor shutting down" << std::endl;
    });
    
    // Monitor progress
    auto last_report = std::chrono::steady_clock::now();
    while (simulations_completed.load() < settings_.num_simulations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count() >= 1) {
            int sims = simulations_completed.load();
            int collected = leaves_collected.load();
            int batches = batches_processed.load();
            float avg_batch = batches > 0 ? float(total_batch_size.load()) / batches : 0;
            
            std::cout << "Progress: " << sims << "/" << settings_.num_simulations 
                      << " simulations | " << collected << " collected | "
                      << batches << " batches (avg: " << avg_batch << ") | "
                      << "Queue sizes: " << leaf_queue.size_approx() << " leaves, "
                      << result_queue.size_approx() << " results | "
                      << "Max queue: " << max_queue_size_seen.load() << " | "
                      << "Nodes: " << current_node_count.load() << " | "
                      << "Memory: " << AggressiveMemoryManager::formatBytes(
                          memory_manager.getCurrentMemoryUsage()) << std::endl;
            
            last_report = now;
        }
    }
    
    // Shutdown sequence
    std::cout << "Starting shutdown sequence..." << std::endl;
    
    // Step 1: Stop collectors from generating new work
    collection_active.store(false);
    std::cout << "  - Collectors signaled to stop" << std::endl;
    
    // Step 2: Wait for remaining items to process (with timeout)
    auto shutdown_start = std::chrono::steady_clock::now();
    const auto shutdown_timeout = std::chrono::seconds(10);
    
    while ((leaf_queue.size_approx() > 0 || result_queue.size_approx() > 0) &&
           std::chrono::steady_clock::now() - shutdown_start < shutdown_timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "  - Waiting for queues to drain: " 
                  << leaf_queue.size_approx() << " leaves, " 
                  << result_queue.size_approx() << " results" << std::endl;
    }
    
    // Step 3: Stop inference processor
    inference_active.store(false);
    std::cout << "  - Inference processor signaled to stop" << std::endl;
    
    // Step 4: Signal deadlock monitor to shutdown
    shutdown_requested.store(true);
    std::cout << "  - Deadlock monitor signaled to stop" << std::endl;
    
    // Step 5: Join threads (they should exit cleanly now)
    std::cout << "  - Joining threads..." << std::endl;
    
    // Join all threads
    for (size_t i = 0; i < collectors.size(); ++i) {
        if (collectors[i].joinable()) {
            std::cout << "    - Joining collector " << i << std::endl;
            collectors[i].join();
        }
    }
    std::cout << "  - All collectors joined" << std::endl;
    
    if (inference_processor.joinable()) {
        std::cout << "    - Joining inference processor" << std::endl;
        inference_processor.join();
    }
    std::cout << "  - Inference processor joined" << std::endl;
    
    for (size_t i = 0; i < backprop_workers.size(); ++i) {
        if (backprop_workers[i].joinable()) {
            std::cout << "    - Joining backprop worker " << i << std::endl;
            backprop_workers[i].join();
        }
    }
    std::cout << "  - All backprop workers joined" << std::endl;
    
    if (deadlock_monitor.joinable()) {
        std::cout << "    - Joining deadlock monitor" << std::endl;
        deadlock_monitor.join();
    }
    std::cout << "  - Deadlock monitor joined" << std::endl;
    
    std::cout << "Shutdown sequence completed" << std::endl;
    
    // Final statistics
    auto search_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start);
    
    std::cout << "✅ TRUE PARALLEL search completed:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "  Simulations: " << simulations_completed.load() << std::endl;
    std::cout << "  Leaves collected: " << leaves_collected.load() << std::endl;
    std::cout << "  Batches: " << batches_processed.load() << std::endl;
    std::cout << "  Avg batch size: " << (batches_processed > 0 ? 
        float(total_batch_size.load()) / batches_processed.load() : 0) << std::endl;
    std::cout << "  Throughput: " << (duration.count() > 0 ? 
        1000.0f * simulations_completed.load() / duration.count() : 0) << " sims/sec" << std::endl;
    std::cout << "  Max queue size: " << max_queue_size_seen.load() << std::endl;
    
    // Final memory report
    std::cout << "\nFinal memory report:" << memory_manager.getMemoryReport() << std::endl;
    
    // Free queue memory tracking
    TRACK_MEMORY_FREE("LeafQueue", max_queue_size * sizeof(LeafEvalRequest));
    TRACK_MEMORY_FREE("ResultQueue", max_queue_size * sizeof(EvalResult));
}

} // namespace mcts
} // namespace alphazero