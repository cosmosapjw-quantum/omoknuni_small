#include "mcts/mcts_engine.h"
#include "mcts/burst_batch_collector.h"
#include "mcts/unified_memory_manager.h"
#include "mcts/mcts_node.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace alphazero {
namespace mcts {

// New optimized search implementation using burst batch collection
void MCTSEngine::executeOptimizedSearchV2(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    auto search_start_time = std::chrono::steady_clock::now();
    
    if (search_roots.empty()) {
        std::cout << "❌ No search roots provided" << std::endl;
        return;
    }
    
    std::shared_ptr<MCTSNode> main_root = search_roots[0];
    if (!main_root) {
        std::cout << "❌ Main root is null" << std::endl;
        return;
    }
    
    
    // Initialize burst batch collector with aggressive settings
    auto burst_collector = std::make_unique<BurstBatchCollector>(
        std::max(size_t(64), static_cast<size_t>(settings_.batch_size)),  // Minimum 64 items per batch
        std::chrono::milliseconds(15)  // Short burst timeout for responsiveness
    );
    
    burst_collector->start();
    
    // Initialize unified memory manager
    auto& memory_manager = UnifiedMemoryManager::getInstance();
    
    // Expand root if not already expanded
    if (!main_root->isTerminal() && !main_root->isFullyExpanded()) {
        expandNonTerminalLeaf(main_root);
    }
    
    // Set up simulation tracking
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        active_simulations_.store(settings_.num_simulations, std::memory_order_release);
    }
    
    std::cout << "🎯 Starting search with " << active_simulations_.load() 
              << " simulations, targeting batches of " << settings_.batch_size << std::endl;
    
    int main_loop_iterations = 0;
    int total_evaluations_requested = 0;
    int total_evaluations_completed = 0;
    
    // Main search loop with burst collection
    while (active_simulations_.load(std::memory_order_acquire) > 0 && 
           main_loop_iterations < settings_.num_simulations * 2) {  // Safety limit
           
        main_loop_iterations++;
        
        // Phase 1: Aggressive leaf generation for burst collection
        int simulations_to_generate = std::min(
            settings_.batch_size,  // Generate up to a full batch per iteration
            active_simulations_.load(std::memory_order_acquire)
        );
        
        if (simulations_to_generate <= 0) {
            break;
        }
        
        // Claim simulations
        int old_sims = active_simulations_.load(std::memory_order_acquire);
        bool claimed = active_simulations_.compare_exchange_weak(
            old_sims, old_sims - simulations_to_generate, std::memory_order_acq_rel);
            
        if (!claimed) {
            continue;
        }
        
        // Generate evaluations rapidly
        int leaves_generated = 0;
        for (int i = 0; i < simulations_to_generate; ++i) {
            try {
                // Select leaf node for evaluation
                auto [leaf, path] = selectLeafNode(main_root);
                
                if (!leaf) {
                    continue;
                }
                
                if (leaf->isTerminal()) {
                    // Handle terminal nodes immediately
                    float value = 0.0f;
                    try {
                        auto game_result = leaf->getState().getGameResult();
                        int current_player = leaf->getState().getCurrentPlayer();
                        if (game_result == core::GameResult::WIN_PLAYER1) {
                            value = current_player == 1 ? 1.0f : -1.0f;
                        } else if (game_result == core::GameResult::WIN_PLAYER2) {
                            value = current_player == 2 ? 1.0f : -1.0f;
                        }
                    } catch (...) {
                        value = 0.0f;
                    }
                    backPropagate(path, value);
                } else {
                    // Request neural network evaluation
                    if (safelyMarkNodeForEvaluation(leaf)) {
                        // Clone state using unified memory manager
                        auto state_clone = memory_manager.cloneGameState(leaf->getState());
                        
                        if (state_clone) {
                            // Create pending evaluation
                            PendingEvaluation pending;
                            pending.node = leaf;
                            pending.path = path;
                            pending.state = state_clone;
                            pending.batch_id = batch_counter_.fetch_add(1, std::memory_order_relaxed);
                            pending.request_id = total_leaves_generated_.fetch_add(1, std::memory_order_relaxed);
                            
                            // Submit to burst collector (non-blocking)
                            burst_collector->submitEvaluation(std::move(pending));
                            leaves_generated++;
                            total_evaluations_requested++;
                        } else {
                            leaf->clearEvaluationFlag();
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error generating leaf: " << e.what() << std::endl;
            }
        }
        
        // Phase 2: Process completed batches
        while (burst_collector->hasPendingBatch()) {
            auto batch = burst_collector->collectBatch();
            if (!batch.empty()) {
                std::cout << "🔥 Processing burst batch of size " << batch.size() << std::endl;
                
                // Process batch with evaluator
                if (processBatchWithEvaluator(batch)) {
                    total_evaluations_completed += batch.size();
                }
            }
        }
        
        // Phase 3: Memory management
        if (main_loop_iterations % 50 == 0) {
            memory_manager.cleanup();
            
            // Print progress
            float completion = 1.0f - (static_cast<float>(active_simulations_.load()) / settings_.num_simulations);
            std::cout << "📊 Progress: " << (completion * 100.0f) << "%, "
                      << "Evaluations: " << total_evaluations_completed << "/" << total_evaluations_requested
                      << ", Avg batch: " << burst_collector->getAverageBatchSize()
                      << ", Memory: " << (memory_manager.getCurrentMemoryUsage() / 1048576) << " MB" << std::endl;
        }
        
        // Phase 4: Adaptive waiting
        if (leaves_generated == 0 && !burst_collector->hasPendingBatch()) {
            // If no progress, wait briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    // Wait for remaining evaluations to complete
    std::cout << "⏳ Waiting for remaining evaluations to complete..." << std::endl;
    
    auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (burst_collector->hasPendingBatch() && 
           std::chrono::steady_clock::now() < wait_deadline) {
        
        auto batch = burst_collector->collectBatch();
        if (!batch.empty()) {
            processBatchWithEvaluator(batch);
            total_evaluations_completed += batch.size();
        }
    }
    
    // Shutdown burst collector
    burst_collector->shutdown();
    
    // Calculate final statistics
    auto search_end_time = std::chrono::steady_clock::now();
    last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end_time - search_start_time);
    
    std::cout << "✅ Optimized search v2 completed:" << std::endl;
    std::cout << "   ⏱️  Search time: " << last_stats_.search_time.count() << "ms" << std::endl;
    std::cout << "   🎯 Evaluations: " << total_evaluations_completed << "/" << total_evaluations_requested << std::endl;
    std::cout << "   📦 Average batch size: " << burst_collector->getAverageBatchSize() << std::endl;
    std::cout << "   💾 Peak memory: " << (memory_manager.getPeakMemoryUsage() / 1048576) << " MB" << std::endl;
    std::cout << "   🔄 Memory efficiency: " << (memory_manager.getMemoryEfficiency() * 100.0f) << "%" << std::endl;
    
    // Update evaluator stats for compatibility
    if (evaluator_) {
        last_stats_.avg_batch_size = burst_collector->getAverageBatchSize();
        last_stats_.total_evaluations = total_evaluations_completed;
    }
    
    search_running_.store(false, std::memory_order_release);
}

bool MCTSEngine::processBatchWithEvaluator(const std::vector<PendingEvaluation>& batch) {
    if (batch.empty() || !evaluator_) {
        return false;
    }
    
    try {
        // Extract states for inference
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.reserve(batch.size());
        
        for (const auto& eval : batch) {
            if (eval.state) {
                auto state_clone = eval.state->clone();
                if (state_clone) {
                    states.push_back(std::move(state_clone));
                }
            }
        }
        
        if (states.empty()) {
            return false;
        }
        
        // Perform batch inference
        auto results = evaluator_->getInferenceFunction()(states);
        
        if (results.size() != states.size()) {
            std::cerr << "❌ Inference result size mismatch" << std::endl;
            return false;
        }
        
        // Apply results to nodes
        for (size_t i = 0; i < std::min(results.size(), batch.size()); ++i) {
            const auto& eval = batch[i];
            if (eval.node && i < results.size()) {
                try {
                    // Apply network output to node
                    auto& output = results[i];
                    
                    // Set prior probabilities if available
                    if (!output.policy.empty()) {
                        eval.node->setPriorProbabilities(output.policy);
                    }
                    
                    // Clear evaluation flags
                    eval.node->clearEvaluationFlag();
                    eval.node->clearAllEvaluationFlags();
                    
                    // Backpropagate value
                    auto path_copy = eval.path;  // Copy path for backpropagation
                    backPropagate(path_copy, output.value);
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error applying network result: " << e.what() << std::endl;
                    if (eval.node) {
                        eval.node->clearAllEvaluationFlags();
                    }
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing batch: " << e.what() << std::endl;
        
        // Clear evaluation flags on error
        for (const auto& eval : batch) {
            if (eval.node) {
                eval.node->clearEvaluationFlag();
            }
        }
        
        return false;
    }
}

} // namespace mcts
} // namespace alphazero