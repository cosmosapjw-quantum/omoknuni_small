#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include <iostream>
#include <random>
#include <numeric>

namespace alphazero {
namespace mcts {

// Add Dirichlet noise to root node policy for exploration
void MCTSEngine::addDirichletNoise(std::shared_ptr<MCTSNode> root) {
    if (!root) {
        return;
    }
    
    // Expand root node if it's not already expanded
    if (root->isLeaf() && !root->isTerminal()) {
        root->expand(settings_.use_progressive_widening,
                   settings_.progressive_widening_c,
                   settings_.progressive_widening_k);
        
        if (root->getChildren().empty()) {
            return;  // No children to add noise to
        }
        
        // Get prior probabilities for the root node
        try {
            auto state_clone = cloneGameState(root->getState());
            if (settings_.num_threads == 0) {
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.push_back(std::unique_ptr<core::IGameState>(state_clone->clone().release()));
                // TODO: Replace with UnifiedInferenceServer synchronous call
                std::vector<NetworkOutput> outputs;
                // Create default outputs since inference server was removed
                outputs.resize(1);
                outputs[0].policy.resize(root->getState().getActionSpaceSize(), 1.0f / root->getState().getActionSpaceSize());
                outputs[0].value = 0.0f;
                if (!outputs.empty()) {
                    root->setPriorProbabilities(outputs[0].policy);
                } else {
                    int action_space_size = root->getState().getActionSpaceSize();
                    root->setPriorProbabilities(createDefaultPolicy(action_space_size));
                }
            } else {
                // Convert shared_ptr to unique_ptr for evaluator
                auto unique_clone = std::unique_ptr<core::IGameState>(state_clone->clone().release());
                // TODO: Replace with UnifiedInferenceServer async call
                // For now, create a completed future with default values
                std::promise<NetworkOutput> promise;
                NetworkOutput default_output;
                default_output.policy.resize(root->getState().getActionSpaceSize(), 1.0f / root->getState().getActionSpaceSize());
                default_output.value = 0.0f;
                promise.set_value(default_output);
                auto future = promise.get_future();
                auto status = future.wait_for(std::chrono::seconds(2));
                if (status == std::future_status::ready) {
                    auto result = future.get();
                    root->setPriorProbabilities(result.policy);
                } else {
                    // Timed out, use uniform policy
                    int action_space_size = root->getState().getActionSpaceSize();
                    root->setPriorProbabilities(createDefaultPolicy(action_space_size));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception getting prior probabilities for root: " << e.what() << std::endl;
            
            // On error, use uniform policy
            int action_space_size = root->getState().getActionSpaceSize();
            root->setPriorProbabilities(createDefaultPolicy(action_space_size));
        }
    
    if (root->getChildren().empty()) {
        return;  // No children to add noise to
    }
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(settings_.dirichlet_alpha, 1.0f);
    std::vector<float> noise;
    noise.reserve(root->getChildren().size());
    
    for (size_t i = 0; i < root->getChildren().size(); ++i) {
        noise.push_back(gamma(random_engine_));
    }
    
    // Normalize noise
    float sum = std::accumulate(noise.begin(), noise.end(), 0.0f);
    if (sum > 0.0f) {
        for (auto& n : noise) {
            n /= sum;
        }
    } else {
        // If sum is zero, use uniform noise
        float uniform_noise = 1.0f / noise.size();
        std::fill(noise.begin(), noise.end(), uniform_noise);
    }
    
    // Apply noise to children's prior probabilities
    for (size_t i = 0; i < root->getChildren().size(); ++i) {
        std::shared_ptr<MCTSNode> child = root->getChildren()[i];
        float prior = child->getPriorProbability();
        float noisy_prior = (1.0f - settings_.dirichlet_epsilon) * prior + 
                          settings_.dirichlet_epsilon * noise[i];
        child->setPriorProbability(noisy_prior);
    }
}
}  // End of addDirichletNoise function

// Process results from pending evaluations
void MCTSEngine::processEvaluationResults() {
    // This method handles evaluation results when using shared queues
    if (!use_shared_queues_ || !shared_result_queue_) {
        return;
    }
    
    std::pair<NetworkOutput, PendingEvaluation> result;
    int processed = 0;
    
    // Process all available results
    while (shared_result_queue_->try_dequeue(result)) {
        auto& output = result.first;
        auto& eval = result.second;
        
        if (eval.node) {
            try {
                eval.node->setPriorProbabilities(output.policy);
                backPropagate(eval.path, output.value);
                eval.node->clearEvaluationFlag();
                processed++;
            } catch (const std::exception& e) {
                std::cerr << "ERROR processing evaluation result: " << e.what() << std::endl;
                // Try to clean up even if processing fails
                try { eval.node->clearEvaluationFlag(); } catch (...) {}
            } catch (...) {
                std::cerr << "ERROR: Unknown exception processing evaluation result" << std::endl;
                // Try to clean up even if processing fails
                try { eval.node->clearEvaluationFlag(); } catch (...) {}
            }
        }
        
        pending_evaluations_.fetch_sub(1, std::memory_order_acq_rel);
    }
    
    if (processed > 0) {
        std::cout << "MCTSEngine::processEvaluationResults - Processed " << processed << " results" << std::endl;
    }
}

// Expand and evaluate a node
float MCTSEngine::expandAndEvaluate(std::shared_ptr<MCTSNode> leaf, const std::vector<std::shared_ptr<MCTSNode>>& path) {
    if (!leaf) {
        std::cerr << "MCTSEngine::expandAndEvaluate - Called with NULL leaf!" << std::endl;
        return 0.0f;
    }

    // Get diagnostic information about the leaf state
    uint64_t leaf_state_hash = 0;
    try {
        leaf_state_hash = leaf->getState().getHash();
    } catch (...) {
        std::cerr << "MCTSEngine::expandAndEvaluate - Could not get hash for logging" << std::endl;
    }

    // Handle terminal states
    if (leaf->isTerminal()) {
        try {
            auto result = leaf->getState().getGameResult();
            float value = 0.0f;
            if (result == core::GameResult::WIN_PLAYER1) {
                value = leaf->getState().getCurrentPlayer() == 1 ? 1.0f : -1.0f;
            } else if (result == core::GameResult::WIN_PLAYER2) {
                value = leaf->getState().getCurrentPlayer() == 2 ? 1.0f : -1.0f;
            }
            return value;
        } catch (const std::exception& e) {
            std::cerr << "Error evaluating terminal state: " << e.what() << std::endl;
            return 0.0f;
        }
    }
    
    // Expand the leaf node
    try {
        leaf->expand(settings_.use_progressive_widening, 
                    settings_.progressive_widening_c, 
                    settings_.progressive_widening_k);
        
        // Store in transposition table if enabled
        if (use_transposition_table_ && transposition_table_) {
            try {
                transposition_table_->store(leaf_state_hash, std::weak_ptr<MCTSNode>(leaf), path.size());
            } catch (const std::exception& e) {
                std::cerr << "Transposition table store failed: " << e.what() << std::endl;
            }
        }
        
        if (leaf->getChildren().empty()) {
            return 0.0f; 
        }
        
        // Evaluate with the neural network
        if (settings_.num_threads == 0) { // SERIAL MODE
            std::cout << "🔍 MCTSEngine::expandAndEvaluate - Using SERIAL MODE for evaluation" << std::endl;
            auto state_clone_serial = cloneGameState(leaf->getState());
            if (!state_clone_serial) {
                throw std::runtime_error("Failed to clone state for evaluation");
            }
            std::vector<std::unique_ptr<core::IGameState>> states_serial;
            states_serial.push_back(std::unique_ptr<core::IGameState>(state_clone_serial->clone().release()));
            
            // Direct neural network evaluation for synchronous mode
            std::vector<NetworkOutput> outputs;
            // Create default outputs since inference server was removed
            outputs.resize(1);
            outputs[0].policy.resize(leaf->getState().getActionSpaceSize(), 1.0f / leaf->getState().getActionSpaceSize());
            outputs[0].value = 0.0f;
            
            if (!outputs.empty()) {
                leaf->setPriorProbabilities(outputs[0].policy);
                return outputs[0].value;
            } else {
                int action_space_size = leaf->getState().getActionSpaceSize();
                leaf->setPriorProbabilities(createDefaultPolicy(action_space_size));
                return 0.0f;
            }
        } else { // PARALLEL MODE (queue for evaluation)
            static std::atomic<int> eval_attempt_counter{0};
            static std::atomic<int> eval_success_counter{0};
            static std::atomic<int> eval_failure_counter{0};
            
            int attempt_id = eval_attempt_counter.fetch_add(1, std::memory_order_relaxed);
            bool log_detail = (attempt_id < 10 || attempt_id % 20 == 0); // More frequent logging for early attempts
            
            if (log_detail) {
                std::cout << "🔍 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                         << "] PARALLEL MODE using " << (use_shared_queues_ ? "shared" : "local") << " queues" 
                         << ", success_rate: " << (attempt_id > 0 ? (eval_success_counter.load() * 100.0 / attempt_id) : 0.0) << "%" 
                         << ", fail_rate: " << (attempt_id > 0 ? (eval_failure_counter.load() * 100.0 / attempt_id) : 0.0) << "%" 
                         << std::endl;
            }
            
            bool can_evaluate = safelyMarkNodeForEvaluation(leaf);
            
            if (log_detail) {
                std::cout << "🔍 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                         << "] Can evaluate node: " << (can_evaluate ? "YES" : "NO") 
                         << ", leaf node addr: " << leaf.get()
                         << ", isBeingEvalulated: " << (leaf->isBeingEvaluated() ? "yes" : "no")
                         << ", hasPendingEvaluation: " << (leaf->hasPendingEvaluation() ? "yes" : "no")
                         << std::endl;
            }
            
            if (can_evaluate) {
                auto state_clone_parallel = cloneGameState(leaf->getState());
                if (!state_clone_parallel) {
                    leaf->clearEvaluationFlag();
                    throw std::runtime_error("Failed to clone state for evaluation");
                }
                
                // Create PendingEvaluation with node and path
                PendingEvaluation pending;
                pending.node = leaf;
                pending.path = path;
                
                // Move the shared_ptr to the state
                pending.state = convertToUniquePtr(state_clone_parallel);
                
                bool queue_success = false;
                
                // Enqueue for evaluation using shared queue for proper batch accumulation
                if (use_shared_queues_ && shared_leaf_queue_) {
                    // Use shared queue for proper batch collection
                    // The BatchAccumulator works best when batchCollectorLoop can collect multiple 
                    // evaluations at once from the shared queue, rather than receiving them one by one
                    for (int attempt = 0; attempt < 3 && !queue_success; attempt++) {
                        if (log_detail) {
                            std::cout << "📩 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                     << "] Attempt " << attempt << " to enqueue to shared queue" << std::endl;
                        }
                        
                        if (attempt > 0) {
                            // Re-create pending for move if previous attempt failed
                            pending.node = leaf;
                            pending.path = path;
                            
                            // Clone the state and convert to unique_ptr
                            auto state_clone = cloneGameState(leaf->getState());
                            if (!state_clone) {
                                leaf->clearEvaluationFlag();
                                throw std::runtime_error("Re-clone failed for enqueue retry");
                            }
                            pending.state = convertToUniquePtr(state_clone);
                        }
                        
                        queue_success = shared_leaf_queue_->enqueue(std::move(pending));
                        
                        if (log_detail) {
                            std::cout << "📩 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                     << "] Shared queue enqueue " << (queue_success ? "SUCCEEDED" : "FAILED") << std::endl;
                        }
                        
                        if (!queue_success && attempt < 2) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                    
                    if (queue_success && external_queue_notify_fn_) {
                        if (log_detail) {
                            std::cout << "🔔 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                     << "] Calling external queue notify function" << std::endl;
                        }
                        external_queue_notify_fn_();
                    }
                } else {
                    if (log_detail) {
                        std::cout << "📩 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                 << "] Enqueueing to local leaf queue" << std::endl;
                    }
                    
                    queue_success = leaf_queue_.enqueue(std::move(pending));
                    
                    if (log_detail) {
                        std::cout << "📩 MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                 << "] Local queue enqueue " << (queue_success ? "SUCCEEDED" : "FAILED") << std::endl;
                    }
                    
                    // UnifiedInferenceServer was removed in simplification
                }
                
                if (!queue_success) {
                    leaf->clearEvaluationFlag();
                    eval_failure_counter.fetch_add(1, std::memory_order_relaxed);
                    
                    if (log_detail) {
                        std::cout << "❌ MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                 << "] FAILED to enqueue leaf for evaluation, cleared flag" << std::endl;
                    }
                } else {
                    pending_evaluations_.fetch_add(1, std::memory_order_acq_rel);
                    leaf->applyVirtualLoss(settings_.virtual_loss);
                    eval_success_counter.fetch_add(1, std::memory_order_relaxed);
                    
                    if (log_detail) {
                        std::cout << "✅ MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                                 << "] Successfully enqueued leaf for evaluation, applied virtual loss" << std::endl;
                    }
                }
            } else {
                leaf->applyVirtualLoss(settings_.virtual_loss);
                
                if (log_detail) {
                    std::cout << "⚠️ MCTSEngine::expandAndEvaluate - [#" << attempt_id 
                             << "] Node cannot be evaluated (already being evaluated), applied virtual loss" << std::endl;
                }
            }
            return 0.0f; 
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error during expansion/evaluation" << std::endl;
        // Try to clean up the node's evaluation flag
        if (leaf && leaf->isBeingEvaluated()) {
            leaf->clearEvaluationFlag();
        }
        
        // On error, try to set a fallback policy
        try {
            if (leaf) {
                int action_space_size = leaf->getState().getActionSpaceSize();
                leaf->setPriorProbabilities(createDefaultPolicy(action_space_size));
            }
        } catch (...) {}
        
        return 0.0f;
    } catch (const std::exception& e) {
        std::cerr << "Error during expansion/evaluation: " << e.what() << std::endl;
        // Try to clean up the node's evaluation flag
        if (leaf && leaf->isBeingEvaluated()) {
            leaf->clearEvaluationFlag();
        }
        
        // On error, try to set a fallback policy
        try {
            if (leaf) {
                int action_space_size = leaf->getState().getActionSpaceSize();
                leaf->setPriorProbabilities(createDefaultPolicy(action_space_size));
            }
        } catch (...) {}
        
        return 0.0f;
    }
}

} // namespace mcts
} // namespace alphazero