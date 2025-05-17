#include "mcts/mcts_taskflow_engine.h"
#include "utils/logger.h"
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace alphazero {
namespace mcts {

// Thread-local random generator
thread_local std::mt19937 MCTSTaskflowEngine::thread_local_gen_{std::random_device{}()};

MCTSTaskflowEngine::MCTSTaskflowEngine(MCTSSettings settings,
                                       std::unique_ptr<nn::NeuralNetwork> nn_model)
    : executor_(settings.num_threads),
      settings_(settings),
      tt_(std::make_unique<TranspositionTable>()),
      node_tracker_(std::make_unique<NodeTracker>()),
      node_pool_(std::make_unique<MCTSNodePool>()),
      leaf_queue_(std::make_unique<moodycamel::ConcurrentQueue<MCTSEngine::PendingEvaluation>>()),
      result_queue_(std::make_unique<moodycamel::ConcurrentQueue<mcts::NetworkOutput>>()) {
    
    // Create evaluator with shared_ptr to allow copying in lambda
    auto nn_shared = std::shared_ptr<nn::NeuralNetwork>(std::move(nn_model));
    evaluator_ = std::make_unique<MCTSEvaluator>(
          [nn_shared](const std::vector<std::unique_ptr<core::IGameState>>& states) -> std::vector<mcts::NetworkOutput> {
              return nn_shared->inference(states);
          },
          settings.batch_size, 
          settings.batch_timeout);
    
    // Start the evaluator with external queues
    evaluator_->setExternalQueues(leaf_queue_.get(), result_queue_.get());
    
    LOG_MCTS_INFO("Created TaskflowEngine with {} worker threads", settings.num_threads);
}

MCTSTaskflowEngine::~MCTSTaskflowEngine() {
    shutdown();
}

void MCTSTaskflowEngine::shutdown() {
    bool expected = false;
    if (shutdown_.compare_exchange_strong(expected, true)) {
        LOG_MCTS_INFO("Shutting down TaskflowEngine...");
        
        // Stop search if running
        search_running_ = false;
        
        // Wait for all tasks to complete
        executor_.wait_for_all();
        
        // Shutdown evaluator
        if (evaluator_) {
            evaluator_->stop();
        }
        
        // Clear the taskflow
        taskflow_.clear();
        
        LOG_MCTS_INFO("TaskflowEngine shutdown complete");
    }
}

std::shared_ptr<MCTSNode> MCTSTaskflowEngine::runSearch(
    std::unique_ptr<core::IGameState> root_state,
    int num_simulations) {
    
    // Reset search state
    shutdown_ = false;
    search_running_ = true;
    
    LOG_MCTS_INFO("Starting search with {} simulations on {} threads", 
                  num_simulations, settings_.num_threads);
    
    // Initialize root node using node pool
    root_ = node_pool_->allocateNode(std::move(root_state), nullptr);
    
    // Add Dirichlet noise to root if configured
    if (settings_.add_dirichlet_noise && !root_->isFullyExpanded()) {
        root_->expand();
        
        // Get number of children for noise generation
        auto& children = root_->getChildren();
        int num_actions = children.size();
        
        if (num_actions > 0) {
            std::gamma_distribution<float> gamma(settings_.dirichlet_alpha, 1.0);
            std::vector<float> noise(num_actions);
            std::vector<float> priors(num_actions);
            
            // Collect existing priors
            for (int i = 0; i < num_actions; ++i) {
                priors[i] = children[i]->getPriorProbability();
            }
            
            // Generate Dirichlet noise
            float sum = 0.0f;
            for (auto& n : noise) {
                n = gamma(thread_local_gen_);
                sum += n;
            }
            for (auto& n : noise) {
                n /= sum;
            }
            
            // Mix priors with noise
            for (size_t i = 0; i < priors.size(); ++i) {
                priors[i] = (1.0f - settings_.dirichlet_epsilon) * priors[i] + 
                          settings_.dirichlet_epsilon * noise[i];
            }
            
            // Apply back to children
            for (size_t i = 0; i < children.size(); ++i) {
                children[i]->setPriorProbability(priors[i]);
            }
        }
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Create taskflow graph for MCTS search
    buildSearchTaskGraph();
    
    // Run simulation batches using Taskflow
    int completed_simulations = 0;
    const int batch_size = std::min(settings_.max_concurrent_simulations, 
                                   num_simulations / 4); // Dynamic batch sizing
    
    while (completed_simulations < num_simulations && search_running_) {
        int remaining = num_simulations - completed_simulations;
        int current_batch = std::min(batch_size, remaining);
        
        // Submit a batch of simulations
        submitSimulationBatch(current_batch);
        
        // Wait for batch completion
        executor_.run(taskflow_).wait();
        
        completed_simulations += current_batch;
        
        // Optional: Print progress
        if (completed_simulations % 100 == 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            float nps = completed_simulations * 1000.0f / ms;
            LOG_MCTS_DEBUG("Progress: {}/{} simulations, {:.1f} sims/sec", 
                          completed_simulations, num_simulations, nps);
        }
    }
    
    // Wait for all remaining evaluations to complete
    while (pending_evaluations_ > 0 || active_simulations_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    float nps = completed_simulations * 1000.0f / duration;
    LOG_MCTS_INFO("Search complete: simulations={}, batch_size={}, nps={:.0f}, depth={}, value={:.3f}", 
                  completed_simulations, settings_.batch_size, nps, root_->getDepth(), root_->getValue());
    
    // Profile metrics
    utils::ProfileMCTSSimulation(completed_simulations);
    utils::ProfileNodesPerSecond(nps);
    utils::ProfileTreeDepth(root_->getDepth());
    
    search_running_ = false;
    return root_;
}

void MCTSTaskflowEngine::buildSearchTaskGraph() {
    taskflow_.clear();
    
    // Create a task for processing results
    auto result_processor = taskflow_.emplace([this]() {
        processResultsTask();
    });
    
    // The main simulation tasks will be added dynamically via submitSimulationBatch
}

void MCTSTaskflowEngine::submitSimulationBatch(int batch_size) {
    // Clear previous batch tasks
    taskflow_.clear();
    
    // Create parallel simulation tasks using Taskflow's parallel_for
    taskflow_.for_each_index(0, batch_size, 1, [this](int idx) {
        treeTraversalTask();
    });
    
    // Create a task to process results after simulations
    auto result_processor = taskflow_.emplace([this, batch_size]() {
        // Process results for this batch
        int processed = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        
        while (processed < batch_size && 
               std::chrono::steady_clock::now() < timeout) {
            mcts::NetworkOutput result;
            if (result_queue_->try_dequeue(result)) {
                // Process the network output result
                pending_evaluations_--;
                processed++;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    });
    
    // Connect the tasks
    auto sim_task = taskflow_.composed_of(taskflow_);
    result_processor.succeed(sim_task);
}

void MCTSTaskflowEngine::treeTraversalTask() {
    active_simulations_++;
    
    // Path from root to leaf
    std::vector<std::shared_ptr<MCTSNode>> path;
    std::shared_ptr<MCTSNode> current = root_;
    path.push_back(current);
    
    // Tree traversal with virtual loss
    while (!current->isLeaf()) {
        current->addVirtualLoss();
        current = current->selectChild(settings_.exploration_constant, 
                                     settings_.use_rave, 
                                     settings_.rave_constant);
        path.push_back(current);
    }
    
    // Apply virtual loss to leaf
    current->addVirtualLoss();
    
    // Expand node if not terminal and not already expanded
    if (!current->isTerminal() && !current->isFullyExpanded()) {
        {
            std::lock_guard<std::mutex> lock(current->getExpansionMutex());
            if (!current->isFullyExpanded()) {
                current->expand();
            }
        }
    }
    
    // Handle terminal nodes
    if (current->isTerminal()) {
        float terminal_value = current->getTerminalValue();
        int current_player = current->getPlayer();
        
        // Immediate backpropagation for terminal nodes
        for (auto& node : path) {
            float value = (node->getPlayer() == current_player) ? 
                         terminal_value : -terminal_value;
            node->update(value);
            node->removeVirtualLoss();
        }
        active_simulations_--;
        return;
    }
    
    // Queue for neural network evaluation
    if (current->needsEvaluation()) {
        current->markEvaluationInProgress();
        
        MCTSEngine::PendingEvaluation pending;
        pending.state = current->getState().clone();
        pending.node = current;
        pending.path = std::move(path);
        
        leaf_queue_->enqueue(std::move(pending));
        pending_evaluations_++;
    }
    
    active_simulations_--;
}

void MCTSTaskflowEngine::processResultsTask() {
    // Results are now processed inline in submitSimulationBatch
    // This method is kept for compatibility but could be removed
}

std::shared_ptr<MCTSNode> MCTSTaskflowEngine::selectBestChild(
    std::shared_ptr<MCTSNode> parent) {
    // Temperature-based move selection
    if (settings_.temperature > 0) {
        std::vector<float> probs;
        std::vector<std::shared_ptr<MCTSNode>> children;
        
        for (auto& child : parent->getChildren()) {
            children.push_back(child);
            float visits = static_cast<float>(child->getVisitCount());
            probs.push_back(std::pow(visits, 1.0f / settings_.temperature));
        }
        
        // Normalize probabilities
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (auto& p : probs) {
            p /= sum;
        }
        
        // Sample from distribution
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return children[dist(thread_local_gen_)];
    } else {
        // Greedy selection (temperature = 0)
        return parent->getMostVisitedChild();
    }
}

void MCTSTaskflowEngine::reset() {
    LOG_MCTS_INFO("Resetting TaskflowEngine...");
    
    // Stop any ongoing search
    search_running_ = false;
    
    // Wait for tasks to complete
    executor_.wait_for_all();
    
    // Clear taskflow
    taskflow_.clear();
    
    // Reset root and clear queues
    root_.reset();
    
    // Clear queues by consuming all items
    MCTSEngine::PendingEvaluation pending;
    while (leaf_queue_->try_dequeue(pending)) {}
    
    mcts::NetworkOutput result;
    while (result_queue_->try_dequeue(result)) {}
    
    // Reset counters
    active_simulations_ = 0;
    pending_evaluations_ = 0;
    
    // Optionally reset transposition table
    if (tt_) {
        tt_->clear();
    }
    
    LOG_MCTS_INFO("Reset complete");
}

} // namespace mcts
} // namespace alphazero