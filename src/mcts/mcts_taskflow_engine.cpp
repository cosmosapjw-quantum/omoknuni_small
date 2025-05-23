#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/unified_inference_server.h"
#include "mcts/burst_coordinator.h"
#include "mcts/mcts_object_pool.h"
#include "utils/logger.h"
#include <taskflow/taskflow.hpp>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <future>
#include <chrono>

namespace alphazero {
namespace mcts {

/**
 * Advanced Taskflow-based MCTS Engine using the new UnifiedInferenceServer + BurstCoordinator architecture
 * Leverages Taskflow's sophisticated task graph execution for maximum parallelism
 */
class MCTSTaskflowEngine {
private:
    // Core architecture components
    std::shared_ptr<nn::NeuralNetwork> neural_network_;
    std::shared_ptr<UnifiedInferenceServer> unified_server_shared_;
    std::unique_ptr<BurstCoordinator> burst_coordinator_;
    // Using singleton MCTSObjectPoolManager instead of instance member
    std::unique_ptr<TranspositionTable> transposition_table_;
    
    // Taskflow execution infrastructure
    tf::Executor executor_;
    tf::Taskflow taskflow_;
    
    // Configuration and state
    MCTSSettings settings_;
    std::shared_ptr<MCTSNode> root_node_;
    
    // Performance tracking
    mutable std::mutex stats_mutex_;
    struct TaskflowStats {
        std::atomic<size_t> total_tasks_executed{0};
        std::atomic<size_t> total_batches_processed{0};
        std::atomic<double> average_task_latency{0.0};
        std::atomic<double> parallel_efficiency{0.0};
        std::chrono::steady_clock::time_point start_time;
    } stats_;

public:
    MCTSTaskflowEngine(const MCTSSettings& settings, 
                       std::shared_ptr<nn::NeuralNetwork> network)
        : neural_network_(std::move(network))
        , executor_(settings.num_threads)
        , settings_(settings) {
        
        if (!neural_network_) {
            throw std::invalid_argument("Neural network cannot be null");
        }
        
        initializeComponents();
        stats_.start_time = std::chrono::steady_clock::now();
    }
    
    ~MCTSTaskflowEngine() {
        shutdown();
    }
    
    SearchResult search(const core::IGameState& root_state) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Initialize root node and expand using singleton object pool manager
        auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
        auto* raw_root_node = object_pool_manager.getNodePool().acquire();
        root_node_ = std::shared_ptr<MCTSNode>(raw_root_node, [&object_pool_manager](MCTSNode* node) {
            object_pool_manager.getNodePool().release(node);
        });
        if (!root_state.isTerminal()) {
            root_node_->expand(false, 
                              settings_.progressive_widening_c,
                              settings_.progressive_widening_k);
        }
        
        // Create sophisticated taskflow graph
        createOptimizedTaskflowGraph(root_state);
        
        // Execute taskflow with advanced parallelization
        auto taskflow_future = executor_.run(taskflow_);
        taskflow_future.wait();
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Generate comprehensive result
        return generateSearchResult(total_time);
    }
    
    void shutdown() {
        if (unified_server_shared_) {
            unified_server_shared_->stop();
        }
        executor_.wait_for_all();
    }
    
    // Performance analytics specific to taskflow execution
    struct TaskflowAnalytics {
        double parallel_efficiency;
        double task_scheduling_overhead;
        size_t total_tasks_executed;
        double average_task_latency_ms;
        double taskflow_throughput;
    };
    
    TaskflowAnalytics getTaskflowAnalytics() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - stats_.start_time).count();
        
        TaskflowAnalytics analytics;
        analytics.parallel_efficiency = stats_.parallel_efficiency.load();
        analytics.total_tasks_executed = stats_.total_tasks_executed.load();
        analytics.average_task_latency_ms = stats_.average_task_latency.load();
        
        if (elapsed > 0.0) {
            analytics.taskflow_throughput = analytics.total_tasks_executed / elapsed;
        } else {
            analytics.taskflow_throughput = 0.0;
        }
        
        // Estimate scheduling overhead
        double theoretical_min_time = analytics.average_task_latency_ms * analytics.total_tasks_executed / settings_.num_threads;
        double actual_time = elapsed * 1000.0; // Convert to ms
        analytics.task_scheduling_overhead = (actual_time - theoretical_min_time) / actual_time;
        
        return analytics;
    }

private:
    void initializeComponents() {
        // Initialize UnifiedInferenceServer with optimized configuration
        UnifiedInferenceServer::ServerConfig server_config;
        server_config.target_batch_size = static_cast<size_t>(settings_.batch_size);
        server_config.min_batch_size = static_cast<size_t>(std::max(4, settings_.batch_size / 8));
        server_config.max_batch_size = static_cast<size_t>(settings_.batch_size * 4);
        server_config.max_batch_wait = std::chrono::milliseconds(static_cast<int>(settings_.batch_timeout.count()));
        server_config.min_batch_wait = std::chrono::milliseconds(1);
        server_config.num_worker_threads = static_cast<size_t>(std::min(8, settings_.num_threads / 3));
        
        unified_server_shared_ = std::make_shared<UnifiedInferenceServer>(neural_network_, server_config);
        unified_server_shared_->start();
        
        // Initialize BurstCoordinator for taskflow integration
        BurstCoordinator::BurstConfig burst_config;
        burst_config.target_burst_size = static_cast<size_t>(settings_.batch_size);
        burst_config.min_burst_size = static_cast<size_t>(std::max(4, settings_.batch_size / 8));
        burst_config.collection_timeout = std::chrono::milliseconds(std::max(3, static_cast<int>(settings_.batch_timeout.count()) / 6));
        burst_config.evaluation_timeout = std::chrono::milliseconds(static_cast<int>(settings_.batch_timeout.count()));
        burst_config.max_parallel_threads = static_cast<size_t>(settings_.num_threads);
        
        burst_coordinator_ = std::make_unique<BurstCoordinator>(unified_server_shared_, burst_config);
        
        // Use singleton object pool manager for memory optimization
        auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
        auto node_stats = object_pool_manager.getNodePool().getStats();
        std::cout << "MCTSTaskflowEngine using object pool (hit rate: " << node_stats.hit_rate << "%)" << std::endl;
        
        if (settings_.use_transposition_table) {
            transposition_table_ = std::make_unique<TranspositionTable>(
                settings_.transposition_table_size_mb);
        }
    }
    
    void createOptimizedTaskflowGraph(const core::IGameState& root_state) {
        taskflow_.clear();
        
        // Calculate optimal task distribution
        const int simulations_per_task = std::max(4, settings_.num_simulations / (settings_.num_threads * 4));
        const int num_tasks = (settings_.num_simulations + simulations_per_task - 1) / simulations_per_task;
        
        std::vector<tf::Task> simulation_tasks;
        simulation_tasks.reserve(num_tasks);
        
        // Create parallel simulation tasks
        for (int task_id = 0; task_id < num_tasks; ++task_id) {
            int task_simulations = std::min(simulations_per_task, 
                                           settings_.num_simulations - task_id * simulations_per_task);
            
            auto task = taskflow_.emplace([this, &root_state, task_simulations, task_id]() {
                executeTaskflowSimulations(root_state, task_simulations, task_id);
            });
            
            task.name("sim_task_" + std::to_string(task_id));
            simulation_tasks.push_back(task);
        }
        
        // Create advanced task dependencies for optimal execution
        createTaskDependencies(simulation_tasks);
        
        // Add specialized tasks for optimization
        addOptimizationTasks(simulation_tasks);
    }
    
    void createTaskDependencies(std::vector<tf::Task>& simulation_tasks) {
        // Create sophisticated dependency graph for optimal resource utilization
        
        // Phase 1: Initial exploration tasks (can run independently)
        size_t phase1_tasks = std::min(static_cast<size_t>(settings_.num_threads), simulation_tasks.size() / 3);
        
        // Phase 2: Intermediate tasks (depend on some phase 1 completion)
        size_t phase2_start = phase1_tasks;
        size_t phase2_tasks = std::min(static_cast<size_t>(settings_.num_threads * 2), 
                                      simulation_tasks.size() - phase2_start);
        
        // Create dependencies: Phase 2 tasks depend on subset of Phase 1
        for (size_t i = phase2_start; i < phase2_start + phase2_tasks && i < simulation_tasks.size(); ++i) {
            // Each phase 2 task depends on one phase 1 task
            size_t dependency_idx = (i - phase2_start) % phase1_tasks;
            simulation_tasks[i].succeed(simulation_tasks[dependency_idx]);
        }
        
        // Phase 3: Final exploitation tasks (depend on phase 2)
        size_t phase3_start = phase2_start + phase2_tasks;
        for (size_t i = phase3_start; i < simulation_tasks.size(); ++i) {
            // Each phase 3 task depends on corresponding phase 2 task
            if (phase2_start + ((i - phase3_start) % phase2_tasks) < simulation_tasks.size()) {
                size_t dependency_idx = phase2_start + ((i - phase3_start) % phase2_tasks);
                simulation_tasks[i].succeed(simulation_tasks[dependency_idx]);
            }
        }
    }
    
    void addOptimizationTasks(const std::vector<tf::Task>& simulation_tasks) {
        // Add performance monitoring task
        auto monitor_task = taskflow_.emplace([this]() {
            monitorTaskflowPerformance();
        });
        monitor_task.name("performance_monitor");
        
        // Add adaptive optimization task
        auto optimize_task = taskflow_.emplace([this]() {
            optimizeTaskflowConfiguration();
        });
        optimize_task.name("adaptive_optimizer");
        
        // Monitor task runs after first few simulation tasks
        if (simulation_tasks.size() >= 4) {
            monitor_task.succeed(simulation_tasks[simulation_tasks.size() / 4]);
        }
        
        // Optimization task runs after monitor and some simulations
        optimize_task.succeed(monitor_task);
        if (simulation_tasks.size() >= 2) {
            optimize_task.succeed(simulation_tasks[simulation_tasks.size() / 2]);
        }
    }
    
    void executeTaskflowSimulations(const core::IGameState& root_state, 
                                   int num_simulations, 
                                   int task_id) {
        auto task_start = std::chrono::steady_clock::now();
        
        std::vector<BurstCoordinator::BurstRequest> requests;
        requests.reserve(num_simulations);
        
        // Collect evaluation candidates using advanced tree traversal
        for (int sim = 0; sim < num_simulations; ++sim) {
            auto selected_leaf = selectOptimizedLeafWithTaskflow(root_node_, task_id);
            if (!selected_leaf.first) {
                break; // Tree exhausted
            }
            
            BurstCoordinator::BurstRequest request;
            request.leaf = selected_leaf.first;
            request.state = createGameStateForPath(root_state, selected_leaf.second);
            
            requests.push_back(std::move(request));
        }
        
        if (!requests.empty()) {
            // Execute coordinated burst evaluation
            auto results = burst_coordinator_->collectAndEvaluate(requests, requests.size());
            
            // Apply results with taskflow-optimized backpropagation
            applyTaskflowResults(requests, results, task_id);
        }
        
        auto task_end = std::chrono::steady_clock::now();
        auto task_duration = std::chrono::duration<double, std::milli>(task_end - task_start).count();
        
        // Update task performance metrics
        updateTaskMetrics(requests.size(), task_duration);
    }
    
    std::pair<std::shared_ptr<MCTSNode>, std::vector<std::shared_ptr<MCTSNode>>>
    selectOptimizedLeafWithTaskflow(std::shared_ptr<MCTSNode> root, int task_id) {
        std::vector<std::shared_ptr<MCTSNode>> path;
        auto current = root;
        
        // Add task-specific exploration bias for better parallelization
        float task_exploration_bias = 1.0f + (task_id % 4) * 0.1f;
        
        while (current && current->isExpanded() && !current->isTerminal()) {
            path.push_back(current);
            
            // Apply virtual loss with task-specific adjustment
            float adjusted_virtual_loss = settings_.virtual_loss * task_exploration_bias;
            current->applyVirtualLoss(adjusted_virtual_loss);
            
            // Select child with task-aware UCB
            auto next_child = current->selectChild(
                settings_.exploration_constant * task_exploration_bias,
                true, // use virtual loss
                adjusted_virtual_loss);
            
            if (!next_child) {
                current->removeVirtualLoss(adjusted_virtual_loss);
                break;
            }
            
            current = next_child;
        }
        
        if (current && (!current->isExpanded() || current->isTerminal())) {
            path.push_back(current);
            return {current, path};
        }
        
        return {nullptr, {}};
    }
    
    std::unique_ptr<core::IGameState> createGameStateForPath(
        const core::IGameState& root_state,
        const std::vector<std::shared_ptr<MCTSNode>>& path) {
        
        auto state = root_state.clone();
        
        // Apply moves along the path to reconstruct the game state
        for (size_t i = 1; i < path.size(); ++i) {
            if (path[i]->getAction() >= 0) {
                state->makeMove(path[i]->getAction());
            }
        }
        
        return state;
    }
    
    void applyTaskflowResults(const std::vector<BurstCoordinator::BurstRequest>& requests,
                             const std::vector<NetworkOutput>& results,
                             int task_id) {
        
        // Apply results with taskflow-specific optimizations
        for (size_t i = 0; i < results.size() && i < requests.size(); ++i) {
            auto& leaf = requests[i].leaf;
            auto& result = results[i];
            
            // Remove virtual loss (with task-specific adjustment)
            float task_bias = 1.0f + (task_id % 4) * 0.1f;
            leaf->removeVirtualLoss(settings_.virtual_loss * task_bias);
            
            // Apply evaluation result with enhanced backpropagation
            leaf->updateRecursive(result.value);
            
            // Expand node with taskflow considerations
            if (!leaf->isExpanded() && !requests[i].state->isTerminal()) {
                leaf->expand(false,
                           settings_.progressive_widening_c,
                           settings_.progressive_widening_k);
            }
            
            // Update transposition table if available
            if (transposition_table_ && requests[i].state) {
                updateTranspositionTable(*requests[i].state, result, leaf);
            }
        }
    }
    
    void updateTranspositionTable(const core::IGameState& state,
                                 const NetworkOutput& result,
                                 std::shared_ptr<MCTSNode> node) {
        if (!transposition_table_) return;
        
        auto hash = state.getHash();
        // TranspositionTable expects weak_ptr and uses its own store method
        transposition_table_->store(hash, std::weak_ptr<MCTSNode>(node), 0);
    }
    
    void monitorTaskflowPerformance() {
        // Monitor and log taskflow execution metrics
        auto analytics = getTaskflowAnalytics();
        
        // Log performance if needed
        if (analytics.parallel_efficiency < 0.6) {
            // Low parallel efficiency detected
            adjustTaskflowStrategy();
        }
    }
    
    void optimizeTaskflowConfiguration() {
        // Adaptive optimization of taskflow execution strategy
        auto server_stats = unified_server_shared_->getStats();
        auto burst_stats = burst_coordinator_->getEfficiencyStats();
        (void)burst_stats; // Mark as used to avoid warning
        
        // Adjust burst coordination based on taskflow performance
        if (server_stats.getAverageBatchSize() < settings_.batch_size * 0.7) {
            auto config = burst_coordinator_->getConfig();
            config.collection_timeout = std::chrono::milliseconds(std::max(1, static_cast<int>(config.collection_timeout.count()) - 1));
            burst_coordinator_->updateConfig(config);
        }
    }
    
    void adjustTaskflowStrategy() {
        // Implement adaptive taskflow strategy adjustments
        // This could involve changing task granularity, dependencies, etc.
    }
    
    void updateTaskMetrics(size_t requests_processed, double latency_ms) {
        stats_.total_tasks_executed.fetch_add(1);
        
        // Update average latency using exponential moving average
        double current_avg = stats_.average_task_latency.load();
        double alpha = 0.1;
        double new_avg = alpha * latency_ms + (1.0 - alpha) * current_avg;
        stats_.average_task_latency.store(new_avg);
        
        // Estimate parallel efficiency
        double theoretical_latency = latency_ms / settings_.num_threads;
        double efficiency = std::min(1.0, theoretical_latency / latency_ms);
        
        double current_efficiency = stats_.parallel_efficiency.load();
        double new_efficiency = alpha * efficiency + (1.0 - alpha) * current_efficiency;
        stats_.parallel_efficiency.store(new_efficiency);
    }
    
    SearchResult generateSearchResult(std::chrono::milliseconds total_time) {
        SearchResult result;
        
        if (!root_node_) {
            return result; // Empty result
        }
        
        // Generate action probabilities
        result.probabilities = generateActionProbabilities();
        result.value = root_node_->getValue();
        
        // Select best action
        result.action = selectBestAction();
        
        // Comprehensive statistics including taskflow metrics
        populateSearchStatistics(result, total_time);
        
        return result;
    }
    
    std::vector<float> generateActionProbabilities() {
        if (!root_node_ || root_node_->getChildren().empty()) {
            return {};
        }
        
        auto children = root_node_->getChildren();
        std::vector<float> probabilities;
        probabilities.reserve(children.size());
        
        // Calculate probabilities based on visit counts with temperature
        float total_visits = 0.0f;
        std::vector<float> visit_powers;
        visit_powers.reserve(children.size());
        
        for (const auto& child : children) {
            float visit_power = std::pow(static_cast<float>(child->getVisitCount()), 
                                       1.0f / settings_.temperature);
            visit_powers.push_back(visit_power);
            total_visits += visit_power;
        }
        
        // Normalize to probabilities
        for (float visit_power : visit_powers) {
            probabilities.push_back(total_visits > 0.0f ? visit_power / total_visits : 0.0f);
        }
        
        return probabilities;
    }
    
    int selectBestAction() {
        if (!root_node_ || root_node_->getChildren().empty()) {
            return 0;
        }
        
        auto children = root_node_->getChildren();
        auto best_child = std::max_element(children.begin(), children.end(),
            [](const auto& a, const auto& b) {
                return a->getVisitCount() < b->getVisitCount();
            });
        
        return best_child != children.end() ? (*best_child)->getAction() : 0;
    }
    
    void populateSearchStatistics(SearchResult& result, std::chrono::milliseconds total_time) {
        result.stats.total_nodes = root_node_ ? root_node_->getVisitCount() : 0;
        result.stats.search_time = total_time;
        result.stats.total_evaluations = stats_.total_tasks_executed.load();
        result.stats.nodes_per_second = total_time.count() > 0 ? 
            static_cast<float>(result.stats.total_evaluations) / (total_time.count() / 1000.0f) : 0.0f;
        
        // Taskflow-specific statistics
        auto taskflow_analytics = getTaskflowAnalytics();
        result.stats.parallel_efficiency = taskflow_analytics.parallel_efficiency;
        result.stats.task_scheduling_overhead = taskflow_analytics.task_scheduling_overhead;
        
        // Unified inference server statistics
        if (unified_server_shared_) {
            auto server_stats = unified_server_shared_->getStats();
            result.stats.avg_batch_size = server_stats.getAverageBatchSize();
            result.stats.avg_batch_latency = std::chrono::milliseconds(static_cast<int64_t>(server_stats.getAverageBatchLatency()));
        }
        
        // Burst coordinator statistics
        if (burst_coordinator_) {
            auto burst_stats = burst_coordinator_->getEfficiencyStats();
            result.stats.burst_efficiency = burst_stats.average_collection_efficiency;
        }
        
        // Transposition table statistics
        if (transposition_table_) {
            result.stats.tt_hit_rate = transposition_table_->hitRate();
            result.stats.tt_size = transposition_table_->size();
        }
    }
};

// Global Taskflow Engine Factory Function
std::unique_ptr<MCTSTaskflowEngine> createTaskflowEngine(const MCTSSettings& settings,
                                                         std::shared_ptr<nn::NeuralNetwork> network) {
    return std::make_unique<MCTSTaskflowEngine>(settings, std::move(network));
}

} // namespace mcts
} // namespace alphazero