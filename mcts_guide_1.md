# The Ultimate Guide to Leaf-Parallelized MCTS with Batch Neural Network Inference

Production-ready Monte Carlo Tree Search (MCTS) with neural network evaluation requires carefully designed parallelization to maximize throughput. This comprehensive guide walks through implementing leaf parallelization with centralized batch inference using moodycamel::ConcurrentQueue in C++.

## Understanding the architecture

Leaf parallelization is one of three main MCTS parallelization strategies, alongside root and tree parallelization. It offers superior GPU utilization by batching neural network evaluations from multiple leaf nodes, making it ideal for AlphaZero-style implementations.

The architecture consists of five core components:

1. **MCTS Core**: Manages selection, expansion, and backpropagation
2. **Leaf Parallelization Layer**: Coordinates multiple parallel simulations
3. **Batch Processor**: Collects evaluation requests into optimal batches
4. **Neural Network Interface**: Performs batched inference on the GPU
5. **Thread Synchronization Mechanisms**: Maintains tree consistency

Let's build this system step-by-step.

## 1. Implementing core data structures

Begin with the foundational node structure, optimized for concurrent access:

```cpp
class MCTSNode {
private:
    GameState state;
    std::atomic<int> visit_count{0};
    std::atomic<float> value_sum{0.0f};
    std::atomic<int> virtual_loss_count{0};
    
    // Neural network outputs
    std::vector<float> policy_priors;
    
    // Tree structure
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    std::vector<Action> actions;
    
    // Threading control
    std::mutex expansion_mutex;
    
public:
    MCTSNode(const GameState& state, MCTSNode* parent = nullptr)
        : state(state), parent(parent) {}
    
    // Node selection using PUCT formula with virtual loss
    MCTSNode* select_child() {
        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        
        float exploration_factor = EXPLORATION_CONSTANT * 
            std::sqrt(static_cast<float>(visit_count.load()));
        
        for (size_t i = 0; i < children.size(); ++i) {
            MCTSNode* child = children[i];
            
            // Get stats (thread-safe reads)
            int child_visits = child->visit_count.load();
            int virtual_losses = child->virtual_loss_count.load();
            float child_value = child->value_sum.load();
            
            // Apply virtual loss penalty
            int effective_visits = child_visits + virtual_losses;
            float effective_value = child_value - virtual_losses;
            
            // PUCT formula (AlphaZero-style)
            float exploitation = effective_visits > 0 ? 
                effective_value / effective_visits : 0.0f;
            float exploration = policy_priors[i] * exploration_factor / 
                (1 + effective_visits);
            
            float score = exploitation + exploration;
            
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
        
        return best_child;
    }
    
    // Virtual loss methods
    void add_virtual_loss() {
        virtual_loss_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    void remove_virtual_loss() {
        virtual_loss_count.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // Other methods (expansion, backpropagation, etc.)
    // ...
};
```

## 2. Setting up concurrent queues for leaf evaluation

Next, implement the queuing system using moodycamel::ConcurrentQueue:

```cpp
#include "concurrentqueue.h"

// Evaluation request structure
struct EvaluationRequest {
    MCTSNode* node;
    GameState state;
    std::promise<NetworkOutput> promise;
    
    EvaluationRequest(MCTSNode* n, const GameState& s) 
        : node(n), state(s) {}
};

// Batch inference manager
class BatchInferenceManager {
private:
    // Queue for collecting evaluation requests
    moodycamel::ConcurrentQueue<EvaluationRequest> request_queue;
    
    // Neural network model
    std::unique_ptr<NeuralNetwork> network;
    
    // Worker thread for batch processing
    std::thread worker_thread;
    std::atomic<bool> shutdown_flag{false};
    
    // Batch processing parameters
    size_t max_batch_size;
    std::chrono::milliseconds batch_timeout;
    
public:
    BatchInferenceManager(size_t batch_size = 16, 
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(5))
        : max_batch_size(batch_size), batch_timeout(timeout) {
        
        // Initialize neural network
        network = std::make_unique<NeuralNetwork>("model_weights.bin");
        
        // Start worker thread
        worker_thread = std::thread(&BatchInferenceManager::process_batches, this);
    }
    
    ~BatchInferenceManager() {
        // Signal shutdown and wait for thread to finish
        shutdown_flag = true;
        worker_thread.join();
    }
    
    // Request evaluation of a state
    std::future<NetworkOutput> evaluate(MCTSNode* node, const GameState& state) {
        EvaluationRequest request(node, state);
        
        // Get future before moving promise into the queue
        std::future<NetworkOutput> future = request.promise.get_future();
        
        // Add to queue
        request_queue.enqueue(std::move(request));
        
        return future;
    }
    
private:
    void process_batches() {
        while (!shutdown_flag) {
            auto batch = collect_batch();
            if (!batch.empty()) {
                process_batch(batch);
            }
        }
    }
    
    std::vector<EvaluationRequest> collect_batch() {
        // Implementation to follow in next section...
    }
    
    void process_batch(std::vector<EvaluationRequest>& batch) {
        // Implementation to follow in next section...
    }
};
```

## 3. Strategies for batching neural network evaluations

Now let's implement the batch collection and processing logic:

```cpp
std::vector<EvaluationRequest> BatchInferenceManager::collect_batch() {
    std::vector<EvaluationRequest> batch;
    
    // Start with a timestamp for timeout calculation
    auto start_time = std::chrono::steady_clock::now();
    
    // Try to collect up to max_batch_size requests
    EvaluationRequest request;
    while (batch.size() < max_batch_size) {
        // Try to dequeue a request
        if (request_queue.try_dequeue(request)) {
            batch.push_back(std::move(request));
        } else {
            // Queue is empty, wait a bit and check timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            
            // If we have at least one request and timed out, or queue is empty, process what we have
            if ((batch.size() > 0 && elapsed > batch_timeout) || 
                request_queue.size_approx() == 0) {
                break;
            }
        }
    }
    
    return batch;
}

void BatchInferenceManager::process_batch(std::vector<EvaluationRequest>& batch) {
    // Prepare input tensor from batch states
    std::vector<GameState> states;
    states.reserve(batch.size());
    
    for (const auto& request : batch) {
        states.push_back(request.state);
    }
    
    // Run batch inference
    std::vector<NetworkOutput> outputs = network->batch_inference(states);
    
    // Distribute results to requesters via promises
    for (size_t i = 0; i < batch.size(); ++i) {
        batch[i].promise.set_value(std::move(outputs[i]));
    }
}
```

## 4. Thread synchronization in MCTS

Let's implement the thread synchronization patterns for the main MCTS algorithm:

```cpp
class LeafParallelMCTS {
private:
    MCTSNode* root;
    BatchInferenceManager inference_manager;
    std::vector<std::thread> worker_threads;
    std::atomic<bool> shutdown_flag{false};
    std::atomic<int> active_simulations{0};
    
    // Thread synchronization
    std::mutex tree_mutex;
    
public:
    LeafParallelMCTS(size_t num_threads, size_t batch_size) 
        : inference_manager(batch_size) {
        
        // Create worker threads
        worker_threads.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            worker_threads.emplace_back(&LeafParallelMCTS::search_worker, this);
        }
    }
    
    ~LeafParallelMCTS() {
        // Signal shutdown and join threads
        shutdown_flag = true;
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Clean up tree
        delete_tree(root);
    }
    
    Action search(const GameState& state, int num_simulations) {
        // Initialize or reset root
        if (root) delete_tree(root);
        root = new MCTSNode(state);
        
        // Reset counters
        active_simulations = 0;
        
        // Run simulations in parallel
        for (int i = 0; i < num_simulations; ++i) {
            // Increment active simulation count
            active_simulations.fetch_add(1, std::memory_order_relaxed);
            
            // Signal workers to start a simulation
            simulation_cv.notify_one();
        }
        
        // Wait for all simulations to complete
        while (active_simulations.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Select best action (most visited child)
        MCTSNode* best_child = nullptr;
        int most_visits = -1;
        
        for (MCTSNode* child : root->children) {
            int visits = child->visit_count.load();
            if (visits > most_visits) {
                most_visits = visits;
                best_child = child;
            }
        }
        
        return best_child ? best_child->action : Action();
    }
    
private:
    void search_worker() {
        while (!shutdown_flag) {
            // Wait for work
            {
                std::unique_lock<std::mutex> lock(simulation_mutex);
                simulation_cv.wait(lock, [this]() {
                    return active_simulations.load() > 0 || shutdown_flag;
                });
                
                if (shutdown_flag) break;
            }
            
            // Run a single simulation
            run_single_simulation();
            
            // Decrement active simulation count
            active_simulations.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    
    void run_single_simulation() {
        // This will be detailed in the next section
    }
};
```

## 5. Virtual loss implementation

Now let's implement the core MCTS algorithm with virtual loss:

```cpp
void LeafParallelMCTS::run_single_simulation() {
    std::vector<MCTSNode*> path;
    MCTSNode* node = root;
    
    // Selection phase - find a leaf node
    while (!node->is_leaf() && !node->is_terminal()) {
        path.push_back(node);
        node = node->select_child();
        
        // Apply virtual loss to discourage other threads from taking this path
        node->add_virtual_loss();
    }
    
    // Expansion phase - if node is not terminal and has visits, expand it
    if (!node->is_terminal() && node->visit_count.load() > 0) {
        std::lock_guard<std::mutex> lock(node->expansion_mutex);
        
        // Recheck conditions after acquiring lock
        if (node->is_leaf() && !node->is_terminal() && node->visit_count.load() > 0) {
            node->expand();
            
            // Choose a child for evaluation
            MCTSNode* child = node->select_child();
            path.push_back(node);
            node = child;
            
            // Apply virtual loss to this child too
            node->add_virtual_loss();
        }
    }
    
    // Evaluation phase
    float value;
    if (node->is_terminal()) {
        // Terminal nodes have a definite outcome
        value = node->get_terminal_value();
    } else {
        // Request neural network evaluation
        std::future<NetworkOutput> future = 
            inference_manager.evaluate(node, node->state);
        
        // Wait for evaluation
        NetworkOutput output = future.get();
        
        // Initialize node with policy
        node->initialize_with_policy(output.policy);
        
        value = output.value;
    }
    
    // Backpropagation phase - update all nodes in the path
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* n = *it;
        
        // Remove virtual loss
        n->remove_virtual_loss();
        
        // Update statistics
        n->visit_count.fetch_add(1, std::memory_order_relaxed);
        n->value_sum.fetch_add(value, std::memory_order_relaxed);
        
        // Flip value for alternating players (assuming two-player zero-sum game)
        value = -value;
    }
}
```

## 6. Handling timeouts and batch size tradeoffs

To properly handle timeouts and adapt batch sizes:

```cpp
class AdaptiveBatchManager : public BatchInferenceManager {
private:
    // Batch size adaptation parameters
    size_t min_batch_size = 1;
    size_t max_batch_size = 64;
    size_t current_batch_size;
    
    // Metrics for adaptation
    std::deque<float> recent_throughputs;
    std::chrono::milliseconds adaptation_interval{1000};
    std::chrono::steady_clock::time_point last_adaptation;
    
    // Timeout adaptation
    std::chrono::milliseconds min_timeout{1};
    std::chrono::milliseconds max_timeout{20};
    std::chrono::milliseconds current_timeout;
    
    // Monitoring metrics
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> batches_processed{0};
    std::atomic<int64_t> cumulative_batch_size{0};
    
public:
    AdaptiveBatchManager(size_t initial_batch_size = 16)
        : current_batch_size(initial_batch_size),
          current_timeout(std::chrono::milliseconds(5)),
          last_adaptation(std::chrono::steady_clock::now()) {
    }
    
protected:
    std::vector<EvaluationRequest> collect_batch() override {
        auto batch = BatchInferenceManager::collect_batch();
        
        // Record metrics
        total_requests.fetch_add(batch.size(), std::memory_order_relaxed);
        batches_processed.fetch_add(1, std::memory_order_relaxed);
        cumulative_batch_size.fetch_add(batch.size(), std::memory_order_relaxed);
        
        // Check if it's time to adapt parameters
        auto now = std::chrono::steady_clock::now();
        if (now - last_adaptation > adaptation_interval) {
            adapt_parameters();
            last_adaptation = now;
        }
        
        return batch;
    }
    
private:
    void adapt_parameters() {
        // Calculate recent throughput
        int64_t requests = total_requests.exchange(0);
        int64_t batches = batches_processed.exchange(0);
        int64_t batch_sizes_sum = cumulative_batch_size.exchange(0);
        
        if (batches == 0) return;
        
        float avg_batch_size = static_cast<float>(batch_sizes_sum) / batches;
        float throughput = static_cast<float>(requests) / 
                          std::chrono::duration_cast<std::chrono::seconds>(
                              adaptation_interval).count();
        
        recent_throughputs.push_back(throughput);
        if (recent_throughputs.size() > 5) {
            recent_throughputs.pop_front();
        }
        
        // Calculate throughput trend
        float trend = 0.0f;
        if (recent_throughputs.size() >= 2) {
            trend = recent_throughputs.back() - recent_throughputs.front();
        }
        
        // Adapt batch size based on trend and current situation
        if (trend > 0 && avg_batch_size >= current_batch_size * 0.9) {
            // Throughput is increasing and batches are nearly full - increase batch size
            current_batch_size = std::min(current_batch_size + 4, max_batch_size);
        } else if (trend < 0 && current_batch_size > min_batch_size) {
            // Throughput is decreasing - try smaller batches
            current_batch_size = std::max(current_batch_size - 2, min_batch_size);
        }
        
        // Adapt timeout based on batch fill rate
        float fill_rate = avg_batch_size / current_batch_size;
        
        if (fill_rate < 0.5) {
            // Batches aren't filling quickly - increase timeout
            auto new_timeout = std::min(
                current_timeout + std::chrono::milliseconds(1),
                max_timeout);
            current_timeout = new_timeout;
        } else if (fill_rate > 0.9) {
            // Batches are filling quickly - decrease timeout
            auto new_timeout = std::max(
                current_timeout - std::chrono::milliseconds(1),
                min_timeout);
            current_timeout = new_timeout;
        }
    }
};
```

## 7. Using std::future/std::promise for returning results

Let's expand the evaluation request with more robust future/promise handling:

```cpp
// Enhanced handling with timeouts and cancellation
class EvaluationManager {
private:
    BatchInferenceManager& inference_manager;
    
    // Track pending evaluations
    struct PendingEval {
        MCTSNode* node;
        std::future<NetworkOutput> future;
        std::chrono::steady_clock::time_point request_time;
    };
    
    std::vector<PendingEval> pending_evaluations;
    std::mutex pending_mutex;
    
public:
    EvaluationManager(BatchInferenceManager& manager) 
        : inference_manager(manager) {}
    
    void request_evaluation(MCTSNode* node, const GameState& state) {
        // Request evaluation from inference manager
        std::future<NetworkOutput> future = 
            inference_manager.evaluate(node, state);
        
        // Record pending evaluation
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_evaluations.push_back({
                node,
                std::move(future),
                std::chrono::steady_clock::now()
            });
        }
    }
    
    void process_completed_evaluations(std::chrono::milliseconds timeout = 
                                      std::chrono::milliseconds(50)) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        
        auto now = std::chrono::steady_clock::now();
        auto it = pending_evaluations.begin();
        
        while (it != pending_evaluations.end()) {
            auto& eval = *it;
            
            // Check if evaluation has completed
            if (eval.future.wait_for(std::chrono::seconds(0)) == 
                std::future_status::ready) {
                
                // Get the result
                NetworkOutput output = eval.future.get();
                
                // Update the node
                eval.node->initialize_with_policy(output.policy);
                eval.node->backpropagate(output.value);
                
                // Remove from pending list
                it = pending_evaluations.erase(it);
            } 
            // Check for timeout
            else if (now - eval.request_time > timeout) {
                // Evaluation timed out - use fallback
                eval.node->initialize_with_default_policy();
                eval.node->backpropagate(0.0f);  // Neutral value
                
                // Remove from pending list
                it = pending_evaluations.erase(it);
            }
            else {
                ++it;
            }
        }
    }
};
```

## 8. Transposition table implementation

Let's implement a thread-safe transposition table:

```cpp
class TranspositionTable {
private:
    struct Entry {
        std::atomic<uint64_t> key{0};
        std::atomic<uint32_t> lock{0};
        std::atomic<int16_t> depth{0};
        std::atomic<int16_t> bound_type{0};
        std::atomic<float> value{0.0f};
        std::atomic<uint32_t> best_move{0};
        std::atomic<uint32_t> nodes_count{0};
        std::vector<float> policy;
        
        // Lock for policy updates
        std::mutex policy_mutex;
    };
    
    std::vector<Entry> table;
    size_t size_mask;
    
public:
    // Bound types
    static constexpr int16_t BOUND_NONE = 0;
    static constexpr int16_t BOUND_UPPER = 1;
    static constexpr int16_t BOUND_LOWER = 2;
    static constexpr int16_t BOUND_EXACT = 3;
    
    TranspositionTable(size_t size) {
        // Round up to power of 2
        size_t power_of_2 = 1;
        while (power_of_2 < size) {
            power_of_2 *= 2;
        }
        
        table.resize(power_of_2);
        size_mask = power_of_2 - 1;
    }
    
    // Thread-safe store
    void store(uint64_t key, int16_t depth, int16_t bound_type, 
               float value, uint32_t best_move, 
               const std::vector<float>& policy) {
        
        size_t index = key & size_mask;
        Entry& entry = table[index];
        
        // Lock entry for update
        uint32_t expected = 0;
        while (!entry.lock.compare_exchange_weak(expected, 1, 
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
            expected = 0;
            std::this_thread::yield();
        }
        
        // Check if this entry should replace the existing one
        bool should_replace = 
            entry.key.load() == 0 ||  // Empty slot
            entry.key.load() == key ||  // Same position
            entry.depth.load() <= depth;  // Greater or equal depth
            
        if (should_replace) {
            entry.key.store(key, std::memory_order_relaxed);
            entry.depth.store(depth, std::memory_order_relaxed);
            entry.bound_type.store(bound_type, std::memory_order_relaxed);
            entry.value.store(value, std::memory_order_relaxed);
            entry.best_move.store(best_move, std::memory_order_relaxed);
            
            // Update nodes count
            entry.nodes_count.store(0, std::memory_order_relaxed);
            
            // Update policy (needs separate lock)
            {
                std::lock_guard<std::mutex> policy_lock(entry.policy_mutex);
                entry.policy = policy;
            }
        }
        
        // Unlock
        entry.lock.store(0, std::memory_order_release);
    }
    
    // Thread-safe lookup
    bool lookup(uint64_t key, int16_t& depth, int16_t& bound_type,
                float& value, uint32_t& best_move,
                std::vector<float>& policy) {
        
        size_t index = key & size_mask;
        Entry& entry = table[index];
        
        // Lock entry for reading
        uint32_t expected = 0;
        while (!entry.lock.compare_exchange_weak(expected, 1, 
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
            expected = 0;
            std::this_thread::yield();
        }
        
        bool found = (entry.key.load(std::memory_order_relaxed) == key);
        
        if (found) {
            depth = entry.depth.load(std::memory_order_relaxed);
            bound_type = entry.bound_type.load(std::memory_order_relaxed);
            value = entry.value.load(std::memory_order_relaxed);
            best_move = entry.best_move.load(std::memory_order_relaxed);
            
            // Increment nodes count
            entry.nodes_count.fetch_add(1, std::memory_order_relaxed);
            
            // Copy policy
            {
                std::lock_guard<std::mutex> policy_lock(entry.policy_mutex);
                policy = entry.policy;
            }
        }
        
        // Unlock
        entry.lock.store(0, std::memory_order_release);
        
        return found;
    }
};
```

## 9. Progressive widening implementation

Finally, let's implement progressive widening to handle large action spaces:

```cpp
class MCTSNodeWithProgressiveWidening : public MCTSNode {
private:
    // Progressive widening parameters
    float k_pw = 1.0f;
    float alpha_pw = 0.5f;
    
    // Available actions before expansion
    std::vector<Action> all_actions;
    
public:
    // Override expansion method to implement progressive widening
    void expand() override {
        std::lock_guard<std::mutex> lock(expansion_mutex);
        
        if (children.empty() && !all_actions.empty()) {
            // First expansion - get all available actions
            all_actions = state.get_legal_actions();
            
            // Sort by policy prior if available
            if (!policy_priors.empty()) {
                // Create index-value pairs
                std::vector<std::pair<int, float>> action_priors;
                for (size_t i = 0; i < all_actions.size(); ++i) {
                    action_priors.emplace_back(i, policy_priors[i]);
                }
                
                // Sort by prior value (descending)
                std::sort(action_priors.begin(), action_priors.end(),
                         [](const auto& a, const auto& b) {
                             return a.second > b.second;
                         });
                
                // Reorder actions and priors
                std::vector<Action> sorted_actions;
                std::vector<float> sorted_priors;
                
                for (const auto& [idx, prior] : action_priors) {
                    sorted_actions.push_back(all_actions[idx]);
                    sorted_priors.push_back(prior);
                }
                
                all_actions = std::move(sorted_actions);
                policy_priors = std::move(sorted_priors);
            }
        }
        
        // Calculate how many actions should be expanded according to formula:
        // count = k * N^alpha
        int visit_count = this->visit_count.load();
        int action_count = std::ceil(k_pw * std::pow(visit_count, alpha_pw));
        
        // Limit to actual available actions
        action_count = std::min(action_count, static_cast<int>(all_actions.size()));
        
        // Expand only up to the current progressive widening limit
        while (children.size() < action_count) {
            size_t action_idx = children.size();
            Action action = all_actions[action_idx];
            
            // Create child state
            GameState child_state = state.apply_action(action);
            
            // Create new node
            MCTSNode* child = new MCTSNodeWithProgressiveWidening(child_state, this);
            child->action = action;
            
            // Set prior probability if available
            if (action_idx < policy_priors.size()) {
                child->prior_probability = policy_priors[action_idx];
            }
            
            // Add to children
            children.push_back(child);
            actions.push_back(action);
        }
    }
    
    // Check if node is fully expanded according to progressive widening
    bool is_fully_expanded() override {
        int visit_count = this->visit_count.load();
        int action_count = std::ceil(k_pw * std::pow(visit_count, alpha_pw));
        
        return children.size() >= action_count &&
               children.size() >= all_actions.size();
    }
};
```

## Putting it all together: The complete system

Let's integrate all these components into a complete, production-ready implementation:

```cpp
class ProductionMCTS {
private:
    // Core components
    std::unique_ptr<MCTSNode> root;
    TranspositionTable trans_table;
    AdaptiveBatchManager batch_manager;
    EvaluationManager eval_manager;
    
    // Thread pool
    ThreadPool thread_pool;
    
    // Configuration
    SearchConfig config;
    
public:
    ProductionMCTS(const SearchConfig& search_config)
        : trans_table(search_config.tt_size),
          batch_manager(search_config.batch_size),
          eval_manager(batch_manager),
          thread_pool(search_config.num_threads),
          config(search_config) {}
    
    Action search(const GameState& state, int num_simulations) {
        // Initialize search
        uint64_t state_hash = state.compute_hash();
        
        // Check transposition table first
        int16_t depth, bound_type;
        float value;
        uint32_t best_move;
        std::vector<float> policy;
        
        if (trans_table.lookup(state_hash, depth, bound_type, 
                              value, best_move, policy)) {
            // Found in transposition table
            if (bound_type == TranspositionTable::BOUND_EXACT && 
                depth >= config.min_tt_depth) {
                
                // We can just return the best move directly
                return Action(best_move);
            }
        }
        
        // Initialize or reset root
        root = std::make_unique<MCTSNodeWithProgressiveWidening>(state);
        
        // Initialize neural network evaluation for root
        std::future<NetworkOutput> root_eval = 
            batch_manager.evaluate(root.get(), state);
        
        NetworkOutput root_output = root_eval.get();
        root->initialize_with_policy(root_output.policy);
        
        // Run parallel simulations
        std::atomic<int> remaining_sims(num_simulations);
        
        // Queue tasks for thread pool
        for (int i = 0; i < config.num_threads; ++i) {
            thread_pool.enqueue([this, &remaining_sims]() {
                while (remaining_sims.fetch_sub(1) > 0) {
                    this->run_simulation();
                    
                    // Process any completed evaluations
                    eval_manager.process_completed_evaluations();
                }
            });
        }
        
        // Wait for all simulations to complete
        thread_pool.wait_idle();
        
        // Process any remaining evaluations
        eval_manager.process_completed_evaluations();
        
        // Select best move (usually most visited child)
        MCTSNode* best_child = nullptr;
        int most_visits = -1;
        
        for (MCTSNode* child : root->children) {
            int visits = child->visit_count.load();
            if (visits > most_visits) {
                most_visits = visits;
                best_child = child;
            }
        }
        
        Action best_action = best_child ? best_child->action : Action();
        
        // Store in transposition table
        trans_table.store(state_hash, config.tt_depth, 
                         TranspositionTable::BOUND_EXACT,
                         root->value_sum.load() / root->visit_count.load(),
                         best_action.get_id(), 
                         root_output.policy);
        
        return best_action;
    }
    
private:
    void run_simulation() {
        std::vector<MCTSNode*> path;
        MCTSNode* node = root.get();
        
        // Selection phase - descend tree with virtual loss
        while (!node->is_leaf() && !node->is_terminal()) {
            path.push_back(node);
            node = node->select_child();
            node->add_virtual_loss();
        }
        
        // Expansion and evaluation phases
        float value;
        if (node->is_terminal()) {
            value = node->get_terminal_value();
        } else {
            // Check transposition table
            uint64_t node_hash = node->state.compute_hash();
            int16_t depth, bound_type;
            float tt_value;
            uint32_t best_move;
            std::vector<float> policy;
            
            if (trans_table.lookup(node_hash, depth, bound_type, 
                                 tt_value, best_move, policy)) {
                // Found in TT, use this value
                value = tt_value;
                
                // Expand node with policy from TT
                if (node->visit_count.load() >= config.expansion_threshold) {
                    node->initialize_with_policy(policy);
                    node->expand();
                }
            } else {
                // Not found in TT, request neural network evaluation
                if (node->visit_count.load() >= config.expansion_threshold) {
                    // Request asynchronous evaluation
                    eval_manager.request_evaluation(node, node->state);
                    
                    // Just use a default value for now - will be updated later
                    value = 0.0f;
                } else {
                    // Node doesn't meet expansion threshold yet
                    value = 0.0f;
                }
            }
        }
        
        // Backpropagation - update all nodes in the path
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            MCTSNode* n = *it;
            n->visit_count.fetch_add(1, std::memory_order_relaxed);
            n->value_sum.fetch_add(value, std::memory_order_relaxed);
            value = -value;  // Flip for alternating players
        }
    }
};
```

## Performance optimization best practices

When optimizing your implementation for production use:

1. **Memory management**:
   - Use memory pools for node allocation
   - Implement node recycling for long-running searches
   - Consider custom allocators for cache-friendly memory layouts

2. **Profiling and monitoring**:
   - Track key metrics like nodes/second, batch utilization, and memory usage
   - Use CPU and GPU profilers to identify bottlenecks
   - Implement adaptive parameters based on runtime performance

3. **Error handling**:
   - Implement robust timeout handling for all operations
   - Use graceful degradation for GPU errors
   - Provide fallback evaluation mechanisms

4. **Testing**:
   - Create deterministic test scenarios with fixed seeds
   - Use ThreadSanitizer to detect race conditions
   - Benchmark against simpler implementations for correctness verification

## Conclusion

Implementing production-ready leaf-parallelized MCTS with batch neural network inference requires careful attention to concurrency, memory management, and GPU utilization. This guide has provided a comprehensive walkthrough of building such a system incrementally, with concrete C++ code examples that address all major components.

Key takeaways:
- Use moodycamel::ConcurrentQueue for efficient task distribution
- Implement virtual loss to encourage thread diversity
- Utilize std::future/std::promise for clean asynchronous communication
- Apply adaptive batch sizing for optimal GPU utilization
- Implement thread-safe transposition tables and progressive widening

By following these patterns, you can create a high-performance MCTS implementation that maximizes throughput while maintaining search quality.