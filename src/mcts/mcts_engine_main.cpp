#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/advanced_memory_pool.h"
#include "mcts/aggressive_memory_manager.h"
#include "nn/resnet_model.h"
#include "utils/debug_monitor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace alphazero {
namespace mcts {

// Define static members (kept for compatibility, but used for server coordination now)
std::mutex MCTSEngine::s_global_evaluator_mutex;
std::atomic<int> MCTSEngine::s_evaluator_init_counter{0};

// Constructor with neural network
MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> neural_net, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      game_state_pool_enabled_(true),
      use_advanced_memory_pool_(true),
      direct_inference_fn_(nullptr) {
    
    // Create transposition table if enabled (always use PHMap)
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128;
        
        // Always use high-performance PHMap implementation
        PHMapTranspositionTable::Config config;
        config.size_mb = tt_size_mb;
        config.num_shards = 0;  // Auto-determine
        config.enable_compression = true;
        config.enable_stats = true;
        transposition_table_ = std::make_unique<PHMapTranspositionTable>(config);
    }
    
    // Create node tracker for pending evaluations
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Initialize advanced memory pool for optimized memory management
    AdvancedMemoryPoolConfig mem_pool_config;
    mem_pool_config.initial_size = 10000;  // Start with 10,000 pre-allocated objects
    mem_pool_config.growth_factor = 1.5;   // Grow by 50% when needed
    mem_pool_config.max_pool_size = 1000000; // Cap at 1 million objects
    memory_pool_ = std::make_unique<AdvancedMemoryPool>(mem_pool_config);
    
    // Initialize node pool for efficient MCTS node allocation
    MCTSNodePool::Config node_pool_config;
    node_pool_config.initial_pool_size = std::min(size_t(10000), size_t(settings.num_simulations * 5));
    node_pool_config.grow_size = std::min(size_t(5000), size_t(settings.num_simulations * 2));
    node_pool_config.max_pool_size = std::min(size_t(1000000), size_t(settings.num_simulations * 50));
    node_pool_ = std::make_unique<MCTSNodePool>(node_pool_config);
    
    // Initialize memory pressure monitor
    MemoryPressureMonitor::Config mem_monitor_config;
    mem_monitor_config.max_memory_bytes = 48ULL * 1024 * 1024 * 1024; // 48GB limit
    mem_monitor_config.warning_threshold = 0.75; // 75% warning
    mem_monitor_config.critical_threshold = 0.85; // 85% critical
    mem_monitor_config.check_interval = std::chrono::milliseconds(500); // Check every 500ms
    
    memory_pressure_monitor_ = std::make_unique<MemoryPressureMonitor>(mem_monitor_config);
    memory_pressure_monitor_->setCleanupCallback(
        [this](MemoryPressureMonitor::PressureLevel level) {
            handleMemoryPressure(level);
        });
    memory_pressure_monitor_->start();
    
    // Initialize GPU memory pool (always enabled)
    GPUMemoryPool::PoolConfig gpu_pool_config;
    gpu_pool_config.initial_pool_size_mb = 1024;  // 1GB initial allocation
    gpu_pool_config.max_pool_size_mb = 4096;      // 4GB max allocation
    gpu_memory_pool_ = std::make_unique<GPUMemoryPool>(gpu_pool_config);
    
    // Initialize dynamic batch manager (always enabled)
    DynamicBatchManager::Config batch_config;
    batch_config.min_batch_size = 1;
    batch_config.max_batch_size = settings.batch_size;
    batch_config.max_wait_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        settings.batch_timeout).count();
    batch_config.target_gpu_utilization_percent = 0.9;
    dynamic_batch_manager_ = std::make_unique<DynamicBatchManager>(batch_config);
    
    // Initialize aggressive memory manager
    auto& aggressive_memory_manager = AggressiveMemoryManager::getInstance();
    AggressiveMemoryManager::Config aggressive_config;
    aggressive_config.warning_threshold_gb = 28.0;
    aggressive_config.critical_threshold_gb = 35.0;
    aggressive_config.emergency_threshold_gb = 40.0;
    aggressive_memory_manager.setConfig(aggressive_config);
    
    // Register cleanup callbacks
    aggressive_memory_manager.registerCleanupCallback("MCTSEngine_NodePool",
        [this](AggressiveMemoryManager::PressureLevel level) {
            if (node_pool_ && level >= AggressiveMemoryManager::PressureLevel::WARNING) {
                node_pool_->compact();
                if (level >= AggressiveMemoryManager::PressureLevel::CRITICAL) {
                    // Release half of the free nodes
                    node_pool_->releaseMemory(1000);
                }
            }
        }, 100);
    
    aggressive_memory_manager.registerCleanupCallback("MCTSEngine_TranspositionTable",
        [this](AggressiveMemoryManager::PressureLevel level) {
            if (level >= AggressiveMemoryManager::PressureLevel::CRITICAL) {
                if (transposition_table_) {
                    transposition_table_->clear();
                }
            }
        }, 90);
    
    aggressive_memory_manager.registerCleanupCallback("MCTSEngine_AdvancedMemoryPool",
        [this](AggressiveMemoryManager::PressureLevel level) {
            if (memory_pool_ && level >= AggressiveMemoryManager::PressureLevel::WARNING) {
                memory_pool_->resetStats();
            }
        }, 80);
    
    // CRITICAL FIX: Register GPU memory cleanup callback
    aggressive_memory_manager.registerCleanupCallback("MCTSEngine_GPUMemory",
        [this](AggressiveMemoryManager::PressureLevel level) {
            if (level >= AggressiveMemoryManager::PressureLevel::WARNING) {
                // Clean GPU memory pool
                if (gpu_memory_pool_) {
                    gpu_memory_pool_->trim(0.5f); // Keep only 50% of unused memory
                }
                
                // Clean neural network tensor pools
                if (neural_network_) {
                    auto resnet = std::dynamic_pointer_cast<nn::ResNetModel>(neural_network_);
                    if (resnet) {
                        resnet->cleanupTensorPool();
                    }
                }
                
                // Force CUDA cache cleanup
                #ifdef WITH_TORCH
                if (torch::cuda::is_available()) {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
                #endif
            }
        }, 110); // Higher priority than other cleanups
    
    // Validate neural network
    if (!neural_net) {
        std::cerr << "ERROR: Null neural network passed to MCTSEngine" << std::endl;
        throw std::invalid_argument("Neural network cannot be null");
    }
    
    // Store neural network reference
    neural_network_ = neural_net;
    
    // Pass GPU memory pool to neural network if it's a ResNetModel
    if (gpu_memory_pool_) {
        auto resnet = std::dynamic_pointer_cast<nn::ResNetModel>(neural_net);
        if (resnet && gpu_memory_pool_) {
            // Convert unique_ptr to shared_ptr with a no-op deleter since we still own it
            auto shared_pool = std::shared_ptr<GPUMemoryPool>(gpu_memory_pool_.get(), [](GPUMemoryPool*){});
            resnet->setGPUMemoryPool(shared_pool);
        }
    }
    
    // Create direct inference function for serial mode
    direct_inference_fn_ = [neural_net](const std::vector<std::unique_ptr<core::IGameState>>& states) -> std::vector<NetworkOutput> {
        return neural_net->inference(states);
    };
    
    // For true serial mode (num_threads <= 0), skip complex inference infrastructure
    if (settings.num_threads <= 0) {
        return; // Skip creating complex infrastructure
    }
    
    // Simplified implementation - using direct batching approach
}

// Constructor with inference function
MCTSEngine::MCTSEngine(InferenceFunction inference_fn, const MCTSSettings& settings)
    : settings_(settings),
      shutdown_(false),
      active_simulations_(0),
      search_running_(false),
      random_engine_(std::random_device()()),
      transposition_table_(nullptr),
      use_transposition_table_(settings.use_transposition_table),
      game_state_pool_enabled_(true),
      use_advanced_memory_pool_(true),
      direct_inference_fn_(inference_fn) {
    
    // Initialize advanced memory pool for optimized memory management
    AdvancedMemoryPoolConfig mem_pool_config;
    mem_pool_config.initial_size = 10000;  // Start with 10,000 pre-allocated objects
    mem_pool_config.growth_factor = 1.5;   // Grow by 50% when needed
    mem_pool_config.max_pool_size = 1000000; // Cap at 1 million objects
    memory_pool_ = std::make_unique<AdvancedMemoryPool>(mem_pool_config);
    
    // Create transposition table if enabled (always use PHMap)
    if (use_transposition_table_) {
        size_t tt_size_mb = settings.transposition_table_size_mb > 0 ? 
                           settings.transposition_table_size_mb : 128;
        
        // Always use high-performance PHMap implementation
        PHMapTranspositionTable::Config config;
        config.size_mb = tt_size_mb;
        config.num_shards = 0;  // Auto-determine
        config.enable_compression = true;
        config.enable_stats = true;
        transposition_table_ = std::make_unique<PHMapTranspositionTable>(config);
    }
    
    // Create node tracker for pending evaluations
    node_tracker_ = std::make_unique<NodeTracker>();
    
    // Initialize memory pressure monitor (same as first constructor)
    MemoryPressureMonitor::Config mem_monitor_config;
    mem_monitor_config.max_memory_bytes = 48ULL * 1024 * 1024 * 1024; // 48GB limit
    mem_monitor_config.warning_threshold = 0.75; // 75% warning
    mem_monitor_config.critical_threshold = 0.85; // 85% critical
    mem_monitor_config.check_interval = std::chrono::milliseconds(500); // Check every 500ms
    
    memory_pressure_monitor_ = std::make_unique<MemoryPressureMonitor>(mem_monitor_config);
    memory_pressure_monitor_->setCleanupCallback(
        [this](MemoryPressureMonitor::PressureLevel level) {
            handleMemoryPressure(level);
        });
    memory_pressure_monitor_->start();
    
    // Validate inference function
    if (!inference_fn) {
        std::cerr << "ERROR: Null inference function passed to MCTSEngine" << std::endl;
        throw std::invalid_argument("Inference function cannot be null");
    }
    
    // For true serial mode (num_threads <= 0), skip complex inference infrastructure
    if (settings.num_threads <= 0) {
        return; // Skip creating complex infrastructure
    }
    
    // SIMPLIFIED: No more UnifiedInferenceServer or BurstCoordinator
        
}

// Start the inference server if not already running
bool MCTSEngine::ensureEvaluatorStarted() {
    // SIMPLIFIED: No more server to start
    return true;
}

// Main search method
SearchResult MCTSEngine::search(const core::IGameState& state) {
    auto start_time = std::chrono::steady_clock::now();

    // Validate the state
    if (!safeGameStateValidation(state)) {
        std::cerr << "Invalid game state, returning default result" << std::endl;
        SearchResult result;
        result.action = -1;
        result.value = 0.0f;
        
        // Use first legal move as fallback
        auto legal_moves = state.getLegalMoves();
        if (!legal_moves.empty()) {
            result.action = legal_moves[0];
            result.probabilities.resize(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        }
        return result;
    }

    // Clear the transposition table for new search
    if (use_transposition_table_ && transposition_table_) {
        transposition_table_->clear();
        // Stats are reset automatically when clearing
    }

    // Reset previous search state
    root_.reset();
    
    // Initialize statistics for new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;

    // Check if state is already terminal
    if (state.isTerminal()) {
        SearchResult result;
        result.action = -1;
        
        try {
            core::GameResult game_res = state.getGameResult();
            int current_player = state.getCurrentPlayer();
            if (game_res == core::GameResult::WIN_PLAYER1) {
                result.value = (current_player == 1) ? 1.0f : -1.0f;
            } else if (game_res == core::GameResult::WIN_PLAYER2) {
                result.value = (current_player == 2) ? 1.0f : -1.0f;
            } else {
                result.value = 0.0f;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error getting terminal value: " << e.what() << std::endl;
            result.value = 0.0f;
        }
        
        result.probabilities.assign(state.getActionSpaceSize(), 0.0f);
        last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
    }

    // Handle serial mode vs parallel mode differently
    if (settings_.num_threads <= 0) {
        // Serial mode: use direct inference, no threading infrastructure needed
        if (!direct_inference_fn_) {
            throw std::runtime_error("Direct inference function not available for serial mode");
        }
    } else {
        // Parallel mode: simplified architecture no longer uses UnifiedInferenceServer
        // Always use traditional evaluator
        if (!ensureEvaluatorStarted()) {
            throw std::runtime_error("Failed to start evaluator");
        }
        
        // UnifiedInferenceServer and BurstCoordinator were removed in simplification
        
        // Inference server was removed in simplification
    }
    
    // Run the search with enhanced error handling
    try {
        runSearch(state);
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: MCTSEngine::search - Exception during runSearch: " << e.what() << std::endl;
        safelyStopEvaluator();
        throw;
    } catch (...) {
        std::cerr << "CRITICAL ERROR: MCTSEngine::search - Unknown exception during runSearch" << std::endl;
        safelyStopEvaluator();
        throw std::runtime_error("Unknown error during search");
    }

    // Calculate search time
    auto end_time = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Prepare search result
    SearchResult result;
    result.action = -1;

    try {
        // Extract action probabilities
        result.probabilities = getActionProbabilities(root_, settings_.temperature);

        // Select action from probabilities
        if (!result.probabilities.empty()) {
            float sum = std::accumulate(result.probabilities.begin(), result.probabilities.end(), 0.0f);
            
            if (std::abs(sum - 1.0f) > 0.1f && sum > 0.0f) {
                // Normalize probabilities
                for (auto& p : result.probabilities) {
                    p /= sum;
                }
            }
            
            // Temperature near zero - deterministic selection (argmax)
            if (settings_.temperature < 0.01f) {
                auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
                if (max_it != result.probabilities.end()) {
                    result.action = std::distance(result.probabilities.begin(), max_it);
                }
            } else {
                // Sample according to the probability distribution
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float r = dist(random_engine_);
                float cumsum = 0.0f;
                
                for (size_t i = 0; i < result.probabilities.size(); i++) {
                    cumsum += result.probabilities[i];
                    if (r <= cumsum) {
                        result.action = static_cast<int>(i);
                        break;
                    }
                }
            }
            
            // Fallback in case of numerical issues
            if (result.action < 0 && !result.probabilities.empty()) {
                auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
                result.action = std::distance(result.probabilities.begin(), max_it);
            }
        } 
        else if (root_ && !root_->getChildren().empty()) {
            // If no probabilities, select most visited child
            int max_visits = -1;
            
            for (size_t i = 0; i < root_->getChildren().size(); i++) {
                auto child = root_->getChildren()[i];
                if (child && child->getVisitCount() > max_visits) {
                    max_visits = child->getVisitCount();
                    if (i < root_->getActions().size()) {
                        result.action = root_->getActions()[i];
                    }
                }
            }
        } 
        else {
            // Last resort - select a valid legal move
            auto legal_moves = state.getLegalMoves();
            if (!legal_moves.empty()) {
                result.action = legal_moves[0];
            }
        }

        // Get value estimate
        result.value = root_ ? root_->getValue() : 0.0f;
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting search results: " << e.what() << std::endl;
        
        if (result.action < 0) {
            try {
                auto legal_moves = state.getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                }
            } catch (...) {
                // Silently ignore errors here
            }
        }
    }
    catch (...) {
        std::cerr << "Unknown error extracting search results" << std::endl;
        
        if (result.action < 0) {
            try {
                auto legal_moves = state.getLegalMoves();
                if (!legal_moves.empty()) {
                    result.action = legal_moves[0];
                }
            } catch (...) {
                // Silently ignore errors here
            }
        }
    }

    // Update statistics
    last_stats_.search_time = search_time;
    // Serial mode: set basic statistics
    last_stats_.avg_batch_size = settings_.batch_size; // Using configured batch size
    last_stats_.avg_batch_latency = std::chrono::milliseconds(1);
    last_stats_.total_evaluations = settings_.num_simulations; // Approximate
    
    if (last_stats_.search_time.count() > 0) {
        last_stats_.nodes_per_second = 1000.0f * last_stats_.total_nodes / 
                                      std::max(1, static_cast<int>(last_stats_.search_time.count()));
    }

    // Add transposition table stats if enabled
    if (use_transposition_table_ && transposition_table_) {
        auto tt_stats = transposition_table_->getStats();
        last_stats_.tt_hit_rate = tt_stats.hit_rate;
        last_stats_.tt_size = transposition_table_->size();
    }
    
    result.stats = last_stats_;
    
    // FIX: Clean up memory after search completes
    // This prevents memory accumulation between searches
    cleanupPendingEvaluations();
    
    return result;
}

// Safely stop the inference server if it was started
void MCTSEngine::safelyStopEvaluator() {
    // UnifiedInferenceServer was removed in simplification
    // Nothing to do here anymore
}

// Configure shared external queues
void MCTSEngine::setSharedExternalQueues(
        moodycamel::ConcurrentQueue<PendingEvaluation>* leaf_queue,
        moodycamel::ConcurrentQueue<std::pair<NetworkOutput, PendingEvaluation>>* result_queue,
        std::function<void()> notify_fn) {
    shared_leaf_queue_ = leaf_queue;
    shared_result_queue_ = result_queue;
    use_shared_queues_ = true;
    external_queue_notify_fn_ = notify_fn;
    
    // UnifiedInferenceServer was removed in simplification
}

// Accessor methods for transposition table
void MCTSEngine::setUseTranspositionTable(bool use) {
    use_transposition_table_ = use;
}

bool MCTSEngine::isUsingTranspositionTable() const {
    return use_transposition_table_;
}

void MCTSEngine::setTranspositionTableSize(size_t size_mb) {
    // Always use PHMap transposition table
    PHMapTranspositionTable::Config config;
    config.size_mb = size_mb;
    config.num_shards = 0;  // Auto-determine
    config.enable_compression = true;
    config.enable_stats = true;
    transposition_table_ = std::make_unique<PHMapTranspositionTable>(config);
}

void MCTSEngine::clearTranspositionTable() {
    if (transposition_table_) {
        transposition_table_->clear();
    }
}

float MCTSEngine::getTranspositionTableHitRate() const {
    if (transposition_table_) {
        auto stats = transposition_table_->getStats();
        return stats.hit_rate;
    }
    return 0.0f;
}

void MCTSEngine::setUsePHMapTransposition(bool use_phmap) {
    // Always use PHMap, so this method is now a no-op
    // Kept for API compatibility
}

// Settings and stats accessors
const MCTSSettings& MCTSEngine::getSettings() const {
    return settings_;
}

void MCTSEngine::updateSettings(const MCTSSettings& settings) {
    settings_ = settings;
}

const MCTSStats& MCTSEngine::getLastStats() const {
    return last_stats_;
}


// Force memory cleanup
void MCTSEngine::monitorMemoryUsage() {
    // Implement memory monitoring and cleanup logic
}

// Move constructor
MCTSEngine::MCTSEngine(MCTSEngine&& other) noexcept {
    // First stop the other engine to ensure thread safety
    other.shutdown_ = true;
    
    // UnifiedInferenceServer was removed in simplification
    
    // Wait for threads to complete
    if (other.result_distributor_worker_.joinable()) {
        try {
            other.result_distributor_worker_.join();
        } catch (...) {
            // Ignore exceptions
        }
    }
    
    // Move resources
    settings_ = std::move(other.settings_);
    last_stats_ = std::move(other.last_stats_);
    // inference_server_ = std::move(other.// inference_server_);
    // burst_coordinator_ = std::move(other.// burst_coordinator_);
    root_ = std::move(other.root_);
    shutdown_ = other.shutdown_.load();
    active_simulations_ = other.active_simulations_.load();
    search_running_ = other.search_running_.load();
    random_engine_ = std::move(other.random_engine_);
    transposition_table_ = std::move(other.transposition_table_);
    use_transposition_table_ = other.use_transposition_table_;
    pending_evaluations_ = other.pending_evaluations_.load();
    batch_counter_ = other.batch_counter_.load();
    total_leaves_generated_ = other.total_leaves_generated_.load();
    total_results_processed_ = other.total_results_processed_.load();
    leaf_queue_ = std::move(other.leaf_queue_);
    batch_queue_ = std::move(other.batch_queue_);
    result_queue_ = std::move(other.result_queue_);
    result_distributor_worker_ = std::move(other.result_distributor_worker_);
    workers_active_ = other.workers_active_.load();
    
    // Clean up source object
    other.workers_active_ = false;
    
    if (other.result_distributor_worker_.joinable()) {
        other.result_distributor_worker_.join();
    }
    
    other.search_running_ = false;
    other.active_simulations_ = 0;
}

// Move assignment operator
MCTSEngine& MCTSEngine::operator=(MCTSEngine&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        shutdown_ = true;
        workers_active_ = false;
        
        if (result_distributor_worker_.joinable()) {
            result_distributor_worker_.join();
        }
        
        safelyStopEvaluator();
        
        // Move resources from other
        settings_ = std::move(other.settings_);
        last_stats_ = std::move(other.last_stats_);
        // inference_server_ = std::move(other.// inference_server_);
        // burst_coordinator_ = std::move(other.// burst_coordinator_);
        root_ = std::move(other.root_);
        shutdown_ = other.shutdown_.load();
        active_simulations_ = other.active_simulations_.load();
        search_running_ = other.search_running_.load();
        random_engine_ = std::move(other.random_engine_);
        transposition_table_ = std::move(other.transposition_table_);
        use_transposition_table_ = other.use_transposition_table_;
        pending_evaluations_ = other.pending_evaluations_.load();
        batch_counter_ = other.batch_counter_.load();
        total_leaves_generated_ = other.total_leaves_generated_.load();
        total_results_processed_ = other.total_results_processed_.load();
        leaf_queue_ = std::move(other.leaf_queue_);
        batch_queue_ = std::move(other.batch_queue_);
        result_queue_ = std::move(other.result_queue_);
        result_distributor_worker_ = std::move(other.result_distributor_worker_);
        workers_active_ = other.workers_active_.load();
        
        // Clean up source object
        other.shutdown_ = true;
        other.workers_active_ = false;
        
        if (other.result_distributor_worker_.joinable()) {
            other.result_distributor_worker_.join();
        }
        
        other.search_running_ = false;
        other.active_simulations_ = 0;
    }
    
    return *this;
}

// Destructor
MCTSEngine::~MCTSEngine() {
    // Phase 1: Signal shutdown to all components
    shutdown_.store(true, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    active_simulations_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Stop memory pressure monitor
    if (memory_pressure_monitor_) {
        memory_pressure_monitor_->stop();
    }
    
    // Phase 2: Stop the evaluator
    safelyStopEvaluator();
    
    // Phase 3: Clear all queues to prevent stuck threads
    {
        PendingEvaluation temp_eval;
        while (leaf_queue_.try_dequeue(temp_eval)) {
            if (temp_eval.node) {
                try {
                    temp_eval.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
        
        BatchInfo temp_batch;
        while (batch_queue_.try_dequeue(temp_batch)) {
            for (auto& eval : temp_batch.evaluations) {
                if (eval.node) {
                    try {
                        eval.node->clearEvaluationFlag();
                    } catch (...) {}
                }
            }
        }
        
        std::pair<NetworkOutput, PendingEvaluation> temp_result;
        while (result_queue_.try_dequeue(temp_result)) {
            if (temp_result.second.node) {
                try {
                    temp_result.second.node->clearEvaluationFlag();
                } catch (...) {}
            }
        }
    }
    
    // Phase 4: Join worker threads
    if (result_distributor_worker_.joinable()) {
        result_distributor_worker_.join();
    }
    
    // Phase 5: Final cleanup
    if (transposition_table_) {
        transposition_table_->clear();
    }
    
    root_.reset();
}

} // namespace mcts
} // namespace alphazero