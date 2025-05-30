#include "mcts/mcts_engine.h"
#include "utils/advanced_memory_monitor.h"
// #include "mcts/aggressive_memory_manager.h" // Using local implementation
#include "utils/progress_bar.h"
#include "utils/shutdown_manager.h"

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include "mcts/mcts_node.h"
#include "utils/debug_monitor.h"
#include "utils/debug_logger.h"
#include "utils/gamestate_pool.h"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <condition_variable>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>
#include <sstream>

// 🚀 TASKFLOW INTEGRATION for work-stealing CPU optimization
#include <taskflow/taskflow.hpp>

// 🚀 INTEGRATE LEGACY MEMORY MANAGEMENT COMPONENTS
// #include "mcts/unified_memory_manager.h" - removed
// #include "mcts/advanced_memory_pool.h" - removed
#include "utils/gpu_memory_manager.h"
// #include "core/tensor_pool.h" // Removed - no longer using tensor pools
#include "utils/memory_tracker.h"

// 🚀 LOCK-FREE BATCH ACCUMULATOR for maximum batching efficiency
// #include "mcts/lock_free_batch_accumulator.h" // Removed in simplification

namespace alphazero {
namespace mcts {

// 🚨 UNIFIED AGGRESSIVE MEMORY CONTROL SYSTEM - Integrates All Legacy Components
class UnifiedAggressiveMemoryController {
private:
    static constexpr size_t GB = 1024 * 1024 * 1024;
    static constexpr size_t MB = 1024 * 1024;
    
    // Memory thresholds (in bytes)
    size_t warning_threshold_ = 28 * GB;    // 28GB warning
    size_t critical_threshold_ = 30 * GB;   // 30GB critical
    size_t emergency_threshold_ = 31 * GB;  // 31GB emergency
    
    std::atomic<size_t> current_memory_usage_{0};
    std::atomic<int> memory_pressure_level_{0}; // 0=normal, 1=warning, 2=critical, 3=emergency
    std::chrono::steady_clock::time_point last_check_time_;
    
    // 🚀 LEGACY MEMORY COMPONENTS INTEGRATION
    // UnifiedMemoryManager* unified_memory_manager_; // Removed in simplification
    utils::GPUMemoryManager* gpu_memory_manager_;
    // core::GlobalTensorPool* tensor_pool_; // Removed
    // std::unique_ptr<AdvancedMemoryPool> // advanced_memory_pool_; // Removed
    bool legacy_components_initialized_;
    
    // 🚀 LEGACY MEMORY STATS TRACKING
    size_t last_gpu_allocated_ = 0;
    // Tensor pool size tracking removed
    size_t last_unified_usage_ = 0;
    
public:
    UnifiedAggressiveMemoryController() 
        : last_check_time_(std::chrono::steady_clock::now())
        , legacy_components_initialized_(false) {
        initializeLegacyComponents();
    }
    
    // Get current memory usage in bytes
    size_t getCurrentMemoryUsage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        // Convert from KB to bytes
        size_t memory_kb = static_cast<size_t>(usage.ru_maxrss);
        return memory_kb * 1024; // Linux reports in KB
    }
    
    // Check memory pressure and return recommended action
    int checkMemoryPressure() {
        auto current_time = std::chrono::steady_clock::now();
        auto time_since_check = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_check_time_);
        
        // Check every 100ms
        if (time_since_check.count() < 100) {
            return memory_pressure_level_.load();
        }
        
        last_check_time_ = current_time;
        size_t current_memory = getCurrentMemoryUsage();
        current_memory_usage_.store(current_memory);
        
        // 🚀 ENHANCED MEMORY ANALYSIS: Include legacy component stats
        size_t total_managed_memory = current_memory;
        
        if (legacy_components_initialized_) {
            try {
                // GPU Memory Manager stats
                if (gpu_memory_manager_) {
                    auto gpu_stats = gpu_memory_manager_->getStats();
                    total_managed_memory += gpu_stats.allocated_bytes;
                    last_gpu_allocated_ = gpu_stats.allocated_bytes;
                }
                
                // Tensor pool stats removed
                
                // UnifiedMemoryManager was removed in simplification
            } catch (...) {
                // Ignore errors in stats collection
            }
        }
        
        int new_pressure_level = 0;
        if (total_managed_memory >= emergency_threshold_) {
            new_pressure_level = 3; // EMERGENCY
        } else if (total_managed_memory >= critical_threshold_) {
            new_pressure_level = 2; // CRITICAL  
        } else if (total_managed_memory >= warning_threshold_) {
            new_pressure_level = 1; // WARNING
        }
        
        int old_level = memory_pressure_level_.exchange(new_pressure_level);
        
        // Log pressure level changes with comprehensive stats
        if (new_pressure_level > old_level) {
            std::cout << "🚨 UNIFIED MEMORY PRESSURE: " << formatMemoryUsage(total_managed_memory) 
                      << " - Level " << new_pressure_level;
            switch (new_pressure_level) {
                case 1: std::cout << " (WARNING)"; break;
                case 2: std::cout << " (CRITICAL)"; break;
                case 3: std::cout << " (EMERGENCY)"; break;
            }
            std::cout << " | System: " << formatMemoryUsage(current_memory);
            if (legacy_components_initialized_) {
                std::cout << " | GPU: " << formatMemoryUsage(last_gpu_allocated_)
                          // Tensor pool stats removed
                          << " | Unified: " << formatMemoryUsage(last_unified_usage_);
            }
            std::cout << std::endl;
        }
        
        return new_pressure_level;
    }
    
    // 🚀 INITIALIZE LEGACY MEMORY COMPONENTS
    void initializeLegacyComponents() {
        if (legacy_components_initialized_) return;
        
        try {
            // Initialize UnifiedMemoryManager
            // unified_memory_manager_ = &// UnifiedMemoryManager::getInstance();
            
            // Initialize GPUMemoryManager  
            gpu_memory_manager_ = &utils::GPUMemoryManager::getInstance();
            gpu_memory_manager_->initialize(2 * GB, 8 * GB); // 2GB initial, 8GB max
            
            // Tensor pool initialization removed
            
            // Advanced memory pool removed
            /*
            AdvancedMemoryPoolConfig config;
            config.initial_size = 50000;
            config.max_pool_size = 500000;
            config.enable_stats = true;
            // advanced_memory_pool_ = std::make_unique<AdvancedMemoryPool>(config);
            */
            
            legacy_components_initialized_ = true;
            std::cout << "🚀 Legacy memory components initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "⚠️  Warning: Some legacy memory components failed to initialize: " << e.what() << std::endl;
        }
    }
    
    // Get adaptive batch size based on memory pressure
    int getAdaptiveBatchSize(int base_batch_size) {
        int pressure = checkMemoryPressure();
        switch (pressure) {
            case 0: return base_batch_size;           // Normal: full batch size
            case 1: return base_batch_size * 3 / 4;   // Warning: 75% of base
            case 2: return base_batch_size / 2;       // Critical: 50% of base  
            case 3: return std::max(4, base_batch_size / 4); // Emergency: 25% of base, min 4
            default: return base_batch_size;
        }
    }
    
    // 🚀 ENHANCED EMERGENCY CLEANUP - Uses All Legacy Components
    void emergencyCleanup() {
        std::cout << "🚨 COMPREHENSIVE EMERGENCY CLEANUP INITIATED!" << std::endl;
        
        // Record memory before cleanup
        size_t memory_before = getCurrentMemoryUsage();
        
        // 1. CUDA Memory cleanup
        #ifdef WITH_TORCH
        if (torch::cuda::is_available()) {
            try {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
                std::cout << "🔧 CUDA cache cleared" << std::endl;
            } catch (...) {
                std::cout << "⚠️  CUDA cleanup failed" << std::endl;
            }
        }
        #endif
        
        // 2. GPU Memory Manager cleanup
        if (legacy_components_initialized_ && gpu_memory_manager_) {
            try {
                auto stats_before = gpu_memory_manager_->getStats();
                std::cout << "🔧 GPU Memory before cleanup: " << formatMemoryUsage(stats_before.allocated_bytes) 
                          << " (peak: " << formatMemoryUsage(stats_before.peak_allocated) << ")" << std::endl;
                
                gpu_memory_manager_->reset();
                
                auto stats_after = gpu_memory_manager_->getStats();
                std::cout << "🔧 GPU Memory after cleanup: " << formatMemoryUsage(stats_after.allocated_bytes) << std::endl;
            } catch (const std::exception& e) {
                std::cout << "⚠️  GPU Manager cleanup error: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "⚠️  GPU Manager cleanup failed" << std::endl;
            }
        }
        
        // 3. Tensor pool cleanup removed
        
        // 4. Advanced Memory Pool removed
        /*
        if (advanced_memory_pool_) {
            try {
                auto stats_before = advanced_memory_pool_->getStats();
                std::cout << "🔧 Advanced Pool before: " << stats_before.nodes_in_use << " nodes, " 
                          << stats_before.states_in_use << " states, " 
                          << stats_before.peak_usage << " peak usage" << std::endl;
                
                // Reset stats to release memory indirectly
                advanced_memory_pool_->resetStats();
                
                auto stats_after = advanced_memory_pool_->getStats();
                std::cout << "🔧 Advanced Pool after: " << stats_after.nodes_in_use << " nodes, " 
                          << stats_after.states_in_use << " states" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "⚠️  Advanced Pool cleanup error: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "⚠️  Advanced Pool cleanup failed" << std::endl;
            }
        }
        */
        
        // 5. UnifiedMemoryManager was removed in simplification
        
        // 6. GameState Pool cleanup
        try {
            auto& pool_manager = utils::GameStatePoolManager::getInstance();
            pool_manager.clearAllPools();
            std::cout << "🔧 GameState pools cleared" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "⚠️  GameState Pool cleanup error: " << e.what() << std::endl;
        } catch (...) {
            std::cout << "⚠️  GameState Pool cleanup failed" << std::endl;
        }
        
        // 7. System-level cleanup
        try {
            // Force garbage collection and system cleanup
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (...) {}
        
        // 8. Verify cleanup effectiveness
        size_t memory_after = getCurrentMemoryUsage();
        size_t memory_saved = (memory_before > memory_after) ? (memory_before - memory_after) : 0;
        
        std::cout << "🧹 COMPREHENSIVE CLEANUP COMPLETED!" << std::endl;
        std::cout << "💾 Memory: " << formatMemoryUsage(memory_before) << " → " 
                  << formatMemoryUsage(memory_after);
        if (memory_saved > 0) {
            std::cout << " (saved: " << formatMemoryUsage(memory_saved) << ")";
        }
        std::cout << std::endl;
        
        // Update internal tracking
        current_memory_usage_.store(memory_after);
        last_gpu_allocated_ = 0;
        // Tensor pool references removed
        last_unified_usage_ = 0;
    }
    
    // 🚀 GET COMPREHENSIVE MEMORY REPORT
    std::string getComprehensiveMemoryReport() {
        std::ostringstream report;
        report << "💾 UNIFIED MEMORY REPORT:\n";
        report << "  System: " << getCurrentMemoryString() << "\n";
        
        if (legacy_components_initialized_) {
            report << "  GPU: " << formatMemoryUsage(last_gpu_allocated_) << "\n";
            // Tensor pool stats removed
            report << "  Unified: " << formatMemoryUsage(last_unified_usage_) << "\n";
            
            // Advanced pool removed
            /*
            if (advanced_memory_pool_) {
                try {
                    auto stats = advanced_memory_pool_->getStats();
                    report << "  Pool allocations: " << stats.total_allocations << "\n";
                    report << "  Pool peak usage: " << stats.peak_usage << "\n";
                } catch (...) {}
            }
            */
        }
        
        int pressure = memory_pressure_level_.load();
        report << "  Pressure: Level " << pressure;
        switch (pressure) {
            case 1: report << " (WARNING ⚠️)"; break;
            case 2: report << " (CRITICAL 🚨)"; break;
            case 3: report << " (EMERGENCY 🔥)"; break;
            default: report << " (NORMAL ✅)"; break;
        }
        
        return report.str();
    }
    
    // Format memory usage for display
    std::string formatMemoryUsage(size_t bytes) {
        if (bytes >= GB) {
            return std::to_string(bytes / GB) + "." + std::to_string((bytes % GB) / (100 * MB)) + "GB";
        } else {
            return std::to_string(bytes / MB) + "MB";
        }
    }
    
    // Get current memory usage for display
    std::string getCurrentMemoryString() {
        return formatMemoryUsage(current_memory_usage_.load());
    }
}; // End of UnifiedAggressiveMemoryController class

// 🚀 TYPE ALIAS for backward compatibility
using AggressiveMemoryController = UnifiedAggressiveMemoryController;

// Renamed from runSearch to better reflect its function
void MCTSEngine::runSearch(const core::IGameState& state) {
    // std::cout << "[MCTS_DEBUG] runSearch: Starting search execution" << std::endl;
    
    // Check for shutdown before starting search
    if (utils::isShutdownRequested()) {
        // std::cout << "[MCTS_DEBUG] runSearch: Shutdown requested, aborting search" << std::endl;
        return;
    }
    
    auto search_start = std::chrono::steady_clock::now();
    
    try {
        // Sequential steps to initialize and run the search

        // Step 1: Create the root node with the current state (always fresh for standard search)
        // std::cout << "[MCTS_DEBUG] runSearch: Creating root node" << std::endl;
        root_ = createRootNode(state);
        // std::cout << "[MCTS_DEBUG] runSearch: Root node created successfully, ptr: " << root_.get() << std::endl;
        
        // Step 2: Initialize game state pool if enabled
        // std::cout << "[MCTS_DEBUG] runSearch: Initializing game state pool" << std::endl;
        initializeGameStatePool(state);
        
        // Step 3: Set up batch parameters for the evaluator
        // std::cout << "[MCTS_DEBUG] runSearch: Setting up batch parameters" << std::endl;
        setupBatchParameters();
        
        // Step 4: Expand the root node to prepare for search
        // std::cout << "[MCTS_DEBUG] runSearch: Expanding root node (isTerminal=" << root_->isTerminal() << ")" << std::endl;
        if (!root_->isTerminal()) {
            expandNonTerminalLeaf(root_);
            // std::cout << "[MCTS_DEBUG] runSearch: Root node expanded successfully" << std::endl;
        }
        
        // Step 5: Reset search statistics and prepare for new search
        resetSearchState();
        
        // Step 6: Create search roots (root parallelization removed - always single root)
        std::vector<std::shared_ptr<MCTSNode>> search_roots;
        search_roots.push_back(root_);
        
        // Step 7: Execute the main search algorithm based on selected method
        auto exec_start = std::chrono::steady_clock::now();
        
        // STREAMLINED SEARCH - Use only the best performing batch tree method
        // std::cout << "[MCTS_DEBUG] runSearch: Starting main search execution (num_threads=" << settings_.num_threads << ")" << std::endl;
        if (!search_roots.empty() && search_roots[0]) {
            if (settings_.num_threads <= 1) {
                // Serial mode for single thread
                // std::cout << "[MCTS_DEBUG] runSearch: Executing simple serial search" << std::endl;
                executeSimpleSerialSearch(search_roots);
                // std::cout << "[MCTS_DEBUG] runSearch: Serial search completed" << std::endl;
            } else {
                // Batch tree search for multi-threaded
                // std::cout << "[MCTS_DEBUG] runSearch: About to call executeBatchedTreeSearch" << std::endl;
                // std::cout << "[MCTS_DEBUG] runSearch: Creating root_state clone" << std::endl;
                auto root_state = state.clone();
                // std::cout << "[MCTS_DEBUG] runSearch: root_state cloned successfully, calling executeBatchedTreeSearch" << std::endl;
                executeBatchedTreeSearch(search_roots[0].get(), std::move(root_state));
                // std::cout << "[MCTS_DEBUG] runSearch: Batched tree search completed" << std::endl;
            }
        } else {
            LOG_SYSTEM_ERROR("runSearch: No valid search roots available");
        }
        
        auto exec_end = std::chrono::steady_clock::now();
        
        // Step 8: No aggregation needed (root parallelization removed)
        
        // Step 9: Update search statistics before returning
        countTreeStatistics();
        
    } catch (const std::exception& e) {
        // Log the error
        std::cerr << "Exception during MCTS search: " << e.what() << std::endl;
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow to allow caller to handle the error
        throw;
    } catch (...) {
        // Handle unknown exceptions
        std::cerr << "Unknown exception during MCTS search" << std::endl;
        
        // Reset search state
        search_running_.store(false, std::memory_order_release);
        active_simulations_.store(0, std::memory_order_release);
        
        // Rethrow with a more descriptive message
        throw std::runtime_error("Unknown error occurred during MCTS search");
    }
}

// Initialize game state pool for better performance
void MCTSEngine::initializeGameStatePool(const core::IGameState& state) {
    if (game_state_pool_enabled_ && !utils::GameStatePoolManager::getInstance().hasPool(state.getGameType())) {
        try {
            // Initialize with reasonable defaults
            size_t pool_size = settings_.num_simulations / 4;  // Estimate based on simulations
            utils::GameStatePoolManager::getInstance().initializePool(state.getGameType(), pool_size);
        } catch (const std::exception& e) {
        }
    }
}

// Reset the search state for a new search
void MCTSEngine::resetSearchState() {
    // Initialize statistics for the new search
    last_stats_ = MCTSStats();
    last_stats_.tt_size = transposition_table_ ? transposition_table_->size() : 0;
    
    total_leaves_generated_.store(0, std::memory_order_release);
    pending_evaluations_.store(0, std::memory_order_release);
    
    // Set search running flag
    search_running_.store(true, std::memory_order_release);
    
    int sim_count = std::max(500, settings_.num_simulations);
    
    active_simulations_.store(0, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    active_simulations_.store(sim_count, std::memory_order_release);
    
    int actual_value = active_simulations_.load(std::memory_order_acquire);
    if (actual_value != sim_count) {
        active_simulations_ = sim_count;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        actual_value = active_simulations_.load(std::memory_order_acquire);
    }
    
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        active_simulations_.store(sim_count, std::memory_order_seq_cst);
    }
}

// Simple serial search implementation that bypasses complex coordinators
void MCTSEngine::executeSimpleSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    // std::cout << "[MCTS_DEBUG] executeSimpleSerialSearch: Starting" << std::endl;
    
    if (search_roots.empty() || !search_roots[0]) {
        std::cerr << "[ERROR] No search roots available" << std::endl;
        return;
    }
    
    std::shared_ptr<MCTSNode> root = search_roots[0];
    // std::cout << "[MCTS_DEBUG] Using root ptr: " << root.get() << std::endl;
    
    // CRITICAL FIX: Adaptive batch sizing for efficiency
    int base_batch_size = settings_.batch_size;
    std::vector<std::shared_ptr<MCTSNode>> batch_leaves;
    std::vector<std::vector<std::shared_ptr<MCTSNode>>> batch_paths;
    batch_leaves.reserve(base_batch_size);
    batch_paths.reserve(base_batch_size);
    
    // Ensure root is expanded
    if (!root->isTerminal() && !root->isExpanded()) {
        // std::cout << "[MCTS_DEBUG] Expanding root node" << std::endl;
        expandNonTerminalLeaf(root);
    }
    
    // std::cout << "[MCTS_DEBUG] Starting main simulation loop for " << settings_.num_simulations << " simulations" << std::endl;
    
    // Run simulations with adaptive batching for efficiency
    for (int sim = 0; sim < settings_.num_simulations && !utils::isShutdownRequested(); ) {
        // Collect a batch of leaf nodes
        batch_leaves.clear();
        batch_paths.clear();
        
        // ADAPTIVE BATCH SIZE: Reduce batch size as we near completion
        int remaining = settings_.num_simulations - sim;
        int target_batch_size = base_batch_size;
        if (remaining < base_batch_size / 4) {
            target_batch_size = remaining;  // Final cleanup
        } else if (remaining < base_batch_size / 2) {
            target_batch_size = base_batch_size / 4;
        } else if (remaining < base_batch_size) {
            target_batch_size = base_batch_size / 2;
        }
        
        // std::cout << "[MCTS_DEBUG] Simulation " << sim << ", target batch size: " << target_batch_size << std::endl;
        
        int batch_start = sim;
        // Collect exactly target_batch_size leaves
        while (batch_leaves.size() < static_cast<size_t>(target_batch_size) && sim < settings_.num_simulations) {
        // Selection phase: traverse tree to find leaf
        std::shared_ptr<MCTSNode> leaf = root;
        std::vector<std::shared_ptr<MCTSNode>> path = {root};
        
        // std::cout << "[MCTS_DEBUG] Starting tree traversal for simulation " << sim << std::endl;
        
        while (!leaf->isTerminal() && leaf->isExpanded() && !leaf->getChildren().empty()) {
            std::shared_ptr<MCTSNode> selected = leaf->selectChild(settings_.exploration_constant);
            if (!selected) {
                // std::cout << "[MCTS_DEBUG] No child selected, breaking" << std::endl;
                break;
            }
            
            // std::cout << "[MCTS_DEBUG] Selected child ptr: " << selected.get() 
            //           << ", action: " << selected->getAction() << std::endl;
            
            // Check transposition table for the selected child
            if (use_transposition_table_ && transposition_table_) {
                uint64_t hash = selected->getState().getHash();
                auto tt_node = transposition_table_->lookup(hash);
                if (tt_node && tt_node != selected) {
                    // std::cout << "[MCTS_DEBUG] Found TT node ptr: " << tt_node.get() 
                    //           << ", visits: " << tt_node->getVisitCount() 
                    //           << " vs selected visits: " << selected->getVisitCount() << std::endl;
                    // Found a transposition - use the stored node if it has more visits
                    if (tt_node->getVisitCount() > selected->getVisitCount()) {
                        // std::cout << "[MCTS_DEBUG] Using TT node instead of selected" << std::endl;
                        selected = tt_node;
                    }
                }
            }
            
            leaf = selected;
            path.push_back(leaf);
        }
        
        // Expansion phase: expand leaf if not terminal
        if (!leaf->isTerminal() && !leaf->isExpanded()) {
            expandNonTerminalLeaf(leaf);
            
            // Store in transposition table if enabled
            if (use_transposition_table_ && transposition_table_) {
                uint64_t hash = leaf->getState().getHash();
                transposition_table_->store(hash, leaf, path.size());
            }
            
            if (!leaf->getChildren().empty()) {
                // Select first child as the new leaf
                leaf = leaf->getChildren()[0];
                path.push_back(leaf);
            }
        }
        
        // Evaluation phase: get value for the leaf
        float value = 0.0f;
        if (leaf->isTerminal()) {
            // Terminal node evaluation
            auto& state = leaf->getState();
            auto result = state.getGameResult();
            switch (result) {
                case core::GameResult::WIN_PLAYER1:
                    value = (leaf->getState().getCurrentPlayer() == 1) ? 1.0f : -1.0f;
                    break;
                case core::GameResult::WIN_PLAYER2:
                    value = (leaf->getState().getCurrentPlayer() == 2) ? 1.0f : -1.0f;
                    break;
                case core::GameResult::DRAW:
                    value = 0.0f;
                    break;
                default:
                    value = 0.0f;
                    break;
            }
        } else {
            // Non-terminal leaf - add to batch
            batch_leaves.push_back(leaf);
            batch_paths.push_back(path);
        }
        
        sim++;
        }
        
        // Process batch if we have any non-terminal leaves
        if (!batch_leaves.empty()) {
            try {
                // Prepare states for batch inference
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.reserve(batch_leaves.size());
                for (auto& leaf : batch_leaves) {
                    states.push_back(leaf->getState().clone());
                }
                
                // Batch inference
                if (direct_inference_fn_) {
                    auto inference_start = std::chrono::steady_clock::now();
                    auto results = direct_inference_fn_(states);
                    auto inference_end = std::chrono::steady_clock::now();
                    auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();
                    
                    // CRITICAL FIX: Track neural network statistics
                    last_stats_.total_evaluations += batch_leaves.size();
                    last_stats_.total_batches_processed += 1;
                    last_stats_.avg_batch_size = (last_stats_.avg_batch_size * (last_stats_.total_batches_processed - 1) + batch_leaves.size()) / last_stats_.total_batches_processed;
                    last_stats_.avg_batch_latency = std::chrono::milliseconds(
                        (last_stats_.avg_batch_latency.count() * (last_stats_.total_batches_processed - 1) + inference_ms) / last_stats_.total_batches_processed
                    );
                    
                    // Process results
                    for (size_t i = 0; i < batch_leaves.size() && i < results.size(); ++i) {
                        auto& leaf = batch_leaves[i];
                        auto& path = batch_paths[i];
                        float value = results[i].value;
                        
                        // Set policy probabilities if leaf has children
                        if (!leaf->getChildren().empty() && !results[i].policy.empty()) {
                            leaf->setPriorProbabilities(results[i].policy);
                        }
                        
                        // Backpropagation
                        for (auto& node : path) {
                            node->update(value);
                            value = -value; // Flip value for opponent
                        }
                    }
                    
                    // Clear states immediately to free memory
                    states.clear();
                    
                    // CRITICAL FIX: Minimal CUDA cleanup for 8GB VRAM - only every 100 batches
                    #ifdef WITH_TORCH
                    if (torch::cuda::is_available() && last_stats_.total_batches_processed % 100 == 0) {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
                    #endif
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Batch inference failed: " << e.what() << std::endl;
            }
        }
    }
}

// Implementation of the serial search approach with leaf batching
void MCTSEngine::executeSerialSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    // Track search start time
    auto search_start_time = std::chrono::steady_clock::now();
    
    // Thread-local leaf storage for batching
    std::vector<PendingEvaluation> leaf_batch;
    const size_t OPTIMAL_BATCH_SIZE = settings_.batch_size;
    leaf_batch.reserve(OPTIMAL_BATCH_SIZE);
    
    // Counters for monitoring
    int consecutive_empty_tries = 0;
    const int MAX_EMPTY_TRIES = 3;
    
    // OPTIMIZED: Burst search persistence counters 
    int consecutive_empty_bursts = 0;
    const int MAX_CONSECUTIVE_EMPTY = 6;
    
    if (search_roots.empty()) {
        return;
    }
    
    if (!search_roots[0]) {
        return;
    }
    
    std::shared_ptr<MCTSNode> main_root = search_roots[0];
    
    if (!main_root->isTerminal()) {
        expandNonTerminalLeaf(main_root);
    }
    
    if (active_simulations_.load(std::memory_order_acquire) <= 0) {
        active_simulations_.store(100, std::memory_order_release);
    }
    
    int main_loop_iterations = 0;
    
    while (active_simulations_.load(std::memory_order_acquire) > 0) {
        main_loop_iterations++;
        
        // CRITICAL FIX: Add termination conditions to prevent infinite loops
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - search_start_time);
        
        // Force termination after reasonable time limit
        if (elapsed_time.count() > 30) { // 30 seconds maximum
            // Emergency stop after 30 seconds
            active_simulations_.store(0, std::memory_order_release);
            break;
        }
        
        // Force termination after too many iterations without progress
        if (main_loop_iterations > 1000) {
            // Emergency stop after too many iterations
            active_simulations_.store(0, std::memory_order_release);
            break;
        }
        
        // Check if we should wait for pending evaluations to complete
        if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        auto batch_start_time = std::chrono::steady_clock::now();
        const auto MAX_BATCH_COLLECTION_TIME = std::chrono::milliseconds(50);
        
        // Force at least one iteration initially
        bool force_execution = true;
        
        const size_t MIN_REQUIRED_BATCH = std::max(static_cast<size_t>(48), static_cast<size_t>(OPTIMAL_BATCH_SIZE * 0.75));
        
        while ((leaf_batch.size() < OPTIMAL_BATCH_SIZE && 
                active_simulations_.load(std::memory_order_acquire) > 0 &&
                (leaf_batch.size() < MIN_REQUIRED_BATCH || (std::chrono::steady_clock::now() - batch_start_time) < MAX_BATCH_COLLECTION_TIME) &&
                pending_evaluations_.load(std::memory_order_acquire) < settings_.batch_size * 4) || 
               force_execution) {
            
            // Reset force execution flag after first iteration
            force_execution = false;
            
            // Start a new simulation by claiming nodes from the counter
            int old_sims = active_simulations_.load(std::memory_order_acquire);
            
            // Don't add more simulations - let the counter decrease to 0
            if (old_sims <= 0) {
                break; // Exit if no simulations left
            }
            
            int simulations_to_claim = std::min(static_cast<int>(OPTIMAL_BATCH_SIZE * 0.9), old_sims);
            simulations_to_claim = std::max(32, simulations_to_claim);
            
            bool claimed = active_simulations_.compare_exchange_weak(
                old_sims, old_sims - simulations_to_claim, std::memory_order_acq_rel);
            
            if (!claimed) {
                // Try again in next iteration - don't add more simulations
                continue;
            }
            
            if (claimed) {
                int leaves_found = 0;
                
                // Direct batch evaluation without BurstCoordinator
                // Collect leaf nodes for batch evaluation
                std::vector<std::shared_ptr<MCTSNode>> batch_nodes;
                std::vector<std::vector<std::shared_ptr<MCTSNode>>> batch_paths;
                
                // Try to collect a batch of leaf nodes
                for (int i = 0; i < simulations_to_claim && batch_nodes.size() < settings_.batch_size; ++i) {
                    // Select a search root
                    auto& current_root = search_roots[i % search_roots.size()];
                    
                    // Select path to leaf
                    std::vector<std::shared_ptr<MCTSNode>> path;
                    auto leaf_result = selectLeafNode(current_root);
                    path = leaf_result.second;
                    
                    if (leaf_result.first && !path.empty()) {
                        auto leaf_node = leaf_result.first;
                        
                        // Skip terminal nodes
                        if (!leaf_node->isTerminal()) {
                            // Expand if needed
                            if (!leaf_node->isExpanded()) {
                                expandNonTerminalLeaf(leaf_node);
                            }
                            
                            batch_nodes.push_back(leaf_node);
                            batch_paths.push_back(path);
                            leaves_found++;
                        }
                    }
                }
                
                // Evaluate batch if we have nodes
                if (!batch_nodes.empty()) {
                    consecutive_empty_bursts = 0;
                    
                    // Prepare states for neural network
                    std::vector<std::unique_ptr<core::IGameState>> states;
                    for (auto& node : batch_nodes) {
                        states.push_back(node->getState().clone());
                    }
                    
                    // Direct neural network evaluation
                    auto results = neural_network_->inference(states);
                    
                    // Backpropagate results
                    for (size_t i = 0; i < results.size() && i < batch_paths.size(); ++i) {
                        backPropagate(batch_paths[i], results[i].value);
                    }
                    
                    total_leaves_generated_.fetch_add(leaves_found, std::memory_order_release);
                } else {
                    consecutive_empty_bursts++;
                    if (consecutive_empty_bursts >= MAX_CONSECUTIVE_EMPTY) {
                        std::cout << "[SEARCH] Max consecutive empty batches reached - stopping search" << std::endl;
                        active_simulations_.store(0, std::memory_order_release);
                        break;
                    }
                }
                
                // Update consecutive empty tries counter
                if (leaves_found == 0) {
                    consecutive_empty_tries++;
                } else {
                    consecutive_empty_tries = 0;
                }
            }
        }
        
        // OLD BATCHING LOGIC REMOVED - Using UnifiedInferenceServer instead
        // The unified inference server handles all batching, queuing, and coordination internally
        // This eliminates the complex manual batch formation that was causing deadlocks
        
        leaf_batch.clear(); // Clear any leftover references
        consecutive_empty_tries = 0;
        
        // Process results directly when using shared queues to prevent deadlock
        processPendingSimulations();
        
        // MEMORY FIX: More frequent and aggressive memory cleanup
        if (main_loop_iterations % 10 == 0) {
            // Clear any cached data that might be accumulating
            if (transposition_table_) {
                // More aggressive clearing to prevent memory buildup
                if (transposition_table_->size() > 10000) {
                    transposition_table_->clear();
                }
            }
            
            // MEMORY FIX: Force GPU memory cleanup
            #ifdef TORCH_CUDA_AVAILABLE
            if (torch::cuda::is_available()) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            #endif
            
            // Force garbage collection if needed
            std::this_thread::yield();
        }
        
        // Adaptive wait based on pending evaluations
        if (pending_evaluations_.load(std::memory_order_acquire) > settings_.batch_size * 3) {
            // If too many pending, wait longer to prevent memory overflow
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } else if (leaf_batch.empty() && consecutive_empty_tries >= MAX_EMPTY_TRIES) {
            // If we can't find leaves, check if we're done
            if (active_simulations_.load(std::memory_order_acquire) == 0) {
                break;  // Exit the loop
            }
            std::this_thread::yield();
        }
    }
    
    // Wait for any remaining evaluations to complete
    waitForSimulationsToComplete(search_start_time);
    
    // Record total search time
    auto search_end_time = std::chrono::steady_clock::now();
    last_stats_.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end_time - search_start_time);
    
    // UnifiedInferenceServer was removed in simplification
    
    // Mark search as completed
    search_running_.store(false, std::memory_order_release);
}

void MCTSEngine::executeParallelSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty()) {
        return;
    }
    
    // Use OpenMP for parallel MCTS search
    const int num_threads = std::min(settings_.num_threads, static_cast<int>(std::thread::hardware_concurrency()));
    const int simulations_per_thread = settings_.num_simulations / num_threads;
    const int remaining_simulations = settings_.num_simulations % num_threads;
    
    std::cout << "🔄 Starting parallel MCTS search with " << num_threads << " threads, " 
              << settings_.num_simulations << " total simulations" << std::endl;
    
    // ATOMIC COORDINATION: Thread activity and batch formation monitoring
    std::atomic<int> active_threads(0);
    std::atomic<int> total_leaves_processed(0);
    std::atomic<int> total_nn_calls(0);
    
    // Lock-free coordination for concurrent batch formation
    std::atomic<size_t> pending_evaluations(0);    // Track pending requests
    std::atomic<size_t> batch_formation_round(0);   // Coordinate batch rounds
    std::atomic<bool> batch_ready_signal(false);    // Signal when batch is ready
    
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        int thread_simulations = simulations_per_thread;
        if (thread_id < remaining_simulations) {
            thread_simulations++;
        }
        
        active_threads.fetch_add(1);
        std::cout << "[THREAD_" << thread_id << "] Started with " << thread_simulations << " simulations" << std::endl;
        
        // Thread-local variables
        std::mt19937 thread_rng(std::random_device{}() + thread_id);
        std::vector<std::shared_ptr<MCTSNode>> thread_path;
        
        // Each thread performs its allocated simulations
        for (int sim = 0; sim < thread_simulations; ++sim) {
            // Select a search root (round-robin distribution)
            auto& current_root = search_roots[sim % search_roots.size()];
            
            // Clear path for this simulation
            thread_path.clear();
            
            // Select path to leaf node using the actual method
            auto leaf_result = selectLeafNodeParallel(current_root, thread_path, thread_rng);
            if (!leaf_result.leaf_node || thread_path.empty()) {
                continue;
            }
            
            auto leaf_node = leaf_result.leaf_node;
            
            // Skip if leaf is terminal
            if (leaf_node->isTerminal()) {
                continue;
            }
            
            // Expand leaf if not already expanded
            if (!leaf_node->isExpanded()) {
                expandNonTerminalLeaf(leaf_node);
            }
            
            // Direct neural network evaluation since inference server was removed
            NetworkOutput result;
            {
                // Use neural network directly
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.push_back(leaf_node->getState().clone());
                
                auto nn_start = std::chrono::steady_clock::now();
                
                // Direct neural network call
                auto results = neural_network_->inference(states);
                
                auto nn_end = std::chrono::steady_clock::now();
                auto nn_duration = std::chrono::duration_cast<std::chrono::microseconds>(nn_end - nn_start);
                
                result = results.empty() ? NetworkOutput{} : results[0];
                
                // PHASE 3: Update atomic counters
                pending_evaluations.fetch_sub(1, std::memory_order_relaxed);
                total_nn_calls.fetch_add(1, std::memory_order_relaxed);
                
                // Remove verbose thread logging
            }
            
            // Backpropagate result through the path
            backpropagateParallel(thread_path, result.value, settings_.virtual_loss);
            total_leaves_processed.fetch_add(1);
            
            // AGGRESSIVE memory cleanup every 10 simulations to prevent stacking
            if (sim % 10 == 0) {
                // Force cleanup regardless of pressure to prevent accumulation
                utils::GameStatePoolManager::getInstance().clearAllPools();
                
                // Thread 0 does additional GPU cleanup
                if (thread_id == 0) {
                    try {
#ifdef WITH_TORCH
                        if (torch::cuda::is_available()) {
                            torch::cuda::synchronize();
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        }
#endif
                    } catch (...) {}
                }
            }
        }
        
        active_threads.fetch_sub(1);
        std::cout << "[THREAD_" << thread_id << "] Completed " << thread_simulations 
                  << " simulations" << std::endl;
    }
    
    std::cout << "✅ Parallel MCTS search completed:" << std::endl;
    std::cout << "  - Total leaves processed: " << total_leaves_processed.load() << std::endl;
    std::cout << "  - Total NN calls: " << total_nn_calls.load() << std::endl;
    std::cout << "  - Average batch formation: " << (total_nn_calls.load() > 0 ? 
                 static_cast<float>(total_leaves_processed.load()) / total_nn_calls.load() : 0.0f) << std::endl;
}

// PERFORMANCE FIX: Implement tree reuse between moves for efficiency
mcts::SearchResult MCTSEngine::searchWithTreeReuse(const core::IGameState& state, int last_action) {
    // Only log if verbose logging is enabled
    auto& progress_manager = utils::SelfPlayProgressManager::getInstance();
    if (progress_manager.isVerboseLoggingEnabled()) {
        std::cout << "🌲 Starting search with tree reuse (last_action: " << last_action << ")" << std::endl;
    }
    
    try {
        // Try to reuse existing tree by transitioning to the new state
        if (root_ && last_action >= 0) {
            // Look for a child node that matches the last action
            for (auto& child : root_->getChildren()) {
                if (child && child->getAction() == last_action) {
                    if (progress_manager.isVerboseLoggingEnabled()) {
                        std::cout << "🌲 Found matching child node, reusing tree structure" << std::endl;
                    }
                    
                    // CRITICAL: Clean up siblings to prevent memory accumulation
                    auto old_root = root_;
                    for (auto& sibling : old_root->getChildren()) {
                        if (sibling && sibling != child) {
                            // Clear children by getting reference and clearing the vector
                            sibling->getChildren().clear();
                        }
                    }
                    
                    // Promote child to new root and detach from old parent
                    auto new_root = child;
                    new_root->setParentDirectly(std::weak_ptr<MCTSNode>()); // Clear parent
                    root_ = new_root;
                    
                    // Run search from existing tree
                    runSearch(state);
                    
                    // Create and return search result
                    SearchResult result;
                    if (root_) {
                        auto best_child = root_->selectBestChildUCB(0.0f, random_engine_);
                        if (best_child) {
                            result.action = best_child->getAction();
                            result.value = best_child->getValue();
                        }
                    }
                    return result;
                }
            }
        }
        
        // Fallback to fresh search if tree reuse isn't possible
        std::cout << "🌲 Tree reuse not possible, starting fresh search" << std::endl;
        return search(state);
        
    } catch (const std::exception& e) {
        std::cerr << "🌲 Error in tree reuse search: " << e.what() << std::endl;
        // Fallback to fresh search on error
        return search(state);
    }
}

// ENHANCED SIMPLIFIED PARALLEL SEARCH: Integrates advanced legacy features + aggressive memory control
// Removed - uses components that were removed in simplification
#if 0
void MCTSEngine::executeSimplifiedParallelSearch(const std::vector<std::shared_ptr<MCTSNode>>& search_roots) {
    if (search_roots.empty()) {
        return;
    }
    
    // 🚨 INITIALIZE UNIFIED AGGRESSIVE MEMORY CONTROL
    static UnifiedAggressiveMemoryController memory_controller;
    
    const int num_threads = std::min(settings_.num_threads, static_cast<int>(std::thread::hardware_concurrency()));
    
    // 🚀 ADAPTIVE MEMORY-CONTROLLED BATCHING
    const int base_target_batch_size = std::max(20, num_threads * 2);
    const int base_max_batch_size = std::min(32, base_target_batch_size * 2);
    
    // 🚨 APPLY MEMORY PRESSURE CONTROL
    const int target_batch_size = memory_controller.getAdaptiveBatchSize(base_target_batch_size);
    const int max_batch_size = memory_controller.getAdaptiveBatchSize(base_max_batch_size);
    const int burst_collection_size = num_threads;
    const int adaptive_timeout_ms = 150;
    
    const int total_simulations = settings_.num_simulations;
    
    std::cout << "🔥 ENHANCED SIMPLIFIED PARALLEL SEARCH: " << num_threads << " threads" << std::endl;
    std::cout << "🎯 MEMORY-CONTROLLED BATCHING: target=" << target_batch_size << ", max=" << max_batch_size 
              << ", burst=" << burst_collection_size << ", timeout=" << adaptive_timeout_ms << "ms" << std::endl;
    std::cout << "💾 Current Memory: " << memory_controller.getCurrentMemoryString() << std::endl;
    
    // 🚀 LOCK-FREE BATCH COORDINATION: Ultra-efficient batching without locks
    LockFreeBatchConfig batch_config;
    batch_config.target_batch_size = std::max(static_cast<size_t>(target_batch_size), static_cast<size_t>(16));
    batch_config.max_wait_time = std::chrono::milliseconds(adaptive_timeout_ms / 4);  // Faster response
    
    LockFreeBatchAccumulator batch_accumulator(batch_config);
    
    std::atomic<int> sims_done(0);
    std::atomic<int> batch_num(0);
    
    // 🚀 ADAPTIVE BATCH SIZING: Dynamic GPU utilization optimization
    std::atomic<int> current_batch_target(target_batch_size);
    auto last_batch_time = std::chrono::steady_clock::now();
    
    // 🚀 TASKFLOW WORK-STEALING OPTIMIZATION: Replace OpenMP with dynamic task scheduling
    tf::Executor executor(num_threads);  // Work-stealing scheduler with optimal thread count
    tf::Taskflow taskflow;
    
    // Create parallel tasks for work-stealing leaf collection
    std::vector<tf::Task> leaf_collection_tasks;
    leaf_collection_tasks.reserve(num_threads * 2);  // 2x tasks for better load balancing
    
    for (int task_id = 0; task_id < num_threads * 2; ++task_id) {
        auto task = taskflow.emplace([&, task_id]() {
            std::mt19937 rng(std::random_device{}() + task_id);
            
            // 🔥 WORK-STEALING LEAF COLLECTION: Dynamic load balancing
            while (sims_done.load() < total_simulations) {
                // Single-leaf collection per iteration - maximum throughput
                std::vector<std::shared_ptr<MCTSNode>> path;
                auto& root = search_roots[sims_done.load() % search_roots.size()];
                auto leaf_result = selectLeafNodeParallel(root, path, rng);
                
                if (!leaf_result.leaf_node || path.empty() || leaf_result.leaf_node->isTerminal()) {
                    sims_done.fetch_add(1);
                    continue;
                }
                
                auto leaf = leaf_result.leaf_node;
                if (!leaf->isExpanded()) {
                    expandNonTerminalLeaf(leaf);
                    }
                
                // 🚀 LOCK-FREE BATCH SUBMISSION: Ultra-efficient request submission
                PendingEvaluation pending_eval(leaf, leaf->getState().clone(), path);
                batch_accumulator.submitRequest(std::move(pending_eval));
                
                // 🚀 LOCK-FREE BATCH COLLECTION: Check for ready batches without blocking
                auto ready_batch = batch_accumulator.collectBatch();
                
                // 🚨 MEMORY PRESSURE OVERRIDE: Force collection if memory critical
                int memory_pressure = memory_controller.checkMemoryPressure();
                if (memory_pressure >= 3 && ready_batch.empty()) {
                    // Emergency: Force immediate small batch processing
                    std::cout << "🚨 EMERGENCY: Forcing batch collection due to memory pressure!" << std::endl;
                    // Wait briefly and try again for any available batches
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    ready_batch = batch_accumulator.collectBatch();
                }
                
                // 🔥 PROCESS READY BATCH: Lock-free batch processing
                if (!ready_batch.empty()) {
                const int batch_id = batch_num.fetch_add(1) + 1;
                const int batch_size_actual = ready_batch.size();
                
                // Prepare batch states from PendingEvaluation
                std::vector<std::unique_ptr<core::IGameState>> states;
                states.reserve(batch_size_actual);
                for (auto& pending_eval : ready_batch) {
                    if (pending_eval.state) {
                        states.push_back(pending_eval.state->clone());
                    }
                }
                
                // 🚀 ADVANCED GPU INFERENCE: Monitor and optimize performance
                auto start = std::chrono::steady_clock::now();
                std::vector<NetworkOutput> results;
                
                if (direct_inference_fn_) {
                    // Direct high-performance neural network call
                    results = direct_inference_fn_(states);
                } else {
                    // Optimized fallback with proper sizing
                    results.resize(states.size());
                    for (auto& r : results) {
                        r.value = 0.0f;
                        r.policy.resize(225, 1.0f/225.0f);
                    }
                }
                
                auto end = std::chrono::steady_clock::now();
                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // 🎯 PERFORMANCE ANALYTICS: GPU utilization + memory tracking
                float throughput = static_cast<float>(batch_size_actual) / (duration_ms.count() + 1);
                float gpu_efficiency = std::min(100.0f, (static_cast<float>(batch_size_actual) / max_batch_size) * 100.0f);
                
                // 🚨 MEMORY MONITORING: Display current usage with batch info
                int current_memory_pressure = memory_controller.checkMemoryPressure();
                std::string memory_status = memory_controller.getCurrentMemoryString();
                std::string pressure_indicator = "";
                switch (current_memory_pressure) {
                    case 1: pressure_indicator = " ⚠️"; break;
                    case 2: pressure_indicator = " 🚨"; break; 
                    case 3: pressure_indicator = " 🔥"; break;
                }
                
                std::cout << "[BATCH-" << batch_id << "] 🚀 " << batch_size_actual << " states → " 
                          << duration_us.count() << "μs (" << duration_ms.count() << "ms) | "
                          << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " states/ms | "
                          << "GPU-Eff: " << std::fixed << std::setprecision(1) << gpu_efficiency << "% | "
                          << "Mem: " << memory_status << pressure_indicator << std::endl;
                
                // 🚨 AGGRESSIVE MEMORY PRESSURE RESPONSE
                if (current_memory_pressure >= 3) {
                    // EMERGENCY: Immediate cleanup
                    memory_controller.emergencyCleanup();
                    current_batch_target.store(std::max(4, current_batch_target.load() / 2));
                    std::cout << "🚨 EMERGENCY: Batch size reduced due to memory pressure" << std::endl;
                } else if (current_memory_pressure >= 2) {
                    // CRITICAL: Aggressive cleanup and batch size reduction
                    #ifdef WITH_TORCH
                    if (torch::cuda::is_available()) {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
                    #endif
                    current_batch_target.store(std::max(8, current_batch_target.load() * 3 / 4));
                    std::cout << "🚨 CRITICAL: Batch size reduced for memory safety" << std::endl;
                } else if (batch_size_actual >= target_batch_size && duration_ms.count() < 10) {
                    // Good performance and normal memory
                    std::cout << "🔥 OPTIMAL: " << batch_size_actual << " states processed efficiently" << std::endl;
                } else if (duration_ms.count() > 100) {
                    // Poor performance -> reduce batch size for memory safety
                    std::cout << "⚠️  Slow processing detected. Reducing batch targets." << std::endl;
                    current_batch_target.store(std::max(4, current_batch_target.load() - 4));
                }
                
                // 🧹 PROACTIVE MEMORY CLEANUP: Prevent accumulation
                if (batch_id % 5 == 0 || current_memory_pressure >= 1) {
                    #ifdef WITH_TORCH
                    if (torch::cuda::is_available()) {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
                    #endif
                }
                
                    // 🚀 LOCK-FREE BACKPROPAGATION: Process PendingEvaluation results
                    for (size_t i = 0; i < ready_batch.size() && i < results.size(); ++i) {
                        backpropagateParallel(ready_batch[i].path, results[i].value, settings_.virtual_loss);
                        sims_done.fetch_add(1);
                    }
                }
            } // End while loop
        }); // End lambda
        
        leaf_collection_tasks.push_back(task);
    }
    
    // 🚀 EXECUTE TASKFLOW: Work-stealing parallel execution
    executor.run(taskflow).wait();
    
    // 🚀 SHUTDOWN LOCK-FREE BATCH ACCUMULATOR
    batch_accumulator.shutdown();
    
    std::cout << "🚀 TASKFLOW: Completed work-stealing execution with " 
              << num_threads * 2 << " dynamic tasks" << std::endl;
    std::cout << "🚀 LOCK-FREE: Accumulated " << sims_done.load() 
              << " requests through lock-free batching" << std::endl;
    
    std::cout << "✅ SIMPLIFIED SEARCH done: " << sims_done.load() << " simulations" << std::endl;
    
    // 🚨 FINAL AGGRESSIVE MEMORY CLEANUP WITH MONITORING
    std::cout << "🧹 Starting comprehensive memory cleanup..." << std::endl;
    std::string pre_cleanup_memory = memory_controller.getCurrentMemoryString();
    
    #ifdef WITH_TORCH
    if (torch::cuda::is_available()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
        // Force synchronization to complete all GPU operations
        torch::cuda::synchronize();
    }
    #endif
    
    // 🔥 CRITICAL: Clear any cached game states to prevent accumulation
    if (game_state_pool_enabled_) {
        try {
            utils::GameStatePoolManager::getInstance().clearAllPools();
        } catch (...) {
            // Ignore cleanup errors but prevent crash
        }
    }
    
    // 🚨 VERIFY CLEANUP EFFECTIVENESS
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Allow cleanup to complete
    std::string post_cleanup_memory = memory_controller.getCurrentMemoryString();
    int final_pressure = memory_controller.checkMemoryPressure();
    
    std::cout << "🧹 Memory cleanup completed: " << pre_cleanup_memory << " → " << post_cleanup_memory;
    if (final_pressure > 0) {
        std::cout << " ⚠️  Pressure level: " << final_pressure;
    }
    std::cout << std::endl;
} // End of executeSimplifiedParallelSearch
#endif // 0

} // namespace mcts
} // namespace alphazero