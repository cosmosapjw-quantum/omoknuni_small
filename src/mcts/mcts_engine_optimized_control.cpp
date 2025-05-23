#include "mcts/mcts_engine.h"
#include "mcts/mcts_node.h"
#include "mcts/unified_inference_server.h"
#include "mcts/burst_coordinator.h"
#include "mcts/mcts_object_pool.h"
#include <iostream>

namespace alphazero {
namespace mcts {

/**
 * Advanced control mechanisms for the optimized MCTS engine using the new architecture
 * Provides runtime configuration and optimization controls for UnifiedInferenceServer + BurstCoordinator
 */

void MCTSEngine::enableAdvancedOptimizations(bool enable) {
    use_advanced_optimizations_.store(enable, std::memory_order_release);
    
    if (enable) {
        // Use singleton object pool manager for advanced optimizations
        auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
        auto node_stats = object_pool_manager.getNodePool().getStats();
        std::cout << "Using advanced MCTSObjectPoolManager singleton for optimizations (Node pool size: " 
                  << node_stats.current_pool_size << ", hit rate: " << node_stats.hit_rate << "%" << ")" << std::endl;
        
        // Configure UnifiedInferenceServer for maximum performance
        if (unified_inference_server_) {
            optimizeInferenceServerConfiguration();
            std::cout << "UnifiedInferenceServer optimized for maximum performance" << std::endl;
        }
        
        // Configure BurstCoordinator for advanced batching
        if (burst_coordinator_) {
            optimizeBurstCoordinatorConfiguration();
            std::cout << "BurstCoordinator optimized for advanced batching strategies" << std::endl;
        }
        
        // Enable advanced transposition table optimizations
        if (transposition_table_) {
            enableTranspositionTableOptimizations();
            std::cout << "Transposition table optimizations enabled" << std::endl;
        }
        
        std::cout << "Advanced MCTS optimizations enabled with new architecture" << std::endl;
    } else {
        disableAdvancedOptimizations();
        std::cout << "Advanced MCTS optimizations disabled" << std::endl;
    }
}

void MCTSEngine::configureAdaptiveBatching(bool enable) {
    adaptive_batching_enabled_.store(enable, std::memory_order_release);
    
    if (enable && unified_inference_server_) {
        // Enable dynamic batch size adaptation using available ServerConfig fields
        auto current_config = unified_inference_server_->getConfig();
        
        // Set parameters for optimal GPU utilization using available fields
        current_config.enable_request_coalescing = true;
        current_config.enable_priority_processing = true;
        current_config.target_batch_size = static_cast<size_t>(settings_.batch_size);
        current_config.min_batch_size = static_cast<size_t>(settings_.batch_size * 0.75);
        
        unified_inference_server_->updateConfig(current_config);
        
        std::cout << "Adaptive batching enabled with target batch size: " << current_config.target_batch_size << std::endl;
    }
}

void MCTSEngine::enableBurstPipelining(bool enable, int pipeline_depth) {
    burst_pipelining_enabled_.store(enable, std::memory_order_release);
    
    if (enable && burst_coordinator_) {
        // Configure burst coordination for better pipelining
        auto config = burst_coordinator_->getConfig();
        
        // Adjust timing for more aggressive collection
        config.collection_timeout = std::chrono::milliseconds(std::max(1, pipeline_depth));
        config.target_burst_size = std::max(8, settings_.batch_size / pipeline_depth);
        config.max_parallel_threads = std::max(1, std::min(pipeline_depth, 8));
        
        burst_coordinator_->updateConfig(config);
        
        std::cout << "Burst pipelining configured with parallel threads: " << config.max_parallel_threads << std::endl;
    }
}

void MCTSEngine::setPerformanceMode(PerformanceMode mode) {
    performance_mode_.store(static_cast<int>(mode), std::memory_order_release);
    
    switch (mode) {
        case PerformanceMode::MaximumAccuracy:
            configureForMaximumAccuracy();
            break;
            
        case PerformanceMode::BalancedPerformance:
            configureForBalancedPerformance();
            break;
            
        case PerformanceMode::MaximumSpeed:
            configureForMaximumSpeed();
            break;
            
        case PerformanceMode::EnergyEfficiency:
            configureForEnergyEfficiency();
            break;
    }
}

void MCTSEngine::enableMemoryOptimizations(bool enable) {
    memory_optimizations_enabled_.store(enable, std::memory_order_release);
    
    if (enable) {
        // Use singleton object pool manager for memory optimizations
        auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
        
        auto pending_stats = object_pool_manager.getPendingEvaluationPool().getStats();
        std::cout << "Memory optimizations enabled using MCTSObjectPoolManager singleton (Pending eval pool size: " 
                  << pending_stats.current_pool_size << ", hit rate: " << pending_stats.hit_rate << "%" << ")" << std::endl;
    }
}

void MCTSEngine::configureConcurrencyOptimizations(int max_threads) {
    max_concurrent_threads_ = std::max(1, std::min(max_threads, static_cast<int>(std::thread::hardware_concurrency())));
    
    // Update UnifiedInferenceServer threading
    if (unified_inference_server_) {
        auto config = unified_inference_server_->getConfig();
        config.num_worker_threads = std::min(static_cast<size_t>(8), static_cast<size_t>(max_concurrent_threads_ / 3));
        unified_inference_server_->updateConfig(config);
    }
    
    // Update BurstCoordinator threading
    if (burst_coordinator_) {
        auto config = burst_coordinator_->getConfig();
        config.max_parallel_threads = static_cast<size_t>(max_concurrent_threads_.load());
        burst_coordinator_->updateConfig(config);
    }
    
    std::cout << "Concurrency optimized for " << max_concurrent_threads_ << " threads" << std::endl;
}

void MCTSEngine::enableRealTimeOptimization(bool enable) {
    real_time_optimization_enabled_.store(enable, std::memory_order_release);
    
    if (enable) {
        // Start real-time optimization thread
        if (!optimization_thread_.joinable()) {
            optimization_thread_ = std::thread(&MCTSEngine::realTimeOptimizationWorker, this);
        }
        
        std::cout << "Real-time optimization enabled" << std::endl;
    } else {
        // Stop real-time optimization
        if (optimization_thread_.joinable()) {
            optimization_thread_.join();
        }
        
        std::cout << "Real-time optimization disabled" << std::endl;
    }
}

MCTSEngine::OptimizationMetrics MCTSEngine::getOptimizationMetrics() const {
    OptimizationMetrics metrics;
    
    // Collect metrics from all components
    if (unified_inference_server_) {
        auto server_stats = unified_inference_server_->getStats();
        metrics.average_batch_size = server_stats.getAverageBatchSize();
        metrics.batch_utilization = metrics.average_batch_size / 
                                   unified_inference_server_->getConfig().target_batch_size;
        metrics.inference_throughput = server_stats.total_requests > 0 ? 
            static_cast<float>(server_stats.total_requests * 1000) / server_stats.cumulative_batch_time_ms : 0.0f;
    }
    
    if (burst_coordinator_) {
        auto burst_stats = burst_coordinator_->getEfficiencyStats();
        metrics.burst_efficiency = burst_stats.average_collection_efficiency;
        metrics.coordination_overhead = 1.0f - burst_stats.target_utilization_rate;
    }
    
    // Use singleton object pool manager for stats
    auto& object_pool_manager = MCTSObjectPoolManager::getInstance();
    auto pool_stats = object_pool_manager.getNodePool().getStats();
    metrics.memory_efficiency = static_cast<float>(pool_stats.pool_hits) / 
                               std::max(1.0f, static_cast<float>(pool_stats.total_allocations));
    metrics.pool_utilization = static_cast<float>(pool_stats.current_pool_size) / 
                              std::max(1.0f, static_cast<float>(pool_stats.current_pool_size));
    
    // Calculate overall optimization score
    metrics.overall_optimization_score = (
        metrics.batch_utilization * 0.3f +
        metrics.burst_efficiency * 0.25f +
        metrics.memory_efficiency * 0.2f +
        std::min(1.0f, metrics.inference_throughput / 1000.0f) * 0.25f
    );
    
    return metrics;
}

// Private optimization methods

void MCTSEngine::optimizeInferenceServerConfiguration() {
    if (!unified_inference_server_) return;
    
    auto config = unified_inference_server_->getConfig();
    
    // Optimize based on hardware capabilities
    config.target_batch_size = std::min(static_cast<size_t>(256), static_cast<size_t>(settings_.batch_size * 2));
    config.max_batch_size = config.target_batch_size * 3;
    config.min_batch_size = std::max(static_cast<size_t>(4), config.target_batch_size / 8);
    
    // Optimize timing parameters
    config.max_batch_wait = std::chrono::milliseconds(std::max(5, static_cast<int>(settings_.batch_timeout.count()) / 2));
    config.min_batch_wait = std::chrono::milliseconds(1);
    
    // Advanced optimization features
    config.enable_request_coalescing = true;
    config.enable_priority_processing = true;
    
    unified_inference_server_->updateConfig(config);
}

void MCTSEngine::optimizeBurstCoordinatorConfiguration() {
    if (!burst_coordinator_) return;
    
    auto config = burst_coordinator_->getConfig();
    
    // Optimize collection parameters using actual BurstConfig fields
    config.target_burst_size = settings_.batch_size;
    config.min_burst_size = std::max(4, settings_.batch_size / 8);
    
    // Optimize timing for better coordination
    config.collection_timeout = std::chrono::milliseconds(std::max(2, static_cast<int>(settings_.batch_timeout.count()) / 8));
    config.evaluation_timeout = std::chrono::milliseconds(static_cast<int>(settings_.batch_timeout.count()));
    
    // Set max parallel threads
    config.max_parallel_threads = std::max(1, settings_.num_threads / 2);
    
    burst_coordinator_->updateConfig(config);
}

void MCTSEngine::enableTranspositionTableOptimizations() {
    if (!transposition_table_) return;
    
    // TranspositionTable optimizations are automatically enabled
    // The current implementation uses parallel_flat_hash_map with built-in concurrency
    // and automatic replacement policies, so no manual optimization is needed
    transposition_table_->resetStats(); // Reset statistics for fresh performance tracking
}

void MCTSEngine::configureForMaximumAccuracy() {
    // Configure all components for maximum accuracy
    settings_.num_simulations = std::max(settings_.num_simulations, 1600);
    settings_.exploration_constant = 1.41f; // Classic UCB constant
    settings_.virtual_loss = 0.5f; // Lower virtual loss for better accuracy
    
    // Configure progressive widening for maximum accuracy
    settings_.progressive_widening_c = 1.5f; // More conservative expansion
    settings_.progressive_widening_k = 8.0f; // Higher threshold for expansion
}

void MCTSEngine::configureForBalancedPerformance() {
    // Optimal balance between speed and accuracy
    settings_.exploration_constant = 1.4f;
    settings_.virtual_loss = 1.0f; // Optimized virtual loss
    
    // Configure progressive widening for balanced performance
    settings_.progressive_widening_c = 2.0f; // Our optimized values
    settings_.progressive_widening_k = 6.0f;
}

void MCTSEngine::configureForMaximumSpeed() {
    // Configure for maximum throughput
    settings_.virtual_loss = 2.0f; // Higher virtual loss for speed
    settings_.exploration_constant = 1.2f; // Lower exploration for faster convergence
    
    // Configure progressive widening for maximum speed
    settings_.progressive_widening_c = 3.0f; // Aggressive expansion
    settings_.progressive_widening_k = 4.0f; // Lower threshold
}

void MCTSEngine::configureForEnergyEfficiency() {
    // Configure for energy-efficient operation
    settings_.num_simulations = std::min(settings_.num_simulations, 400);
    settings_.virtual_loss = 1.5f; // Moderate virtual loss
    
    // Reduce batch sizes for energy efficiency
    if (unified_inference_server_) {
        auto config = unified_inference_server_->getConfig();
        config.target_batch_size = std::max(static_cast<size_t>(16), config.target_batch_size / 2);
        unified_inference_server_->updateConfig(config);
    }
}

void MCTSEngine::disableAdvancedOptimizations() {
    // Reset all optimization flags
    use_advanced_optimizations_.store(false, std::memory_order_release);
    adaptive_batching_enabled_.store(false, std::memory_order_release);
    burst_pipelining_enabled_.store(false, std::memory_order_release);
    memory_optimizations_enabled_.store(false, std::memory_order_release);
    real_time_optimization_enabled_.store(false, std::memory_order_release);
    
    // Stop optimization thread if running
    if (optimization_thread_.joinable()) {
        optimization_thread_.join();
    }
}

void MCTSEngine::realTimeOptimizationWorker() {
    while (real_time_optimization_enabled_.load(std::memory_order_acquire)) {
        // Perform real-time optimization every 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Get current performance metrics
        auto metrics = getOptimizationMetrics();
        
        // Adaptive optimization based on current performance
        if (metrics.batch_utilization < 0.6f && unified_inference_server_) {
            // Low batch utilization - adjust collection timeout
            auto config = burst_coordinator_->getConfig();
            config.collection_timeout = std::chrono::milliseconds(std::max(1, static_cast<int>(config.collection_timeout.count()) - 1));
            burst_coordinator_->updateConfig(config);
        }
        
        if (metrics.burst_efficiency < 0.7f && burst_coordinator_) {
            // Low coordination efficiency - adjust burst strategy
            auto config = burst_coordinator_->getConfig();
            config.target_burst_size = std::max(8, static_cast<int>(config.target_burst_size * 1.1));
            burst_coordinator_->updateConfig(config);
        }
        
        if (metrics.memory_efficiency < 0.8f) {
            // Low memory efficiency - the singleton object pool manager handles maintenance automatically
            std::cout << "Memory efficiency low (" << metrics.memory_efficiency << "), object pool manager handles maintenance automatically" << std::endl;
        }
    }
}

} // namespace mcts
} // namespace alphazero