// src/mcts/enhanced_mcts_engine.cpp
#include "mcts/enhanced_mcts_engine.h"
#include "utils/logger.h"
#include <torch/cuda.h>
#include <cuda_runtime.h>

namespace alphazero {
namespace mcts {

EnhancedMCTSEngine::EnhancedMCTSEngine(
    const EnhancedSettings& settings,
    std::shared_ptr<nn::NeuralNetwork> network,
    core::GameType game_type
) : MCTSEngine(network, static_cast<const MCTSSettings&>(settings)), 
    enhanced_settings_(settings) {
    
    // Initialize GPU memory pool
    if (settings.use_gpu_memory_pool) {
        gpu_pool_ = std::make_shared<GPUMemoryPool>(settings.gpu_pool_config);
        LOG_MCTS_INFO("GPU memory pool initialized with {} MB", 
                      settings.gpu_pool_config.initial_pool_size_mb);
    }
    
    // Initialize dynamic batch manager
    if (settings.use_dynamic_batching) {
        batch_manager_ = std::make_shared<DynamicBatchManager>(settings.batch_manager_config);
        LOG_MCTS_INFO("Dynamic batch manager initialized (mode: {})", 
                      static_cast<int>(settings.batch_manager_config.optimization_mode));
    }
    
    // Initialize advanced transposition table
    if (settings.use_advanced_tt) {
        advanced_tt_ = std::make_shared<AdvancedTranspositionTable>(settings.tt_config);
        setUseTranspositionTable(true);
        LOG_MCTS_INFO("Advanced transposition table initialized with {} MB", 
                      settings.tt_config.size_mb);
    }
    
    // Note: Multi-instance NN manager would need a different constructor
    // since it expects (model_path, num_instances, nn_config)
    // For now, we'll use the single network instance from base class
    
    last_stats_update_ = std::chrono::steady_clock::now();
}

EnhancedMCTSEngine::~EnhancedMCTSEngine() {
    // Cleanup is handled by smart pointers
}

SearchResult EnhancedMCTSEngine::search(const core::IGameState& root_state) {
    // Prefetch transposition table entries if available
    if (advanced_tt_ && enhanced_settings_.use_advanced_tt) {
        std::vector<uint64_t> prefetch_hashes;
        prefetch_hashes.reserve(8);
        
        // Get likely positions to explore
        auto legal_moves = root_state.getLegalMoves();
        for (size_t i = 0; i < std::min(size_t(8), legal_moves.size()); ++i) {
            auto state_copy = root_state.clone();
            state_copy->makeMove(legal_moves[i]);
            prefetch_hashes.push_back(state_copy->getHash());
        }
        
        advanced_tt_->prefetch(prefetch_hashes.data(), prefetch_hashes.size());
    }
    
    // Use base search with enhanced components
    auto result = MCTSEngine::search(root_state);
    
    // Update performance metrics
    updatePerformanceMetrics();
    
    return result;
}

void EnhancedMCTSEngine::evaluateBatchWithPool(PooledBatch& batch) {
    if (batch.nodes.empty()) return;
    
    // Prepare for GPU inference with memory pool
    int batch_size = batch.nodes.size();
    int device_id = 0; // Default device
    
    if (torch::cuda::is_available()) {
        cudaSetDevice(device_id);
    }
    
    // Get input tensor dimensions from first state
    auto first_state = batch.nodes[0]->getState().clone();
    auto tensor_repr = first_state->getTensorRepresentation();
    
    int channels = tensor_repr.size();
    int height = tensor_repr[0].size();
    int width = tensor_repr[0][0].size();
    
    // Allocate GPU memory from pool if available
    torch::Tensor batch_input;
    std::shared_ptr<GPUMemoryPool::MemoryBlock> memory_block;
    
    if (gpu_pool_) {
        size_t tensor_size = batch_size * channels * height * width * sizeof(float);
        memory_block = gpu_pool_->allocateBlock(tensor_size, device_id);
        
        // Create tensor from pooled memory
        batch_input = torch::from_blob(
            memory_block->device_ptr,
            {batch_size, channels, height, width},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id)
        );
    } else {
        // Fallback to regular allocation
        batch_input = torch::zeros({batch_size, channels, height, width}, 
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id));
    }
    
    // Copy state data to batch tensor
    for (int i = 0; i < batch_size; ++i) {
        auto state = batch.nodes[i]->getState().clone();
        auto state_tensor = state->getTensorRepresentation();
        
        // Copy to batch tensor
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    batch_input[i][c][h][w] = state_tensor[c][h][w];
                }
            }
        }
    }
    
    // Perform inference
    torch::Tensor policy_output, value_output;
    
    // Use the neural network from base class
    if (neural_network_) {
        // Convert batch tensor to vector of states for inference
        std::vector<std::unique_ptr<core::IGameState>> batch_states;
        batch_states.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batch_states.push_back(batch.nodes[i]->getState().clone());
        }
        
        auto outputs = neural_network_->inference(batch_states);
        
        // Extract policy and value tensors
        if (!outputs.empty()) {
            // Create tensors from outputs
            policy_output = torch::zeros({batch_size, static_cast<long>(outputs[0].policy.size())});
            value_output = torch::zeros({batch_size, 1});
            
            for (int i = 0; i < batch_size && i < static_cast<int>(outputs.size()); ++i) {
                for (size_t j = 0; j < outputs[i].policy.size(); ++j) {
                    policy_output[i][j] = outputs[i].policy[j];
                }
                value_output[i][0] = outputs[i].value;
            }
        }
    }
    
    // Process outputs and update nodes
    for (int i = 0; i < batch_size; ++i) {
        // Extract policy for this node
        std::vector<float> policy(policy_output[i].data_ptr<float>(), 
                                policy_output[i].data_ptr<float>() + policy_output[i].numel());
        
        // Extract value
        float value = value_output[i].item<float>();
        
        // Update node (this would need to be added to MCTSNode interface)
        // For now, we'll just log
        LOG_MCTS_DEBUG("Node {} evaluated with value: {}", i, value);
        
        // Note: In a complete implementation, we would update the node here:
        // batch.nodes[i]->setEvaluationResult(policy, value);
    }
    
    // Store memory block for later cleanup
    if (memory_block) {
        batch.memory_blocks.push_back(memory_block);
    }
    
    // Update stats
    if (batch_manager_) {
        batch_manager_->updateMetrics(batch_size, 
                                  std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - last_stats_update_).count(),
                                  batch_size); // queue_depth_at_start
    }
}

void EnhancedMCTSEngine::collectDynamicBatch(std::vector<MCTSNode*>& batch, int optimal_size) {
    batch.clear();
    batch.reserve(optimal_size);
    
    // This would need access to the pending evaluations queue from base class
    // For now, just a placeholder implementation
    LOG_MCTS_DEBUG("Collecting dynamic batch with optimal size: {}", optimal_size);
}

void EnhancedMCTSEngine::updatePerformanceMetrics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_update_);
    
    if (elapsed.count() >= 1) {
        // Update GPU utilization metrics
        if (gpu_pool_) {
            auto pool_stats = gpu_pool_->getStats();
            float hit_rate = pool_stats.num_reuses > 0 ? 
                           static_cast<float>(pool_stats.num_reuses) / 
                           static_cast<float>(pool_stats.num_allocations + pool_stats.num_reuses) : 0.0f;
            LOG_MCTS_DEBUG("GPU pool hit rate: {:.2f}%, allocated: {} bytes", 
                         hit_rate * 100.0f,
                         pool_stats.total_allocated);
        }
        
        // Update batch manager metrics
        if (batch_manager_) {
            auto batch_stats = batch_manager_->getStats();
            LOG_MCTS_DEBUG("Dynamic batching - avg size: {:.1f}, avg time: {:.2f} ms",
                         batch_stats.avg_batch_size,
                         batch_stats.avg_inference_time_ms);
        }
        
        // Update transposition table metrics
        if (advanced_tt_) {
            auto tt_stats = advanced_tt_->getStats();
            LOG_MCTS_DEBUG("Advanced TT - hit rate: {:.2f}%, entries: {}k",
                         tt_stats.hit_rate * 100.0f,
                         tt_stats.num_entries / 1000);
        }
        
        last_stats_update_ = now;
    }
}

EnhancedMCTSEngine::EnhancedStats EnhancedMCTSEngine::getEnhancedStats() const {
    EnhancedStats stats;
    
    // Get base stats
    stats.base_stats = getLastStats();
    
    // Get GPU pool stats
    if (gpu_pool_) {
        stats.gpu_pool_stats = gpu_pool_->getStats();
    }
    
    // Get batch manager stats
    if (batch_manager_) {
        stats.batch_stats = batch_manager_->getStats();
    }
    
    // Get transposition table stats
    if (advanced_tt_) {
        stats.tt_stats = advanced_tt_->getStats();
    }
    
    // Calculate GPU utilization (simplified)
    stats.gpu_utilization = 0.0f;
    if (torch::cuda::is_available()) {
        // This would need proper CUDA metrics
        stats.gpu_utilization = 0.85f; // Placeholder
    }
    
    // Calculate memory efficiency
    stats.memory_efficiency = 0.0f;
    if (gpu_pool_) {
        auto pool_stats = gpu_pool_->getStats();
        if (pool_stats.total_allocated > 0) {
            // Calculate efficiency as reuse ratio
            stats.memory_efficiency = static_cast<float>(pool_stats.num_reuses) / 
                                    static_cast<float>(pool_stats.num_allocations + pool_stats.num_reuses);
        }
    }
    
    return stats;
}

// Factory implementation
std::unique_ptr<EnhancedMCTSEngine> EnhancedMCTSEngineFactory::create(
    const YAML::Node& config,
    std::shared_ptr<nn::NeuralNetwork> network,
    core::GameType game_type
) {
    auto settings = parseConfig(config);
    return std::make_unique<EnhancedMCTSEngine>(settings, network, game_type);
}

EnhancedMCTSEngine::EnhancedSettings EnhancedMCTSEngineFactory::parseConfig(const YAML::Node& config) {
    EnhancedMCTSEngine::EnhancedSettings settings;
    
    // Parse base MCTS settings
    if (config["mcts"]) {
        auto mcts_config = config["mcts"];
        settings.num_simulations = mcts_config["num_simulations"].as<int>(800);
        settings.num_threads = mcts_config["num_threads"].as<int>(12);
        settings.batch_size = mcts_config["batch_size"].as<int>(256);
        settings.exploration_constant = mcts_config["exploration_constant"].as<float>(1.4f);
        settings.virtual_loss = mcts_config["virtual_loss"].as<int>(3);
        settings.use_transposition_table = mcts_config["use_transposition_table"].as<bool>(true);
        settings.batch_timeout = std::chrono::milliseconds(
            mcts_config["batch_timeout_ms"].as<int>(5));
    }
    
    // Parse enhanced settings
    if (config["enhanced"]) {
        auto enhanced = config["enhanced"];
        
        // GPU pool config
        if (enhanced["gpu_pool"]) {
            auto pool = enhanced["gpu_pool"];
            settings.use_gpu_memory_pool = pool["enabled"].as<bool>(true);
            settings.gpu_pool_config.initial_pool_size_mb = pool["initial_size_mb"].as<size_t>(1024);
            settings.gpu_pool_config.max_pool_size_mb = pool["max_size_mb"].as<size_t>(4096);
            settings.gpu_pool_config.pinned_memory_size_mb = pool["pinned_memory_size_mb"].as<size_t>(128);
            settings.gpu_pool_config.enable_peer_access = pool["enable_peer_access"].as<bool>(true);
            settings.gpu_pool_config.use_unified_memory = pool["use_unified_memory"].as<bool>(false);
        }
        
        // Dynamic batching config
        if (enhanced["dynamic_batching"]) {
            auto batch = enhanced["dynamic_batching"];
            settings.use_dynamic_batching = batch["enabled"].as<bool>(true);
            settings.batch_manager_config.min_batch_size = batch["min_size"].as<int>(64);
            settings.batch_manager_config.max_batch_size = batch["max_size"].as<int>(512);
            settings.batch_manager_config.preferred_batch_size = batch["preferred_size"].as<int>(256);
            settings.batch_manager_config.max_wait_time_ms = batch["max_wait_time_ms"].as<int>(5);
            settings.batch_manager_config.target_gpu_utilization_percent = batch["target_gpu_utilization"].as<int>(85);
        }
        
        // Advanced TT config
        if (enhanced["transposition_table"]) {
            auto tt = enhanced["transposition_table"];
            settings.use_advanced_tt = tt["enabled"].as<bool>(true);
            settings.tt_config.size_mb = tt["size_mb"].as<size_t>(256);
            settings.tt_config.num_buckets = tt["num_buckets"].as<size_t>(4);
            settings.tt_config.num_hash_functions = tt["num_hash_functions"].as<int>(4);
            settings.tt_config.enable_compression = tt["enable_compression"].as<bool>(true);
            settings.tt_config.enable_prefetch = tt["enable_prefetch"].as<bool>(true);
            settings.tt_config.verify_hash = tt["verify_hash"].as<bool>(true);
            settings.tt_config.load_factor = tt["load_factor"].as<float>(0.75f);
            settings.tt_config.max_age = tt["max_age"].as<int>(10);
        }
        
        // Multi-instance NN
        settings.nn_instances_per_engine = enhanced["nn_instances"].as<int>(1);
    }
    
    return settings;
}

} // namespace mcts
} // namespace alphazero