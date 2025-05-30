#include "mcts/shared_eval_wrapper.h"
#include "utils/logger.h"

namespace alphazero {
namespace mcts {

std::atomic<bool> SharedEvalWrapper::enabled_{false};
std::mutex SharedEvalWrapper::mutex_;

void SharedEvalWrapper::initialize(
    std::shared_ptr<nn::NeuralNetwork> network,
    size_t batch_size,
    size_t min_batch_size,
    float timeout_ms,
    size_t num_threads) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (enabled_) {
        LOG_MCTS_WARN("SharedEvalWrapper already initialized");
        return;
    }
    
    // Configure and start the shared evaluation server
    SharedEvaluationServer::Config config;
    config.max_batch_size = batch_size;
    config.min_batch_size = min_batch_size;
    config.batch_timeout_ms = timeout_ms;
    config.num_worker_threads = num_threads;
    
    GlobalEvaluationServer::initialize(network, config);
    enabled_ = true;
    
    LOG_MCTS_INFO("SharedEvalWrapper initialized with batch_size={}, threads={}", 
                  batch_size, num_threads);
}

void SharedEvalWrapper::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!enabled_) {
        return;
    }
    
    GlobalEvaluationServer::shutdown();
    enabled_ = false;
    
    LOG_MCTS_INFO("SharedEvalWrapper shutdown");
}

bool SharedEvalWrapper::isEnabled() {
    return enabled_.load();
}

SharedEvaluationServer::Stats SharedEvalWrapper::getStats() {
    auto* server = GlobalEvaluationServer::get();
    if (server) {
        return server->getStats();
    }
    return SharedEvaluationServer::Stats{};
}

MCTSEngine::InferenceFunction SharedEvalWrapper::createSharedInferenceFunction() {
    return [](const std::vector<std::unique_ptr<core::IGameState>>& states) 
        -> std::vector<NetworkOutput> {
        
        auto* server = GlobalEvaluationServer::get();
        if (!server) {
            throw std::runtime_error("Shared evaluation server not initialized");
        }
        
        std::vector<NetworkOutput> results;
        results.reserve(states.size());
        
        // Submit all states and collect futures
        std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
        futures.reserve(states.size());
        
        for (const auto& state : states) {
            futures.push_back(server->evaluate(state->clone()));
        }
        
        // Wait for results
        for (auto& future : futures) {
            auto [policy, value] = future.get();
            NetworkOutput output;
            output.policy = std::move(policy);
            output.value = value;
            results.push_back(std::move(output));
        }
        
        return results;
    };
}

}  // namespace mcts
}  // namespace alphazero