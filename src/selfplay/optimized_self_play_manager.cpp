#include "selfplay/optimized_self_play_manager.h"
#include "core/game_export.h"
#include "utils/logger.h"
#include <algorithm>
#include <random>
#ifdef __linux__
#include <pthread.h>
#endif

namespace alphazero {
namespace selfplay {

OptimizedSelfPlayManager::OptimizedSelfPlayManager(
    std::shared_ptr<mcts::MultiInstanceNNManager> nn_manager,
    const SelfPlaySettings& settings,
    const std::string& game_type)
    : nn_manager_(nn_manager)
    , settings_(settings)
    , game_type_(game_type)
    , start_time_(std::chrono::steady_clock::now()) {
    
    int num_workers = settings.num_parallel_games;
    LOG_GAME_INFO("Initializing OptimizedSelfPlayManager with {} workers", num_workers);
    
    // Create worker contexts
    workers_.reserve(num_workers);
    
    for (int i = 0; i < num_workers; ++i) {
        auto context = std::make_unique<WorkerContext>();
        context->worker_id = i;
        
        // Get independent neural network instance
        context->neural_net = nn_manager_->getInstance(i);
        if (!context->neural_net) {
            LOG_GAME_ERROR("Failed to get neural network for worker {}", i);
            throw std::runtime_error("Failed to initialize worker");
        }
        
        // Create MCTS engine with independent neural network
        context->engine = std::make_unique<mcts::MCTSEngine>(
            context->neural_net, 
            settings_.mcts_settings
        );
        
        LOG_GAME_INFO("Worker {} initialized", i);
        
        // Start worker thread
        context->thread = std::thread(&OptimizedSelfPlayManager::workerLoop, this, context.get());
        
        // Set thread affinity to distribute across CPU cores (Linux-specific)
        #ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
        pthread_setaffinity_np(context->thread.native_handle(), sizeof(cpuset), &cpuset);
        #endif
        
        workers_.push_back(std::move(context));
    }
    
    LOG_GAME_INFO("All workers started successfully");
}

OptimizedSelfPlayManager::~OptimizedSelfPlayManager() {
    // Signal workers to stop
    should_stop_ = true;
    queue_cv_.notify_all();
    
    // Wait for all workers to finish
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
    }
}

void OptimizedSelfPlayManager::workerLoop(WorkerContext* context) {
    LOG_GAME_INFO("Worker {} started", context->worker_id);
    
    while (!should_stop_) {
        int game_id = -1;
        
        // Get work from queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !work_queue_.empty() || should_stop_; 
            });
            
            if (should_stop_) break;
            
            if (!work_queue_.empty()) {
                game_id = work_queue_.front();
                work_queue_.pop();
            }
        }
        
        if (game_id >= 0) {
            // Play one game
            auto start = std::chrono::high_resolution_clock::now();
            
            GameRecord record = playOneGame(context);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end - start).count();
            
            // Update metrics
            context->games_completed++;
            context->total_game_time.store(context->total_game_time.load() + duration);
            context->total_moves += record.states.size();
            
            // Store completed game
            {
                std::lock_guard<std::mutex> lock(completed_mutex_);
                completed_games_.push_back(std::move(record));
                total_games_completed_++;
            }
            
            LOG_GAME_DEBUG("Worker {} completed game {} in {}s ({} moves)", 
                          context->worker_id, game_id, duration, record.states.size());
        }
    }
    
    LOG_GAME_INFO("Worker {} stopped", context->worker_id);
}

GameRecord OptimizedSelfPlayManager::playOneGame(WorkerContext* context) {
    GameRecord record;
    record.game_id = std::to_string(context->worker_id) + "_" + 
                    std::to_string(context->games_completed.load());
    
    // Create initial game state
    core::GameType game_type_enum;
    if (game_type_ == "gomoku") {
        game_type_enum = core::GameType::GOMOKU;
    } else if (game_type_ == "chess") {
        game_type_enum = core::GameType::CHESS;
    } else if (game_type_ == "go") {
        game_type_enum = core::GameType::GO;
    } else {
        LOG_GAME_ERROR("Unknown game type: {}", game_type_);
        return record;
    }
    
    auto state = core::GameRegistry::instance().createGame(game_type_enum);
    
    // Temperature schedule
    auto getTemperature = [this](int move_number) -> float {
        if (move_number < settings_.temperature_threshold) {
            return settings_.high_temperature;
        }
        return settings_.low_temperature;  // Lower temperature after threshold
    };
    
    std::mt19937 rng(std::random_device{}());
    int move_number = 0;
    
    while (!state->isTerminal()) {
        // Run MCTS search
        auto search_result = context->engine->search(*state);
        int best_move = search_result.action;
        
        // Get move probabilities from search result
        std::vector<float> policy = search_result.probabilities;
        
        // Store training data
        record.states.push_back(state->clone());
        record.policies.push_back(policy);
        
        // Select move based on temperature
        float temperature = getTemperature(move_number);
        int selected_move = best_move;
        
        if (temperature > 0.0f && !policy.empty()) {
            // Sample from policy distribution
            std::vector<float> probs = policy;
            
            // Apply temperature
            float sum = 0.0f;
            for (auto& p : probs) {
                p = std::pow(p, 1.0f / temperature);
                sum += p;
            }
            
            // Normalize
            if (sum > 0) {
                for (auto& p : probs) {
                    p /= sum;
                }
                
                // Sample
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int sampled_index = dist(rng);
                
                // Convert index back to move
                auto legal_moves = state->getLegalMoves();
                if (sampled_index < static_cast<int>(legal_moves.size())) {
                    selected_move = legal_moves[sampled_index];
                }
            }
        }
        
        // Make the move
        state->makeMove(selected_move);
        move_number++;
        
        // Clear tree for next search (except root reuse if enabled)
        // Clear tree for next search
        // Note: clearTree functionality might need to be implemented
    }
    
    // Get final outcome
    // Get the winner (0 = draw, 1 = player 1, 2 = player 2)
    float final_value = 0.0f;
    if (state->isTerminal()) {
        // For two-player games, we need to determine who won
        // This is game-specific, but generally:
        // If the game ended and it's player 2's turn, player 1 won
        // We'll use a simple heuristic based on move count
        if (move_number % 2 == 1) {
            // Odd number of moves = player 1 won
            final_value = 1.0f;
        } else {
            // Even number of moves = player 2 won
            final_value = -1.0f;
        }
        
        // Check for draw (if the state supports it)
        // For now, we'll assume no draws in gomoku
    }
    
    // Propagate values (alternating perspective)
    float current_value = final_value;
    for (int i = static_cast<int>(record.states.size()) - 1; i >= 0; --i) {
        record.values.push_back(current_value);
        current_value = -current_value;  // Flip perspective
    }
    std::reverse(record.values.begin(), record.values.end());
    
    return record;
}

std::vector<GameRecord> OptimizedSelfPlayManager::generateGames(int num_games) {
    startAsyncGeneration(num_games);
    waitForCompletion();
    return collectCompletedGames();
}

void OptimizedSelfPlayManager::startAsyncGeneration(int num_games) {
    LOG_GAME_INFO("Starting generation of {} games", num_games);
    
    total_games_requested_ += num_games;
    
    // Add games to work queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (int i = 0; i < num_games; ++i) {
            work_queue_.push(i);
        }
    }
    
    // Notify all workers
    queue_cv_.notify_all();
}

std::vector<GameRecord> OptimizedSelfPlayManager::collectCompletedGames() {
    std::vector<GameRecord> games;
    
    {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        games = std::move(completed_games_);
        completed_games_.clear();
    }
    
    return games;
}

void OptimizedSelfPlayManager::waitForCompletion() {
    // Wait until all requested games are completed
    while (total_games_completed_ < total_games_requested_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Print progress
        int completed = total_games_completed_.load();
        int requested = total_games_requested_.load();
        if (completed % 10 == 0 && completed > 0) {
            LOG_GAME_INFO("Progress: {}/{} games completed", completed, requested);
        }
    }
}

void OptimizedSelfPlayManager::printStatistics() const {
    auto now = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(now - start_time_).count();
    
    LOG_GAME_INFO("=== Performance Statistics ===");
    LOG_GAME_INFO("Total games: {}", total_games_completed_.load());
    LOG_GAME_INFO("Total time: {} seconds", total_time);
    LOG_GAME_INFO("Games/sec: {}", total_games_completed_.load() / total_time);
    
    int total_moves = 0;
    double total_worker_time = 0.0;
    
    for (const auto& worker : workers_) {
        int games = worker->games_completed.load();
        double time = worker->total_game_time.load();
        int moves = worker->total_moves.load();
        
        total_moves += moves;
        total_worker_time += time;
        
        double avg_time = (games > 0) ? time / games : 0.0;
        double avg_moves = (games > 0) ? static_cast<double>(moves) / games : 0.0;
        
        LOG_GAME_INFO("Worker {}: Games={} AvgTime={}s AvgMoves={}",
                     worker->worker_id, games, avg_time, avg_moves);
    }
    
    LOG_GAME_INFO("Total moves: {} Moves/sec: {}", total_moves, total_moves / total_time);
    
    // Print neural network statistics
    nn_manager_->printStatistics();
}

} // namespace selfplay
} // namespace alphazero