#include "selfplay/unified_parallel_manager.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace alphazero {
namespace selfplay {

UnifiedParallelManager::UnifiedParallelManager(
    std::shared_ptr<nn::NeuralNetwork> neural_net,
    const SelfPlaySettings& settings)
    : SelfPlayManager(neural_net, settings) {
    
    num_parallel_games_ = settings.num_parallel_games;
    if (num_parallel_games_ <= 0) {
        int cpu_cores = std::thread::hardware_concurrency();
        int threads_per_game = settings.mcts_settings.num_threads;
        num_parallel_games_ = std::max(3, cpu_cores / threads_per_game);
    }
}

UnifiedParallelManager::~UnifiedParallelManager() = default;

std::vector<GameData> UnifiedParallelManager::generateGamesUnified(
    core::GameType game_type,
    size_t num_games,
    int board_size) {
    
    std::cout << "\n🚀 UNIFIED PARALLEL GAME GENERATION" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Total games to generate: " << num_games << std::endl;
    std::cout << "  - Parallel games: " << num_parallel_games_ << std::endl;
    std::cout << "  - MCTS threads per game: " << settings_.mcts_settings.num_threads << std::endl;
    std::cout << "  - Total CPU threads: " << num_parallel_games_ * settings_.mcts_settings.num_threads << std::endl;
    std::cout << "  - Unified batch size: " << settings_.mcts_settings.batch_size << std::endl;
    std::cout << "  - UNIFIED GPU batching across ALL games! 🔥" << std::endl;
    std::cout << std::endl;
    
    // Create shared evaluator
    auto shared_evaluator = std::make_unique<SharedEvaluator>();
    shared_evaluator->neural_net = neural_net_;
    shared_evaluator->batch_size = settings_.mcts_settings.batch_size;
    shared_evaluator->batch_timeout_ms = settings_.mcts_settings.batch_timeout.count();
    
    // Start unified evaluator thread
    shared_evaluator->evaluator_thread = std::thread(
        &UnifiedParallelManager::unifiedEvaluatorThread, this, shared_evaluator.get());
    
    // Results storage
    std::vector<GameData> all_games;
    all_games.reserve(num_games);
    
    // Process games in batches
    size_t games_completed = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (games_completed < num_games) {
        // Determine batch size
        size_t batch_size = std::min(num_parallel_games_, num_games - games_completed);
        
        // Create game workers
        std::vector<std::unique_ptr<GameWorker>> workers;
        for (size_t i = 0; i < batch_size; ++i) {
            auto worker = std::make_unique<GameWorker>();
            worker->worker_id = games_completed + i;
            
            // Create MCTS engine using shared queues
            worker->engine = createSharedQueueEngine(shared_evaluator.get());
            
            // Start worker thread
            worker->thread = std::thread(
                &UnifiedParallelManager::gameWorkerThread, this,
                worker.get(), game_type, board_size, shared_evaluator.get());
            
            workers.push_back(std::move(worker));
        }
        
        // Monitor progress
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            size_t batch_completed = 0;
            for (const auto& worker : workers) {
                if (worker->completed.load()) {
                    batch_completed++;
                }
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            
            if (elapsed_seconds > 0) {
                double games_per_second = (games_completed + batch_completed) / (double)elapsed_seconds;
                size_t queue_size = shared_evaluator->request_queue.size_approx();
                
                std::cout << "\rProgress: " << (games_completed + batch_completed) << "/" << num_games 
                          << " games | Speed: " << std::fixed << std::setprecision(2) 
                          << games_per_second << " games/sec"
                          << " | Eval Queue: " << queue_size
                          << " | Active games: " << (batch_size - batch_completed) << std::flush;
            }
            
            if (batch_completed == batch_size) {
                break;
            }
        }
        
        // Collect results
        for (auto& worker : workers) {
            worker->thread.join();
            all_games.push_back(std::move(worker->game_data));
        }
        
        games_completed += batch_size;
    }
    
    // Shutdown evaluator
    shared_evaluator->shutdown = true;
    shared_evaluator->evaluator_thread.join();
    
    std::cout << "\n✅ Unified parallel generation complete!" << std::endl;
    std::cout << "   Average GPU utilization should be 70%+ throughout! 🎉" << std::endl;
    
    return all_games;
}

void UnifiedParallelManager::unifiedEvaluatorThread(SharedEvaluator* evaluator) {
    std::cout << "🔥 Unified evaluator thread started - serving ALL games from single queue" << std::endl;
    
    std::vector<mcts::EvaluationRequest> batch;
    batch.reserve(evaluator->batch_size);
    std::vector<std::shared_ptr<mcts::MCTSNode>> nodes;
    nodes.reserve(evaluator->batch_size);
    
    auto last_batch_time = std::chrono::steady_clock::now();
    size_t total_evaluations = 0;
    size_t total_batches = 0;
    
    while (!evaluator->shutdown.load()) {
        batch.clear();
        nodes.clear();
        
        // Collect batch with timeout
        auto timeout_point = std::chrono::steady_clock::now() + 
                           std::chrono::milliseconds(evaluator->batch_timeout_ms);
        
        while (batch.size() < evaluator->batch_size && 
               std::chrono::steady_clock::now() < timeout_point) {
            
            mcts::EvaluationRequest req;
            if (evaluator->request_queue.try_dequeue(req)) {
                nodes.push_back(req.node);
                batch.push_back(std::move(req));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        if (batch.empty()) {
            continue;
        }
        
        // Process batch
        auto batch_start = std::chrono::steady_clock::now();
        
        // Prepare states for neural network
        std::vector<std::unique_ptr<core::IGameState>> states;
        states.reserve(batch.size());
        for (auto& req : batch) {
            states.push_back(std::move(req.state));
        }
        
        // Run inference
        auto outputs = evaluator->neural_net->inference(states);
        
        // Send results back through promises
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value(outputs[i]);
        }
        
        auto batch_duration = std::chrono::steady_clock::now() - batch_start;
        auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_duration).count();
        
        total_evaluations += batch.size();
        total_batches++;
        
        // Log every 20 batches
        if (total_batches % 20 == 0) {
            auto total_duration = std::chrono::steady_clock::now() - last_batch_time;
            auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(total_duration).count();
            
            if (total_seconds > 0) {
                double avg_batch_size = total_evaluations / (double)total_batches;
                double throughput = total_evaluations / (double)total_seconds;
                
                std::cout << "\n📊 Unified GPU Stats: Avg batch: " << avg_batch_size 
                          << " | Throughput: " << throughput << " evals/sec"
                          << " | Latest batch: " << batch.size() << " in " << batch_ms << "ms" 
                          << std::endl;
            }
        }
    }
    
    std::cout << "\n🛑 Unified evaluator thread shutdown" << std::endl;
}

void UnifiedParallelManager::gameWorkerThread(
    GameWorker* worker,
    core::GameType game_type,
    int board_size,
    SharedEvaluator* evaluator) {
    
    // Create initial game state
    std::unique_ptr<core::IGameState> game_state;
    switch (game_type) {
        case core::GameType::GOMOKU:
            game_state = std::make_unique<games::gomoku::GomokuState>(board_size);
            break;
        case core::GameType::CHESS:
            game_state = std::make_unique<games::chess::ChessState>();
            break;
        case core::GameType::GO:
            game_state = std::make_unique<games::go::GoState>(board_size);
            break;
        default:
            throw std::runtime_error("Unknown game type");
    }
    
    // Initialize game data
    worker->game_data.game_id = "unified_game_" + std::to_string(worker->worker_id);
    worker->game_data.game_type = game_type;
    worker->game_data.board_size = board_size;
    
    // Play game
    int move_count = 0;
    while (!game_state->isTerminal()) {
        // Run MCTS search
        auto search_result = worker->engine->search(*game_state);
        
        // Select move based on temperature
        int action = search_result.action;
        if (move_count < settings_.temperature_threshold && settings_.high_temperature > 0) {
            // Apply temperature for exploration
            // (simplified - in real implementation would sample from distribution)
        }
        
        // Record move and policy
        worker->game_data.moves.push_back(action);
        worker->game_data.policies.push_back(search_result.probabilities);
        
        // Make move
        game_state->makeMove(action);
        move_count++;
    }
    
    // Record game result
    auto result = game_state->getGameResult();
    switch (result) {
        case core::GameResult::WIN_PLAYER1:
            worker->game_data.winner = 1;
            break;
        case core::GameResult::WIN_PLAYER2:
            worker->game_data.winner = 2;
            break;
        default:
            worker->game_data.winner = 0;
    }
    
    worker->completed = true;
}

std::unique_ptr<mcts::MCTSEngine> UnifiedParallelManager::createSharedQueueEngine(
    SharedEvaluator* evaluator) {
    
    // Create custom inference function that uses shared queues
    auto inference_fn = [evaluator](const std::vector<std::unique_ptr<core::IGameState>>& states) 
        -> std::vector<mcts::NetworkOutput> {
        
        std::vector<mcts::NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        // For each state, create evaluation request with promise
        std::vector<std::future<mcts::NetworkOutput>> futures;
        
        for (const auto& state : states) {
            mcts::EvaluationRequest req;
            req.state = state->clone();
            req.action_space_size = state->getActionSpaceSize();
            
            auto future = req.promise.get_future();
            futures.push_back(std::move(future));
            
            evaluator->request_queue.enqueue(std::move(req));
        }
        
        // Collect results
        for (auto& future : futures) {
            outputs.push_back(future.get());
        }
        
        return outputs;
    };
    
    // Create engine with custom inference function
    return std::make_unique<mcts::MCTSEngine>(inference_fn, settings_.mcts_settings);
}

} // namespace selfplay  
} // namespace alphazero