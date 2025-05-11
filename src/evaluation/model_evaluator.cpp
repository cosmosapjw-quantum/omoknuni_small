// src/evaluation/model_evaluator.cpp
#include "evaluation/model_evaluator.h"
#include <chrono>
#include <thread>
#include <sstream>
#include <future>
#include <cmath>
#include <iostream>
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"

namespace alphazero {
namespace evaluation {

ModelEvaluator::ModelEvaluator(std::shared_ptr<nn::NeuralNetwork> model_first,
                             std::shared_ptr<nn::NeuralNetwork> model_second,
                             const EvaluationSettings& settings)
    : model_first_(model_first),
      model_second_(model_second),
      settings_(settings) {
    
    // Initialize random number generator
    if (settings_.random_seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(settings_.random_seed));
    }
    
    // Create MCTS engines for each worker thread
    engines_first_.reserve(settings_.num_parallel_games);
    engines_second_.reserve(settings_.num_parallel_games);
    
    for (int i = 0; i < settings_.num_parallel_games; ++i) {
        engines_first_.emplace_back(std::make_unique<mcts::MCTSEngine>(
            model_first_, settings_.mcts_settings_first));
        engines_second_.emplace_back(std::make_unique<mcts::MCTSEngine>(
            model_second_, settings_.mcts_settings_second));
    }
}

ModelEvaluator::ModelEvaluator(ModelEvaluator&& other) noexcept
    : model_first_(std::move(other.model_first_)),
      model_second_(std::move(other.model_second_)),
      settings_(std::move(other.settings_)),
      engines_first_(std::move(other.engines_first_)),
      engines_second_(std::move(other.engines_second_)),
      rng_(std::move(other.rng_)) {}

ModelEvaluator& ModelEvaluator::operator=(ModelEvaluator&& other) noexcept {
    if (this != &other) {
        model_first_ = std::move(other.model_first_);
        model_second_ = std::move(other.model_second_);
        settings_ = std::move(other.settings_);
        engines_first_ = std::move(other.engines_first_);
        engines_second_ = std::move(other.engines_second_);
        rng_ = std::move(other.rng_);
    }
    return *this;
}

TournamentResult ModelEvaluator::runTournament(core::GameType game_type, int board_size) {
    TournamentResult result;
    result.total_games = settings_.num_games;
    
    // Create list of matches to play
    std::vector<std::tuple<int, bool>> matches;  // (position_id, first_model_as_player1)
    
    for (int i = 0; i < settings_.num_games; ++i) {
        // Alternate which model plays as Player 1
        bool first_model_as_player1 = (i % 2 == 0);
        
        // Select a random starting position
        std::uniform_int_distribution<int> pos_dist(0, settings_.num_start_positions - 1);
        int position_id = pos_dist(rng_);
        
        matches.emplace_back(position_id, first_model_as_player1);
    }
    
    // Play matches in parallel batches
    result.matches.reserve(settings_.num_games);
    
    for (size_t i = 0; i < matches.size(); i += settings_.num_parallel_games) {
        size_t batch_size = std::min(settings_.num_parallel_games, 
                                    static_cast<int>(matches.size() - i));
        
        std::vector<std::future<MatchResult>> futures;
        futures.reserve(batch_size);
        
        // Launch worker threads
        for (size_t j = 0; j < batch_size; ++j) {
            auto [position_id, first_model_as_player1] = matches[i + j];
            
            futures.push_back(std::async(std::launch::async, 
                                        &ModelEvaluator::playMatch, this, 
                                        game_type, board_size, 
                                        position_id, first_model_as_player1));
        }
        
        // Collect results
        for (auto& future : futures) {
            MatchResult match_result = future.get();
            
            // Update statistics
            if (match_result.result == 0) {
                result.draws++;
            } else if ((match_result.result == 1 && match_result.first_model_as_player1) ||
                       (match_result.result == 2 && !match_result.first_model_as_player1)) {
                result.wins_first++;
            } else {
                result.wins_second++;
            }
            
            result.matches.push_back(std::move(match_result));
        }
    }
    
    // Calculate ELO difference
    result.elo_diff = calculateEloDiff(result.wins_first, result.wins_second, result.draws);
    
    return result;
}

MatchResult ModelEvaluator::playMatch(core::GameType game_type, int board_size, 
                                     int position_id, bool first_model_as_player1) {
    // Select a thread-specific MCTS engine
    int thread_id = 0;
    {
        std::hash<std::thread::id> hasher;
        thread_id = hasher(std::this_thread::get_id()) % engines_first_.size();
    }
    
    mcts::MCTSEngine& engine_first = *engines_first_[thread_id];
    mcts::MCTSEngine& engine_second = *engines_second_[thread_id];
    
    // Create game
    auto game = createGame(game_type, board_size, position_id);
    
    // Prepare match result
    MatchResult result;
    result.match_id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                     "_" + std::to_string(thread_id);
    result.game_type = game_type;
    result.board_size = board_size;
    result.first_model_as_player1 = first_model_as_player1;
    result.result = 0;  // Default to draw
    
    // Calculate max moves if not specified
    int max_moves = settings_.max_moves;
    if (max_moves <= 0) {
        // Default to 5 times the action space size or 1000, whichever is smaller
        max_moves = std::min(5 * game->getActionSpaceSize(), 1000);
    }
    
    // Start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Play until terminal state or max moves
    while (!game->isTerminal() && static_cast<int>(result.moves.size()) < max_moves) {
        // Determine which model to use
        bool use_first_model = (game->getCurrentPlayer() == 1) ? first_model_as_player1 : 
                                                              !first_model_as_player1;
        
        mcts::MCTSEngine& engine = use_first_model ? engine_first : engine_second;
        
        // Run search with low temperature
        auto current_settings = engine.getSettings();
        current_settings.temperature = 0.1f;  // Low temperature for deterministic play
        current_settings.add_dirichlet_noise = false;  // No noise for evaluation
        engine.updateSettings(current_settings);
        
        // Run search
        auto search_result = engine.search(*game);
        
        // Make move
        game->makeMove(search_result.action);
        result.moves.push_back(search_result.action);
        
        // Check if terminal
        if (game->isTerminal()) {
            auto game_result = game->getGameResult();
            if (game_result == core::GameResult::WIN_PLAYER1) {
                result.result = 1;
            } else if (game_result == core::GameResult::WIN_PLAYER2) {
                result.result = 2;
            }
            break;
        }
    }
    
    // End time
    auto end_time = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    return result;
}

float ModelEvaluator::calculateEloDiff(int wins, int losses, int draws) {
    if (wins + losses + draws == 0) {
        return 0.0f;
    }
    
    float score = (wins + 0.5f * draws) / (wins + losses + draws);
    
    // Prevent division by zero or log of zero
    if (score == 0.0f) score = 0.001f;
    if (score == 1.0f) score = 0.999f;
    
    // ELO formula
    return -400.0f * std::log10((1.0f / score) - 1.0f);
}

const EvaluationSettings& ModelEvaluator::getSettings() const {
    return settings_;
}

void ModelEvaluator::updateSettings(const EvaluationSettings& settings) {
    settings_ = settings;
    
    // Update MCTS settings in engines
    for (auto& engine : engines_first_) {
        engine->updateSettings(settings_.mcts_settings_first);
    }
    
    for (auto& engine : engines_second_) {
        engine->updateSettings(settings_.mcts_settings_second);
    }
}

std::unique_ptr<core::IGameState> ModelEvaluator::createGame(core::GameType game_type, 
                                                           int board_size, 
                                                           int position_id) {
    // Create game based on type
    std::unique_ptr<core::IGameState> game;
    
    switch (game_type) {
        case core::GameType::CHESS: {
            game = std::make_unique<::alphazero::chess::ChessState>();
            // TODO: Implement Fischer Random starting positions if needed
            break;
        }
        case core::GameType::GO: {
            game = std::make_unique<::alphazero::go::GoState>(board_size > 0 ? board_size : 19);
            break;
        }
        case core::GameType::GOMOKU: {
            game = std::make_unique<::alphazero::games::gomoku::GomokuState>(board_size > 0 ? board_size : 15);
            break;
        }
        default:
            throw std::runtime_error("Unsupported game type");
    }
    
    return game;
}

} // namespace evaluation
} // namespace alphazero