// src/python/bindings.cpp
// OPTIMIZED: Compatible with new BurstCoordinator + UnifiedInferenceServer architecture
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>

#include "core/igamestate.h"
#include "core/game_export.h"
#include "mcts/mcts_engine.h"

#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"

namespace py = pybind11;

namespace alphazero {
namespace python {

// Convert Python numpy arrays to C++ tensor representation
std::vector<std::vector<std::vector<float>>> numpyToTensor(const py::array_t<float>& array) {
    py::buffer_info info = array.request();
    
    if (info.ndim != 3) {
        throw std::runtime_error("Input array must be 3-dimensional");
    }
    
    size_t channels = info.shape[0];
    size_t height = info.shape[1];
    size_t width = info.shape[2];
    
    std::vector<std::vector<std::vector<float>>> tensor(
        channels, std::vector<std::vector<float>>(
            height, std::vector<float>(width, 0.0f)));
    
    float* data = static_cast<float*>(info.ptr);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                tensor[c][h][w] = data[c * height * width + h * width + w];
            }
        }
    }
    
    return tensor;
}

// Convert C++ tensor representation to Python numpy arrays
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
    if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
        return py::array_t<float>();
    }
    
    size_t channels = tensor.size();
    size_t height = tensor[0].size();
    size_t width = tensor[0][0].size();
    
    py::array_t<float> array({channels, height, width});
    py::buffer_info info = array.request();
    float* data = static_cast<float*>(info.ptr);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                data[c * height * width + h * width + w] = tensor[c][h][w];
            }
        }
    }
    
    return array;
}

// Wrapper for neural network inference
class NeuralNetWrapper {
public:
    using InferenceCallback = std::function<py::tuple(py::array_t<float>)>;
    
    NeuralNetWrapper(InferenceCallback callback) : callback_(callback) {}
    
    std::vector<mcts::NetworkOutput> inference(const std::vector<std::unique_ptr<core::IGameState>>& states) {
        // Prepare batch
        std::vector<py::array_t<float>> batch;
        batch.reserve(states.size());
        
        for (const auto& state : states) {
            auto tensor = state->getEnhancedTensorRepresentation();
            batch.push_back(tensorToNumpy(tensor));
        }
        
        // Combine into a single batch tensor
        py::array_t<float> batch_tensor;
        if (!batch.empty()) {
            py::buffer_info info = batch[0].request();
            size_t channels = info.shape[0];
            size_t height = info.shape[1];
            size_t width = info.shape[2];
            
            batch_tensor = py::array_t<float>({batch.size(), channels, height, width});
            py::buffer_info batch_info = batch_tensor.request();
            float* batch_data = static_cast<float*>(batch_info.ptr);
            
            for (size_t b = 0; b < batch.size(); ++b) {
                py::buffer_info state_info = batch[b].request();
                float* state_data = static_cast<float*>(state_info.ptr);
                
                for (size_t i = 0; i < channels * height * width; ++i) {
                    batch_data[b * channels * height * width + i] = state_data[i];
                }
            }
        } else {
            return {}; // Empty batch
        }
        
        // Call Python inference function
        py::tuple result;
        try {
            result = callback_(batch_tensor);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Neural network inference error: ") + e.what());
        }
        
        // Extract policy and value
        py::array_t<float> policy_batch = result[0].cast<py::array_t<float>>();
        py::array_t<float> value_batch = result[1].cast<py::array_t<float>>();
        
        py::buffer_info policy_info = policy_batch.request();
        py::buffer_info value_info = value_batch.request();
        
        float* policy_data = static_cast<float*>(policy_info.ptr);
        float* value_data = static_cast<float*>(value_info.ptr);
        
        // Convert to C++ format
        std::vector<mcts::NetworkOutput> outputs;
        outputs.reserve(states.size());
        
        for (size_t i = 0; i < states.size(); ++i) {
            mcts::NetworkOutput output;
            
            // Extract policy
            size_t policy_size = policy_info.shape[1];
            output.policy.resize(policy_size);
            for (size_t j = 0; j < policy_size; ++j) {
                output.policy[j] = policy_data[i * policy_size + j];
            }
            
            // Extract value
            output.value = value_data[i];
            
            outputs.push_back(std::move(output));
        }
        
        return outputs;
    }
    
private:
    InferenceCallback callback_;
};

// Self-play manager
class SelfPlayManager {
public:
    SelfPlayManager(const mcts::MCTSSettings& settings, NeuralNetWrapper::InferenceCallback callback)
        : neural_net_(callback),
          mcts_engine_([this](const std::vector<std::unique_ptr<core::IGameState>>& states) {
              return neural_net_.inference(states);
          }, settings) {}
    
    py::tuple generate_game(const std::string& game_type, int max_moves = 1000) {
        // Create game
        auto game = createGame(game_type);
        if (!game) {
            throw std::runtime_error("Invalid game type: " + game_type);
        }
        
        std::vector<int> moves;
        std::vector<std::vector<float>> policies;
        int winner = 0; // 0: draw, 1: player 1, 2: player 2
        
        // Maximum temperature moves
        int temperature_threshold = std::min(30, max_moves / 2);
        
        // Play until terminal state or max moves
        while (!game->isTerminal() && static_cast<int>(moves.size()) < max_moves) {
            // Set temperature
            auto settings = mcts_engine_.getSettings();
            if (static_cast<int>(moves.size()) < temperature_threshold) {
                settings.temperature = 1.0f; // Exploration
            } else {
                settings.temperature = 0.1f; // Exploitation
            }
            mcts_engine_.updateSettings(settings);
            
            // Run search
            auto result = mcts_engine_.search(*game);
            
            // Store policy
            policies.push_back(result.probabilities);
            
            // Make move
            game->makeMove(result.action);
            moves.push_back(result.action);
            
            // Check if terminal
            if (game->isTerminal()) {
                auto game_result = game->getGameResult();
                if (game_result == core::GameResult::WIN_PLAYER1) {
                    winner = 1;
                } else if (game_result == core::GameResult::WIN_PLAYER2) {
                    winner = 2;
                }
                break;
            }
        }
        
        // Convert moves and policies to numpy arrays
        py::array_t<int> moves_array(moves.size());
        py::buffer_info moves_info = moves_array.request();
        int* moves_data = static_cast<int*>(moves_info.ptr);
        
        for (size_t i = 0; i < moves.size(); ++i) {
            moves_data[i] = moves[i];
        }
        
        // Create policies array
        size_t num_policies = policies.size();
        size_t policy_size = policies.empty() ? 0 : policies[0].size();
        
        py::array_t<float> policies_array({num_policies, policy_size});
        py::buffer_info policies_info = policies_array.request();
        float* policies_data = static_cast<float*>(policies_info.ptr);
        
        for (size_t i = 0; i < num_policies; ++i) {
            for (size_t j = 0; j < policy_size; ++j) {
                policies_data[i * policy_size + j] = policies[i][j];
            }
        }
        
        return py::make_tuple(moves_array, policies_array, winner);
    }
    
    py::dict evaluate_position(const std::string& game_type, const std::vector<int>& moves) {
        // Create game
        auto game = createGame(game_type);
        if (!game) {
            throw std::runtime_error("Invalid game type: " + game_type);
        }
        
        // Apply moves
        for (int move : moves) {
            if (!game->isLegalMove(move)) {
                throw std::runtime_error("Illegal move: " + std::to_string(move));
            }
            game->makeMove(move);
            
            if (game->isTerminal()) {
                break;
            }
        }
        
        // Run search
        auto result = mcts_engine_.search(*game);
        
        // Create return dictionary
        py::dict ret;
        ret["value"] = result.value;
        ret["policy"] = py::array_t<float>(result.probabilities.size(), result.probabilities.data());
        ret["nodes"] = result.stats.total_nodes;
        ret["time_ms"] = result.stats.search_time.count();
        
        return ret;
    }
    
    const mcts::MCTSStats& get_last_stats() const {
        return mcts_engine_.getLastStats();
    }
    
private:
    std::unique_ptr<core::IGameState> createGame(const std::string& game_type) {
        if (game_type == "chess") {
            return std::make_unique<games::chess::ChessState>();
        } else if (game_type == "go") {
            return std::make_unique<games::go::GoState>();
        } else if (game_type == "gomoku") {
            return std::make_unique<games::gomoku::GomokuState>();
        } else {
            return nullptr;
        }
    }
    
    NeuralNetWrapper neural_net_;
    mcts::MCTSEngine mcts_engine_;
};

// Module definition
PYBIND11_MODULE(alphazero_py, m) {
    m.doc() = "AlphaZero Python bindings";
    
    // Game types
    py::enum_<core::GameType>(m, "GameType")
        .value("UNKNOWN", core::GameType::UNKNOWN)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .value("GOMOKU", core::GameType::GOMOKU)
        .export_values();
    
    // Game result
    py::enum_<core::GameResult>(m, "GameResult")
        .value("ONGOING", core::GameResult::ONGOING)
        .value("WIN_PLAYER1", core::GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", core::GameResult::WIN_PLAYER2)
        .value("DRAW", core::GameResult::DRAW)
        .export_values();
    
    // MCTS Settings
    py::class_<mcts::MCTSSettings>(m, "MCTSSettings")
        .def(py::init<>())
        .def_readwrite("num_simulations", &mcts::MCTSSettings::num_simulations)
        .def_readwrite("num_threads", &mcts::MCTSSettings::num_threads)
        .def_readwrite("batch_size", &mcts::MCTSSettings::batch_size)
        .def_readwrite("exploration_constant", &mcts::MCTSSettings::exploration_constant)
        .def_readwrite("virtual_loss", &mcts::MCTSSettings::virtual_loss)
        .def_readwrite("add_dirichlet_noise", &mcts::MCTSSettings::add_dirichlet_noise)
        .def_readwrite("dirichlet_alpha", &mcts::MCTSSettings::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &mcts::MCTSSettings::dirichlet_epsilon)
        .def_readwrite("temperature", &mcts::MCTSSettings::temperature);
    
    // MCTS Stats
    py::class_<mcts::MCTSStats>(m, "MCTSStats")
        .def(py::init<>())
        .def_readonly("total_nodes", &mcts::MCTSStats::total_nodes)
        .def_readonly("max_depth", &mcts::MCTSStats::max_depth)
        .def_readonly("search_time", &mcts::MCTSStats::search_time)
        .def_readonly("avg_batch_size", &mcts::MCTSStats::avg_batch_size)
        .def_readonly("avg_batch_latency", &mcts::MCTSStats::avg_batch_latency)
        .def_readonly("total_evaluations", &mcts::MCTSStats::total_evaluations)
        .def_readonly("nodes_per_second", &mcts::MCTSStats::nodes_per_second);
    
    // Neural network wrapper
    py::class_<NeuralNetWrapper>(m, "NeuralNetWrapper")
        .def(py::init<NeuralNetWrapper::InferenceCallback>());
    
    // Self-play manager
    py::class_<SelfPlayManager>(m, "SelfPlayManager")
        .def(py::init<const mcts::MCTSSettings&, NeuralNetWrapper::InferenceCallback>())
        .def("generate_game", &SelfPlayManager::generate_game,
             py::arg("game_type"), py::arg("max_moves") = 1000)
        .def("evaluate_position", &SelfPlayManager::evaluate_position,
             py::arg("game_type"), py::arg("moves"))
        .def("get_last_stats", &SelfPlayManager::get_last_stats);
    
    // Game interface
    py::class_<core::IGameState>(m, "IGameState")
        .def("get_legal_moves", &core::IGameState::getLegalMoves)
        .def("is_legal_move", &core::IGameState::isLegalMove)
        .def("make_move", &core::IGameState::makeMove)
        .def("undo_move", &core::IGameState::undoMove)
        .def("is_terminal", &core::IGameState::isTerminal)
        .def("get_game_result", &core::IGameState::getGameResult)
        .def("get_current_player", &core::IGameState::getCurrentPlayer)
        .def("get_board_size", &core::IGameState::getBoardSize)
        .def("get_action_space_size", &core::IGameState::getActionSpaceSize)
        .def("get_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getTensorRepresentation());
        })
        .def("get_enhanced_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getEnhancedTensorRepresentation());
        })
        .def("get_hash", &core::IGameState::getHash)
        .def("action_to_string", &core::IGameState::actionToString)
        .def("string_to_action", &core::IGameState::stringToAction)
        .def("to_string", &core::IGameState::toString)
        .def("get_move_history", &core::IGameState::getMoveHistory);
    
    // Game factory
    m.def("create_game", [](core::GameType type) {
        return core::GameFactory::createGame(type);
    });
    
    // Game-specific classes
    py::class_<games::chess::ChessState, core::IGameState>(m, "ChessState")
        .def(py::init<>());

    py::class_<games::go::GoState, core::IGameState>(m, "GoState")
        .def(py::init<>());

    py::class_<games::gomoku::GomokuState, core::IGameState>(m, "GomokuState")
        .def(py::init<>());
}

} // namespace python
} // namespace alphazero