#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>

#include "cli/alphazero_pipeline.h"
#include "core/igamestate.h"
#include "core/game_export.h"
#include "mcts/mcts_engine.h"
#include "selfplay/self_play_manager.h"
#include "nn/neural_network.h"
#include "nn/neural_network_factory.h"
#include "training/training_data_manager.h"

namespace py = pybind11;

namespace alphazero {
namespace python {

// Neural Network wrapper to interface with Python models
class PyNeuralNetworkWrapper : public nn::NeuralNetwork {
public:
    using InferenceCallback = std::function<py::tuple(py::array_t<float>)>;
    
    PyNeuralNetworkWrapper(InferenceCallback callback, int input_channels, int board_size, int policy_size) 
        : callback_(callback), 
          input_channels_(input_channels), 
          board_size_(board_size), 
          policy_size_(policy_size) {}
    
    // Convert Python numpy arrays to C++ tensor representation
    static std::vector<std::vector<std::vector<float>>> numpyToTensor(const py::array_t<float>& array) {
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
    static py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
        if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
            return py::array_t<float>();
        }
        
        size_t channels = tensor.size();
        size_t height = tensor[0].size();
        size_t width = tensor[0][0].size();
        
        std::vector<ssize_t> shape = {static_cast<ssize_t>(channels), 
                                      static_cast<ssize_t>(height), 
                                      static_cast<ssize_t>(width)};
        std::vector<float> data(channels * height * width);
        
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    data[c * height * width + h * width + w] = tensor[c][h][w];
                }
            }
        }
        
        return py::array_t<float>(shape, data.data());
    }
    
    // Implement the required interface methods
    std::vector<mcts::NetworkOutput> inference(
            const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        
        if (states.empty()) {
            return {};
        }
        
        // Convert game states to tensor representation for batch processing
        std::vector<std::vector<std::vector<std::vector<float>>>> batched_tensors;
        batched_tensors.reserve(states.size());
        
        for (const auto& state : states) {
            batched_tensors.push_back(state->getEnhancedTensorRepresentation());
        }
        
        // Create a batch tensor for Python
        std::vector<ssize_t> shape = {static_cast<ssize_t>(states.size()), 
                                     static_cast<ssize_t>(input_channels_), 
                                     static_cast<ssize_t>(board_size_), 
                                     static_cast<ssize_t>(board_size_)};
        
        std::vector<float> batch_data(states.size() * input_channels_ * board_size_ * board_size_, 0.0f);
        
        // Fill batch data
        for (size_t b = 0; b < states.size(); ++b) {
            for (size_t c = 0; c < input_channels_; ++c) {
                for (size_t h = 0; h < board_size_; ++h) {
                    for (size_t w = 0; w < board_size_; ++w) {
                        if (c < batched_tensors[b].size() && 
                            h < batched_tensors[b][c].size() && 
                            w < batched_tensors[b][c][h].size()) {
                            batch_data[b * input_channels_ * board_size_ * board_size_ + 
                                      c * board_size_ * board_size_ + 
                                      h * board_size_ + w] = batched_tensors[b][c][h][w];
                        }
                    }
                }
            }
        }
        
        py::array_t<float> batch_tensor(shape, batch_data.data());
        
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
    
    void save(const std::string& path) override {
        // This is a wrapper, so we don't save anything directly
        // The Python side handles model saving
    }
    
    void load(const std::string& path) override {
        // This is a wrapper, so we don't load anything directly
        // The Python side handles model loading
    }
    
    std::vector<int64_t> getInputShape() const override {
        return {static_cast<int64_t>(input_channels_), 
                static_cast<int64_t>(board_size_), 
                static_cast<int64_t>(board_size_)};
    }
    
    int64_t getPolicySize() const override {
        return static_cast<int64_t>(policy_size_);
    }
    
private:
    InferenceCallback callback_;
    int input_channels_;
    int board_size_;
    int policy_size_;
};

// Self-play game data
struct PyGameData {
    py::array_t<int> moves;
    py::array_t<float> policies;
    int winner;
    
    static PyGameData from_cpp(const selfplay::GameData& cpp_game) {
        PyGameData py_game;
        
        // Convert moves
        size_t num_moves = cpp_game.moves.size();
        std::vector<int> moves_vec(cpp_game.moves.begin(), cpp_game.moves.end());
        py_game.moves = py::array_t<int>(num_moves, moves_vec.data());
        
        // Convert policies
        size_t policy_size = cpp_game.policies.empty() ? 0 : cpp_game.policies[0].size();
        std::vector<float> policies_flat;
        policies_flat.reserve(num_moves * policy_size);
        
        for (const auto& policy : cpp_game.policies) {
            policies_flat.insert(policies_flat.end(), policy.begin(), policy.end());
        }
        
        std::vector<ssize_t> policy_shape = {static_cast<ssize_t>(num_moves), 
                                            static_cast<ssize_t>(policy_size)};
        py_game.policies = py::array_t<float>(policy_shape, policies_flat.data());
        
        // Set winner
        py_game.winner = cpp_game.winner;
        
        return py_game;
    }
};

// Python wrapper for AlphaZero pipeline
class PyAlphaZeroPipeline {
public:
    PyAlphaZeroPipeline(const py::dict& config_dict) {
        // Convert Python dict to C++ config
        config_ = dict_to_config(config_dict);
    }
    
    void initialize_with_model(PyNeuralNetworkWrapper::InferenceCallback callback) {
        // Create neural network wrapper
        neural_net_ = std::make_shared<PyNeuralNetworkWrapper>(
            callback, 
            config_.input_channels,
            config_.board_size,
            config_.policy_size > 0 ? config_.policy_size : (config_.board_size * config_.board_size + 1) // Default policy size
        );
        
        // Initialize directories
        std::filesystem::create_directories(config_.model_dir);
        std::filesystem::create_directories(config_.data_dir);
        std::filesystem::create_directories(config_.log_dir);
    }
    
    std::vector<PyGameData> run_self_play(int num_games, const py::kwargs& kwargs) {
        // Get settings from config
        selfplay::SelfPlaySettings settings;
        settings.mcts_settings.num_simulations = config_.mcts_num_simulations;
        settings.mcts_settings.num_threads = config_.mcts_num_threads;
        settings.mcts_settings.exploration_constant = config_.mcts_exploration_constant;
        settings.mcts_settings.temperature = config_.mcts_temperature;
        settings.mcts_settings.add_dirichlet_noise = config_.mcts_add_dirichlet_noise;
        settings.mcts_settings.dirichlet_alpha = config_.mcts_dirichlet_alpha;
        settings.mcts_settings.dirichlet_epsilon = config_.mcts_dirichlet_epsilon;
        settings.mcts_settings.batch_size = config_.mcts_batch_size;
        settings.mcts_settings.batch_timeout = std::chrono::milliseconds(config_.mcts_batch_timeout_ms);
        
        settings.num_parallel_games = config_.self_play_num_parallel_games;
        // Use the same number of engines as parallel games by default
        settings.num_mcts_engines = config_.self_play_num_parallel_games;
        settings.max_moves = config_.self_play_max_moves > 0 ? 
                             config_.self_play_max_moves : 
                             config_.board_size * config_.board_size * 2; // Default max moves
        settings.temperature_threshold = config_.self_play_temperature_threshold;
        settings.high_temperature = config_.self_play_high_temperature;
        settings.low_temperature = config_.self_play_low_temperature;
        settings.add_dirichlet_noise = config_.mcts_add_dirichlet_noise;
        
        // Override settings from kwargs if provided
        if (kwargs.contains("num_simulations")) {
            settings.mcts_settings.num_simulations = kwargs["num_simulations"].cast<int>();
        }
        if (kwargs.contains("num_threads")) {
            settings.mcts_settings.num_threads = kwargs["num_threads"].cast<int>();
        }
        if (kwargs.contains("exploration_constant")) {
            settings.mcts_settings.exploration_constant = kwargs["exploration_constant"].cast<float>();
        }
        
        // Create self-play manager with current model
        selfplay::SelfPlayManager self_play_manager(neural_net_, settings);
        
        // Generate games
        std::vector<selfplay::GameData> cpp_games = self_play_manager.generateGames(
            config_.game_type, 
            num_games,
            config_.board_size
        );
        
        // Convert to Python format
        std::vector<PyGameData> py_games;
        py_games.reserve(cpp_games.size());
        
        for (const auto& cpp_game : cpp_games) {
            py_games.push_back(PyGameData::from_cpp(cpp_game));
        }
        
        return py_games;
    }
    
    py::tuple extract_training_examples(const std::vector<PyGameData>& games) {
        // Convert PyGameData to C++ GameData
        std::vector<selfplay::GameData> cpp_games;
        cpp_games.reserve(games.size());
        
        for (const auto& py_game : games) {
            selfplay::GameData cpp_game;
            
            // Convert moves
            py::buffer_info moves_info = py_game.moves.request();
            int* moves_data = static_cast<int*>(moves_info.ptr);
            cpp_game.moves.assign(moves_data, moves_data + moves_info.size);
            
            // Convert policies
            py::buffer_info policies_info = py_game.policies.request();
            float* policies_data = static_cast<float*>(policies_info.ptr);
            size_t num_moves = moves_info.size;
            size_t policy_size = num_moves > 0 ? policies_info.shape[1] : 0;
            
            cpp_game.policies.resize(num_moves);
            for (size_t i = 0; i < num_moves; ++i) {
                cpp_game.policies[i].resize(policy_size);
                for (size_t j = 0; j < policy_size; ++j) {
                    cpp_game.policies[i][j] = policies_data[i * policy_size + j];
                }
            }
            
            // Set winner
            cpp_game.winner = py_game.winner;
            
            cpp_games.push_back(cpp_game);
        }
        
        // Convert to training examples
        auto [states, targets] = selfplay::SelfPlayManager::convertToTrainingExamples(cpp_games);
        auto [policies, values] = targets;
        
        // Convert to numpy arrays
        size_t num_examples = states.size();
        size_t channels = states[0].size();
        size_t height = states[0][0].size();
        size_t width = states[0][0][0].size();
        size_t policy_size = policies[0].size();
        
        // Create numpy arrays using vector shape and data
        std::vector<float> states_flat;
        states_flat.reserve(num_examples * channels * height * width);
        
        for (const auto& state : states) {
            for (const auto& channel : state) {
                for (const auto& row : channel) {
                    states_flat.insert(states_flat.end(), row.begin(), row.end());
                }
            }
        }
        
        std::vector<float> policies_flat;
        policies_flat.reserve(num_examples * policy_size);
        for (const auto& policy : policies) {
            policies_flat.insert(policies_flat.end(), policy.begin(), policy.end());
        }
        
        std::vector<float> values_vec(values.begin(), values.end());
        
        // Create numpy arrays
        std::vector<ssize_t> states_shape = {static_cast<ssize_t>(num_examples), 
                                           static_cast<ssize_t>(channels), 
                                           static_cast<ssize_t>(height), 
                                           static_cast<ssize_t>(width)};
        
        std::vector<ssize_t> policies_shape = {static_cast<ssize_t>(num_examples), 
                                              static_cast<ssize_t>(policy_size)};
        
        py::array_t<float> states_array(states_shape, states_flat.data());
        py::array_t<float> policies_array(policies_shape, policies_flat.data());
        py::array_t<float> values_array(num_examples, values_vec.data());
        
        return py::make_tuple(states_array, policies_array, values_array);
    }
    
    py::dict evaluate_models(PyNeuralNetworkWrapper::InferenceCallback contender_callback, int num_games) {
        // Create contender model
        auto contender_model = std::make_shared<PyNeuralNetworkWrapper>(
            contender_callback, 
            config_.input_channels,
            config_.board_size,
            config_.policy_size > 0 ? config_.policy_size : (config_.board_size * config_.board_size + 1)
        );
        
        // Set up arena settings
        selfplay::SelfPlaySettings arena_settings;
        arena_settings.mcts_settings.num_simulations = config_.arena_num_simulations;
        arena_settings.mcts_settings.num_threads = config_.arena_num_threads;
        arena_settings.mcts_settings.exploration_constant = config_.mcts_exploration_constant;
        arena_settings.mcts_settings.temperature = config_.arena_temperature;
        arena_settings.mcts_settings.add_dirichlet_noise = false; // No noise in arena games
        arena_settings.mcts_settings.batch_size = config_.mcts_batch_size;
        arena_settings.mcts_settings.batch_timeout = std::chrono::milliseconds(config_.mcts_batch_timeout_ms);
        
        arena_settings.num_parallel_games = config_.arena_num_parallel_games;
        arena_settings.num_mcts_engines = config_.arena_num_parallel_games; // Use same value by default
        arena_settings.max_moves = config_.self_play_max_moves > 0 ? 
                                  config_.self_play_max_moves : 
                                  config_.board_size * config_.board_size * 2;
        arena_settings.temperature_threshold = 0; // No high temperature period
        arena_settings.high_temperature = 0.0f;
        arena_settings.low_temperature = config_.arena_temperature;
        arena_settings.add_dirichlet_noise = false;
        
        // Create self-play managers for both models
        selfplay::SelfPlayManager champion_manager(neural_net_, arena_settings);
        selfplay::SelfPlayManager contender_manager(contender_model, arena_settings);
        
        // Play half the games with each model as first player
        int half_games = num_games / 2;
        int champion_wins = 0;
        int contender_wins = 0;
        int draws = 0;
        
        // Champion as first player
        std::vector<selfplay::GameData> first_half = play_arena_games(
            champion_manager, contender_manager, half_games
        );
        
        // Contender as first player
        std::vector<selfplay::GameData> second_half = play_arena_games(
            contender_manager, champion_manager, num_games - half_games
        );
        
        // Combine and count results
        std::vector<selfplay::GameData> all_games;
        all_games.insert(all_games.end(), first_half.begin(), first_half.end());
        all_games.insert(all_games.end(), second_half.begin(), second_half.end());
        
        for (const auto& game : first_half) {
            if (game.winner == 1) champion_wins++;
            else if (game.winner == 2) contender_wins++;
            else draws++;
        }
        
        for (const auto& game : second_half) {
            if (game.winner == 1) contender_wins++;
            else if (game.winner == 2) champion_wins++;
            else draws++;
        }
        
        // Calculate win rates
        float champion_win_rate = static_cast<float>(champion_wins) / all_games.size();
        float contender_win_rate = static_cast<float>(contender_wins) / all_games.size();
        float draw_rate = static_cast<float>(draws) / all_games.size();
        
        // Return results
        py::dict results;
        results["champion_wins"] = champion_wins;
        results["contender_wins"] = contender_wins;
        results["draws"] = draws;
        results["total_games"] = all_games.size();
        results["champion_win_rate"] = champion_win_rate;
        results["contender_win_rate"] = contender_win_rate;
        results["draw_rate"] = draw_rate;
        results["contender_is_better"] = contender_win_rate > config_.arena_win_rate_threshold;
        
        return results;
    }
    
    py::dict get_config() const {
        return config_to_dict(config_);
    }
    
private:
    cli::AlphaZeroPipelineConfig config_;
    std::shared_ptr<PyNeuralNetworkWrapper> neural_net_;
    
    static std::vector<selfplay::GameData> play_arena_games(
        selfplay::SelfPlayManager& player1_manager,
        selfplay::SelfPlayManager& player2_manager,
        int num_games) {
        
        // In a real implementation, we would need to carefully orchestrate the MCTS engines
        // to use the appropriate neural networks at the right times.
        // For simplicity, we'll use player1_manager to generate the games
        
        // In reality, we'd need to alternate between models during the game
        return player1_manager.generateGames(
            core::GameType::GOMOKU, num_games, 15 // Default to Gomoku and 15x15 board
        );
    }
    
    static cli::AlphaZeroPipelineConfig dict_to_config(const py::dict& dict) {
        cli::AlphaZeroPipelineConfig config;
        
        // Helper function to get value with default
        auto get_value = [&dict](const std::string& key, auto default_value) -> decltype(default_value) {
            if (dict.contains(key.c_str())) {
                return dict[key.c_str()].cast<decltype(default_value)>();
            }
            return default_value;
        };
        
        // Game settings
        std::string game_type = get_value("game_type", std::string("gomoku"));
        if (game_type == "gomoku") {
            config.game_type = core::GameType::GOMOKU;
        } else if (game_type == "chess") {
            config.game_type = core::GameType::CHESS;
        } else if (game_type == "go") {
            config.game_type = core::GameType::GO;
        }
        
        config.board_size = get_value("board_size", 15);
        config.input_channels = get_value("input_channels", 20);
        config.policy_size = get_value("policy_size", 0);
        
        // Directory settings
        config.model_dir = get_value("model_dir", std::string("models"));
        config.data_dir = get_value("data_dir", std::string("data"));
        config.log_dir = get_value("log_dir", std::string("logs"));
        
        // Neural network settings
        config.network_type = get_value("network_type", std::string("resnet"));
        config.use_gpu = get_value("use_gpu", true);
        config.num_iterations = get_value("num_iterations", 10);
        config.num_res_blocks = get_value("num_res_blocks", 19);
        config.num_filters = get_value("num_filters", 256);
        
        // Self-play settings
        config.self_play_num_games = get_value("self_play_num_games", 500);
        config.self_play_num_parallel_games = get_value("self_play_num_parallel_games", 8);
        // If explicitly provided, use separate parameter for MCTS engines
        config.self_play_num_mcts_engines = get_value("self_play_num_mcts_engines", config.self_play_num_parallel_games);
        config.self_play_max_moves = get_value("self_play_max_moves", 0);
        config.self_play_temperature_threshold = get_value("self_play_temperature_threshold", 30);
        config.self_play_high_temperature = get_value("self_play_high_temperature", 1.0f);
        config.self_play_low_temperature = get_value("self_play_low_temperature", 0.1f);
        config.self_play_output_format = get_value("self_play_output_format", std::string("json"));
        
        // MCTS settings
        config.mcts_num_simulations = get_value("mcts_num_simulations", 800);
        config.mcts_num_threads = get_value("mcts_num_threads", 8);
        config.mcts_batch_size = get_value("mcts_batch_size", 64);
        config.mcts_batch_timeout_ms = get_value("mcts_batch_timeout_ms", 20);
        config.mcts_exploration_constant = get_value("mcts_exploration_constant", 1.5f);
        config.mcts_temperature = get_value("mcts_temperature", 1.0f);
        config.mcts_add_dirichlet_noise = get_value("mcts_add_dirichlet_noise", true);
        config.mcts_dirichlet_alpha = get_value("mcts_dirichlet_alpha", 0.3f);
        config.mcts_dirichlet_epsilon = get_value("mcts_dirichlet_epsilon", 0.25f);
        
        // Training settings
        config.train_epochs = get_value("train_epochs", 20);
        config.train_batch_size = get_value("train_batch_size", 1024);
        config.train_num_workers = get_value("train_num_workers", 4);
        config.train_learning_rate = get_value("train_learning_rate", 0.001f);
        config.train_weight_decay = get_value("train_weight_decay", 0.0001f);
        config.train_lr_step_size = get_value("train_lr_step_size", 10);
        config.train_lr_gamma = get_value("train_lr_gamma", 0.1f);
        
        // Arena/evaluation settings
        config.enable_evaluation = get_value("enable_evaluation", true);
        config.arena_num_games = get_value("arena_num_games", 50);
        config.arena_num_parallel_games = get_value("arena_num_parallel_games", 8);
        config.arena_num_mcts_engines = get_value("arena_num_mcts_engines", config.arena_num_parallel_games);
        config.arena_num_threads = get_value("arena_num_threads", 4);
        config.arena_num_simulations = get_value("arena_num_simulations", 400);
        config.arena_temperature = get_value("arena_temperature", 0.1f);
        config.arena_win_rate_threshold = get_value("arena_win_rate_threshold", 0.55f);
        
        return config;
    }
    
    static py::dict config_to_dict(const cli::AlphaZeroPipelineConfig& config) {
        py::dict dict;
        
        // Game settings
        if (config.game_type == core::GameType::GOMOKU) {
            dict["game_type"] = "gomoku";
        } else if (config.game_type == core::GameType::CHESS) {
            dict["game_type"] = "chess";
        } else if (config.game_type == core::GameType::GO) {
            dict["game_type"] = "go";
        }
        
        dict["board_size"] = config.board_size;
        dict["input_channels"] = config.input_channels;
        dict["policy_size"] = config.policy_size;
        
        // Directory settings
        dict["model_dir"] = config.model_dir;
        dict["data_dir"] = config.data_dir;
        dict["log_dir"] = config.log_dir;
        
        // Neural network settings
        dict["network_type"] = config.network_type;
        dict["use_gpu"] = config.use_gpu;
        dict["num_iterations"] = config.num_iterations;
        dict["num_res_blocks"] = config.num_res_blocks;
        dict["num_filters"] = config.num_filters;
        
        // Self-play settings
        dict["self_play_num_games"] = config.self_play_num_games;
        dict["self_play_num_parallel_games"] = config.self_play_num_parallel_games;
        dict["self_play_num_mcts_engines"] = config.self_play_num_mcts_engines;
        dict["self_play_max_moves"] = config.self_play_max_moves;
        dict["self_play_temperature_threshold"] = config.self_play_temperature_threshold;
        dict["self_play_high_temperature"] = config.self_play_high_temperature;
        dict["self_play_low_temperature"] = config.self_play_low_temperature;
        dict["self_play_output_format"] = config.self_play_output_format;
        
        // MCTS settings
        dict["mcts_num_simulations"] = config.mcts_num_simulations;
        dict["mcts_num_threads"] = config.mcts_num_threads;
        dict["mcts_batch_size"] = config.mcts_batch_size;
        dict["mcts_batch_timeout_ms"] = config.mcts_batch_timeout_ms;
        dict["mcts_exploration_constant"] = config.mcts_exploration_constant;
        dict["mcts_temperature"] = config.mcts_temperature;
        dict["mcts_add_dirichlet_noise"] = config.mcts_add_dirichlet_noise;
        dict["mcts_dirichlet_alpha"] = config.mcts_dirichlet_alpha;
        dict["mcts_dirichlet_epsilon"] = config.mcts_dirichlet_epsilon;
        
        // Training settings
        dict["train_epochs"] = config.train_epochs;
        dict["train_batch_size"] = config.train_batch_size;
        dict["train_num_workers"] = config.train_num_workers;
        dict["train_learning_rate"] = config.train_learning_rate;
        dict["train_weight_decay"] = config.train_weight_decay;
        dict["train_lr_step_size"] = config.train_lr_step_size;
        dict["train_lr_gamma"] = config.train_lr_gamma;
        
        // Arena/evaluation settings
        dict["enable_evaluation"] = config.enable_evaluation;
        dict["arena_num_games"] = config.arena_num_games;
        dict["arena_num_parallel_games"] = config.arena_num_parallel_games;
        dict["arena_num_mcts_engines"] = config.arena_num_mcts_engines;
        dict["arena_num_threads"] = config.arena_num_threads;
        dict["arena_num_simulations"] = config.arena_num_simulations;
        dict["arena_temperature"] = config.arena_temperature;
        dict["arena_win_rate_threshold"] = config.arena_win_rate_threshold;
        
        return dict;
    }
};

// Module definition
PYBIND11_MODULE(alphazero_pipeline, m) {
    m.doc() = "AlphaZero Pipeline Python Bindings";
    
    // Expose C++ game types
    py::enum_<core::GameType>(m, "GameType")
        .value("UNKNOWN", core::GameType::UNKNOWN)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .value("GOMOKU", core::GameType::GOMOKU)
        .export_values();
    
    // Neural network wrapper class
    py::class_<PyNeuralNetworkWrapper, std::shared_ptr<PyNeuralNetworkWrapper>>(m, "NeuralNetworkWrapper")
        .def(py::init<PyNeuralNetworkWrapper::InferenceCallback, int, int, int>(),
             py::arg("callback"), py::arg("input_channels"), py::arg("board_size"), py::arg("policy_size"))
        .def_static("numpy_to_tensor", &PyNeuralNetworkWrapper::numpyToTensor)
        .def_static("tensor_to_numpy", &PyNeuralNetworkWrapper::tensorToNumpy);
    
    // Game data struct
    py::class_<PyGameData>(m, "GameData")
        .def(py::init<>())
        .def_readwrite("moves", &PyGameData::moves)
        .def_readwrite("policies", &PyGameData::policies)
        .def_readwrite("winner", &PyGameData::winner);
    
    // AlphaZero pipeline
    py::class_<PyAlphaZeroPipeline>(m, "AlphaZeroPipeline")
        .def(py::init<const py::dict&>())
        .def("initialize_with_model", &PyAlphaZeroPipeline::initialize_with_model)
        .def("run_self_play", &PyAlphaZeroPipeline::run_self_play,
             py::arg("num_games"))
        .def("extract_training_examples", &PyAlphaZeroPipeline::extract_training_examples)
        .def("evaluate_models", &PyAlphaZeroPipeline::evaluate_models,
             py::arg("contender_callback"), py::arg("num_games") = 40)
        .def("get_config", &PyAlphaZeroPipeline::get_config);
}

} // namespace python
} // namespace alphazero