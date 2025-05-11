// src/training/training_data_manager.cpp
#include "training/training_data_manager.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace alphazero {
namespace training {

TrainingDataManager::TrainingDataManager(const TrainingDataSettings& settings)
    : settings_(settings),
      total_examples_(0) {
    
    // Initialize random number generator
    if (settings_.random_seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(settings_.random_seed));
    }
}

void TrainingDataManager::addGames(const std::vector<selfplay::GameData>& games, int iteration) {
    // Get examples from games
    auto [states, targets] = selfplay::SelfPlayManager::convertToTrainingExamples(games);
    auto& [policies, values] = targets;
    
    // Create examples vector if it doesn't exist
    if (examples_by_iteration_.find(iteration) == examples_by_iteration_.end()) {
        examples_by_iteration_[iteration] = std::vector<TrainingExample>();
        iterations_.push_back(iteration);
        
        // Sort iterations
        std::sort(iterations_.begin(), iterations_.end());
    }
    
    // Add examples to the appropriate iteration
    auto& examples = examples_by_iteration_[iteration];
    size_t prev_examples = examples.size();
    examples.reserve(prev_examples + states.size());
    
    size_t move_number = 0;
    std::string current_game_id;
    
    for (size_t i = 0; i < states.size(); ++i) {
        // Check if we're starting a new game
        if (i > 0 && (i % games[0].moves.size() == 0)) {
            move_number = 0;
        }
        
        // Get game ID
        size_t game_idx = i / games[0].moves.size();
        if (game_idx < games.size()) {
            current_game_id = games[game_idx].game_id;
        } else {
            current_game_id = "unknown";
        }
        
        // Create example
        TrainingExample example;
        example.state = states[i];
        example.policy = policies[i];
        example.value = values[i];
        example.game_id = current_game_id;
        example.move_number = move_number++;
        
        examples.push_back(std::move(example));
    }
    
    // Update total examples
    total_examples_ += (examples.size() - prev_examples);
    
    // Trim old iterations if needed
    trimOldIterations();
}

std::vector<TrainingExample> TrainingDataManager::sampleBatch(size_t batch_size) {
    if (batch_size == 0) {
        batch_size = settings_.batch_size;
    }
    
    // Check if we have enough examples
    if (total_examples_ == 0) {
        return {};
    }
    
    // Determine which iterations to sample from
    std::vector<int> sample_iterations;
    
    if (iterations_.size() <= settings_.sample_recent_iterations) {
        // Sample from all iterations
        sample_iterations = iterations_;
    } else {
        // Sample from recent iterations
        size_t start_idx = iterations_.size() - settings_.sample_recent_iterations;
        sample_iterations.assign(iterations_.begin() + start_idx, iterations_.end());
    }
    
    // Create distribution for selecting iterations
    std::vector<size_t> examples_per_iteration;
    size_t total_examples_in_sample = 0;
    
    for (int iter : sample_iterations) {
        size_t num_examples = examples_by_iteration_[iter].size();
        examples_per_iteration.push_back(num_examples);
        total_examples_in_sample += num_examples;
    }
    
    std::discrete_distribution<size_t> iter_dist(examples_per_iteration.begin(), 
                                                examples_per_iteration.end());
    
    // Sample examples
    std::vector<TrainingExample> batch;
    batch.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Select iteration
        size_t iter_idx = iter_dist(rng_);
        int iteration = sample_iterations[iter_idx];
        
        // Select example from iteration
        const auto& examples = examples_by_iteration_[iteration];
        std::uniform_int_distribution<size_t> example_dist(0, examples.size() - 1);
        size_t example_idx = example_dist(rng_);
        
        batch.push_back(examples[example_idx]);
    }
    
    return batch;
}

size_t TrainingDataManager::getTotalExamples() const {
    return total_examples_;
}

std::unordered_map<int, size_t> TrainingDataManager::getExamplesPerIteration() const {
    std::unordered_map<int, size_t> result;
    
    for (const auto& [iteration, examples] : examples_by_iteration_) {
        result[iteration] = examples.size();
    }
    
    return result;
}

void TrainingDataManager::save(const std::string& directory, const std::string& format) {
    // Create directory if it doesn't exist
    std::filesystem::create_directories(directory);
    
    // Save metadata
    std::ofstream metadata(directory + "/metadata.txt");
    metadata << "total_examples: " << total_examples_ << "\n";
    metadata << "iterations: ";
    for (int iter : iterations_) {
        metadata << iter << " ";
    }
    metadata << "\n";
    
    // Save examples for each iteration
    for (int iteration : iterations_) {
        std::string iter_dir = directory + "/iteration_" + std::to_string(iteration);
        std::filesystem::create_directories(iter_dir);
        
        const auto& examples = examples_by_iteration_[iteration];
        
        if (format == "binary") {
            // Save in binary format
            std::string filename = iter_dir + "/examples.bin";
            std::ofstream file(filename, std::ios::binary);
            
            // Write header
            int32_t magic = 0x415A5445;  // "AZTE" (AlphaZero Training Examples)
            int32_t version = 1;
            int32_t num_examples = static_cast<int32_t>(examples.size());
            
            file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            file.write(reinterpret_cast<const char*>(&version), sizeof(version));
            file.write(reinterpret_cast<const char*>(&num_examples), sizeof(num_examples));
            
            // Write examples
            for (const auto& example : examples) {
                // Write state dimensions
                int32_t channels = static_cast<int32_t>(example.state.size());
                int32_t height = channels > 0 ? static_cast<int32_t>(example.state[0].size()) : 0;
                int32_t width = (channels > 0 && height > 0) ? 
                               static_cast<int32_t>(example.state[0][0].size()) : 0;
                
                file.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
                file.write(reinterpret_cast<const char*>(&height), sizeof(height));
                file.write(reinterpret_cast<const char*>(&width), sizeof(width));
                
                // Write state data
                for (const auto& channel : example.state) {
                    for (const auto& row : channel) {
                        file.write(reinterpret_cast<const char*>(row.data()), width * sizeof(float));
                    }
                }
                
                // Write policy
                int32_t policy_size = static_cast<int32_t>(example.policy.size());
                file.write(reinterpret_cast<const char*>(&policy_size), sizeof(policy_size));
                file.write(reinterpret_cast<const char*>(example.policy.data()), 
                          policy_size * sizeof(float));
                
                // Write value
                file.write(reinterpret_cast<const char*>(&example.value), sizeof(example.value));
                
                // Write game ID
                int32_t game_id_length = static_cast<int32_t>(example.game_id.length());
                file.write(reinterpret_cast<const char*>(&game_id_length), sizeof(game_id_length));
                file.write(example.game_id.c_str(), game_id_length);
                
                // Write move number
                int32_t move_number = static_cast<int32_t>(example.move_number);
                file.write(reinterpret_cast<const char*>(&move_number), sizeof(move_number));
            }
        } else {
            throw std::runtime_error("Unsupported format: " + format);
        }
    }
}

void TrainingDataManager::load(const std::string& directory, const std::string& format) {
    // Clear existing data
    clear();
    
    // Check if directory exists
    if (!std::filesystem::exists(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory);
    }
    
    // Load metadata
    std::ifstream metadata(directory + "/metadata.txt");
    if (!metadata) {
        throw std::runtime_error("Metadata file not found");
    }
    
    std::string line;
    while (std::getline(metadata, line)) {
        if (line.find("iterations:") == 0) {
            std::istringstream iss(line.substr(11));  // Skip "iterations: "
            int iteration;
            while (iss >> iteration) {
                iterations_.push_back(iteration);
            }
        }
    }
    
    // Load examples for each iteration
    for (int iteration : iterations_) {
        std::string iter_dir = directory + "/iteration_" + std::to_string(iteration);
        
        if (!std::filesystem::exists(iter_dir)) {
            std::cerr << "Iteration directory not found: " << iter_dir << std::endl;
            continue;
        }
        
        if (format == "binary") {
            // Load from binary format
            std::string filename = iter_dir + "/examples.bin";
            if (!std::filesystem::exists(filename)) {
                std::cerr << "Examples file not found: " << filename << std::endl;
                continue;
            }
            
            std::ifstream file(filename, std::ios::binary);
            
            // Read header
            int32_t magic, version, num_examples;
            
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            file.read(reinterpret_cast<char*>(&version), sizeof(version));
            file.read(reinterpret_cast<char*>(&num_examples), sizeof(num_examples));
            
            // Validate magic number
            if (magic != 0x415A5445) {
                std::cerr << "Invalid file format: " << filename << std::endl;
                continue;
            }
            
            // Read examples
            std::vector<TrainingExample> examples;
            examples.reserve(num_examples);
            
            for (int i = 0; i < num_examples; ++i) {
                TrainingExample example;
                
                // Read state dimensions
                int32_t channels, height, width;
                file.read(reinterpret_cast<char*>(&channels), sizeof(channels));
                file.read(reinterpret_cast<char*>(&height), sizeof(height));
                file.read(reinterpret_cast<char*>(&width), sizeof(width));
                
                // Read state data
                example.state.resize(channels, std::vector<std::vector<float>>(
                    height, std::vector<float>(width)));
                
                for (int32_t c = 0; c < channels; ++c) {
                    for (int32_t h = 0; h < height; ++h) {
                        file.read(reinterpret_cast<char*>(example.state[c][h].data()), 
                                 width * sizeof(float));
                    }
                }
                
                // Read policy
                int32_t policy_size;
                file.read(reinterpret_cast<char*>(&policy_size), sizeof(policy_size));
                example.policy.resize(policy_size);
                file.read(reinterpret_cast<char*>(example.policy.data()), 
                         policy_size * sizeof(float));
                
                // Read value
                file.read(reinterpret_cast<char*>(&example.value), sizeof(example.value));
                
                // Read game ID
                int32_t game_id_length;
                file.read(reinterpret_cast<char*>(&game_id_length), sizeof(game_id_length));
                example.game_id.resize(game_id_length);
                file.read(&example.game_id[0], game_id_length);
                
                // Read move number
                int32_t move_number;
                file.read(reinterpret_cast<char*>(&move_number), sizeof(move_number));
                example.move_number = move_number;
                
                examples.push_back(std::move(example));
            }
            
            // Add examples to the appropriate iteration
            examples_by_iteration_[iteration] = std::move(examples);
            total_examples_ += examples_by_iteration_[iteration].size();
        } else {
            throw std::runtime_error("Unsupported format: " + format);
        }
    }
}

void TrainingDataManager::clear() {
    examples_by_iteration_.clear();
    iterations_.clear();
    total_examples_ = 0;
}

void TrainingDataManager::trimOldIterations() {
    // Check if we need to trim
    if (total_examples_ <= settings_.max_examples) {
        return;
    }
    
    // Sort iterations
    std::sort(iterations_.begin(), iterations_.end());
    
    // Start removing oldest iterations
    size_t examples_to_remove = total_examples_ - settings_.max_examples;
    
    while (examples_to_remove > 0 && 
           iterations_.size() > settings_.min_iterations_to_keep) {
        int oldest_iteration = iterations_.front();
        size_t examples_in_oldest = examples_by_iteration_[oldest_iteration].size();
        
        if (examples_in_oldest <= examples_to_remove) {
            // Remove entire iteration
            examples_to_remove -= examples_in_oldest;
            total_examples_ -= examples_in_oldest;
            examples_by_iteration_.erase(oldest_iteration);
            iterations_.erase(iterations_.begin());
        } else {
            // Remove some examples from oldest iteration
            auto& examples = examples_by_iteration_[oldest_iteration];
            examples.erase(examples.begin(), examples.begin() + examples_to_remove);
            total_examples_ -= examples_to_remove;
            examples_to_remove = 0;
        }
    }
}

} // namespace training
} // namespace alphazero