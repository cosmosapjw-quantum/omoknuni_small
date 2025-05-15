// include/training/training_data_manager.h
#ifndef ALPHAZERO_TRAINING_DATA_MANAGER_H
#define ALPHAZERO_TRAINING_DATA_MANAGER_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <random>
#include "selfplay/self_play_manager.h"
#include "core/export_macros.h"

namespace alphazero {
namespace training {

/**
 * @brief Training example structure
 */
struct ALPHAZERO_API TrainingExample {
    // State tensor representation
    std::vector<std::vector<std::vector<float>>> state;
    
    // Policy target
    std::vector<float> policy;
    
    // Value target
    float value;
    
    // Game ID this example came from
    std::string game_id;
    
    // Move number in the game
    int move_number;
};

/**
 * @brief Settings for the training data manager
 */
struct ALPHAZERO_API TrainingDataSettings {
    // Maximum number of examples to keep in memory
    size_t max_examples = 500000;
    
    // Batch size for sampling
    size_t batch_size = 2048;
    
    // Random seed (-1 for time-based)
    int64_t random_seed = -1;
    
    // Value of samples from more recent iterations
    size_t sample_recent_iterations = 20;
    
    // Minimum number of iterations to keep
    size_t min_iterations_to_keep = 10;
};

/**
 * @brief Manager for training data
 */
class ALPHAZERO_API TrainingDataManager {
public:
    /**
     * @brief Constructor
     * 
     * @param settings Training data settings
     */
    explicit TrainingDataManager(const TrainingDataSettings& settings = TrainingDataSettings());
    
    /**
     * @brief Add self-play games to the training data
     * 
     * @param games Vector of game data
     * @param iteration Current iteration
     */
    void addGames(const std::vector<selfplay::GameData>& games, int iteration);
    
    /**
     * @brief Sample a batch of training examples
     * 
     * @param batch_size Size of the batch (0 for default)
     * @return Vector of training examples
     */
    std::vector<TrainingExample> sampleBatch(size_t batch_size = 0);
    
    /**
     * @brief Get the total number of examples
     * 
     * @return Number of examples
     */
    size_t getTotalExamples() const;
    
    /**
     * @brief Get the number of examples per iteration
     * 
     * @return Map of iteration to number of examples
     */
    std::unordered_map<int, size_t> getExamplesPerIteration() const;
    
    /**
     * @brief Save training data to disk
     * 
     * @param directory Directory to save to
     * @param format Format ("binary" or "tfrecord")
     */
    void save(const std::string& directory, const std::string& format = "binary");
    
    /**
     * @brief Load training data from disk
     * 
     * @param directory Directory to load from
     * @param format Format ("binary" or "tfrecord")
     */
    void load(const std::string& directory, const std::string& format = "binary");
    
    /**
     * @brief Clear all training data
     */
    void clear();
    
private:
    // Settings
    TrainingDataSettings settings_;

    // Training examples per iteration
    std::unordered_map<int, std::vector<TrainingExample>> examples_by_iteration_;
    
    // Total number of examples
    size_t total_examples_;
    
    // Current iterations (sorted)
    std::vector<int> iterations_;
    
    // Random number generator
    std::mt19937 rng_;
    
    /**
     * @brief Trim old iterations if needed
     */
    void trimOldIterations();
};

} // namespace training
} // namespace alphazero

#endif // ALPHAZERO_TRAINING_DATA_MANAGER_H