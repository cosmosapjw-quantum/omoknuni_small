#include "selfplay/self_play_manager.h"
#include "nn/neural_network.h"
#include "core/game_export.h"
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdlib>

using namespace alphazero;

// A mock neural network implementation for testing
class MockNeuralNetwork : public nn::NeuralNetwork {
public:
    MockNeuralNetwork() : policy_size_(225) {} // Default to 15x15
    
    // Implement required interface methods
    std::vector<mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        
        std::vector<mcts::NetworkOutput> results;
        for (const auto& state : states) {
            mcts::NetworkOutput output;
            int action_space = state->getActionSpaceSize();
            output.policy.resize(action_space, 1.0f / action_space);
            output.value = 0.0f;
            results.push_back(output);
        }
        return results;
    }
    
    void save(const std::string& path) override {
        // Mock implementation - do nothing
        std::cout << "Mock save to " << path << std::endl;
    }
    
    void load(const std::string& path) override {
        // Mock implementation - do nothing
        std::cout << "Mock load from " << path << std::endl;
    }
    
    std::vector<int64_t> getInputShape() const override {
        // Input shape is typically [channels, height, width]
        return {3, 15, 15}; // Default to 15x15 board with 3 channels
    }
    
    int64_t getPolicySize() const override {
        return policy_size_;
    }
    
private:
    int64_t policy_size_;
};

// Simple test function to verify self-play for a specific game type
bool test_self_play_for_game(core::GameType game_type, int num_games, int board_size) {
    try {
        std::cout << "Testing self-play for " << core::gameTypeToString(game_type) 
                  << " with board size " << board_size << std::endl;
        
        // Create temporary directory for output
        std::string output_dir = "/tmp/alphazero_test_" + 
                                 core::gameTypeToString(game_type) + 
                                 std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(output_dir);
        
        // Create mock neural network
        auto network = std::make_shared<MockNeuralNetwork>();
        
        // Create self-play settings with minimal MCTS settings
        selfplay::SelfPlaySettings sp_settings;
        sp_settings.num_parallel_games = 1;
        sp_settings.num_mcts_engines = 1;
        sp_settings.max_moves = board_size * board_size;  // Allow full games
        sp_settings.mcts_settings.num_simulations = 1;  // Minimal search
        sp_settings.mcts_settings.num_threads = 1;
        
        // Create self-play manager
        selfplay::SelfPlayManager manager(network, sp_settings);
        
        // Generate games - our modified implementation will use random moves
        auto games = manager.generateGames(game_type, num_games, board_size);
        
        // Verify games
        std::cout << "Generated " << games.size() << " games" << std::endl;
        
        if (games.size() != num_games) {
            std::cerr << "ERROR: Expected " << num_games << " games, got " 
                      << games.size() << std::endl;
            return false;
        }
        
        // Check each game
        for (const auto& game : games) {
            std::cout << "Game ID: " << game.game_id << ", moves: " 
                      << game.moves.size() << ", winner: " << game.winner << std::endl;
            
            // Verify move count
            if (game.moves.empty()) {
                std::cerr << "ERROR: Game has no moves" << std::endl;
                return false;
            }
            
            // Verify policies
            if (game.policies.size() != game.moves.size()) {
                std::cerr << "ERROR: Game has " << game.moves.size() 
                          << " moves but " << game.policies.size() 
                          << " policies" << std::endl;
                return false;
            }
        }
        
        // Save games
        manager.saveGames(games, output_dir, "json");
        
        // Check that saved files exist
        bool files_exist = false;
        try {
            if (std::filesystem::exists(output_dir)) {
                // Just check if the directory is not empty
                files_exist = !std::filesystem::is_empty(output_dir);
                std::cout << "Verified game files were saved successfully" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error checking saved files: " << e.what() << std::endl;
        }
        
        // Cleanup
        try {
            std::filesystem::remove_all(output_dir);
        } catch (...) {
            std::cerr << "Warning: Failed to remove temp directory" << std::endl;
        }
        
        return files_exist;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

int main(int, char**) {
    try {
        std::cout << "Starting self-play game tests..." << std::endl;
        
        bool success = true;
        
        // Test Gomoku
        std::cout << "\n=== Testing Gomoku ===" << std::endl;
        if (!test_self_play_for_game(core::GameType::GOMOKU, 1, 9)) {
            std::cerr << "Gomoku test failed!" << std::endl;
            success = false;
        }
        
        // Test Chess
        std::cout << "\n=== Testing Chess ===" << std::endl;
        if (!test_self_play_for_game(core::GameType::CHESS, 1, 8)) {
            std::cerr << "Chess test failed!" << std::endl;
            success = false;
        }
        
        // Test Go
        std::cout << "\n=== Testing Go ===" << std::endl;
        if (!test_self_play_for_game(core::GameType::GO, 1, 9)) {
            std::cerr << "Go test failed!" << std::endl;
            success = false;
        }
        
        if (success) {
            std::cout << "\nAll self-play game tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cerr << "\nSome self-play game tests FAILED!" << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}