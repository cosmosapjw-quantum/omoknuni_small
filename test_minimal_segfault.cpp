#include <iostream>
#include <memory>
#include <vector>
#include <functional>

// Forward declare only what we need
namespace alphazero {
namespace mcts {
    struct NetworkOutput {
        std::vector<float> policy;
        float value;
    };
    
    struct MCTSSettings {
        int num_simulations = 1;
        int num_threads = 0;
        int batch_size = 1;
        float exploration_constant = 1.5f;
        bool use_transposition_table = false;
    };
}

namespace core {
    class IGameState {
    public:
        virtual ~IGameState() = default;
        virtual int getActionSpaceSize() const = 0;
        virtual bool isTerminal() const = 0;
        virtual std::vector<int> getLegalMoves() const = 0;
    };
}
}

using InferenceFunction = std::function<std::vector<alphazero::mcts::NetworkOutput>(const std::vector<std::unique_ptr<alphazero::core::IGameState>>&)>;

// Simple mock inference function
std::vector<alphazero::mcts::NetworkOutput> mockInference(const std::vector<std::unique_ptr<alphazero::core::IGameState>>& states) {
    // std::cout << "[DEBUG] mockInference called with " << states.size() << " states" << std::endl;
    std::vector<alphazero::mcts::NetworkOutput> outputs;
    for (size_t i = 0; i < states.size(); ++i) {
        alphazero::mcts::NetworkOutput output;
        output.policy = std::vector<float>(2, 0.5f);  // Simple uniform policy
        output.value = 0.0f;
        outputs.push_back(output);
    }
    return outputs;
}

int main() {
    // std::cout << "[DEBUG] Starting minimal segfault test" << std::endl;
    
    try {
        // Test if the function pointer works
        // std::cout << "[DEBUG] Testing mock inference function" << std::endl;
        std::vector<std::unique_ptr<alphazero::core::IGameState>> empty_states;
        auto results = mockInference(empty_states);
        // std::cout << "[DEBUG] Mock inference returned " << results.size() << " results" << std::endl;
        
        // Test settings creation
        // std::cout << "[DEBUG] Creating MCTSSettings" << std::endl;
        alphazero::mcts::MCTSSettings settings;
        settings.num_simulations = 1;
        settings.num_threads = 0;
        settings.use_transposition_table = false;
        // std::cout << "[DEBUG] Settings created successfully" << std::endl;
        
        // std::cout << "[DEBUG] Test completed successfully - no segfault here" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}