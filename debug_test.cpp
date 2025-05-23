#include <iostream>
#include <memory>
#include "mcts/mcts_node.h"

// Simple mock game state for debugging
class DebugMockGameState : public alphazero::core::IGameState {
public:
    DebugMockGameState() : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN), terminal_(false) {
        std::cout << "DebugMockGameState constructor called" << std::endl;
    }
    
    DebugMockGameState(const DebugMockGameState& other)
        : alphazero::core::IGameState(alphazero::core::GameType::UNKNOWN),
          terminal_(other.terminal_) {
        std::cout << "DebugMockGameState copy constructor called" << std::endl;
    }
    
    ~DebugMockGameState() {
        std::cout << "DebugMockGameState destructor called" << std::endl;
    }
    
    void setTerminal(bool terminal) { terminal_ = terminal; }
    
    std::vector<int> getLegalMoves() const override { 
        std::cout << "getLegalMoves() called, returning {0, 1, 2}" << std::endl;
        return {0, 1, 2}; 
    }
    
    bool isLegalMove(int action) const override { return action >= 0 && action <= 2; }
    
    void makeMove(int action) override {
        std::cout << "makeMove(" << action << ") called" << std::endl;
    }
    
    bool undoMove() override { return false; }
    
    bool isTerminal() const override { 
        std::cout << "isTerminal() called, returning " << terminal_ << std::endl;
        return terminal_; 
    }
    
    alphazero::core::GameResult getGameResult() const override { return alphazero::core::GameResult::DRAW; }
    int getCurrentPlayer() const override { return 1; }
    int getBoardSize() const override { return 3; }
    int getActionSpaceSize() const override { return 9; }
    
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override {
        return std::vector<std::vector<std::vector<float>>>
               (2, std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));
    }
    
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return getTensorRepresentation();
    }
    
    uint64_t getHash() const override { return 0; }
    
    std::unique_ptr<IGameState> clone() const override {
        std::cout << "clone() called" << std::endl;
        return std::make_unique<DebugMockGameState>(*this);
    }
    
    std::string actionToString(int action) const override { return std::to_string(action); }
    std::optional<int> stringToAction(const std::string& moveStr) const override {
        try { return std::stoi(moveStr); } catch (...) { return std::nullopt; }
    }
    std::string toString() const override { return "DebugMockGameState"; }
    bool equals(const IGameState& other) const override { return false; }
    std::vector<int> getMoveHistory() const override { return {}; }
    bool validate() const override { return true; }
    
    void copyFrom(const IGameState& source) override {
        const DebugMockGameState* debug_source = dynamic_cast<const DebugMockGameState*>(&source);
        if (!debug_source) {
            throw std::invalid_argument("Cannot copy from non-DebugMockGameState");
        }
        terminal_ = debug_source->terminal_;
    }
    
private:
    bool terminal_;
};

int main() {
    std::cout << "=== Starting debug test ===" << std::endl;
    
    try {
        // Create the game state
        std::cout << "1. Creating game state..." << std::endl;
        auto game_state = std::make_unique<DebugMockGameState>();
        game_state->setTerminal(false);
        
        // Create the node
        std::cout << "2. Creating node..." << std::endl;
        auto node = alphazero::mcts::MCTSNode::create(std::move(game_state));
        
        // Expand the node
        std::cout << "3. Expanding node..." << std::endl;
        node->expand();
        std::cout << "3a. Node expanded successfully" << std::endl;
        
        // Check children
        std::cout << "4. Checking children..." << std::endl;
        auto& children = node->getChildren();
        std::cout << "4a. Number of children: " << children.size() << std::endl;
        
        if (children.size() != 3) {
            std::cout << "ERROR: Expected 3 children, got " << children.size() << std::endl;
            return 1;
        }
        
        // Set prior probabilities
        std::cout << "5. Setting prior probabilities..." << std::endl;
        children[0]->setPriorProbability(0.7f);
        children[1]->setPriorProbability(0.2f);
        children[2]->setPriorProbability(0.1f);
        std::cout << "5a. Prior probabilities set" << std::endl;
        
        // Test selectChild
        std::cout << "6. Testing selectChild..." << std::endl;
        auto selected = node->selectChild(1.0f);
        std::cout << "6a. selectChild returned" << std::endl;
        
        if (!selected) {
            std::cout << "ERROR: selectChild returned null" << std::endl;
            return 1;
        }
        
        std::cout << "6b. Selected child has visit count: " << selected->getVisitCount() << std::endl;
        
        // Test update
        std::cout << "7. Testing update..." << std::endl;
        selected->update(-1.0f);
        std::cout << "7a. Update completed" << std::endl;
        
        std::cout << "7b. After update, visit count: " << selected->getVisitCount() << std::endl;
        std::cout << "7c. After update, value: " << selected->getValue() << std::endl;
        
        // Test second selectChild
        std::cout << "8. Testing second selectChild..." << std::endl;
        auto second_selected = node->selectChild(1.0f);
        std::cout << "8a. Second selectChild returned" << std::endl;
        
        if (!second_selected) {
            std::cout << "ERROR: Second selectChild returned null" << std::endl;
            return 1;
        }
        
        std::cout << "=== Test completed successfully ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "ERROR: Unknown exception caught" << std::endl;
        return 1;
    }
}