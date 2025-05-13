// src/games/go/go_state.cpp
#include "games/go/go_state.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace alphazero {
namespace games {
namespace go {

// Constructor
GoState::GoState(int board_size, float komi, bool chinese_rules, bool enforce_superko)
    : IGameState(core::GameType::GO),
      board_size_(board_size),
      current_player_(1),  // Black goes first
      komi_(komi),
      chinese_rules_(chinese_rules),
      ko_point_(-1),
      consecutive_passes_(0),
      hash_dirty_(true),
      zobrist_(board_size, 2, 2)  // board_size, 2 piece types (black and white), 2 players
{
    // Validate board size
    if (board_size != 9 && board_size != 13 && board_size != 19) {
        board_size_ = 19;  // Default to standard 19x19 if invalid
    }

    // Initialize board with empty intersections
    board_.resize(board_size_ * board_size_, 0);
    
    // Initialize capture counts
    captured_stones_.resize(3, 0);  // Index 0 unused, 1=Black, 2=White
    
    // Initialize Zobrist hash
    hash_ = 0;
    hash_dirty_ = true;
    
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, enforce_superko);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );

    // Add named features for hash calculation
    zobrist_.addFeature("ko_point", board_size_ * board_size_ + 1);  // All positions + none
    zobrist_.addFeature("rules", 2);          // Chinese or Japanese rules
    zobrist_.addFeature("komi", 16);          // Discretized komi values
}

// Copy constructor
GoState::GoState(const GoState& other)
    : IGameState(core::GameType::GO),
      board_size_(other.board_size_),
      current_player_(other.current_player_),
      board_(other.board_),
      komi_(other.komi_),
      chinese_rules_(other.chinese_rules_),
      ko_point_(other.ko_point_),
      captured_stones_(other.captured_stones_),
      consecutive_passes_(other.consecutive_passes_),
      move_history_(other.move_history_),
      position_history_(other.position_history_),
      full_move_history_(other.full_move_history_),
      dead_stones_(other.dead_stones_),
      zobrist_(other.zobrist_),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_)
{
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, other.rules_->isSuperkoenforced());
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );
    
    // Ensure a fresh cache for this copy
    rules_->invalidateCache();
}

// Assignment operator
GoState& GoState::operator=(const GoState& other) {
    if (this != &other) {
        board_size_ = other.board_size_;
        current_player_ = other.current_player_;
        board_ = other.board_;
        komi_ = other.komi_;
        chinese_rules_ = other.chinese_rules_;
        ko_point_ = other.ko_point_;
        captured_stones_ = other.captured_stones_;
        consecutive_passes_ = other.consecutive_passes_;
        move_history_ = other.move_history_;
        position_history_ = other.position_history_;
        full_move_history_ = other.full_move_history_;
        dead_stones_ = other.dead_stones_;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
        
        // Reinitialize rules
        rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, other.rules_->isSuperkoenforced());
        
        // Set up board accessor functions for rules
        rules_->setBoardAccessor(
            [this](int pos) { return this->getStone(pos); },
            [this](int pos) { return this->isInBounds(pos); },
            [this](int pos) { return this->getAdjacentPositions(pos); }
        );
        
        // Ensure a fresh cache for this copy
        rules_->invalidateCache();
    }
    return *this;
}

// IGameState interface implementation
std::vector<int> GoState::getLegalMoves() const {
    std::vector<int> legalMoves;
    
    // Add pass move (-1)
    legalMoves.push_back(-1);
    
    // Check all board positions
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isValidMove(pos)) {
            // If we're enforcing superko, check that too
            if (rules_->isSuperkoenforced()) {
                // Create a temporary copy to test for superko
                GoState tempState(*this);
                
                // Apply the move without updating history
                tempState.setStone(pos, tempState.current_player_);
                
                // Process any captures
                std::vector<StoneGroup> opponentGroups = tempState.rules_->findGroups(3 - tempState.current_player_);
                for (const auto& group : opponentGroups) {
                    if (group.liberties.empty()) {
                        tempState.captureGroup(group);
                    }
                }
                
                // Check for superko
                uint64_t newHash = tempState.getHash();
                if (!checkForSuperko(newHash)) {
                    legalMoves.push_back(pos);
                }
            } else {
                // If superko is not enforced, add all valid moves
                legalMoves.push_back(pos);
            }
        }
    }
    
    return legalMoves;
}

void GoState::makeMove(int action) {
    if (!isLegalMove(action)) {
        throw std::runtime_error("Illegal move attempted");
    }
    
    // Create a record for this move
    MoveRecord record;
    record.action = action;
    record.ko_point = ko_point_;
    record.consecutive_passes = consecutive_passes_;
    
    // Handle pass
    if (action == -1) {
        consecutive_passes_++;
        ko_point_ = -1;  // Clear ko point on pass
        
        // Record move
        move_history_.push_back(action);
        full_move_history_.push_back(record);
    } else {
        // Reset consecutive passes
        consecutive_passes_ = 0;
        
        // CRITICAL FIX: Always clear ko point for any non-pass move
        ko_point_ = -1;
        
        // Place stone
        setStone(action, current_player_);
        
        // Explicitly invalidate cache before finding groups for capture processing
        rules_->invalidateCache(); 

        // Check for captures
        std::vector<StoneGroup> opponentGroups = rules_->findGroups(3 - current_player_);
        std::vector<StoneGroup> capturedGroups;
        int capturedStones = 0;
        
        for (const auto& group : opponentGroups) {
            if (group.liberties.empty()) {
                capturedGroups.push_back(group);
                capturedStones += group.stones.size();
                // Save captured positions for undo
                for (int pos : group.stones) {
                    record.captured_positions.push_back(pos);
                }
            }
        }
        
        // Process captures
        for (const auto& group : capturedGroups) {
            captureGroup(group);
        }
        
        // ONLY set a ko point if exactly one stone was captured
        if (capturedGroups.size() == 1 && capturedGroups[0].stones.size() == 1) {
            ko_point_ = *capturedGroups[0].stones.begin();
        }
        
        // Update capture count
        captured_stones_[current_player_] += capturedStones;
        
        // Record move
        move_history_.push_back(action);
        full_move_history_.push_back(record);
        
        // Record position for ko/superko detection
        position_history_.push_back(getHash());
    }
    
    // Switch players
    current_player_ = 3 - current_player_;
    
    // Invalidate hash after all state changes
    invalidateHash();
}

bool GoState::isLegalMove(int action) const {
    if (action == -1) {
        return true;  // Pass is always legal
    }
    
    // First check basic validity
    if (!isValidMove(action)) {
        return false;
    }
    
    // Check basic ko rule
    if (action == ko_point_) {
        return false;  // Ko violation
    }
    
    // If not enforcing superko, we're done
    if (!rules_->isSuperkoenforced()) {
        return true;
    }
    
    // Create a temporary copy to test for superko
    GoState tempState(*this);
    
    // CRITICAL FIX: Clear ko point first, just like in makeMove
    tempState.ko_point_ = -1;
    
    // Apply the move to tempState
    tempState.setStone(action, tempState.current_player_);
    
    // Process captures exactly as makeMove does
    std::vector<StoneGroup> opponentGroups = tempState.rules_->findGroups(3 - tempState.current_player_);
    std::vector<StoneGroup> capturedGroups;
    
    for (const auto& group : opponentGroups) {
        if (group.liberties.empty()) {
            capturedGroups.push_back(group);
            for (int pos : group.stones) {
                tempState.setStone(pos, 0);
            }
        }
    }
    
    // ONLY set a ko point if exactly one stone was captured
    if (capturedGroups.size() == 1 && capturedGroups[0].stones.size() == 1) {
        tempState.ko_point_ = *capturedGroups[0].stones.begin();
    }
    
    // CRITICAL FIX: Calculate hash BEFORE switching players (matching makeMove's timing)
    tempState.invalidateHash();
    uint64_t newHash = tempState.getHash();
    
    // Check for superko violation
    for (uint64_t hash : position_history_) {
        if (hash == newHash) {
            return false;  // Superko violation
        }
    }
    
    return true;
}

bool GoState::undoMove() {
    if (full_move_history_.empty()) {
        return false;
    }
    
    // Get last move record
    MoveRecord lastMove = full_move_history_.back();
    full_move_history_.pop_back();
    
    // Remove from move history
    if (!move_history_.empty()) {
        move_history_.pop_back();
    }
    
    // Remove last position from history
    if (!position_history_.empty()) {
        position_history_.pop_back();
    }
    
    // Switch back to previous player
    current_player_ = 3 - current_player_;
    
    // Restore ko point
    ko_point_ = lastMove.ko_point;
    
    // Restore consecutive passes
    consecutive_passes_ = lastMove.consecutive_passes;
    
    // If it was a pass, we're done
    if (lastMove.action == -1) {
        // Invalidate hash
        invalidateHash();
        return true;
    }
    
    // Remove the stone
    setStone(lastMove.action, 0);
    
    // Restore captured stones
    for (int pos : lastMove.captured_positions) {
        setStone(pos, 3 - current_player_);  // Opponent's color
    }
    
    // Update captured stones count
    captured_stones_[current_player_] -= lastMove.captured_positions.size();
    
    // Invalidate hash
    invalidateHash();
    
    // Invalidate rules cache
    rules_->invalidateCache();
    
    return true;
}

bool GoState::isTerminal() const {
    // Game ends when both players pass consecutively
    return consecutive_passes_ >= 2;
}

core::GameResult GoState::getGameResult() const {
    if (!isTerminal()) {
        return core::GameResult::ONGOING;
    }
    
    // Calculate scores
    auto [blackScore, whiteScore] = calculateScore();
    
    if (blackScore > whiteScore) {
        return core::GameResult::WIN_PLAYER1;  // Black wins
    } else if (whiteScore > blackScore) {
        return core::GameResult::WIN_PLAYER2;  // White wins
    } else {
        return core::GameResult::DRAW;  // Draw
    }
}

int GoState::getCurrentPlayer() const {
    return current_player_;
}

int GoState::getBoardSize() const {
    return board_size_;
}

int GoState::getActionSpaceSize() const {
    return board_size_ * board_size_ + 1;  // +1 for pass
}

std::vector<std::vector<std::vector<float>>> GoState::getTensorRepresentation() const {
    // Basic 3-plane representation (black, white, turn)
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size_, 
            std::vector<float>(board_size_, 0.0f)));
    
    // Fill first two planes with stone positions
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            
            if (stone == 1) {
                tensor[0][y][x] = 1.0f;  // Black stones
            } else if (stone == 2) {
                tensor[1][y][x] = 1.0f;  // White stones
            }
        }
    }
    
    // Third plane: current player
    float playerValue = (current_player_ == 1) ? 1.0f : 0.0f;
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            tensor[2][y][x] = playerValue;
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GoState::getEnhancedTensorRepresentation() const {
    // Start with the basic representation
    std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
    
    // Add additional planes for enhanced features
    
    // 4: Liberties of black groups (normalized)
    // 5: Liberties of white groups (normalized)
    std::vector<std::vector<float>> blackLiberties(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> whiteLiberties(board_size_, std::vector<float>(board_size_, 0.0f));
    
    // Get groups from cache (will be calculated if needed)
    auto blackGroups = rules_->findGroups(1);
    auto whiteGroups = rules_->findGroups(2);
    
    for (const auto& group : blackGroups) {
        float libertyCount = static_cast<float>(group.liberties.size());
        float normalizedLiberties = std::min(1.0f, libertyCount / 10.0f);  // Normalize to [0,1]
        
        for (int pos : group.stones) {
            int y = pos / board_size_;
            int x = pos % board_size_;
            blackLiberties[y][x] = normalizedLiberties;
        }
    }
    
    for (const auto& group : whiteGroups) {
        float libertyCount = static_cast<float>(group.liberties.size());
        float normalizedLiberties = std::min(1.0f, libertyCount / 10.0f);  // Normalize to [0,1]
        
        for (int pos : group.stones) {
            int y = pos / board_size_;
            int x = pos % board_size_;
            whiteLiberties[y][x] = normalizedLiberties;
        }
    }
    
    tensor.push_back(blackLiberties);
    tensor.push_back(whiteLiberties);
    
    // 6: Ko point
    std::vector<std::vector<float>> koPlane(board_size_, std::vector<float>(board_size_, 0.0f));
    if (ko_point_ >= 0) {
        int y = ko_point_ / board_size_;
        int x = ko_point_ % board_size_;
        koPlane[y][x] = 1.0f;
    }
    tensor.push_back(koPlane);
    
    // 7-8: Distance transforms from borders
    std::vector<std::vector<float>> distanceX(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> distanceY(board_size_, std::vector<float>(board_size_, 0.0f));
    
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            // Normalize distances to [0,1]
            distanceX[y][x] = static_cast<float>(std::min(x, board_size_ - 1 - x)) / (board_size_ / 2);
            distanceY[y][x] = static_cast<float>(std::min(y, board_size_ - 1 - y)) / (board_size_ / 2);
        }
    }
    
    tensor.push_back(distanceX);
    tensor.push_back(distanceY);
    
    return tensor;
}

uint64_t GoState::getHash() const {
    if (hash_dirty_) {
        updateHash();
    }
    return hash_;
}

std::unique_ptr<core::IGameState> GoState::clone() const {
    return std::make_unique<GoState>(*this);
}

std::string GoState::actionToString(int action) const {
    if (action == -1) {
        return "pass";
    }
    
    if (action < 0 || action >= board_size_ * board_size_) {
        return "invalid";
    }
    
    std::pair<int, int> coords = actionToCoord(action);
    int x = coords.first;
    int y = coords.second;
    
    // Convert to Go coordinates (A-T, skipping I, 1-19)
    char colChar = 'A' + x;
    if (colChar >= 'I') {
        colChar++;  // Skip 'I'
    }
    
    return std::string(1, colChar) + std::to_string(board_size_ - y);
}

std::optional<int> GoState::stringToAction(const std::string& moveStr) const {
    if (moveStr == "pass" || moveStr == "PASS" || moveStr == "Pass") {
        return -1;
    }
    
    if (moveStr.length() < 2) {
        return std::nullopt;
    }
    
    char colChar = std::toupper(moveStr[0]);
    
    // Skip 'I' as it's not used in Go notation
    if (colChar == 'I') {
        return std::nullopt;
    }
    
    // Adjust for 'I' being skipped
    int x;
    if (colChar >= 'J') {
        x = colChar - 'A' - 1;
    } else {
        x = colChar - 'A';
    }
    
    // Parse row
    int y;
    try {
        y = board_size_ - std::stoi(moveStr.substr(1));
    } catch (...) {
        return std::nullopt;
    }
    
    if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
        return std::nullopt;
    }
    
    return coordToAction(x, y);
}

std::string GoState::toString() const {
    std::stringstream ss;
    
    // Print column headers
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print board
    for (int y = 0; y < board_size_; ++y) {
        ss << std::setw(2) << (board_size_ - y) << " ";
        
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            
            if (stone == 0) {
                // Check if this is a ko point
                if (pos == ko_point_) {
                    ss << "k ";
                } else if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "d ";  // Mark dead stones
                } else {
                    ss << ". ";
                }
            } else if (stone == 1) {
                if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "x ";  // Dead black stone
                } else {
                    ss << "X ";  // Black
                }
            } else if (stone == 2) {
                if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "o ";  // Dead white stone
                } else {
                    ss << "O ";  // White
                }
            }
        }
        
        ss << (board_size_ - y) << std::endl;
    }
    
    // Print column headers again
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print game info
    ss << "Current player: " << (current_player_ == 1 ? "Black" : "White") << std::endl;
    ss << "Captures - Black: " << captured_stones_[1] << ", White: " << captured_stones_[2] << std::endl;
    ss << "Komi: " << komi_ << std::endl;
    ss << "Rules: " << (chinese_rules_ ? "Chinese" : "Japanese") << std::endl;
    ss << "Superko enforcement: " << (rules_->isSuperkoenforced() ? "Yes" : "No") << std::endl;
    
    if (isTerminal()) {
        auto [blackScore, whiteScore] = calculateScore();
        
        ss << "Game over!" << std::endl;
        ss << "Final score - Black: " << blackScore << ", White: " << whiteScore 
           << " (with komi " << komi_ << ")" << std::endl;
        
        if (blackScore > whiteScore) {
            ss << "Black wins by " << (blackScore - whiteScore) << " points" << std::endl;
        } else if (whiteScore > blackScore) {
            ss << "White wins by " << (whiteScore - blackScore) << " points" << std::endl;
        } else {
            ss << "Game ended in a draw" << std::endl;
        }
    }
    
    return ss.str();
}

bool GoState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::GO) {
        return false;
    }
    
    try {
        const GoState& otherGo = dynamic_cast<const GoState&>(other);
        
        if (board_size_ != otherGo.board_size_ || 
            current_player_ != otherGo.current_player_ ||
            ko_point_ != otherGo.ko_point_ ||
            komi_ != otherGo.komi_ ||
            chinese_rules_ != otherGo.chinese_rules_ ||
            consecutive_passes_ != otherGo.consecutive_passes_ ||
            captured_stones_ != otherGo.captured_stones_ ||
            dead_stones_ != otherGo.dead_stones_) {
            return false;
        }
        
        // Compare board positions
        return board_ == otherGo.board_;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> GoState::getMoveHistory() const {
    return move_history_;
}

bool GoState::validate() const {
    // Check board size
    if (board_size_ != 9 && board_size_ != 13 && board_size_ != 19) {
        return false;
    }
    
    // Check current player
    if (current_player_ != 1 && current_player_ != 2) {
        return false;
    }
    
    // Check ko point
    if (ko_point_ >= board_size_ * board_size_) {
        return false;
    }
    
    // Check captured stones
    if (captured_stones_.size() != 3) {
        return false;
    }
    
    // Count stones of each color on the board
    int black_count = 0;
    int white_count = 0;
    
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isInBounds(pos)) {
            int stone = getStone(pos);
            if (stone == 1) black_count++;
            else if (stone == 2) white_count++;
        }
    }
    
    // Check if stone counts make sense in relation to captures
    // Black goes first, so if it's black's turn, white should have placed equal stones
    // If it's white's turn, black should have one more stone
    if ((current_player_ == 1 && black_count != white_count + captured_stones_[1] - captured_stones_[2]) ||
        (current_player_ == 2 && black_count != white_count + 1 + captured_stones_[1] - captured_stones_[2])) {
        return false;
    }
    
    return true;
}

// Go-specific methods
int GoState::getStone(int pos) const {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return 0;  // Out of bounds, return empty
    }
    return board_[pos];
}

int GoState::getStone(int x, int y) const {
    if (!isInBounds(x, y)) {
        return 0;  // Out of bounds, return empty
    }
    return board_[y * board_size_ + x];
}

void GoState::setStone(int pos, int stone) {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return;  // Out of bounds, do nothing
    }
    board_[pos] = stone;
    invalidateHash();
    rules_->invalidateCache();
}

void GoState::setStone(int x, int y, int stone) {
    if (!isInBounds(x, y)) {
        return;  // Out of bounds, do nothing
    }
    board_[y * board_size_ + x] = stone;
    invalidateHash();
    rules_->invalidateCache();
}

int GoState::getCapturedStones(int player) const {
    if (player != 1 && player != 2) {
        return 0;
    }
    return captured_stones_[player];
}

float GoState::getKomi() const {
    return komi_;
}

bool GoState::isChineseRules() const {
    return chinese_rules_;
}

bool GoState::isEnforcingSuperko() const {
    return rules_->isSuperkoenforced();
}

std::pair<int, int> GoState::actionToCoord(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return {-1, -1};
    }
    
    int y = action / board_size_;
    int x = action % board_size_;
    
    return {x, y};
}

int GoState::coordToAction(int x, int y) const {
    if (!isInBounds(x, y)) {
        return -1;
    }
    
    return y * board_size_ + x;
}

int GoState::getKoPoint() const {
    return ko_point_;
}

std::vector<int> GoState::getTerritoryOwnership(const std::unordered_set<int>& dead_stones) const {
    // Combine local dead stones with any provided
    std::unordered_set<int> all_dead_stones = dead_stones;
    all_dead_stones.insert(dead_stones_.begin(), dead_stones_.end());
    
    return rules_->getTerritoryOwnership(all_dead_stones);
}

bool GoState::isInsideTerritory(int pos, int player, const std::unordered_set<int>& dead_stones) const {
    // Combine local dead stones with any provided
    std::unordered_set<int> all_dead_stones = dead_stones;
    all_dead_stones.insert(dead_stones_.begin(), dead_stones_.end());
    
    std::vector<int> territory = rules_->getTerritoryOwnership(all_dead_stones);
    if (pos < 0 || pos >= static_cast<int>(territory.size())) {
        return false;
    }
    return territory[pos] == player;
}

void GoState::markDeadStones(const std::unordered_set<int>& positions) {
    dead_stones_ = positions;
    invalidateHash();
    rules_->invalidateCache();
}

const std::unordered_set<int>& GoState::getDeadStones() const {
    return dead_stones_;
}

void GoState::clearDeadStones() {
    dead_stones_.clear();
    invalidateHash();
    rules_->invalidateCache();
}

std::pair<float, float> GoState::calculateScore() const {
    return rules_->calculateScores(captured_stones_, komi_, dead_stones_);
}

// Helper methods
std::vector<int> GoState::getAdjacentPositions(int pos) const {
    std::vector<int> adjacentPositions;
    int x, y;
    std::tie(x, y) = actionToCoord(pos);
    
    // Check orthogonally adjacent positions
    for (const auto& direction : std::vector<std::pair<int, int>>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
        int newX = x + direction.first;
        int newY = y + direction.second;
        
        if (isInBounds(newX, newY)) {
            adjacentPositions.push_back(coordToAction(newX, newY));
        }
    }
    
    return adjacentPositions;
}

bool GoState::isInBounds(int x, int y) const {
    return x >= 0 && x < board_size_ && y >= 0 && y < board_size_;
}

bool GoState::isInBounds(int pos) const {
    return pos >= 0 && pos < board_size_ * board_size_;
}

void GoState::invalidateHash() {
    hash_dirty_ = true;
}

void GoState::captureGroup(const StoneGroup& group) {
    // Remove all stones in the group
    for (int pos : group.stones) {
        setStone(pos, 0);  // This should be correctly removing stones
    }
    
    // Explicitly invalidate the rules cache after removing stones
    rules_->invalidateCache();
}

void GoState::captureStones(const std::unordered_set<int>& positions) {
    for (int pos : positions) {
        setStone(pos, 0);
    }
    // Note: setStone already invalidates the cache
}

bool GoState::isValidMove(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return false;
    }
    
    // Check if the intersection is empty
    if (getStone(action) != 0) {
        return false;
    }
    
    // Check for suicide rule
    if (rules_->isSuicidalMove(action, current_player_)) {
        return false;
    }
    
    return true;
}

bool GoState::checkForSuperko(uint64_t new_hash) const {
    // Check if this position has appeared before
    for (uint64_t hash : position_history_) {
        if (hash == new_hash) {
            return true;  // Position repetition found
        }
    }
    return false;
}

void GoState::updateHash() const {
    hash_ = 0;
    
    // Hash board position
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!isInBounds(pos)) continue;
        
        int stone = getStone(pos);
        if (stone != 0) {
            int pieceIdx = stone - 1;  // Convert to 0-based index
            hash_ ^= zobrist_.getPieceHash(pieceIdx, pos);
        }
    }
    
    // Hash current player
    hash_ ^= zobrist_.getPlayerHash(current_player_ - 1);
    
    // Hash ko point
    if (ko_point_ >= 0) {
        hash_ ^= zobrist_.getFeatureHash("ko_point", ko_point_);
    }
    
    // Hash the rule variant
    if (chinese_rules_) {
        hash_ ^= zobrist_.getFeatureHash("rules", 1);  // Chinese rules
    } else {
        hash_ ^= zobrist_.getFeatureHash("rules", 0);  // Japanese rules
    }
    
    // Hash komi value (discretized)
    int komi_int = static_cast<int>(komi_ * 2);  // Convert to half-points
    hash_ ^= zobrist_.getFeatureHash("komi", komi_int & 0xF);  // Use lower 4 bits
    
    hash_dirty_ = false;
}

} // namespace go
} // namespace games
} // namespace alphazero