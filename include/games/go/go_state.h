// include/alphazero/games/go/go_state.h
#ifndef GO_STATE_H
#define GO_STATE_H

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <optional>
#include <atomic>
#include "core/igamestate.h"
#include "utils/zobrist_hash.h"
#include "games/go/go_rules.h"
#include "core/export_macros.h"

namespace alphazero {
namespace games {
namespace go {

// Define move history structure to support undo
struct ALPHAZERO_API MoveRecord {
    int action;                            // The action that was taken
    int ko_point;                          // Ko point before this move
    std::vector<int> captured_positions;   // Positions of stones captured by this move
    int consecutive_passes;                // Consecutive passes before this move
};

/**
 * @brief Implementation of Go game state
 */
class ALPHAZERO_API GoState : public core::IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param board_size Board size (9, 13, or 19)
     * @param komi Komi value
     * @param chinese_rules Whether to use Chinese rules
     * @param enforce_superko Whether to enforce positional superko (true for Chinese, false for Japanese)
     */
    GoState(int board_size = 19, float komi = 7.5f, bool chinese_rules = true, bool enforce_superko = true);
    
    /**
     * @brief Copy constructor
     */
    GoState(const GoState& other);
    
    /**
     * @brief Destructor - returns cached tensors to pool
     */
    ~GoState();
    
    /**
     * @brief Assignment operator
     */
    GoState& operator=(const GoState& other);
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    core::GameResult getGameResult() const override;
    int getCurrentPlayer() const override;
    int getBoardSize() const override;
    int getActionSpaceSize() const override;
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<core::IGameState> clone() const override;
    void copyFrom(const core::IGameState& source) override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const core::IGameState& other) const override;
    std::vector<int> getMoveHistory() const override;
    bool validate() const override;
    
    // Go-specific methods
    /**
     * @brief Get stone at a position
     * 
     * @param pos Position index
     * @return 0 for empty, 1 for black, 2 for white
     */
    int getStone(int pos) const;
    
    /**
     * @brief Get stone at coordinates
     * 
     * @param x Row
     * @param y Column
     * @return 0 for empty, 1 for black, 2 for white
     */
    int getStone(int x, int y) const;
    
    /**
     * @brief Place a stone at a position
     * 
     * @param pos Position index
     * @param stone Stone value (1 for black, 2 for white)
     */
    void setStone(int pos, int stone);
    
    /**
     * @brief Place a stone at coordinates
     * 
     * @param x Row
     * @param y Column
     * @param stone Stone value (1 for black, 2 for white)
     */
    void setStone(int x, int y, int stone);
    
    /**
     * @brief Get captured stones count
     * 
     * @param player Player (1 for black, 2 for white)
     * @return Number of stones captured by the player
     */
    int getCapturedStones(int player) const;
    
    /**
     * @brief Get komi value
     * 
     * @return Komi value
     */
    float getKomi() const;
    
    /**
     * @brief Check if using Chinese rules
     * 
     * @return true if using Chinese rules, false if Japanese
     */
    bool isChineseRules() const;
    
    /**
     * @brief Check if enforcing superko rule
     * 
     * @return true if enforcing superko, false otherwise
     */
    bool isEnforcingSuperko() const;
    
    /**
     * @brief Convert action to coordinates
     * 
     * @param action Action index
     * @return Pair of (x, y) coordinates
     */
    std::pair<int, int> actionToCoord(int action) const;
    
    /**
     * @brief Convert coordinates to action
     * 
     * @param x Row
     * @param y Column
     * @return Action index
     */
    int coordToAction(int x, int y) const;
    
    /**
     * @brief Get current ko point
     * 
     * @return Ko point index, or -1 if none
     */
    int getKoPoint() const;
    
    /**
     * @brief Get territory ownership
     * 
     * @param dead_stones Set of positions containing dead stones to remove before scoring
     * @return Vector of territory ownership (0 for neutral, 1 for black, 2 for white)
     */
    std::vector<int> getTerritoryOwnership(const std::unordered_set<int>& dead_stones = {}) const;
    
    /**
     * @brief Check if a point is inside territory
     * 
     * @param pos Position index
     * @param player Player (1 for black, 2 for white)
     * @param dead_stones Set of positions containing dead stones
     * @return true if position is inside player's territory
     */
    bool isInsideTerritory(int pos, int player, const std::unordered_set<int>& dead_stones = {}) const;
    
    /**
     * @brief Mark stones as dead for scoring
     * 
     * @param positions Set of positions to mark as dead
     */
    void markDeadStones(const std::unordered_set<int>& positions);
    
    /**
     * @brief Get the set of currently marked dead stones
     * 
     * @return Set of positions of dead stones
     */
    const std::unordered_set<int>& getDeadStones() const;
    
    /**
     * @brief Clear the set of marked dead stones
     */
    void clearDeadStones();
    
    /**
     * @brief Calculate final score with current dead stones
     * 
     * @return Pair of (black_score, white_score)
     */
    std::pair<float, float> calculateScore() const;
    
private:
    int board_size_;
    int current_player_;
    std::vector<int> board_;
    float komi_;
    bool chinese_rules_;
    
    // Game state tracking
    int ko_point_;
    std::vector<int> captured_stones_;
    int consecutive_passes_;
    std::vector<int> move_history_;
    std::vector<uint64_t> position_history_;
    std::vector<MoveRecord> full_move_history_; // Detailed history for undo
    
    // Dead stones for scoring
    std::unordered_set<int> dead_stones_;
    
    // Zobrist hashing
    core::ZobristHash zobrist_;
    mutable uint64_t hash_;
    mutable bool hash_dirty_;
    
    // PERFORMANCE FIX: Cached tensor representations to avoid expensive recomputation
    mutable std::vector<std::vector<std::vector<float>>> cached_tensor_repr_;
    mutable std::vector<std::vector<std::vector<float>>> cached_enhanced_tensor_repr_;
    mutable std::atomic<bool> tensor_cache_dirty_{true};
    mutable std::atomic<bool> enhanced_tensor_cache_dirty_{true};
    
    // PERFORMANCE FIX: Cache expensive group analysis results
    mutable std::vector<StoneGroup> cached_groups_[2]; // [0] = black groups, [1] = white groups
    mutable std::atomic<bool> groups_cache_dirty_{true};
    
    // Rules
    std::shared_ptr<GoRules> rules_;
    
    // Helper methods
    std::vector<int> getAdjacentPositions(int pos) const;
    bool isInBounds(int x, int y) const;
    bool isInBounds(int pos) const;
    void invalidateHash();
    void captureGroup(const StoneGroup& group);
    void captureStones(const std::unordered_set<int>& positions);
    void clearTensorCache() const;
    
    // Check if a move is valid
    bool isValidMove(int action) const;
    bool checkForSuperko(uint64_t new_hash) const;
    
    // Hash calculation
    void updateHash() const;
};

} // namespace go
} // namespace games
} // namespace alphazero

#endif // GO_STATE_H