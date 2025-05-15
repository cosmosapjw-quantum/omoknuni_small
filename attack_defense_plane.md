# Detailed Implementation Guide for Attack/Defense Plane in AlphaZero

This guide provides specific implementation details for adding attack and defense planes to the neural network input tensor for Go, Gomoku, and Chess games. All changes will be integrated into the existing attack_defense_module.h/cpp files.

## 1. Changes to attack_defense_module.h

First, extend the header file with these new method declarations:

```
// Add to attack_defense_module.h

class AttackDefenseModule {
public:
    // Existing constructor and methods...
    
    // New method to generate attack/defense planes for neural network input
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    generate_attack_defense_planes(const std::unique_ptr<alphazero::core::IGameState>& state);
    
    // New method to get normalized attack/defense scores for a specific move
    std::pair<float, float> get_move_scores(
        const std::unique_ptr<alphazero::core::IGameState>& state,
        int move);

private:
    // Existing member variables and methods...
    
    // Game-specific implementations
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    generate_go_planes(const std::unique_ptr<alphazero::core::IGameState>& state);
    
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    generate_gomoku_planes(const std::unique_ptr<alphazero::core::IGameState>& state);
    
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    generate_chess_planes(const std::unique_ptr<alphazero::core::IGameState>& state);
    
    // Helper methods for chess implementation
    float get_piece_value(int piece_type) const;
    float calculate_attack_value(const std::vector<std::vector<int>>& board, int row, int col, int player);
    float calculate_defense_value(const std::vector<std::vector<int>>& board, int row, int col, int player);
    bool is_capture_move(const std::vector<std::vector<int>>& board, int from_row, int from_col, 
                         int to_row, int to_col, int player);
};
```

## 2. Changes to attack_defense_module.cpp

Add the main implementation and dispatcher logic:

```
// Add to attack_defense_module.cpp

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
AttackDefenseModule::generate_attack_defense_planes(
    const std::unique_ptr<alphazero::core::IGameState>& state) {
    
    // Dispatch to the appropriate game-specific implementation
    switch (state->getGameType()) {
        case alphazero::core::GameType::GO:
            return generate_go_planes(state);
        
        case alphazero::core::GameType::GOMOKU:
            return generate_gomoku_planes(state);
        
        case alphazero::core::GameType::CHESS:
            return generate_chess_planes(state);
        
        default:
            // Return empty planes for unknown game types
            return {
                std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f)),
                std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f))
            };
    }
}

std::pair<float, float> AttackDefenseModule::get_move_scores(
    const std::unique_ptr<alphazero::core::IGameState>& state,
    int move) {
    
    // Create clone states for before and after
    auto before_state = state->clone();
    auto after_state = state->clone();
    
    // Apply the move to the after state
    after_state->makeMove(move);
    
    // Generate planes for both states
    auto [attack_before, defense_before] = generate_attack_defense_planes(before_state);
    auto [attack_after, defense_after] = generate_attack_defense_planes(after_state);
    
    // Extract the move coordinates
    int row = move / board_size_;
    int col = move % board_size_;
    
    // Calculate the differences
    float attack_diff = attack_after[row][col] - attack_before[row][col];
    float defense_diff = defense_after[row][col] - defense_before[row][col];
    
    return {attack_diff, defense_diff};
}
```

## 3. Game-Specific Implementations

### 3.1. Go Implementation

For Go, focus on liberty counting and threat detection:

```
// Add to attack_defense_module.cpp

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
AttackDefenseModule::generate_go_planes(const std::unique_ptr<alphazero::core::IGameState>& state) {
    // Convert the state to a 2D board representation
    std::vector<std::vector<int>> board(board_size_, std::vector<int>(board_size_, 0));
    
    // Fill board with stone positions
    // This will need to be adapted based on your specific Go state implementation
    for (int row = 0; row < board_size_; row++) {
        for (int col = 0; col < board_size_; col++) {
            int pos = row * board_size_ + col;
            int stone = /* get stone from state at position pos */;
            board[row][col] = stone;
        }
    }
    
    int current_player = state->getCurrentPlayer();
    int opponent = (current_player == 1) ? 2 : 1;
    
    // Initialize attack and defense planes
    std::vector<std::vector<float>> attack_plane(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> defense_plane(board_size_, std::vector<float>(board_size_, 0.0f));
    
    // For each empty position
    for (int row = 0; row < board_size_; row++) {
        for (int col = 0; col < board_size_; col++) {
            if (board[row][col] == 0) {  // Empty position
                // Create temporary board with the move
                auto temp_board = board;
                temp_board[row][col] = current_player;
                
                // Calculate attack value - count opponent stones that would be captured
                int attack_value = count_captures(temp_board, row, col, opponent);
                attack_plane[row][col] = static_cast<float>(attack_value);
                
                // Calculate defense value - count own groups that would get additional liberties
                int defense_value = count_liberty_gains(temp_board, row, col, current_player);
                defense_plane[row][col] = static_cast<float>(defense_value);
            }
        }
    }
    
    // Normalize values to a reasonable range (0-1)
    normalize_plane(attack_plane, 5.0f);  // Assuming max capture value around 5 stones
    normalize_plane(defense_plane, 3.0f);  // Assuming max liberty gain around 3
    
    return {attack_plane, defense_plane};
}

// Helper methods for Go (pseudocode, implement based on your Go rules implementation)
int AttackDefenseModule::count_captures(const std::vector<std::vector<int>>& board, 
                                       int row, int col, int opponent_color) {
    // Count adjacent opponent groups with exactly one liberty (that would be captured)
    // This depends on your specific Go rules implementation
    // ...
}

int AttackDefenseModule::count_liberty_gains(const std::vector<std::vector<int>>& board,
                                           int row, int col, int player_color) {
    // Count friendly groups that would gain liberties by this move
    // This depends on your specific Go rules implementation
    // ...
}

void AttackDefenseModule::normalize_plane(std::vector<std::vector<float>>& plane, float max_value) {
    for (auto& row : plane) {
        for (auto& val : row) {
            val = std::min(1.0f, val / max_value);
        }
    }
}
```

### 3.2. Gomoku Implementation

For Gomoku, focus on detecting threat patterns using the existing code:

```
// Add to attack_defense_module.cpp

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
AttackDefenseModule::generate_gomoku_planes(const std::unique_ptr<alphazero::core::IGameState>& state) {
    // Convert the state to a 2D board representation
    std::vector<std::vector<int>> board(board_size_, std::vector<int>(board_size_, 0));
    
    // Fill board with stone positions
    // This will need to be adapted based on your specific Gomoku state implementation
    for (int row = 0; row < board_size_; row++) {
        for (int col = 0; col < board_size_; col++) {
            int pos = row * board_size_ + col;
            int stone = /* get stone from state at position pos */;
            board[row][col] = stone;
        }
    }
    
    int current_player = state->getCurrentPlayer();
    
    // Initialize attack and defense planes
    std::vector<std::vector<float>> attack_plane(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> defense_plane(board_size_, std::vector<float>(board_size_, 0.0f));
    
    // For each empty position, calculate attack and defense values
    for (int row = 0; row < board_size_; row++) {
        for (int col = 0; col < board_size_; col++) {
            if (board[row][col] == 0) {  // Empty position
                // Create a batch with just this position
                std::vector<std::vector<std::vector<int>>> board_batch = {board};
                std::vector<int> chosen_moves = {row * board_size_ + col};
                std::vector<int> player_batch = {current_player};
                
                // Use existing methods to compute attack and defense bonuses
                std::vector<float> attack_bonuses = compute_attack_bonus(board_batch, chosen_moves, player_batch);
                std::vector<float> defense_bonuses = compute_defense_bonus(board_batch, chosen_moves, player_batch);
                
                // Store values in the planes
                attack_plane[row][col] = attack_bonuses[0];
                defense_plane[row][col] = defense_bonuses[0];
            }
        }
    }
    
    // Normalize values to a reasonable range (0-1)
    normalize_plane(attack_plane, 2.0f);  // Adjust based on your typical attack bonus range
    normalize_plane(defense_plane, 2.0f);  // Adjust based on your typical defense bonus range
    
    return {attack_plane, defense_plane};
}
```

### 3.3. Chess Implementation

For Chess, focus on material values and piece interactions:

```
// Add to attack_defense_module.cpp

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
AttackDefenseModule::generate_chess_planes(const std::unique_ptr<alphazero::core::IGameState>& state) {
    // Initialize 8x8 planes for standard chess board
    std::vector<std::vector<float>> attack_plane(8, std::vector<float>(8, 0.0f));
    std::vector<std::vector<float>> defense_plane(8, std::vector<float>(8, 0.0f));
    
    // Convert to 2D board representation with piece types and colors
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    std::vector<std::vector<int>> piece_colors(8, std::vector<int>(8, 0));
    
    // Fill board with piece information
    // This will need to be adapted based on your specific Chess state implementation
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int square = row * 8 + col;
            // Get piece type (e.g., PAWN=1, KNIGHT=2, etc.) and color (1=white, 2=black)
            int piece_type = /* get piece type from state at square */;
            int piece_color = /* get piece color from state at square */;
            board[row][col] = piece_type;
            piece_colors[row][col] = piece_color;
        }
    }
    
    int current_player = state->getCurrentPlayer();
    
    // Calculate attack and defense values for each square
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            // For empty squares, consider the value if a piece moves there
            if (board[row][col] == 0) {
                // Get all legal moves that end at this square
                std::vector<int> moves_to_square = get_moves_to_square(state, row, col);
                
                float max_attack_value = 0.0f;
                float max_defense_value = 0.0f;
                
                for (int move : moves_to_square) {
                    // Extract move details
                    int from_row = /* get from_row from move */;
                    int from_col = /* get from_col from move */;
                    
                    // Check if this is a capture move
                    if (is_capture_move(board, piece_colors, from_row, from_col, row, col, current_player)) {
                        float capture_value = get_piece_value(board[row][col]);
                        max_attack_value = std::max(max_attack_value, capture_value);
                    }
                    
                    // Calculate defense value - pieces this move would defend
                    float defense_value = calculate_defense_value(board, piece_colors, row, col, current_player);
                    max_defense_value = std::max(max_defense_value, defense_value);
                }
                
                attack_plane[row][col] = max_attack_value;
                defense_plane[row][col] = max_defense_value;
            }
            else {
                // For occupied squares, calculate current attack/defense value
                if (piece_colors[row][col] == current_player) {
                    // Current player's piece - what does it attack/defend?
                    attack_plane[row][col] = calculate_attack_value(board, piece_colors, row, col, current_player);
                    defense_plane[row][col] = calculate_defense_value(board, piece_colors, row, col, current_player);
                }
            }
        }
    }
    
    // Normalize values based on piece values (queen = 9 is highest)
    normalize_plane(attack_plane, 9.0f);
    normalize_plane(defense_plane, 9.0f);
    
    return {attack_plane, defense_plane};
}

// Helper methods for chess implementation
float AttackDefenseModule::get_piece_value(int piece_type) const {
    // Standard chess piece values
    constexpr float piece_values[] = {0.0f, 1.0f, 3.0f, 3.0f, 5.0f, 9.0f, 0.0f};
    if (piece_type >= 0 && piece_type < 7) {
        return piece_values[piece_type];
    }
    return 0.0f;
}

float AttackDefenseModule::calculate_attack_value(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_colors,
    int row, int col, int player) {
    
    // Calculate the total value of opponent pieces this piece attacks
    float attack_value = 0.0f;
    
    // Get all squares this piece attacks
    std::vector<std::pair<int, int>> attacked_squares = get_attacked_squares(board, piece_colors, row, col, player);
    
    for (const auto& [target_row, target_col] : attacked_squares) {
        if (piece_colors[target_row][target_col] != 0 && piece_colors[target_row][target_col] != player) {
            // Opponent piece
            attack_value += get_piece_value(board[target_row][target_col]);
        }
    }
    
    return attack_value;
}

float AttackDefenseModule::calculate_defense_value(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_colors,
    int row, int col, int player) {
    
    // Calculate the total value of friendly pieces this piece defends
    float defense_value = 0.0f;
    
    // Get all squares this piece defends
    std::vector<std::pair<int, int>> defended_squares = get_defended_squares(board, piece_colors, row, col, player);
    
    for (const auto& [target_row, target_col] : defended_squares) {
        if (piece_colors[target_row][target_col] == player) {
            // Friendly piece
            defense_value += get_piece_value(board[target_row][target_col]);
        }
    }
    
    return defense_value;
}

bool AttackDefenseModule::is_capture_move(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_colors,
    int from_row, int from_col, int to_row, int to_col, int player) {
    
    // A move is a capture if the destination has an opponent piece
    return (piece_colors[to_row][to_col] != 0 && piece_colors[to_row][to_col] != player);
}

// These methods would need to be implemented based on your chess rules implementation
std::vector<int> AttackDefenseModule::get_moves_to_square(
    const std::unique_ptr<alphazero::core::IGameState>& state, int row, int col) {
    // Get all legal moves that end at the specified square
    // ...
}

std::vector<std::pair<int, int>> AttackDefenseModule::get_attacked_squares(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_colors,
    int row, int col, int player) {
    // Get all squares attacked by the piece at (row, col)
    // This depends on piece type and chess movement rules
    // ...
}

std::vector<std::pair<int, int>> AttackDefenseModule::get_defended_squares(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_colors,
    int row, int col, int player) {
    // Get all squares defended by the piece at (row, col)
    // ...
}
```

## 4. Claude Prompt for Implementation

Use this prompt to generate the actual implementation code:

```
You are a C++ expert specializing in game AI and the AlphaZero algorithm. I need you to implement attack and defense plane functionality for the neural network input tensor in my AlphaZero project.

The implementation should add two additional planes to the tensor representation:
1. An attack plane showing the attack value for each position
2. A defense plane showing the defense value for each position

The implementation needs to handle three game types:
- Go: Based on stone captures and liberty gains
- Gomoku: Based on threat patterns like open threes and open fours
- Chess: Based on material values of attacked/defended pieces

I want to integrate this functionality into my existing AttackDefenseModule (attack_defense_module.h/cpp) without creating separate files.

Please implement:
1. The complete modifications to attack_defense_module.h
2. The complete modifications to attack_defense_module.cpp including all necessary helper methods

Ensure the implementation follows these design principles:
- Proper game type detection and dispatch to specific implementations
- Efficient code that can handle batch processing
- Normalization of values to ensure consistency across game types
- Proper integration with the existing neural network input pipeline

The existing AttackDefenseModule already has methods for computing attack and defense bonuses for Gomoku, but needs to be extended to generate full planes and handle Chess and Go.

Assume the board representation is standardized: 0 for empty spaces, 1/2 for player stones/pieces.
```

This implementation guide provides a comprehensive approach to adding attack and defense planes to your neural network input tensor across all three game types while keeping all the changes within the existing attack_defense_module.h/cpp files.