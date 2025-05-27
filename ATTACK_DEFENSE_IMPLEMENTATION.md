# Attack/Defense Planes Implementation Summary

## Overview
Successfully implemented attack and defense feature planes for the MCTS neural network, extending the input channels from 17 to 19 for Gomoku, Chess, and Go games.

## Changes Made

### 1. Refactored Attack/Defense Module (`include/utils/attack_defense_module.h`, `src/utils/attack_defense_module.cpp`)
- Created base class `AttackDefenseModule` with pure virtual methods
- Implemented game-specific subclasses:
  - `GomokuAttackDefenseModule`: Counts open-three and open-four patterns
  - `ChessAttackDefenseModule`: Calculates piece-value weighted attack/defense scores
  - `GoAttackDefenseModule`: Evaluates captures, atari, liberties, and eye potential
- Added factory function `createAttackDefenseModule()` for game-type based instantiation
- Added `compute_planes()` method to generate attack/defense planes for neural network input

### 2. Updated Game State Representations
- **GomokuState** (`src/games/gomoku/gomoku_state.cpp`):
  - Modified `getEnhancedTensorRepresentation()` to return 19 channels
  - Channels 0-15: History pairs (8 pairs × 2 players)
  - Channel 16: Color plane
  - Channel 17: Attack plane
  - Channel 18: Defense plane

### 3. Neural Network Integration
- **GPU Optimizer** (`src/nn/gpu_optimizer.cpp`):
  - Updated `stateToTensor()` to use `getEnhancedTensorRepresentation()` when channels > 3
  - Added support for Chess in addition to Gomoku and Go

- **Enhanced MCTS Engine** (`src/mcts/enhanced_mcts_engine.cpp`):
  - Modified to use enhanced representation based on channel count
  - Automatically detects and uses appropriate tensor representation

- **Neural Network Factory** (`src/nn/neural_network_factory.cpp`):
  - Updated to handle 19 channels for Gomoku models
  - Added backward compatibility for 17-channel models

### 4. Configuration Updates
- All YAML configuration files updated to use `input_channels: 19`
- Configurations affected:
  - `config_ddw_balanced.yaml`
  - `config_ddw_randwire_optimized.yaml`
  - `config_ddw_randwire_aggressive.yaml`
  - `config_ddw_test.yaml`
  - `config_optimized_memory.yaml`
  - `config_resnet_optimized.yaml`
  - `config_test_signal.yaml`

## Implementation Details

### Gomoku Attack/Defense Calculation
- **Attack Score**: Increase in threat count (open-threes, open-fours) after move
- **Defense Score**: Decrease in opponent threat count after move
- Uses pattern matching with sliding windows for horizontal, vertical, and diagonal threats

### Chess Attack/Defense Calculation (Simplified)
- **Attack Score**: Sum of enemy piece values newly attacked
- **Defense Score**: Sum of friendly piece values newly defended
- Piece values: Pawn=1, Knight/Bishop=3, Rook=5, Queen=9, King=100
- Note: Current implementation is simplified; full chess move generation needed for production

### Go Attack/Defense Calculation
- **Attack Score**: Weighted sum of:
  - Capture value (stones captured)
  - Atari creation (putting groups in atari)
  - Liberty pressure
  - Eye destruction potential
- **Defense Score**: Weighted sum of:
  - Capture prevention
  - Atari escape
  - Liberty gain
  - Eye creation potential

## Testing
Created `test_attack_defense.cpp` to verify:
1. Correct number of channels (19)
2. Attack/defense plane computation
3. Integration with all three game types

## Usage
The system automatically uses 19 channels when:
- Config file specifies `input_channels: 19`
- Neural network expects more than 3 channels
- Game states call `getEnhancedTensorRepresentation()`

## Benefits
1. **Improved MCTS Performance**: Attack/defense information helps guide tree search
2. **Faster Learning**: Neural network receives explicit tactical information
3. **Better Move Selection**: Direct encoding of tactical threats and defenses
4. **Game-Specific Optimization**: Each game type has tailored attack/defense metrics

## Future Improvements
1. Implement full chess move generation for accurate attack/defense calculation
2. Enhance Go implementation with more sophisticated life/death analysis
3. Add normalization to ensure consistent value ranges across games
4. Consider additional tactical features (pins, forks, ko threats, etc.)
5. Optimize computation for real-time performance