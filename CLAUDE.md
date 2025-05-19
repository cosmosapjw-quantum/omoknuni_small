# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Building the Project

### Initial Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
cmake --build . --config Release --parallel
```

### Clean Build
```bash
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
cmake --build . --config Release --parallel
```

### Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
cmake --build . --config Debug --parallel
```

## Testing

### Run All Tests
```bash
cd build
./bin/Release/all_tests
```

### Run Specific Test Suite
```bash
./bin/Release/mcts_tests      # MCTS component tests
./bin/Release/gomoku_tests    # Gomoku game tests
./bin/Release/chess_tests     # Chess game tests
./bin/Release/go_tests        # Go game tests
./bin/Release/core_tests      # Core abstraction tests
```

### Run Single Test
```bash
./bin/Release/mcts_tests --gtest_filter="MCTSNodeTest.UpdateQ"
./bin/Release/mcts_tests --gtest_filter="MCTSEngineTest.*"
```

### Run Tests with Verbose Output
```bash
./bin/Release/all_tests --gtest_print_time=1 --gtest_color=yes
```

## CLI Usage

### Run Self-Play
```bash
./bin/Release/omoknuni_cli self-play --config ../config.yaml
```

### Train Model
```bash
./bin/Release/omoknuni_cli train --config ../config.yaml
```

### Evaluate Model
```bash
./bin/Release/omoknuni_cli eval --config ../config.yaml
```

### Interactive Play
```bash
./bin/Release/omoknuni_cli play --config ../config.yaml
```

## Key Architecture Components

### 1. Game Abstraction Layer
- `IGameState` interface provides unified API for all games
- Implementations: `GomokuState`, `ChessState`, `GoState`
- Each game implements move generation, validation, and outcome detection
- Zobrist hashing provides efficient state comparison

### 2. MCTS Engine
- **Leaf Parallelization**: Multiple threads expand tree, leaf states queued for batch evaluation
- **External Queue Mechanism**: Uses `moodycamel::ConcurrentQueue` for lock-free producer/consumer
- **Virtual Loss**: Prevents thread collisions during tree expansion
- **Transposition Table**: Shares search information across identical positions
- **Progressive Widening**: Controls branching factor based on visit count
- **Key Classes**:
  - `MCTSEngine`: Main search controller
  - `MCTSEvaluator`: Batch neural network evaluator
  - `MCTSNode`: Tree node with UCB calculations
  - `NodePool`: Memory pool for efficient node allocation

### 3. Neural Network Integration
- Uses PyTorch C++ API (libtorch) for CUDA inference
- DDW-RandWire-ResNet architecture with policy and value heads
- Batch inference for GPU efficiency (configurable batch size)
- Model checkpointing and loading from disk

### 4. Self-Play System
- `SelfPlayManager` orchestrates multiple game engines
- Configurable parallelism and temperature settings
- Game records saved as flat files (JSON format)
- ELO tracking for model comparison

### 5. Thread Safety Considerations
- Nodes use atomic operations for visit counts
- Queue operations are lock-free
- Virtual loss prevents race conditions during selection
- State pools reduce allocation contention

## Common Issues and Solutions

### MCTS Stalling
- Check batch timeout settings (`mcts_batch_timeout_ms`)
- Verify batch size is achievable (`mcts_batch_size`)
- Ensure evaluator thread is active
- Monitor queue sizes for deadlock conditions

### GPU Memory Issues
- Reduce `num_filters` and `num_res_blocks` in config
- Lower `mcts_batch_size`
- Check for memory leaks with `nvidia-smi`

### Build Failures
- Ensure CUDA toolkit is installed (>= 11.7)
- Check libtorch version matches CUDA version
- Verify Python development headers are installed
- Clear CMake cache if switching configurations

## Configuration

Key config.yaml parameters:
- `game_type`: gomoku, chess, or go
- `mcts_num_simulations`: Tree search depth
- `mcts_num_threads`: Worker thread count
- `mcts_batch_size`: GPU batch size
- `mcts_batch_timeout_ms`: Max wait for batch
- `mcts_virtual_loss`: Thread collision prevention
- `mcts_exploration_constant`: UCB exploration term

## Development Notes

### Adding a New Game
1. Create state class implementing `IGameState`
2. Implement `getAllLegalMoves()`, `makeMove()`, `isTerminal()`
3. Add game factory in `GameFactory`
4. Update config parser for game-specific settings

### Modifying MCTS
- Selection phase in `MCTSNode::selectBestChild()`
- Expansion in `MCTSEngine::expandNode()`
- Evaluation via `MCTSEvaluator::evaluationLoop()`
- Backpropagation in `MCTSNode::updateRecursive()`

### Debugging MCTS
- Use `MCTS_DEBUG` compile flag for verbose output
- Monitor queue sizes with debug prints
- Track node expansions per thread
- Log batch sizes and inference times