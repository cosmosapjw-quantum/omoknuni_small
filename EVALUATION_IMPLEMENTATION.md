# AlphaZero Evaluation/Arena Implementation

This document describes the evaluation/arena system implemented for the AlphaZero training pipeline.

## Overview

The evaluation system plays matches between the current best model (champion) and newly trained models (contenders) to determine if the new model is an improvement. This ensures that the training process only keeps models that demonstrate actual playing strength improvement.

## Components

### 1. Python Arena Module (`python/alphazero/arena.py`)

The core arena implementation that handles:
- Playing individual games between two models
- Managing match statistics (wins, losses, draws)
- Supporting both C++ and Python-only evaluation modes
- Parallel game execution for efficiency

Key features:
- **Flexible backend**: Can use C++ MCTS engine or simplified Python evaluation
- **Fair evaluation**: Alternates which player goes first
- **Configurable parameters**: MCTS simulations, temperature, batch size

### 2. Evaluation Script (`alphazero_evaluate.py`)

Command-line script that:
- Loads champion and contender models
- Runs arena matches using configured parameters
- Determines if the new model is better based on win rate threshold
- Saves evaluation results in JSON format
- Updates the best model if contender wins

Usage:
```bash
python3 alphazero_evaluate.py \
    --config config.yaml \
    --iteration 5 \
    --checkpoint-dir checkpoints/
```

### 3. C++ Arena Implementation (`include/evaluation/arena.h`, `src/evaluation/arena.cpp`)

High-performance C++ implementation that:
- Uses the full MCTS engine for accurate game play
- Supports parallel game execution with thread pools
- Provides detailed game statistics (nodes searched, duration)
- Integrates seamlessly with existing MCTS infrastructure

### 4. Shell Script Integration

The `run_alphazero_training_with_progress.sh` script now includes proper evaluation:
- Phase 4 calls the evaluation script
- Checks if the new model is better
- Updates the best model only if improvement is demonstrated
- Falls back gracefully if evaluation fails

## Configuration

Evaluation parameters in `config_alphazero_train.yaml`:

```yaml
# Pipeline settings
pipeline:
  evaluation_threshold: 0.55  # Win rate needed to become champion

# Evaluation settings  
evaluation:
  num_games: 100              # Total games to play
  num_parallel_games: 10      # Concurrent games
  mcts_simulations: 1600      # MCTS sims per move (higher = stronger)
  temperature: 0.1            # Low for deterministic play
```

## How It Works

1. **Model Loading**: Both champion and contender models are loaded
2. **Arena Match**: Models play a configured number of games
3. **Fair Play**: Players alternate who goes first to ensure fairness
4. **Statistics**: Win/loss/draw counts are tracked
5. **Decision**: If contender wins ≥ threshold (default 55%), it becomes the new champion
6. **Model Update**: Best model is updated only on improvement

## Testing

Run the test suite to verify the evaluation system:

```bash
python3 test_evaluation.py
```

This tests:
- Basic arena functionality
- Model comparison logic
- Evaluation script integration
- Result saving and loading

## Benefits

1. **Quality Assurance**: Only genuinely better models are kept
2. **Training Stability**: Prevents regression from bad training iterations
3. **Measurable Progress**: Provides concrete win rates for each iteration
4. **Flexibility**: Works with or without C++ acceleration

## Future Enhancements

- ELO rating system for tracking model strength over time
- Distributed evaluation across multiple machines
- Opening book analysis for more varied games
- Position-specific evaluation for debugging