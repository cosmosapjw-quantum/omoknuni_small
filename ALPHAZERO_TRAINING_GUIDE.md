# AlphaZero Training Pipeline Guide

This guide explains how to use the complete AlphaZero training pipeline for training game-playing AI models.

## Overview

The AlphaZero training pipeline consists of three main phases that repeat in a loop:

1. **Self-Play**: Generate training games using the current best model
2. **Training**: Train the neural network on the generated games
3. **Evaluation**: Compare the new model against the previous best

## Quick Start

For a quick test run with minimal configuration:

```bash
./run_alphazero_simple.sh
```

This will run 10 iterations with 100 games per iteration.

## Full Pipeline

For production training with all features:

```bash
./run_alphazero_training.sh config_alphazero_train.yaml
```

## Configuration

The main configuration file `config_alphazero_train.yaml` contains all training parameters:

### Key Parameters

- `pipeline.num_iterations`: Total training iterations (default: 100)
- `pipeline.games_per_iteration`: Games to generate per iteration (default: 1000)
- `pipeline.evaluation_games`: Games for model evaluation (default: 100)
- `pipeline.evaluation_threshold`: Win rate needed to update model (default: 0.55)

### MCTS Settings

- `mcts.num_simulations`: Simulations per move (default: 800)
- `mcts.batch_size`: Neural network batch size (default: 256)
- `mcts.exploration_constant`: UCB exploration parameter (default: 1.25)

### Neural Network

Two architectures are supported:

1. **ResNet** (traditional)
   ```yaml
   network_type: resnet
   num_filters: 128
   num_res_blocks: 10
   ```

2. **DDW-RandWire** (experimental)
   ```yaml
   network_type: ddw_randwire
   ddw_channels: 64
   ddw_num_blocks: 6
   ```

### Training Hyperparameters

- `training.batch_size`: Training batch size (default: 512)
- `training.learning_rate`: Initial learning rate (default: 0.002)
- `training.epochs_per_iteration`: Training epochs (default: 10)

## Directory Structure

```
omoknuni_small/
├── checkpoints/alphazero/     # Model checkpoints
│   ├── best_model.pt          # Best model so far
│   ├── current_model.pt       # Current training model
│   └── model_iter_N.pt        # Checkpoint for iteration N
├── data/
│   ├── self_play_games/       # Generated game records
│   ├── training_data/         # Processed training data
│   └── evaluation_games/      # Evaluation match records
├── logs/alphazero/            # Training logs
│   ├── selfplay_N.log         # Self-play logs
│   ├── training_N.log         # Training logs
│   └── tensorboard/           # TensorBoard data
└── tensorboard/               # TensorBoard visualizations
```

## Monitoring Training

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir logs/alphazero/tensorboard --port 6006
```

Then open http://localhost:6006 in your browser.

### Memory Usage

Monitor system memory during training:

```bash
python3 monitor_memory_usage.py
```

### GPU Utilization

Check GPU usage:

```bash
watch -n 1 nvidia-smi
```

## Resume Training

To resume from a checkpoint:

```bash
./run_alphazero_training.sh config_alphazero_train.yaml checkpoints/alphazero/model_iter_50.pt
```

## Performance Optimization

### Memory Management

For systems with limited memory:

1. Reduce `pipeline.games_per_iteration`
2. Lower `pipeline.training_window_size`
3. Decrease `mcts.batch_size`
4. Enable `memory.empty_cuda_cache_on_pressure`

### Training Speed

To speed up training:

1. Increase `self_play.parallel_self_play_workers`
2. Enable `training.use_amp` (mixed precision)
3. Increase `training.accumulation_steps`
4. Use larger `training.batch_size`

### GPU Utilization

For better GPU usage:

1. Increase `mcts.batch_size` (up to GPU memory limit)
2. Reduce `mcts.batch_timeout_ms`
3. Enable `neural_network.gpu_memory_fraction: 0.8`

## Game-Specific Settings

### Gomoku 9x9

```yaml
game_type: gomoku
board_size: 9
input_channels: 19  # With attack/defense planes
mcts.num_simulations: 800
```

### Gomoku 15x15

```yaml
game_type: gomoku
board_size: 15
input_channels: 19
mcts.num_simulations: 1200
```

### Chess

```yaml
game_type: chess
board_size: 8
input_channels: 119  # AlphaZero-style features
mcts.num_simulations: 1600
```

### Go

```yaml
game_type: go
board_size: 19
input_channels: 19
mcts.num_simulations: 1600
```

## Troubleshooting

### Out of Memory Errors

1. Check memory usage: `free -h`
2. Reduce batch sizes in configuration
3. Enable memory cleanup options
4. Restart training with cleared cache

### Slow Training

1. Check CPU usage: `htop`
2. Verify GPU is being used: `nvidia-smi`
3. Reduce `mcts.num_simulations`
4. Increase parallel workers

### Model Not Improving

1. Check evaluation threshold (may be too high)
2. Increase `pipeline.games_per_iteration`
3. Adjust learning rate schedule
4. Check for bugs in game implementation

## Advanced Features

### Distributed Training

Enable multi-GPU or multi-node training:

```yaml
distributed:
  enabled: true
  backend: nccl
  world_size: 4  # Number of GPUs
```

### Custom Evaluation

Implement custom evaluation metrics:

```python
# In alphazero_train.py
def custom_evaluation(new_model, old_model):
    # Your evaluation logic
    return win_rate
```

### Data Augmentation

Enable rotation and reflection augmentation:

```yaml
training:
  use_augmentation: true
  augmentation_types:
    - rotation
    - reflection
```

## Best Practices

1. **Start Small**: Begin with fewer iterations and games to verify setup
2. **Monitor Metrics**: Use TensorBoard to track training progress
3. **Save Checkpoints**: Keep checkpoints every few iterations
4. **Validate Results**: Periodically play against the model manually
5. **Tune Gradually**: Adjust one parameter at a time

## Example Training Run

```bash
# 1. Build the project
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

# 2. Verify setup
./build/bin/Release/omoknuni_cli_final --help

# 3. Start training
./run_alphazero_training.sh config_alphazero_train.yaml

# 4. Monitor progress
tensorboard --logdir logs/alphazero/tensorboard

# 5. Test the model
./build/bin/Release/omoknuni_cli_final play \
    --model checkpoints/alphazero/best_model.pt \
    --mcts-simulations 1600
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `logs/alphazero/`
3. Verify configuration parameters
4. Test with simplified settings first