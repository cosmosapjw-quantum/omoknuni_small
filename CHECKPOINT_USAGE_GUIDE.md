# AlphaZero Checkpoint Auto-Saving/Loading Guide

This guide explains the checkpoint functionality that has been added to the AlphaZero training pipeline.

## Overview

The checkpoint system automatically saves training progress at regular intervals and allows you to resume training from where you left off. This is useful for:
- Long training runs that might be interrupted
- Experimenting with different hyperparameters from a saved state
- Recovering from crashes or system failures

## Features

### 1. Automatic Checkpoint Saving
- Checkpoints are saved every N iterations (configurable)
- Each checkpoint includes:
  - Model weights
  - Iteration number
  - Training metadata
  - Configuration used

### 2. Automatic Checkpoint Loading
- Training automatically resumes from the latest checkpoint if available
- Can be disabled if you want to start fresh

### 3. Checkpoint Management
- Automatically cleans up old checkpoints to save disk space
- Keeps only the most recent N checkpoints (configurable)

## Configuration

Add these settings to your config YAML file:

```yaml
# Checkpoint settings
checkpoint_dir: checkpoints/alphazero_gomoku
pipeline:
  checkpoint_interval: 5        # Save checkpoint every 5 iterations
  resume_from_checkpoint: true  # Resume from latest checkpoint if available
training:
  keep_checkpoint_max: 10      # Keep only the latest 10 checkpoints
```

## Usage

### Running Training with Checkpoints

#### Using the Shell Script
```bash
# Normal usage - will automatically resume from checkpoint if available
./run_alphazero_training_with_progress.sh config_alphazero_train.yaml

# Force fresh start (ignore existing checkpoints)
./run_alphazero_training_with_progress.sh config_alphazero_train.yaml false false

# Parameters: config_file verbose resume_from_checkpoint
```

#### Using Python Directly
```bash
# Resume from checkpoint (default)
python python/alphazero/alphazero_trainer.py --config config_alphazero_train.yaml

# Force fresh start
python python/alphazero/alphazero_trainer.py --config config_alphazero_train.yaml --no-resume

# Specify custom checkpoint directory
python python/alphazero/alphazero_trainer.py --config config_alphazero_train.yaml --checkpoint-dir my_checkpoints
```

### Checkpoint Structure

Checkpoints are saved in the following structure:
```
checkpoints/alphazero_gomoku/
├── checkpoint_iter_5.json      # Metadata for iteration 5
├── model_iter_5.pt            # Model weights for iteration 5
├── checkpoint_iter_10.json     # Metadata for iteration 10
├── model_iter_10.pt           # Model weights for iteration 10
└── ...
```

### Monitoring Progress

When training resumes from a checkpoint, you'll see messages like:
```
📂 Found checkpoint at iteration 15
🔄 Resuming from iteration 16
✅ Successfully loaded checkpoint from iteration 15
```

### Testing Checkpoint Functionality

Run the test script to verify checkpoints work correctly:
```bash
# Test all checkpoint features
python test_checkpoint.py

# Test only save/load
python test_checkpoint.py --test save-load

# Test only resume functionality
python test_checkpoint.py --test resume
```

## Advanced Usage

### Manual Checkpoint Management

You can manually manage checkpoints if needed:

```python
from alphazero.alphazero_trainer import AlphaZeroTrainer

# Load trainer
trainer = AlphaZeroTrainer(config_path="config.yaml")

# Save checkpoint manually
trainer._save_checkpoint(iteration=42)

# Find latest checkpoint
latest = trainer._find_latest_checkpoint()
print(f"Latest checkpoint: iteration {latest['iteration']}")

# Load specific checkpoint
trainer._load_checkpoint(latest)
```

### Emergency Checkpoints

If training crashes, an emergency checkpoint is automatically saved with the current state.

## Troubleshooting

### Checkpoint Not Found
If you see "No checkpoint found, starting from iteration 0":
- Check that `checkpoint_dir` exists and contains checkpoint files
- Verify `resume_from_checkpoint` is set to `true` in config
- Check file permissions

### Checkpoint Loading Failed
If checkpoint loading fails:
- Verify the model file exists alongside the checkpoint JSON
- Check that the model architecture hasn't changed
- Look for error messages in the log files

### Disk Space Issues
If running out of disk space:
- Reduce `keep_checkpoint_max` to keep fewer checkpoints
- Manually delete old checkpoints from the checkpoint directory
- Check that checkpoint cleanup is working properly

## Best Practices

1. **Regular Intervals**: Set `checkpoint_interval` based on iteration time
   - Fast iterations (< 5 min): checkpoint every 10-20 iterations
   - Slow iterations (> 30 min): checkpoint every 1-5 iterations

2. **Disk Space**: Plan for checkpoint storage
   - Each checkpoint = model size + metadata
   - Keep enough checkpoints to recover from issues
   - But not so many that you run out of disk space

3. **Backup Important Checkpoints**: For milestone checkpoints, copy them to a backup location

4. **Version Control**: Keep your config file in version control to reproduce training

## Implementation Details

The checkpoint system is implemented in:
- **Python**: `python/alphazero/alphazero_trainer.py` - Core checkpoint logic with save/load/resume functionality
- **Shell**: `run_alphazero_training_with_progress.sh` - Shell script integration with automatic resume support

The system uses JSON files for metadata and PyTorch's native `.pt` format for model weights.