#!/bin/bash

# Quick performance test - just 5 games to check timing improvements
echo "🚀 Quick Performance Test"
echo "========================="

# Set environment
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12

# Create temporary config with fewer games
cat > config_test_quick.yaml << 'EOF'
game_type: gomoku
board_size: 15
input_channels: 19

mcts:
  num_simulations: 200
  num_threads: 8
  exploration_constant: 1.0
  batch_size: 128
  batch_timeout_ms: 20
  use_transposition_table: true

neural_network:
  model_type: resnet
  num_filters: 64
  num_res_blocks: 10

self_play:
  num_games: 5  # Just 5 games for quick test
  parallel_games: 6

model_path: models/model.pt
output_dir: data/self_play_games
mcts_temp_threshold: 30
mcts_temperature: 1.0
mcts_virtual_loss: 1.0
save_interval: 1
EOF

# Run test
echo "Running quick test with optimized MCTS..."
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    timeout 120s ./build/bin/Release/omoknuni_cli_final self-play config_test_quick.yaml

echo "✅ Quick test completed!"