#!/bin/bash

# Run optimized AlphaZero pipeline with better batching configuration
echo "Starting optimized AlphaZero pipeline with enhanced batching..."

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Run the pipeline with the optimized configuration
./build/bin/Release/omoknuni_cli self-play \
    --config config.yaml \
    --output-dir data/self_play_games_optimized \
    --num-games 10