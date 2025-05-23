#!/bin/bash
# Run self-play with parallel game generation

echo "Starting parallel self-play with optimized GPU utilization..."
echo "============================================="

# Set environment for CUDA
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Run self-play with parallel configuration
./build/bin/Release/omoknuni_cli self-play --config config_parallel_games.yaml

echo "Parallel self-play completed."