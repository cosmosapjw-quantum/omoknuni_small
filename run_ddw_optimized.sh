#!/bin/bash

# Run script for DDW-RandWire-ResNet with optimized settings
# Based on the fixed ResNet configuration

echo "🚀 Starting DDW-RandWire-ResNet Self-Play with Optimized Settings"
echo "=============================================================="

# Clean up any existing DDW model to ensure fresh start
echo "Cleaning up existing models..."
rm -f models/ddw_randwire_model.pt

# Environment setup (from fixed config)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12  # Match MCTS threads
export MKL_NUM_THREADS=12
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060 Ti architecture

# PyTorch memory settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.7"

# Start memory monitor in background
echo "Starting memory monitor..."
python3 monitor_memory_usage.py &
MONITOR_PID=$!

# Give monitor time to start
sleep 2

# Run self-play with DDW optimized config
echo "Starting DDW self-play with optimized settings..."
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_ddw_randwire_optimized.yaml

# Stop memory monitor
echo "Stopping memory monitor..."
kill $MONITOR_PID 2>/dev/null

echo "✅ DDW self-play completed!"
echo "Check memory_usage.log and memory_usage_plot.png for memory usage analysis"