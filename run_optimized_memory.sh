#!/bin/bash

# Optimized run script with memory-efficient settings
# For Ryzen 9 5900X + RTX 3060 Ti

echo "🚀 Starting Optimized Self-Play with Memory Management"
echo "===================================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12  # Limit OpenMP threads to prevent oversubscription
export MKL_NUM_THREADS=12
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060 Ti architecture

# PyTorch memory settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.7"

# Start memory monitor in background
# echo "Starting memory monitor..."
# python3 monitor_memory_usage.py &
# MONITOR_PID=$!

# # Give monitor time to start
# sleep 2

# Run self-play with optimized config
echo "Starting self-play with optimized settings..."
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_fixed_memory_cpu.yaml

# Stop memory monitor
# echo "Stopping memory monitor..."
# kill $MONITOR_PID 2>/dev/null

echo "✅ Self-play completed!"
# echo "Check memory_usage.log and memory_usage_plot.png for memory usage analysis"