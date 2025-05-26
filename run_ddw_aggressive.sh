#!/bin/bash

# Run script for DDW-RandWire-ResNet with aggressive settings
# For maximum GPU utilization while maintaining stability

echo "🔥 Starting DDW-RandWire-ResNet Self-Play with Aggressive Settings"
echo "================================================================"

# Clean up any existing DDW model to ensure fresh start
echo "Cleaning up existing models..."
rm -f models/ddw_randwire_model_aggressive.pt

# Environment setup for aggressive performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20  # Use more threads for aggressive config
export MKL_NUM_THREADS=20
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060 Ti architecture

# PyTorch memory settings - more aggressive
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,garbage_collection_threshold:0.9"

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# Start GPU monitor for aggressive settings
echo "Starting GPU utilization monitor..."
nvidia-smi dmon -i 0 -s pucm -d 5 > gpu_monitor_aggressive.log &
GPU_MONITOR_PID=$!

# Start memory monitor in background
echo "Starting memory monitor..."
python3 monitor_memory_usage.py &
MONITOR_PID=$!

# Give monitors time to start
sleep 2

# Run self-play with DDW aggressive config
echo "Starting DDW self-play with aggressive settings..."
echo "Target: 90%+ GPU utilization"
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_ddw_randwire_aggressive.yaml

# Stop monitors
echo "Stopping monitors..."
kill $MONITOR_PID 2>/dev/null
kill $GPU_MONITOR_PID 2>/dev/null

echo "✅ Aggressive DDW self-play completed!"
echo "Check the following logs:"
echo "  - memory_usage.log: System memory usage"
echo "  - gpu_monitor_aggressive.log: GPU utilization stats"
echo "  - memory_usage_plot.png: Memory usage visualization"