#!/bin/bash

# Run script for balanced DDW-RandWire-ResNet configuration
# Prevents CPU bottlenecks while maintaining good GPU utilization

echo "🏃 Starting DDW-RandWire-ResNet with Balanced Settings"
echo "======================================================"

# Clean up any existing models
echo "Cleaning up existing models..."
rm -f models/ddw_randwire_balanced.pt

# Environment setup for balanced performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12  # Match config threads
export MKL_NUM_THREADS=12
export TORCH_CUDA_ARCH_LIST="8.6"

# PyTorch memory settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.7"

# CPU affinity (optional - bind to specific cores to reduce context switching)
# taskset -c 0-11 can be added before the command if needed

# Start memory monitor
echo "Starting memory monitor..."
python3 monitor_memory_usage.py &
MONITOR_PID=$!

# Give monitor time to start
sleep 2

# Run self-play with balanced config
echo "Starting DDW self-play with balanced settings..."
echo "Target: 60-75% GPU utilization without CPU bottlenecks"
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_ddw_balanced.yaml

# Stop memory monitor
echo "Stopping memory monitor..."
kill $MONITOR_PID 2>/dev/null

echo "✅ Balanced DDW self-play completed!"
echo "Check memory_usage.log for detailed memory analysis"