#!/bin/bash

echo "Running optimized MCTS with GPU memory fixes..."
echo "Hardware: Ryzen 9 5900X, RTX 3060 Ti (8GB VRAM)"
echo "=========================================="

# Set environment variables for optimal performance
export OMP_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060 Ti architecture

# Clear GPU memory before starting
nvidia-smi --gpu-reset -i 0 2>/dev/null || true

# Monitor GPU memory in background
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > gpu_monitor_optimized.log &
MONITOR_PID=$!

# Run the optimized configuration
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_optimized_gpu_fix.yaml

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

echo "=========================================="
echo "Run complete. Check gpu_monitor_optimized.log for GPU memory usage."
echo "Compare with previous runs to verify VRAM leak is fixed."