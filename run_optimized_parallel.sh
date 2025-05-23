#!/bin/bash
# Optimized parallel self-play for sustained 70%+ utilization

echo "🚀 Starting OPTIMIZED parallel self-play"
echo "========================================"
echo "Configuration:"
echo "  - 4 parallel games (vs 3)"
echo "  - 6 MCTS threads per game (24 total)"
echo "  - Larger batches (48)"
echo "  - Shorter timeout (15ms)"
echo ""

# Set environment for maximum performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24

# CPU affinity settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# Disable CPU frequency scaling for consistent performance
echo "Setting CPU governor to performance mode..."
sudo cpupower frequency-set -g performance 2>/dev/null || true

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Start monitoring in background
echo "Starting utilization monitor..."
python3 monitor_utilization.py &
MONITOR_PID=$!

# Give monitor time to start
sleep 2

# Run optimized self-play
./build/bin/Release/omoknuni_cli self-play --config config_optimized_parallel.yaml

# Kill monitor
kill $MONITOR_PID 2>/dev/null

echo ""
echo "Optimized parallel self-play completed."

# Reset CPU governor
sudo cpupower frequency-set -g ondemand 2>/dev/null || true