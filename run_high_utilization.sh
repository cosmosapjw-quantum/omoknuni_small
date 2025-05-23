#!/bin/bash
# Run self-play with settings optimized for 70%+ CPU and GPU utilization

echo "🔥 Starting HIGH CPU/GPU UTILIZATION self-play!"
echo "==============================================="
echo "Key optimizations:"
echo "  - 8 MCTS threads per game (vs 4)"
echo "  - 3 parallel games = 24 total CPU threads"  
echo "  - Larger batch size (64)"
echo "  - Shorter batch timeout (10ms)"
echo "  - Higher virtual loss (5)"
echo ""

# Set environment for maximum performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24

# Thread affinity for better CPU utilization
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Monitor system utilization in background
echo "Starting system monitors..."
(
    while true; do
        # Get CPU usage
        CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        
        # Get GPU usage  
        GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
        
        echo -ne "\r🖥️  CPU: ${CPU}% | 🎮 GPU: ${GPU}% | Target: 70%+ for both"
        sleep 2
    done
) &
MONITOR_PID=$!

# Run self-play with high utilization config
./build/bin/Release/omoknuni_cli self-play --config config_high_cpu_gpu.yaml

# Kill monitor
kill $MONITOR_PID 2>/dev/null

echo ""
echo "High utilization self-play completed."