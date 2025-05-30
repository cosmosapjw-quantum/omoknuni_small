#!/bin/bash

echo "GPU Utilization Monitor for MCTS Benchmark"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found! Please install NVIDIA drivers."
    exit 1
fi

# Create output directory
mkdir -p benchmark_results

# Start monitoring in background
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > benchmark_results/gpu_utilization.csv &
MONITOR_PID=$!

echo "GPU monitoring started (PID: $MONITOR_PID)"
echo "Data being saved to: benchmark_results/gpu_utilization.csv"
echo ""

# Also show live stats
echo "Live GPU Stats (Ctrl+C to stop):"
echo "---------------------------------"
watch -n 1 "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader,nounits | awk -F',' '{printf \"GPU: %3d%% | Mem: %3d%% | Used: %5d MB | Temp: %2d°C\n\", \$1, \$2, \$3, \$4}'"

# Cleanup
kill $MONITOR_PID 2>/dev/null
echo "Monitoring stopped."