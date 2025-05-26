#!/bin/bash

echo "=========================================="
echo "Testing GPU Memory Fixes and CPU/GPU Balance"
echo "Hardware: Ryzen 9 5900X (24T), RTX 3060 Ti (8GB)"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=0

# Kill any existing monitoring
pkill -f "nvidia-smi.*gpu_monitor"
pkill -f "monitor_cpu"

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv -l 2 > gpu_monitor_test.log &
GPU_PID=$!

# Start CPU monitoring
(while true; do 
    echo "$(date +%s),$(mpstat 1 1 | grep Average | awk '{print 100-$12}')" >> cpu_monitor_test.log
    sleep 1
done) &
CPU_PID=$!

# Clear GPU memory before starting
nvidia-smi --gpu-reset -i 0 2>/dev/null || true

echo "Starting test run..."
echo "Monitor files: gpu_monitor_test.log, cpu_monitor_test.log"
echo ""

# Run the optimized version with library path
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    time ./build/bin/Release/omoknuni_cli_final self-play config_optimized_gpu_fix.yaml

# Stop monitoring
kill $GPU_PID 2>/dev/null
kill $CPU_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Test Complete! Analyzing results..."
echo "=========================================="

# Analyze GPU memory usage
echo ""
echo "GPU Memory Analysis:"
INITIAL_MEM=$(head -10 gpu_monitor_test.log | grep -v timestamp | head -1 | cut -d',' -f2 | tr -d ' MiB')
PEAK_MEM=$(cut -d',' -f2 gpu_monitor_test.log | grep -v memory | tr -d ' MiB' | sort -n | tail -1)
FINAL_MEM=$(tail -10 gpu_monitor_test.log | grep -v timestamp | tail -1 | cut -d',' -f2 | tr -d ' MiB')
AVG_GPU_UTIL=$(cut -d',' -f4 gpu_monitor_test.log | grep -v utilization | tr -d ' %' | awk '{sum+=$1; count++} END {print sum/count}')

echo "  Initial VRAM: ${INITIAL_MEM} MiB"
echo "  Peak VRAM: ${PEAK_MEM} MiB"
echo "  Final VRAM: ${FINAL_MEM} MiB"
echo "  Average GPU Utilization: ${AVG_GPU_UTIL}%"

# Check for memory leak
LEAK_MB=$((FINAL_MEM - INITIAL_MEM))
if [ $LEAK_MB -gt 500 ]; then
    echo "  ⚠️  WARNING: Possible memory leak detected (${LEAK_MB} MiB growth)"
else
    echo "  ✅ Memory usage stable (${LEAK_MB} MiB growth)"
fi

# Analyze CPU usage
echo ""
echo "CPU Usage Analysis:"
AVG_CPU=$(awk -F',' '{sum+=$2; count++} END {print sum/count}' cpu_monitor_test.log)
echo "  Average CPU Utilization: ${AVG_CPU}%"

if (( $(echo "$AVG_CPU < 50" | bc -l) )); then
    echo "  ⚠️  WARNING: Low CPU utilization"
elif (( $(echo "$AVG_CPU > 80" | bc -l) )); then
    echo "  ✅ Good CPU utilization"
else
    echo "  ⚠️  Moderate CPU utilization"
fi

echo ""
echo "Full logs saved to:"
echo "  - gpu_monitor_test.log"
echo "  - cpu_monitor_test.log"