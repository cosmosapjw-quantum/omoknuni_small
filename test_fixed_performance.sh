#!/bin/bash

# Test script to validate memory and CPU fixes
# Monitors CPU usage, memory usage, and GPU utilization

echo "🚀 Starting performance test with memory and CPU fixes..."
echo "=================================================="

# Configuration
TEST_DURATION=120  # Run for 2 minutes
MONITOR_INTERVAL=2  # Check every 2 seconds
CONFIG_FILE="config_fixed_memory_cpu.yaml"
LOG_DIR="test_logs_$(date +%Y%m%d_%H%M%S)"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to get current memory usage
get_memory_usage() {
    ps aux | grep omoknuni_cli_final | grep -v grep | awk '{print $6}' | head -1
}

# Function to get CPU usage
get_cpu_usage() {
    ps aux | grep omoknuni_cli_final | grep -v grep | awk '{print $3}' | head -1
}

# Function to get GPU memory usage
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

# Function to get GPU utilization
get_gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
}

# Start monitoring in background
echo "Starting background monitors..."
(
    echo "Time,CPU%,Memory_KB,GPU_Memory_MB,GPU_Util%" > "$LOG_DIR/performance.csv"
    while true; do
        TIMESTAMP=$(date +%s)
        CPU=$(get_cpu_usage)
        MEM=$(get_memory_usage)
        GPU_MEM=$(get_gpu_memory)
        GPU_UTIL=$(get_gpu_utilization)
        
        if [ ! -z "$CPU" ]; then
            echo "$TIMESTAMP,$CPU,$MEM,$GPU_MEM,$GPU_UTIL" >> "$LOG_DIR/performance.csv"
            echo "[$(date +%H:%M:%S)] CPU: ${CPU}% | RAM: $((MEM/1024))MB | VRAM: ${GPU_MEM}MB | GPU: ${GPU_UTIL}%"
        fi
        
        sleep $MONITOR_INTERVAL
    done
) &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Stopping monitors..."
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
    
    # Analyze results
    echo -e "\n📊 PERFORMANCE ANALYSIS:"
    echo "========================"
    
    if [ -f "$LOG_DIR/performance.csv" ]; then
        # Calculate averages
        AVG_CPU=$(awk -F, 'NR>1 {sum+=$2; count++} END {if(count>0) print sum/count; else print 0}' "$LOG_DIR/performance.csv")
        AVG_MEM=$(awk -F, 'NR>1 {sum+=$3; count++} END {if(count>0) print sum/count/1024; else print 0}' "$LOG_DIR/performance.csv")
        AVG_GPU_MEM=$(awk -F, 'NR>1 {sum+=$4; count++} END {if(count>0) print sum/count; else print 0}' "$LOG_DIR/performance.csv")
        AVG_GPU_UTIL=$(awk -F, 'NR>1 {sum+=$5; count++} END {if(count>0) print sum/count; else print 0}' "$LOG_DIR/performance.csv")
        
        # Find peaks
        PEAK_MEM=$(awk -F, 'NR>1 {if($3>max) max=$3} END {print max/1024}' "$LOG_DIR/performance.csv")
        PEAK_GPU_MEM=$(awk -F, 'NR>1 {if($4>max) max=$4} END {print max}' "$LOG_DIR/performance.csv")
        
        # Check for memory leaks (increasing trend)
        FIRST_MEM=$(awk -F, 'NR==2 {print $3}' "$LOG_DIR/performance.csv")
        LAST_MEM=$(awk -F, 'NR>1 {last=$3} END {print last}' "$LOG_DIR/performance.csv")
        MEM_GROWTH=$(echo "scale=2; ($LAST_MEM - $FIRST_MEM) / 1024" | bc)
        
        FIRST_GPU=$(awk -F, 'NR==2 {print $4}' "$LOG_DIR/performance.csv")
        LAST_GPU=$(awk -F, 'NR>1 {last=$4} END {print last}' "$LOG_DIR/performance.csv")
        GPU_GROWTH=$(echo "scale=2; $LAST_GPU - $FIRST_GPU" | bc)
        
        echo "Average CPU Usage: ${AVG_CPU}%"
        echo "Average RAM Usage: ${AVG_MEM}MB (Peak: ${PEAK_MEM}MB)"
        echo "Average VRAM Usage: ${AVG_GPU_MEM}MB (Peak: ${PEAK_GPU_MEM}MB)"
        echo "Average GPU Utilization: ${AVG_GPU_UTIL}%"
        echo ""
        echo "Memory Growth: ${MEM_GROWTH}MB"
        echo "VRAM Growth: ${GPU_GROWTH}MB"
        
        # Determine if fixes worked
        echo -e "\n🎯 FIX VALIDATION:"
        
        # CPU usage should be > 50%
        if (( $(echo "$AVG_CPU > 50" | bc -l) )); then
            echo "✅ CPU Usage: GOOD (${AVG_CPU}% > 50%)"
        else
            echo "❌ CPU Usage: LOW (${AVG_CPU}% < 50%)"
        fi
        
        # Memory growth should be minimal
        if (( $(echo "$MEM_GROWTH < 100" | bc -l) )); then
            echo "✅ Memory Growth: CONTROLLED (${MEM_GROWTH}MB < 100MB)"
        else
            echo "❌ Memory Growth: HIGH (${MEM_GROWTH}MB > 100MB)"
        fi
        
        # VRAM growth should be minimal
        if (( $(echo "$GPU_GROWTH < 500" | bc -l) )); then
            echo "✅ VRAM Growth: CONTROLLED (${GPU_GROWTH}MB < 500MB)"
        else
            echo "❌ VRAM Growth: HIGH (${GPU_GROWTH}MB > 500MB)"
        fi
        
        # GPU utilization should be high
        if (( $(echo "$AVG_GPU_UTIL > 70" | bc -l) )); then
            echo "✅ GPU Utilization: GOOD (${AVG_GPU_UTIL}% > 70%)"
        else
            echo "⚠️  GPU Utilization: SUBOPTIMAL (${AVG_GPU_UTIL}% < 70%)"
        fi
    fi
    
    echo -e "\nLogs saved to: $LOG_DIR/"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Run the self-play with monitoring
echo -e "\n🎮 Starting self-play with config: $CONFIG_FILE"
echo "Running for $TEST_DURATION seconds..."
echo "=================================================="

# Run with timeout
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
timeout $TEST_DURATION ./build/bin/Release/omoknuni_cli_final self-play $CONFIG_FILE 2>&1 | tee "$LOG_DIR/output.log"

# Cleanup will be called automatically by trap