#!/bin/bash

# Enhanced performance test script with accurate CPU monitoring
# Uses pidstat for better CPU usage tracking

echo "🚀 Starting enhanced performance test..."
echo "=================================================="

# Configuration
TEST_DURATION=120  # Run for 2 minutes
MONITOR_INTERVAL=2  # Check every 2 seconds
CONFIG_FILE="config_optimized_memory.yaml"
LOG_DIR="test_logs_$(date +%Y%m%d_%H%M%S)"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if pidstat is available
if ! command -v pidstat &> /dev/null; then
    echo "Installing sysstat for better CPU monitoring..."
    sudo apt-get update && sudo apt-get install -y sysstat
fi

# Function to get process PID
get_pid() {
    pgrep -f "omoknuni_cli_final self-play" | head -1
}

# Function to get current memory usage (in MB)
get_memory_usage() {
    local pid=$1
    if [ ! -z "$pid" ]; then
        ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}'
    fi
}

# Function to get CPU usage using pidstat
get_cpu_usage() {
    local pid=$1
    if [ ! -z "$pid" ]; then
        pidstat -p $pid 1 1 2>/dev/null | grep -E "^Average:" | awk '{print $8}'
    fi
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
echo "Starting enhanced monitoring..."
(
    echo "[MEMORY_MONITOR] Starting monitoring..."
    echo "Time,PID,CPU%,Memory_MB,GPU_Memory_MB,GPU_Util%" > "$LOG_DIR/performance.csv"
    
    while true; do
        PID=$(get_pid)
        if [ ! -z "$PID" ]; then
            TIMESTAMP=$(date +%s)
            
            # Get CPU usage (this takes 1 second due to pidstat)
            CPU=$(get_cpu_usage $PID)
            
            # Get other metrics
            MEM=$(get_memory_usage $PID)
            GPU_MEM=$(get_gpu_memory)
            GPU_UTIL=$(get_gpu_utilization)
            
            if [ ! -z "$CPU" ] && [ ! -z "$MEM" ]; then
                echo "$TIMESTAMP,$PID,$CPU,$MEM,$GPU_MEM,$GPU_UTIL" >> "$LOG_DIR/performance.csv"
                printf "[%s] PID: %s | CPU: %.1f%% | RAM: %.0fMB | VRAM: %sMB | GPU: %s%%\n" \
                    "$(date +%H:%M:%S)" "$PID" "$CPU" "$MEM" "$GPU_MEM" "$GPU_UTIL"
            fi
        else
            echo "[$(date +%H:%M:%S)] Waiting for process to start..."
        fi
        
        sleep $MONITOR_INTERVAL
    done
) &
MONITOR_PID=$!

# Alternative monitoring using top for cross-validation
(
    while true; do
        PID=$(get_pid)
        if [ ! -z "$PID" ]; then
            TOP_CPU=$(top -b -n 2 -d 1 -p $PID 2>/dev/null | tail -1 | awk '{print $9}')
            if [ ! -z "$TOP_CPU" ]; then
                echo "[TOP] CPU: ${TOP_CPU}%" >> "$LOG_DIR/top_cpu.log"
            fi
        fi
        sleep 5
    done
) &
TOP_MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo -e "\n[MEMORY_MONITOR] Stopped monitoring"
    echo "🛑 Stopping monitors..."
    kill $MONITOR_PID 2>/dev/null
    kill $TOP_MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
    wait $TOP_MONITOR_PID 2>/dev/null
    
    # Analyze results
    echo -e "\n📊 PERFORMANCE ANALYSIS:"
    echo "========================"
    
    if [ -f "$LOG_DIR/performance.csv" ]; then
        # Calculate averages (skip header)
        AVG_CPU=$(awk -F, 'NR>1 && $3!="" {sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}' "$LOG_DIR/performance.csv")
        AVG_MEM=$(awk -F, 'NR>1 && $4!="" {sum+=$4; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}' "$LOG_DIR/performance.csv")
        AVG_GPU_MEM=$(awk -F, 'NR>1 && $5!="" {sum+=$5; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}' "$LOG_DIR/performance.csv")
        AVG_GPU_UTIL=$(awk -F, 'NR>1 && $6!="" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}' "$LOG_DIR/performance.csv")
        
        # Find peaks
        PEAK_CPU=$(awk -F, 'NR>1 && $3!="" {if($3>max) max=$3} END {printf "%.2f", max}' "$LOG_DIR/performance.csv")
        PEAK_MEM=$(awk -F, 'NR>1 && $4!="" {if($4>max) max=$4} END {printf "%.2f", max}' "$LOG_DIR/performance.csv")
        PEAK_GPU_MEM=$(awk -F, 'NR>1 && $5!="" {if($5>max) max=$5} END {printf "%.2f", max}' "$LOG_DIR/performance.csv")
        
        # Get first and last valid readings for growth calculation
        FIRST_MEM=$(awk -F, 'NR>1 && $4!="" {print $4; exit}' "$LOG_DIR/performance.csv")
        LAST_MEM=$(awk -F, 'NR>1 && $4!="" {last=$4} END {print last}' "$LOG_DIR/performance.csv")
        MEM_GROWTH=$(echo "scale=2; $LAST_MEM - $FIRST_MEM" | bc)
        
        FIRST_GPU=$(awk -F, 'NR>1 && $5!="" {print $5; exit}' "$LOG_DIR/performance.csv")
        LAST_GPU=$(awk -F, 'NR>1 && $5!="" {last=$5} END {print last}' "$LOG_DIR/performance.csv")
        GPU_GROWTH=$(echo "scale=2; $LAST_GPU - $FIRST_GPU" | bc)
        
        # Count data points
        DATA_POINTS=$(awk -F, 'NR>1 && $3!="" {count++} END {print count}' "$LOG_DIR/performance.csv")
        
        echo "Data points collected: $DATA_POINTS"
        echo "Average CPU Usage: ${AVG_CPU}% (Peak: ${PEAK_CPU}%)"
        echo "Average RAM Usage: ${AVG_MEM}MB (Peak: ${PEAK_MEM}MB)"
        echo "Average VRAM Usage: ${AVG_GPU_MEM}MB (Peak: ${PEAK_GPU_MEM}MB)"
        echo "Average GPU Utilization: ${AVG_GPU_UTIL}%"
        echo ""
        echo "Memory Growth: ${MEM_GROWTH}MB"
        echo "VRAM Growth: ${GPU_GROWTH}MB"
        
        # Show top-based CPU average for comparison
        if [ -f "$LOG_DIR/top_cpu.log" ]; then
            TOP_AVG=$(awk '{sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count}' "$LOG_DIR/top_cpu.log")
            echo "CPU Usage (top validation): ${TOP_AVG}%"
        fi
        
        # Determine if fixes worked
        echo -e "\n🎯 FIX VALIDATION:"
        
        # For multi-threaded apps, CPU can be > 100% (100% per core)
        # With 24 threads, we expect at least 200% (2 cores) usage
        if (( $(echo "$AVG_CPU > 200" | bc -l) )); then
            echo "✅ CPU Usage: EXCELLENT (${AVG_CPU}% > 200%)"
        elif (( $(echo "$AVG_CPU > 100" | bc -l) )); then
            echo "✅ CPU Usage: GOOD (${AVG_CPU}% > 100%)"
        elif (( $(echo "$AVG_CPU > 50" | bc -l) )); then
            echo "⚠️  CPU Usage: MODERATE (${AVG_CPU}% > 50%)"
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
        elif (( $(echo "$AVG_GPU_UTIL > 30" | bc -l) )); then
            echo "⚠️  GPU Utilization: MODERATE (${AVG_GPU_UTIL}% > 30%)"
        else
            echo "❌ GPU Utilization: LOW (${AVG_GPU_UTIL}% < 30%)"
        fi
    fi
    
    echo -e "\nLogs saved to: $LOG_DIR/"
    
    # Show sample of raw data for debugging
    echo -e "\nSample performance data:"
    head -n 5 "$LOG_DIR/performance.csv"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Run the self-play with monitoring
echo -e "\n🎮 Starting self-play with config: $CONFIG_FILE"
echo "Running for $TEST_DURATION seconds..."
echo "=================================================="

# Kill any existing instances
pkill -f "omoknuni_cli_final" 2>/dev/null
sleep 2

# Run with timeout
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
timeout $TEST_DURATION ./build/bin/Release/omoknuni_cli_final self-play $CONFIG_FILE 2>&1 | tee "$LOG_DIR/output.log"

# Cleanup will be called automatically by trap