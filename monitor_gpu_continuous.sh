#!/bin/bash
# Continuous GPU monitoring with 0.5 second intervals

echo "🔍 GPU Continuous Monitoring (0.5s intervals)"
echo "Press Ctrl+C to stop"
echo ""

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found"
    exit 1
fi

# Create log file
LOG_FILE="gpu_continuous_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Header
printf "%-10s %-6s %-8s %-8s %-8s %-8s %-10s %-8s\n" \
    "Time" "GPU%" "SM%" "MEM%" "TEMP" "POWER" "CLOCK" "BATCH" | tee "$LOG_FILE"
printf "%-10s %-6s %-8s %-8s %-8s %-8s %-10s %-8s\n" \
    "----" "----" "----" "----" "----" "-----" "-----" "-----" | tee -a "$LOG_FILE"

# Track idle periods
IDLE_COUNT=0
TOTAL_COUNT=0
LAST_GPU_UTIL=0

while true; do
    # Get current time
    TIME=$(date +%H:%M:%S.%1N)
    
    # Get detailed GPU stats
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits | head -1)
    
    GPU_UTIL=$(echo $GPU_STATS | awk -F', ' '{print $1}')
    MEM_UTIL=$(echo $GPU_STATS | awk -F', ' '{print $2}')
    GPU_MEM_USED=$(echo $GPU_STATS | awk -F', ' '{print $3}')
    GPU_MEM_TOTAL=$(echo $GPU_STATS | awk -F', ' '{print $4}')
    GPU_MEM_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))
    GPU_TEMP=$(echo $GPU_STATS | awk -F', ' '{print $5}')
    GPU_POWER=$(echo $GPU_STATS | awk -F', ' '{print $6}')
    GPU_CLOCK=$(echo $GPU_STATS | awk -F', ' '{print $7}')
    
    # Get SM utilization (Streaming Multiprocessor)
    SM_UTIL=$(nvidia-smi dmon -c 1 -s u 2>/dev/null | tail -1 | awk '{print $3}' || echo "N/A")
    
    # Count batches from log (if available)
    BATCH_COUNT=$(tail -100 /tmp/alphazero_gpu.log 2>/dev/null | grep -c "Batch" || echo "0")
    
    # Track idle periods
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if [ "$GPU_UTIL" -lt 10 ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
        MARKER="❌"
    elif [ "$GPU_UTIL" -ge 70 ]; then
        MARKER="✅"
    else
        MARKER="⚠️"
    fi
    
    # Print stats
    printf "%-10s %-6s %-8s %-8s %-8s %-8s %-10s %-8s %s\n" \
        "$TIME" "${GPU_UTIL}%" "${SM_UTIL}%" "${GPU_MEM_PERCENT}%" "${GPU_TEMP}°C" "${GPU_POWER}W" "${GPU_CLOCK}MHz" "$BATCH_COUNT" "$MARKER" | tee -a "$LOG_FILE"
    
    # Calculate idle percentage every 20 samples (10 seconds)
    if [ $((TOTAL_COUNT % 20)) -eq 0 ] && [ $TOTAL_COUNT -gt 0 ]; then
        IDLE_PERCENT=$((IDLE_COUNT * 100 / TOTAL_COUNT))
        echo "📊 GPU Idle Time: ${IDLE_PERCENT}% (${IDLE_COUNT}/${TOTAL_COUNT} samples)" | tee -a "$LOG_FILE"
        
        if [ $IDLE_PERCENT -lt 20 ]; then
            echo "✅ Excellent GPU utilization!" | tee -a "$LOG_FILE"
        elif [ $IDLE_PERCENT -lt 40 ]; then
            echo "⚠️  Moderate GPU idle time" | tee -a "$LOG_FILE"
        else
            echo "❌ High GPU idle time - optimization needed" | tee -a "$LOG_FILE"
        fi
    fi
    
    # Sleep for 0.5 seconds
    sleep 0.5
done