#!/bin/bash
# Performance monitoring script for AlphaZero

echo "🔍 Starting performance monitoring for AlphaZero"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found. GPU monitoring disabled."
    GPU_AVAILABLE=0
else
    GPU_AVAILABLE=1
fi

# Create log file
LOG_FILE="performance_monitor_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Header
printf "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" \
    "Time" "CPU%" "MEM%" "GPU%" "GPU_MEM" "GPU_TEMP" "POWER" "Processes" | tee "$LOG_FILE"
printf "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" \
    "----" "----" "----" "----" "-------" "--------" "-----" "---------" | tee -a "$LOG_FILE"

# Monitoring loop
while true; do
    # Get current time
    TIME=$(date +%H:%M:%S)
    
    # Get CPU usage (average across all cores)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Get memory usage
    MEM_USAGE=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
    
    # Get GPU stats if available
    if [ $GPU_AVAILABLE -eq 1 ]; then
        GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | head -1)
        GPU_UTIL=$(echo $GPU_STATS | awk -F', ' '{print $1}')
        GPU_MEM_USED=$(echo $GPU_STATS | awk -F', ' '{print $2}')
        GPU_MEM_TOTAL=$(echo $GPU_STATS | awk -F', ' '{print $3}')
        GPU_MEM_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))
        GPU_TEMP=$(echo $GPU_STATS | awk -F', ' '{print $4}')
        GPU_POWER=$(echo $GPU_STATS | awk -F', ' '{print $5}')
        
        # Count omoknuni processes
        PROC_COUNT=$(pgrep -f omoknuni_cli | wc -l)
    else
        GPU_UTIL="N/A"
        GPU_MEM_PERCENT="N/A"
        GPU_TEMP="N/A"
        GPU_POWER="N/A"
        PROC_COUNT=$(pgrep -f omoknuni_cli | wc -l)
    fi
    
    # Print stats
    printf "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" \
        "$TIME" "${CPU_USAGE}%" "${MEM_USAGE}%" "${GPU_UTIL}%" "${GPU_MEM_PERCENT}%" "${GPU_TEMP}°C" "${GPU_POWER}W" "$PROC_COUNT" | tee -a "$LOG_FILE"
    
    # Check for target utilization
    if [ "$GPU_AVAILABLE" -eq 1 ] && [ "$GPU_UTIL" != "N/A" ]; then
        if [ "${GPU_UTIL%.*}" -ge 70 ]; then
            echo "✅ Target GPU utilization (70%+) achieved!" | tee -a "$LOG_FILE"
        fi
    fi
    
    # Sleep for 2 seconds
    sleep 2
done