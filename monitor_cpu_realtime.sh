#!/bin/bash

# Real-time CPU monitoring script
echo "🔍 Real-time CPU Monitor for omoknuni_cli_final"
echo "============================================="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get all omoknuni processes
    PIDS=$(pgrep -f "omoknuni_cli_final self-play")
    
    if [ ! -z "$PIDS" ]; then
        for PID in $PIDS; do
            # Method 1: Using ps
            PS_CPU=$(ps -p $PID -o %cpu= 2>/dev/null | tr -d ' ')
            
            # Method 2: Using top (more accurate for multi-threaded)
            TOP_OUTPUT=$(top -b -n 1 -p $PID 2>/dev/null | tail -1)
            TOP_CPU=$(echo "$TOP_OUTPUT" | awk '{print $9}')
            
            # Method 3: Calculate from /proc/stat (most accurate)
            if [ -f /proc/$PID/stat ]; then
                STAT=($(cat /proc/$PID/stat))
                UTIME=${STAT[13]}
                STIME=${STAT[14]}
                START_TIME=${STAT[21]}
                
                # Get system uptime and CPU count
                UPTIME=$(cat /proc/uptime | awk '{print $1}')
                CPU_COUNT=$(nproc)
                
                # Calculate CPU usage
                TOTAL_TIME=$((UTIME + STIME))
                SECONDS=$(($(date +%s) - START_TIME/100))
                if [ $SECONDS -gt 0 ]; then
                    CPU_PERCENT=$(echo "scale=2; $TOTAL_TIME / $SECONDS / $CPU_COUNT" | bc)
                fi
            fi
            
            # Get thread count
            THREADS=$(ls /proc/$PID/task 2>/dev/null | wc -l)
            
            # Get memory usage
            MEM_KB=$(ps -p $PID -o rss= 2>/dev/null)
            MEM_MB=$((MEM_KB / 1024))
            
            # Display
            printf "[%s] PID: %-8s | Threads: %-3s | CPU(ps): %-6s%% | CPU(top): %-6s%% | RAM: %-6sMB\n" \
                "$(date +%H:%M:%S)" "$PID" "$THREADS" "$PS_CPU" "$TOP_CPU" "$MEM_MB"
        done
    else
        echo "[$(date +%H:%M:%S)] No omoknuni_cli_final process found"
    fi
    
    sleep 1
done