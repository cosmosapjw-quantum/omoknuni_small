#!/bin/bash

# Test script specifically for VRAM leak validation
echo "🚀 Testing VRAM memory leak fix..."
echo "=================================="

# Configuration
TEST_DURATION=60  # Run for 1 minute
CONFIG_FILE="config_fixed_memory_cpu.yaml"

# Function to get VRAM usage
get_vram_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

# Initial VRAM reading
INITIAL_VRAM=$(get_vram_mb)
echo "Initial VRAM: ${INITIAL_VRAM}MB"

# Start monitoring in background
(
    while true; do
        VRAM=$(get_vram_mb)
        echo "[$(date +%H:%M:%S)] VRAM: ${VRAM}MB"
        sleep 2
    done
) &
MONITOR_PID=$!

# Run self-play
echo -e "\nStarting self-play test..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
timeout $TEST_DURATION ./build/bin/Release/omoknuni_cli_final self-play $CONFIG_FILE 2>&1 | grep -E "(VRAM|GPU|Memory|game|TASKFLOW)" &
SELFPLAY_PID=$!

# Wait for test to complete
wait $SELFPLAY_PID

# Stop monitor
kill $MONITOR_PID 2>/dev/null

# Final VRAM reading
FINAL_VRAM=$(get_vram_mb)
echo -e "\n=================================="
echo "VRAM Usage Summary:"
echo "Initial: ${INITIAL_VRAM}MB"
echo "Final: ${FINAL_VRAM}MB"
VRAM_GROWTH=$((FINAL_VRAM - INITIAL_VRAM))
echo "Growth: ${VRAM_GROWTH}MB"

if [ $VRAM_GROWTH -gt 500 ]; then
    echo "❌ VRAM LEAK DETECTED: Growth > 500MB"
else
    echo "✅ VRAM STABLE: Growth < 500MB"
fi