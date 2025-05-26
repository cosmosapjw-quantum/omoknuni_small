#!/bin/bash

echo "Testing shutdown behavior..."
echo "Press Ctrl+C to test graceful shutdown"
echo ""

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Start the program and let it run for a bit
./build/bin/Release/omoknuni_cli_final self-play config_test_signal.yaml &
PID=$!

echo "Started process with PID: $PID"

# Monitor the process
while true; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process $PID has terminated"
        break
    fi
    sleep 1
done

# Check for any remaining processes
echo ""
echo "Checking for remaining processes..."
REMAINING=$(pgrep -f omoknuni_cli_final)
if [ -z "$REMAINING" ]; then
    echo "SUCCESS: No remaining omoknuni processes found"
else
    echo "WARNING: Found remaining processes: $REMAINING"
    ps aux | grep omoknuni_cli_final | grep -v grep
fi

echo "Test completed"