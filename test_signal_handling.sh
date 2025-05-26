#!/bin/bash

echo "Testing signal handling in omoknuni_cli_final..."
echo "This test will:"
echo "1. Start the program"
echo "2. Send Ctrl+C after 5 seconds"
echo "3. Check if the process terminates cleanly"
echo ""

# Function to check if process is still running
check_process() {
    if ps -p $1 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start the program in background
echo "Starting omoknuni_cli_final..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
./build/bin/Release/omoknuni_cli_final self-play config_test_signal.yaml &
PID=$!

echo "Process started with PID: $PID"

# Wait a bit for the program to start
sleep 5

# Send first SIGINT (Ctrl+C)
echo ""
echo "Sending first SIGINT (Ctrl+C) to process $PID..."
kill -INT $PID

# Wait and check if process is shutting down gracefully
sleep 3

if check_process $PID; then
    echo "Process still running after first SIGINT. Sending second SIGINT..."
    kill -INT $PID
    sleep 2
    
    if check_process $PID; then
        echo "ERROR: Process still running after second SIGINT!"
        echo "Force killing process..."
        kill -9 $PID
        exit 1
    else
        echo "SUCCESS: Process terminated after second SIGINT"
    fi
else
    echo "SUCCESS: Process terminated gracefully after first SIGINT"
fi

# Check for any remaining omoknuni processes
echo ""
echo "Checking for any remaining omoknuni processes..."
REMAINING=$(pgrep -f omoknuni_cli_final)
if [ -z "$REMAINING" ]; then
    echo "SUCCESS: No remaining omoknuni processes found"
else
    echo "WARNING: Found remaining processes: $REMAINING"
    echo "Process details:"
    ps aux | grep omoknuni_cli_final | grep -v grep
fi

# Check GPU memory usage
echo ""
echo "Checking GPU memory..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

echo ""
echo "Signal handling test completed"