#!/bin/bash

echo "🧹 Cleaning up stuck processes and memory..."

# Kill all omoknuni processes
echo "Killing omoknuni processes..."
for pid in $(pgrep -f omoknuni); do
    echo "Killing PID: $pid"
    kill -9 $pid 2>/dev/null
done

# Kill monitor processes
echo "Killing monitor processes..."
pkill -9 -f monitor_memory
pkill -9 -f "python.*monitor"

# Wait a moment
sleep 2

# Clear system caches (requires sudo)
echo "To clear system caches, run:"
echo "  sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"

# Show current memory usage
echo ""
echo "Current memory usage:"
free -h

echo ""
echo "GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check for any remaining processes
remaining=$(pgrep -f omoknuni | wc -l)
if [ $remaining -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: $remaining omoknuni processes still running"
    echo "You may need to run: sudo kill -9 \$(pgrep -f omoknuni)"
else
    echo ""
    echo "✅ All omoknuni processes terminated"
fi

# Python cleanup for stuck torch processes
echo ""
echo "If GPU memory is still held, you can run this Python script:"
echo "python3 -c \"import torch; torch.cuda.empty_cache(); print('GPU cache cleared')\""