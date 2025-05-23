#\!/bin/bash

echo "🚨 Testing Aggressive Memory Control System"
echo "Features: Real-time monitoring, adaptive batching, emergency cleanup"
echo ""

cd build

# Clean start
rm -f ../models/model.pt
rm -f /tmp/memory_controlled_test.log

# Set library path
export LD_LIBRARY_PATH=./lib/Release:$LD_LIBRARY_PATH

# Monitor memory usage in background
echo "💾 Starting memory monitoring..."
echo "Time(s),Memory(GB),Pressure" > /tmp/memory_monitor.csv
(
    for i in {1..300}; do
        mem_kb=$(cat /proc/self/status  < /dev/null |  grep VmRSS | awk '{print $2}')
        mem_gb=$(echo "scale=2; $mem_kb / 1024 / 1024" | bc -l)
        echo "$i,$mem_gb,0" >> /tmp/memory_monitor.csv
        sleep 1
    done
) &
MONITOR_PID=$\!

echo "🎯 Testing with increased configuration and memory control..."
timeout 180 ./bin/Release/omoknuni_cli self-play --config ../config.yaml --num-games 10 2>&1 | tee /tmp/memory_controlled_test.log

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "📊 BATCH PERFORMANCE WITH MEMORY CONTROL:"
echo "=========================================="
grep -E "(BATCH-.*🚀.*states.*→)" /tmp/memory_controlled_test.log | head -15

echo ""
echo "🚨 MEMORY PRESSURE EVENTS:"
echo "=========================="
grep -E "(MEMORY PRESSURE|EMERGENCY|CRITICAL|⚠️|🚨|🔥)" /tmp/memory_controlled_test.log | head -10

echo ""
echo "💾 MEMORY CLEANUP EFFECTIVENESS:"
echo "==============================="
grep -E "(Memory cleanup completed|🧹)" /tmp/memory_controlled_test.log | head -5

echo ""
echo "🎯 ADAPTIVE BATCH SIZE CHANGES:"
echo "==============================="
grep -E "(Batch size reduced|OPTIMAL)" /tmp/memory_controlled_test.log | head -5
