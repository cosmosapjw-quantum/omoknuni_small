#\!/bin/bash

echo "🛡️  Testing Memory-Safe Enhanced Parallel Search"
echo "Target: 20-24 state batches, sustained operation without OOM"
echo ""

cd build

# Clean everything to start fresh
rm -f ../models/model.pt
rm -f /tmp/memory_safe_test.log

# Set library path
export LD_LIBRARY_PATH=./lib/Release:$LD_LIBRARY_PATH

# Monitor memory during execution
echo "🎯 Testing memory-safe configuration with monitoring..."
timeout 300 ./bin/Release/omoknuni_cli self-play --config ../config.yaml --num-games 3 2>&1  < /dev/null |  tee /tmp/memory_safe_test.log

echo ""
echo "📊 BATCH PERFORMANCE SUMMARY:"
echo "=============================="
grep -E "(BATCH-.*🚀.*states.*→)" /tmp/memory_safe_test.log | head -10

echo ""
echo "🛡️  MEMORY SAFETY ANALYSIS:"
echo "==========================="
grep -E "(Memory cleanup|GOOD GPU|Slow GPU)" /tmp/memory_safe_test.log | head -5

echo ""
echo "⚡ THROUGHPUT RESULTS:"
echo "====================="
grep -E "Throughput.*[1-9]\\.[0-9]" /tmp/memory_safe_test.log | head -5
