#\!/bin/bash

echo "🚀 Testing Enhanced Simplified Parallel Search - Optimized Configuration"
echo "Target: 28-32 state batches, 20%+ GPU efficiency, sustained performance"
echo ""

cd build

# Clean old model to force regeneration with new parameters
rm -f ../models/model.pt

# Set library path
export LD_LIBRARY_PATH=./lib/Release:$LD_LIBRARY_PATH

# Run with optimized config - test single game first
echo "🎯 Testing single game with enhanced batching..."
timeout 120 ./bin/Release/omoknuni_cli self-play --config ../config.yaml --num-games 1 2>&1  < /dev/null |  tee /tmp/optimized_test.log

echo ""
echo "📊 BATCH PERFORMANCE SUMMARY:"
echo "=============================="
grep -E "(BATCH-.*🚀.*states.*→)" /tmp/optimized_test.log | head -10

echo ""
echo "🎯 GPU EFFICIENCY ANALYSIS:"
echo "==========================="
grep -E "GPU-Eff.*[1-9][0-9]" /tmp/optimized_test.log | head -5

echo ""
echo "⚡ TOP THROUGHPUT RESULTS:"
echo "=========================="
grep -E "Throughput.*[2-9]\\.[0-9]" /tmp/optimized_test.log | head -5
