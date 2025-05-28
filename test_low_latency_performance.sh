#!/bin/bash

# Test script for low-latency performance with optimized settings

echo "Building with optimizations..."
cd build
cmake --build . --config Release --parallel $(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

cd ..

echo -e "\n========================================="
echo "Testing Low Latency Performance (<100ms target)"
echo "========================================="

# Test 1: Standard batch tree (baseline)
echo -e "\n1. Testing standard batch tree (baseline):"
./build/bin/Release/omoknuni_cli_final self-play \
    --config config_alphazero_train.yaml \
    --num-games 5 \
    --parallel-games 1 \
    --verbose 2>&1 | grep -E "Move time:|Batch #|states processed|GPU-Eff|Throughput" | head -20

# Test 2: Low latency configuration
echo -e "\n2. Testing low latency configuration:"
./build/bin/Release/omoknuni_cli_final self-play \
    --config config_low_latency.yaml \
    --num-games 5 \
    --parallel-games 1 \
    --verbose 2>&1 | grep -E "Move time:|UltraFastBatch|states processed|sims/sec|Timing -" | head -20

# Test 3: Benchmark move generation speed
echo -e "\n3. Benchmarking move generation speed:"
./build/bin/Release/omoknuni_cli_final benchmark \
    --config config_low_latency.yaml \
    --num-positions 100 \
    --verbose 2>&1 | grep -E "Average move time:|Min:|Max:|Median:|throughput"

echo -e "\n========================================="
echo "Performance Test Complete"
echo "========================================="