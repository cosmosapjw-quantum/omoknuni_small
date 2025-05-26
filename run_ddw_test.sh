#!/bin/bash

# Script to run DDW-RandWire-ResNet test with proper environment setup

# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# For CPU-only testing, disable CUDA
export CUDA_VISIBLE_DEVICES=""

echo "Running DDW-RandWire-ResNet test..."
echo "================================="

# First run minimal test
echo "1. Running minimal test..."
./build/bin/Release/test_ddw_minimal

if [ $? -eq 0 ]; then
    echo -e "\n✓ Minimal test passed\n"
    
    echo "2. Running full DDW test..."
    ./build/bin/Release/test_ddw_randwire_resnet
    
    if [ $? -eq 0 ]; then
        echo -e "\n✓ Full test passed"
    else
        echo -e "\n✗ Full test failed"
    fi
else
    echo -e "\n✗ Minimal test failed"
fi

# To run with GPU, comment out the CUDA_VISIBLE_DEVICES line above
# and run with:
# CUDA_VISIBLE_DEVICES=0 ./run_ddw_test.sh