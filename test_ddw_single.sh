#!/bin/bash

# Single worker test for DDW-RandWire-ResNet

echo "🧪 Testing DDW-RandWire-ResNet with single worker"
echo "================================================"

# Clean up any existing model
rm -f models/ddw_randwire_model.pt

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Run with single worker config
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release \
    ./build/bin/Release/omoknuni_cli_final self-play config_ddw_randwire_optimized.yaml \
    --num-parallel-workers 1 \
    --num-games 1

echo "✅ Test completed!"