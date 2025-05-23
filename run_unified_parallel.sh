#!/bin/bash
# Run self-play with UNIFIED parallel generation for sustained GPU utilization

echo "🔥 Starting UNIFIED parallel self-play - 70%+ sustained GPU utilization!"
echo "======================================================================="
echo "This implementation uses a single GPU batch collector across ALL games"
echo ""

# Set environment for CUDA
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Run self-play with unified parallel configuration
./build/bin/Release/omoknuni_cli self-play --config config_unified_parallel.yaml

echo ""
echo "Unified parallel self-play completed."