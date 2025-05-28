#!/bin/bash

# Optimized self-play runner with aggressive memory management

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_MAXSIZE=536870912  # 512MB CUDA kernel cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Limit memory fragmentation

# Run with optimized settings
echo "Starting optimized self-play..."
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release ./build/bin/Release/omoknuni_cli_final self-play "$@"