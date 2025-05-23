#!/bin/bash
# High throughput AlphaZero pipeline with optimizations

echo "🚀 Starting HIGH THROUGHPUT AlphaZero pipeline..."
echo "⚡ Target: 70%+ CPU/GPU utilization"
echo "🔧 Optimizations:"
echo "   - Async memory cleanup (no blocking)"
echo "   - 16 MCTS threads with 64 batch size"
echo "   - 25ms batch timeout for rapid processing"
echo "   - Burst mode with adaptive batching"
echo ""

# Set environment variables for performance
export OMP_NUM_THREADS=16
export TORCH_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Set thread affinity for better performance
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Run with high throughput config
cd "$(dirname "$0")"
./build/bin/Release/omoknuni_cli self-play --config config_high_throughput.yaml "$@"