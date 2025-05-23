#!/bin/bash
# Final optimized AlphaZero pipeline for consistent high performance

echo "🚀 Starting FINAL OPTIMIZED AlphaZero pipeline"
echo "✅ Based on working taskflow with pause fixes"
echo ""
echo "🔧 Key Optimizations:"
echo "   - Dynamic batch sizing (50-100% fill)"
echo "   - Adaptive timeouts (5-40ms)"
echo "   - Bulk dequeue operations"
echo "   - Yield instead of sleep"
echo "   - Optimized for 12 threads + 48 batch"
echo ""

# Environment for best performance on Ryzen 9 5900X + RTX 3060 Ti
export OMP_NUM_THREADS=6
export TORCH_NUM_THREADS=6
export MKL_NUM_THREADS=6
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Set library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# CPU affinity - use first 12 physical cores
export OMP_PROC_BIND=close
export OMP_PLACES="{0:12:1}"

# Set process priority
renice -n -5 $$ 2>/dev/null || true

# Run with ultra performance config
cd "$(dirname "$0")"
exec taskset -c 0-11 ./build/bin/Release/omoknuni_cli self-play --config config_ultra_performance.yaml "$@"