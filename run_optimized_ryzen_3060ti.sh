#!/bin/bash
# Optimized AlphaZero pipeline for Ryzen 9 5900X + RTX 3060 Ti

echo "🚀 Starting OPTIMIZED AlphaZero pipeline for Ryzen 9 5900X + RTX 3060 Ti"
echo "⚡ Hardware Configuration:"
echo "   - CPU: 12 cores / 24 threads @ 3.7-4.5 GHz"
echo "   - GPU: RTX 3060 Ti with 8GB VRAM, 4864 CUDA cores"
echo "   - RAM: 64GB"
echo ""
echo "🔧 Optimizations Applied:"
echo "   - 8 MCTS threads (avoid thread contention)"
echo "   - 32 batch size (optimal for 8GB VRAM)"
echo "   - Limited OpenMP threads for tensor prep"
echo "   - CUDA streams for async transfers"
echo "   - Thread-local tensor caching"
echo "   - CPU affinity optimization"
echo ""

# Set environment variables optimized for your hardware
# MCTS uses 8 threads, limit OpenMP to 4 to avoid contention
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Enable CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# PyTorch specific optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Set library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# CPU affinity settings for Ryzen 9 5900X
# Use physical cores 0-7 for MCTS threads (avoiding CCX boundaries)
export OMP_PROC_BIND=close
export OMP_PLACES="{0:8}"

# Set process priority
renice -n -5 $$ 2>/dev/null || true

# Clear GPU memory before starting
if command -v nvidia-smi &> /dev/null; then
    echo "Clearing GPU memory..."
    nvidia-smi --gpu-reset 2>/dev/null || true
fi

# Run with optimized config
cd "$(dirname "$0")"
exec taskset -c 0-15 ./build/bin/Release/omoknuni_cli self-play --config config_optimized_fixed.yaml "$@"