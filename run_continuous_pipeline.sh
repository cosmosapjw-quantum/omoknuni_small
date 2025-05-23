#!/bin/bash
# Ultra-performance continuous pipeline for zero GPU idle time

echo "🚀 Starting CONTINUOUS PIPELINE AlphaZero"
echo "⚡ Zero GPU Idle Time Configuration"
echo ""
echo "🔧 Key Optimizations:"
echo "   - Double buffering for continuous GPU feed"
echo "   - No sleep operations, only yields"
echo "   - Bulk dequeue for efficiency"
echo "   - Reduced batch timeout for faster collection"
echo "   - Optimized thread priorities"
echo ""

# Environment for maximum performance
export OMP_NUM_THREADS=6
export TORCH_NUM_THREADS=6
export MKL_NUM_THREADS=6
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

# Set library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# CPU affinity - use physical cores only
export OMP_PROC_BIND=close
export OMP_PLACES="{0:12:1}"

# Set high priority
sudo renice -n -10 $$ 2>/dev/null || renice -n -5 $$ 2>/dev/null || true

# Clear GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "Optimizing GPU settings..."
    # Set GPU to persistence mode
    sudo nvidia-smi -pm 1 2>/dev/null || true
    # Set performance mode
    sudo nvidia-smi -pl 200 2>/dev/null || true
fi

# Run with continuous pipeline config
cd "$(dirname "$0")"
exec taskset -c 0-11 ./build/bin/Release/omoknuni_cli self-play --config config_ultra_performance.yaml "$@"