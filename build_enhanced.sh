#!/bin/bash

# Build script for enhanced AlphaZero with ultra performance optimizations

echo "Building ENHANCED AlphaZero with all optimizations..."
echo "Features:"
echo "  - GPU memory pooling for zero-copy operations"
echo "  - Dynamic batch sizing based on queue depth"
echo "  - Advanced transposition table with cuckoo hashing"
echo "  - Multi-instance neural networks"
echo "  - Thread-local memory management"
echo ""

# Clean build directory
cd build
rm -rf *

# Configure with all optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DWITH_TORCH=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto -fvisibility=hidden -fopenmp" \
    -DCMAKE_CUDA_FLAGS="-O3 -arch=sm_86 --use_fast_math" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

# Build with all cores
cmake --build . --config Release --parallel $(nproc)

echo ""
echo "Build complete!"
echo ""
echo "To run the enhanced version with ultra performance:"
echo "  ./bin/Release/omoknuni_cli_enhanced self-play-enhanced --config ../config_enhanced_ultra_performance.yaml"
echo ""
echo "Monitor performance with:"
echo "  - nvidia-smi (GPU utilization)"
echo "  - htop (CPU utilization)"
echo "  - Check logs for detailed statistics"