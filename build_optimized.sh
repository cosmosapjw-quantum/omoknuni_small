#!/bin/bash

# Build script for optimized AlphaZero with true parallel MCTS

echo "Building optimized AlphaZero with independent neural networks..."

# Clean build directory
cd build
rm -rf *

# Configure with optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DWITH_TORCH=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto -fvisibility=hidden" \
    -DCMAKE_CUDA_FLAGS="-O3 -arch=sm_86" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON

# Build with all cores
cmake --build . --config Release --parallel $(nproc)

echo "Build complete!"
echo ""
echo "To run the optimized version:"
echo "  ./bin/Release/omoknuni_cli_optimized self-play-optimized --config ../config_optimized_true_parallel.yaml"
echo ""
echo "Key optimizations:"
echo "  - Independent neural network instances per MCTS engine"
echo "  - Removed excessive CUDA synchronization"
echo "  - Thread-local memory management"
echo "  - Async inference pipeline"
echo "  - CPU thread affinity"
echo "  - Optimized tensor buffers"