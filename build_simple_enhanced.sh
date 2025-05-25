#!/bin/bash

# Build script for simple enhanced version that works with existing code

echo "Building Simple Enhanced AlphaZero..."
echo "This version includes:"
echo "  - Multi-instance neural networks (already implemented)"
echo "  - Optimized self-play manager (already implemented)"
echo "  - Transposition table (existing implementation)"
echo "  - Thread affinity support"
echo ""

# Go to build directory
cd build

# Regenerate CMake files to pick up the new target
echo "Regenerating CMake files..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON -DBUILD_SHARED_LIBS=ON

# Just build the simple enhanced CLI target
echo "Building simple enhanced CLI..."
make omoknuni_cli_simple_enhanced -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "To run the simple enhanced version:"
    echo "  ./bin/Release/omoknuni_cli_simple_enhanced self-play --config ../config_simple_enhanced.yaml"
    echo ""
    echo "Or use the already working optimized version:"
    echo "  ./bin/Release/omoknuni_cli_optimized self-play-optimized --config ../config_optimized_true_parallel.yaml"
else
    echo ""
    echo "Build failed. The optimized version should still work:"
    echo "  ./bin/Release/omoknuni_cli_optimized self-play-optimized --config ../config_optimized_true_parallel.yaml"
fi