#!/bin/bash
# Quick rebuild script after removing verbose logging

cd build
echo "Rebuilding with reduced logging..."
cmake --build . --config Release --parallel 8

echo "Build complete!"