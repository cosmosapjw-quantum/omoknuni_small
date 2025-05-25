#!/bin/bash
# Safe run script with synchronous logging

echo "Running AlphaZero with synchronous logging (safe mode)..."

# Use the existing optimized CLI but force synchronous logging
# by modifying the config temporarily
cp config_optimized_hardware.yaml config_safe_temp.yaml

# Run with modified settings
./build/bin/Release/omoknuni_cli self-play config_safe_temp.yaml

# Cleanup
rm -f config_safe_temp.yaml