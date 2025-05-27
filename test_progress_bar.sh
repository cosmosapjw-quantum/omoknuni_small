#!/bin/bash
# Test script to demonstrate progress bar vs verbose logging

echo "Testing AlphaZero Self-Play with Progress Bar"
echo "============================================="
echo ""
echo "1. Running with progress bar (quiet mode):"
echo ""
./build/bin/Release/omoknuni_cli_final self-play config_ddw_balanced.yaml

echo ""
echo ""
echo "2. Running with verbose logging:"
echo ""
./build/bin/Release/omoknuni_cli_final self-play config_ddw_balanced.yaml --verbose