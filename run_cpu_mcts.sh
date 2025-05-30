#!/bin/bash

# CPU MCTS Mode Runner
# Runs MCTS with CPU-based evaluation (default mode)

set -e  # Exit on any error

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
DEFAULT_CONFIG="config_optimized_minimal.yaml"
BINARY_PATH="./build//bin/Release/omoknuni_cli_final"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [CONFIG_FILE] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  self-play    Run self-play training (default)"
    echo "  train        Run model training"
    echo "  eval         Run model evaluation"
    echo "  play         Interactive play mode"
    echo ""
    echo "Options:"
    echo "  --config FILE        Use specific config file (default: $DEFAULT_CONFIG)"
    echo "  --verbose           Enable verbose output"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 self-play                           # Run with default config"
    echo "  $0 self-play config_low_latency.yaml  # Run with specific config"
    echo "  $0 train --verbose                    # Run training with verbose output"
    echo "  $0 eval config_minimal_test.yaml      # Run evaluation"
    echo ""
    echo "Note: This script uses CPU-based MCTS evaluation for maximum compatibility."
}

# Parse command line arguments
COMMAND="self-play"
CONFIG_FILE=""
VERBOSE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        self-play|train|eval|play)
            COMMAND="$1"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *.yaml|*.yml)
            CONFIG_FILE="$1"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Set default config if not provided
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

# Check if binary exists
if [[ ! -f "$BINARY_PATH" ]]; then
    print_error "Binary not found: $BINARY_PATH"
    print_info "Please build the project first:"
    print_info "  cd build && cmake --build . --config Release --parallel"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Available config files:"
    ls -1 config*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

# Display run information
print_info "Running CPU MCTS Mode"
print_info "Command: $COMMAND"
print_info "Config: $CONFIG_FILE"
print_info "MCTS Mode: CPU (standard evaluation)"
print_info "Binary: $BINARY_PATH"

if [[ -n "$VERBOSE" ]]; then
    print_info "Verbose output: enabled"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    print_info "Extra arguments:$EXTRA_ARGS"
fi

echo ""

# Build the command
CMD="$BINARY_PATH $COMMAND $CONFIG_FILE --mcts-mode=cpu $VERBOSE $EXTRA_ARGS"

print_info "Executing: $CMD"
echo ""

# Run the command
if eval "$CMD"; then
    echo ""
    print_success "CPU MCTS execution completed successfully"
else
    echo ""
    print_error "CPU MCTS execution failed with exit code $?"
    exit 1
fi