#!/bin/bash

# Unified MCTS Mode Runner
# Automatically selects best MCTS mode or allows manual selection

set -e  # Exit on any error

# Configuration
DEFAULT_CONFIG="config_optimized_minimal.yaml"
BINARY_PATH="./bin/Release/omoknuni_cli_final"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_highlight() {
    echo -e "${CYAN}[MCTS]${NC} $1"
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
    echo "  --mcts-mode MODE     Force MCTS mode: cpu, gpu, or auto (default: auto)"
    echo "  --verbose           Enable verbose output"
    echo "  --help              Show this help message"
    echo ""
    echo "MCTS Modes:"
    echo "  cpu     Standard CPU-based MCTS (maximum compatibility)"
    echo "  gpu     GPU-enhanced MCTS (requires CUDA-capable GPU)"
    echo "  auto    Automatically detect best available mode (default)"
    echo ""
    echo "Examples:"
    echo "  $0 self-play                           # Auto-detect best mode"
    echo "  $0 self-play --mcts-mode=gpu           # Force GPU mode"
    echo "  $0 self-play --mcts-mode=cpu           # Force CPU mode"
    echo "  $0 self-play config_low_latency.yaml  # Auto-detect with specific config"
    echo "  $0 train --verbose                    # Auto-detect with verbose output"
    echo ""
}

# Function to check GPU availability (quiet)
check_gpu_available() {
    command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null
}

# Function to determine best MCTS mode
determine_mcts_mode() {
    local mode="$1"
    
    if [[ "$mode" == "auto" ]]; then
        print_info "Auto-detecting best MCTS mode..."
        
        if check_gpu_available; then
            local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            if [[ $gpu_memory -ge 4096 ]]; then
                print_success "GPU detected with ${gpu_memory} MB memory - using GPU mode"
                echo "gpu"
            else
                print_warning "GPU detected but low memory (${gpu_memory} MB) - using CPU mode"
                echo "cpu"
            fi
        else
            print_info "No suitable GPU detected - using CPU mode"
            echo "cpu"
        fi
    else
        echo "$mode"
    fi
}

# Parse command line arguments
COMMAND="self-play"
CONFIG_FILE=""
MCTS_MODE="auto"
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
        --mcts-mode)
            MCTS_MODE="$2"
            shift 2
            ;;
        --mcts-mode=*)
            MCTS_MODE="${1#*=}"
            shift
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

# Validate MCTS mode
case "$MCTS_MODE" in
    cpu|gpu|auto)
        ;;
    *)
        print_error "Invalid MCTS mode: $MCTS_MODE"
        print_info "Valid modes: cpu, gpu, auto"
        exit 1
        ;;
esac

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

# Determine final MCTS mode
FINAL_MODE=$(determine_mcts_mode "$MCTS_MODE")

# Display run information
print_highlight "MCTS Configuration:"
print_info "Command: $COMMAND"
print_info "Config: $CONFIG_FILE"
print_info "Requested Mode: $MCTS_MODE"
print_info "Selected Mode: $FINAL_MODE"
print_info "Binary: $BINARY_PATH"

if [[ -n "$VERBOSE" ]]; then
    print_info "Verbose output: enabled"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    print_info "Extra arguments:$EXTRA_ARGS"
fi

echo ""

# Mode-specific information
case "$FINAL_MODE" in
    gpu)
        print_highlight "Using GPU-Enhanced MCTS:"
        print_info "  ✓ GPU batch evaluation"
        print_info "  ✓ GPU tree storage"
        print_info "  ✓ Shared evaluation server"
        print_info "  ✓ CUDA optimizations"
        ;;
    cpu)
        print_highlight "Using CPU-Based MCTS:"
        print_info "  ✓ Standard neural network evaluation"
        print_info "  ✓ CPU tree operations"
        print_info "  ✓ Maximum compatibility"
        ;;
esac

echo ""

# Build the command
CMD="$BINARY_PATH $COMMAND $CONFIG_FILE --mcts-mode=$FINAL_MODE $VERBOSE $EXTRA_ARGS"

print_info "Executing: $CMD"
echo ""

# Run the command
start_time=$(date +%s)

if eval "$CMD"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    print_success "MCTS execution completed successfully"
    print_info "Execution time: ${duration} seconds"
    print_info "Mode used: $FINAL_MODE"
    
    # Show GPU info if available and used
    if [[ "$FINAL_MODE" == "gpu" ]] && command -v nvidia-smi &> /dev/null; then
        print_info "Final GPU status:"
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | \
        awk '{printf "  Memory: %d/%d MB, Temperature: %d°C\n", $1, $2, $3}'
    fi
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    print_error "MCTS execution failed with exit code $?"
    print_info "Execution time: ${duration} seconds"
    print_info "Mode used: $FINAL_MODE"
    
    # Suggest fallback
    if [[ "$FINAL_MODE" == "gpu" ]]; then
        print_info ""
        print_info "Try fallback to CPU mode:"
        print_info "  $0 $COMMAND $CONFIG_FILE --mcts-mode=cpu $VERBOSE $EXTRA_ARGS"
    fi
    
    exit 1
fi