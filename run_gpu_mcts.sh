#!/bin/bash

# GPU MCTS Mode Runner
# Runs MCTS with GPU-enhanced evaluation and tree operations

set -e  # Exit on any error

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
DEFAULT_CONFIG="config_optimized_minimal.yaml"
BINARY_PATH="./build/bin/Release/omoknuni_cli_final"

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
    echo "GPU Requirements:"
    echo "  - CUDA-capable GPU (Compute Capability 7.0+)"
    echo "  - Sufficient GPU memory (recommended: 8GB+)"
    echo "  - CUDA 11.7+ and cuDNN installed"
    echo ""
    echo "Note: This script uses GPU-enhanced MCTS for improved performance."
}

# Function to check GPU availability
check_gpu() {
    print_info "Checking GPU availability..."
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi
    
    # Check GPU status
    if nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $gpu_count GPU(s) available"
        print_info "GPU Memory: ${gpu_memory} MB"
        
        # Check memory
        if [[ $gpu_memory -lt 4096 ]]; then
            print_warning "GPU memory is low (<4GB). Consider reducing batch size."
        fi
        return 0
    else
        print_error "GPU not accessible. nvidia-smi failed."
        return 1
    fi
}

# Function to check CUDA
check_cuda() {
    print_info "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        print_success "CUDA detected: version $cuda_version"
        return 0
    else
        print_warning "CUDA compiler (nvcc) not found in PATH"
        return 1
    fi
}

# Parse command line arguments
COMMAND="self-play"
CONFIG_FILE=""
VERBOSE=""
EXTRA_ARGS=""
SKIP_GPU_CHECK=false

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
        --skip-gpu-check)
            SKIP_GPU_CHECK=true
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
    print_info "Please build the project first with CUDA support:"
    print_info "  cd build"
    print_info "  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON"
    print_info "  cmake --build . --config Release --parallel"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Available config files:"
    ls -1 config*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

# GPU checks (unless skipped)
if [[ "$SKIP_GPU_CHECK" == false ]]; then
    print_info "Performing GPU compatibility checks..."
    
    gpu_available=true
    if ! check_gpu; then
        gpu_available=false
    fi
    
    if ! check_cuda; then
        print_warning "CUDA not detected. GPU features may be limited."
    fi
    
    if [[ "$gpu_available" == false ]]; then
        print_error "GPU not available or not accessible."
        print_info "You can:"
        print_info "  1. Use CPU mode instead: ./run_cpu_mcts.sh $COMMAND $CONFIG_FILE"
        print_info "  2. Skip GPU check: $0 $COMMAND $CONFIG_FILE --skip-gpu-check"
        print_info "  3. Fix GPU setup and try again"
        exit 1
    fi
    
    echo ""
fi

# Display run information
print_info "Running GPU MCTS Mode"
print_info "Command: $COMMAND"
print_info "Config: $CONFIG_FILE"
print_info "MCTS Mode: GPU (enhanced evaluation + tree operations)"
print_info "Binary: $BINARY_PATH"

if [[ -n "$VERBOSE" ]]; then
    print_info "Verbose output: enabled"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    print_info "Extra arguments:$EXTRA_ARGS"
fi

echo ""

# Build the command
CMD="$BINARY_PATH $COMMAND $CONFIG_FILE --mcts-mode=gpu $VERBOSE $EXTRA_ARGS"

print_info "Executing: $CMD"
echo ""

# Run the command
if eval "$CMD"; then
    echo ""
    print_success "GPU MCTS execution completed successfully"
    
    # Show GPU memory usage after completion
    if command -v nvidia-smi &> /dev/null && [[ "$SKIP_GPU_CHECK" == false ]]; then
        print_info "Final GPU memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | \
        awk '{printf "  Used: %d MB / Total: %d MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    fi
else
    echo ""
    print_error "GPU MCTS execution failed with exit code $?"
    
    # Show GPU status on failure
    if command -v nvidia-smi &> /dev/null && [[ "$SKIP_GPU_CHECK" == false ]]; then
        print_info "GPU status after failure:"
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | \
        awk '{printf "  Memory: %d/%d MB, Temperature: %d°C\n", $1, $2, $3}'
    fi
    
    exit 1
fi