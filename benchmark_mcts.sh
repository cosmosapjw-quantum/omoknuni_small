#!/bin/bash

# MCTS Performance Benchmark Script
# Compare CPU vs GPU MCTS performance

set -e  # Exit on any error

# Configuration
DEFAULT_CONFIG="config_minimal_test.yaml"
BINARY_PATH="./bin/Release/omoknuni_cli_final"
BENCHMARK_SIMULATIONS=100

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

print_benchmark() {
    echo -e "${MAGENTA}[BENCHMARK]${NC} $1"
}

print_result() {
    echo -e "${CYAN}[RESULT]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE        Use specific config file (default: $DEFAULT_CONFIG)"
    echo "  --simulations N      Number of simulations per benchmark (default: $BENCHMARK_SIMULATIONS)"
    echo "  --command CMD        Command to benchmark (default: self-play)"
    echo "  --cpu-only          Only benchmark CPU mode"
    echo "  --gpu-only          Only benchmark GPU mode"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Full CPU vs GPU benchmark"
    echo "  $0 --simulations=200                 # More thorough benchmark"
    echo "  $0 --config=config_low_latency.yaml # Use specific config"
    echo "  $0 --cpu-only                       # CPU-only benchmark"
    echo ""
}

# Function to check GPU availability
check_gpu_available() {
    command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null
}

# Function to run benchmark
run_benchmark() {
    local mode="$1"
    local config="$2"
    local command="$3"
    
    print_benchmark "Running $mode MCTS benchmark..."
    print_info "Config: $config"
    print_info "Command: $command"
    print_info "Mode: $mode"
    
    local start_time=$(date +%s.%N)
    
    # Run the command and capture output
    local cmd="$BINARY_PATH $command $config --mcts-mode=$mode"
    local output_file="/tmp/mcts_benchmark_${mode}_$$.log"
    
    if timeout 300 $cmd > "$output_file" 2>&1; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        # Extract performance metrics from output
        local simulations=$(grep -o "simulations: [0-9]*" "$output_file" | tail -1 | cut -d' ' -f2 || echo "0")
        local evaluations=$(grep -o "evaluations: [0-9]*" "$output_file" | tail -1 | cut -d' ' -f2 || echo "0")
        local nodes=$(grep -o "nodes: [0-9]*" "$output_file" | tail -1 | cut -d' ' -f2 || echo "0")
        
        # Calculate performance metrics
        local sims_per_sec=$(echo "scale=2; $simulations / $duration" | bc)
        local evals_per_sec=$(echo "scale=2; $evaluations / $duration" | bc)
        
        print_success "$mode benchmark completed"
        print_result "Duration: ${duration}s"
        print_result "Simulations: $simulations (${sims_per_sec}/s)"
        print_result "Evaluations: $evaluations (${evals_per_sec}/s)"
        print_result "Nodes: $nodes"
        
        # GPU-specific metrics
        if [[ "$mode" == "gpu" ]] && command -v nvidia-smi &> /dev/null; then
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            print_result "GPU Utilization: ${gpu_util}%"
            print_result "GPU Memory: ${gpu_memory} MB"
        fi
        
        rm -f "$output_file"
        
        # Return results as space-separated values
        echo "$duration $simulations $evaluations $nodes $sims_per_sec $evals_per_sec"
    else
        print_error "$mode benchmark failed or timed out"
        rm -f "$output_file"
        echo "0 0 0 0 0 0"
    fi
}

# Parse command line arguments
CONFIG_FILE=""
SIMULATIONS=""
COMMAND="self-play"
CPU_ONLY=false
GPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --simulations=*)
            SIMULATIONS="${1#*=}"
            shift
            ;;
        --command)
            COMMAND="$2"
            shift 2
            ;;
        --command=*)
            COMMAND="${1#*=}"
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set defaults
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

if [[ -z "$SIMULATIONS" ]]; then
    SIMULATIONS="$BENCHMARK_SIMULATIONS"
fi

# Check prerequisites
if [[ ! -f "$BINARY_PATH" ]]; then
    print_error "Binary not found: $BINARY_PATH"
    print_info "Please build the project first"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check GPU availability
gpu_available=false
if check_gpu_available; then
    gpu_available=true
fi

if [[ "$GPU_ONLY" == true ]] && [[ "$gpu_available" == false ]]; then
    print_error "GPU-only benchmark requested but GPU not available"
    exit 1
fi

# Show benchmark configuration
print_benchmark "MCTS Performance Benchmark"
print_info "Config: $CONFIG_FILE"
print_info "Command: $COMMAND"
print_info "Target simulations: $SIMULATIONS"
print_info "GPU available: $gpu_available"

echo ""

# Results storage
declare -a results
declare -a modes

# Run CPU benchmark
if [[ "$GPU_ONLY" == false ]]; then
    print_benchmark "=== CPU MCTS Benchmark ==="
    cpu_results=$(run_benchmark "cpu" "$CONFIG_FILE" "$COMMAND")
    results+=("$cpu_results")
    modes+=("CPU")
    echo ""
fi

# Run GPU benchmark
if [[ "$CPU_ONLY" == false ]] && [[ "$gpu_available" == true ]]; then
    print_benchmark "=== GPU MCTS Benchmark ==="
    gpu_results=$(run_benchmark "gpu" "$CONFIG_FILE" "$COMMAND")
    results+=("$gpu_results")
    modes+=("GPU")
    echo ""
fi

# Summary comparison
if [[ ${#results[@]} -gt 1 ]]; then
    print_benchmark "=== Performance Comparison ==="
    
    # Parse results
    cpu_data=(${results[0]})
    gpu_data=(${results[1]})
    
    cpu_duration=${cpu_data[0]}
    cpu_simulations=${cpu_data[1]}
    cpu_sims_per_sec=${cpu_data[4]}
    
    gpu_duration=${gpu_data[0]}
    gpu_simulations=${gpu_data[1]}
    gpu_sims_per_sec=${gpu_data[4]}
    
    # Calculate speedup
    if [[ "$cpu_duration" != "0" ]] && [[ "$gpu_duration" != "0" ]]; then
        speedup=$(echo "scale=2; $cpu_duration / $gpu_duration" | bc)
        throughput_ratio=$(echo "scale=2; $gpu_sims_per_sec / $cpu_sims_per_sec" | bc)
        
        print_result "Performance Summary:"
        print_result "  CPU: ${cpu_duration}s (${cpu_sims_per_sec} sims/s)"
        print_result "  GPU: ${gpu_duration}s (${gpu_sims_per_sec} sims/s)"
        print_result "  Speedup: ${speedup}x"
        print_result "  Throughput ratio: ${throughput_ratio}x"
        
        if (( $(echo "$speedup > 1.5" | bc -l) )); then
            print_success "GPU mode shows significant performance improvement!"
        elif (( $(echo "$speedup > 1.1" | bc -l) )); then
            print_success "GPU mode shows moderate performance improvement"
        elif (( $(echo "$speedup > 0.9" | bc -l) )); then
            print_warning "GPU and CPU performance are similar"
        else
            print_warning "CPU mode is faster - check GPU configuration"
        fi
    fi
    
    echo ""
    print_info "Recommendation:"
    if (( $(echo "$speedup > 1.2" | bc -l) )); then
        print_success "Use GPU mode for better performance"
        print_info "  ./run_mcts.sh $COMMAND $CONFIG_FILE --mcts-mode=gpu"
    else
        print_info "Use CPU mode for maximum compatibility"
        print_info "  ./run_mcts.sh $COMMAND $CONFIG_FILE --mcts-mode=cpu"
    fi
fi

echo ""
print_benchmark "Benchmark completed"