#!/bin/bash
# Enhanced AlphaZero Training Pipeline with GPU MCTS Support

set -e  # Exit on any error

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Default values
CONFIG_FILE="${1:-config_alphazero_train.yaml}"
VERBOSE="${2:-false}"
RESUME_FROM_CHECKPOINT="${3:-true}"
MCTS_MODE="${4:-auto}"  # auto, cpu, gpu

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

print_phase() {
    echo -e "${MAGENTA}[PHASE]${NC} $1"
}

print_training() {
    echo -e "${CYAN}[TRAINING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [CONFIG_FILE] [VERBOSE] [RESUME_FROM_CHECKPOINT] [MCTS_MODE]"
    echo ""
    echo "Parameters:"
    echo "  CONFIG_FILE            Configuration file (default: config_alphazero_train.yaml)"
    echo "  VERBOSE                Enable verbose output: true/false (default: false)"
    echo "  RESUME_FROM_CHECKPOINT Resume from checkpoint: true/false (default: true)"
    echo "  MCTS_MODE             MCTS mode: auto/cpu/gpu (default: auto)"
    echo ""
    echo "MCTS Modes:"
    echo "  auto    Automatically detect best available mode"
    echo "  cpu     Force CPU-based MCTS (maximum compatibility)"
    echo "  gpu     Force GPU-enhanced MCTS (requires CUDA)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use defaults with auto MCTS mode"
    echo "  $0 config.yaml                       # Use specific config with auto mode"
    echo "  $0 config.yaml true                  # Enable verbose output"
    echo "  $0 config.yaml true true gpu         # Force GPU MCTS mode"
    echo "  $0 config.yaml false false cpu       # Force CPU MCTS mode"
    echo ""
    echo "Alternative usage with flags:"
    echo "  $0 --config=config.yaml --mcts-mode=gpu --verbose --no-resume"
    echo ""
}

# Parse alternative flag-based arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mcts-mode=*)
            MCTS_MODE="${1#*=}"
            shift
            ;;
        --mcts-mode)
            MCTS_MODE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --no-verbose)
            VERBOSE="false"
            shift
            ;;
        --resume)
            RESUME_FROM_CHECKPOINT="true"
            shift
            ;;
        --no-resume)
            RESUME_FROM_CHECKPOINT="false"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Handle positional arguments
            if [[ -z "$CONFIG_FILE" || "$CONFIG_FILE" == "config_alphazero_train.yaml" ]]; then
                CONFIG_FILE="$1"
            elif [[ "$VERBOSE" == "false" ]]; then
                VERBOSE="$1"
            elif [[ "$RESUME_FROM_CHECKPOINT" == "true" ]]; then
                RESUME_FROM_CHECKPOINT="$1"
            elif [[ "$MCTS_MODE" == "auto" ]]; then
                MCTS_MODE="$1"
            fi
            shift
            ;;
    esac
done

# Validate MCTS mode
case "$MCTS_MODE" in
    auto|cpu|gpu)
        ;;
    *)
        print_error "Invalid MCTS mode: $MCTS_MODE"
        print_info "Valid modes: auto, cpu, gpu"
        exit 1
        ;;
esac

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Function to check GPU availability
check_gpu_available() {
    command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null
}

# Function to determine best MCTS mode
determine_mcts_mode() {
    local mode="$1"
    
    if [[ "$mode" == "auto" ]]; then
        print_info "Auto-detecting best MCTS mode for training..."
        
        if check_gpu_available; then
            local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            if [[ $gpu_memory -ge 6144 ]]; then  # 6GB minimum for training
                print_success "GPU detected with ${gpu_memory} MB memory - using GPU mode for training"
                echo "gpu"
            else
                print_warning "GPU detected but insufficient memory (${gpu_memory} MB < 6GB) - using CPU mode"
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

# Determine final MCTS mode
FINAL_MCTS_MODE=$(determine_mcts_mode "$MCTS_MODE")

# Parse configuration values
EXPERIMENT_NAME=$(grep "experiment_name:" "$CONFIG_FILE" | awk '{print $2}')
CHECKPOINT_DIR=$(grep "checkpoint_dir:" "$CONFIG_FILE" | awk '{print $2}')
LOG_DIR=$(grep "log_dir:" "$CONFIG_FILE" | awk '{print $2}')
TENSORBOARD_DIR=$(grep "tensorboard_dir:" "$CONFIG_FILE" | awk '{print $2}')
NUM_ITERATIONS=$(grep "num_iterations:" "$CONFIG_FILE" | awk '{print $2}')
GAMES_PER_ITERATION=$(grep "games_per_iteration:" "$CONFIG_FILE" | awk '{print $2}')
CHECKPOINT_INTERVAL=$(grep "checkpoint_interval:" "$CONFIG_FILE" | awk '{print $2}')
KEEP_CHECKPOINT_MAX=$(grep "keep_checkpoint_max:" "$CONFIG_FILE" | awk '{print $2}')
BOARD_SIZE=$(grep "board_size:" "$CONFIG_FILE" | awk '{print $2}')
INPUT_CHANNELS=$(grep "input_channels:" "$CONFIG_FILE" | awk '{print $2}')

# Set defaults if not in config
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-5}
KEEP_CHECKPOINT_MAX=${KEEP_CHECKPOINT_MAX:-10}

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR" "data/training"

# Set up logging with MCTS mode
LOG_FILE="$LOG_DIR/alphazero_training_${FINAL_MCTS_MODE}_$(date +%Y%m%d_%H%M%S).log"

# Check for existing checkpoint to resume from
START_ITERATION=1
if [ "$RESUME_FROM_CHECKPOINT" = "true" ] && [ -d "$CHECKPOINT_DIR" ]; then
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -1 "$CHECKPOINT_DIR"/checkpoint_iter_*.json 2>/dev/null | sort -V | tail -n 1)
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        # Extract iteration number from checkpoint filename
        CHECKPOINT_ITER=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint_iter_\([0-9]*\).*/\1/')
        if [ ! -z "$CHECKPOINT_ITER" ]; then
            START_ITERATION=$((CHECKPOINT_ITER + 1))
            print_success "Found checkpoint at iteration $CHECKPOINT_ITER"
            print_info "Resuming from iteration $START_ITERATION"
        fi
    fi
fi

# Display training configuration
print_training "Enhanced AlphaZero Training Pipeline"
echo "================================="
print_info "Experiment: $EXPERIMENT_NAME"
print_info "Start Iteration: $START_ITERATION"
print_info "Total Iterations: $NUM_ITERATIONS"
print_info "Games per iteration: $GAMES_PER_ITERATION"
print_info "MCTS Mode: $FINAL_MCTS_MODE"
print_info "Checkpoints: $CHECKPOINT_DIR"
print_info "Checkpoint interval: $CHECKPOINT_INTERVAL"
print_info "Max checkpoints: $KEEP_CHECKPOINT_MAX"
print_info "Tensorboard: $TENSORBOARD_DIR"
print_info "Log file: $LOG_FILE"
echo "================================="

# Mode-specific information
case "$FINAL_MCTS_MODE" in
    gpu)
        print_training "GPU-Enhanced Training Configuration:"
        print_info "  ✓ GPU batch evaluation for faster self-play"
        print_info "  ✓ GPU tree operations for improved throughput"
        print_info "  ✓ Shared evaluation server for efficiency"
        print_info "  ✓ CUDA optimizations enabled"
        
        # Show GPU status
        if command -v nvidia-smi &> /dev/null; then
            print_info "GPU Status:"
            nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | \
            awk '{printf "  GPU: %s, Memory: %d MB, Temp: %d°C\n", $1, $2, $3}'
        fi
        ;;
    cpu)
        print_training "CPU-Based Training Configuration:"
        print_info "  ✓ Standard neural network evaluation"
        print_info "  ✓ CPU tree operations"
        print_info "  ✓ Maximum compatibility mode"
        print_info "  ✓ Stable and reliable execution"
        ;;
esac

echo ""

# Start TensorBoard in background
print_info "Starting TensorBoard..."
tensorboard --logdir="$TENSORBOARD_DIR" --port 6006 --bind_all &
TB_PID=$!
print_success "TensorBoard running at http://localhost:6006"

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Shutting down..."
    if [ ! -z "$TB_PID" ]; then
        kill $TB_PID 2>/dev/null
    fi
    exit
}
trap cleanup INT TERM

# Function to trace model for TorchScript optimization
trace_model() {
    local model_path=$1
    local traced_path="traced_resnet_${BOARD_SIZE}.pt"
    
    # Check if model is already a TorchScript model
    if python3 -c "import torch; torch.jit.load('$model_path'); print('Already TorchScript')" 2>/dev/null | grep -q "Already TorchScript"; then
        print_success "Model is already in TorchScript format"
        # Copy to traced path if different
        if [ "$model_path" != "$traced_path" ]; then
            cp "$model_path" "$traced_path"
            print_info "Copied TorchScript model to: $traced_path"
        fi
        return 0
    fi
    
    print_info "Tracing model for TorchScript optimization..."
    
    if [ -f "$model_path" ]; then
        python3 trace_model_for_cpp.py \
            --model "$model_path" \
            --output "$traced_path" \
            --board-size "$BOARD_SIZE" \
            --input-channels "$INPUT_CHANNELS" \
            --batch-size 1 \
            --device cuda 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ] && [ -f "$traced_path" ]; then
            print_success "Model traced successfully: $traced_path"
            return 0
        else
            print_warning "Model tracing failed, continuing without TorchScript optimization"
            return 1
        fi
    else
        print_warning "Model file not found: $model_path"
        return 1
    fi
}

# Function to save checkpoint
save_checkpoint() {
    local iteration=$1
    local checkpoint_file="$CHECKPOINT_DIR/checkpoint_iter_${iteration}.json"
    local model_file="$CHECKPOINT_DIR/model_iter_${iteration}.pt"
    
    print_info "Saving checkpoint at iteration $iteration"
    
    # Create checkpoint metadata
    cat > "$checkpoint_file" <<EOF
{
    "iteration": $iteration,
    "timestamp": "$(date -Iseconds)",
    "experiment_name": "$EXPERIMENT_NAME",
    "mcts_mode": "$FINAL_MCTS_MODE",
    "total_games": $((iteration * GAMES_PER_ITERATION)),
    "model_path": "$model_file",
    "config_file": "$CONFIG_FILE"
}
EOF
    
    # Copy the current model
    if [ -f "models/model.pt" ]; then
        cp "models/model.pt" "$model_file"
        
        # Trace model for TorchScript optimization
        trace_model "$model_file"
    fi
    
    # Clean up old checkpoints if exceeding max
    local checkpoint_count=$(ls -1 "$CHECKPOINT_DIR"/checkpoint_iter_*.json 2>/dev/null | wc -l)
    if [ $checkpoint_count -gt $KEEP_CHECKPOINT_MAX ]; then
        print_info "Cleaning old checkpoints (keeping latest $KEEP_CHECKPOINT_MAX)"
        ls -1t "$CHECKPOINT_DIR"/checkpoint_iter_*.json | tail -n +$((KEEP_CHECKPOINT_MAX + 1)) | while read f; do
            base=$(basename "$f" .json)
            rm -f "$f" "$CHECKPOINT_DIR/${base}.pt"
            print_info "  Removed: $base"
        done
    fi
}

# Function to load checkpoint
load_checkpoint() {
    local checkpoint_file=$1
    if [ -f "$checkpoint_file" ]; then
        local model_path=$(grep '"model_path"' "$checkpoint_file" | cut -d'"' -f4)
        if [ -f "$model_path" ]; then
            print_info "Loading model from checkpoint: $model_path"
            cp "$model_path" "models/model.pt"
            return 0
        fi
    fi
    return 1
}

# Load checkpoint if resuming
if [ $START_ITERATION -gt 1 ]; then
    PREV_ITERATION=$((START_ITERATION - 1))
    CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint_iter_${PREV_ITERATION}.json"
    if load_checkpoint "$CHECKPOINT_FILE"; then
        print_success "Successfully loaded checkpoint from iteration $PREV_ITERATION"
        
        # Trace the loaded model for TorchScript optimization
        if [ -f "models/model.pt" ]; then
            trace_model "models/model.pt"
        fi
    else
        print_warning "Failed to load checkpoint, starting with current model"
    fi
else
    # For first iteration, trace initial model if it exists
    if [ -f "models/model.pt" ]; then
        print_info "Tracing initial model for TorchScript optimization..."
        trace_model "models/model.pt"
    fi
fi

# Performance tracking
declare -a iteration_times
declare -a selfplay_times
declare -a training_times

# Main training loop
for ((i=$START_ITERATION; i<=$NUM_ITERATIONS; i++)); do
    iteration_start=$(date +%s)
    
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_training "Iteration $i/$NUM_ITERATIONS (MCTS: $FINAL_MCTS_MODE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Phase 1: Self-play with chosen MCTS mode
    print_phase "Phase 1: Self-play generation ($GAMES_PER_ITERATION games)"
    print_info "Using $FINAL_MCTS_MODE MCTS mode for self-play"
    
    selfplay_start=$(date +%s)
    
    # Build self-play command with MCTS mode
    SELF_PLAY_CMD="./build/bin/Release/omoknuni_cli_final self-play $CONFIG_FILE --mcts-mode=$FINAL_MCTS_MODE"
    if [ "$VERBOSE" = "true" ]; then
        SELF_PLAY_CMD="$SELF_PLAY_CMD --verbose"
    fi
    
    print_info "Executing: $SELF_PLAY_CMD"
    
    if ! $SELF_PLAY_CMD 2>&1 | tee -a "$LOG_FILE"; then
        print_error "Self-play failed!"
        cleanup
    fi
    
    selfplay_end=$(date +%s)
    selfplay_duration=$((selfplay_end - selfplay_start))
    selfplay_times+=($selfplay_duration)
    
    print_success "Self-play completed in ${selfplay_duration}s"
    
    # Show GPU status after self-play if using GPU mode
    if [[ "$FINAL_MCTS_MODE" == "gpu" ]] && command -v nvidia-smi &> /dev/null; then
        print_info "GPU status after self-play:"
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | \
        awk '{printf "  Memory: %d/%d MB, Temperature: %d°C\n", $1, $2, $3}'
    fi
    
    # Phase 2: Data preparation
    print_phase "Phase 2: Preparing training data"
    if ! python3 prepare_training_data.py \
        --game-dir "data/self_play_games" \
        --output-dir "data/training" \
        --iteration $i \
        --config "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        print_error "Data preparation failed!"
        cleanup
    fi
    
    # Phase 3: Neural network training
    print_phase "Phase 3: Training neural network"
    training_start=$(date +%s)
    
    if ! python3 alphazero_train.py \
        --config "$CONFIG_FILE" \
        --iteration $i \
        --data-dir "data/training" 2>&1 | tee -a "$LOG_FILE"; then
        print_error "Training failed!"
        cleanup
    fi
    
    training_end=$(date +%s)
    training_duration=$((training_end - training_start))
    training_times+=($training_duration)
    
    print_success "Neural network training completed in ${training_duration}s"
    
    # Trace the newly trained model for TorchScript optimization
    if [ -f "models/model.pt" ]; then
        trace_model "models/model.pt"
    fi
    
    # Phase 4: Model evaluation
    print_phase "Phase 4: Evaluating new model"
    
    # Run evaluation script with MCTS mode
    EVALUATION_OUTPUT="$LOG_DIR/evaluation_iter_${i}.json"
    EVAL_CMD="python3 alphazero_evaluate.py --config $CONFIG_FILE --iteration $i --checkpoint-dir $CHECKPOINT_DIR --output $EVALUATION_OUTPUT --mcts-mode $FINAL_MCTS_MODE"
    
    if $EVAL_CMD 2>&1 | tee -a "$LOG_FILE"; then
        # Check if new model is better
        if python3 -c "import json; d=json.load(open('$EVALUATION_OUTPUT')); exit(0 if d.get('contender_is_better', False) else 1)" 2>/dev/null; then
            print_success "New model is better and has been promoted to champion!"
        else
            print_success "Previous model remains champion"
            # Restore best model
            if [ -f "models/best_model.pt" ]; then
                cp "models/best_model.pt" "models/model.pt"
            fi
        fi
    else
        print_warning "Evaluation failed, keeping current model"
        # In case of evaluation failure, keep the new model anyway
        if [ -f "$CHECKPOINT_DIR/model_iter_${i}.pt" ]; then
            cp "$CHECKPOINT_DIR/model_iter_${i}.pt" "models/model.pt"
        fi
    fi
    
    # Save checkpoint based on interval or if it's the last iteration
    if [ $((i % CHECKPOINT_INTERVAL)) -eq 0 ] || [ $i -eq $NUM_ITERATIONS ]; then
        save_checkpoint $i
    fi
    
    # Memory cleanup
    print_info "Running memory cleanup..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Generate ELO visualization if ratings exist
    if [ -f "models/elo_ratings.json" ]; then
        print_info "Generating ELO visualizations..."
        python3 visualize_elo.py --elo-file models/elo_ratings.json --output-dir "$LOG_DIR" 2>&1 | tee -a "$LOG_FILE" || true
        
        # Print current ELO scores
        print_info "Current ELO Ratings:"
        python3 -c "
import json
with open('models/elo_ratings.json', 'r') as f:
    data = json.load(f)
    ratings = data['ratings']
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for model, rating in sorted_ratings[:5]:  # Show top 5
        print(f'  {model:<20} {rating:>7.1f}')
" 2>/dev/null || print_info "  Unable to read ELO ratings"
    fi
    
    iteration_end=$(date +%s)
    iteration_duration=$((iteration_end - iteration_start))
    iteration_times+=($iteration_duration)
    
    print_success "Iteration $i completed successfully in ${iteration_duration}s!"
    
    # Performance summary
    print_info "Performance Summary:"
    print_info "  Self-play: ${selfplay_duration}s"
    print_info "  Training: ${training_duration}s"
    print_info "  Total iteration: ${iteration_duration}s"
    
    # Estimate remaining time
    if [ ${#iteration_times[@]} -gt 0 ]; then
        avg_time=$(( $(IFS=+; echo "${iteration_times[*]}") / ${#iteration_times[@]} ))
        remaining_iterations=$((NUM_ITERATIONS - i))
        estimated_remaining=$((avg_time * remaining_iterations))
        hours=$((estimated_remaining / 3600))
        minutes=$(((estimated_remaining % 3600) / 60))
        print_info "Estimated remaining time: ${hours}h ${minutes}m"
    fi
done

# Final performance summary
if [ ${#iteration_times[@]} -gt 0 ]; then
    echo -e "\n📊 Training Performance Summary (MCTS: $FINAL_MCTS_MODE)"
    echo "================================="
    
    total_time=$(IFS=+; echo "${iteration_times[*]}")
    avg_iteration=$(( total_time / ${#iteration_times[@]} ))
    
    if [ ${#selfplay_times[@]} -gt 0 ]; then
        avg_selfplay=$(( $(IFS=+; echo "${selfplay_times[*]}") / ${#selfplay_times[@]} ))
        print_info "Average self-play time: ${avg_selfplay}s"
    fi
    
    if [ ${#training_times[@]} -gt 0 ]; then
        avg_training=$(( $(IFS=+; echo "${training_times[*]}") / ${#training_times[@]} ))
        print_info "Average training time: ${avg_training}s"
    fi
    
    print_info "Average iteration time: ${avg_iteration}s"
    print_info "Total training time: ${total_time}s"
    
    hours=$((total_time / 3600))
    minutes=$(((total_time % 3600) / 60))
    print_info "Total duration: ${hours}h ${minutes}m"
fi

print_success "Training completed successfully!"
print_info "Final model saved to: models/model.pt"
print_info "Final checkpoint: $CHECKPOINT_DIR/checkpoint_iter_${NUM_ITERATIONS}.json"
print_info "Training logs: $LOG_FILE"
print_info "TensorBoard data: $TENSORBOARD_DIR"
print_info "MCTS mode used: $FINAL_MCTS_MODE"

# Save final checkpoint if not already saved
if [ $((NUM_ITERATIONS % CHECKPOINT_INTERVAL)) -ne 0 ]; then
    save_checkpoint $NUM_ITERATIONS
fi

# Final GPU status if using GPU mode
if [[ "$FINAL_MCTS_MODE" == "gpu" ]] && command -v nvidia-smi &> /dev/null; then
    print_info "Final GPU status:"
    nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | \
    awk '{printf "  Memory: %d/%d MB, Temperature: %d°C\n", $1, $2, $3}'
fi

# Cleanup
cleanup