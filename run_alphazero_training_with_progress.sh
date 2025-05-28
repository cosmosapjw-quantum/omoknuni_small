#!/bin/bash
# AlphaZero Training Pipeline with Progress Bar

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
CONFIG_FILE="${1:-config_alphazero_train.yaml}"
VERBOSE="${2:-false}"
RESUME_FROM_CHECKPOINT="${3:-true}"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

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

# Set up logging
LOG_FILE="$LOG_DIR/alphazero_training_$(date +%Y%m%d_%H%M%S).log"

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
            echo "📂 Found checkpoint at iteration $CHECKPOINT_ITER"
            echo "🔄 Resuming from iteration $START_ITERATION"
        fi
    fi
fi

echo "🏁 Starting AlphaZero Training Pipeline"
echo "================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start Iteration: $START_ITERATION"
echo "Total Iterations: $NUM_ITERATIONS"
echo "Games per iteration: $GAMES_PER_ITERATION"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "Max checkpoints: $KEEP_CHECKPOINT_MAX"
echo "Tensorboard: $TENSORBOARD_DIR"
echo "Log file: $LOG_FILE"
echo "================================"

# Start TensorBoard in background
echo "📊 Starting TensorBoard..."
tensorboard --logdir="$TENSORBOARD_DIR" --port 6006 --bind_all &
TB_PID=$!
echo "TensorBoard running at http://localhost:6006"

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
        echo "✅ Model is already in TorchScript format"
        # Copy to traced path if different
        if [ "$model_path" != "$traced_path" ]; then
            cp "$model_path" "$traced_path"
            echo "📋 Copied TorchScript model to: $traced_path"
        fi
        return 0
    fi
    
    echo "🔧 Tracing model for TorchScript optimization..."
    
    if [ -f "$model_path" ]; then
        python3 trace_model_for_cpp.py \
            --model "$model_path" \
            --output "$traced_path" \
            --board-size "$BOARD_SIZE" \
            --input-channels "$INPUT_CHANNELS" \
            --batch-size 1 \
            --device cuda 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ] && [ -f "$traced_path" ]; then
            echo "✅ Model traced successfully: $traced_path"
            return 0
        else
            echo "⚠️  Model tracing failed, continuing without TorchScript optimization"
            return 1
        fi
    else
        echo "⚠️  Model file not found: $model_path"
        return 1
    fi
}

# Function to save checkpoint
save_checkpoint() {
    local iteration=$1
    local checkpoint_file="$CHECKPOINT_DIR/checkpoint_iter_${iteration}.json"
    local model_file="$CHECKPOINT_DIR/model_iter_${iteration}.pt"
    
    echo "💾 Saving checkpoint at iteration $iteration"
    
    # Create checkpoint metadata
    cat > "$checkpoint_file" <<EOF
{
    "iteration": $iteration,
    "timestamp": "$(date -Iseconds)",
    "experiment_name": "$EXPERIMENT_NAME",
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
        echo "🧹 Cleaning old checkpoints (keeping latest $KEEP_CHECKPOINT_MAX)"
        ls -1t "$CHECKPOINT_DIR"/checkpoint_iter_*.json | tail -n +$((KEEP_CHECKPOINT_MAX + 1)) | while read f; do
            base=$(basename "$f" .json)
            rm -f "$f" "$CHECKPOINT_DIR/${base}.pt"
            echo "  Removed: $base"
        done
    fi
}

# Function to load checkpoint
load_checkpoint() {
    local checkpoint_file=$1
    if [ -f "$checkpoint_file" ]; then
        local model_path=$(grep '"model_path"' "$checkpoint_file" | cut -d'"' -f4)
        if [ -f "$model_path" ]; then
            echo "📥 Loading model from checkpoint: $model_path"
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
        echo "✅ Successfully loaded checkpoint from iteration $PREV_ITERATION"
        
        # Trace the loaded model for TorchScript optimization
        if [ -f "models/model.pt" ]; then
            trace_model "models/model.pt"
        fi
    else
        echo "⚠️  Failed to load checkpoint, starting with current model"
    fi
else
    # For first iteration, trace initial model if it exists
    if [ -f "models/model.pt" ]; then
        echo "🔧 Tracing initial model for TorchScript optimization..."
        trace_model "models/model.pt"
    fi
fi

# Main training loop
for ((i=$START_ITERATION; i<=$NUM_ITERATIONS; i++)); do
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Iteration $i/$NUM_ITERATIONS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Phase 1: Self-play
    echo -e "\n📋 Phase 1: Self-play generation ($GAMES_PER_ITERATION games)"
    
    # Add verbose flag if requested
    SELF_PLAY_CMD="./build/bin/Release/omoknuni_cli_final self-play $CONFIG_FILE"
    if [ "$VERBOSE" = "true" ]; then
        SELF_PLAY_CMD="$SELF_PLAY_CMD --verbose"
    fi
    
    if ! $SELF_PLAY_CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo "❌ Self-play failed!"
        cleanup
    fi
    
    # Phase 2: Data preparation
    echo -e "\n📋 Phase 2: Preparing training data"
    if ! python3 prepare_training_data.py \
        --game-dir "data/self_play_games" \
        --output-dir "data/training" \
        --iteration $i \
        --config "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        echo "❌ Data preparation failed!"
        cleanup
    fi
    
    # Phase 3: Neural network training
    echo -e "\n📋 Phase 3: Training neural network"
    if ! python3 alphazero_train.py \
        --config "$CONFIG_FILE" \
        --iteration $i \
        --data-dir "data/training" 2>&1 | tee -a "$LOG_FILE"; then
        echo "❌ Training failed!"
        cleanup
    fi
    
    # Trace the newly trained model for TorchScript optimization
    if [ -f "models/model.pt" ]; then
        trace_model "models/model.pt"
    fi
    
    # Phase 4: Model evaluation
    echo -e "\n📋 Phase 4: Evaluating new model"
    
    # Run evaluation script
    EVALUATION_OUTPUT="$LOG_DIR/evaluation_iter_${i}.json"
    if python3 alphazero_evaluate.py \
        --config "$CONFIG_FILE" \
        --iteration $i \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --output "$EVALUATION_OUTPUT" 2>&1 | tee -a "$LOG_FILE"; then
        
        # Check if new model is better
        if python3 -c "import json; d=json.load(open('$EVALUATION_OUTPUT')); exit(0 if d.get('contender_is_better', False) else 1)" 2>/dev/null; then
            echo "✅ New model is better and has been promoted to champion!"
        else
            echo "✅ Previous model remains champion"
            # Restore best model
            if [ -f "models/best_model.pt" ]; then
                cp "models/best_model.pt" "models/model.pt"
            fi
        fi
    else
        echo "⚠️  Evaluation failed, keeping current model"
        # In case of evaluation failure, keep the new model anyway
        cp "$CHECKPOINT_DIR/model_iter_${i}.pt" "models/model.pt"
    fi
    
    # Save checkpoint based on interval or if it's the last iteration
    if [ $((i % CHECKPOINT_INTERVAL)) -eq 0 ] || [ $i -eq $NUM_ITERATIONS ]; then
        save_checkpoint $i
    fi
    
    # Memory cleanup
    echo -e "\n🧹 Running memory cleanup..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Generate ELO visualization if ratings exist
    if [ -f "models/elo_ratings.json" ]; then
        echo -e "\n📊 Generating ELO visualizations..."
        python3 visualize_elo.py --elo-file models/elo_ratings.json --output-dir "$LOG_DIR" 2>&1 | tee -a "$LOG_FILE" || true
        
        # Print current ELO scores
        echo -e "\n📈 Current ELO Ratings:"
        python3 -c "
import json
with open('models/elo_ratings.json', 'r') as f:
    data = json.load(f)
    ratings = data['ratings']
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for model, rating in sorted_ratings[:5]:  # Show top 5
        print(f'  {model:<20} {rating:>7.1f}')
" 2>/dev/null || echo "  Unable to read ELO ratings"
    fi
    
    echo -e "\n✅ Iteration $i completed successfully!"
done

echo -e "\n🎉 Training completed successfully!"
echo "Final model saved to: models/model.pt"
echo "Final checkpoint: $CHECKPOINT_DIR/checkpoint_iter_${NUM_ITERATIONS}.json"
echo "Training logs: $LOG_FILE"
echo "TensorBoard data: $TENSORBOARD_DIR"

# Save final checkpoint if not already saved
if [ $((NUM_ITERATIONS % CHECKPOINT_INTERVAL)) -ne 0 ]; then
    save_checkpoint $NUM_ITERATIONS
fi

# Cleanup
cleanup