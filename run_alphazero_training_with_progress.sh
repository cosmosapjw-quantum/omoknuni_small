#!/bin/bash
# AlphaZero Training Pipeline with Progress Bar

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
CONFIG_FILE="${1:-config_alphazero_train.yaml}"
VERBOSE="${2:-false}"

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

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR" "data/training"

# Set up logging
LOG_FILE="$LOG_DIR/alphazero_training_$(date +%Y%m%d_%H%M%S).log"

echo "🏁 Starting AlphaZero Training Pipeline"
echo "================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Iterations: $NUM_ITERATIONS"
echo "Games per iteration: $GAMES_PER_ITERATION"
echo "Checkpoints: $CHECKPOINT_DIR"
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

# Main training loop
for ((i=1; i<=$NUM_ITERATIONS; i++)); do
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
    
    # Phase 4: Model evaluation
    echo -e "\n📋 Phase 4: Evaluating new model"
    # For now, we'll just copy the model as the evaluation isn't implemented
    cp "$CHECKPOINT_DIR/model_iter_${i}.pt" "models/model.pt"
    echo "✅ Model updated"
    
    # Save checkpoint
    if [ $((i % 5)) -eq 0 ]; then
        echo -e "\n💾 Saving checkpoint at iteration $i"
        cp "$CHECKPOINT_DIR/model_iter_${i}.pt" "$CHECKPOINT_DIR/checkpoint_${i}.pt"
    fi
    
    # Memory cleanup
    echo -e "\n🧹 Running memory cleanup..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    echo -e "\n✅ Iteration $i completed successfully!"
done

echo -e "\n🎉 Training completed successfully!"
echo "Final model saved to: models/model.pt"
echo "Training logs: $LOG_FILE"
echo "TensorBoard data: $TENSORBOARD_DIR"

# Cleanup
cleanup