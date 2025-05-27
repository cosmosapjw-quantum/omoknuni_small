#!/bin/bash

# Simplified AlphaZero Training Script
# Quick start version with essential features

set -e

# Configuration
CONFIG_FILE="config_alphazero_train.yaml"
NUM_ITERATIONS=10
GAMES_PER_ITERATION=100
EVAL_GAMES=50

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎯 AlphaZero Training (Simplified)${NC}"
echo "===================================="

# Create directories
mkdir -p checkpoints/alphazero
mkdir -p logs/alphazero
mkdir -p data/{self_play_games,training_data,evaluation_games}

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release

# Initialize model
CURRENT_MODEL="checkpoints/alphazero/current_model.pt"
BEST_MODEL="checkpoints/alphazero/best_model.pt"

if [ ! -f "$CURRENT_MODEL" ]; then
    echo -e "${YELLOW}Initializing random model...${NC}"
    # Use a simple self-play run to generate initial model
    ./build/bin/Release/omoknuni_cli_final self-play \
        --config config_ddw_balanced.yaml \
        --num-games 1 \
        --output models/init_model.pt
    cp models/init_model.pt "$CURRENT_MODEL"
    cp models/init_model.pt "$BEST_MODEL"
fi

# Training loop
for ITER in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo -e "${BLUE}=== Iteration $ITER / $NUM_ITERATIONS ===${NC}"
    
    # 1. Self-play
    echo -e "${YELLOW}Generating self-play games...${NC}"
    SELFPLAY_DIR="data/self_play_games/iter_${ITER}"
    mkdir -p "$SELFPLAY_DIR"
    
    ./build/bin/Release/omoknuni_cli_final self-play \
        --config config_ddw_balanced.yaml \
        --model "$CURRENT_MODEL" \
        --num-games $GAMES_PER_ITERATION \
        --output-dir "$SELFPLAY_DIR" \
        2>&1 | tee "logs/alphazero/selfplay_${ITER}.log"
    
    # 2. Prepare training data
    echo -e "${YELLOW}Preparing training data...${NC}"
    python3 prepare_training_data.py \
        --input-dir "$SELFPLAY_DIR" \
        --output "data/training_data/iter_${ITER}.npz" \
        --board-size 9
    
    # 3. Train network
    echo -e "${YELLOW}Training neural network...${NC}"
    python3 alphazero_train.py \
        --config "$CONFIG_FILE" \
        --data "data/training_data/iter_${ITER}.npz" \
        --iteration $ITER \
        --checkpoint "$CURRENT_MODEL" \
        --output "checkpoints/alphazero/model_iter_${ITER}.pt" \
        --tensorboard-dir "logs/alphazero/tensorboard/iter_${ITER}"
    
    # Update current model
    cp "checkpoints/alphazero/model_iter_${ITER}.pt" "$CURRENT_MODEL"
    
    # 4. Evaluate (simplified - just compare win rates)
    echo -e "${YELLOW}Evaluating new model...${NC}"
    
    # For now, always accept the new model
    # In a real implementation, you would play matches between old and new
    echo -e "${GREEN}✓ Model updated${NC}"
    cp "$CURRENT_MODEL" "$BEST_MODEL"
    
    # Save progress
    echo "$ITER" > "checkpoints/alphazero/latest_iteration.txt"
    
    # Cleanup old data
    if [ $ITER -gt 5 ]; then
        rm -rf "data/self_play_games/iter_$((ITER-5))"
        rm -f "data/training_data/iter_$((ITER-5)).npz"
    fi
done

echo ""
echo -e "${GREEN}✅ Training completed!${NC}"
echo "Best model saved to: $BEST_MODEL"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir logs/alphazero/tensorboard"