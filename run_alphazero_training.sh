#!/bin/bash

# Full AlphaZero Training Pipeline Script
# For Ryzen 9 5900X + RTX 3060 Ti

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${1:-config_alphazero_train.yaml}"
RESUME_FROM_CHECKPOINT="${2:-}"

echo -e "${BLUE}🎯 AlphaZero Training Pipeline${NC}"
echo "=================================="
echo "Config: $CONFIG_FILE"
echo "Date: $(date)"
echo ""

# Check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: nvidia-smi not found. Please install CUDA.${NC}"
        exit 1
    fi
    
    # Check Python packages
    python3 -c "import torch; import numpy; import yaml" 2>/dev/null || {
        echo -e "${RED}ERROR: Required Python packages not found.${NC}"
        echo "Please install: torch, numpy, pyyaml"
        exit 1
    }
    
    # Check executable
    if [ ! -f "./build/bin/Release/omoknuni_cli_final" ]; then
        echo -e "${RED}ERROR: omoknuni_cli_final not found. Please build the project first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
}

# Environment setup
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # CUDA settings
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_LAUNCH_BLOCKING=0
    
    # Threading settings for Ryzen 9 5900X
    export OMP_NUM_THREADS=12
    export MKL_NUM_THREADS=12
    export NUMEXPR_NUM_THREADS=12
    
    # PyTorch settings
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.7"
    export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060 Ti
    
    # Library path
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
    
    echo -e "${GREEN}✓ Environment configured${NC}"
}

# Create directory structure
create_directories() {
    echo -e "${YELLOW}Creating directory structure...${NC}"
    
    # Extract paths from config
    CHECKPOINT_DIR=$(grep "checkpoint_dir:" $CONFIG_FILE | awk '{print $2}')
    LOG_DIR=$(grep "log_dir:" $CONFIG_FILE | awk '{print $2}')
    TENSORBOARD_DIR=$(grep "tensorboard_dir:" $CONFIG_FILE | awk '{print $2}')
    
    # Create directories
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$TENSORBOARD_DIR"
    mkdir -p "data/self_play_games"
    mkdir -p "data/training_data"
    mkdir -p "data/evaluation_games"
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Start monitoring services
start_monitoring() {
    echo -e "${YELLOW}Starting monitoring services...${NC}"
    
    # Start TensorBoard
    if command -v tensorboard &> /dev/null; then
        tensorboard --logdir="$TENSORBOARD_DIR" --port=6006 --bind_all &
        TENSORBOARD_PID=$!
        echo -e "${GREEN}✓ TensorBoard started at http://localhost:6006${NC}"
    fi
    
    # Start memory monitor
    python3 monitor_memory_usage.py --interval 30 --output "$LOG_DIR/memory_usage.log" &
    MEMORY_MONITOR_PID=$!
    echo -e "${GREEN}✓ Memory monitor started${NC}"
    
    # Start GPU monitor
    nvidia-smi dmon -s pucvmet -d 30 -o DT > "$LOG_DIR/gpu_usage.log" &
    GPU_MONITOR_PID=$!
    echo -e "${GREEN}✓ GPU monitor started${NC}"
}

# Main training loop
run_training() {
    echo -e "${BLUE}Starting AlphaZero training...${NC}"
    echo ""
    
    # Parse configuration values
    NUM_ITERATIONS=$(grep "num_iterations:" $CONFIG_FILE | awk '{print $2}')
    GAMES_PER_ITER=$(grep "games_per_iteration:" $CONFIG_FILE | awk '{print $2}')
    
    # Initialize best model
    BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
    CURRENT_MODEL="$CHECKPOINT_DIR/current_model.pt"
    
    # Resume from checkpoint if specified
    if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
        echo -e "${YELLOW}Resuming from checkpoint: $RESUME_FROM_CHECKPOINT${NC}"
        cp "$RESUME_FROM_CHECKPOINT" "$CURRENT_MODEL"
        START_ITER=$(echo "$RESUME_FROM_CHECKPOINT" | grep -oE 'iter_[0-9]+' | grep -oE '[0-9]+')
        START_ITER=$((START_ITER + 1))
    else
        START_ITER=1
        # Initialize with random model
        echo -e "${YELLOW}Initializing with random model...${NC}"
        ./build/bin/Release/omoknuni_cli_final init-model --config $CONFIG_FILE --output "$CURRENT_MODEL"
        cp "$CURRENT_MODEL" "$BEST_MODEL"
    fi
    
    # Training iterations
    for ITER in $(seq $START_ITER $NUM_ITERATIONS); do
        echo ""
        echo -e "${BLUE}=== Iteration $ITER / $NUM_ITERATIONS ===${NC}"
        echo "Start time: $(date)"
        
        ITER_START=$(date +%s)
        
        # Phase 1: Self-play
        echo -e "${YELLOW}Phase 1: Self-play data generation${NC}"
        SELFPLAY_DIR="data/self_play_games/iter_${ITER}"
        mkdir -p "$SELFPLAY_DIR"
        
        ./build/bin/Release/omoknuni_cli_final self-play \
            --config $CONFIG_FILE \
            --model "$CURRENT_MODEL" \
            --output-dir "$SELFPLAY_DIR" \
            --num-games $GAMES_PER_ITER \
            --parallel-workers 8 \
            2>&1 | tee "$LOG_DIR/selfplay_iter_${ITER}.log"
        
        echo -e "${GREEN}✓ Generated $GAMES_PER_ITER games${NC}"
        
        # Phase 2: Training
        echo -e "${YELLOW}Phase 2: Neural network training${NC}"
        TRAIN_DATA="data/training_data/iter_${ITER}.npz"
        
        # Convert games to training data
        ./build/bin/Release/omoknuni_cli_final prepare-training \
            --input-dir "$SELFPLAY_DIR" \
            --output "$TRAIN_DATA" \
            --window-size 500000 \
            2>&1 | tee -a "$LOG_DIR/training_iter_${ITER}.log"
        
        # Train the network
        ./build/bin/Release/omoknuni_cli_final train \
            --config $CONFIG_FILE \
            --model "$CURRENT_MODEL" \
            --data "$TRAIN_DATA" \
            --output "$CHECKPOINT_DIR/model_iter_${ITER}.pt" \
            --tensorboard-dir "$TENSORBOARD_DIR/iter_${ITER}" \
            2>&1 | tee -a "$LOG_DIR/training_iter_${ITER}.log"
        
        # Update current model
        cp "$CHECKPOINT_DIR/model_iter_${ITER}.pt" "$CURRENT_MODEL"
        echo -e "${GREEN}✓ Training completed${NC}"
        
        # Phase 3: Evaluation
        echo -e "${YELLOW}Phase 3: Model evaluation${NC}"
        EVAL_DIR="data/evaluation_games/iter_${ITER}"
        mkdir -p "$EVAL_DIR"
        
        WIN_RATE=$(./build/bin/Release/omoknuni_cli_final evaluate \
            --config $CONFIG_FILE \
            --new-model "$CURRENT_MODEL" \
            --old-model "$BEST_MODEL" \
            --output-dir "$EVAL_DIR" \
            --num-games 100 \
            2>&1 | tee "$LOG_DIR/evaluation_iter_${ITER}.log" | \
            grep "Win rate:" | awk '{print $3}')
        
        echo -e "Win rate: ${WIN_RATE}"
        
        # Update best model if threshold met
        THRESHOLD=$(grep "evaluation_threshold:" $CONFIG_FILE | awk '{print $2}')
        if (( $(echo "$WIN_RATE > $THRESHOLD" | bc -l) )); then
            echo -e "${GREEN}✓ New best model found! (${WIN_RATE} > ${THRESHOLD})${NC}"
            cp "$CURRENT_MODEL" "$BEST_MODEL"
            echo "$ITER" > "$CHECKPOINT_DIR/best_iteration.txt"
        else
            echo -e "${YELLOW}Model did not improve (${WIN_RATE} <= ${THRESHOLD})${NC}"
            # Revert to best model
            cp "$BEST_MODEL" "$CURRENT_MODEL"
        fi
        
        # Calculate iteration time
        ITER_END=$(date +%s)
        ITER_TIME=$((ITER_END - ITER_START))
        echo "Iteration time: $((ITER_TIME / 60)) minutes"
        
        # Save iteration summary
        cat << EOF >> "$LOG_DIR/training_summary.log"
Iteration: $ITER
Date: $(date)
Games generated: $GAMES_PER_ITER
Win rate: $WIN_RATE
Model updated: $([[ $(echo "$WIN_RATE > $THRESHOLD" | bc -l) -eq 1 ]] && echo "Yes" || echo "No")
Duration: $((ITER_TIME / 60)) minutes
---
EOF
        
        # Cleanup old data if needed
        if [ $ITER -gt 10 ]; then
            echo -e "${YELLOW}Cleaning up old data...${NC}"
            find data/self_play_games -name "iter_*" -type d | sort -V | head -n -10 | xargs rm -rf
            find data/training_data -name "iter_*.npz" | sort -V | head -n -10 | xargs rm -f
        fi
        
        echo -e "${GREEN}✓ Iteration $ITER completed${NC}"
    done
    
    echo ""
    echo -e "${GREEN}🎉 Training completed!${NC}"
    echo "Best model: $BEST_MODEL"
    echo "Best iteration: $(cat "$CHECKPOINT_DIR/best_iteration.txt")"
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill monitoring processes
    [ ! -z "$TENSORBOARD_PID" ] && kill $TENSORBOARD_PID 2>/dev/null
    [ ! -z "$MEMORY_MONITOR_PID" ] && kill $MEMORY_MONITOR_PID 2>/dev/null
    [ ! -z "$GPU_MONITOR_PID" ] && kill $GPU_MONITOR_PID 2>/dev/null
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    check_dependencies
    setup_environment
    create_directories
    start_monitoring
    
    # Wait a bit for monitors to start
    sleep 5
    
    run_training
}

# Run main function
main

echo -e "${GREEN}✅ AlphaZero training pipeline completed successfully!${NC}"