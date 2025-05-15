#!/usr/bin/env python3
"""
AlphaZero Training Pipeline Main Script

This script integrates the C++ backend with the Python training pipeline
for training an AlphaZero-style neural network for board games.
"""

import os
import sys
import time
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "build"))

# Import our AlphaZero trainer
from alphazero.alphazero_trainer import AlphaZeroTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/alphazero_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('alphazero.main')

def create_default_config():
    """Create a default configuration file if none exists."""
    config = {
        # Game settings
        "game_type": "gomoku",
        "board_size": 15,
        "input_channels": 20,
        "policy_size": 0,  # Auto-calculated from board size
        
        # Directory settings
        "model_dir": "models",
        "data_dir": "data",
        "log_dir": "logs",
        
        # Neural network settings
        "network_type": "resnet",
        "use_gpu": True,
        "num_iterations": 10,
        "num_res_blocks": 19,
        "num_filters": 256,
        
        # Self-play settings
        "self_play_num_games": 100,  # Reduced for quicker testing
        "self_play_num_parallel_games": 4,
        "self_play_max_moves": 0,  # Auto-calculated
        "self_play_temperature_threshold": 30,
        "self_play_high_temperature": 1.0,
        "self_play_low_temperature": 0.1,
        
        # MCTS settings
        "mcts_num_simulations": 100,  # Reduced for quicker testing
        "mcts_num_threads": 4,
        "mcts_batch_size": 8,
        "mcts_batch_timeout_ms": 20,
        "mcts_exploration_constant": 1.5,
        "mcts_temperature": 1.0,
        "mcts_add_dirichlet_noise": True,
        "mcts_dirichlet_alpha": 0.3,
        "mcts_dirichlet_epsilon": 0.25,
        
        # Training settings
        "train_epochs": 10,
        "train_batch_size": 256,
        "train_num_workers": 4,
        "train_learning_rate": 0.001,
        "train_weight_decay": 0.0001,
        "train_lr_step_size": 5,
        "train_lr_gamma": 0.1,
        
        # Arena/evaluation settings
        "enable_evaluation": True,
        "arena_num_games": 20,  # Reduced for quicker testing
        "arena_num_parallel_games": 4,
        "arena_num_threads": 4,
        "arena_num_simulations": 100,  # Reduced for quicker testing
        "arena_temperature": 0.1,
        "arena_win_rate_threshold": 0.55
    }
    
    return config

def create_config_file(path):
    """Create a default configuration file."""
    config = create_default_config()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Created default configuration file at: {path}")
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AlphaZero Training Pipeline')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of iterations to run (overrides config value)')
    parser.add_argument('--self-play-games', type=int, default=None,
                        help='Number of self-play games per iteration (overrides config value)')
    parser.add_argument('--mode', type=str, choices=['train', 'selfplay', 'eval'], default='train',
                        help='Operation mode: train (full pipeline), selfplay (generate games only), eval (evaluate model)')
    parser.add_argument('--game-type', type=str, choices=['gomoku', 'chess', 'go'], default=None,
                        help='Game type (overrides config value)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    
    return parser.parse_args()

def main():
    """Main function to run AlphaZero pipeline."""
    args = parse_args()
    
    # Create default config file if it doesn't exist
    if not os.path.exists(args.config):
        config = create_config_file(args.config)
    else:
        # Load configuration from file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.iterations is not None:
        config['num_iterations'] = args.iterations
        
    if args.self_play_games is not None:
        config['self_play_num_games'] = args.self_play_games
        
    if args.game_type is not None:
        config['game_type'] = args.game_type
        
    if args.no_gpu:
        config['use_gpu'] = False
    
    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Initialize AlphaZero trainer
    logger.info("Initializing AlphaZero trainer")
    trainer = AlphaZeroTrainer(config_dict=config)
    
    # Run in selected mode
    if args.mode == 'train':
        logger.info(f"Running full training pipeline for {config['num_iterations']} iterations")
        trainer.run()
    elif args.mode == 'selfplay':
        # Just run one iteration of self-play
        logger.info(f"Running self-play only for {config['self_play_num_games']} games")
        iteration_dir = trainer._run_self_play(
            f"{config['data_dir']}/selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info(f"Self-play completed. Games saved to {iteration_dir}")
    elif args.mode == 'eval':
        # Evaluate the current model
        logger.info("Running model evaluation")
        evaluation_dir = f"{config['data_dir']}/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(evaluation_dir, exist_ok=True)
        results = trainer._evaluate_model(evaluation_dir)
        logger.info(f"Evaluation complete: {results}")
    
    logger.info("AlphaZero pipeline completed successfully")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in AlphaZero pipeline: {e}")
        sys.exit(1)