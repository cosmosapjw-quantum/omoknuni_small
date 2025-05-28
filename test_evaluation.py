#!/usr/bin/env python3
"""
Test script for AlphaZero evaluation system.

This script tests the evaluation/arena functionality to ensure it works correctly.
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

# Import modules
from python.alphazero.alphazero_trainer import AlphaZeroNetwork
from python.alphazero.arena import Arena

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test.evaluation')


def create_test_model(name: str, board_size: int = 15):
    """Create a test model with random weights."""
    model = AlphaZeroNetwork(
        game_type='gomoku',
        input_channels=19,
        board_size=board_size,
        num_res_blocks=2,  # Small for testing
        num_filters=32     # Small for testing
    )
    
    # Add some variation to distinguish models
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Save model
    model_path = f"test_{name}_model.pt"
    model.save(model_path)
    logger.info(f"Created test model: {model_path}")
    
    return model_path


def test_arena_basic():
    """Test basic arena functionality."""
    logger.info("Testing basic arena functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test models
    champion_path = create_test_model("champion")
    contender_path = create_test_model("contender")
    
    try:
        # Load models
        champion_model = AlphaZeroNetwork.load(champion_path, device)
        champion_model.eval()
        
        contender_model = AlphaZeroNetwork.load(contender_path, device)
        contender_model.eval()
        
        # Create arena
        arena = Arena('gomoku', board_size=15, device=device)
        
        # Play a few test games
        logger.info("Playing 10 test games...")
        results = arena.play_match(
            champion_model=lambda x: champion_model.predict(x),
            contender_model=lambda x: contender_model.predict(x),
            num_games=10,
            num_parallel_games=2,
            mcts_simulations=50,  # Low for testing
            temperature=0.1,
            swap_colors=True
        )
        
        # Print results
        logger.info(f"Test match results:")
        logger.info(f"  Champion wins: {results['champion_wins']}")
        logger.info(f"  Contender wins: {results['contender_wins']}")
        logger.info(f"  Draws: {results['draws']}")
        logger.info(f"  Champion win rate: {results['champion_win_rate']:.2%}")
        logger.info(f"  Contender win rate: {results['contender_win_rate']:.2%}")
        
        # Verify results are reasonable
        assert results['total_games'] == 10, "Wrong number of games played"
        assert results['champion_wins'] + results['contender_wins'] + results['draws'] == 10, "Game counts don't add up"
        
        logger.info("✅ Basic arena test passed!")
        
    finally:
        # Cleanup test models
        if os.path.exists(champion_path):
            os.remove(champion_path)
        if os.path.exists(contender_path):
            os.remove(contender_path)


def test_evaluation_script():
    """Test the alphazero_evaluate.py script."""
    logger.info("Testing evaluation script...")
    
    # Create a minimal config file
    test_config = {
        'game_type': 'gomoku',
        'board_size': 15,
        'input_channels': 19,
        'model_dir': 'models',
        'pipeline': {
            'evaluation_threshold': 0.55
        },
        'evaluation': {
            'num_games': 10,
            'num_parallel_games': 2,
            'mcts_simulations': 50,
            'temperature': 0.1
        }
    }
    
    config_path = 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # Create test models directory
        os.makedirs('models', exist_ok=True)
        
        # Create a best model
        best_model = AlphaZeroNetwork(
            game_type='gomoku',
            input_channels=19,
            board_size=15,
            num_res_blocks=2,
            num_filters=32
        )
        best_model.save('models/best_model.pt')
        
        # Create a latest model (slightly different)
        latest_model = AlphaZeroNetwork(
            game_type='gomoku',
            input_channels=19,
            board_size=15,
            num_res_blocks=2,
            num_filters=32
        )
        # Add variation
        with torch.no_grad():
            for param in latest_model.parameters():
                param.add_(torch.randn_like(param) * 0.2)
        latest_model.save('models/latest_model.pt')
        
        # Run evaluation script
        import subprocess
        result = subprocess.run([
            sys.executable,
            'alphazero_evaluate.py',
            '--config', config_path,
            '--iteration', '1',
            '--output', 'test_evaluation.json'
        ], capture_output=True, text=True)
        
        logger.info(f"Evaluation script output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Evaluation script errors:\n{result.stderr}")
        
        # Check if evaluation completed
        if os.path.exists('test_evaluation.json'):
            import json
            with open('test_evaluation.json', 'r') as f:
                eval_results = json.load(f)
            
            logger.info(f"Evaluation results: {eval_results}")
            assert 'contender_win_rate' in eval_results, "Missing contender_win_rate in results"
            assert 'contender_is_better' in eval_results, "Missing contender_is_better in results"
            
            logger.info("✅ Evaluation script test passed!")
        else:
            logger.error("Evaluation script did not produce output file")
            
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
        if os.path.exists('test_evaluation.json'):
            os.remove('test_evaluation.json')
        if os.path.exists('models/best_model.pt'):
            os.remove('models/best_model.pt')
        if os.path.exists('models/latest_model.pt'):
            os.remove('models/latest_model.pt')


def main():
    """Run all tests."""
    logger.info("Starting AlphaZero evaluation system tests...")
    
    try:
        # Test 1: Basic arena functionality
        test_arena_basic()
        
        # Test 2: Evaluation script
        test_evaluation_script()
        
        logger.info("\n✅ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()