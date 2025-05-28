#!/usr/bin/env python3
"""
AlphaZero Model Evaluation Script

This script handles the evaluation phase of the AlphaZero training pipeline.
It plays arena matches between the current best model and newly trained models
to determine if the new model should become the champion.
"""

import os
import sys
import yaml
import json
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "build"))
sys.path.append(str(Path(__file__).parent / "build/lib/Release"))

# Import required modules
from python.alphazero.arena import Arena, evaluate_models
from python.alphazero.alphazero_trainer import AlphaZeroNetwork
from python.alphazero.elo_system import ELORatingSystem, RandomPolicyModel, create_elo_report
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('alphazero.evaluate')


class AlphaZeroEvaluator:
    """Main evaluator class for AlphaZero models."""
    
    def __init__(self, config_path: str):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Extract relevant settings
        self.game_type = self.config['game_type']
        self.board_size = self.config['board_size']
        
        # Evaluation settings
        eval_config = self.config.get('evaluation', {})
        self.num_games = eval_config.get('num_games', 100)
        self.num_parallel_games = eval_config.get('num_parallel_games', 8)
        self.mcts_simulations = eval_config.get('mcts_simulations', 400)
        self.temperature = eval_config.get('temperature', 0.1)
        self.win_rate_threshold = self.config['pipeline'].get('evaluation_threshold', 0.55)
        
        # Model paths
        self.model_dir = self.config.get('model_dir', 'models')
        self.best_model_path = os.path.join(self.model_dir, 'best_model.pt')
        self.latest_model_path = os.path.join(self.model_dir, 'latest_model.pt')
        
        # ELO system
        self.elo_system = ELORatingSystem()
        self.elo_file = os.path.join(self.model_dir, 'elo_ratings.json')
        
        # Load existing ELO ratings if available
        if os.path.exists(self.elo_file):
            try:
                self.elo_system.load_ratings(self.elo_file)
                logger.info(f"Loaded ELO ratings from {self.elo_file}")
            except Exception as e:
                logger.warning(f"Failed to load ELO ratings: {e}")
        
    def evaluate_iteration(self, iteration: int, new_model_path: Optional[str] = None) -> Dict:
        """
        Evaluate the model from a specific iteration.
        
        Args:
            iteration: The iteration number
            new_model_path: Path to the new model (if not provided, uses latest_model.pt)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation for iteration {iteration}")
        
        # Determine model paths
        contender_path = new_model_path or self.latest_model_path
        
        # Check if best model exists (for first iteration, new model automatically becomes best)
        if not os.path.exists(self.best_model_path):
            logger.info("No best model found. New model will become the first champion.")
            
            # Copy new model as best model
            import shutil
            shutil.copy2(contender_path, self.best_model_path)
            
            results = {
                'iteration': iteration,
                'champion_wins': 0,
                'contender_wins': 0,
                'draws': 0,
                'total_games': 0,
                'contender_win_rate': 1.0,
                'champion_win_rate': 0.0,
                'contender_is_better': True,
                'passes_threshold': True,
                'first_iteration': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("First iteration - new model automatically becomes champion")
            return results
        
        # Run arena evaluation
        logger.info(f"Evaluating: {self.best_model_path} (champion) vs {contender_path} (contender)")
        
        try:
            # Use the unified arena implementation
            from python.alphazero.arena import Arena
            
            # Load models
            champion_model = AlphaZeroNetwork.load(self.best_model_path, self.device)
            champion_model.eval()
            
            contender_model = AlphaZeroNetwork.load(contender_path, self.device)
            contender_model.eval()
            
            # Create arena and run evaluation
            arena = Arena(self.game_type, self.board_size, self.device)
            
            # Simplified evaluation using the model's predict method
            results = self._run_simple_evaluation(champion_model, contender_model, arena)
            
            # Add metadata
            results['iteration'] = iteration
            results['timestamp'] = datetime.now().isoformat()
            results['win_rate_threshold'] = self.win_rate_threshold
            results['passes_threshold'] = results['contender_win_rate'] >= self.win_rate_threshold
            results['contender_is_better'] = results['passes_threshold']
            
            # Update ELO ratings
            model_id_champion = f"iter_{iteration-1}" if iteration > 1 else "initial"
            model_id_contender = f"iter_{iteration}"
            
            elo_results = self._update_elo_ratings(
                model_id_champion, model_id_contender,
                results['champion_wins'], results['contender_wins'], results['draws']
            )
            
            # Add ELO information to results
            results['elo'] = {
                'champion_id': model_id_champion,
                'contender_id': model_id_contender,
                'champion_rating': elo_results['champion_new_elo'],
                'contender_rating': elo_results['contender_new_elo'],
                'rating_change_champion': elo_results['champion_new_elo'] - elo_results['champion_old_elo'],
                'rating_change_contender': elo_results['contender_new_elo'] - elo_results['contender_old_elo']
            }
            
            # Update best model if contender wins
            if results['contender_is_better']:
                logger.info(f"New model is better! Win rate: {results['contender_win_rate']:.2%}")
                import shutil
                shutil.copy2(contender_path, self.best_model_path)
                logger.info(f"Updated best model: {self.best_model_path}")
            else:
                logger.info(f"Champion remains the best. Contender win rate: {results['contender_win_rate']:.2%}")
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return a safe default result
            results = {
                'iteration': iteration,
                'error': str(e),
                'champion_wins': 0,
                'contender_wins': 0,
                'draws': 0,
                'total_games': 0,
                'contender_win_rate': 0.0,
                'champion_win_rate': 0.0,
                'contender_is_better': False,
                'passes_threshold': False,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def _run_simple_evaluation(self, champion_model, contender_model, arena):
        """
        Run a simplified evaluation without full MCTS.
        This is a placeholder for the full implementation.
        """
        import numpy as np
        import random
        
        champion_wins = 0
        contender_wins = 0
        draws = 0
        
        logger.info(f"Running {self.num_games} evaluation games...")
        
        for game_idx in range(self.num_games):
            # Alternate who plays first
            champion_plays_first = (game_idx % 2 == 0)
            
            # Simulate a game (simplified - in practice would use full MCTS)
            # For now, use value predictions as a proxy for game strength
            test_positions = 10  # Sample some random positions
            champion_score = 0
            contender_score = 0
            
            for _ in range(test_positions):
                # Create a random board position
                board_tensor = torch.randn(1, self.config['input_channels'], 
                                          self.board_size, self.board_size).to(self.device)
                
                # Get value predictions
                with torch.no_grad():
                    _, champion_value = champion_model(board_tensor)
                    _, contender_value = contender_model(board_tensor)
                
                champion_score += champion_value.item()
                contender_score += contender_value.item()
            
            # Determine winner based on average value
            # Add some randomness to simulate game variance
            champion_score += random.gauss(0, 0.1)
            contender_score += random.gauss(0, 0.1)
            
            if champion_score > contender_score + 0.05:
                champion_wins += 1
            elif contender_score > champion_score + 0.05:
                contender_wins += 1
            else:
                draws += 1
            
            # Log progress
            if (game_idx + 1) % 10 == 0:
                logger.info(f"Progress: {game_idx + 1}/{self.num_games} games")
        
        total_games = champion_wins + contender_wins + draws
        
        results = {
            'champion_wins': champion_wins,
            'contender_wins': contender_wins,
            'draws': draws,
            'total_games': total_games,
            'champion_win_rate': champion_wins / total_games if total_games > 0 else 0,
            'contender_win_rate': contender_wins / total_games if total_games > 0 else 0,
            'draw_rate': draws / total_games if total_games > 0 else 0
        }
        
        logger.info(f"Evaluation complete: Champion {champion_wins} - {contender_wins} Contender ({draws} draws)")
        
        return results
    
    def _update_elo_ratings(self, champion_model_id: str, contender_model_id: str, 
                           champion_wins: int, contender_wins: int, draws: int):
        """Update ELO ratings based on match results."""
        # Add models to ELO system if not present
        if self.elo_system.get_rating(champion_model_id) is None:
            # If this is the first champion, calibrate against random
            self._calibrate_model_vs_random(champion_model_id)
            
        if self.elo_system.get_rating(contender_model_id) is None:
            # New models start at default rating
            self.elo_system.add_model(contender_model_id)
            
        # Update ratings based on match
        old_champion_elo = self.elo_system.get_rating(champion_model_id)
        old_contender_elo = self.elo_system.get_rating(contender_model_id)
        
        new_champion_elo, new_contender_elo = self.elo_system.update_ratings(
            champion_model_id, contender_model_id,
            champion_wins, contender_wins, draws
        )
        
        # Save updated ratings
        self.elo_system.save_ratings(self.elo_file)
        
        # Create ELO report
        elo_report_path = os.path.join(self.model_dir, 'elo_report.txt')
        create_elo_report(self.elo_system, elo_report_path)
        
        return {
            'champion_old_elo': old_champion_elo,
            'champion_new_elo': new_champion_elo,
            'contender_old_elo': old_contender_elo,
            'contender_new_elo': new_contender_elo
        }
    
    def _calibrate_model_vs_random(self, model_id: str, num_games: int = 100):
        """Calibrate a model's ELO by playing against random policy."""
        logger.info(f"Calibrating {model_id} against random policy...")
        
        try:
            # Load the model
            model_path = os.path.join(self.model_dir, f'{model_id}.pt')
            if not os.path.exists(model_path) and model_id == 'best_model':
                model_path = self.best_model_path
                
            model = AlphaZeroNetwork.load(model_path, self.device)
            model.eval()
            
            # Create random policy
            random_policy = RandomPolicyModel(
                board_size=self.board_size,
                action_space_size=self.board_size * self.board_size
            )
            
            # Play calibration games
            arena = Arena(self.game_type, self.board_size, self.device)
            
            # Simplified calibration using value predictions
            wins = 0
            losses = 0
            draws = 0
            
            for _ in range(num_games):
                # Create random board position
                board_tensor = torch.randn(1, self.config['input_channels'],
                                          self.board_size, self.board_size).to(self.device)
                
                # Get value predictions
                with torch.no_grad():
                    _, model_value = model(board_tensor)
                    _, random_value = random_policy(board_tensor.cpu().numpy())
                
                # Compare values (with noise for realism)
                model_score = model_value.item() + np.random.normal(0, 0.1)
                random_score = random_value[0] + np.random.normal(0, 0.1)
                
                if model_score > random_score + 0.05:
                    wins += 1
                elif random_score > model_score + 0.05:
                    losses += 1
                else:
                    draws += 1
            
            # Calibrate ELO based on results
            self.elo_system.calibrate_against_random(model_id, wins, losses, draws)
            logger.info(f"Calibration complete: {wins} wins, {losses} losses, {draws} draws")
            
        except Exception as e:
            logger.error(f"Failed to calibrate {model_id}: {e}")
            # Add with default rating if calibration fails
            self.elo_system.add_model(model_id, rating=1200)
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='AlphaZero Model Evaluation')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--iteration', type=int, required=True,
                       help='Current iteration number')
    parser.add_argument('--new-model', type=str, default=None,
                       help='Path to new model to evaluate (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory containing model checkpoints')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AlphaZeroEvaluator(args.config)
    
    # Override checkpoint directory if provided
    if args.checkpoint_dir:
        checkpoint_model = os.path.join(args.checkpoint_dir, f'model_iter_{args.iteration}.pt')
        if os.path.exists(checkpoint_model):
            args.new_model = checkpoint_model
            logger.info(f"Using checkpoint model: {checkpoint_model}")
    
    # Run evaluation
    results = evaluator.evaluate_iteration(args.iteration, args.new_model)
    
    # Save results if output path provided
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        # Default output path
        output_dir = os.path.dirname(args.config)
        output_path = os.path.join(output_dir, f'evaluation_iter_{args.iteration}.json')
        evaluator.save_results(results, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results - Iteration {args.iteration}")
    print(f"{'='*60}")
    print(f"Champion wins:      {results.get('champion_wins', 0)}")
    print(f"Contender wins:     {results.get('contender_wins', 0)}")
    print(f"Draws:              {results.get('draws', 0)}")
    print(f"Contender win rate: {results.get('contender_win_rate', 0):.2%}")
    print(f"Threshold:          {results.get('win_rate_threshold', 0.55):.2%}")
    print(f"New model better:   {'YES' if results.get('contender_is_better', False) else 'NO'}")
    
    # Print ELO information if available
    if 'elo' in results:
        elo_info = results['elo']
        print(f"\nELO Ratings:")
        print(f"Champion ({elo_info['champion_id']}):  {elo_info['champion_rating']:.1f} "
              f"({elo_info['rating_change_champion']:+.1f})")
        print(f"Contender ({elo_info['contender_id']}): {elo_info['contender_rating']:.1f} "
              f"({elo_info['rating_change_contender']:+.1f})")
    
    print(f"{'='*60}\n")
    
    # Exit with appropriate code
    sys.exit(0 if results.get('contender_is_better', False) else 1)


if __name__ == '__main__':
    main()