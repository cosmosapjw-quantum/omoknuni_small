"""
ELO Rating System for AlphaZero Model Evaluation

This module implements an ELO rating system to track model strength over time.
The random policy is anchored at ELO 0, providing a consistent baseline.

Based on the standard ELO formula with modifications for ML model evaluation.
"""

import os
import json
import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger('alphazero.elo')


class ELORatingSystem:
    """
    ELO rating system for tracking model strength.
    
    Uses the standard ELO formula:
    - Expected score: E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    - Rating update: R_new = R_old + K * (S - E)
    
    Where:
    - R is the rating
    - K is the K-factor (learning rate)
    - S is the actual score (1 for win, 0.5 for draw, 0 for loss)
    - E is the expected score
    """
    
    def __init__(self, 
                 k_factor_new: float = 32.0,
                 k_factor_established: float = 16.0,
                 games_until_established: int = 20,
                 scale_factor: float = 400.0,
                 initial_rating: float = 1500.0):
        """
        Initialize the ELO rating system.
        
        Args:
            k_factor_new: K-factor for new models (higher = faster adaptation)
            k_factor_established: K-factor for established models
            games_until_established: Number of games before a model is considered established
            scale_factor: Scale factor for probability calculation (standard is 400)
            initial_rating: Initial rating for new models (before anchoring)
        """
        self.k_factor_new = k_factor_new
        self.k_factor_established = k_factor_established
        self.games_until_established = games_until_established
        self.scale_factor = scale_factor
        self.initial_rating = initial_rating
        
        # Model ratings and game counts
        self.ratings: Dict[str, float] = {}
        self.game_counts: Dict[str, int] = {}
        self.rating_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Initialize random policy as anchor at ELO 0
        self.add_model("random_policy", rating=0.0)
        
    def add_model(self, model_id: str, rating: Optional[float] = None) -> float:
        """
        Add a new model to the rating system.
        
        Args:
            model_id: Unique identifier for the model
            rating: Initial rating (if None, uses default initial rating)
            
        Returns:
            The initial rating assigned
        """
        if model_id in self.ratings:
            logger.warning(f"Model {model_id} already exists, keeping current rating")
            return self.ratings[model_id]
            
        if rating is None:
            rating = self.initial_rating
            
        self.ratings[model_id] = rating
        self.game_counts[model_id] = 0
        self.rating_history[model_id] = [(datetime.now(), rating)]
        
        logger.info(f"Added model {model_id} with initial rating {rating}")
        return rating
        
    def get_k_factor(self, model_id: str) -> float:
        """Get the appropriate K-factor for a model based on game count."""
        games_played = self.game_counts.get(model_id, 0)
        if games_played < self.games_until_established:
            return self.k_factor_new
        return self.k_factor_established
        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Uses the standard ELO probability formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / scale))
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / self.scale_factor))
        
    def update_ratings(self, model_a: str, model_b: str, 
                      wins_a: int, wins_b: int, draws: int) -> Tuple[float, float]:
        """
        Update ratings based on match results.
        
        Args:
            model_a: ID of first model
            model_b: ID of second model
            wins_a: Number of wins for model A
            wins_b: Number of wins for model B
            draws: Number of draws
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Ensure models exist
        if model_a not in self.ratings:
            self.add_model(model_a)
        if model_b not in self.ratings:
            self.add_model(model_b)
            
        # Get current ratings
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        
        # Calculate total games
        total_games = wins_a + wins_b + draws
        if total_games == 0:
            logger.warning("No games played, ratings unchanged")
            return rating_a, rating_b
            
        # Calculate actual scores
        score_a = (wins_a + 0.5 * draws) / total_games
        score_b = (wins_b + 0.5 * draws) / total_games
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a  # = self.expected_score(rating_b, rating_a)
        
        # Get K-factors
        k_a = self.get_k_factor(model_a)
        k_b = self.get_k_factor(model_b)
        
        # Update ratings
        new_rating_a = rating_a + k_a * total_games * (score_a - expected_a)
        new_rating_b = rating_b + k_b * total_games * (score_b - expected_b)
        
        # Apply updates
        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b
        
        # Update game counts
        self.game_counts[model_a] += total_games
        self.game_counts[model_b] += total_games
        
        # Update history
        now = datetime.now()
        self.rating_history[model_a].append((now, new_rating_a))
        self.rating_history[model_b].append((now, new_rating_b))
        
        # Log update
        logger.info(f"ELO Update: {model_a} {rating_a:.1f} -> {new_rating_a:.1f} "
                   f"({'+' if new_rating_a > rating_a else ''}{new_rating_a - rating_a:.1f})")
        logger.info(f"ELO Update: {model_b} {rating_b:.1f} -> {new_rating_b:.1f} "
                   f"({'+' if new_rating_b > rating_b else ''}{new_rating_b - rating_b:.1f})")
        
        return new_rating_a, new_rating_b
        
    def get_rating(self, model_id: str) -> Optional[float]:
        """Get current rating for a model."""
        return self.ratings.get(model_id)
        
    def get_all_ratings(self) -> Dict[str, float]:
        """Get all current ratings."""
        return self.ratings.copy()
        
    def get_sorted_ratings(self) -> List[Tuple[str, float]]:
        """Get models sorted by rating (highest first)."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        
    def save_ratings(self, filepath: str):
        """Save rating data to file."""
        data = {
            'ratings': self.ratings,
            'game_counts': self.game_counts,
            'config': {
                'k_factor_new': self.k_factor_new,
                'k_factor_established': self.k_factor_established,
                'games_until_established': self.games_until_established,
                'scale_factor': self.scale_factor,
                'initial_rating': self.initial_rating
            },
            'history': {
                model_id: [(t.isoformat(), r) for t, r in history]
                for model_id, history in self.rating_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved ELO ratings to {filepath}")
        
    def load_ratings(self, filepath: str):
        """Load rating data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.ratings = data['ratings']
        self.game_counts = data['game_counts']
        
        # Load config if present
        if 'config' in data:
            config = data['config']
            self.k_factor_new = config.get('k_factor_new', self.k_factor_new)
            self.k_factor_established = config.get('k_factor_established', self.k_factor_established)
            self.games_until_established = config.get('games_until_established', self.games_until_established)
            self.scale_factor = config.get('scale_factor', self.scale_factor)
            self.initial_rating = config.get('initial_rating', self.initial_rating)
            
        # Load history if present
        if 'history' in data:
            self.rating_history = {
                model_id: [(datetime.fromisoformat(t), r) for t, r in history]
                for model_id, history in data['history'].items()
            }
        else:
            # Initialize empty history
            self.rating_history = {
                model_id: [(datetime.now(), rating)]
                for model_id, rating in self.ratings.items()
            }
            
        logger.info(f"Loaded ELO ratings from {filepath}")
        
    def calibrate_against_random(self, model_id: str, wins: int, losses: int, draws: int):
        """
        Calibrate a model's rating based on performance against random policy.
        
        This establishes the initial rating more accurately than the default.
        """
        total_games = wins + losses + draws
        if total_games == 0:
            return
            
        win_rate = (wins + 0.5 * draws) / total_games
        
        # Calculate implied rating difference using inverse of ELO formula
        # win_rate = 1 / (1 + 10^(-diff/400))
        # Solving for diff: diff = 400 * log10(win_rate / (1 - win_rate))
        
        if win_rate == 1.0:
            rating_diff = 800  # Cap at very high difference
        elif win_rate == 0.0:
            rating_diff = -800  # Cap at very low difference
        else:
            rating_diff = self.scale_factor * math.log10(win_rate / (1 - win_rate))
            
        # Set rating relative to random policy (ELO 0)
        calibrated_rating = 0 + rating_diff
        
        self.ratings[model_id] = calibrated_rating
        self.rating_history[model_id].append((datetime.now(), calibrated_rating))
        
        logger.info(f"Calibrated {model_id} to ELO {calibrated_rating:.1f} "
                   f"based on {win_rate:.1%} win rate vs random")


class RandomPolicyModel:
    """
    Random policy model that serves as the ELO 0 baseline.
    """
    
    def __init__(self, board_size: int = 15, action_space_size: int = None):
        """
        Initialize random policy model.
        
        Args:
            board_size: Size of the game board
            action_space_size: Size of action space (if None, uses board_size^2)
        """
        self.board_size = board_size
        self.action_space_size = action_space_size or (board_size * board_size)
        
    def predict(self, board_state):
        """
        Return uniform random policy and neutral value.
        
        Args:
            board_state: Current board state (ignored)
            
        Returns:
            Tuple of (policy, value) where policy is uniform and value is 0
        """
        # Uniform random policy
        policy = np.ones(self.action_space_size) / self.action_space_size
        
        # Neutral value (no preference)
        value = 0.0
        
        return policy, np.array([value])
        
    def __call__(self, board_state):
        """Allow model to be called directly."""
        return self.predict(board_state)


def create_elo_report(elo_system: ELORatingSystem, output_path: str = "elo_report.txt"):
    """
    Create a human-readable ELO rating report.
    
    Args:
        elo_system: The ELO rating system
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AlphaZero Model ELO Ratings Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Current ratings
        f.write("Current Model Rankings:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Rank':<6} {'Model':<20} {'ELO':<8} {'Games':<8}\n")
        f.write("-"*40 + "\n")
        
        sorted_ratings = elo_system.get_sorted_ratings()
        for rank, (model_id, rating) in enumerate(sorted_ratings, 1):
            games = elo_system.game_counts[model_id]
            f.write(f"{rank:<6} {model_id:<20} {rating:>7.1f} {games:>7}\n")
            
        # Rating differences from baseline
        f.write("\n\nRating Differences from Random Policy (ELO 0):\n")
        f.write("-"*40 + "\n")
        
        random_rating = elo_system.get_rating("random_policy") or 0
        for model_id, rating in sorted_ratings:
            if model_id != "random_policy":
                diff = rating - random_rating
                win_prob = elo_system.expected_score(rating, random_rating)
                f.write(f"{model_id:<20} {diff:>+8.1f} (Win prob: {win_prob:.1%})\n")
                
        f.write("\n" + "="*60 + "\n")
        
    logger.info(f"Created ELO report: {output_path}")


if __name__ == "__main__":
    # Example usage
    elo = ELORatingSystem()
    
    # Add some models
    elo.add_model("model_v1", rating=1200)
    elo.add_model("model_v2", rating=1200)
    
    # Calibrate model_v1 against random
    elo.calibrate_against_random("model_v1", wins=75, losses=20, draws=5)
    
    # Update based on match between v1 and v2
    elo.update_ratings("model_v1", "model_v2", wins_a=55, wins_b=40, draws=5)
    
    # Show current ratings
    print("\nCurrent Ratings:")
    for model, rating in elo.get_sorted_ratings():
        print(f"{model}: {rating:.1f}")
        
    # Save ratings
    elo.save_ratings("test_elo_ratings.json")
    
    # Create report
    create_elo_report(elo, "test_elo_report.txt")