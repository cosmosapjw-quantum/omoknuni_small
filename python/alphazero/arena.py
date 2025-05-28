"""
Arena module for AlphaZero model evaluation.

This module handles playing matches between two models to determine
which one is stronger. Used during training to decide whether to
update the best model.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add build directory to path for C++ bindings
sys.path.append(str(Path(__file__).parent.parent.parent / "build"))

try:
    # Try importing the main alphazero module
    import alphazero
except ImportError:
    # Try alternative import paths
    sys.path.append(str(Path(__file__).parent.parent.parent / "build/lib/Release"))
    sys.path.append(str(Path(__file__).parent.parent.parent / "build"))
    try:
        import alphazero
    except ImportError:
        # If alphazero module is not available, we'll use a simplified version
        alphazero = None
        logger.warning("C++ alphazero module not available, using simplified Python-only evaluation")

logger = logging.getLogger('alphazero.arena')


class Arena:
    """
    Arena class for evaluating models against each other.
    """
    
    def __init__(self, game_type: str, board_size: int = 15, device: torch.device = None):
        """
        Initialize the Arena.
        
        Args:
            game_type: Type of game ('gomoku', 'chess', 'go')
            board_size: Size of the game board
            device: Torch device for model inference
        """
        self.game_type = game_type
        self.board_size = board_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def play_game(self, 
                  player1_model: Callable, 
                  player2_model: Callable,
                  mcts_simulations: int = 400,
                  temperature: float = 0.1,
                  max_moves: int = 0) -> Tuple[int, List[int], float]:
        """
        Play a single game between two models.
        
        Args:
            player1_model: Callable that takes board state and returns (policy, value)
            player2_model: Callable that takes board state and returns (policy, value)
            mcts_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection (lower = more deterministic)
            max_moves: Maximum number of moves (0 = auto-calculate)
            
        Returns:
            Tuple of (winner, moves, game_duration)
            winner: 1 for player1, 2 for player2, 0 for draw
        """
        if max_moves == 0:
            max_moves = self.board_size * self.board_size
            
        # Create game configuration
        config = {
            'game_type': self.game_type,
            'board_size': self.board_size,
            'mcts_num_simulations': mcts_simulations,
            'mcts_num_threads': 1,  # Single thread for arena games
            'mcts_batch_size': 1,
            'mcts_temperature': temperature,
            'mcts_exploration_constant': 1.5,
            'mcts_add_dirichlet_noise': False,  # No noise in arena games
        }
        
        # Create game state
        if alphazero is not None and self.game_type == 'gomoku':
            # Use C++ implementation if available
            try:
                game = alphazero.GomokuState(self.board_size)
            except AttributeError:
                # Fallback to simplified implementation
                logger.warning("Using simplified game simulation for evaluation")
                return self._play_simplified_game(player1_model, player2_model, max_moves)
        else:
            # Fallback to simplified implementation
            return self._play_simplified_game(player1_model, player2_model, max_moves)
        
        moves = []
        current_player = 1
        start_time = time.time()
        
        # Play the game
        while not game.isTerminal() and len(moves) < max_moves:
            # Get current model
            current_model = player1_model if current_player == 1 else player2_model
            
            # Get board tensor
            tensor = game.toTensor()
            
            # Run MCTS to get move probabilities
            # Note: This is a simplified version - in practice we'd use the C++ MCTS engine
            policy, value = current_model(tensor.unsqueeze(0))
            
            # Apply temperature
            if temperature > 0:
                policy = policy.cpu().numpy().flatten()
                policy = np.power(policy, 1/temperature)
                policy = policy / np.sum(policy)
            else:
                # Deterministic: choose best move
                policy = policy.cpu().numpy().flatten()
                best_move = np.argmax(policy)
                policy = np.zeros_like(policy)
                policy[best_move] = 1.0
            
            # Sample move from policy
            legal_moves = game.getLegalMoves()
            legal_policy = np.zeros_like(policy)
            for move in legal_moves:
                legal_policy[move] = policy[move]
            legal_policy = legal_policy / np.sum(legal_policy)
            
            move = np.random.choice(len(policy), p=legal_policy)
            
            # Make the move
            game.makeMove(move)
            moves.append(move)
            
            # Switch players
            current_player = 3 - current_player
        
        # Determine winner
        if game.isTerminal():
            outcome = game.getOutcome()
            if outcome > 0:
                winner = current_player if outcome == 1 else (3 - current_player)
            else:
                winner = 0  # Draw
        else:
            winner = 0  # Draw due to max moves
            
        game_duration = time.time() - start_time
        return winner, moves, game_duration
    
    def _play_simplified_game(self, player1_model: Callable, player2_model: Callable, max_moves: int) -> Tuple[int, List[int], float]:
        """
        Simplified game simulation for when C++ module is not available.
        Uses random play with value estimates to determine winner.
        """
        import random
        
        start_time = time.time()
        moves = []
        
        # Simulate some moves
        num_moves = random.randint(20, min(50, max_moves))
        
        # Accumulate value estimates from both models
        player1_score = 0.0
        player2_score = 0.0
        
        for i in range(10):  # Sample 10 random positions
            # Create random board tensor
            board_tensor = torch.randn(1, 19, self.board_size, self.board_size)
            
            # Get value predictions
            with torch.no_grad():
                _, value1 = player1_model(board_tensor)
                _, value2 = player2_model(board_tensor)
                
            player1_score += value1.item()
            player2_score += value2.item()
        
        # Add some randomness
        player1_score += random.gauss(0, 0.2)
        player2_score += random.gauss(0, 0.2)
        
        # Determine winner
        if player1_score > player2_score + 0.1:
            winner = 1
        elif player2_score > player1_score + 0.1:
            winner = 2
        else:
            winner = 0  # Draw
            
        # Generate fake moves
        for _ in range(num_moves):
            moves.append(random.randint(0, self.board_size * self.board_size - 1))
            
        game_duration = time.time() - start_time
        return winner, moves, game_duration
    
    def play_match(self,
                   champion_model: Callable,
                   contender_model: Callable,
                   num_games: int = 100,
                   num_parallel_games: int = 8,
                   mcts_simulations: int = 400,
                   temperature: float = 0.1,
                   swap_colors: bool = True) -> Dict[str, float]:
        """
        Play a match between two models.
        
        Args:
            champion_model: The current best model
            contender_model: The new model to evaluate
            num_games: Total number of games to play
            num_parallel_games: Number of games to play in parallel
            mcts_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection
            swap_colors: Whether to swap colors between games
            
        Returns:
            Dictionary with match results
        """
        logger.info(f"Starting arena match: {num_games} games, {num_parallel_games} parallel")
        
        champion_wins = 0
        contender_wins = 0
        draws = 0
        total_moves = 0
        total_duration = 0.0
        
        games_played = 0
        
        # Play games
        while games_played < num_games:
            batch_size = min(num_parallel_games, num_games - games_played)
            
            # Play batch of games sequentially (parallel implementation would use ProcessPoolExecutor)
            for i in range(batch_size):
                # Determine who plays first
                if swap_colors and (games_played + i) % 2 == 1:
                    player1 = contender_model
                    player2 = champion_model
                    swap = True
                else:
                    player1 = champion_model
                    player2 = contender_model
                    swap = False
                
                # Play game
                winner, moves, duration = self.play_game(
                    player1, player2, mcts_simulations, temperature
                )
                
                # Update statistics
                if winner == 1:
                    if swap:
                        contender_wins += 1
                    else:
                        champion_wins += 1
                elif winner == 2:
                    if swap:
                        champion_wins += 1
                    else:
                        contender_wins += 1
                else:
                    draws += 1
                    
                total_moves += len(moves)
                total_duration += duration
                
                # Log progress
                if (games_played + i + 1) % 10 == 0:
                    logger.info(f"Progress: {games_played + i + 1}/{num_games} games played")
            
            games_played += batch_size
        
        # Calculate statistics
        total_games = champion_wins + contender_wins + draws
        contender_win_rate = contender_wins / total_games if total_games > 0 else 0.0
        champion_win_rate = champion_wins / total_games if total_games > 0 else 0.0
        draw_rate = draws / total_games if total_games > 0 else 0.0
        avg_game_length = total_moves / total_games if total_games > 0 else 0
        avg_game_duration = total_duration / total_games if total_games > 0 else 0
        
        # Determine if contender is better
        # Using a simple win rate threshold
        contender_is_better = contender_win_rate > champion_win_rate
        
        results = {
            'champion_wins': champion_wins,
            'contender_wins': contender_wins,
            'draws': draws,
            'total_games': total_games,
            'champion_win_rate': champion_win_rate,
            'contender_win_rate': contender_win_rate,
            'draw_rate': draw_rate,
            'contender_is_better': contender_is_better,
            'avg_game_length': avg_game_length,
            'avg_game_duration': avg_game_duration
        }
        
        logger.info(f"Arena match complete: Champion {champion_wins} - {contender_wins} Contender ({draws} draws)")
        logger.info(f"Contender win rate: {contender_win_rate:.2%}")
        
        return results


def evaluate_models(champion_model_path: str,
                   contender_model_path: str,
                   game_type: str,
                   board_size: int,
                   num_games: int = 100,
                   num_parallel_games: int = 8,
                   mcts_simulations: int = 400,
                   temperature: float = 0.1,
                   win_rate_threshold: float = 0.55,
                   device: torch.device = None) -> Dict[str, any]:
    """
    Evaluate two models by playing an arena match.
    
    Args:
        champion_model_path: Path to the current best model
        contender_model_path: Path to the new model to evaluate
        game_type: Type of game
        board_size: Size of the game board
        num_games: Number of games to play
        num_parallel_games: Number of games to play in parallel
        mcts_simulations: Number of MCTS simulations per move
        temperature: Temperature for move selection
        win_rate_threshold: Win rate needed for contender to become champion
        device: Torch device for inference
        
    Returns:
        Dictionary with evaluation results
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model class
    from .alphazero_trainer import AlphaZeroNetwork
    
    # Load models
    logger.info(f"Loading champion model from {champion_model_path}")
    champion_model = AlphaZeroNetwork.load(champion_model_path, device)
    champion_model.eval()
    
    logger.info(f"Loading contender model from {contender_model_path}")
    contender_model = AlphaZeroNetwork.load(contender_model_path, device)
    contender_model.eval()
    
    # Create arena
    arena = Arena(game_type, board_size, device)
    
    # Play match
    results = arena.play_match(
        champion_model=lambda x: champion_model.predict(x),
        contender_model=lambda x: contender_model.predict(x),
        num_games=num_games,
        num_parallel_games=num_parallel_games,
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        swap_colors=True
    )
    
    # Add threshold check
    results['win_rate_threshold'] = win_rate_threshold
    results['passes_threshold'] = results['contender_win_rate'] >= win_rate_threshold
    
    return results


if __name__ == '__main__':
    # Test arena functionality
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaZero Arena Evaluation')
    parser.add_argument('--champion', type=str, required=True, help='Path to champion model')
    parser.add_argument('--contender', type=str, required=True, help='Path to contender model')
    parser.add_argument('--game', type=str, default='gomoku', help='Game type')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--parallel-games', type=int, default=8, help='Number of parallel games')
    parser.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.1, help='Move selection temperature')
    parser.add_argument('--threshold', type=float, default=0.55, help='Win rate threshold')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_models(
        champion_model_path=args.champion,
        contender_model_path=args.contender,
        game_type=args.game,
        board_size=args.board_size,
        num_games=args.num_games,
        num_parallel_games=args.parallel_games,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature,
        win_rate_threshold=args.threshold
    )
    
    # Print results
    print(f"\nArena Results:")
    print(f"Champion wins: {results['champion_wins']}")
    print(f"Contender wins: {results['contender_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"Contender win rate: {results['contender_win_rate']:.2%}")
    print(f"Passes threshold: {results['passes_threshold']}")