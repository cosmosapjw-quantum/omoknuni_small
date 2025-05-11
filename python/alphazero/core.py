# python/alphazero/core.py
import os
import sys
import numpy as np
import time
from pathlib import Path

# Add binary directory to path
def _find_module():
    # Try common locations
    search_paths = [
        Path(__file__).parent / "../../build/lib/Release",
        Path(__file__).parent / "../../build/lib/Debug",
        Path(__file__).parent / "../../build/lib",
        Path(__file__).parent / "../../lib",
    ]
    
    for path in search_paths:
        path = path.resolve()
        if path.exists():
            return str(path)
            
    raise ImportError("Could not find alphazero_py module")

sys.path.append(_find_module())

try:
    import alphazero_py
except ImportError:
    raise ImportError("Failed to import alphazero_py. Make sure it's built correctly.")

class AlphaZeroCore:
    """High-level wrapper for the AlphaZero C++ implementation"""
    
    def __init__(self, neural_network, mcts_settings=None):
        """
        Initialize the AlphaZero core.
        
        Args:
            neural_network: A callable that takes a batch of board tensors and returns (policy, value) tuple
            mcts_settings: Optional MCTSSettings object
        """
        self.neural_network = neural_network
        
        # Create default settings if none provided
        if mcts_settings is None:
            mcts_settings = alphazero_py.MCTSSettings()
            mcts_settings.num_simulations = 800
            mcts_settings.num_threads = 4
            mcts_settings.batch_size = 8
            mcts_settings.exploration_constant = 1.5
        
        # Create self-play manager
        self.self_play = alphazero_py.SelfPlayManager(mcts_settings, self._inference_callback)
    
    def _inference_callback(self, tensor):
        """Wrapper for neural network inference"""
        policy, value = self.neural_network(tensor)
        return policy, value
    
    def generate_game(self, game_type, max_moves=1000):
        """
        Generate a self-play game.
        
        Args:
            game_type: String name of game ("chess", "go", "gomoku")
            max_moves: Maximum number of moves before forced draw
            
        Returns:
            Tuple of (moves, policies, winner)
        """
        return self.self_play.generate_game(game_type, max_moves)
    
    def evaluate_position(self, game_type, moves):
        """
        Evaluate a position after a sequence of moves.
        
        Args:
            game_type: String name of game
            moves: List of integer moves
            
        Returns:
            Dictionary with value, policy, nodes, time_ms
        """
        return self.self_play.evaluate_position(game_type, moves)
    
    def get_last_stats(self):
        """Get statistics from the last search"""
        return self.self_play.get_last_stats()
    
    @staticmethod
    def create_game(game_type):
        """Create a game state object"""
        if isinstance(game_type, str):
            game_type = {
                "chess": alphazero_py.GameType.CHESS,
                "go": alphazero_py.GameType.GO,
                "gomoku": alphazero_py.GameType.GOMOKU
            }.get(game_type.lower())
            
            if game_type is None:
                raise ValueError(f"Unknown game type: {game_type}")
        
        return alphazero_py.create_game(game_type)

# Example usage
if __name__ == "__main__":
    # Create a dummy neural network
    class DummyNetwork:
        def __call__(self, tensor):
            batch_size = tensor.shape[0]
            action_size = 19*19  # Gomoku
            
            policy = np.ones((batch_size, action_size)) / action_size
            value = np.zeros(batch_size)
            
            return policy, value
    
    # Create AlphaZero core
    nn = DummyNetwork()
    az = AlphaZeroCore(nn)
    
    # Generate a game
    start_time = time.time()
    moves, policies, winner = az.generate_game("gomoku", max_moves=10)
    end_time = time.time()
    
    print(f"Generated game with {len(moves)} moves in {end_time - start_time:.2f} seconds")
    print(f"Winner: {'Draw' if winner == 0 else f'Player {winner}'}")
    
    # Evaluate a position
    result = az.evaluate_position("gomoku", [])
    print(f"Start position evaluation: value={result['value']:.4f}, nodes={result['nodes']}")