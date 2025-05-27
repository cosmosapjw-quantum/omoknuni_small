#!/usr/bin/env python3
"""
Prepare training data from self-play games
Converts JSON game files to efficient training format
"""

import os
import json
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Converts self-play games to training data"""
    
    def __init__(self, board_size: int, window_size: int = 500000):
        self.board_size = board_size
        self.window_size = window_size
        self.positions = deque(maxlen=window_size)
        
    def load_game_file(self, file_path: str) -> int:
        """Load a single game file and extract training positions"""
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            # Extract game metadata
            winner = game_data.get('winner', 0)
            moves = game_data.get('moves', [])
            
            # Skip invalid games
            if winner == 0 or len(moves) < 10:
                return 0
            
            # Process each position in the game
            positions_added = 0
            for i, move_data in enumerate(moves):
                # Extract state (board position)
                state = self._parse_state(move_data.get('state', {}))
                
                # Extract policy (move probabilities)
                policy = self._parse_policy(move_data.get('mcts_probs', {}))
                
                # Calculate value based on game outcome
                # Value is from the perspective of the player to move
                player_to_move = move_data.get('player', 1)
                if player_to_move == winner:
                    value = 1.0
                elif winner == 3:  # Draw
                    value = 0.0
                else:
                    value = -1.0
                
                # Add position to training data
                self.positions.append({
                    'state': state,
                    'policy': policy,
                    'value': value,
                    'game_id': game_data.get('game_id', 'unknown'),
                    'move_number': i
                })
                positions_added += 1
            
            return positions_added
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return 0
    
    def _parse_state(self, state_data: Dict) -> np.ndarray:
        """Parse board state into tensor format"""
        # This should match the format expected by your neural network
        # For now, create a placeholder
        
        # Assuming state_data contains board representation
        # You'll need to adapt this based on your actual data format
        
        # Create 19-channel representation (with attack/defense planes)
        channels = 19
        state = np.zeros((channels, self.board_size, self.board_size), dtype=np.float32)
        
        # Parse board positions
        if 'board' in state_data:
            board = state_data['board']
            # Channel 0: current player stones
            # Channel 1: opponent stones
            # Channels 2-15: move history
            # Channel 16: color to play
            # Channel 17: attack plane
            # Channel 18: defense plane
            
            # This is a placeholder - implement based on your format
            pass
        
        return state
    
    def _parse_policy(self, mcts_probs: Dict) -> np.ndarray:
        """Parse MCTS probabilities into policy vector"""
        policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        
        # Convert move probabilities
        for move_str, prob in mcts_probs.items():
            try:
                if move_str.isdigit():
                    move_idx = int(move_str)
                    policy[move_idx] = prob
                else:
                    # Parse coordinate format (e.g., "d4")
                    col = ord(move_str[0]) - ord('a')
                    row = int(move_str[1:]) - 1
                    move_idx = row * self.board_size + col
                    policy[move_idx] = prob
            except:
                continue
        
        # Normalize policy
        if policy.sum() > 0:
            policy /= policy.sum()
        
        return policy
    
    def load_directory(self, directory: str) -> int:
        """Load all game files from a directory"""
        total_positions = 0
        game_files = list(Path(directory).glob("*.json"))
        
        logger.info(f"Found {len(game_files)} game files in {directory}")
        
        for i, game_file in enumerate(game_files):
            positions_added = self.load_game_file(str(game_file))
            total_positions += positions_added
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(game_files)} games, "
                           f"{total_positions} positions total")
        
        return total_positions
    
    def save_training_data(self, output_path: str, shuffle: bool = True):
        """Save training data in efficient format"""
        if len(self.positions) == 0:
            logger.error("No training positions to save")
            return
        
        # Convert to lists
        positions_list = list(self.positions)
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(positions_list)
        
        # Extract arrays
        states = np.array([p['state'] for p in positions_list], dtype=np.float32)
        policies = np.array([p['policy'] for p in positions_list], dtype=np.float32)
        values = np.array([p['value'] for p in positions_list], dtype=np.float32)
        
        # Save as compressed numpy archive
        np.savez_compressed(
            output_path,
            states=states,
            policies=policies,
            values=values,
            metadata={
                'num_positions': len(positions_list),
                'board_size': self.board_size,
                'window_size': self.window_size
            }
        )
        
        logger.info(f"Saved {len(positions_list)} training positions to {output_path}")
        
        # Print statistics
        logger.info(f"Value distribution: "
                   f"Wins: {(values == 1.0).sum()}, "
                   f"Losses: {(values == -1.0).sum()}, "
                   f"Draws: {(values == 0.0).sum()}")


def merge_training_files(input_files: List[str], output_file: str, max_positions: int = 500000):
    """Merge multiple training data files"""
    all_states = []
    all_policies = []
    all_values = []
    
    for file_path in input_files:
        try:
            data = np.load(file_path)
            all_states.append(data['states'])
            all_policies.append(data['policies'])
            all_values.append(data['values'])
            logger.info(f"Loaded {len(data['states'])} positions from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Concatenate all data
    if all_states:
        states = np.concatenate(all_states, axis=0)
        policies = np.concatenate(all_policies, axis=0)
        values = np.concatenate(all_values, axis=0)
        
        # Limit to max positions (keep most recent)
        if len(states) > max_positions:
            states = states[-max_positions:]
            policies = policies[-max_positions:]
            values = values[-max_positions:]
        
        # Shuffle
        indices = np.random.permutation(len(states))
        states = states[indices]
        policies = policies[indices]
        values = values[indices]
        
        # Save merged data
        np.savez_compressed(
            output_file,
            states=states,
            policies=policies,
            values=values,
            metadata={
                'num_positions': len(states),
                'source_files': len(input_files)
            }
        )
        
        logger.info(f"Saved {len(states)} merged positions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare AlphaZero training data')
    parser.add_argument('--input-dir', type=str, help='Directory containing game files')
    parser.add_argument('--input-files', type=str, nargs='+', help='Input training files to merge')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--board-size', type=int, default=9, help='Board size')
    parser.add_argument('--window-size', type=int, default=500000, help='Maximum positions to keep')
    parser.add_argument('--merge', action='store_true', help='Merge existing training files')
    
    args = parser.parse_args()
    
    if args.merge:
        # Merge existing training files
        merge_training_files(args.input_files, args.output, args.window_size)
    else:
        # Convert game files to training data
        preparer = TrainingDataPreparer(args.board_size, args.window_size)
        
        # Load games
        if args.input_dir:
            preparer.load_directory(args.input_dir)
        elif args.input_files:
            for file_path in args.input_files:
                preparer.load_game_file(file_path)
        
        # Save training data
        preparer.save_training_data(args.output)


if __name__ == '__main__':
    main()