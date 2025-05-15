#!/usr/bin/env python3
import json
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle

class OmoknuniVisualizer:
    """Visualizer for Omoknuni self-play game data"""
    
    def __init__(self, data_path):
        """Initialize the visualizer with path to a game data file"""
        self.game_data = self.load_game_data(data_path)
        self.current_move = 0
        self.max_moves = len(self.game_data["moves"])
        self.board_size = self.game_data["board_size"]
        self.game_type = self.game_data["game_type"]
        self.game_id = self.game_data["game_id"]
        
        # Game type mapping
        self.game_type_names = {
            1: "Chess",
            2: "Go",
            3: "Gomoku"
        }
        
        # Player colors
        self.player_colors = {
            0: 'None',  # Empty
            1: 'Black',  # BLACK
            2: 'White'   # WHITE
        }
        
        # Set up the visualization
        self.setup_visualization()
        
    def load_game_data(self, data_path):
        """Load game data from JSON file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def action_to_coords(self, action):
        """Convert action index to board coordinates (row, col)"""
        row = action // self.board_size
        col = action % self.board_size
        return row, col
    
    def setup_visualization(self):
        """Set up the matplotlib visualization"""
        # Create figure with GridSpec for better control
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.suptitle(f'Game {self.game_id} - {self.game_type_names.get(self.game_type, "Unknown")}', fontsize=16)
        
        # Create a grid with dedicated colorbar space (narrower colorbar)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 0.85, 0.05])
        self.board_ax = self.fig.add_subplot(gs[0])  # Left subplot
        self.policy_ax = self.fig.add_subplot(gs[1])  # Middle subplot
        self.cbar_ax = self.fig.add_subplot(gs[2])  # Right subplot for colorbar (narrower)
        
        # Set up board plot
        self.board_ax.set_title('Game Board')
        self.board_ax.set_xlim(-1, self.board_size)
        self.board_ax.set_ylim(-1, self.board_size)
        self.board_ax.set_aspect('equal')
        
        # Draw the grid for Gomoku or Go
        if self.game_type in [2, 3]:  # Go or Gomoku
            for i in range(self.board_size):
                self.board_ax.axhline(i, color='black', linewidth=0.5)
                self.board_ax.axvline(i, color='black', linewidth=0.5)
                
            # Label the axes
            self.board_ax.set_xticks(range(self.board_size))
            self.board_ax.set_yticks(range(self.board_size))
            self.board_ax.set_xticklabels([chr(65+i) for i in range(self.board_size)])
            self.board_ax.set_yticklabels([str(i+1) for i in range(self.board_size)])
        
        # Set up policy heatmap plot
        self.policy_ax.set_title('Policy Distribution')
        
        # Button axes for navigation
        self.prev_button_ax = plt.axes([0.2, 0.05, 0.1, 0.04])
        self.next_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        
        # Create navigation buttons
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')
        
        # Connect button click events
        self.prev_button.on_clicked(self.prev_move)
        self.next_button.on_clicked(self.next_move)
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, '', ha='center')
        
        # Initialize board state
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)
        self.stone_artists = []
        
        # For storing the colorbar
        self.policy_colorbar = None
        
        # Draw initial state
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization for the current move"""
        # Clear previous stones
        for artist in self.stone_artists:
            artist.remove()
        self.stone_artists = []
        
        # Reset board state
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Apply moves up to current_move
        current_player = 1  # BLACK starts
        for i in range(self.current_move):
            action = self.game_data["moves"][i]
            row, col = self.action_to_coords(action)
            self.board_state[row, col] = current_player
            current_player = 3 - current_player  # Switch between 1 and 2
        
        # Draw stones for Gomoku
        if self.game_type == 3:  # Gomoku
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board_state[row, col] > 0:
                        color = 'black' if self.board_state[row, col] == 1 else 'white'
                        edgecolor = 'white' if self.board_state[row, col] == 1 else 'black'
                        circle = Circle((col, self.board_size - 1 - row), 0.4, 
                                      facecolor=color, edgecolor=edgecolor, zorder=2)
                        self.board_ax.add_patch(circle)
                        self.stone_artists.append(circle)
                        
                        # Highlight the last move
                        if i == self.current_move - 1 and row == self.action_to_coords(self.game_data["moves"][i])[0] and col == self.action_to_coords(self.game_data["moves"][i])[1]:
                            highlight = Circle((col, self.board_size - 1 - row), 0.2, 
                                             facecolor='red', alpha=0.5, zorder=3)
                            self.board_ax.add_patch(highlight)
                            self.stone_artists.append(highlight)
        
        # Show policy distribution if available
        if "policies" in self.game_data and len(self.game_data["policies"]) > self.current_move:
            policy = self.game_data["policies"][self.current_move]
            policy_grid = np.zeros((self.board_size, self.board_size))
            
            # Convert flat policy to grid
            for i in range(self.board_size * self.board_size):
                if i < len(policy):  # Ensure we don't go out of bounds
                    row, col = self.action_to_coords(i)
                    policy_grid[row, col] = policy[i]
            
            # Clear previous heatmap and retain axis
            self.policy_ax.clear()
            self.policy_ax.set_title('Policy Distribution')
            
            # Show heatmap
            im = self.policy_ax.imshow(policy_grid, cmap='viridis', origin='upper')
            
            # Clear the colorbar axis
            self.cbar_ax.clear()
            
            # Create colorbar in the dedicated axis
            self.policy_colorbar = self.fig.colorbar(im, cax=self.cbar_ax)
            
            # Match the policy grid to the board orientation
            self.policy_ax.set_xticks(range(self.board_size))
            self.policy_ax.set_yticks(range(self.board_size))
            self.policy_ax.set_xticklabels([chr(65+i) for i in range(self.board_size)])
            self.policy_ax.set_yticklabels([str(i+1) for i in range(self.board_size)])
        
        # Update status text
        current_player_text = self.player_colors[1 if self.current_move % 2 == 0 else 2]
        winner_text = ""
        if self.current_move == self.max_moves:
            winner = self.game_data.get("winner", 0)
            winner_text = f" - Winner: {self.player_colors.get(winner, 'Draw')}"
        
        self.status_text.set_text(f'Move {self.current_move}/{self.max_moves} - Next: {current_player_text}{winner_text}')
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def next_move(self, event):
        """Go to next move"""
        if self.current_move < self.max_moves:
            self.current_move += 1
            self.update_visualization()
    
    def prev_move(self, event):
        """Go to previous move"""
        if self.current_move > 0:
            self.current_move -= 1
            self.update_visualization()
    
    def show(self):
        """Show the visualization"""
        # Don't use tight_layout at all - it's causing issues with the colorbar
        # Instead, manually set appropriate margins
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Omoknuni self-play games')
    parser.add_argument('game_file', type=str, nargs='?', 
                        help='Path to a specific game JSON file')
    parser.add_argument('--list', action='store_true',
                        help='List available game files')
    parser.add_argument('--latest', action='store_true',
                        help='Visualize the latest game')
    args = parser.parse_args()
    
    # Default data path
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'selfplay')
    
    if args.list:
        # List available game files
        game_files = sorted(glob.glob(os.path.join(data_path, '*.json')))
        print(f"Found {len(game_files)} game files:")
        for file in game_files:
            print(f"  {os.path.basename(file)}")
        return
    
    # Determine which game file to visualize
    if args.game_file:
        # Use the specified file
        game_file = args.game_file
        if not os.path.exists(game_file):
            # Try looking in the data directory
            game_file = os.path.join(data_path, os.path.basename(game_file))
    elif args.latest:
        # Find the latest game file
        game_files = sorted(glob.glob(os.path.join(data_path, '*.json')))
        if not game_files:
            print("No game files found in data/selfplay directory")
            return
        game_file = game_files[-1]
    else:
        # No file specified, list available games and prompt
        game_files = sorted(glob.glob(os.path.join(data_path, '*.json')))
        if not game_files:
            print("No game files found in data/selfplay directory")
            return
        
        print("Available game files:")
        for i, file in enumerate(game_files):
            print(f"{i+1}. {os.path.basename(file)}")
        
        # Choose a game file
        try:
            choice = int(input("Enter game number to visualize (or 0 to exit): "))
            if choice <= 0 or choice > len(game_files):
                return
            game_file = game_files[choice-1]
        except (ValueError, KeyboardInterrupt):
            return
    
    # Create and show the visualizer
    visualizer = OmoknuniVisualizer(game_file)
    visualizer.show()

if __name__ == "__main__":
    main()