# python/examples/train_alphazero.py
import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))
from alphazero.core import AlphaZeroCore, alphazero_py

# Neural network model for Gomoku
class GomokuNetwork(nn.Module):
    def __init__(self, board_size=15, num_channels=128):
        super(GomokuNetwork, self).__init__()
        
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # Common layers
        self.conv1 = nn.Conv2d(17, num_channels, 3, padding=1)  # 17 input planes for Gomoku
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.resblocks = nn.ModuleList([
            self._build_residual_block(num_channels) for _ in range(10)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, self.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def _build_residual_block(self, num_channels):
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
    
    def forward(self, x):
        # Common layers
        x = torch.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.resblocks:
            residual = x
            x = torch.relu(x + block(x))
        
        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = torch.log_softmax(policy, dim=1)
        
        # Value head
        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * self.board_size * self.board_size)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, x):
        """Wrapper for inference during self-play"""
        x = torch.FloatTensor(x)
        with torch.no_grad():
            policy, value = self(x)
            return torch.softmax(policy, dim=1).numpy(), value.numpy()

# Dataset for training
class SelfPlayDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

# Example training loop
def train(model, dataset, args):
    """Simple training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        
        for states, target_policies, target_values in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            
            # Forward pass
            policy_logits, value = model(states)
            
            # Compute loss
            policy_loss = -torch.sum(target_policies * policy_logits) / target_policies.size(0)
            value_loss = torch.mean((value - target_values) ** 2)
            loss = policy_loss + value_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Loss={total_loss/len(dataloader):.4f}, "
              f"Policy={policy_loss_sum/len(dataloader):.4f}, "
              f"Value={value_loss_sum/len(dataloader):.4f}")
    
    return model

# Self-play data generation
def generate_self_play_data(args):
    """Generate self-play data using the latest model"""
    # Load model
    model = GomokuNetwork(board_size=args.board_size)
    
    # Load weights if available
    model_path = Path(args.model_dir) / "latest_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    # Initialize AlphaZero core
    mcts_settings = alphazero_py.MCTSSettings()
    mcts_settings.num_simulations = args.num_simulations
    mcts_settings.num_threads = args.num_threads
    mcts_settings.batch_size = args.batch_size
    mcts_settings.exploration_constant = args.exploration
    mcts_settings.add_dirichlet_noise = True
    
    az = AlphaZeroCore(model.predict, mcts_settings)
    
    # Generate games
    all_states = []
    all_policies = []
    all_values = []
    
    for i in range(args.num_games):
        print(f"Generating game {i+1}/{args.num_games}...")
        start_time = time.time()
        
        # Generate a game
        moves, policies, winner = az.generate_game("gomoku", max_moves=args.max_moves)
        
        end_time = time.time()
        print(f"Game completed in {end_time - start_time:.2f}s, "
              f"{len(moves)} moves, winner: {'Draw' if winner == 0 else f'Player {winner}'}")
        
        # Convert moves to board states
        game = az.create_game("gomoku")
        states = []
        
        # Reset game
        for move in moves:
            # Store state
            states.append(game.get_enhanced_tensor_representation())
            
            # Make move
            game.make_move(move)
        
        # Calculate game values based on winner
        values = []
        if winner == 0:  # Draw
            values = [0.0] * len(moves)
        else:
            # Set value based on whether each player won
            player = 1
            for _ in range(len(moves)):
                values.append(1.0 if player == winner else -1.0)
                player = 3 - player  # Toggle between 1 and 2
        
        # Store data
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
    
    # Convert to numpy arrays
    states_np = np.array(all_states)
    policies_np = np.array(all_policies)
    values_np = np.array(all_values).reshape(-1, 1)
    
    print(f"Generated {len(states_np)} training examples")
    return states_np, policies_np, values_np

# Main training loop
def main(args):
    """Main training loop"""
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize model
    model = GomokuNetwork(board_size=args.board_size)
    
    # Load existing model if available
    model_path = Path(args.model_dir) / "latest_model.pt"
    if model_path.exists() and not args.new_model:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    # Training loop
    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration+1}/{args.iterations} ===")
        
        # Generate self-play data
        states, policies, values = generate_self_play_data(args)
        
        # Create dataset
        dataset = SelfPlayDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(policies),
            torch.FloatTensor(values)
        )
        
        # Train model
        model = train(model, dataset, args)
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Also save a snapshot for this iteration
        torch.save(model.state_dict(), Path(args.model_dir) / f"model_iter_{iteration+1}.pt")
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=5, help="Number of training iterations")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # Self-play parameters
    parser.add_argument("--num_games", type=int, default=5, help="Number of self-play games per iteration")
    parser.add_argument("--num_simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of MCTS threads")
    parser.add_argument("--exploration", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--max_moves", type=int, default=200, help="Maximum moves per game")
    
    # Game parameters
    parser.add_argument("--board_size", type=int, default=15, help="Gomoku board size")
    
    # Misc
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--new_model", action="store_true", help="Start with a new model")
    
    args = parser.parse_args()
    
    main(args)