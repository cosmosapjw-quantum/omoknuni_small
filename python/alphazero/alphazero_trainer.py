import os
import sys
import time
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import multiprocessing
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent.parent / "build"))

# Import C++ bindings
try:
    import alphazero_pipeline as azp
except ImportError:
    raise ImportError("Failed to import alphazero_pipeline. Make sure it's built correctly.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/alphazero_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('alphazero')

# AlphaZero dataset for training
class AlphaZeroDataset(Dataset):
    def __init__(self, states, policies, values, device=torch.device('cpu')):
        """
        Initialize the AlphaZero training dataset.
        
        Args:
            states: Tensor of board states [N, C, H, W]
            policies: Tensor of MCTS policies [N, action_space]
            values: Tensor of game results [N]
            device: Torch device to store tensors on
        """
        if isinstance(states, np.ndarray):
            self.states = torch.FloatTensor(states).to(device)
        else:
            self.states = states.to(device)
            
        if isinstance(policies, np.ndarray):
            self.policies = torch.FloatTensor(policies).to(device)
        else:
            self.policies = policies.to(device)
            
        if isinstance(values, np.ndarray):
            self.values = torch.FloatTensor(values).to(device)
        else:
            self.values = values.to(device)
            
        assert len(self.states) == len(self.policies) == len(self.values), "Dataset sizes don't match"

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'policies': self.policies[idx],
            'values': self.values[idx]
        }

# Neural network for AlphaZero
class AlphaZeroNetwork(nn.Module):
    def __init__(self, 
                 game_type: str = 'gomoku',
                 input_channels: int = 20,
                 board_size: int = 15, 
                 num_res_blocks: int = 19,
                 num_filters: int = 256,
                 policy_size: int = None):
        """
        Initialize the AlphaZero neural network with ResNet architecture.
        
        Args:
            game_type: Type of game ('gomoku', 'chess', 'go')
            input_channels: Number of input channels
            board_size: Size of the board (e.g., 15 for 15x15 Gomoku)
            num_res_blocks: Number of residual blocks
            num_filters: Number of filters in convolutional layers
            policy_size: Size of policy head output (if None, calculated from board_size)
        """
        super(AlphaZeroNetwork, self).__init__()
        
        self.game_type = game_type
        self.board_size = board_size
        self.input_channels = input_channels
        
        # Calculate policy size if not provided
        if policy_size is None:
            # Default to board_size^2 + 1 (for pass move in Go)
            self.policy_size = board_size * board_size + 1
        else:
            self.policy_size = policy_size
            
        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._build_residual_block(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, self.policy_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()  # Value output between -1 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_residual_block(self, num_filters):
        """Build a residual block with two convolutional layers."""
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters)
        )
    
    def _initialize_weights(self):
        """Initialize weights according to the AlphaZero paper."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input tensor should be [batch_size, input_channels, board_size, board_size]
        x = self.conv_block(x)
        
        # Residual blocks with skip connections
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = torch.relu(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
        return policy_logits, value
    
    def predict(self, x):
        """Make prediction for C++ callback (expects numpy array input)."""
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
            
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Move to same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Forward pass with no gradient tracking
        with torch.no_grad():
            policy_logits, value = self(x)
            policy = torch.softmax(policy_logits, dim=1)
            
        # Convert back to numpy
        policy_np = policy.cpu().numpy()
        value_np = value.cpu().numpy()
        
        return policy_np, value_np
    
    def save(self, path):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'game_type': self.game_type,
            'board_size': self.board_size,
            'input_channels': self.input_channels,
            'policy_size': self.policy_size,
            'architecture': {
                'num_res_blocks': len(self.res_blocks),
                'num_filters': self.conv_block[0].out_channels
            }
        }, path)
        
    @classmethod
    def load(cls, path, device=None):
        """Load model from disk."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with same architecture
        model = cls(
            game_type=checkpoint.get('game_type', 'gomoku'),
            input_channels=checkpoint.get('input_channels', 20),
            board_size=checkpoint.get('board_size', 15),
            num_res_blocks=checkpoint['architecture']['num_res_blocks'],
            num_filters=checkpoint['architecture']['num_filters'],
            policy_size=checkpoint.get('policy_size', None)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class AlphaZeroTrainer:
    def __init__(self, config_path=None, config_dict=None):
        """
        Initialize the AlphaZero trainer.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Dict of configuration parameters (alternative to config_path)
        """
        # Load configuration
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._default_config()
            
        # Create directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create neural network
        self.model = self._create_neural_network()
        
        # Initialize C++ pipeline
        self.pipeline = azp.AlphaZeroPipeline(self.config)
        self.pipeline.initialize_with_model(self.model.predict)
        
        logger.info("AlphaZero trainer initialized successfully")
        
    def _default_config(self):
        """Create default configuration."""
        return {
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
            "self_play_num_games": 500,
            "self_play_num_parallel_games": 8,
            "self_play_max_moves": 0,  # Auto-calculated
            "self_play_temperature_threshold": 30,
            "self_play_high_temperature": 1.0,
            "self_play_low_temperature": 0.1,
            
            # MCTS settings
            "mcts_num_simulations": 800,
            "mcts_num_threads": 8,
            "mcts_batch_size": 64,
            "mcts_batch_timeout_ms": 20,
            "mcts_exploration_constant": 1.5,
            "mcts_temperature": 1.0,
            "mcts_add_dirichlet_noise": True,
            "mcts_dirichlet_alpha": 0.3,
            "mcts_dirichlet_epsilon": 0.25,
            
            # Training settings
            "train_epochs": 20,
            "train_batch_size": 1024,
            "train_num_workers": 4,
            "train_learning_rate": 0.001,
            "train_weight_decay": 0.0001,
            "train_lr_step_size": 10,
            "train_lr_gamma": 0.1,
            
            # Arena/evaluation settings
            "enable_evaluation": True,
            "arena_num_games": 50,
            "arena_num_parallel_games": 8,
            "arena_num_threads": 4,
            "arena_num_simulations": 400,
            "arena_temperature": 0.1,
            "arena_win_rate_threshold": 0.55
        }
        
    def _create_neural_network(self):
        """Create and initialize the neural network."""
        # Initialize model
        model = AlphaZeroNetwork(
            game_type=self.config['game_type'],
            input_channels=self.config['input_channels'],
            board_size=self.config['board_size'],
            num_res_blocks=self.config['num_res_blocks'],
            num_filters=self.config['num_filters'],
            policy_size=self.config['policy_size'] if self.config['policy_size'] > 0 else None
        )
        
        # Check if there's an existing best model to load
        best_model_path = os.path.join(self.config['model_dir'], 'best_model.pt')
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            try:
                model = AlphaZeroNetwork.load(best_model_path, self.device)
            except Exception as e:
                logger.error(f"Failed to load best model: {e}")
                logger.info("Creating new model instead")
        else:
            logger.info("No existing model found, using fresh model")
            model = model.to(self.device)
            
        return model
    
    def run_iteration(self, iteration):
        """Run a single iteration of the AlphaZero pipeline."""
        logger.info(f"Starting iteration {iteration}")
        start_time = time.time()
        
        # Create iteration directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        iteration_dir = os.path.join(self.config['data_dir'], f"iteration_{iteration}_{timestamp}")
        os.makedirs(iteration_dir, exist_ok=True)
        os.makedirs(os.path.join(iteration_dir, "selfplay"), exist_ok=True)
        os.makedirs(os.path.join(iteration_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(iteration_dir, "evaluation"), exist_ok=True)
        
        # 1. Self-play phase
        logger.info(f"Starting self-play phase for iteration {iteration}")
        games = self._run_self_play(iteration_dir)
        logger.info(f"Self-play completed with {len(games)} games")
        
        # 2. Training phase
        logger.info(f"Starting training phase for iteration {iteration}")
        train_loss = self._train_neural_network(games, iteration_dir)
        logger.info(f"Training completed with final loss: {train_loss}")
        
        # Save latest model
        latest_model_path = os.path.join(self.config['model_dir'], 'latest_model.pt')
        self.model.save(latest_model_path)
        logger.info(f"Saved latest model to {latest_model_path}")
        
        # Also save a copy in the iteration directory
        iter_model_path = os.path.join(iteration_dir, 'training', 'model.pt')
        self.model.save(iter_model_path)
        
        # 3. Evaluation phase (if enabled and not first iteration)
        if self.config['enable_evaluation'] and iteration > 0:
            logger.info(f"Starting evaluation phase for iteration {iteration}")
            evaluation_result = self._evaluate_model(iteration_dir)
            
            if evaluation_result['contender_is_better']:
                logger.info(f"New model from iteration {iteration} is better than previous best")
                best_model_path = os.path.join(self.config['model_dir'], 'best_model.pt')
                self.model.save(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
            else:
                logger.info("Previous best model remains champion")
                # Reload best model
                best_model_path = os.path.join(self.config['model_dir'], 'best_model.pt')
                self.model = AlphaZeroNetwork.load(best_model_path, self.device)
        else:
            # For first iteration, always save as best model
            best_model_path = os.path.join(self.config['model_dir'], 'best_model.pt')
            self.model.save(best_model_path)
            logger.info(f"First iteration model saved as best model to {best_model_path}")
        
        # Log iteration summary
        elapsed_time = time.time() - start_time
        logger.info(f"Iteration {iteration} completed in {elapsed_time:.2f} seconds")
        
        # Write summary to file
        self._log_iteration_summary(iteration, len(games), train_loss, elapsed_time)
        
        return iteration_dir
    
    def _run_self_play(self, iteration_dir):
        """Run self-play to generate games."""
        # Generate games
        num_games = self.config['self_play_num_games']
        logger.info(f"Generating {num_games} self-play games")
        
        games = self.pipeline.run_self_play(num_games)
        logger.info(f"Generated {len(games)} games")
        
        # Save some metadata about the games
        self._save_game_metadata(games, iteration_dir)
        
        return games
    
    def _save_game_metadata(self, games, iteration_dir):
        """Save metadata about self-play games."""
        metadata = {
            "num_games": len(games),
            "winners": {
                "player1": sum(1 for game in games if game.winner == 1),
                "player2": sum(1 for game in games if game.winner == 2),
                "draw": sum(1 for game in games if game.winner == 0)
            },
            "avg_game_length": sum(len(game.moves) for game in games) / len(games) if games else 0
        }
        
        # Save metadata
        with open(os.path.join(iteration_dir, "selfplay", "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f)
    
    def _train_neural_network(self, games, iteration_dir):
        """Train neural network on self-play data."""
        # Convert games to training examples
        logger.info("Converting games to training examples")
        states, policies, values = self.pipeline.extract_training_examples(games)
        
        logger.info(f"Created {len(states)} training examples")
        
        # Create dataset
        dataset = AlphaZeroDataset(states, policies, values)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            num_workers=self.config['train_num_workers'],
            pin_memory=True
        )
        
        # Set up optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['train_learning_rate'],
            weight_decay=self.config['train_weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['train_lr_step_size'],
            gamma=self.config['train_lr_gamma']
        )
        
        # Training loop
        logger.info(f"Starting training for {self.config['train_epochs']} epochs")
        self.model.train()
        
        best_loss = float('inf')
        best_state = None
        final_loss = 0.0
        
        for epoch in range(self.config['train_epochs']):
            # Track metrics
            epoch_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            batch_count = 0
            
            # Iterate through batches
            for batch in data_loader:
                # Move batch to device
                states = batch['states'].to(self.device)
                target_policies = batch['policies'].to(self.device)
                target_values = batch['values'].to(self.device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                policy_logits, value = self.model(states)
                
                # Calculate losses
                # Policy loss: cross entropy
                policy_loss = nn.functional.cross_entropy(policy_logits, target_policies)
                
                # Value loss: MSE
                value_loss = nn.functional.mse_loss(value, target_values)
                
                # Combined loss
                loss = policy_loss + value_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                batch_count += 1
                
                # Log progress occasionally
                if batch_count % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config['train_epochs']}, "
                                f"Batch {batch_count}/{len(data_loader)}, "
                                f"Loss: {loss.item():.4f} "
                                f"(Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")
            
            # Step scheduler
            scheduler.step()
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / batch_count
            avg_policy_loss = policy_loss_sum / batch_count
            avg_value_loss = value_loss_sum / batch_count
            
            logger.info(f"Epoch {epoch+1}/{self.config['train_epochs']} completed. "
                       f"Avg Loss: {avg_epoch_loss:.4f} "
                       f"(Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
            
            # Save model if loss improved
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_state = self.model.state_dict().copy()
                
                epoch_model_path = os.path.join(iteration_dir, "training", f"epoch_{epoch+1}_model.pt")
                self.model.save(epoch_model_path)
                logger.info(f"Saved best epoch model with loss: {best_loss:.4f}")
            
            # Save final epoch result
            final_loss = avg_epoch_loss
        
        # Load best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Loaded best model state with loss: {best_loss:.4f}")
        
        return final_loss
    
    def _evaluate_model(self, iteration_dir):
        """Evaluate latest model against best model."""
        # Create a copy of the current model for evaluation
        contender_model = AlphaZeroNetwork.load(
            os.path.join(self.config['model_dir'], 'latest_model.pt'),
            self.device
        )
        
        # Evaluate models
        logger.info(f"Playing {self.config['arena_num_games']} arena games")
        results = self.pipeline.evaluate_models(contender_model.predict, self.config['arena_num_games'])
        
        # Log results
        logger.info(f"Evaluation complete: Champion wins: {results['champion_wins']}, "
                   f"Contender wins: {results['contender_wins']}, "
                   f"Draws: {results['draws']}")
        logger.info(f"Contender win rate: {results['contender_win_rate']:.2f}, "
                   f"Threshold: {self.config['arena_win_rate_threshold']:.2f}")
        
        # Save results
        results_path = os.path.join(iteration_dir, "evaluation", "results.yaml")
        with open(results_path, 'w') as f:
            yaml.dump(dict(results), f)
        
        return results
    
    def _log_iteration_summary(self, iteration, num_games, train_loss, elapsed_time):
        """Log iteration summary to file."""
        summary_path = os.path.join(self.config['log_dir'], "iteration_summary.csv")
        file_exists = os.path.exists(summary_path)
        
        with open(summary_path, 'a') as f:
            # Write header if new file
            if not file_exists:
                f.write("Iteration,Timestamp,NumGames,TrainLoss,ElapsedTime\n")
            
            # Write data
            f.write(f"{iteration},{datetime.now().strftime('%Y%m%d_%H%M%S')},"
                   f"{num_games},{train_loss:.6f},{elapsed_time:.2f}\n")
    
    def run(self):
        """Run the complete AlphaZero pipeline for all iterations."""
        num_iterations = self.config['num_iterations']
        logger.info(f"Starting AlphaZero pipeline with {num_iterations} iterations")
        
        for i in range(num_iterations):
            try:
                self.run_iteration(i)
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                raise
        
        logger.info("AlphaZero pipeline completed successfully")


def main():
    """Main function to run AlphaZero pipeline from command line."""
    parser = argparse.ArgumentParser(description="AlphaZero Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations to run")
    args = parser.parse_args()
    
    # Create trainer
    trainer = AlphaZeroTrainer(config_path=args.config)
    
    # Override iterations if specified
    if args.iterations is not None:
        trainer.config['num_iterations'] = args.iterations
    
    # Run pipeline
    trainer.run()


if __name__ == "__main__":
    main()