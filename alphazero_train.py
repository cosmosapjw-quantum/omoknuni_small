#!/usr/bin/env python3
"""
AlphaZero Training Pipeline in Python
Handles the neural network training portion of the AlphaZero algorithm
"""

import os
import sys
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from collections import deque
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training data"""
    
    def __init__(self, data_files: List[str], config: dict, max_positions: int = 500000):
        self.config = config
        self.positions = deque(maxlen=max_positions)
        self.board_size = config['board_size']
        self.augment = config['training'].get('use_augmentation', True)
        
        # Load data from files
        for file_path in data_files:
            if file_path.endswith('.npz'):
                self._load_npz(file_path)
            elif file_path.endswith('.json'):
                self._load_json(file_path)
    
    def _load_npz(self, file_path: str):
        """Load data from numpy compressed file"""
        try:
            data = np.load(file_path)
            states = data['states']
            policies = data['policies']
            values = data['values']
            
            for i in range(len(states)):
                self.positions.append((states[i], policies[i], values[i]))
            
            logger.info(f"Loaded {len(states)} positions from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def _load_json(self, file_path: str):
        """Load data from JSON game file"""
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            # Convert game to training positions
            for position in game_data.get('positions', []):
                state = np.array(position['state'], dtype=np.float32)
                policy = np.array(position['policy'], dtype=np.float32)
                value = position['value']
                self.positions.append((state, policy, value))
            
            logger.info(f"Loaded {len(game_data.get('positions', []))} positions from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        state, policy, value = self.positions[idx]
        
        # Apply augmentation if enabled
        if self.augment and random.random() < 0.5:
            state, policy = self._augment(state, policy)
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        )
    
    def _augment(self, state: np.ndarray, policy: np.ndarray):
        """Apply data augmentation (rotation and reflection)"""
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            state = np.rot90(state, k, axes=(1, 2))
            policy = np.rot90(policy.reshape(self.board_size, self.board_size), k)
            policy = policy.flatten()
        
        # Random reflection
        if random.random() < 0.5:
            state = np.flip(state, axis=2)
            policy = np.fliplr(policy.reshape(self.board_size, self.board_size))
            policy = policy.flatten()
        
        return state, policy


class AlphaZeroTrainer:
    """Main trainer class for AlphaZero"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = self.config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize tensorboard
        self.writer = None
        self.global_step = 0
        
        # Training history
        self.history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
    
    def _create_model(self):
        """Create the neural network model"""
        # This is a placeholder - should load the actual C++ model
        # For now, create a simple PyTorch model for demonstration
        
        class SimpleAlphaZeroNet(nn.Module):
            def __init__(self, board_size, input_channels, num_filters, num_blocks):
                super().__init__()
                self.board_size = board_size
                self.input_channels = input_channels
                
                # Initial convolution
                self.initial_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
                self.initial_bn = nn.BatchNorm2d(num_filters)
                
                # Residual blocks
                self.blocks = nn.ModuleList([
                    self._make_residual_block(num_filters)
                    for _ in range(num_blocks)
                ])
                
                # Policy head
                self.policy_conv = nn.Conv2d(num_filters, 32, 1)
                self.policy_bn = nn.BatchNorm2d(32)
                self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)
                
                # Value head
                self.value_conv = nn.Conv2d(num_filters, 32, 1)
                self.value_bn = nn.BatchNorm2d(32)
                self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
                self.value_fc2 = nn.Linear(256, 1)
            
            def _make_residual_block(self, channels):
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels)
                )
            
            def forward(self, x):
                # Initial convolution
                x = F.relu(self.initial_bn(self.initial_conv(x)))
                
                # Residual blocks
                for block in self.blocks:
                    residual = x
                    x = F.relu(block(x) + residual)
                
                # Policy head
                policy = F.relu(self.policy_bn(self.policy_conv(x)))
                policy = policy.view(policy.size(0), -1)
                policy = self.policy_fc(policy)
                
                # Value head
                value = F.relu(self.value_bn(self.value_conv(x)))
                value = value.view(value.size(0), -1)
                value = F.relu(self.value_fc1(value))
                value = torch.tanh(self.value_fc2(value))
                
                return policy, value
        
        return SimpleAlphaZeroNet(
            board_size=self.config['board_size'],
            input_channels=self.config['input_channels'],
            num_filters=self.config['neural_network']['num_filters'],
            num_blocks=self.config['neural_network']['num_res_blocks']
        )
    
    def _create_optimizer(self):
        """Create optimizer"""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if self.config['training']['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.config['training']['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config['training']['optimizer']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config['training']['lr_schedule'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_steps_per_iteration'],
                eta_min=self.config['training']['lr_min']
            )
        elif self.config['training']['lr_schedule'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=0.9
            )
        else:
            return None
    
    def train_iteration(self, data_files: List[str], iteration: int, tensorboard_dir: str):
        """Train for one iteration"""
        logger.info(f"Starting training iteration {iteration}")
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Create dataset and dataloader
        dataset = AlphaZeroDataset(data_files, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training settings
        epochs = self.config['training']['epochs_per_iteration']
        accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        gradient_clip = self.config['training'].get('gradient_clip', 1.0)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            num_batches = 0
            
            for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        pred_policies, pred_values = self.model(states)
                        
                        # Calculate losses
                        policy_loss = F.cross_entropy(pred_policies, target_policies)
                        value_loss = F.mse_loss(pred_values.squeeze(), target_values)
                        
                        total_loss = (
                            self.config['training']['policy_loss_weight'] * policy_loss +
                            self.config['training']['value_loss_weight'] * value_loss
                        )
                        total_loss = total_loss / accumulation_steps
                else:
                    pred_policies, pred_values = self.model(states)
                    
                    # Calculate losses
                    policy_loss = F.cross_entropy(pred_policies, target_policies)
                    value_loss = F.mse_loss(pred_values.squeeze(), target_values)
                    
                    total_loss = (
                        self.config['training']['policy_loss_weight'] * policy_loss +
                        self.config['training']['value_loss_weight'] * value_loss
                    )
                    total_loss = total_loss / accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Update weights
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
                
                # Update metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item() * accumulation_steps
                num_batches += 1
                
                # Log to tensorboard
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/Value', value_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/Total', total_loss.item() * accumulation_steps, self.global_step)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                self.global_step += 1
                
                # Early stopping check
                if self.global_step >= self.config['training']['max_steps_per_iteration']:
                    break
            
            # Log epoch metrics
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Policy Loss: {avg_policy_loss:.4f}, "
                       f"Value Loss: {avg_value_loss:.4f}, "
                       f"Total Loss: {avg_total_loss:.4f}")
            
            self.history['policy_loss'].append(avg_policy_loss)
            self.history['value_loss'].append(avg_value_loss)
            self.history['total_loss'].append(avg_total_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Close tensorboard writer
        self.writer.close()
    
    def save_checkpoint(self, path: str, iteration: int):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('iteration', 0)


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--data', type=str, nargs='+', required=True, help='Training data files')
    parser.add_argument('--iteration', type=int, required=True, help='Training iteration')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
    parser.add_argument('--tensorboard-dir', type=str, required=True, help='TensorBoard directory')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AlphaZeroTrainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train_iteration(args.data, args.iteration, args.tensorboard_dir)
    
    # Save checkpoint
    trainer.save_checkpoint(args.output, args.iteration)
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()