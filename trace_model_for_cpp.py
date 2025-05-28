#!/usr/bin/env python3
"""
Script to trace PyTorch models for use with libtorch C++
Since torch::jit::trace is not available in C++, models must be traced in Python first.
"""

import torch
import torch.nn as nn
import argparse
import os
from typing import Tuple

class SimpleResNet(nn.Module):
    """Simple ResNet for demonstration - replace with your actual model"""
    def __init__(self, input_channels: int, board_size: int, num_filters: int = 128, num_blocks: int = 10):
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        
        # Input convolution
        self.input_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input
        x = torch.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.relu(x + residual)
        return x

def trace_model(model_path: str = None, output_path: str = "traced_model.pt", 
                board_size: int = 15, input_channels: int = 19, batch_size: int = 1,
                device: str = "cuda"):
    """
    Trace a PyTorch model for use with libtorch C++
    
    Args:
        model_path: Path to existing .pth model file (optional)
        output_path: Path to save traced model
        board_size: Board size for the game
        input_channels: Number of input channels
        batch_size: Batch size for tracing
        device: Device to trace on (cuda or cpu)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            # First try to load as a regular PyTorch model
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("Loaded as regular PyTorch model")
        except Exception as e:
            # If that fails, try loading as TorchScript
            try:
                model = torch.jit.load(model_path, map_location=device)
                print("Model is already a TorchScript archive, no need to trace")
                # Save to output path and return
                model.save(output_path)
                print(f"TorchScript model copied to: {output_path}")
                return
            except Exception as e2:
                print(f"Failed to load model: {e}")
                print(f"Also failed as TorchScript: {e2}")
                raise
    else:
        print("Creating new model for tracing")
        model = SimpleResNet(input_channels, board_size)
        model.to(device)
    
    # Set to eval mode
    model.eval()
    
    # Create example input
    example_input = torch.randn(batch_size, input_channels, board_size, board_size, device=device)
    
    # Trace the model
    print(f"Tracing model with input shape: {example_input.shape}")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for inference (optional)
    print("Optimizing for inference...")
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"Traced model saved to: {output_path}")
    
    # Verify the traced model
    print("Verifying traced model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    # Test with example input
    with torch.no_grad():
        try:
            traced_output = loaded_model(example_input)
            
            # Check output format
            if isinstance(traced_output, (list, tuple)):
                print(f"Model outputs {len(traced_output)} tensors")
                for i, out in enumerate(traced_output):
                    print(f"  Output {i}: shape {out.shape}")
            else:
                print(f"Model output shape: {traced_output.shape}")
                
            print("✅ Traced model verified successfully!")
        except Exception as e:
            print(f"⚠️  Error testing traced model: {e}")
    
    print("Model tracing complete!")
    
    # Generate example C++ code
    print("\nExample C++ code to load this model:")
    print("""
    #include <torch/script.h>
    
    // Load the model
    torch::jit::Module model = torch::jit::load("{0}");
    model.to(torch::kCUDA);  // or torch::kCPU
    model.eval();
    
    // Inference
    torch::NoGradGuard no_grad;
    auto input = torch::randn({{{1}, {2}, {3}, {3}}}, torch::kCUDA);
    
    // Forward pass
    auto outputs = model.forward({{input}}).toTuple();
    auto policy = outputs->elements()[0].toTensor();
    auto value = outputs->elements()[1].toTensor();
    """.format(output_path, batch_size, input_channels, board_size))

def main():
    parser = argparse.ArgumentParser(description="Trace PyTorch models for libtorch C++")
    parser.add_argument("--model", type=str, help="Path to existing model file")
    parser.add_argument("--output", type=str, default="traced_model.pt", help="Output path for traced model")
    parser.add_argument("--board-size", type=int, default=15, help="Board size")
    parser.add_argument("--input-channels", type=int, default=19, help="Number of input channels")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tracing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to trace on")
    
    args = parser.parse_args()
    
    trace_model(
        model_path=args.model,
        output_path=args.output,
        board_size=args.board_size,
        input_channels=args.input_channels,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()