#!/usr/bin/env python3
"""Quick script to check model type"""
import torch
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model.pt"

try:
    # Try loading as TorchScript
    model = torch.jit.load(model_path)
    print(f"✅ {model_path} is a TorchScript model")
    print(f"Model type: {type(model)}")
    
    # Check if it has the expected forward signature
    print("\nTrying to get model info...")
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 19, 15, 15)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                print(f"Model outputs: {len(output)} tensors")
                for i, o in enumerate(output):
                    print(f"  Output {i}: shape {o.shape}")
            else:
                print(f"Model output shape: {output.shape}")
    except Exception as e:
        print(f"Error running model: {e}")
        
except Exception as e1:
    print(f"Not a TorchScript model: {e1}")
    
    try:
        # Try as regular PyTorch model
        model = torch.load(model_path, weights_only=False)
        print(f"✅ {model_path} is a regular PyTorch model")
        print(f"Model type: {type(model)}")
    except Exception as e2:
        print(f"Failed to load as regular model: {e2}")