"""
AlphaZero Python Package

A Python package for AlphaZero-style self-play, training, and evaluation.
"""

__version__ = '0.1.0'

import sys
import os
from pathlib import Path

# Add binary directories to path for compiled modules
_module_path = Path(__file__).parent

# Try different possible binary locations
_binary_locations = [
    _module_path / '../../build/lib/Release',
    _module_path / '../../build/lib/Debug',
    _module_path / '../../build/lib',
    _module_path / '../../lib',
]

for path in _binary_locations:
    if path.exists():
        sys.path.append(str(path.resolve()))

# Import core functionality
try:
    from .core import AlphaZeroCore, alphazero_py
except ImportError:
    # Handle import error gracefully
    import warnings
    warnings.warn(
        "Failed to import AlphaZero core modules. "
        "Make sure the C++ backend is properly built and installed."
    )

# Import trainer
try:
    from .alphazero_trainer import AlphaZeroTrainer, AlphaZeroNetwork, AlphaZeroDataset
except ImportError:
    # Handle import error gracefully
    import warnings
    warnings.warn(
        "Failed to import AlphaZero trainer modules. "
        "This may be due to missing dependencies or incomplete installation."
    )