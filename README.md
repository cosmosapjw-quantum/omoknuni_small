# Omoknuni: AlphaZero Multi-Game AI Engine

Omoknuni is a high-performance AI engine written in C++ that learns to play board games (Gomoku, Chess, and Go) at an expert level using the AlphaZero algorithm. The engine combines Monte Carlo Tree Search (MCTS) with deep neural networks for position evaluation, all without relying on handcrafted heuristics.

## Key Features

- **Game Abstraction Layer**: Supports Gomoku, Chess, and Go with a unified interface
- **MCTS Engine**: Multi-threaded with virtual loss, progressive widening, and transposition tables
- **Neural Network Integration**: DDW-RandWire-ResNet architecture via libtorch (CUDA)
- **Leaf-Parallelization & Batch Inference**: Efficient GPU utilization with batched evaluations
- **Python CLI & Bindings**: Full pipeline accessible through Python with pybind11
- **Self-Play & Training Pipeline**: End-to-end training workflow

## System Requirements

- **OS**: Linux (64-bit) or Windows (64-bit)
- **Hardware**: Recommended NVIDIA RTX 3060 Ti or better, Ryzen 5900X or equivalent, 64 GB RAM
- **Software**: CUDA toolkit, Python 3.7+, C++17 compiler

## Installation

### Prerequisites

- CMake 3.14 or newer
- CUDA 11.7 or newer
- libtorch 2.0.1 or newer
- Python 3.7+
- pybind11 2.10.4+
- nlohmann_json
- yaml-cpp

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/omoknuni/omoknuni.git
   cd omoknuni
   ```

2. Build the C++ library and Python bindings:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
   cmake --build . --config Release --parallel
   ```

3. Install the Python package:
   ```bash
   cd ..
   pip install -e .
   ```

## Usage

### Configuration

Create a configuration file `config.yaml` with your desired settings:

```yaml
# Game settings
game_type: gomoku
board_size: 15

# Neural network settings
network_type: resnet
num_res_blocks: 19
num_filters: 256

# Self-play settings
self_play_num_games: 100
mcts_num_simulations: 800
mcts_num_threads: 8
```

### Running the AlphaZero Pipeline

From Python:

```python
from alphazero.alphazero_trainer import AlphaZeroTrainer

# Initialize and run the pipeline
trainer = AlphaZeroTrainer(config_path='config.yaml')
trainer.run()
```

From the command line:

```bash
python -m alphazero_main --config config.yaml
```

### Modes

- **Train**: Run the full pipeline (self-play, training, evaluation)
- **Self-Play Only**: Generate games without training
- **Evaluation**: Evaluate a trained model

## License

This project is licensed under the MIT License - see the LICENSE file for details.