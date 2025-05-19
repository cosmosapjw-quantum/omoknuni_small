# Omoknuni: AlphaZero Multi-Game AI Engine

Omoknuni is a high-performance AI engine written in C++ that learns to play board games (Gomoku, Chess, and Go) at an expert level using the AlphaZero algorithm. The engine combines Monte Carlo Tree Search (MCTS) with deep neural networks for position evaluation, all without relying on handcrafted heuristics.

## Key Features

- **Game Abstraction Layer**: Supports Gomoku, Chess, and Go with a unified interface
- **Advanced MCTS Engine**: 
  - Multi-threaded with virtual loss for collision prevention
  - Progressive widening to control tree branching factor
  - RAVE (Rapid Action Value Estimation) for improved action values
  - Root parallelization for running multiple trees simultaneously
  - Transposition tables for position caching
  - Lock-free concurrent queues for efficient communication
- **Neural Network Integration**: DDW-RandWire-ResNet architecture via libtorch (CUDA)
- **Leaf-Parallelization & Batch Inference**: Centralized GPU batch evaluator with configurable batch size and timeout
- **Python CLI & Bindings**: Full pipeline accessible through Python with pybind11
- **Self-Play & Training Pipeline**: End-to-end training workflow with ELO tracking

## Project Structure

```
omoknuni_small/
├── CMakeLists.txt                    # Main build configuration
├── README.md                         # This file
├── config.yaml                       # Main configuration file
├── config_*.yaml                     # Various configuration presets
├── setup.py                          # Python package setup
├── visualization.py                  # Visualization utilities
├── .gitignore
├── .vscode/                          # VS Code workspace settings
├── .claude/                          # Claude AI assistant files
│
├── MCTS_guide.md                     # MCTS implementation guide
├── docs/                             # Documentation files
│   ├── app_flow_document.md
│   ├── app_flowchart.md
│   ├── attack_defense_plane.md
│   ├── backend_structure_document.md
│   ├── frontend_guidelines_document.md
│   ├── go_improvement.md
│   ├── implementation_plan.md
│   ├── package_recommendations.md
│   ├── project_requirements_document.md
│   ├── security_guideline_document.md
│   └── tech_stack_document.md
│
├── include/                          # C++ header files
│   ├── alphazero_export.h
│   ├── cudnn.h
│   │
│   ├── cli/                          # Command-line interface
│   │   ├── alphazero_pipeline.h
│   │   └── cli_manager.h
│   │
│   ├── core/                         # Core game abstractions
│   │   ├── export_macros.h
│   │   ├── game_export.h
│   │   ├── igamestate.h
│   │   └── illegal_move_exception.h
│   │
│   ├── evaluation/                   # Neural network evaluation
│   │   └── model_evaluator.h
│   │
│   ├── games/                        # Game-specific implementations
│   │   ├── chess/
│   │   │   ├── chess_rules.h
│   │   │   ├── chess_state.h
│   │   │   ├── chess_types.h
│   │   │   └── chess960.h
│   │   ├── go/
│   │   │   ├── go_rules.h
│   │   │   └── go_state.h
│   │   └── gomoku/
│   │       ├── gomoku_rules.h
│   │       └── gomoku_state.h
│   │
│   ├── mcts/                         # Monte Carlo Tree Search
│   │   ├── evaluation_types.h
│   │   ├── mcts_engine.h
│   │   ├── mcts_evaluator.h
│   │   ├── mcts_node.h
│   │   ├── node_tracker.h
│   │   └── transposition_table.h
│   │
│   ├── nn/                           # Neural network models
│   │   ├── ddw_randwire_resnet.h
│   │   ├── neural_network.h
│   │   ├── neural_network_factory.h
│   │   └── resnet_model.h
│   │
│   ├── selfplay/                     # Self-play management
│   │   └── self_play_manager.h
│   │
│   ├── training/                     # Training utilities
│   │   ├── data_loader.h
│   │   ├── dataset.h
│   │   └── training_data_manager.h
│   │
│   ├── utils/                        # Utility classes
│   │   ├── attack_defense_module.h
│   │   ├── cuda_utils.h
│   │   ├── debug_monitor.h
│   │   ├── device_utils.h
│   │   ├── gamestate_pool.h
│   │   ├── hash_specializations.h
│   │   ├── memory_tracker.h
│   │   └── zobrist_hash.h
│   │
│   ├── third_party/                  # Third-party libraries
│   │   ├── concurrentqueue.h
│   │   └── lightweightsemaphore.h
│   │
│   └── nlohmann/                     # JSON library (detailed structure omitted)
│
├── src/                              # C++ source files
│   ├── cli/                          # CLI implementations
│   │   ├── alphazero_cli_pipeline.cpp
│   │   ├── alphazero_pipeline.cpp
│   │   ├── cli_manager.cpp
│   │   └── omoknuni_cli.cpp
│   │
│   ├── core/                         # Core implementations
│   │   ├── game_export.cpp
│   │   └── igamestate.cpp
│   │
│   ├── evaluation/                   # Evaluation implementations
│   │   └── model_evaluator.cpp
│   │
│   ├── games/                        # Game implementations
│   │   ├── chess/
│   │   │   ├── chess_rules.cpp
│   │   │   ├── chess_state.cpp
│   │   │   └── chess960.cpp
│   │   ├── go/
│   │   │   ├── go_rules.cpp
│   │   │   └── go_state.cpp
│   │   └── gomoku/
│   │       ├── gomoku_rules.cpp
│   │       └── gomoku_state.cpp
│   │
│   ├── mcts/                         # MCTS implementations
│   │   ├── evaluation_types.cpp
│   │   ├── mcts_engine.cpp
│   │   ├── mcts_evaluator.cpp
│   │   ├── mcts_node.cpp
│   │   ├── mcts_node_pending_eval.cpp
│   │   ├── node_tracker.cpp
│   │   └── transposition_table.cpp
│   │
│   ├── nn/                           # Neural network implementations
│   │   ├── ddw_randwire_resnet.cpp
│   │   ├── neural_network_factory.cpp
│   │   └── resnet_model.cpp
│   │
│   ├── python/                       # Python bindings
│   │   ├── alphazero_bindings.cpp
│   │   └── bindings.cpp
│   │
│   ├── selfplay/                     # Self-play implementations
│   │   └── self_play_manager.cpp
│   │
│   ├── training/                     # Training implementations
│   │   ├── data_loader.cpp
│   │   ├── dataset.cpp
│   │   └── training_data_manager.cpp
│   │
│   └── utils/                        # Utility implementations
│       ├── attack_defense_module.cpp
│       ├── debug_monitor.cpp
│       ├── gamestate_pool.cpp
│       ├── hash_specializations.cpp
│       ├── memory_tracker.cpp
│       └── zobrist_hash.cpp
│
├── python/                           # Python package
│   ├── alphazero/
│   │   ├── __init__.py
│   │   ├── alphazero_trainer.py
│   │   └── core.py
│   ├── alphazero_main.py
│   └── examples/
│       ├── alphazero_training.cpp
│       ├── self_play_libtorch.cpp
│       └── train_alphazero.py
│
├── tests/                            # Test files
│   ├── all_tests_main.cpp
│   ├── core_tests_main.cpp
│   ├── test_main.h
│   │
│   ├── cli/
│   │   └── cli_manager_test.cpp
│   │
│   ├── core/
│   │   ├── game_export_test.cpp
│   │   └── igamestate_test.cpp
│   │
│   ├── evaluation/
│   │   └── model_evaluator_test.cpp
│   │
│   ├── games/
│   │   ├── chess/
│   │   │   └── chess_test.cpp
│   │   ├── go/
│   │   │   └── go_test.cpp
│   │   └── gomoku/
│   │       └── gomoku_test.cpp
│   │
│   ├── integration/
│   │   ├── mcts_with_nn_test.cpp
│   │   ├── self_play_integration_test.cpp
│   │   └── games/
│   │       └── self_play_games_test.cpp
│   │
│   ├── mcts/
│   │   ├── debug_test.cpp
│   │   ├── mcts_engine_test.cpp
│   │   ├── mcts_evaluator_test.cpp
│   │   ├── mcts_node_test.cpp
│   │   ├── transposition_integration_test.cpp
│   │   └── transposition_table_test.cpp
│   │
│   ├── nn/
│   │   └── neural_network_test.cpp
│   │
│   ├── selfplay/
│   │   └── self_play_manager_test.cpp
│   │
│   └── training/
│       └── training_data_manager_test.cpp
│
├── build/                            # Build directory (auto-generated)
├── data/                             # Data storage
│   └── self_play_games/
├── models/                           # Model storage
└── temp/                             # Temporary files
```

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

# MCTS settings
mcts_num_simulations: 800
mcts_num_threads: 12
mcts_batch_size: 256
mcts_batch_timeout_ms: 5
mcts_exploration_constant: 1.4
mcts_virtual_loss: 3

# Progressive widening
mcts_use_progressive_widening: true
mcts_progressive_widening_c: 1.0
mcts_progressive_widening_k: 10.0

# Root parallelization
mcts_use_root_parallelization: true
mcts_num_root_workers: 4

# RAVE settings
mcts_use_rave: true
mcts_rave_constant: 3000.0

# Transposition table
mcts_use_transposition_table: true
mcts_transposition_table_size_mb: 128

# Self-play settings
self_play_num_games: 100
self_play_temperature: 1.0
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

## Performance Tuning

### Recommended Settings by Hardware

**High-End GPU (RTX 3090/4090)**:
```yaml
mcts_batch_size: 512
mcts_num_threads: 16
mcts_num_root_workers: 8
```

**Mid-Range GPU (RTX 3060 Ti)**:
```yaml
mcts_batch_size: 256
mcts_num_threads: 12
mcts_num_root_workers: 4
```

**CPU-Only**:
```yaml
mcts_batch_size: 16
mcts_num_threads: 8
mcts_use_root_parallelization: false
```

## Troubleshooting

### Common Issues

1. **Low GPU Utilization**:
   - Increase `mcts_batch_size`
   - Decrease `mcts_batch_timeout_ms`
   - Check if evaluator thread is running with `nvidia-smi`

2. **Out of Memory**:
   - Reduce `mcts_batch_size`
   - Lower `mcts_max_concurrent_simulations`
   - Decrease neural network size (`num_filters`, `num_res_blocks`)

3. **Slow Search Speed**:
   - Enable transposition tables
   - Use progressive widening
   - Optimize thread count to match CPU cores

4. **Build Errors**:
   - Ensure CUDA toolkit matches libtorch version
   - Check CMake version (>= 3.14)
   - Verify Python development headers are installed

### Debug Options

Enable verbose logging:
```bash
export MCTS_DEBUG=1
export MCTS_VERBOSE=1
```

Monitor performance:
```bash
nvidia-smi dmon -s pucvmet
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.