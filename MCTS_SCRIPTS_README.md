# MCTS Mode Scripts

This directory contains convenient shell scripts to run the MCTS system in different modes and benchmark performance.

## Available Scripts

### 🚀 `run_mcts.sh` - Unified MCTS Runner (Recommended)
Automatically detects the best available MCTS mode or allows manual selection.

```bash
# Auto-detect best mode (recommended)
./run_mcts.sh self-play

# Force specific mode
./run_mcts.sh self-play --mcts-mode=gpu
./run_mcts.sh self-play --mcts-mode=cpu

# With custom config
./run_mcts.sh self-play config_low_latency.yaml --verbose

# Training with auto-mode
./run_mcts.sh train --verbose
```

**Features:**
- ✅ Automatic GPU detection and mode selection
- ✅ Graceful fallback from GPU to CPU
- ✅ Comprehensive error checking
- ✅ Performance timing and GPU monitoring
- ✅ Colored output for easy reading

---

### 🖥️ `run_cpu_mcts.sh` - CPU Mode Only
Forces CPU-based MCTS for maximum compatibility.

```bash
# Basic usage
./run_cpu_mcts.sh self-play

# With custom config
./run_cpu_mcts.sh self-play config_optimized_minimal.yaml

# Training mode
./run_cpu_mcts.sh train --verbose

# Evaluation mode
./run_cpu_mcts.sh eval config_minimal_test.yaml
```

**Best for:**
- 🔧 Systems without CUDA/GPU support
- 🛠️ Debugging and development
- 📊 Baseline performance measurements
- 🔒 Maximum stability and compatibility

---

### 🔥 `run_gpu_mcts.sh` - GPU Mode Only
Forces GPU-enhanced MCTS with comprehensive GPU checks.

```bash
# Basic usage (with GPU checks)
./run_gpu_mcts.sh self-play

# Skip GPU compatibility checks
./run_gpu_mcts.sh self-play --skip-gpu-check

# With custom config
./run_gpu_mcts.sh self-play config_optimized_minimal.yaml --verbose

# Training with GPU acceleration
./run_gpu_mcts.sh train --verbose
```

**Features:**
- 🔍 Pre-flight GPU compatibility checks
- 📊 GPU memory and utilization monitoring
- ⚡ GPU-enhanced batch evaluation
- 🌲 GPU tree storage and operations
- 🔄 CUDA graph optimization support

**Requirements:**
- CUDA-capable GPU (Compute Capability 7.0+)
- 4GB+ GPU memory (8GB+ recommended)
- CUDA 11.7+ and cuDNN

---

### 📊 `benchmark_mcts.sh` - Performance Benchmark
Compare CPU vs GPU MCTS performance.

```bash
# Full benchmark (CPU vs GPU)
./benchmark_mcts.sh

# CPU-only benchmark
./benchmark_mcts.sh --cpu-only

# GPU-only benchmark  
./benchmark_mcts.sh --gpu-only

# Custom configuration
./benchmark_mcts.sh --config=config_low_latency.yaml --simulations=200

# Quick benchmark
./benchmark_mcts.sh --simulations=50
```

**Output includes:**
- ⏱️ Execution time comparison
- 🔢 Simulations per second
- 📈 Performance speedup ratios
- 💾 GPU memory usage
- 🎯 Automatic mode recommendations

---

## Quick Start Guide

### 1. First Run (Auto-mode)
```bash
# Let the system choose the best mode
./run_mcts.sh self-play
```

### 2. Performance Comparison
```bash
# Compare CPU vs GPU performance
./benchmark_mcts.sh
```

### 3. Production Usage
```bash
# Use the recommended mode for your system
./run_mcts.sh self-play config_optimized_minimal.yaml --verbose
```

## Mode Comparison

| Feature | CPU Mode | GPU Mode | Auto Mode |
|---------|----------|----------|-----------|
| **Compatibility** | ✅ Universal | 🔶 CUDA required | ✅ Adaptive |
| **Setup** | 🟢 Simple | 🟡 CUDA install | 🟢 Simple |
| **Performance** | 🔵 Baseline | 🚀 2-10x faster | 🎯 Best available |
| **Memory Usage** | 🟢 Low RAM | 🟡 GPU memory | 🎯 Optimal |
| **Debugging** | 🟢 Easy | 🟡 More complex | 🔵 Mode-dependent |

## Configuration Files

The scripts work with various configuration files:

```bash
# Recommended configs by use case
./run_mcts.sh self-play config_optimized_minimal.yaml    # General purpose
./run_mcts.sh self-play config_low_latency.yaml          # Low latency
./run_mcts.sh self-play config_minimal_test.yaml         # Quick testing
./run_mcts.sh train config_alphazero_train.yaml          # Training
```

## Troubleshooting

### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Test GPU mode with checks disabled
./run_gpu_mcts.sh self-play --skip-gpu-check

# Fall back to CPU mode
./run_cpu_mcts.sh self-play
```

### Build Issues
```bash
# Rebuild with CUDA support
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DWITH_TORCH=ON
cmake --build . --config Release --parallel
```

### Configuration Issues
```bash
# List available configs
ls config*.yaml

# Use minimal test config
./run_mcts.sh self-play config_minimal_test.yaml
```

## Environment Variables

```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Disable GPU mode fallback
export FORCE_CPU_ONLY=1

# Enable debug output
export MCTS_DEBUG=1
```

## Performance Tips

### For CPU Mode:
- Use `config_optimized_minimal.yaml`
- Set thread count to match CPU cores
- Consider reducing simulation count for faster iterations

### For GPU Mode:
- Ensure 8GB+ GPU memory for large batch sizes
- Use `config_low_latency.yaml` for optimal GPU utilization
- Monitor GPU temperature and throttling

### For Training:
- Use GPU mode for faster convergence
- Increase batch sizes with GPU mode
- Monitor memory usage during long runs

## Examples

### Development Workflow
```bash
# Quick test
./run_mcts.sh self-play config_minimal_test.yaml

# Performance check
./benchmark_mcts.sh --simulations=100

# Production run
./run_mcts.sh self-play config_optimized_minimal.yaml --verbose
```

### Training Workflow
```bash
# Benchmark first
./benchmark_mcts.sh --command=train

# Use best mode for training
./run_mcts.sh train config_alphazero_train.yaml --verbose
```

### Evaluation Workflow
```bash
# Quick evaluation
./run_mcts.sh eval config_minimal_test.yaml

# Full evaluation with GPU
./run_gpu_mcts.sh eval config_optimized_minimal.yaml --verbose
```