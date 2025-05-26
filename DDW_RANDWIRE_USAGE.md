# DDW-RandWire-ResNet Usage Guide

## Overview

The DDW-RandWire-ResNet is a hybrid architecture that combines:
- **DDW (Differentiable Dynamic Wiring)**: Instance-aware routing that adapts connections based on input
- **RandWire**: Random graph-based network topology
- **ResNet**: Residual connections for training stability

## Quick Start

### 1. Basic Configuration

Create a YAML config file with DDW settings:

```yaml
# Basic DDW configuration
network_type: ddw_randwire
ddw_channels: 128
ddw_num_blocks: 20
ddw_num_nodes: 32
ddw_graph_method: watts_strogatz
ddw_dynamic_routing: true
```

### 2. Run Self-Play

```bash
./bin/Release/omoknuni_cli_final self-play config_ddw_randwire_optimized.yaml
```

## Configuration Parameters

### Network Architecture

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `network_type` | Network architecture type | "resnet" | "ddw_randwire" |
| `ddw_channels` | Number of channels in blocks | 128 | 64-256 |
| `ddw_num_blocks` | Number of RandWire blocks | 20 | 10-30 |
| `ddw_num_nodes` | Nodes per RandWire block | 32 | 16-64 |
| `ddw_dynamic_routing` | Enable instance-aware routing | true | true/false |

### Graph Generation Methods

#### 1. Watts-Strogatz (Small-World)
```yaml
ddw_graph_method: watts_strogatz
ddw_ws_p: 0.75      # Rewiring probability (0.0-1.0)
ddw_ws_k: 4         # Initial neighbors (even number)
```

#### 2. Erdős-Rényi (Random)
```yaml
ddw_graph_method: erdos_renyi
ddw_er_edge_prob: 0.1  # Edge probability (0.0-1.0)
```

#### 3. Barabási-Albert (Scale-Free)
```yaml
ddw_graph_method: barabasi_albert
ddw_ba_m: 5         # Edges per new node
```

## Hardware-Specific Configurations

### RTX 3060 Ti (8GB VRAM) Configurations

#### Conservative (Stable)
```yaml
ddw_channels: 96
ddw_num_blocks: 15
mcts_batch_size: 96
num_parallel_workers: 8
```

#### Aggressive (Maximum Performance)
```yaml
ddw_channels: 128
ddw_num_blocks: 20
mcts_batch_size: 128
num_parallel_workers: 12
```

### Memory Usage Estimation

| Configuration | VRAM Usage | System RAM |
|---------------|------------|------------|
| Conservative | ~6GB | ~16GB |
| Aggressive | ~7.5GB | ~32GB |

## Performance Tuning

### 1. GPU Utilization

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

Target: 85-95% utilization

### 2. Batch Size Optimization

Start with smaller batch sizes and increase:
```yaml
mcts_batch_size: 64   # Start here
mcts_batch_size: 96   # If stable
mcts_batch_size: 128  # Maximum for 8GB
```

### 3. Worker Count

Formula: `workers = min(CPU_threads / 2, GPU_memory / worker_memory)`

For Ryzen 9 5900X:
- Conservative: 8 workers
- Aggressive: 12 workers

## Troubleshooting

### Out of Memory Errors

1. Reduce `ddw_channels` by 32
2. Reduce `mcts_batch_size` by 32
3. Reduce `num_parallel_workers` by 2

### Low GPU Utilization

1. Increase `num_parallel_workers`
2. Increase `mcts_batch_size`
3. Reduce `mcts_batch_timeout_ms`

### Training Instability

1. Enable `ddw_dynamic_routing: false` temporarily
2. Reduce learning rate
3. Use smaller `ddw_num_nodes`

## Advanced Features

### Dynamic Routing Analysis

To visualize routing patterns:
```python
# In training script
model.eval()
with torch.no_grad():
    # Get routing weights
    output = model(input, use_dynamic=True)
    # Analyze edge weights in each block
```

### Graph Structure Comparison

Compare different graph methods:
```bash
# Test each method
for method in watts_strogatz erdos_renyi barabasi_albert; do
    sed -i "s/ddw_graph_method: .*/ddw_graph_method: $method/" config.yaml
    ./run_test.sh
done
```

## Best Practices

1. **Start Conservative**: Begin with optimized config, then tune
2. **Monitor Temperatures**: GPU should stay below 80°C
3. **Save Checkpoints**: Every 1000 games or 1 hour
4. **Profile Performance**: Use `enable_profiling: true` initially
5. **Validate Models**: Test new models before full training

## Expected Performance

With RTX 3060 Ti + Ryzen 9 5900X:

| Metric | Conservative | Aggressive |
|--------|--------------|------------|
| Games/hour | 150-200 | 250-350 |
| GPU Usage | 80-90% | 90-98% |
| Positions/sec | 1500-2000 | 2500-3500 |
| Memory Usage | 6GB VRAM | 7.5GB VRAM |