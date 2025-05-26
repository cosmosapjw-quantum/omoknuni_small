# GPU Memory Optimization Summary

## Problems Identified

### 1. VRAM Memory Leaks
- **torch::from_blob** in GPUMemoryPool created unmanaged tensors that weren't properly freed
- TensorPool only cleaned up every 10 inferences, causing accumulation
- Missing memory tracking on exception paths
- No aggressive GPU cleanup after inference

### 2. CPU/GPU Imbalance
- Small batch sizes (32-64) underutilized RTX 3060 Ti's 4864 CUDA cores
- Blocking CUDA synchronization after each batch
- No pipelining between CPU tree search and GPU evaluation
- CPU threads blocked waiting for GPU operations

## Implemented Fixes

### Memory Leak Fixes

1. **Replace torch::from_blob (gpu_memory_pool.cpp:267-290)**
   - Changed to create properly managed tensors with torch::empty
   - Added explicit memory copy to ensure lifecycle management
   - Prevents dangling references to GPU memory

2. **Aggressive GPU Cleanup (resnet_model.cpp:763-791)**
   - Clear tensor pool after EVERY inference (not just every 10)
   - Force CUDA cache cleanup after each inference
   - Added device synchronization across all GPUs
   - Cleanup interval reduced from 10 to 5 for periodic cleanups

3. **RAII Memory Guards (resnet_model.cpp:289-300)**
   - Added CPUTensorGuard for automatic memory tracking
   - Ensures CPU tensors are properly tracked and freed
   - Handles exception cases properly

4. **Exception Cleanup (resnet_model.cpp:817-841)**
   - Added GPU cleanup in all exception handlers
   - Ensures memory is freed even on errors
   - Prevents accumulation from failed inferences

5. **GPU Cleanup Callback (mcts_engine_main.cpp:128-153)**
   - Registered high-priority GPU memory cleanup with AggressiveMemoryManager
   - Cleans GPU memory pool, neural network tensors, and CUDA cache
   - Triggered on memory pressure warnings

### CPU/GPU Balance Fixes

1. **Increased Batch Sizes (mcts_engine_simple_batch.cpp:63-70)**
   - Minimum batch size: 64 (up from flexible sizing)
   - Maximum batch size: 256 (to prevent OOM)
   - Better utilizes RTX 3060 Ti's parallel processing

2. **Asynchronous Pipeline (mcts_engine_simple_batch.cpp:148-230)**
   - GPU inference runs asynchronously while CPU prepares next batch
   - Uses std::async for non-blocking inference
   - Half CPU threads prepare next batch while GPU processes current
   - Eliminates CPU idle time during GPU processing

3. **Removed Blocking Sync (mcts_engine_simple_batch.cpp:186-200)**
   - Changed from blocking torch::cuda::synchronize() every batch
   - Non-blocking cleanup only every 20 batches
   - Allows GPU to continue processing without interruption

## Expected Results

### Memory Usage
- VRAM should stabilize instead of continuously growing
- Peak VRAM usage reduced by aggressive cleanup
- No memory leaks on long-running processes

### Performance
- 2-4x improvement in GPU utilization
- Higher throughput (states/second)
- Better CPU/GPU overlap reducing total time

### Monitoring
- Run with `./run_optimized_gpu_fix.sh`
- Check `gpu_monitor_optimized.log` for VRAM usage
- Compare inference times in console output

## Hardware-Specific Optimizations

For RTX 3060 Ti (8GB VRAM, 4864 CUDA cores):
- Batch size 128 optimal for memory/performance balance
- 2GB GPU memory pool allocation
- 20 CPU threads (leaving 4 for system/GPU management)
- Aggressive cleanup prevents hitting 8GB limit

## Configuration Changes

See `config_optimized_gpu_fix.yaml`:
- `mcts_batch_size: 128`
- `mcts_batch_timeout_ms: 5`
- `enable_aggressive_memory_cleanup: true`
- `memory_cleanup_interval_batches: 5`
- `gpu_memory_pool_size_mb: 2048`
- `enable_async_inference: true`