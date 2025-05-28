# Shutdown Fix Summary

## Problem
The aggressive shutdown mechanism was not working properly. When pressing Ctrl+C, the code would stall and not exit. After forcefully exiting the terminal window, there were remaining processes visible in nvidia-smi.

## Root Causes
1. **Blocking GPU operations**: `torch::cuda::synchronize()` and `c10::cuda::CUDACachingAllocator::emptyCache()` were blocking indefinitely
2. **Long sleep intervals**: The memory manager's monitoring loop used non-interruptible sleep
3. **Signal handlers not registered early**: Signal handlers were registered after other initialization
4. **No timeout on GPU cleanup**: GPU synchronization during cleanup could hang forever

## Fixes Applied

### 1. Interruptible Sleep in Memory Manager
```cpp
// Old: std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
// New: Check for shutdown every 100ms during sleep
while (!shutdown_ && std::chrono::steady_clock::now() - sleep_start < sleep_duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

### 2. GPU Operations with Timeout
```cpp
// Wrap GPU sync in a separate thread with timeout
std::atomic<bool> sync_done(false);
std::thread gpu_sync([&sync_done]() {
    try {
        torch::cuda::synchronize();
        sync_done.store(true);
    } catch (...) {}
});

// Wait maximum 2 seconds for GPU sync
auto start = std::chrono::steady_clock::now();
while (!sync_done.load() && 
       std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
```

### 3. Early Signal Handler Registration
```cpp
int main(int argc, char* argv[]) {
    // Register signal handlers FIRST
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Initialize shutdown manager
    utils::initializeShutdownManager();
    
    // Then other initialization...
}
```

### 4. Aggressive Force Exit
- First Ctrl+C: Graceful shutdown with 5-second timeout
- Second Ctrl+C: Immediate force exit using `std::abort()`
- Third Ctrl+C: Absolute force exit using `_exit(1)`
- Watchdog thread automatically forces exit after 5 seconds

### 5. Skip GPU Cleanup on Shutdown
During game generation, GPU cleanup is skipped if shutdown is requested:
```cpp
if (torch::cuda::is_available() && !utils::isShutdownRequested()) {
    // GPU cleanup with timeout
}
```

## Testing
To test the fix:
1. Run self-play: `./bin/Release/omoknuni_cli_final self-play config.yaml`
2. Press Ctrl+C once - should see graceful shutdown message
3. If it doesn't exit within 5 seconds, it will force exit automatically
4. Check `nvidia-smi` - no processes should remain

## Additional Improvements
- All GPU operations now have timeouts
- Shutdown checks added to all long-running loops
- Detached threads for GPU operations that might hang
- Removed blocking GPU sync calls from critical shutdown paths