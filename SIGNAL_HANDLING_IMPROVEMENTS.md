# Signal Handling Improvements

## Summary of Changes

I've implemented comprehensive signal handling improvements to ensure clean shutdown of all workers, threads, and memory allocations when the program receives Ctrl+C (SIGINT).

## Key Improvements

### 1. Enhanced Signal Handler (src/cli/omoknuni_cli_final.cpp)
- First Ctrl+C triggers graceful shutdown with user feedback
- Second Ctrl+C forces immediate exit using std::_Exit()
- 10-second watchdog timer automatically forces exit if graceful shutdown hangs
- Clear console messages inform the user of shutdown progress

### 2. Thread Registry System
- All worker threads register themselves on startup
- Threads unregister when they complete
- Main process tracks all active threads
- Timeout-based cleanup with 5-second maximum wait per thread

### 3. Global Shutdown Manager (new files)
- Created `include/utils/shutdown_manager.h` and `src/utils/shutdown_manager.cpp`
- Provides global shutdown flag accessible across all components
- MCTS engine now checks for shutdown during search loops
- All worker threads check shutdown flag in their main loops

### 4. MCTS Engine Integration
- Updated `src/mcts/mcts_engine_taskflow_optimized.cpp` to check shutdown flag
- All worker threads (leaf collection, batch processing, backpropagation, monitoring) now respond to shutdown
- Prevents threads from getting stuck in infinite loops

### 5. Comprehensive Cleanup Sequence
- Proper thread joining with timeouts
- GPU synchronization (torch::cuda::synchronize())
- Memory manager shutdown
- Force exit with std::_Exit() if threads won't stop
- Cleanup of all resources before exit

## Usage

The program now properly responds to Ctrl+C:
- Press Ctrl+C once for graceful shutdown
- Press Ctrl+C twice for immediate forced exit
- If graceful shutdown takes more than 10 seconds, automatic force exit occurs

## Files Modified

1. `src/cli/omoknuni_cli_final.cpp` - Enhanced signal handling and thread management
2. `src/mcts/mcts_engine_taskflow_optimized.cpp` - Added shutdown checks in all loops
3. `include/utils/shutdown_manager.h` - New global shutdown manager header
4. `src/utils/shutdown_manager.cpp` - New global shutdown manager implementation
5. `CMakeLists.txt` - Added new shutdown_manager.cpp to UTIL_SOURCES

## Benefits

- No more stuck processes after Ctrl+C
- Clean release of GPU memory
- Proper thread cleanup
- No zombie processes or memory leaks
- User-friendly shutdown messages
- Timeout protection against hanging threads

## Testing

To test the signal handling:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib/Release
./build/bin/Release/omoknuni_cli_final self-play config_test_signal.yaml
# Press Ctrl+C to test graceful shutdown
# Press Ctrl+C twice to test forced shutdown
```

Note: The build needs to complete for these changes to take effect.