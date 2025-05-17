#ifndef ALPHAZERO_PROFILER_H
#define ALPHAZERO_PROFILER_H

// Tracy Profiler integration
// This provides nanosecond-resolution profiling for CPU, GPU, memory, and locks

#ifdef TRACY_ENABLE
    #include <tracy/Tracy.hpp>
    #include <tracy/TracyC.h>
    
    // CPU profiling macros
    #define PROFILE_SCOPE(name) ZoneScoped
    #define PROFILE_SCOPE_N(name) ZoneScopedN(name)
    #define PROFILE_SCOPE_C(name, color) ZoneScopedNC(name, color)
    
    // Frame marking for logical work units
    #define PROFILE_FRAME_MARK FrameMark
    #define PROFILE_FRAME_MARK_N(name) FrameMarkNamed(name)
    
    // GPU profiling (requires TRACY_ENABLE_GPU)
    #ifdef TRACY_ENABLE_GPU
        #define PROFILE_GPU_ZONE(name) TracyGpuZone(name)
        #define PROFILE_GPU_COLLECT TracyGpuCollect
    #else
        #define PROFILE_GPU_ZONE(name)
        #define PROFILE_GPU_COLLECT
    #endif
    
    // Memory allocation tracking
    #define PROFILE_ALLOC(ptr, size) TracyAlloc(ptr, size)
    #define PROFILE_FREE(ptr) TracyFree(ptr)
    #define PROFILE_ALLOC_N(ptr, size, name) TracyAllocN(ptr, size, name)
    #define PROFILE_FREE_N(ptr, name) TracyFreeN(ptr, name)
    
    // Lock profiling
    #define PROFILE_LOCKABLE(type, var) TracyLockable(type, var)
    #define PROFILE_LOCKABLE_N(type, var, name) TracyLockableN(type, var, name)
    #define PROFILE_SHARED_LOCKABLE(type, var) TracySharedLockable(type, var)
    #define PROFILE_SHARED_LOCKABLE_N(type, var, name) TracySharedLockableN(type, var, name)
    
    // Custom value plotting
    #define PROFILE_PLOT(name, val) TracyPlot(name, val)
    #define PROFILE_MESSAGE(msg) TracyMessage(msg, strlen(msg))
    #define PROFILE_MESSAGE_L(msg) TracyMessageL(msg)
    
    // App info
    #define PROFILE_SET_THREAD_NAME(name) tracy::SetThreadName(name)
    
#else // TRACY_ENABLE not defined
    
    // No-op macros when profiling is disabled
    #define PROFILE_SCOPE(name)
    #define PROFILE_SCOPE_N(name)
    #define PROFILE_SCOPE_C(name, color)
    
    #define PROFILE_FRAME_MARK
    #define PROFILE_FRAME_MARK_N(name)
    
    #define PROFILE_GPU_ZONE(name)
    #define PROFILE_GPU_COLLECT
    
    #define PROFILE_ALLOC(ptr, size)
    #define PROFILE_FREE(ptr)
    #define PROFILE_ALLOC_N(ptr, size, name)
    #define PROFILE_FREE_N(ptr, name)
    
    #define PROFILE_LOCKABLE(type, var) type var
    #define PROFILE_LOCKABLE_N(type, var, name) type var
    #define PROFILE_SHARED_LOCKABLE(type, var) type var
    #define PROFILE_SHARED_LOCKABLE_N(type, var, name) type var
    
    #define PROFILE_PLOT(name, val)
    #define PROFILE_MESSAGE(msg)
    #define PROFILE_MESSAGE_L(msg)
    
    #define PROFILE_SET_THREAD_NAME(name)
    
#endif // TRACY_ENABLE

namespace alphazero {
namespace utils {

// Helper RAII class for custom profiling zones
class ProfileZone {
public:
#ifdef TRACY_ENABLE
    explicit ProfileZone(const char* name) {
        static const ___tracy_source_location_data loc{name, __FUNCTION__, __FILE__, __LINE__, 0};
        zone_ctx_ = ___tracy_emit_zone_begin_callstack(&loc, 1, true);
    }
    
    ~ProfileZone() {
        ___tracy_emit_zone_end(zone_ctx_);
    }
    
    void SetText(const char* text) {
        ___tracy_emit_zone_text(zone_ctx_, text, strlen(text));
    }
    
    void SetValue(uint64_t value) {
        ___tracy_emit_zone_value(zone_ctx_, value);
    }
    
private:
    TracyCZoneCtx zone_ctx_;
#else
    explicit ProfileZone(const char* name) {}
    void SetText(const char* text) {}
    void SetValue(uint64_t value) {}
#endif
};

// MCTS-specific profiling helpers
inline void ProfileMCTSSimulation(int simulation_count) {
    PROFILE_PLOT("MCTS Simulations", static_cast<int64_t>(simulation_count));
}

inline void ProfileMCTSBatchSize(int batch_size) {
    PROFILE_PLOT("MCTS Batch Size", static_cast<int64_t>(batch_size));
}

inline void ProfileNNInferenceTime(float time_ms) {
    PROFILE_PLOT("NN Inference Time (ms)", static_cast<float>(time_ms));
}

inline void ProfileGPUUtilization(float utilization_percent) {
    PROFILE_PLOT("GPU Utilization %", static_cast<float>(utilization_percent));
}

inline void ProfileMemoryUsage(size_t bytes) {
    PROFILE_PLOT("Memory Usage (MB)", static_cast<int64_t>(bytes / (1024 * 1024)));
}

inline void ProfileTreeDepth(int depth) {
    PROFILE_PLOT("MCTS Tree Depth", static_cast<int64_t>(depth));
}

inline void ProfileNodesPerSecond(float nps) {
    PROFILE_PLOT("Nodes/sec", static_cast<float>(nps));
}

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_PROFILER_H