#ifndef ALPHAZERO_MEMORY_ALLOCATOR_H
#define ALPHAZERO_MEMORY_ALLOCATOR_H

// Using mimalloc as the high-performance memory allocator
// This header provides a centralized configuration for memory allocation

#ifdef USE_MIMALLOC
    #include <mimalloc.h>
    // Note: Avoid mimalloc-override.h to prevent conflicts with PyTorch
    // Use mimalloc functions explicitly instead of global overrides
    
    // Optional: Define custom allocation functions for specific use cases
    namespace alphazero {
    namespace memory {
        // Explicit mimalloc functions to avoid global override conflicts
        inline void* malloc(size_t size) {
            return mi_malloc(size);
        }
        
        inline void* calloc(size_t count, size_t size) {
            return mi_calloc(count, size);
        }
        
        inline void* realloc(void* ptr, size_t newsize) {
            return mi_realloc(ptr, newsize);
        }
        
        inline void free(void* ptr) {
            mi_free(ptr);
        }
        
        // Use mimalloc's aligned allocation for SIMD operations
        inline void* aligned_alloc(size_t alignment, size_t size) {
            return mi_aligned_alloc(alignment, size);
        }
        
        inline void aligned_free(void* ptr) {
            mi_free(ptr);
        }
        
        // Memory statistics functions
        inline void print_stats() {
            mi_stats_print(nullptr);
        }
        
        inline size_t get_peak_rss() {
            #if MI_STATS
            mi_stats_t stats;
            mi_stats_get(&stats);
            return stats.peak.allocated;
            #else
            // If stats are not available, return 0
            return 0;
            #endif
        }
    }
    }
#else
    // Fallback to standard allocators if mimalloc is not available
    #include <cstdlib>
    
    namespace alphazero {
    namespace memory {
        inline void* aligned_alloc(size_t alignment, size_t size) {
            #ifdef _WIN32
                return _aligned_malloc(size, alignment);
            #else
                return std::aligned_alloc(alignment, size);
            #endif
        }
        
        inline void aligned_free(void* ptr) {
            #ifdef _WIN32
                _aligned_free(ptr);
            #else
                std::free(ptr);
            #endif
        }
        
        inline void print_stats() {
            // No-op for standard allocator
        }
        
        inline size_t get_peak_rss() {
            return 0; // Not available with standard allocator
        }
    }
    }
#endif

#endif // ALPHAZERO_MEMORY_ALLOCATOR_H