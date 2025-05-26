#ifndef ALPHAZERO_UTILS_SHUTDOWN_MANAGER_H
#define ALPHAZERO_UTILS_SHUTDOWN_MANAGER_H

#include <atomic>
#include "core/export_macros.h"

namespace alphazero {
namespace utils {

// Global shutdown flag accessible across the application
extern ALPHAZERO_API std::atomic<bool> g_shutdown_requested;

// Initialize the shutdown flag (call once in main)
ALPHAZERO_API void initializeShutdownManager();

// Check if shutdown has been requested
inline bool isShutdownRequested() {
    return g_shutdown_requested.load(std::memory_order_acquire);
}

// Request shutdown
inline void requestShutdown() {
    g_shutdown_requested.store(true, std::memory_order_release);
}

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_UTILS_SHUTDOWN_MANAGER_H