#include "utils/shutdown_manager.h"

namespace alphazero {
namespace utils {

// Definition of the global shutdown flag
std::atomic<bool> g_shutdown_requested(false);

void initializeShutdownManager() {
    g_shutdown_requested.store(false, std::memory_order_release);
}

} // namespace utils
} // namespace alphazero