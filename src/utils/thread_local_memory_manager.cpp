#include "utils/thread_local_memory_manager.h"
#include <mutex>

namespace alphazero {
namespace utils {

// Static member definition
std::vector<ThreadLocalMemoryManager*> ThreadLocalMemoryManager::all_managers_;

} // namespace utils
} // namespace alphazero