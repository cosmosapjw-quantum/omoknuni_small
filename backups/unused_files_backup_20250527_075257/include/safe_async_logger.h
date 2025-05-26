#ifndef SAFE_ASYNC_LOGGER_H
#define SAFE_ASYNC_LOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "core/export_macros.h"

namespace alphazero {
namespace utils {

/**
 * Thread-safe async logger with proper lifecycle management
 * 
 * Features:
 * - Proper thread pool initialization order
 * - Safe shutdown mechanism
 * - Graceful handling of log messages during shutdown
 * - No segfaults from destroyed thread pools
 */
class ALPHAZERO_API SafeAsyncLogger {
public:
    struct Config {
        std::string log_dir = "logs";
        spdlog::level::level_enum console_level = spdlog::level::info;
        spdlog::level::level_enum file_level = spdlog::level::debug;
        size_t max_file_size = 1048576 * 50; // 50MB
        size_t max_files = 10;
        size_t thread_pool_size = 32768;
        size_t thread_pool_threads = 4;
        bool async_logging = true;
        std::chrono::seconds flush_interval = std::chrono::seconds(3);
    };
    
    // Singleton instance
    static SafeAsyncLogger& getInstance() {
        static SafeAsyncLogger instance;
        return instance;
    }
    
    // Initialize the logger system
    void init(const Config& config = Config());
    
    // Get logger instances
    std::shared_ptr<spdlog::logger> getMCTSLogger();
    std::shared_ptr<spdlog::logger> getNNLogger();
    std::shared_ptr<spdlog::logger> getGameLogger();
    std::shared_ptr<spdlog::logger> getSystemLogger();
    
    // Safe shutdown
    void shutdown();
    
    // Check if safe to log
    bool isSafeToLog() const { return initialized_ && !shutting_down_; }
    
    // Flush all loggers
    void flushAll();
    
private:
    SafeAsyncLogger() = default;
    ~SafeAsyncLogger() { shutdown(); }
    
    // Prevent copying
    SafeAsyncLogger(const SafeAsyncLogger&) = delete;
    SafeAsyncLogger& operator=(const SafeAsyncLogger&) = delete;
    
    // Create logger with proper error handling
    std::shared_ptr<spdlog::logger> createLogger(
        const std::string& name,
        const Config& config);
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutting_down_{false};
    mutable std::mutex mutex_;
    std::condition_variable shutdown_cv_;
    
    // Logger instances
    std::shared_ptr<spdlog::logger> mcts_logger_;
    std::shared_ptr<spdlog::logger> nn_logger_;
    std::shared_ptr<spdlog::logger> game_logger_;
    std::shared_ptr<spdlog::logger> system_logger_;
    
    // Configuration
    Config config_;
};

// Safe logging macros that check shutdown state
#define SAFE_LOG_MCTS_INFO(...) \
    do { \
        if (alphazero::utils::SafeAsyncLogger::getInstance().isSafeToLog()) { \
            auto logger = alphazero::utils::SafeAsyncLogger::getInstance().getMCTSLogger(); \
            if (logger) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__); \
        } \
    } while(0)

#define SAFE_LOG_NN_INFO(...) \
    do { \
        if (alphazero::utils::SafeAsyncLogger::getInstance().isSafeToLog()) { \
            auto logger = alphazero::utils::SafeAsyncLogger::getInstance().getNNLogger(); \
            if (logger) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__); \
        } \
    } while(0)

#define SAFE_LOG_GAME_INFO(...) \
    do { \
        if (alphazero::utils::SafeAsyncLogger::getInstance().isSafeToLog()) { \
            auto logger = alphazero::utils::SafeAsyncLogger::getInstance().getGameLogger(); \
            if (logger) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__); \
        } \
    } while(0)

#define SAFE_LOG_SYSTEM_INFO(...) \
    do { \
        if (alphazero::utils::SafeAsyncLogger::getInstance().isSafeToLog()) { \
            auto logger = alphazero::utils::SafeAsyncLogger::getInstance().getSystemLogger(); \
            if (logger) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__); \
        } \
    } while(0)

} // namespace utils
} // namespace alphazero

#endif // SAFE_ASYNC_LOGGER_H