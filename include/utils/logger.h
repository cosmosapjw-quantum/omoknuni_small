#ifndef ALPHAZERO_LOGGER_H
#define ALPHAZERO_LOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/async.h>
#include <memory>
#include <string>
#include "core/export_macros.h"

namespace alphazero {
namespace utils {

/**
 * Production-grade logging using spdlog
 * 
 * Features:
 * - Thread-safe async logging
 * - Multiple sinks (console + file)
 * - Configurable log levels
 * - Rotating log files
 * - Structured logging with custom formatters
 * - Performance-optimized for MCTS operations
 */
class ALPHAZERO_API Logger {
public:
    // Logger initialization
    static void init(const std::string& log_dir = "logs",
                    spdlog::level::level_enum console_level = spdlog::level::info,
                    spdlog::level::level_enum file_level = spdlog::level::debug,
                    size_t max_file_size = 1048576 * 50, // 50MB
                    size_t max_files = 10,
                    bool async_logging = true);
    
    // Get logger instances
    static std::shared_ptr<spdlog::logger> get_mcts_logger();
    static std::shared_ptr<spdlog::logger> get_nn_logger();
    static std::shared_ptr<spdlog::logger> get_game_logger();
    static std::shared_ptr<spdlog::logger> get_system_logger();
    
    // Convenience functions
    static void shutdown();
    static void flush_all();
    static void set_level(spdlog::level::level_enum level);
    
private:
    static bool initialized_;
    static std::shared_ptr<spdlog::logger> mcts_logger_;
    static std::shared_ptr<spdlog::logger> nn_logger_;
    static std::shared_ptr<spdlog::logger> game_logger_;
    static std::shared_ptr<spdlog::logger> system_logger_;
    
    // Create logger with specific sinks
    static std::shared_ptr<spdlog::logger> create_logger(
        const std::string& name,
        const std::string& log_dir,
        spdlog::level::level_enum console_level,
        spdlog::level::level_enum file_level,
        size_t max_file_size,
        size_t max_files,
        bool async);
};

// Convenience macros for logging
#define LOG_MCTS_TRACE(...) SPDLOG_LOGGER_TRACE(alphazero::utils::Logger::get_mcts_logger(), __VA_ARGS__)
#define LOG_MCTS_DEBUG(...) SPDLOG_LOGGER_DEBUG(alphazero::utils::Logger::get_mcts_logger(), __VA_ARGS__)
#define LOG_MCTS_INFO(...)  SPDLOG_LOGGER_INFO(alphazero::utils::Logger::get_mcts_logger(), __VA_ARGS__)
#define LOG_MCTS_WARN(...)  SPDLOG_LOGGER_WARN(alphazero::utils::Logger::get_mcts_logger(), __VA_ARGS__)
#define LOG_MCTS_ERROR(...) SPDLOG_LOGGER_ERROR(alphazero::utils::Logger::get_mcts_logger(), __VA_ARGS__)

#define LOG_NN_TRACE(...) SPDLOG_LOGGER_TRACE(alphazero::utils::Logger::get_nn_logger(), __VA_ARGS__)
#define LOG_NN_DEBUG(...) SPDLOG_LOGGER_DEBUG(alphazero::utils::Logger::get_nn_logger(), __VA_ARGS__)
#define LOG_NN_INFO(...)  SPDLOG_LOGGER_INFO(alphazero::utils::Logger::get_nn_logger(), __VA_ARGS__)
#define LOG_NN_WARN(...)  SPDLOG_LOGGER_WARN(alphazero::utils::Logger::get_nn_logger(), __VA_ARGS__)
#define LOG_NN_ERROR(...) SPDLOG_LOGGER_ERROR(alphazero::utils::Logger::get_nn_logger(), __VA_ARGS__)

#define LOG_GAME_TRACE(...) SPDLOG_LOGGER_TRACE(alphazero::utils::Logger::get_game_logger(), __VA_ARGS__)
#define LOG_GAME_DEBUG(...) SPDLOG_LOGGER_DEBUG(alphazero::utils::Logger::get_game_logger(), __VA_ARGS__)
#define LOG_GAME_INFO(...)  SPDLOG_LOGGER_INFO(alphazero::utils::Logger::get_game_logger(), __VA_ARGS__)
#define LOG_GAME_WARN(...)  SPDLOG_LOGGER_WARN(alphazero::utils::Logger::get_game_logger(), __VA_ARGS__)
#define LOG_GAME_ERROR(...) SPDLOG_LOGGER_ERROR(alphazero::utils::Logger::get_game_logger(), __VA_ARGS__)

#define LOG_SYSTEM_TRACE(...) SPDLOG_LOGGER_TRACE(alphazero::utils::Logger::get_system_logger(), __VA_ARGS__)
#define LOG_SYSTEM_DEBUG(...) SPDLOG_LOGGER_DEBUG(alphazero::utils::Logger::get_system_logger(), __VA_ARGS__)
#define LOG_SYSTEM_INFO(...)  SPDLOG_LOGGER_INFO(alphazero::utils::Logger::get_system_logger(), __VA_ARGS__)
#define LOG_SYSTEM_WARN(...)  SPDLOG_LOGGER_WARN(alphazero::utils::Logger::get_system_logger(), __VA_ARGS__)
#define LOG_SYSTEM_ERROR(...) SPDLOG_LOGGER_ERROR(alphazero::utils::Logger::get_system_logger(), __VA_ARGS__)

// Structured logging helpers
struct MCTSLogData {
    int simulations;
    int batch_size;
    float nps; // nodes per second
    int depth;
    float value;
    
    friend std::ostream& operator<<(std::ostream& os, const MCTSLogData& data) {
        os << fmt::format("sims={} batch={} nps={:.1f} depth={} value={:.3f}",
                         data.simulations, data.batch_size, data.nps, 
                         data.depth, data.value);
        return os;
    }
};

struct NNLogData {
    int batch_size;
    float inference_time_ms;
    float gpu_utilization;
    size_t memory_used;
    
    friend std::ostream& operator<<(std::ostream& os, const NNLogData& data) {
        os << fmt::format("batch={} time={:.2f}ms gpu={:.1f}% mem={}MB",
                         data.batch_size, data.inference_time_ms,
                         data.gpu_utilization, data.memory_used / (1024*1024));
        return os;
    }
};

} // namespace utils
} // namespace alphazero

#endif // ALPHAZERO_LOGGER_H