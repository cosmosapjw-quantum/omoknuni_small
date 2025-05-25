#include "utils/logger.h"
#include <filesystem>
#include <iostream>
#include <thread>
#include <chrono>

namespace alphazero {
namespace utils {

// Static member definitions
bool Logger::initialized_ = false;
std::shared_ptr<spdlog::logger> Logger::mcts_logger_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::nn_logger_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::game_logger_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::system_logger_ = nullptr;

void Logger::init(const std::string& log_dir,
                  spdlog::level::level_enum console_level,
                  spdlog::level::level_enum file_level,
                  size_t max_file_size,
                  size_t max_files,
                  bool async_logging) {
    if (initialized_) {
        return;
    }
    
    try {
        // Create log directory if it doesn't exist
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }
        
        // CRITICAL FIX: Initialize thread pool BEFORE creating async loggers
        if (async_logging) {
            spdlog::init_thread_pool(32768, 4); // Larger queue and more threads
            spdlog::flush_every(std::chrono::seconds(3));
        }
        
        // Create loggers AFTER thread pool initialization
        mcts_logger_ = create_logger("mcts", log_dir, console_level, file_level, 
                                    max_file_size, max_files, async_logging);
        nn_logger_ = create_logger("neural_net", log_dir, console_level, file_level, 
                                  max_file_size, max_files, async_logging);
        game_logger_ = create_logger("game", log_dir, console_level, file_level, 
                                    max_file_size, max_files, async_logging);
        system_logger_ = create_logger("system", log_dir, console_level, file_level, 
                                      max_file_size, max_files, async_logging);
        
        // Set global log level
        spdlog::set_level(console_level);
        
        initialized_ = true;
        system_logger_->info("Logging system initialized. Log directory: {}", log_dir);
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

std::shared_ptr<spdlog::logger> Logger::create_logger(
    const std::string& name,
    const std::string& log_dir,
    spdlog::level::level_enum console_level,
    spdlog::level::level_enum file_level,
    size_t max_file_size,
    size_t max_files,
    bool async) {
    
    std::vector<spdlog::sink_ptr> sinks;
    
    // Console sink with color
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(console_level);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%t] %v");
    sinks.push_back(console_sink);
    
    // Rotating file sink
    std::string log_path = log_dir + "/" + name + ".log";
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_path, max_file_size, max_files);
    file_sink->set_level(file_level);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [%t] [%s:%#] %v");
    sinks.push_back(file_sink);
    
    // Create logger
    std::shared_ptr<spdlog::logger> logger;
    if (async) {
        logger = std::make_shared<spdlog::async_logger>(
            name, sinks.begin(), sinks.end(), 
            spdlog::thread_pool(),
            spdlog::async_overflow_policy::overrun_oldest); // Don't block, drop old messages
    } else {
        logger = std::make_shared<spdlog::logger>(
            name, sinks.begin(), sinks.end());
    }
    
    logger->set_level(spdlog::level::trace); // Allow all levels, sinks filter
    logger->flush_on(spdlog::level::warn);   // Auto flush on warnings and errors
    
    // Register logger
    spdlog::register_logger(logger);
    
    return logger;
}

std::shared_ptr<spdlog::logger> Logger::get_mcts_logger() {
    if (!initialized_) {
        init();
    }
    return mcts_logger_;
}

std::shared_ptr<spdlog::logger> Logger::get_nn_logger() {
    if (!initialized_) {
        init();
    }
    return nn_logger_;
}

std::shared_ptr<spdlog::logger> Logger::get_game_logger() {
    if (!initialized_) {
        init();
    }
    return game_logger_;
}

std::shared_ptr<spdlog::logger> Logger::get_system_logger() {
    if (!initialized_) {
        init();
    }
    return system_logger_;
}

void Logger::shutdown() {
    if (initialized_) {
        // Flush all pending messages
        flush_all();
        
        // Give time for async loggers to process remaining messages
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Drop logger references to prevent use after free
        mcts_logger_.reset();
        nn_logger_.reset();
        game_logger_.reset();
        system_logger_.reset();
        
        // Shutdown spdlog (this destroys the thread pool)
        spdlog::shutdown();
        initialized_ = false;
    }
}

void Logger::flush_all() {
    if (initialized_) {
        mcts_logger_->flush();
        nn_logger_->flush();
        game_logger_->flush();
        system_logger_->flush();
    }
}

void Logger::set_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
    if (initialized_) {
        mcts_logger_->set_level(level);
        nn_logger_->set_level(level);
        game_logger_->set_level(level);
        system_logger_->set_level(level);
    }
}

} // namespace utils
} // namespace alphazero