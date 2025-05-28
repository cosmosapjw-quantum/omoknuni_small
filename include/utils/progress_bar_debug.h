#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <fstream>

namespace alphazero {
namespace utils {

// Global mutex for console output to prevent interference
static std::mutex g_console_mutex;
static std::atomic<int> g_progress_bar_counter{0};
static std::ofstream g_debug_log;
static std::once_flag g_debug_log_init;

class ProgressBarDebug {
public:
    ProgressBarDebug(int total, const std::string& prefix = "", int width = 50)
        : total_(total), current_(0), prefix_(prefix), width_(width), 
          start_time_(std::chrono::steady_clock::now()), is_active_(true),
          instance_id_(g_progress_bar_counter.fetch_add(1)) {
        
        // Initialize debug log once
        std::call_once(g_debug_log_init, []() {
            g_debug_log.open("progress_debug.log", std::ios::out | std::ios::trunc);
            g_debug_log << "Progress Bar Debug Log\n";
            g_debug_log << "=====================\n";
        });
        
        // Log creation
        {
            std::lock_guard<std::mutex> lock(g_console_mutex);
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            g_debug_log << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                       << "] ProgressBar #" << instance_id_ 
                       << " CREATED (total=" << total << ", prefix='" << prefix << "')\n";
            g_debug_log.flush();
        }
        
        // Start progress display thread
        display_thread_ = std::thread(&ProgressBarDebug::displayLoop, this);
    }
    
    ~ProgressBarDebug() {
        // Log destruction
        {
            std::lock_guard<std::mutex> lock(g_console_mutex);
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            g_debug_log << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                       << "] ProgressBar #" << instance_id_ 
                       << " DESTROYING (current=" << current_.load() 
                       << "/" << total_ << ")\n";
            g_debug_log.flush();
        }
        
        is_active_ = false;
        if (display_thread_.joinable()) {
            display_thread_.join();
        }
        
        // Clear line and print final state
        {
            std::lock_guard<std::mutex> lock(g_console_mutex);
            std::cout << "\r" << std::string(120, ' ') << "\r";
            display(true);
            std::cout << std::endl;
        }
    }
    
    void update(int increment = 1) {
        current_ += increment;
        last_update_time_ = std::chrono::steady_clock::now();
    }
    
    void setPostfix(const std::string& postfix) {
        std::lock_guard<std::mutex> lock(postfix_mutex_);
        postfix_ = postfix;
    }
    
    void complete() {
        current_ = total_;
    }

private:
    void displayLoop() {
        while (is_active_) {
            display(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void display(bool final = false) {
        std::lock_guard<std::mutex> lock(g_console_mutex);
        
        int current = current_.load();
        if (current > total_) current = total_;
        
        float progress = static_cast<float>(current) / total_;
        int filled = static_cast<int>(progress * width_);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        std::stringstream ss;
        ss << "\r" << prefix_ << " [#" << instance_id_ << "] [";
        
        // Progress bar
        for (int i = 0; i < width_; ++i) {
            if (i < filled) {
                ss << "=";
            } else if (i == filled) {
                ss << ">";
            } else {
                ss << " ";
            }
        }
        
        ss << "] " << current << "/" << total_ 
           << " (" << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%)";
        
        // Time info
        if (elapsed > 0 && current > 0) {
            float rate = static_cast<float>(current) / elapsed;
            int eta = (total_ - current) / rate;
            ss << " [" << formatTime(elapsed) << "<" << formatTime(eta) 
               << ", " << std::fixed << std::setprecision(2) << rate << " games/s]";
        }
        
        // Postfix
        {
            std::lock_guard<std::mutex> lock(postfix_mutex_);
            if (!postfix_.empty()) {
                ss << " " << postfix_;
            }
        }
        
        // Clear to end of line
        ss << "          ";
        
        std::cout << ss.str() << std::flush;
        
        // Also log to file periodically
        static int log_counter = 0;
        if (++log_counter % 10 == 0 || final) {
            auto time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            g_debug_log << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                       << "] ProgressBar #" << instance_id_ 
                       << " displaying: " << current << "/" << total_ << "\n";
            g_debug_log.flush();
        }
    }
    
    std::string formatTime(int seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;
        int secs = seconds % 60;
        
        std::stringstream ss;
        if (hours > 0) {
            ss << hours << ":" << std::setfill('0') << std::setw(2) << minutes 
               << ":" << std::setw(2) << secs;
        } else {
            ss << minutes << ":" << std::setfill('0') << std::setw(2) << secs;
        }
        return ss.str();
    }
    
    int total_;
    std::atomic<int> current_;
    std::string prefix_;
    int width_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    std::atomic<bool> is_active_;
    std::thread display_thread_;
    std::mutex postfix_mutex_;
    std::string postfix_;
    int instance_id_;
};

} // namespace utils
} // namespace alphazero