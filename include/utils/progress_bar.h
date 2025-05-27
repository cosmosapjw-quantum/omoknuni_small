#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace alphazero {
namespace utils {

class ProgressBar {
public:
    ProgressBar(int total, const std::string& prefix = "", int width = 50)
        : total_(total), current_(0), prefix_(prefix), width_(width), 
          start_time_(std::chrono::steady_clock::now()), is_active_(true) {
        // Start progress display thread
        display_thread_ = std::thread(&ProgressBar::displayLoop, this);
    }
    
    ~ProgressBar() {
        is_active_ = false;
        if (display_thread_.joinable()) {
            display_thread_.join();
        }
        // Print final state
        display(true);
        std::cout << std::endl;
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
        int current = current_.load();
        if (current > total_) current = total_;
        
        float progress = static_cast<float>(current) / total_;
        int filled = static_cast<int>(progress * width_);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        std::stringstream ss;
        ss << "\r" << prefix_ << " [";
        
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
};

// Global progress manager for self-play
class SelfPlayProgressManager {
public:
    static SelfPlayProgressManager& getInstance() {
        static SelfPlayProgressManager instance;
        return instance;
    }
    
    void startGames(int total_games) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (progress_bar_) {
            progress_bar_.reset();
        }
        progress_bar_ = std::make_unique<ProgressBar>(total_games, "Self-play");
        games_completed_ = 0;
        total_games_ = total_games;
    }
    
    void completeGame(const std::string& game_id, int moves, float outcome) {
        std::lock_guard<std::mutex> lock(mutex_);
        games_completed_++;
        
        if (progress_bar_) {
            progress_bar_->update();
            
            // Update postfix with latest game info
            std::stringstream ss;
            ss << "Last: " << game_id << " (" << moves << " moves, outcome: " 
               << std::fixed << std::setprecision(1) << outcome << ")";
            progress_bar_->setPostfix(ss.str());
        }
    }
    
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (progress_bar_) {
            progress_bar_->complete();
            progress_bar_.reset();
        }
    }
    
    bool isVerboseLoggingEnabled() const {
        return verbose_logging_;
    }
    
    void setVerboseLogging(bool enabled) {
        verbose_logging_ = enabled;
    }

private:
    SelfPlayProgressManager() : games_completed_(0), total_games_(0), verbose_logging_(false) {}
    
    std::unique_ptr<ProgressBar> progress_bar_;
    std::mutex mutex_;
    int games_completed_;
    int total_games_;
    std::atomic<bool> verbose_logging_;
};

} // namespace utils
} // namespace alphazero