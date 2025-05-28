#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <memory>
#include <cstdio>
#include <unistd.h>

namespace alphazero {
namespace utils {

// Global mutex for progress bar display
static std::mutex g_progress_display_mutex;
static std::atomic<bool> g_progress_bar_active{false};

class ProgressBar {
public:
    ProgressBar(int total, const std::string& prefix = "", int width = 50)
        : total_(total), current_(0), prefix_(prefix), width_(width), 
          start_time_(std::chrono::steady_clock::now()), is_active_(true), is_valid_(false),
          is_tty_(isatty(fileno(stderr))) {
        
        // Check if another progress bar is already active
        bool expected = false;
        if (!g_progress_bar_active.compare_exchange_strong(expected, true)) {
            // Another progress bar is active, mark this as invalid
            is_valid_ = false;
            is_active_ = false;
            return;
        }
        
        is_valid_ = true;
        // Ensure we start on a clean line
        std::cerr << std::endl;
        // Start progress display thread
        display_thread_ = std::thread(&ProgressBar::displayLoop, this);
    }
    
    ~ProgressBar() {
        if (!is_valid_) {
            return;
        }
        
        is_active_ = false;
        if (display_thread_.joinable()) {
            display_thread_.join();
        }
        // Print final state with carriage return to ensure clean line
        std::cerr << "\r" << std::string(120, ' ') << "\r";
        display(true);
        std::cerr << std::endl;
        
        // Mark progress bar as inactive
        g_progress_bar_active = false;
    }
    
    void update(int increment = 1) {
        if (!is_valid_) return;
        current_ += increment;
        last_update_time_ = std::chrono::steady_clock::now();
    }
    
    void setPostfix(const std::string& postfix) {
        if (!is_valid_) return;
        std::lock_guard<std::mutex> lock(postfix_mutex_);
        postfix_ = postfix;
    }
    
    void complete() {
        if (!is_valid_) return;
        current_ = total_;
    }
    
    int getCurrent() const {
        return current_.load();
    }
    
    bool isValid() const {
        return is_valid_;
    }

private:
    void displayLoop() {
        int last_displayed = -1;
        auto last_display_time = std::chrono::steady_clock::now();
        
        while (is_active_) {
            if (is_valid_) {
                int current_val = current_.load();
                auto now = std::chrono::steady_clock::now();
                auto time_since_display = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_display_time).count();
                
                // Update every 1% progress or every 30 seconds
                int progress_percent = (current_val * 100) / total_;
                int last_progress_percent = (last_displayed * 100) / total_;
                
                bool significant_progress = progress_percent > last_progress_percent;
                bool timeout = time_since_display >= 30;
                
                if ((significant_progress || timeout) && current_val != last_displayed) {
                    display(false);
                    last_displayed = current_val;
                    last_display_time = now;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    void display(bool final = false) {
        if (!is_valid_) return;
        
        std::lock_guard<std::mutex> lock(g_progress_display_mutex);
        
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
        
        // No need for extra spaces with ANSI clearing
        
        // Output with proper line handling
        if (!final || is_active_) {
            // When piped through tee, just print simple progress updates
            std::cerr << "Progress: " << current << "/" << total_ 
                     << " (" << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%)";
            if (elapsed > 0 && current > 0) {
                float rate = static_cast<float>(current) / elapsed;
                std::cerr << " - " << std::fixed << std::setprecision(2) << rate << " games/s";
            }
            {
                std::lock_guard<std::mutex> lock(postfix_mutex_);
                if (!postfix_.empty()) {
                    std::cerr << " - " << postfix_;
                }
            }
            std::cerr << std::endl;
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
    bool is_valid_ = false;
    bool is_tty_ = false;
};

// Global progress manager for self-play
class __attribute__((visibility("default"))) SelfPlayProgressManager {
public:
    static SelfPlayProgressManager& getInstance() {
        static SelfPlayProgressManager instance;
        return instance;
    }
    
    // Clear any existing progress bar (for clean shutdown)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (progress_bar_) {
            progress_bar_.reset();
        }
        games_completed_ = 0;
        total_games_ = 0;
    }
    
    void startGames(int total_games) {
        // Do nothing - we don't create progress bars from workers
        return;
    }
    
    void startGlobalProgress(int total_games) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Starting global progress tracking
        
        // Clean up any existing progress bar
        if (progress_bar_) {
            progress_bar_.reset();
            // Wait to ensure cleanup
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        // Create the single global progress bar
        progress_bar_ = std::make_unique<ProgressBar>(total_games, "Self-play");
        games_completed_ = 0;
        total_games_ = total_games;
    }
    
    void completeGame(const std::string& game_id, int moves, float outcome) {
        std::lock_guard<std::mutex> lock(mutex_);
        games_completed_++;
        
        // Store pointer locally to ensure it's not changed while we use it
        auto* bar_ptr = progress_bar_.get();
        
        if (bar_ptr && bar_ptr->isValid()) {
            bar_ptr->update();
            
            // Update postfix with latest game info
            std::stringstream ss;
            ss << "Last: " << game_id << " (" << moves << " moves, outcome: " 
               << std::fixed << std::setprecision(1) << outcome << ")";
            bar_ptr->setPostfix(ss.str());
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

    bool isGlobalProgressActive() const {
        // Always return true to prevent worker progress bars
        return true;
    }

private:
    SelfPlayProgressManager() : games_completed_(0), total_games_(0), verbose_logging_(false) {
    }
    
    ~SelfPlayProgressManager() {
    }
    
    std::unique_ptr<ProgressBar> progress_bar_;
    mutable std::mutex mutex_;
    int games_completed_;
    int total_games_;
    std::atomic<bool> verbose_logging_;
};

} // namespace utils
} // namespace alphazero