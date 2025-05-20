#ifndef ALPHAZERO_MCTS_ADAPTIVE_BACKOFF_H
#define ALPHAZERO_MCTS_ADAPTIVE_BACKOFF_H

#include <thread>
#include <chrono>
#include <functional>
#include <atomic>

namespace alphazero {
namespace mcts {

/**
 * @brief Utility class for efficient polling with adaptive backoff
 * 
 * This class provides methods for waiting with exponential backoff to reduce CPU usage
 * while still being responsive. It replaces condition variables with polling that adapts
 * based on system load and wait duration.
 */
class AdaptiveBackoff {
public:
    /**
     * @brief Constructor
     * 
     * @param max_yield_count Maximum number of yield attempts before sleeping
     * @param min_sleep_us Minimum sleep time in microseconds
     * @param max_sleep_us Maximum sleep time in microseconds
     */
    AdaptiveBackoff(int max_yield_count = 10, 
                  int min_sleep_us = 100, 
                  int max_sleep_us = 5000)
        : max_yield_count_(max_yield_count),
          min_sleep_us_(min_sleep_us),
          max_sleep_us_(max_sleep_us),
          spin_count_(0) {
    }

    /**
     * @brief Waits until predicate returns true or timeout expires
     * 
     * @param predicate Function returning true when waiting should stop
     * @param timeout Maximum time to wait
     * @return true if predicate returned true, false on timeout
     */
    bool wait_for(std::function<bool()> predicate, std::chrono::milliseconds timeout) {
        // Calculate deadline
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        // First try yielding for responsiveness
        for (int i = 0; i < max_yield_count_; ++i) {
            if (predicate()) return true;
            std::this_thread::yield();
        }
        
        // Use exponentially increasing sleep times
        int sleep_us = min_sleep_us_;
        
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) return true;
            
            // Sleep with current backoff
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            
            // Increase sleep time exponentially, capped at max_sleep_us_
            sleep_us = std::min(sleep_us * 2, max_sleep_us_);
            
            // Reset if we're sleeping too long
            if (sleep_us >= max_sleep_us_) {
                sleep_us = min_sleep_us_;
            }
        }
        
        // Final check
        return predicate();
    }

    /**
     * @brief Waits until predicate returns true or deadline is reached
     * 
     * @param predicate Function returning true when waiting should stop
     * @param deadline Time point to stop waiting
     * @return true if predicate returned true, false on timeout
     */
    bool wait_until(std::function<bool()> predicate, 
                   std::chrono::steady_clock::time_point deadline) {
        
        // First try yielding for responsiveness
        for (int i = 0; i < max_yield_count_; ++i) {
            if (predicate()) return true;
            std::this_thread::yield();
        }
        
        // Use exponentially increasing sleep times
        int sleep_us = min_sleep_us_;
        
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) return true;
            
            // Sleep with current backoff
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            
            // Increase sleep time exponentially, capped at max_sleep_us_
            sleep_us = std::min(sleep_us * 2, max_sleep_us_);
            
            // Reset if we're sleeping too long
            if (sleep_us >= max_sleep_us_) {
                sleep_us = min_sleep_us_;
            }
        }
        
        // Final check
        return predicate();
    }
    
    /**
     * @brief Waits with fixed sleep intervals
     * 
     * @param sleep_time Time to sleep on each iteration
     * @param iterations Number of iterations to wait
     * @param predicate Function returning true when waiting should stop
     * @return true if predicate returned true, false if all iterations elapsed
     */
    bool wait_fixed(std::chrono::microseconds sleep_time, int iterations,
                   std::function<bool()> predicate) {
        for (int i = 0; i < iterations; ++i) {
            if (predicate()) return true;
            std::this_thread::sleep_for(sleep_time);
        }
        return predicate();
    }
    
    /**
     * @brief Adaptive spin waiting based on system load
     * 
     * @param predicate Function returning true when waiting should stop
     * @param max_spins Maximum number of spin iterations before sleeping
     * @return true if predicate returned true, false otherwise
     */
    bool spin_wait(std::function<bool()> predicate, int max_spins = 1000) {
        // First try pure spinning
        for (int i = 0; i < max_spins; ++i) {
            if (predicate()) {
                // Reset spin count on success
                spin_count_ = 0;
                return true;
            }
            
            // Just burn CPU cycles for a few iterations
            for (volatile int j = 0; j < 10; ++j) {}
        }
        
        // If spinning failed, increment spin count
        spin_count_++;
        
        // If we've failed too many times, sleep to reduce CPU usage
        if (spin_count_ > 10) {
            std::this_thread::sleep_for(std::chrono::microseconds(min_sleep_us_));
            
            // Gradually increase sleep time
            min_sleep_us_ = std::min(min_sleep_us_ * 2, max_sleep_us_);
            
            // Reset sleep time if it gets too large
            if (min_sleep_us_ >= max_sleep_us_ / 2) {
                min_sleep_us_ = 100;
            }
        }
        
        return predicate();
    }

private:
    int max_yield_count_;
    int min_sleep_us_;
    int max_sleep_us_;
    std::atomic<int> spin_count_;
};

/**
 * @brief Global function for waiting with backoff
 * 
 * @param predicate Function returning true when waiting should stop
 * @param timeout Maximum time to wait
 * @return true if predicate returned true, false on timeout
 */
inline bool wait_with_backoff(std::function<bool()> predicate, 
                             std::chrono::milliseconds timeout) {
    static AdaptiveBackoff backoff;
    return backoff.wait_for(predicate, timeout);
}

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_ADAPTIVE_BACKOFF_H