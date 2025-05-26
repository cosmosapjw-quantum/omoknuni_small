#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <vector>

std::atomic<bool> g_shutdown(false);
std::atomic<int> g_signal_count(0);

void signalHandler(int signal) {
    if (signal == SIGINT) {
        int count = ++g_signal_count;
        if (count == 1) {
            std::cout << "\n*** First SIGINT received - graceful shutdown ***" << std::endl;
            g_shutdown = true;
        } else {
            std::cout << "\n*** Second SIGINT received - force exit ***" << std::endl;
            std::_Exit(0);
        }
    }
}

void worker(int id) {
    std::cout << "Worker " << id << " started" << std::endl;
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Worker " << id << " shutting down" << std::endl;
}

int main() {
    std::signal(SIGINT, signalHandler);
    
    std::cout << "Starting test program with 3 workers..." << std::endl;
    std::cout << "Press Ctrl+C to test signal handling" << std::endl;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for threads
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All workers stopped - exiting cleanly" << std::endl;
    return 0;
}