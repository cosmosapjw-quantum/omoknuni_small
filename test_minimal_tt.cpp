#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Test 1: Basic output works" << std::endl;
    
    // Test if chrono is causing issues
    std::cout << "Test 2: Creating chrono duration" << std::endl;
    auto timeout = std::chrono::milliseconds(1);
    std::cout << "Test 2 passed: " << timeout.count() << "ms" << std::endl;
    
    // Test if thread creation works
    std::cout << "Test 3: Thread test" << std::endl;
    std::thread t([]() {
        std::cout << "Thread executed" << std::endl;
    });
    t.join();
    std::cout << "Test 3 passed" << std::endl;
    
    return 0;
}