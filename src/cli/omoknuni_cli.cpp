// src/cli/omoknuni_cli.cpp
#include "cli/cli_manager.h"
#include "core/game_export.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <thread>
#include <dlfcn.h>
#include <cstdarg>  // For va_start, va_end
#include <signal.h>  // For signal handling
#include <unistd.h>  // For alarm()

// Make debug monitor available to all functions
#include "utils/debug_monitor.h"

// Configure PyTorch CUDA initialization safely
// These defines must come before any torch headers are included
#define PYTORCH_NO_CUDA_INIT_OVERRIDE 0 // Allow proper CUDA initialization
#define USE_TORCH 1  // Enable torch functionality
#define C10_CUDA_DRIVER_INIT 1 // Enable driver initialization

// Watchdog timer variables
volatile sig_atomic_t g_watchdog_timer_expired = 0;
constexpr int WATCHDOG_TIMEOUT_SECONDS = 30; // 30 seconds timeout

// Signal handler for watchdog timer
void watchdog_handler(int sig) {
    g_watchdog_timer_expired = 1;
    std::cerr << "\n[WATCHDOG] Timer expired - program may be stuck\n" << std::endl;
}

// Setup watchdog timer
void setup_watchdog() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = watchdog_handler;
    sigaction(SIGALRM, &sa, NULL);

    // Set alarm for WATCHDOG_TIMEOUT_SECONDS
    alarm(WATCHDOG_TIMEOUT_SECONDS);
}

// Reset watchdog timer
void reset_watchdog() {
    alarm(WATCHDOG_TIMEOUT_SECONDS);
}

// Cancel watchdog timer
void cancel_watchdog() {
    alarm(0);
}

// Debug logging level
enum DebugLevel {
    NONE = 0,
    BASIC = 1,
    DETAILED = 2,
    VERBOSE = 3
};

// Global debug level (default: BASIC)
DebugLevel g_debug_level = BASIC;

// Output debug message based on current debug level
void debug_print(DebugLevel level, const char* format, ...) {
    if (level <= g_debug_level) {
        va_list args;
        va_start(args, format);

        // Print to stderr for immediate visibility
        fprintf(stderr, "[DEBUG L%d] ", level);
        vfprintf(stderr, format, args);
        fprintf(stderr, "\n");
        fflush(stderr);

        // Also log to debug file
        FILE* debug_file = fopen("/tmp/cli_debug.log", "a");
        if (debug_file) {
            // Add timestamp
            auto now = std::chrono::system_clock::now();
            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
            auto epoch = now_ms.time_since_epoch();
            auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);

            fprintf(debug_file, "[%ld][DEBUG L%d] ", value.count(), level);
            vfprintf(debug_file, format, args);
            fprintf(debug_file, "\n");
            fclose(debug_file);
        }

        va_end(args);
    }
}

// Check if a specific library is available
bool check_library(const char* libname) {
    debug_print(DETAILED, "Checking for library: %s", libname);

    void* handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        debug_print(DETAILED, "Could not load library %s: %s", libname, dlerror());
        return false;
    }

    debug_print(DETAILED, "Successfully loaded library: %s", libname);
    dlclose(handle);
    return true;
}

// Simple diagnostic version with enhanced debugging
int main(int argc, char** argv) {
    // Immediate basic console output for debugging with timestamp
    fprintf(stderr, "[DEBUG-INIT] omoknuni_cli starting...\n");
    fflush(stderr);
    // Setup watchdog timer to prevent infinite hanging
    setup_watchdog();
    fprintf(stderr, "[DEBUG-INIT] Watchdog timer set up\n");
    fflush(stderr);

    // Set debug level from environment variable if present
    const char* debug_env = getenv("OMOKNUNI_DEBUG");
    if (debug_env) {
        g_debug_level = static_cast<DebugLevel>(atoi(debug_env));
    }

    // Check if watchdog timer already expired (possible stuck process)
    if (g_watchdog_timer_expired) {
        std::cerr << "[WATCHDOG] Program appears stuck before initialization completed" << std::endl;
        return 1;
    }

    // Initialize debug log file (clear it)
    FILE* debug_file = fopen("/tmp/cli_debug.log", "w");
    if (debug_file) {
        fprintf(debug_file, "CLI debug log initialized\n");
        fclose(debug_file);
    }

    // Force output immediately in case of issues
    debug_print(BASIC, "omoknuni_cli starting...");

    // Configure CUDA properly instead of disabling it
    debug_print(BASIC, "Setting up environment variables for CUDA");
    setenv("OMP_NUM_THREADS", "1", 1);      // Limit OpenMP threads
    setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32", 1); // Limit memory allocation

    // Check if CUDA is available
    debug_print(BASIC, "Checking for CUDA devices...");

    // Explicitly enable CUDA with proper settings
    debug_print(BASIC, "Enabling CUDA support with optimal settings");

    // Explicitly enable CUDA
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);  // Use first device
    setenv("USE_CUDA", "1", 1);              // Enable CUDA in PyTorch

    // Disable any CUDA blocking variables that might be set
    unsetenv("PYTORCH_NO_CUDA");             // Make sure PyTorch uses CUDA

    // Set some performance optimizations for CUDA
    setenv("CUDA_LAUNCH_BLOCKING", "0", 1);    // Asynchronous kernel launches
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8", 1); // Increased stream capacity

    // Log that we're using CUDA
    fprintf(stderr, "CUDA has been explicitly enabled for the application\n");
    fflush(stderr);

    // Log command line arguments
    debug_print(BASIC, "CLI started with %d arguments", argc);
    fprintf(stderr, "CLI started with %d arguments\n", argc);
    for (int i = 0; i < argc; i++) {
        debug_print(DETAILED, "  arg[%d] = %s", i, argv[i]);
        fprintf(stderr, "  arg[%d] = %s\n", i, argv[i]);
    }

    // Reset watchdog timer after arg processing
    reset_watchdog();

    // Print environment information
    debug_print(DETAILED, "Current directory: %s",
            std::filesystem::current_path().string().c_str());

    const char* ld_path = getenv("LD_LIBRARY_PATH");
    debug_print(DETAILED, "LD_LIBRARY_PATH: %s", ld_path ? ld_path : "not set");

    // Check for critical libraries
    debug_print(BASIC, "Checking for alphazero library only");

    // First check alphazero library - only check this one
    // Look in both the current directory and the build/lib/Debug directory
    const char* search_paths[] = {
        "libalphazero.so",                      // Current directory
        "./build/lib/Debug/libalphazero.so",    // Debug build
        "./build/lib/Release/libalphazero.so",  // Release build
        "../lib/Debug/libalphazero.so",         // Relative from bin/Debug
        "../lib/Release/libalphazero.so"        // Relative from bin/Release
    };
    
    bool has_alphazero = false;
    for (const char* path : search_paths) {
        debug_print(DETAILED, "Trying to load library from: %s", path);
        fprintf(stderr, "Trying to load library from: %s\n", path);
        has_alphazero = check_library(path);
        if (has_alphazero) {
            debug_print(BASIC, "Successfully loaded library from: %s", path);
            fprintf(stderr, "Successfully loaded library from: %s\n", path);
            break;
        }
    }

    debug_print(BASIC, "Library status summary:");
    debug_print(BASIC, "  alphazero: %s", has_alphazero ? "YES" : "NO");

    // Show help with flushing to ensure output is visible
    printf("Omoknuni CLI - AlphaZero Implementation (CPU-only mode)\n");
    printf("Usage: omoknuni_cli <command> [options]\n");
    printf("Available commands: self-play, train, eval, play\n");
    printf("For more information, run 'omoknuni_cli <command> --help'\n");
    fflush(stdout); // Ensure output is flushed

    // Create CLI manager
    try {
        // Reset watchdog before creating CLI manager
        reset_watchdog();

        debug_print(BASIC, "Registering game implementations");
        fprintf(stderr, "[DEBUG-INIT] About to register games\n");
        fflush(stderr);

        // Register basic game implementations in core
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::GOMOKU,
            []() {
                fprintf(stderr, "[DEBUG-INIT] Creating GOMOKU game instance\n");
                fflush(stderr);
                try {
                    return std::make_unique<alphazero::games::gomoku::GomokuState>();
                } catch (const std::exception& e) {
                    fprintf(stderr, "[DEBUG-ERROR] Exception creating GOMOKU game: %s\n", e.what());
                    fflush(stderr);
                    throw;
                } catch (...) {
                    fprintf(stderr, "[DEBUG-ERROR] Unknown exception creating GOMOKU game\n");
                    fflush(stderr);
                    throw;
                }
            }
        );
        
        fprintf(stderr, "[DEBUG-INIT] GOMOKU game registered\n");
        fflush(stderr);
        
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::CHESS,
            []() {
                fprintf(stderr, "[DEBUG-INIT] Creating CHESS game instance\n");
                fflush(stderr);
                try {
                    return std::make_unique<alphazero::games::chess::ChessState>();
                } catch (const std::exception& e) {
                    fprintf(stderr, "[DEBUG-ERROR] Exception creating CHESS game: %s\n", e.what());
                    fflush(stderr);
                    throw;
                } catch (...) {
                    fprintf(stderr, "[DEBUG-ERROR] Unknown exception creating CHESS game\n");
                    fflush(stderr);
                    throw;
                }
            }
        );
        
        fprintf(stderr, "[DEBUG-INIT] CHESS game registered\n");
        fflush(stderr);
        
        alphazero::core::GameRegistry::instance().registerGame(
            alphazero::core::GameType::GO,
            []() {
                fprintf(stderr, "[DEBUG-INIT] Creating GO game instance\n");
                fflush(stderr);
                try {
                    return std::make_unique<alphazero::games::go::GoState>();
                } catch (const std::exception& e) {
                    fprintf(stderr, "[DEBUG-ERROR] Exception creating GO game: %s\n", e.what());
                    fflush(stderr);
                    throw;
                } catch (...) {
                    fprintf(stderr, "[DEBUG-ERROR] Unknown exception creating GO game\n");
                    fflush(stderr);
                    throw;
                }
            }
        );
        
        fprintf(stderr, "[DEBUG-INIT] GO game registered\n");
        fflush(stderr);

        debug_print(BASIC, "Creating CLI manager");
        fprintf(stderr, "[DEBUG-INIT] About to create CLI manager\n");
        fflush(stderr);

        // Create CLI manager
        alphazero::cli::CLIManager cli;

        fprintf(stderr, "[DEBUG-INIT] CLI manager created\n");
        fflush(stderr);
        
        debug_print(BASIC, "Adding command handlers");
        fprintf(stderr, "[DEBUG-INIT] About to add command handlers\n");
        fflush(stderr);

        // Add dummy commands that just print help
        cli.addCommand("self-play", "Generate self-play games for training",
            [](const std::vector<std::string>& args) {
                debug_print(BASIC, "Self-play command handler called with %zu arguments", args.size());
                printf("Self-play command (CPU-only diagnostic version)\n");
                return 0;
            }
        );

        cli.addCommand("train", "Train neural network from self-play data",
            [](const std::vector<std::string>& args) {
                debug_print(BASIC, "Train command handler called with %zu arguments", args.size());
                printf("Train command (CPU-only diagnostic version)\n");
                return 0;
            }
        );

        cli.addCommand("eval", "Evaluate model strength",
            [](const std::vector<std::string>& args) {
                debug_print(BASIC, "Eval command handler called with %zu arguments", args.size());
                printf("Eval command (CPU-only diagnostic version)\n");
                return 0;
            }
        );

        cli.addCommand("play", "Play against AI",
            [](const std::vector<std::string>& args) {
                debug_print(BASIC, "Play command handler called with %zu arguments", args.size());
                printf("Play command (CPU-only diagnostic version)\n");
                return 0;
            }
        );

        // Run CLI with arguments
        debug_print(BASIC, "Running CLI manager with %d arguments", argc);
        printf("Running CLI with %d arguments\n", argc);
        fflush(stdout); // Force immediate output
        
        fprintf(stderr, "[DEBUG-INIT] All command handlers added\n");
        fflush(stderr);

        // Reset watchdog before running CLI
        reset_watchdog();

        debug_print(BASIC, "Calling cli.run() - if the program hangs after this message, the issue is in CLIManager::run()");
        fprintf(stderr, "[DEBUG-INIT] About to call cli.run()\n");
        fflush(stderr);

        // Run with watchdog checking
        int result = -1;
        if (g_watchdog_timer_expired) {
            fprintf(stderr, "Watchdog expired before CLI execution - emergency exit\n");
            return 1;
        }

        fprintf(stderr, "[DEBUG-INIT] Calling CLIManager::run with argc=%d\n", argc);
        for (int i = 0; i < argc; i++) {
            fprintf(stderr, "[DEBUG-INIT]   argv[%d] = %s\n", i, argv[i]);
        }
        fflush(stderr);
        
        result = cli.run(argc, argv);
        
        fprintf(stderr, "[DEBUG-INIT] cli.run() returned %d\n", result);
        fflush(stderr);

        // Cancel watchdog timer since we've completed successfully
        cancel_watchdog();
        debug_print(BASIC, "CLI manager returned with code %d", result);

        return result;
    }
    catch (const std::exception& e) {
        debug_print(BASIC, "Exception caught: %s", e.what());
        printf("ERROR: %s\n", e.what());
        return 1;
    }
    catch (...) {
        debug_print(BASIC, "Unknown exception caught");
        printf("ERROR: Unknown exception occurred\n");
        return 1;
    }
}