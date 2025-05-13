// src/cli/cli_manager.cpp
#include "cli/cli_manager.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>

namespace alphazero {
namespace cli {

CLIManager::CLIManager() {
    // Default program name, will be overridden in run()
    program_name_ = "omoknuni-cli";
}

void CLIManager::addCommand(const std::string& command, 
                           const std::string& description, 
                           CommandHandler handler) {
    handlers_[command] = std::move(handler);
    descriptions_[command] = description;
}

int CLIManager::executeCommand(const std::string& command,
                              const std::vector<std::string>& args) {
    fprintf(stderr, "DEBUG: CLIManager::executeCommand() called with command='%s'\n", command.c_str());
    fflush(stderr);

    auto it = handlers_.find(command);
    if (it == handlers_.end()) {
        fprintf(stderr, "DEBUG: Unknown command: %s\n", command.c_str());
        fflush(stderr);
        std::cerr << "Unknown command: " << command << std::endl;
        printHelp();
        return 1;
    }

    fprintf(stderr, "DEBUG: Executing handler for command '%s'\n", command.c_str());
    fflush(stderr);

    try {
        // Execute the command handler
        int result = it->second(args);

        fprintf(stderr, "DEBUG: Handler for command '%s' returned %d\n", command.c_str(), result);
        fflush(stderr);

        return result;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "DEBUG: Exception in command handler '%s': %s\n", command.c_str(), e.what());
        fflush(stderr);
        std::cerr << "Error executing command: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        fprintf(stderr, "DEBUG: Unknown exception in command handler '%s'\n", command.c_str());
        fflush(stderr);
        std::cerr << "Unknown error executing command" << std::endl;
        return 1;
    }
}

int CLIManager::run(int argc, char** argv) {
    fprintf(stderr, "DEBUG: CLIManager::run() starting with argc=%d\n", argc);
    fflush(stderr);

    std::cout << "TRACE: CLIManager::run() starting with argc=" << argc << std::endl;
    std::cout.flush();

    // Store program name
    if (argc > 0) {
        std::filesystem::path path(argv[0]);
        program_name_ = path.filename().string();
        fprintf(stderr, "DEBUG: Program name: %s\n", program_name_.c_str());
        fflush(stderr);

        std::cout << "TRACE: Program name: " << program_name_ << std::endl;
        std::cout.flush();
    }

    // Check for help
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            if (i + 1 < argc) {
                // Command-specific help
                printHelp(argv[i + 1]);
            } else {
                // General help
                printHelp();
            }
            return 0;
        }
    }

    // Extract command and arguments
    if (argc < 2) {
        printHelp();
        return 0;
    }

    std::string command = argv[1];
    std::vector<std::string> args;

    for (int i = 2; i < argc; i++) {
        args.push_back(argv[i]);
    }

    // Execute command
    fprintf(stderr, "DEBUG: About to execute command: %s\n", command.c_str());
    fflush(stderr);

    std::cout << "TRACE: About to execute command: " << command << std::endl;
    std::cout.flush();

    int result = executeCommand(command, args);

    fprintf(stderr, "DEBUG: Command execution completed with result: %d\n", result);
    fflush(stderr);

    std::cout << "TRACE: Command execution completed with result: " << result << std::endl;
    std::cout.flush();

    return result;
}

const std::unordered_map<std::string, std::string>& CLIManager::getCommandDescriptions() const {
    return descriptions_;
}

void CLIManager::printHelp(const std::string& command) {
    fprintf(stderr, "DEBUG: CLIManager::printHelp() called with command='%s'\n", command.c_str());
    fflush(stderr);

    if (!command.empty() && handlers_.find(command) != handlers_.end()) {
        // Command-specific help
        fprintf(stderr, "DEBUG: Printing command-specific help for '%s'\n", command.c_str());
        fflush(stderr);
        std::cout << "Usage: " << program_name_ << " " << command << " [options]" << std::endl;
        std::cout << descriptions_.at(command) << std::endl;
        std::cout << std::endl;
        std::cout << "For more information, run '" << program_name_ << " " << command << " --help'" << std::endl;
    } else {
        // General help
        fprintf(stderr, "DEBUG: Printing general help\n");
        fflush(stderr);
        std::cout << "Usage: " << program_name_ << " <command> [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "Available commands:" << std::endl;
        
        // Sort commands for consistent output
        std::vector<std::string> commands;
        for (const auto& [cmd, _] : handlers_) {
            commands.push_back(cmd);
        }
        std::sort(commands.begin(), commands.end());
        
        // Find the longest command for proper alignment
        size_t max_length = 0;
        for (const auto& cmd : commands) {
            max_length = std::max(max_length, cmd.length());
        }
        
        // Print commands and descriptions
        for (const auto& cmd : commands) {
            std::cout << "  " << std::left << std::setw(max_length + 2) << cmd;
            auto it = descriptions_.find(cmd);
            if (it != descriptions_.end()) {
                std::cout << it->second;
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "For more information on a command, run '" << program_name_ << " <command> --help'" << std::endl;
    }
}

const std::string& CLIManager::getProgramName() const {
    return program_name_;
}

} // namespace cli
} // namespace alphazero