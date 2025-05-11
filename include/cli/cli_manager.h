// include/cli/cli_manager.h
#ifndef ALPHAZERO_CLI_MANAGER_H
#define ALPHAZERO_CLI_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include "core/export_macros.h"

namespace alphazero {
namespace cli {

/**
 * @brief Command handler function type
 */
using CommandHandler = std::function<int(const std::vector<std::string>&)>;

/**
 * @brief CLI manager for handling commands
 */
class ALPHAZERO_API CLIManager {
public:
    /**
     * @brief Constructor
     */
    CLIManager();
    
    /**
     * @brief Add a command handler
     * 
     * @param command Command name
     * @param description Command description
     * @param handler Handler function
     */
    void addCommand(const std::string& command, 
                   const std::string& description, 
                   CommandHandler handler);
    
    /**
     * @brief Execute a command
     * 
     * @param command Command name
     * @param args Command arguments
     * @return Exit code
     */
    int executeCommand(const std::string& command, 
                      const std::vector<std::string>& args);
    
    /**
     * @brief Parse command line arguments and execute command
     * 
     * @param argc Argument count
     * @param argv Argument values
     * @return Exit code
     */
    int run(int argc, char** argv);
    
    /**
     * @brief Get the command descriptions
     * 
     * @return Map of commands to descriptions
     */
    const std::unordered_map<std::string, std::string>& getCommandDescriptions() const;
    
    /**
     * @brief Print help message
     * 
     * @param command Optional command for specific help
     */
    void printHelp(const std::string& command = "");
    
    /**
     * @brief Get global program name
     * 
     * @return Program name
     */
    const std::string& getProgramName() const;
    
private:
    // Program name
    std::string program_name_;
    
    // Command handlers
    std::unordered_map<std::string, CommandHandler> handlers_;
    
    // Command descriptions
    std::unordered_map<std::string, std::string> descriptions_;
};

} // namespace cli
} // namespace alphazero

#endif // ALPHAZERO_CLI_MANAGER_H