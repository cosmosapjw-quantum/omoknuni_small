// tests/cli/cli_manager_test.cpp
#include <gtest/gtest.h>
#include "cli/cli_manager.h"
#include <string>
#include <vector>

using namespace alphazero;

class CLIManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cli = std::make_unique<cli::CLIManager>();
        
        // Add test commands
        cli->addCommand("test", "Test command", 
                      [](const std::vector<std::string>& args) {
                          return args.empty() ? 0 : std::stoi(args[0]);
                      });
        
        cli->addCommand("echo", "Echo arguments", 
                      [](const std::vector<std::string>& args) {
                          for (const auto& arg : args) {
                              std::cout << arg << " ";
                          }
                          std::cout << std::endl;
                          return 0;
                      });
    }
    
    std::unique_ptr<cli::CLIManager> cli;
};

// Test adding commands
TEST_F(CLIManagerTest, AddCommand) {
    auto descriptions = cli->getCommandDescriptions();
    
    // Check that test commands were added
    EXPECT_EQ(descriptions.size(), 3);  // test, echo, help
    EXPECT_EQ(descriptions["test"], "Test command");
    EXPECT_EQ(descriptions["echo"], "Echo arguments");
    EXPECT_EQ(descriptions["help"], "Display help information");
}

// Test executing command
TEST_F(CLIManagerTest, ExecuteCommand) {
    // Test command with no arguments
    int result = cli->executeCommand("test", {});
    EXPECT_EQ(result, 0);
    
    // Test command with argument
    result = cli->executeCommand("test", {"42"});
    EXPECT_EQ(result, 42);
    
    // Test unknown command
    result = cli->executeCommand("unknown", {});
    EXPECT_EQ(result, 1);
}

// Test help command
TEST_F(CLIManagerTest, HelpCommand) {
    // Help should return 0
    int result = cli->executeCommand("help", {});
    EXPECT_EQ(result, 0);
    
    // Help for specific command
    result = cli->executeCommand("help", {"test"});
    EXPECT_EQ(result, 0);
}

// Only include the main function when not building as part of the test suite
#if !defined(CUSTOM_MAIN_USED) && !defined(BUILDING_TEST_SUITE)
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif