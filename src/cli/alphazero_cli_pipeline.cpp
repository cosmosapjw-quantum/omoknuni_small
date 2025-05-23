#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include "cli/alphazero_pipeline.h"
#include "cli/cli_manager.h"
#include "core/game_export.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"
#include "selfplay/self_play_manager.h"

namespace alphazero {
namespace cli {

// CLI command handler for the complete AlphaZero pipeline
int runPipelineCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: No configuration file specified." << std::endl;
        std::cerr << "Usage: omoknuni-cli pipeline <config_file.yaml>" << std::endl;
        return 1;
    }

    // Get config file path
    const std::string& config_path = args[0];
    
    // Run the pipeline with the config file
    std::cout << "Starting AlphaZero pipeline with config: " << config_path << std::endl;
    try {
        return runAlphaZeroPipelineFromConfig(config_path);
    }
    catch (const std::exception& e) {
        std::cerr << "Error running pipeline: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace cli
} // namespace alphazero