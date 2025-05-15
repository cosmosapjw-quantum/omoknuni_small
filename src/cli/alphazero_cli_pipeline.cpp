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
#include "evaluation/model_evaluator.h"

namespace alphazero {
namespace cli {

// CLI command handler for the complete AlphaZero pipeline
int runPipelineCommand(const std::vector<std::string>& args) {
    std::cout << "Executing AlphaZero training pipeline..." << std::endl;
    
    // Parse arguments for configuration
    std::string config_path;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--config" && i + 1 < args.size()) {
            config_path = args[i + 1];
            break;
        }
    }
    
    if (config_path.empty()) {
        std::cerr << "Error: Config file path not specified. Use --config <path>" << std::endl;
        return 1;
    }
    
    try {
        // Parse the pipeline configuration
        PipelineConfig config = parsePipelineConfig(config_path);
        
        // Run the AlphaZero pipeline
        return runAlphaZeroPipeline(config);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during AlphaZero pipeline: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace cli
} // namespace alphazero