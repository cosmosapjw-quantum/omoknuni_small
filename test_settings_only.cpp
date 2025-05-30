#include <iostream>
#include <cstdio>

// Test just the MCTSSettings construction
int main() {
    fprintf(stderr, "[TEST] Starting minimal settings test\n");
    fflush(stderr);
    
    try {
        fprintf(stderr, "[TEST] About to include mcts_engine.h\n");
        fflush(stderr);
        
        #include "mcts/mcts_engine.h"
        
        fprintf(stderr, "[TEST] Header included successfully\n");
        fflush(stderr);
        
        fprintf(stderr, "[TEST] Creating MCTSSettings\n");
        fflush(stderr);
        
        alphazero::mcts::MCTSSettings settings;
        
        fprintf(stderr, "[TEST] MCTSSettings created successfully\n");
        fflush(stderr);
        
        settings.num_simulations = 1;
        settings.use_transposition_table = false;
        
        fprintf(stderr, "[TEST] Settings modified successfully\n");
        fflush(stderr);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Exception: %s\n", e.what());
        return 1;
    }
    
    fprintf(stderr, "[TEST] Test completed successfully\n");
    return 0;
}