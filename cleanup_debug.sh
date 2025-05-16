#!/bin/bash

# Clean up debug output from MCTS files
echo "Cleaning up debug output..."

# Remove std::cout debug lines in MCTS engine
sed -i 's/std::cout.*MCTSEngine.*std::endl;//g' src/mcts/mcts_engine.cpp
sed -i 's/std::cout.*\[SEARCH\].*std::endl;//g' src/mcts/mcts_engine.cpp
sed -i 's/std::cout.*\[ENGINE\].*std::endl;//g' src/mcts/mcts_engine.cpp
sed -i 's/MCTS_LOG_VERBOSE.*;//g' src/mcts/mcts_engine.cpp
sed -i 's/MCTS_LOG_DEBUG.*;//g' src/mcts/mcts_engine.cpp

# Remove debug output in evaluator
sed -i 's/std::cout.*\[EVALUATOR\].*std::endl;//g' src/mcts/mcts_evaluator.cpp
sed -i 's/std::cout.*MCTSEvaluator.*std::endl;//g' src/mcts/mcts_evaluator.cpp

# Remove debug output in self play manager
sed -i 's/std::cout.*SelfPlayManager.*std::endl;//g' src/selfplay/self_play_manager.cpp
sed -i 's/std::cout.*Game [0-9].*std::endl;//g' src/selfplay/self_play_manager.cpp

# Remove debug test files
rm -f tests/memory_leak_test.cpp
rm -f tests/nn_mem_test.cpp
rm -f tests/simple_mem_test.cpp
rm -f debug_mem_leak.sh
rm -f config_debug.yaml
rm -f src/mcts/mcts_engine_clean.cpp

echo "Cleanup complete!"