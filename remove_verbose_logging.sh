#!/bin/bash

# Script to remove verbose logging while keeping error reports

echo "Removing verbose logging from C++ files..."

# Function to process a file
process_file() {
    local file="$1"
    echo "Processing: $file"
    
    # Create a temporary file
    temp_file="${file}.tmp"
    
    # Remove or comment out verbose logging patterns
    sed -e '/LOG_SYSTEM_INFO/d' \
        -e '/LOG_GAME_INFO/d' \
        -e '/LOG_MCTS_INFO/d' \
        -e '/LOG_NEURAL_NET_INFO/d' \
        -e '/std::cout.*Starting/d' \
        -e '/std::cout.*Progress:/d' \
        -e '/std::cout.*Batch [0-9]/d' \
        -e '/std::cout.*Creating/d' \
        -e '/std::cout.*Initialized/d' \
        -e '/std::cout.*Performance/d' \
        -e '/std::cout.*Configuration/d' \
        -e '/std::cout.*Worker [0-9]/d' \
        -e '/std::cout.*Memory:/d' \
        -e '/std::cout.*Throughput:/d' \
        -e '/std::cout.*📊/d' \
        -e '/std::cout.*✅/d' \
        -e '/std::cout.*🚀/d' \
        -e '/std::cout.*🌲/d' \
        -e '/std::cout.*🛡️/d' \
        -e '/std::cout.*🎯/d' \
        -e '/std::cout.*🏊/d' \
        -e '/std::cout.*\[CONFIG_VERIFY\]/d' \
        -e '/std::cout.*\[MCTS_PERF\]/d' \
        -e '/std::cout.*\[MCTS_ROUTING\]/d' \
        -e '/std::cout.*\[MEMORY_MONITOR\]/d' \
        -e '/std::cout.*SelfPlayManager:/d' \
        -e '/std::cout.*MCTSEngine:/d' \
        -e '/std::cout.*NeuralNetworkFactory:/d' \
        -e '/std::cout.*AggressiveMemoryManager/d' \
        -e '/std::cout.*"  "/d' \
        -e '/std::cout << "=/d' \
        -e '/std::cout.*Initial memory:/d' \
        -e '/std::cout.*Final throughput:/d' \
        -e '/std::cout.*Search completed/d' \
        -e '/std::cout.*Game.*Started/d' \
        -e '/std::cout.*Move.*completed/d' \
        "$file" > "$temp_file"
    
    # Replace the original file
    mv "$temp_file" "$file"
}

# Find and process all C++ source files
find src -name "*.cpp" -type f | while read -r file; do
    process_file "$file"
done

# Process header files too
find include -name "*.h" -type f | while read -r file; do
    process_file "$file"
done

echo "Verbose logging removal complete!"