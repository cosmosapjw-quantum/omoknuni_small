# Simple CMake script to build the debug test
add_executable(debug_test
    tests/mcts/debug_test.cpp
)

target_link_libraries(debug_test
    gtest
    alphazero
)

target_include_directories(debug_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)