cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0146 OLD)

# Add vcpkg integration if available
if(EXISTS "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    
    # Add vcpkg directories to search paths
    list(APPEND CMAKE_PREFIX_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows")
    list(APPEND CMAKE_LIBRARY_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/lib")
    list(APPEND CMAKE_INCLUDE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/include")
endif()

project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)

# Find pthreads (installed via vcpkg)
if(WIN32)
    # With vcpkg toolchain, find_package(Threads) should pick up pthreads
    find_package(Threads QUIET) 
    if(Threads_FOUND)
        message(STATUS "Found Threads (pthreads via vcpkg): ${CMAKE_THREAD_LIBS_INIT}")
        # The Threads::Threads target will be available for linking.
        # Add any necessary compile definitions if pthreads needs them (usually not for basic use)
        # For example, if PTHREADS_WIN32 is needed by your code with pthreads-win32
        # add_compile_definitions(PTHREADS_WIN32) 
    else()
        message(WARNING "Standard CMake FindThreads did not find pthreads. Please ensure 'pthreads:x64-windows' is correctly installed via vcpkg and vcpkg integration is active. Threading may not work correctly.")
    endif()
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# Windows-specific configurations (MSVC 2022)
if(MSVC)
    # Use multi-threaded compilation
    add_compile_options(/MP)
    # Use more restrictive warnings
    add_compile_options(/W4)
    # Disable warnings about unsafe standard functions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Prevent warnings from being treated as errors
    add_compile_options(/WX-)
    # Enable AVX2 instructions if available
    add_compile_options(/arch:AVX2)
    # Enable permissive mode for better standards compliance
    add_compile_options(/permissive-)
    # Optimize for speed in Release mode
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    # For gcc/clang
    add_compile_options(-Wall -Wextra)
    # Enable AVX2 instructions if available
    if(NOT APPLE)
        add_compile_options(-mavx2)
    endif()
    # Optimize for speed in Release mode
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" ON)
option(WITH_TORCH "Build with PyTorch support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example executables" OFF)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find PyTorch if required
if(WITH_TORCH)
    # Set environment variables for PyTorch's cuDNN detection
    set(ENV{CUDNN_LIBRARY} "C:/Program Files/NVIDIA/CUDNN/v9.7/lib/12.8/x64/cudnn.lib")
    set(ENV{CUDNN_INCLUDE_DIR} "C:/Program Files/NVIDIA/CUDNN/v9.7/include/12.8")
    set(ENV{CUDNN_ROOT_DIR} "C:/Program Files/NVIDIA/CUDNN/v9.7")

    # Set standard CMake variables to hint at cuDNN location for PyTorch's find scripts
    set(CUDNN_ROOT_DIR "C:/Program Files/NVIDIA/CUDNN/v9.7" CACHE PATH "Path to cuDNN root directory" FORCE)
    set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA/CUDNN/v9.7/include/12.8" CACHE PATH "Path to cuDNN include directory" FORCE)
    set(CUDNN_LIBRARY_DIR "C:/Program Files/NVIDIA/CUDNN/v9.7/lib/12.8/x64" CACHE PATH "Path to cuDNN library directory" FORCE)

    # Add WITH_TORCH definition for your project's source code
    add_compile_definitions(WITH_TORCH)

    # Attempt to find CUDA Toolkit FIRST. This is crucial for PyTorch's CMake scripts that depend on CUDA targets.
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CMake found CUDA Toolkit Version: ${CUDA_VERSION_STRING}. CUDA targets like CUDA::nvtoolsext should be available.")
        
        if(NOT TARGET CUDA::nvToolsExt)
            # Look specifically for nvperf_host.lib that we know exists
            set(NVPERF_HOST_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/lib64/nvperf_host.lib")
            
            if(EXISTS "${NVPERF_HOST_LIBRARY}")
                message(STATUS "Found nvperf_host library: ${NVPERF_HOST_LIBRARY}")
                add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
                set_target_properties(CUDA::nvToolsExt PROPERTIES
                    IMPORTED_LOCATION "${NVPERF_HOST_LIBRARY}")
                    
                # Make sure the include directory is correct
                set_target_properties(CUDA::nvToolsExt PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/include")
            else()
                message(WARNING "nvperf_host library not found at expected location. Creating dummy target.")
                add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            endif()
        endif()
    else()
        message(WARNING "CMake could not find the CUDA Toolkit. If PyTorch is built with CUDA support, its configuration might fail or it might run in CPU mode.")
    endif()

    # Try to find cuDNN before PyTorch
    find_library(CUDNN_LIBRARY
        NAMES cudnn cudnn_static cudnn64_8
        PATHS "C:/Program Files/NVIDIA/CUDNN/v9.7/lib/12.8/x64"
        NO_DEFAULT_PATH)
    
    if(CUDNN_LIBRARY)
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
        # Set a flag that PyTorch can use
        set(CUDNN_FOUND TRUE)
        set(USE_CUDNN 1 CACHE BOOL "Use cuDNN" FORCE)
        add_compile_definitions(USE_CUDNN=1)
        
        # Force PyTorch to use cuDNN by setting these variables in the cache
        set(TORCH_USE_CUDNN 1 CACHE BOOL "Use cuDNN with PyTorch" FORCE)
        set(USE_CUDNN 1 CACHE BOOL "PyTorch cuDNN flag" FORCE) 
        set(CAFFE2_USE_CUDNN 1 CACHE BOOL "Caffe2 cuDNN flag" FORCE)
        # Also set environment variable
        set(ENV{USE_CUDNN} "1")
    else()
        message(WARNING "cuDNN library not found at specified location. Please verify that the cudnn.lib exists at C:/Program Files/NVIDIA/CUDNN/v9.7/lib/12.8")
    endif()

    # Also explicitly check if cudnn.h exists
    if(NOT EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
        message(WARNING "cudnn.h not found at ${CUDNN_INCLUDE_DIR}. Please verify the path is correct.")
    endif()

    # Attempt to find PyTorch in default locations
    find_package(Torch QUIET HINTS ${Torch_DIR} 
                 $ENV{TORCH_HOME}/share/cmake/Torch 
                 C:/libtorch/share/cmake/Torch)

    # If not found, try the specified fallback location
    if(NOT Torch_FOUND)
        message(STATUS "PyTorch not found in default paths. Attempting to locate PyTorch in C:/libtorch.")
        set(Torch_DIR "C:/libtorch/share/cmake/Torch" CACHE PATH "Path to PyTorch CMake config" FORCE)
        find_package(Torch QUIET)
    endif()

    # Process if PyTorch was found (either by default or fallback)
    if(Torch_FOUND)
        message(STATUS "PyTorch found: ${TORCH_VERSION}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

        # Force TORCH_USE_CUDNN if we found cuDNN
        if(CUDNN_FOUND)
            add_compile_definitions(TORCH_USE_CUDNN=1)
            add_compile_definitions(USE_CUDNN=1)
        endif()

        # Check if PyTorch itself found and enabled CUDA and CUDNN
        # These variables (TORCH_USE_CUDA, TORCH_USE_CUDNN) are set by PyTorch's CMake scripts.
        if(DEFINED TORCH_USE_CUDA AND DEFINED TORCH_USE_CUDNN) # Check if variables are defined
            if(TORCH_USE_CUDA AND TORCH_USE_CUDNN)
                message(STATUS "PyTorch reports CUDA and cuDNN are enabled.")
                add_compile_definitions(USE_CUDNN=1) # Define USE_CUDNN for your project's code
            else()
                message(WARNING "PyTorch reports CUDA or cuDNN is NOT enabled. (TORCH_USE_CUDA: ${TORCH_USE_CUDA}, TORCH_USE_CUDNN: ${TORCH_USE_CUDNN}). Your project's cuDNN features might be disabled or not performant.")
            endif()
        else()
            message(WARNING "PyTorch's CUDA/cuDNN status variables (TORCH_USE_CUDA, TORCH_USE_CUDNN) are not defined. Cannot reliably determine cuDNN status from PyTorch.")
        endif()
    else()
        message(WARNING "PyTorch not found even after checking C:/libtorch. Neural network features will remain disabled.")
        set(WITH_TORCH OFF) # Update the option to reflect that PyTorch is not available
        # NN_SOURCES and other dependent parts are already conditional on WITH_TORCH.
    endif()
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found. Python bindings will be built.")
        add_compile_definitions(BUILD_PYTHON_BINDINGS)
    else()
        message(WARNING "pybind11 not found. Python bindings will be disabled.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# Find OpenMP for parallel MCTS
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Parallel MCTS will be enabled.")
    add_compile_definitions(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Parallel MCTS will be disabled.")
endif()

# Define core sources
set(CORE_SOURCES
    src/core/igamestate.cpp
    src/core/game_export.cpp
)

# Define chess sources
set(CHESS_SOURCES
    src/games/chess/chess_state.cpp
    src/games/chess/chess_rules.cpp
    src/games/chess/chess960.cpp
)

# Define go sources
set(GO_SOURCES
    src/games/go/go_state.cpp
    src/games/go/go_rules.cpp
)

# Define gomoku sources
set(GOMOKU_SOURCES
    src/games/gomoku/gomoku_state.cpp
    src/games/gomoku/gomoku_rules.cpp
)

# Define util sources
set(UTIL_SOURCES
    src/utils/zobrist_hash.cpp
    src/utils/attack_defense_module.cpp
    src/utils/hash_specializations.cpp
)

# Define neural network sources (only if PyTorch is available)
set(NN_SOURCES "")
if(WITH_TORCH)
    set(NN_SOURCES
        src/nn/ddw_randwire_resnet.cpp
    )
endif()

# Define MCTS sources
set(MCTS_SOURCES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/mcts.cpp")
    set(MCTS_SOURCES
        src/mcts/mcts.cpp
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/mcts_node.cpp")
        list(APPEND MCTS_SOURCES src/mcts/mcts_node.cpp)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/mcts_tree.cpp")
        list(APPEND MCTS_SOURCES src/mcts/mcts_tree.cpp)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/parallel_mcts.cpp" AND OpenMP_CXX_FOUND)
        list(APPEND MCTS_SOURCES src/mcts/parallel_mcts.cpp)
    endif()
endif()

# Group all sources
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${CHESS_SOURCES}
    ${GO_SOURCES}
    ${GOMOKU_SOURCES}
    ${UTIL_SOURCES}
    ${NN_SOURCES}
    ${MCTS_SOURCES}
)

# Create the library
if(BUILD_SHARED_LIBS)
    # Create shared library
    add_library(alphazero SHARED ${ALL_SOURCES})
    target_compile_definitions(alphazero PRIVATE ALPHAZERO_EXPORTS)
else()
    # Create static library
    add_library(alphazero STATIC ${ALL_SOURCES})
endif()

# Link with dependencies
if(WITH_TORCH)
    target_link_libraries(alphazero ${TORCH_LIBRARIES})
    
    # Add post-build step to copy PyTorch DLLs
    if(WIN32)
        # Get directory of Torch libraries
        if(EXISTS "C:/libtorch/lib")
            set(TORCH_LIB_DIR "C:/libtorch/lib")
        elseif(DEFINED Torch_DIR AND EXISTS "${Torch_DIR}/../../../lib")
            # Try to infer the lib directory from Torch_DIR
            get_filename_component(TORCH_BASE_DIR "${Torch_DIR}/../../.." ABSOLUTE)
            set(TORCH_LIB_DIR "${TORCH_BASE_DIR}/lib")
        endif()
        
        if(DEFINED TORCH_LIB_DIR)
            # Copy PyTorch DLLs to bin directory
            add_custom_command(TARGET alphazero POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TORCH_LIB_DIR}/torch_cpu.dll"
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/torch_cpu.dll"
                COMMENT "Copying PyTorch DLLs to output directory"
            )
            
            # Copy c10 library if it exists
            if(EXISTS "${TORCH_LIB_DIR}/c10.dll")
                add_custom_command(TARGET alphazero POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TORCH_LIB_DIR}/c10.dll"
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/c10.dll"
                    COMMENT "Copying c10.dll to output directory"
                )
            endif()
            
            # Copy c10_cuda library if it exists
            if(EXISTS "${TORCH_LIB_DIR}/c10_cuda.dll")
                add_custom_command(TARGET alphazero POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TORCH_LIB_DIR}/c10_cuda.dll"
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/c10_cuda.dll"
                    COMMENT "Copying c10_cuda.dll to output directory"
                )
            endif()
            
            # Copy CUDA libraries if they exist
            if(EXISTS "${TORCH_LIB_DIR}/torch_cuda.dll")
                add_custom_command(TARGET alphazero POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TORCH_LIB_DIR}/torch_cuda.dll"
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/torch_cuda.dll"
                    COMMENT "Copying torch_cuda.dll to output directory"
                )
            endif()
            
            message(STATUS "Added post-build commands to copy PyTorch DLLs from ${TORCH_LIB_DIR}")
        else()
            message(WARNING "Could not determine PyTorch library directory. DLLs will not be copied automatically.")
        endif()
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero OpenMP::OpenMP_CXX)
endif()

# Python module
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
endif()

# Example executables
if(BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/chess_self_play.cpp")
        add_executable(chess_self_play examples/chess_self_play.cpp)
        target_link_libraries(chess_self_play alphazero)
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/go_self_play.cpp")
        add_executable(go_self_play examples/go_self_play.cpp)
        target_link_libraries(go_self_play alphazero)
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/gomoku_self_play.cpp")
        add_executable(gomoku_self_play examples/gomoku_self_play.cpp)
        target_link_libraries(gomoku_self_play alphazero)
    endif()
endif()

# Testing with Google Test
if(BUILD_TESTS)
    # Include CTest to enable testing
    include(CTest)
    enable_testing()
    
    # Setup GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Add include directory for tests
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_BINARY_DIR}/include)
    
    # Create a custom target to copy all PyTorch DLLs to the test directory
    if(WIN32 AND WITH_TORCH AND EXISTS "C:/libtorch/lib")
        add_custom_target(copy_test_dlls ALL)
        
        # Function to copy a DLL file if it exists
        function(copy_dll_if_exists src_file)
            if(EXISTS "${src_file}")
                add_custom_command(
                    TARGET copy_test_dlls
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${src_file}"
                    "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
                    COMMENT "Copying ${src_file} to output directory"
                )
            endif()
        endfunction()
        
        # Copy all required DLLs from libtorch
        file(GLOB TORCH_DLLS "C:/libtorch/lib/*.dll")
        foreach(DLL_FILE ${TORCH_DLLS})
            copy_dll_if_exists("${DLL_FILE}")
        endforeach()
    endif()
    
    # Define test sources
    set(CORE_TEST_SOURCES
        tests/core/igamestate_test.cpp
        tests/core/game_export_test.cpp
    )
    
    set(CHESS_TEST_SOURCES
        tests/games/chess/chess_test.cpp
    )
    
    set(GO_TEST_SOURCES
        tests/games/go/go_test.cpp
    )
    
    set(GOMOKU_TEST_SOURCES
        tests/games/gomoku/gomoku_test.cpp
    )
    
    # MCTS test sources
    set(MCTS_TEST_SOURCES "")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mcts/mcts_test.cpp")
        set(MCTS_TEST_SOURCES
            tests/mcts/mcts_test.cpp
        )
    endif()
    
    set(ALL_TEST_SOURCES
        ${CORE_TEST_SOURCES}
        ${CHESS_TEST_SOURCES}
        ${GO_TEST_SOURCES}
        ${GOMOKU_TEST_SOURCES}
        ${MCTS_TEST_SOURCES}
    )
    
    # Function to add test executable
    function(add_test_executable name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} alphazero gtest gtest_main)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${name} OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${name} COMMAND ${name})
        # Add dependency on copy_test_dlls target if it exists
        if(TARGET copy_test_dlls)
            add_dependencies(${name} copy_test_dlls)
        endif()
    endfunction()
    
    # Add test executables
    # Core tests need special handling to avoid multiple main functions
    add_executable(core_tests 
        tests/core_tests_main.cpp
        ${CORE_TEST_SOURCES}
    )
    target_link_libraries(core_tests alphazero gtest)
    target_compile_definitions(core_tests PRIVATE CUSTOM_MAIN_USED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(core_tests OpenMP::OpenMP_CXX)
    endif()
    # Add dependency on copy_test_dlls target if it exists
    if(TARGET copy_test_dlls)
        add_dependencies(core_tests copy_test_dlls)
    endif()
    add_test(NAME core_tests COMMAND core_tests)

    # Other individual tests can use the standard approach
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/games/chess/chess_test.cpp")
        add_test_executable(chess_tests ${CHESS_TEST_SOURCES})
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/games/go/go_test.cpp")
        add_test_executable(go_tests ${GO_TEST_SOURCES})
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/games/gomoku/gomoku_test.cpp")
        add_test_executable(gomoku_tests ${GOMOKU_TEST_SOURCES})
    endif()
    
    # Add MCTS tests if they exist
    if(MCTS_TEST_SOURCES)
        add_test_executable(mcts_tests ${MCTS_TEST_SOURCES})
    endif()
    
    # Add all tests combined - but with a custom approach to avoid multiple main functions
    add_executable(all_tests 
        tests/all_tests_main.cpp
        ${ALL_TEST_SOURCES}
    )
    target_link_libraries(all_tests alphazero gtest)  # Note: no gtest_main here
    target_compile_definitions(all_tests PRIVATE CUSTOM_MAIN_USED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(all_tests OpenMP::OpenMP_CXX)
    endif()
    # Add dependency on copy_test_dlls target if it exists
    if(TARGET copy_test_dlls)
        add_dependencies(all_tests copy_test_dlls)
    endif()
    add_test(NAME all_tests COMMAND all_tests)
endif()

# Install targets
install(TARGETS alphazero
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/ DESTINATION include)