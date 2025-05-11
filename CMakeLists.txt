cmake_minimum_required(VERSION 3.20)          # 3.14 → 3.20 for generator-expr fixes
cmake_policy(SET CMP0146 OLD)                 # keep your chosen policy

# ─────────────────────────────── vcpkg integration ─────────────────────────
if(EXISTS "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")

    list(APPEND CMAKE_PREFIX_PATH  "$ENV{USERPROFILE}/vcpkg/installed/x64-windows")
    list(APPEND CMAKE_LIBRARY_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/lib")
    list(APPEND CMAKE_INCLUDE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/include")
endif()

project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)

# ────────────────────────────────────── Threads (Win) ──────────────────────────
if(WIN32)
    find_package(Threads QUIET)
    if(Threads_FOUND)
        message(STATUS "Found Threads (pthreads via vcpkg): ${CMAKE_THREAD_LIBS_INIT}")
    else()
        message(WARNING "pthreads:x64-windows not found – threading disabled.")
    endif()
endif()

# ────────────────────────────────── Standard / warnings ────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

if(MSVC)
    add_compile_options(/MP /W4 /WX- /arch:AVX2 /permissive- /utf-8)
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    add_compile_options(-Wall -Wextra)
    if(NOT APPLE)
        add_compile_options(-mavx2)
    endif()
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

# ───────────────────────────────────── Options ─────────────────────────────────
option(BUILD_PYTHON_BINDINGS  "Build Python bindings (DISABLED due to pybind11 issues)"  ON)
option(BUILD_TESTS            "Build tests"            ON)
option(WITH_TORCH             "Build with PyTorch"     ON)
option(BUILD_SHARED_LIBS      "Build shared libs"      ON)
option(BUILD_EXAMPLES         "Build examples"         ON)

include(GNUInstallDirs)

# ─────────────────────────── Configure library exports ────────────────────────
# Define macros for symbol visibility in shared libraries
if(WIN32)
    if(BUILD_SHARED_LIBS)
        set(ALPHAZERO_DLL_EXPORT "__declspec(dllexport)")
        set(ALPHAZERO_DLL_IMPORT "__declspec(dllimport)")
    else()
        set(ALPHAZERO_DLL_EXPORT "")
        set(ALPHAZERO_DLL_IMPORT "")
    endif()
else()
    set(ALPHAZERO_DLL_EXPORT "__attribute__((visibility(\"default\")))")
    set(ALPHAZERO_DLL_IMPORT "")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/alphazero_export.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/alphazero_export.h"
)

# ───────────────────────────────────── Tests setup ────────────────────────────
if(BUILD_TESTS)
    include(CTest)
    enable_testing()

    # Setup googletest first before defining any of our libraries or targets
    # Use find_package as googletest is provided by vcpkg
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest (via vcpkg): ${GTEST_VERSION}")
    
    # Set property to tell tests that GTest is static
    # vcpkg usually builds GTest as static by default.
    # This definition helps the test code understand GTest's linkage.
    set(GTEST_LINKED_AS_SHARED_LIBRARY FALSE)
endif()

# ───────────────────────────────────── Torch / CUDA ────────────────────────────
if(WITH_TORCH)
    # --- cuDNN hints (edit if your layout differs) ----------------------------
    # User needs to ensure CUDNN_ROOT points to a cuDNN version compatible with the detected CUDA_VERSION
    set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.7" CACHE PATH "cuDNN root directory (MUST match CUDA version used by Torch)")
    # The following paths are derived assuming a standard cuDNN layout for a *specific* CUDA version.
    # If CUDA toolkit is 12.6, these paths should correspond to cuDNN for CUDA 12.6.
    # Currently, user's CUDNN_ROOT points to v9.7 which seems to have a 12.8 subdir. This will likely cause issues if Torch uses CUDA 12.6.
    set(CUDNN_INCLUDE_DIR_PATH "${CUDNN_ROOT}/include/12.8") # This path component (12.8) should match the CUDA version Torch will use.
    set(CUDNN_LIBRARY_DIR_PATH "${CUDNN_ROOT}/lib/12.8/x64")   # Same here for 12.8.
    set(CUDNN_LIBRARY_FILE_PATH "${CUDNN_LIBRARY_DIR_PATH}/cudnn.lib")

    set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_DIR_PATH}" CACHE PATH "Path to cuDNN include directory" FORCE)
    set(CUDNN_LIBRARY "${CUDNN_LIBRARY_FILE_PATH}" CACHE FILEPATH "Path to cuDNN library" FORCE)
    set(CUDNN_ROOT_DIR "${CUDNN_ROOT}" CACHE PATH "Path to cuDNN root directory (alternative hint)" FORCE)

    set(ENV{CUDNN_LIBRARY}      "${CUDNN_LIBRARY_FILE_PATH}")
    set(ENV{CUDNN_INCLUDE_DIR}  "${CUDNN_INCLUDE_DIR_PATH}")
    set(ENV{CUDNN_ROOT_DIR}     "${CUDNN_ROOT}")
    
    message(STATUS "Configuring with cuDNN hints:")
    message(STATUS "  CUDNN_ROOT (user hint): ${CUDNN_ROOT}")
    message(STATUS "  CUDNN_INCLUDE_DIR (cache hint for Torch): ${CUDNN_INCLUDE_DIR}")
    message(STATUS "  CUDNN_LIBRARY (cache hint for Torch): ${CUDNN_LIBRARY}")
    message(STATUS "  CUDNN_ROOT_DIR (cache hint for Torch): ${CUDNN_ROOT_DIR}")

    if(NOT EXISTS "${CUDNN_INCLUDE_DIR}")
        message(WARNING "cuDNN include directory (pointed to by CUDNN_INCLUDE_DIR) not found: ${CUDNN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${CUDNN_LIBRARY}")
        message(WARNING "cuDNN library (pointed to by CUDNN_LIBRARY) not found: ${CUDNN_LIBRARY}")
    endif()

    find_package(CUDA QUIET) # Find CUDA first to get CUDA_VERSION_STRING
    if(CUDA_FOUND)
        message(STATUS "CUDA toolkit ${CUDA_VERSION_STRING} found.")
        add_compile_definitions(TORCH_USE_CUDA=1)

        # Attempt to give Torch the exact CUDA version string to bypass its detection
        if(DEFINED CUDA_VERSION_STRING)
            set(TORCH_CUDA_VERSION "${CUDA_VERSION_STRING}" CACHE STRING "Pre-set CUDA version for Torch" FORCE)
            message(STATUS "Setting TORCH_CUDA_VERSION to ${TORCH_CUDA_VERSION} to aid Torch's CUDA detection.")
        endif()
        
        if(NOT TARGET CUDA::nvToolsExt)
            set(NVPERF_HOST_LIB "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/lib64/nvperf_host.lib")
            if(EXISTS "${NVPERF_HOST_LIB}")
                message(STATUS "Using nvperf_host.lib as replacement for nvToolsExt")
                add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
                set_target_properties(CUDA::nvToolsExt PROPERTIES
                    IMPORTED_LOCATION "${NVPERF_HOST_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/include")
            else()
                message(WARNING "nvperf_host.lib not found, creating interface-only target")
                add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            endif()
        endif()
    else()
        message(WARNING "CUDA toolkit not found – Torch will build CPU-only.")
    endif()

    set(Torch_DIR "C:/libtorch/share/cmake/Torch" CACHE PATH "Torch_DIR")
    find_package(Torch REQUIRED)

    file(REAL_PATH "${Torch_DIR}/../../../lib" Torch_LIB_DIR)
endif()

# ────────────────────────────────── Python / OpenMP ────────────────────────────
set(PYBIND11_FOUND FALSE)
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "NOTE: Python bindings are currently troublesome due to pybind11 compatibility issues.")
    message(STATUS "      If they fail to build, consider disabling BUILD_PYTHON_BINDINGS option.")
    
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python interpreter: ${Python_EXECUTABLE}")
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_cmake_dir_from_python
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE pybind11_get_cmake_dir_result
            ERROR_QUIET
        )
        if(pybind11_get_cmake_dir_result EQUAL 0 AND EXISTS "${pybind11_cmake_dir_from_python}")
            message(STATUS "Found pybind11 CMake directory via Python: ${pybind11_cmake_dir_from_python}")
            list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir_from_python}")
            # The pybind11_DIR hint is usually not needed if CMAKE_PREFIX_PATH is set correctly,
            # but can be an alternative if find_package still struggles.
            # set(pybind11_DIR "${pybind11_cmake_dir_from_python}" CACHE PATH "pybind11 CMake directory from Python" FORCE)
        else()
            message(STATUS "Could not get pybind11 CMake directory from Python interpreter. Will rely on standard search paths.")
        endif()
    else()
        message(WARNING "Python interpreter not found, cannot automatically locate pip-installed pybind11's CMake files.")
    endif()

    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "Building Python bindings with pybind11 ${pybind11_VERSION}")
        # find_package(pybind11) should make pybind11_add_module available.
        # However, if it doesn't (e.g. due to how pybind11 is installed/packaged),
        # explicitly including pybind11.cmake might be necessary.
        # It defines python_add_library which pybind11_add_module uses.
        if(NOT COMMAND python_add_library) # Check if the command is already known
            if(pybind11_cmake_dir_from_python AND EXISTS "${pybind11_cmake_dir_from_python}/pybind11.cmake")
                message(STATUS "python_add_library not found, explicitly including ${pybind11_cmake_dir_from_python}/pybind11.cmake")
                include("${pybind11_cmake_dir_from_python}/pybind11.cmake")
            elseif(pybind11_DIR AND EXISTS "${pybind11_DIR}/pybind11.cmake") # Fallback if pybind11_DIR was set
                message(STATUS "python_add_library not found, explicitly including ${pybind11_DIR}/pybind11.cmake")
                include("${pybind11_DIR}/pybind11.cmake")
            else()
                message(WARNING "python_add_library command is not defined and could not locate pybind11.cmake to include it.")
                set(BUILD_PYTHON_BINDINGS OFF)
                message(WARNING "Disabling Python bindings due to python_add_library missing.")
            endif()
        endif()
        set(PYBIND11_FOUND TRUE)
    else()
        message(WARNING "pybind11 not found - Python bindings will be disabled. Searched CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

if(PYBIND11_FOUND AND BUILD_PYTHON_BINDINGS)
  # This section may fail if pybind11_add_module doesn't work correctly
  # It's in a separate if check to ensure the rest of the build continues even if this part fails
  message(STATUS "Attempting to build Python module...")
  
  # Try a direct approach to help define missing command
  # This might not work, but worth trying
  if(NOT COMMAND python_add_library)
    message(STATUS "Trying manual pybind11 commands workaround...")
    # Some older versions of pybind11 used this approach:
    macro(pybind11_add_module target_name)
      add_library(${target_name} MODULE ${ARGN})
      target_include_directories(${target_name} PRIVATE ${PYTHON_INCLUDE_DIRS})
      target_link_libraries(${target_name} PRIVATE ${PYTHON_LIBRARIES})
      set_target_properties(${target_name} PROPERTIES 
          PREFIX ""
          SUFFIX "${PYTHON_MODULE_EXTENSION}")
    endmacro()
  endif()
  
  # Only try to create Python module if we believe we can
  if(COMMAND pybind11_add_module)
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
  else()
    message(WARNING "Skipping Python module creation - missing required commands")
  endif()
endif()

find_package(nlohmann_json CONFIG REQUIRED)
message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")

find_package(yaml-cpp CONFIG REQUIRED)
message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    add_compile_definitions(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ─────────────────────────────────── Source sets ───────────────────────────────
set(CORE_SOURCES      src/core/igamestate.cpp src/core/game_export.cpp)
set(CHESS_SOURCES     src/games/chess/chess_state.cpp src/games/chess/chess_rules.cpp src/games/chess/chess960.cpp)
set(GO_SOURCES        src/games/go/go_state.cpp   src/games/go/go_rules.cpp)
set(GOMOKU_SOURCES    src/games/gomoku/gomoku_state.cpp src/games/gomoku/gomoku_rules.cpp)
set(UTIL_SOURCES      src/utils/zobrist_hash.cpp src/utils/attack_defense_module.cpp src/utils/hash_specializations.cpp)

# Conditionally define NN_SOURCES based on WITH_TORCH
set(NN_SOURCES "") # Initialize with non-Torch NN files if any, or empty
if(WITH_TORCH)
    list(APPEND NN_SOURCES
        src/nn/ddw_randwire_resnet.cpp
        src/nn/resnet_model.cpp
        src/nn/neural_network_factory.cpp
    )
endif()

set(MCTS_SOURCES
    src/mcts/mcts_node.cpp
    src/mcts/evaluation_types.cpp
    src/mcts/mcts_evaluator.cpp
    src/mcts/mcts_engine.cpp
)
set(SELFPLAY_SOURCES
    src/selfplay/self_play_manager.cpp
)
set(TRAINING_SOURCES
    src/training/training_data_manager.cpp
)
set(EVALUATION_SOURCES
    src/evaluation/model_evaluator.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/mcts.cpp")
    list(APPEND MCTS_SOURCES src/mcts/mcts.cpp)
    foreach(f IN ITEMS mcts_node.cpp mcts_tree.cpp parallel_mcts.cpp)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mcts/${f}")
            list(APPEND MCTS_SOURCES "src/mcts/${f}")
        endif()
    endforeach()
endif()

set(ALL_SOURCES
    ${CORE_SOURCES} ${CHESS_SOURCES} ${GO_SOURCES} ${GOMOKU_SOURCES}
    ${UTIL_SOURCES} ${NN_SOURCES} ${MCTS_SOURCES}
    ${SELFPLAY_SOURCES} ${TRAINING_SOURCES} ${EVALUATION_SOURCES}
)

# ───────────────────────────── alphazero library target ───────────────────────
add_library(alphazero ${ALL_SOURCES})

target_include_directories(alphazero 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Define ALPHAZERO_EXPORTS when building the library to ensure symbols are exported
if(BUILD_SHARED_LIBS)
    target_compile_definitions(alphazero PRIVATE -DALPHAZERO_EXPORTS)
endif()

if(MSVC)
    # Always use multithreaded DLL runtime (/MD) for both library and tests for consistency
    # This avoids linkage errors when mixing runtime libraries
    target_compile_options(alphazero PRIVATE "/MD$<$<CONFIG:Debug>:d>")
endif()

# Link required libraries
target_link_libraries(alphazero PUBLIC Threads::Threads)
target_link_libraries(alphazero PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(alphazero PUBLIC yaml-cpp::yaml-cpp)

if(WITH_TORCH)
    target_link_libraries(alphazero PUBLIC "${TORCH_LIBRARIES}")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(alphazero PRIVATE /W4 /WX-)
else()
    target_compile_options(alphazero PRIVATE -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-sign-compare)
endif()

# ───────────────────────────── Torch DLL copy helper ──────────────────────────
if(WIN32 AND WITH_TORCH)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/MinSizeRel")
    
    message(STATUS "Copying PyTorch runtime DLLs to output directories")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/Debug")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/Release")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/MinSizeRel")
endif()

# ───────────────────────────── Example executables ────────────────────────────
if(BUILD_EXAMPLES)
    if(WITH_TORCH) # Examples might depend on Torch
      add_executable(alphazero_training examples/alphazero_training.cpp)
      target_link_libraries(alphazero_training PRIVATE alphazero nlohmann_json::nlohmann_json)
      
      add_executable(self_play_libtorch examples/self_play_libtorch.cpp)
      target_link_libraries(self_play_libtorch PRIVATE alphazero nlohmann_json::nlohmann_json)
    endif()

    foreach(game IN ITEMS chess go gomoku)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${game}_self_play.cpp")
            add_executable(${game}_self_play "examples/${game}_self_play.cpp")
            target_link_libraries(${game}_self_play alphazero)
        endif()
    endforeach()
endif()

# ───────────────────────────────────── Tests ──────────────────────────────────
if(BUILD_TESTS)
    set(CORE_TEST_SOURCES   tests/core/igamestate_test.cpp  tests/core/game_export_test.cpp)
    set(CHESS_TEST_SOURCES  tests/games/chess/chess_test.cpp)
    set(GO_TEST_SOURCES     tests/games/go/go_test.cpp)
    set(GOMOKU_TEST_SOURCES tests/games/gomoku/gomoku_test.cpp)
    set(MCTS_TEST_SOURCES   tests/mcts/mcts_node_test.cpp  tests/mcts/mcts_evaluator_test.cpp
                            tests/mcts/mcts_engine_test.cpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mcts/mcts_test.cpp")
        list(APPEND MCTS_TEST_SOURCES tests/mcts/mcts_test.cpp)
    endif()

    function(make_test name)
        add_executable(${name} ${ARGN})

        # Tell tests that GTest is being linked statically.
        # This definition should be set to 0 or OFF.
        # We ensure BUILD_SHARED_LIBS is OFF when gtest itself is configured,
        # and this definition ensures the test code itself sees gtest as static.
        target_compile_definitions(${name} PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=0)
        
        # Common libraries for all tests
        target_link_libraries(${name} PRIVATE alphazero)
        
        # Add the appropriate gtest libraries
        # Make sure to include the GTest include directories
        get_target_property(GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
        if(GTEST_INCLUDE_DIRS)
            target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
        else()
            # Fallback - add FetchContent's googletest source directories
            target_include_directories(${name} PRIVATE 
                "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include"
                "${CMAKE_BINARY_DIR}/_deps/googletest-src/googlemock/include")
        endif()
        
        if (${name} STREQUAL "all_tests" OR ${name} STREQUAL "core_tests")
            # These tests (all_tests, core_tests) provide their own main.cpp
            target_link_libraries(${name} PRIVATE GTest::gmock GTest::gtest)
            target_compile_definitions(${name} PRIVATE CUSTOM_MAIN_USED)
            if (${name} STREQUAL "all_tests")
                target_compile_definitions(${name} PRIVATE BUILDING_TEST_SUITE)
            endif()
        else()
            # These tests use the default main provided by gtest_main
            target_link_libraries(${name} PRIVATE GTest::gtest_main GTest::gtest GTest::gmock)
        endif()

        if(OpenMP_CXX_FOUND)
            target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        
        # Make sure each test executable links to the correct runtime libraries
        if(MSVC)
            # Use multithreaded DLL runtime - same as the alphazero library
            target_compile_options(${name} PRIVATE "/MD$<$<CONFIG:Debug>:d>")
        endif()
        
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    make_test(core_tests    tests/core_tests_main.cpp ${CORE_TEST_SOURCES})
    make_test(chess_tests   ${CHESS_TEST_SOURCES})
    make_test(go_tests      ${GO_TEST_SOURCES})
    make_test(gomoku_tests  ${GOMOKU_TEST_SOURCES})
    make_test(neural_network_test tests/nn/neural_network_test.cpp)
    make_test(mcts_with_nn_test tests/integration/mcts_with_nn_test.cpp)
    make_test(self_play_manager_test tests/selfplay/self_play_manager_test.cpp)
    make_test(training_data_manager_test tests/training/training_data_manager_test.cpp)
    make_test(model_evaluator_test tests/evaluation/model_evaluator_test.cpp)
    if(MCTS_TEST_SOURCES)
        make_test(mcts_tests ${MCTS_TEST_SOURCES})
    endif()
    make_test(all_tests     tests/all_tests_main.cpp
                             ${CORE_TEST_SOURCES} ${CHESS_TEST_SOURCES}
                             ${GO_TEST_SOURCES} ${GOMOKU_TEST_SOURCES}
                             ${MCTS_TEST_SOURCES})
    
    # Check if googletest was built as static libraries as expected
    if(TARGET GTest::gtest)
        get_target_property(gtest_type GTest::gtest TYPE)
        if(gtest_type STREQUAL "INTERFACE_LIBRARY" OR gtest_type STREQUAL "STATIC_LIBRARY")
            # INTERFACE_LIBRARY is also acceptable if it correctly links to a static one
            message(STATUS "GoogleTest (GTest::gtest) found as ${gtest_type}, assuming static linkage as per vcpkg default.")
        elseif(NOT gtest_type STREQUAL "STATIC_LIBRARY")
            message(WARNING "GoogleTest (GTest::gtest) is of type ${gtest_type}, not STATIC_LIBRARY. This may cause linking errors if not intended.")
        else()
            message(STATUS "GoogleTest is correctly building as static libraries.")
        endif()
    else()
        message(WARNING "GTest::gtest target not found after find_package(GTest). This is unexpected.")
    endif()
endif()

# ───────────────────────────────── Installation ───────────────────────────────
install(TARGETS alphazero
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION include)
