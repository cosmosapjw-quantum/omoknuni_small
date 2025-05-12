# cmake_minimum_required(VERSION 3.20)          # 3.14 → 3.20 for generator-expr fixes
cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0146 OLD)                 # keep your chosen policy

# Set CUDA architectures early to influence compiler detection for CUDA 12.8+
# This helps avoid nvcc trying to compile for deprecated default architectures like sm_52 during its ID check.
# Using one of the target architectures (e.g., 86 for Ada Lovelace or Hopper H100 if appropriate, or a common one like 75 for Turing)
# set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures for compiler detection and build")
# message(STATUS "Early setting CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES} to aid CUDA compiler ID.")

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
option(BUILD_EXAMPLES         "Build examples"         OFF)

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
    message(STATUS "Providing CUDNN_ROOT hint to PyTorch: ${CUDNN_ROOT}")
    # PyTorch's find_package(CUDNN) should use CUDNN_ROOT to find include/library.
    # We remove previously forced CUDNN_INCLUDE_DIR, CUDNN_LIBRARY, CUDNN_FOUND, USE_CUDNN
    # and related ENV VARS to let PyTorch's detection logic work more naturally.

    find_package(CUDA QUIET) # Find CUDA first to get CUDA_VERSION_STRING
    if(CUDA_FOUND)
        message(STATUS "CUDA toolkit ${CUDA_VERSION_STRING} found.")
        add_compile_definitions(TORCH_USE_CUDA=1)

        # Attempt to give Torch the exact CUDA version string to bypass its detection
        if(DEFINED CUDA_VERSION_STRING)
            set(TORCH_CUDA_VERSION "${CUDA_VERSION_STRING}" CACHE STRING "Pre-set CUDA version for Torch" FORCE)
            message(STATUS "Setting TORCH_CUDA_VERSION to ${TORCH_CUDA_VERSION} to aid Torch's CUDA detection.")
        endif()
        
        # Explicitly set CUDA architectures to avoid auto-detection failures
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
        message(STATUS "Explicitly setting CUDA architectures to: ${CMAKE_CUDA_ARCHITECTURES}")
        
        # Alternative NVCC flags approach if needed
        set(TORCH_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86")
        message(STATUS "Setting explicit TORCH_NVCC_FLAGS: ${TORCH_NVCC_FLAGS}")
        
        # Skip Torch's GPU detection by setting the architecture list directly
        set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6" CACHE STRING "CUDA architectures for PyTorch" FORCE)
        message(STATUS "Setting TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}")
        
        if(NOT TARGET CUDA::nvToolsExt)
            set(NVPERF_HOST_LIB "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/extras/CUPTI/lib64/nvperf_host.lib")
            if(EXISTS "${NVPERF_HOST_LIB}")
                message(STATUS "Using nvperf_host.lib as replacement for nvToolsExt")
                add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
                set_target_properties(CUDA::nvToolsExt PROPERTIES
                    IMPORTED_LOCATION "${NVPERF_HOST_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/extras/CUPTI/include")
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

        # If essential pybind11 commands are not available after find_package, try to include pybind11.cmake manually.
        # We check for both python_add_library (original check) and pybind11_add_module (modern command).
        if(NOT COMMAND python_add_library AND NOT COMMAND pybind11_add_module)
            set(PYBIND11_CMAKE_FILE_TO_INCLUDE "") # Reset variable

            # Attempt 1: Use path from pybind11.get_cmake_dir()
            if(pybind11_cmake_dir_from_python)
                set(CANDIDATE_PATH_FROM_PYTHON "${pybind11_cmake_dir_from_python}/pybind11.cmake")
                file(TO_CMAKE_PATH "${CANDIDATE_PATH_FROM_PYTHON}" NORMALIZED_PATH_PYTHON)
                if(EXISTS "${NORMALIZED_PATH_PYTHON}")
                    set(PYBIND11_CMAKE_FILE_TO_INCLUDE "${NORMALIZED_PATH_PYTHON}")
                endif()
            endif()

            # Attempt 2: Fallback to pybind11_DIR (if different from python derived path and Attempt 1 failed)
            if(PYBIND11_CMAKE_FILE_TO_INCLUDE STREQUAL "" AND pybind11_DIR)
                if(NOT pybind11_cmake_dir_from_python OR NOT ("${pybind11_DIR}" STREQUAL "${pybind11_cmake_dir_from_python}"))
                    set(CANDIDATE_PATH_FROM_DIR "${pybind11_DIR}/pybind11.cmake")
                    file(TO_CMAKE_PATH "${CANDIDATE_PATH_FROM_DIR}" NORMALIZED_PATH_DIR)
                    if(EXISTS "${NORMALIZED_PATH_DIR}")
                        set(PYBIND11_CMAKE_FILE_TO_INCLUDE "${NORMALIZED_PATH_DIR}")
                    endif()
                endif()
            endif()

            if(NOT PYBIND11_CMAKE_FILE_TO_INCLUDE STREQUAL "")
                message(STATUS "Manually including ${PYBIND11_CMAKE_FILE_TO_INCLUDE} to define pybind11 commands.")
                include("${PYBIND11_CMAKE_FILE_TO_INCLUDE}")
                # Check again if commands are available after include
                if(NOT COMMAND python_add_library AND NOT COMMAND pybind11_add_module)
                    message(WARNING "Pybind11 commands (python_add_library/pybind11_add_module) still not available after including ${PYBIND11_CMAKE_FILE_TO_INCLUDE}.")
                    set(BUILD_PYTHON_BINDINGS OFF) # Disable bindings if include didn't make commands available
                else()
                    message(STATUS "Pybind11 commands now available after manual include.")
                endif()
            else()
                message(WARNING "Required pybind11 commands not found, and could not locate pybind11.cmake to include it (tried paths derived from python executable and pybind11_DIR).")
                set(BUILD_PYTHON_BINDINGS OFF) # Disable bindings if .cmake file not found/included
            endif()
        else()
            message(STATUS "Pybind11 commands (python_add_library or pybind11_add_module) already available after find_package.")
        endif()
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
    src/mcts/transposition_table.cpp
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
    set(TORCH_CORE_DLLS
        "c10.dll"
        "torch.dll"
        "torch_cpu.dll"
        "torch_global_deps.dll" # Often bundles other dependencies like OpenMP runtimes
        "asmjit.dll"
        "fbgemm.dll"
        "dnnl.dll"
        "libiomp5md.dll"        # Intel OpenMP runtime
        "libiompstubs5md.dll"   # Intel OpenMP stubs - problematic one, copy if exists
        "pthreadpool.dll"
        "uv.dll"
    )

    set(TORCH_CUDA_DLLS "")
    if(CUDA_FOUND) # Or could check TORCH_USE_CUDA after find_package(Torch)
        set(TORCH_CUDA_DLLS
            "c10_cuda.dll"
            "torch_cuda.dll"
            "nvfuser_codegen.dll"
        )
        # Try to find any cuDNN runtime DLLs bundled with PyTorch in its lib directory
        file(GLOB BUNDLED_CUDNN_DLLS RELATIVE "${Torch_LIB_DIR}" "${Torch_LIB_DIR}/cudnn*.dll")
        if(BUNDLED_CUDNN_DLLS)
            message(STATUS "Found bundled cuDNN DLLs in Torch lib: ${BUNDLED_CUDNN_DLLS}")
            list(APPEND TORCH_CUDA_DLLS ${BUNDLED_CUDNN_DLLS})
        endif()
        file(GLOB BUNDLED_ZLIB_DLLS RELATIVE "${Torch_LIB_DIR}" "${Torch_LIB_DIR}/zlib*.dll") # e.g. zlibwapi.dll for cuDNN
        if(BUNDLED_ZLIB_DLLS)
            message(STATUS "Found bundled zlib DLLs in Torch lib: ${BUNDLED_ZLIB_DLLS}")
            list(APPEND TORCH_CUDA_DLLS ${BUNDLED_ZLIB_DLLS})
        endif()
    endif()

    set(ALL_TORCH_DLLS ${TORCH_CORE_DLLS} ${TORCH_CUDA_DLLS})
    list(REMOVE_DUPLICATES ALL_TORCH_DLLS)

    set(TARGET_CONFIGS Debug Release RelWithDebInfo MinSizeRel)
    # If CMAKE_CONFIGURATION_TYPES is set (e.g. by Visual Studio generator), prefer it.
    # Otherwise, if it's a single-config generator, CMAKE_BUILD_TYPE will be the one.
    # The explicit list above is a fallback for wider compatibility in copy logic.
    if(CMAKE_CONFIGURATION_TYPES)
        set(EFFECTIVE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(EFFECTIVE_CONFIGS ${CMAKE_BUILD_TYPE})
    else()
        set(EFFECTIVE_CONFIGS ${TARGET_CONFIGS}) # Fallback to the full list
    endif()

    message(STATUS "Preparing to copy selected PyTorch runtime DLLs to output directories.")
    foreach(CONFIG ${EFFECTIVE_CONFIGS})
        set(DEST_DIR "${CMAKE_BINARY_DIR}/bin/${CONFIG}")
        file(MAKE_DIRECTORY "${DEST_DIR}") # Ensure directory exists

        message(STATUS "Copying PyTorch DLLs for ${CONFIG} configuration to ${DEST_DIR}")
        foreach(DLL_NAME ${ALL_TORCH_DLLS})
            set(DLL_PATH "${Torch_LIB_DIR}/${DLL_NAME}")
            if(EXISTS "${DLL_PATH}")
                file(COPY "${DLL_PATH}" DESTINATION "${DEST_DIR}")
            else()
                # Be less alarming for potentially optional DLLs.
                message(STATUS "Skipping copy of non-existent or optional DLL: ${DLL_PATH} for ${CONFIG} config.")
            endif()
        endforeach()
    endforeach()
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

    # Check if googletest is built as static or shared library
    get_target_property(gtest_type GTest::gtest TYPE)
    if(gtest_type STREQUAL "SHARED_LIBRARY")
        set(GTEST_IS_SHARED TRUE)
        message(STATUS "GoogleTest is built as shared library - adjusting test configuration")
    else()
        set(GTEST_IS_SHARED FALSE)
        message(STATUS "GoogleTest is built as ${gtest_type} - treating as static library")
    endif()

    function(make_test name)
        add_executable(${name} ${ARGN})

        # Set GTest linkage definition based on actual library type
        if(GTEST_IS_SHARED)
            target_compile_definitions(${name} PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=1)
        else()
            target_compile_definitions(${name} PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=0)
        endif()
        
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
        
        # For Windows shared library builds, set the PATH environment variable
        # to include the vcpkg directory with the GTest DLLs and the test executable's directory
        if(WIN32 AND GTEST_IS_SHARED)
            file(TO_NATIVE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/bin" VCPKG_BIN_NATIVE_PATH)
            file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/bin" PROJECT_BIN_DIR_NATIVE_PART)
            # $<CONFIG> will be evaluated by CTest at test time.
            set(CTEST_CONFIG_SPECIFIC_BIN_PATH "${PROJECT_BIN_DIR_NATIVE_PART}/$<CONFIG>")

            set_tests_properties(${name} PROPERTIES
                ENVIRONMENT "PATH=${VCPKG_BIN_NATIVE_PATH};${CTEST_CONFIG_SPECIFIC_BIN_PATH};$ENV{PATH}")
        endif()
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
    make_test(transposition_table_test tests/mcts/transposition_table_test.cpp)
    make_test(transposition_integration_test tests/mcts/transposition_integration_test.cpp)
    if(MCTS_TEST_SOURCES)
        make_test(mcts_tests ${MCTS_TEST_SOURCES})
    endif()
    make_test(all_tests     tests/all_tests_main.cpp
                             ${CORE_TEST_SOURCES} ${CHESS_TEST_SOURCES}
                             ${GO_TEST_SOURCES} ${GOMOKU_TEST_SOURCES}
                             ${MCTS_TEST_SOURCES})
    
    # Increase stack size for all_tests on MSVC to prevent stack overflow
    if(MSVC AND TARGET all_tests)
        target_link_options(all_tests PRIVATE "/STACK:8388608") # 8MB
    endif()
    
    # Accept shared library for GTest
    if(TARGET GTest::gtest)
        get_target_property(gtest_type GTest::gtest TYPE)
        if(gtest_type STREQUAL "INTERFACE_LIBRARY" OR gtest_type STREQUAL "STATIC_LIBRARY")
            message(STATUS "GoogleTest (GTest::gtest) found as ${gtest_type}, static linkage will be used.")
        elseif(gtest_type STREQUAL "SHARED_LIBRARY")
            message(STATUS "GoogleTest (GTest::gtest) found as shared library, adjusted build settings accordingly.")
        else()
            message(WARNING "GoogleTest (GTest::gtest) is of type ${gtest_type}, which is unexpected. Watch for linking errors.")
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
