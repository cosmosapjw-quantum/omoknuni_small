# Force use of system gcc compiler (must be before project declaration)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

cmake_minimum_required(VERSION 3.20)
# Only set CMP0146 policy if CMake version supports it
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# Set CUDA architectures early to influence compiler detection for CUDA 12.8+
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures for compiler detection and build")
message(STATUS "Early setting CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES} to aid CUDA compiler ID.")

# ───────────────────────────── Standard Linux paths ─────────────────────────
# Add common system paths for finding packages
list(APPEND CMAKE_PREFIX_PATH  "/usr/local")
list(APPEND CMAKE_LIBRARY_PATH "/usr/local/lib")
list(APPEND CMAKE_INCLUDE_PATH "/usr/local/include")

project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)

# ────────────────────────────────────── Threads ──────────────────────────
find_package(Threads QUIET)
if(Threads_FOUND)
    message(STATUS "Found Threads: ${CMAKE_THREAD_LIBS_INIT}")
else()
    message(WARNING "Threads not found – threading disabled.")
endif()

# ────────────────────────────────── Standard / warnings ────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Explicitly use the new C++11 ABI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Add debug flags for batch analysis
add_compile_definitions(DEBUG_BATCH_PROCESSING=1)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# Compiler options for GCC/Clang
add_compile_options(-Wall -Wextra)
if(NOT APPLE)
    add_compile_options(-mavx2)
endif()
add_compile_options($<$<CONFIG:Release>:-O3>)

# ───────────────────────────────────── Options ─────────────────────────────────
option(BUILD_PYTHON_BINDINGS  "Build Python bindings (DISABLED due to pybind11 issues)"  ON)
option(BUILD_TESTS            "Build tests"            ON)
option(WITH_TORCH             "Build with PyTorch"     ON)
option(BUILD_SHARED_LIBS      "Build shared libs"      ON)
option(BUILD_EXAMPLES         "Build examples"         OFF)

include(GNUInstallDirs)

# Make sure the include/cli directory is recognized
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/cli")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/cli")
endif()

# ─────────────────────────── Configure library exports ────────────────────────
# Define macros for symbol visibility in shared libraries
set(ALPHAZERO_DLL_EXPORT "__attribute__((visibility(\"default\")))")
set(ALPHAZERO_DLL_IMPORT "")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/alphazero_export.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/alphazero_export.h"
)

# ───────────────────────────────────── Tests setup ────────────────────────────
if(BUILD_TESTS)
    include(CTest)
    enable_testing()

    # Setup googletest first before defining any of our libraries or targets
    # Use find_package for system-installed GTest
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest: ${GTEST_VERSION}")
    
    # Set property to tell tests that GTest is static
    # System GTest is usually built as static by default.
    set(GTEST_LINKED_AS_SHARED_LIBRARY FALSE)
endif()

# ───────────────────────────────────── Torch / CUDA ────────────────────────────
if(WITH_TORCH)
    # --- Find CUDA first to detect available GPUs ---
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA toolkit ${CUDA_VERSION_STRING} found at ${CUDA_TOOLKIT_ROOT_DIR}")
        add_compile_definitions(TORCH_USE_CUDA=1)

        # Attempt to find CUDA include and library paths if not already set
        if(NOT CUDA_INCLUDE_DIRS AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/include")
            set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
            message(STATUS "Setting CUDA_INCLUDE_DIRS to ${CUDA_INCLUDE_DIRS}")
        endif()
        
        if(NOT CUDA_CUDART_LIBRARY)
            find_library(CUDA_CUDART_LIBRARY cudart
                PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                      /usr/local/cuda/lib64
                      /usr/lib/x86_64-linux-gnu
                NO_DEFAULT_PATH)
            if(CUDA_CUDART_LIBRARY)
                message(STATUS "Found CUDA cudart library: ${CUDA_CUDART_LIBRARY}")
            endif()
        endif()

        # --- cuDNN hints for Linux ---
        # Try standard Linux cuDNN locations
        if(EXISTS "/usr/local/cuda/cudnn")
            set(CUDNN_ROOT "/usr/local/cuda/cudnn" CACHE PATH "cuDNN root directory (MUST match CUDA version used by Torch)")
        elseif(EXISTS "/usr/local/cudnn")
            set(CUDNN_ROOT "/usr/local/cudnn" CACHE PATH "cuDNN root directory (MUST match CUDA version used by Torch)")
        elseif(EXISTS "/usr/include/cudnn.h")
            set(CUDNN_ROOT "/usr" CACHE PATH "cuDNN root directory (system installation)")
        endif()
        
        # First try to find cuDNN directly in libtorch directory (if Torch was already found)
        if(WITH_TORCH AND Torch_LIB_DIR)
            find_library(CUDNN_LIBRARY
                NAMES cudnn.9 cudnn libcudnn
                PATHS "${Torch_LIB_DIR}"
                DOC "Path to cuDNN library in libtorch"
                NO_DEFAULT_PATH
            )

            if(CUDNN_LIBRARY)
                message(STATUS "Found cuDNN library in libtorch directory: ${CUDNN_LIBRARY}")
            endif()
        endif()

        # If not found in libtorch, check for apt-installed cuDNN
        if(NOT CUDNN_LIBRARY)
            find_library(CUDNN_LIBRARY
                NAMES cudnn libcudnn
                PATHS /usr/lib/x86_64-linux-gnu
                      /usr/lib
                      /usr/lib64
                DOC "Path to cuDNN library installed by apt"
                NO_DEFAULT_PATH
            )
        endif()
        
        if(CUDNN_LIBRARY)
            message(STATUS "Found cuDNN library (apt-installed): ${CUDNN_LIBRARY}")
            # If not already set, set CUDNN_ROOT to the system root for apt packages
            if(NOT CUDNN_ROOT)
                set(CUDNN_ROOT "/usr" CACHE PATH "cuDNN root directory (apt installation)" FORCE)
            endif()
            # Export as environment variable for PyTorch to find
            set(ENV{CUDNN_LIBRARY} "${CUDNN_LIBRARY}")
        endif()

        # Check for cudnn.h - first in libtorch include directory
        if(WITH_TORCH)
            find_path(CUDNN_INCLUDE_DIR
                NAMES cudnn.h
                PATHS /opt/libtorch/include
                      "${Torch_DIR}/../../include"
                DOC "Path to cuDNN include directory in libtorch"
                NO_DEFAULT_PATH
            )

            if(CUDNN_INCLUDE_DIR)
                message(STATUS "Found cuDNN include directory in libtorch: ${CUDNN_INCLUDE_DIR}")
            endif()
        endif()

        # If not found in libtorch, check system paths
        if(NOT CUDNN_INCLUDE_DIR)
            find_path(CUDNN_INCLUDE_DIR
                NAMES cudnn.h
                PATHS /usr/include
                      /usr/local/include
                      /usr/local/cuda/include
                DOC "Path to cuDNN include directory"
                NO_DEFAULT_PATH
            )
        endif()
        
        if(CUDNN_INCLUDE_DIR)
            message(STATUS "Found cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
            # Export as environment variable for PyTorch to find
            set(ENV{CUDNN_INCLUDE_DIR} "${CUDNN_INCLUDE_DIR}")
        endif()
        
        if(CUDNN_ROOT)
            message(STATUS "Providing CUDNN_ROOT hint to PyTorch: ${CUDNN_ROOT}")
            # Add CUDNN_ROOT to the environment for PyTorch's find_package to use
            set(ENV{CUDNN_ROOT} "${CUDNN_ROOT}")
        endif()

        # Attempt to give Torch the exact CUDA version string to bypass its detection
        if(DEFINED CUDA_VERSION_STRING)
            set(TORCH_CUDA_VERSION "${CUDA_VERSION_STRING}" CACHE STRING "Pre-set CUDA version for Torch" FORCE)
            message(STATUS "Setting TORCH_CUDA_VERSION to ${TORCH_CUDA_VERSION} to aid Torch's CUDA detection.")
        endif()

        # Explicitly set CUDA architectures to avoid auto-detection failures
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # Alternative NVCC flags approach if needed
        set(TORCH_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86")
        message(STATUS "Setting explicit TORCH_NVCC_FLAGS: ${TORCH_NVCC_FLAGS}")

        # Skip Torch's GPU detection by setting the architecture list directly
        set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6" CACHE STRING "CUDA architectures for PyTorch" FORCE)
        message(STATUS "Setting TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}")

        # Create nvToolsExt target if missing (needed by some PyTorch versions)
        if(NOT TARGET CUDA::nvToolsExt)
            # Try to find it in different locations for Linux
            set(NVTOOLSEXT_PATHS
                "/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so"
                "/usr/local/cuda/lib64/libnvToolsExt.so"
                "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so"
                "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1"
            )
            
            # Try each potential path
            set(NVPERF_HOST_LIB "")
            foreach(PATH ${NVTOOLSEXT_PATHS})
                if(EXISTS "${PATH}")
                    set(NVPERF_HOST_LIB "${PATH}")
                    break()
                endif()
            endforeach()

            # Set include path if library was found
            if(NVPERF_HOST_LIB)
                message(STATUS "Using nvToolsExt replacement: ${NVPERF_HOST_LIB}")
                
                # Extract directory for include path
                get_filename_component(LIB_DIR "${NVPERF_HOST_LIB}" DIRECTORY)
                get_filename_component(CUPTI_DIR "${LIB_DIR}" DIRECTORY)
                
                # Try to find the include directory
                if(EXISTS "${CUPTI_DIR}/include")
                    set(NVPERF_HOST_INCLUDE "${CUPTI_DIR}/include")
                elseif(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include")
                    set(NVPERF_HOST_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include")
                elseif(EXISTS "/usr/local/cuda/include")
                    set(NVPERF_HOST_INCLUDE "/usr/local/cuda/include")
                endif()
                
                add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
                set_target_properties(CUDA::nvToolsExt PROPERTIES
                    IMPORTED_LOCATION "${NVPERF_HOST_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${NVPERF_HOST_INCLUDE}")
            else()
                message(WARNING "nvToolsExt or equivalent not found, creating interface-only target")
                add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            endif()
        endif()
    else()
        message(WARNING "CUDA toolkit not found – Torch will build CPU-only.")
    endif()

    # PyTorch paths for Linux
    # Try multiple common Linux locations for PyTorch
    set(TORCH_SEARCH_PATHS
        "/opt/libtorch/share/cmake/Torch"
        "/usr/local/lib/python3/dist-packages/torch/share/cmake/Torch"
    )
    
    # Add paths for specific Python versions
    foreach(PYTHON_VERSION 3.6 3.7 3.8 3.9 3.10 3.11 3.12)
        list(APPEND TORCH_SEARCH_PATHS 
            "/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/share/cmake/Torch"
            "/usr/lib/python${PYTHON_VERSION}/dist-packages/torch/share/cmake/Torch"
        )
    endforeach()
    
    # Find the first existing path
    set(DEFAULT_TORCH_DIR "")
    foreach(PATH ${TORCH_SEARCH_PATHS})
        if(EXISTS "${PATH}")
            set(DEFAULT_TORCH_DIR "${PATH}")
            break()
        endif()
    endforeach()
    
    # If not found, default to a common location
    if(DEFAULT_TORCH_DIR STREQUAL "")
        set(DEFAULT_TORCH_DIR "/usr/local/lib/python3/dist-packages/torch/share/cmake/Torch")
    endif()

    # Set explicit path to PyTorch
    set(Torch_DIR "/opt/libtorch-2.7.0-cu128/share/cmake/Torch" CACHE PATH "Directory containing TorchConfig.cmake" FORCE)
    message(STATUS "Looking for PyTorch in: ${Torch_DIR}")

    # Set option for CUDA fallback mode
    option(ENABLE_CUDA_FALLBACK "Enable fallback mode for CUDA functions" ON)
    option(DISABLE_CUDNN "Disable cuDNN usage completely" OFF)

    # Conditional CUDNN configuration
    if(DISABLE_CUDNN)
        set(USE_CUDNN 0 CACHE BOOL "Disable cuDNN usage in fallback mode" FORCE)
        set(CAFFE2_USE_CUDNN 0 CACHE BOOL "Disable cuDNN usage for Caffe2 in fallback mode" FORCE)
        add_definitions(-DUSE_CUDNN=0)
        message(STATUS "Building with cuDNN disabled")
    else()
        # Explicitly force cuDNN usage
        set(USE_CUDNN 1 CACHE BOOL "Enable cuDNN usage" FORCE)
        set(CAFFE2_USE_CUDNN 1 CACHE BOOL "Enable cuDNN usage for Caffe2" FORCE)
        add_definitions(-DUSE_CUDNN=1)
        message(STATUS "Building with cuDNN explicitly enabled")
    endif()

    # Add CUDA fallback mode definition
    if(ENABLE_CUDA_FALLBACK)
        add_definitions(-DENABLE_CUDA_FALLBACK=1)
        message(STATUS "Building with CUDA fallback mode enabled (CPU-only execution if CUDA fails)")
    else()
        add_definitions(-DENABLE_CUDA_FALLBACK=0)
        message(STATUS "Building with strict CUDA requirements (will fail if CUDA not available)")
    endif()

    # Explicitly set cuDNN paths before finding PyTorch
    set(CUDNN_LIBRARY_PATH "/opt/libtorch-2.7.0-cu128/lib/libcudnn.so.9" CACHE PATH "Path to cuDNN library")
    set(CUDNN_INCLUDE_PATH "/usr/include/x86_64-linux-gnu" CACHE PATH "Path to cuDNN headers")
    set(CUDNN_ROOT "/opt/libtorch-2.7.0-cu128" CACHE PATH "Path to cuDNN root")

    # Ensure these variables are set for Torch's internal cuDNN lookup
    set(CUDNN_LIBRARY ${CUDNN_LIBRARY_PATH})
    set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_PATH})
    set(CUDNN_INCLUDE_PATH ${CUDNN_INCLUDE_PATH})
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY_PATH})
    set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARY_PATH})
    set(CUDNN_FOUND TRUE CACHE BOOL "cuDNN found status")

    # Set environment variables for Torch
    set(ENV{CUDNN_ROOT} "/opt/libtorch-2.7.0-cu128")
    set(ENV{CUDNN_LIBRARY} "/opt/libtorch-2.7.0-cu128/lib/libcudnn.so.9")
    set(ENV{CUDNN_INCLUDE_DIR} "/usr/include/x86_64-linux-gnu")

    # Add system include directory for cuDNN headers
    include_directories(SYSTEM "/usr/include/x86_64-linux-gnu")

    # Force CMake to find cuDNN
    message(STATUS "Explicitly configured cuDNN:")
    message(STATUS "  CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    message(STATUS "  CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "  CUDNN_ROOT: ${CUDNN_ROOT}")

    find_package(Torch REQUIRED)
    message(STATUS "Found PyTorch: ${Torch_DIR}")

    file(REAL_PATH "${Torch_DIR}/../../../lib" Torch_LIB_DIR)
    message(STATUS "PyTorch library directory: ${Torch_LIB_DIR}")
endif()

# ────────────────────────────────── Python / OpenMP ────────────────────────────
set(PYBIND11_FOUND FALSE)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python interpreter: ${Python_EXECUTABLE} (${Python_VERSION})")
        message(STATUS "Found Python libraries: ${Python_LIBRARIES}")
        message(STATUS "Found Python include dirs: ${Python_INCLUDE_DIRS}")
        
        # First try to get pybind11 paths from Python
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
            set(pybind11_DIR "${pybind11_cmake_dir_from_python}" CACHE PATH "pybind11 CMake directory from Python" FORCE)
        else()
            # Try to find pybind11 in common locations
            foreach(PYTHON_VERSION 3.6 3.7 3.8 3.9 3.10 3.11 3.12)
                set(CANDIDATE_PATH "/usr/local/lib/python${PYTHON_VERSION}/dist-packages/pybind11/share/cmake/pybind11")
                if(EXISTS "${CANDIDATE_PATH}")
                    message(STATUS "Found potential pybind11 path: ${CANDIDATE_PATH}")
                    list(APPEND CMAKE_PREFIX_PATH "${CANDIDATE_PATH}")
                    set(pybind11_DIR "${CANDIDATE_PATH}" CACHE PATH "pybind11 CMake directory" FORCE)
                    break()
                endif()
                
                set(CANDIDATE_PATH "/usr/lib/python${PYTHON_VERSION}/dist-packages/pybind11/share/cmake/pybind11")
                if(EXISTS "${CANDIDATE_PATH}")
                    message(STATUS "Found potential pybind11 path: ${CANDIDATE_PATH}")
                    list(APPEND CMAKE_PREFIX_PATH "${CANDIDATE_PATH}")
                    set(pybind11_DIR "${CANDIDATE_PATH}" CACHE PATH "pybind11 CMake directory" FORCE)
                    break()
                endif()
            endforeach()
        endif()
    else()
        message(WARNING "Python interpreter not found, cannot automatically locate pip-installed pybind11's CMake files.")
    endif()

    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "Building Python bindings with pybind11 ${pybind11_VERSION}")
        set(PYBIND11_FOUND TRUE)
        
        # If essential pybind11 commands are not available after find_package, try to include pybind11.cmake manually.
        if(NOT COMMAND pybind11_add_module)
            # Try to find pybind11.cmake
            if(pybind11_DIR)
                set(POTENTIAL_CMAKE_FILE "${pybind11_DIR}/pybind11Config.cmake")
                if(EXISTS "${POTENTIAL_CMAKE_FILE}")
                    include("${POTENTIAL_CMAKE_FILE}")
                    message(STATUS "Included ${POTENTIAL_CMAKE_FILE}")
                endif()
                
                set(POTENTIAL_CMAKE_FILE "${pybind11_DIR}/pybind11.cmake")
                if(EXISTS "${POTENTIAL_CMAKE_FILE}")
                    include("${POTENTIAL_CMAKE_FILE}")
                    message(STATUS "Included ${POTENTIAL_CMAKE_FILE}")
                endif()
                
                # Look in parent directories
                get_filename_component(PYBIND11_PARENT_DIR "${pybind11_DIR}" DIRECTORY)
                set(POTENTIAL_CMAKE_FILE "${PYBIND11_PARENT_DIR}/pybind11.cmake")
                if(EXISTS "${POTENTIAL_CMAKE_FILE}")
                    include("${POTENTIAL_CMAKE_FILE}")
                    message(STATUS "Included ${POTENTIAL_CMAKE_FILE}")
                endif()
            endif()
            
            # Define a simple pybind11_add_module if it's still not available
            if(NOT COMMAND pybind11_add_module)
                message(STATUS "Defining custom pybind11_add_module")
                function(pybind11_add_module target_name)
                    add_library(${target_name} MODULE ${ARGN})
                    target_include_directories(${target_name} PRIVATE ${Python_INCLUDE_DIRS})
                    target_link_libraries(${target_name} PRIVATE ${Python_LIBRARIES})
                    
                    # Handle suffix for Linux
                    set_target_properties(${target_name} PROPERTIES PREFIX "" SUFFIX ".so")
                    
                    # No need for lib prefix
                    set_target_properties(${target_name} PROPERTIES PREFIX "")
                endfunction()
            endif()
        endif()
    else()
        message(WARNING "pybind11 not found - Python bindings will be disabled.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

if(PYBIND11_FOUND AND BUILD_PYTHON_BINDINGS AND COMMAND pybind11_add_module)
    message(STATUS "Building Python module...")
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
    
    # Add AlphaZero pipeline bindings
    message(STATUS "Building AlphaZero pipeline Python module...")
    pybind11_add_module(alphazero_pipeline src/python/alphazero_bindings.cpp)
    target_link_libraries(alphazero_pipeline PRIVATE alphazero)
    
    if(Python_FOUND)
        # Set the Python module installation path
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(PYTHON_SITE_PACKAGES)
            message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")
            install(TARGETS alphazero_py DESTINATION "${PYTHON_SITE_PACKAGES}")
            install(TARGETS alphazero_pipeline DESTINATION "${PYTHON_SITE_PACKAGES}")
        endif()
    endif()
else()
    message(STATUS "Skipping Python module creation")
endif()

find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
else()
    # Try to find nlohmann_json directly from common system paths
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS
            /usr/include
            /usr/local/include
            /usr/local/include/nlohmann
            /home/${USER}/anaconda3/include
            ${CMAKE_PREFIX_PATH}
        NO_DEFAULT_PATH
    )
    
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann_json include dir: ${NLOHMANN_JSON_INCLUDE_DIR}")
        include_directories(BEFORE SYSTEM ${NLOHMANN_JSON_INCLUDE_DIR})

        # Add system include directory directly to be safe
        include_directories(BEFORE SYSTEM "/usr/include")
        
        # Create the target manually if find_package failed
        if(NOT TARGET nlohmann_json::nlohmann_json)
            add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
            set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}")
        endif()
        
        # Check if this is a single-file version or multi-file version
        if(EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp" AND NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/adl_serializer.hpp")
            # Single-file version detected, extract individual files from json.hpp
            message(STATUS "Single-file json.hpp detected. Extracting component headers...")
            
            set(JSON_HEADERS_DIR "${CMAKE_BINARY_DIR}/include/nlohmann")
            file(MAKE_DIRECTORY "${JSON_HEADERS_DIR}")
            
            # Copy the original json.hpp
            file(COPY "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp" DESTINATION "${JSON_HEADERS_DIR}")
            
            # Create the missing component headers
            foreach(HEADER_NAME adl_serializer detail json_fwd)
                file(WRITE "${JSON_HEADERS_DIR}/${HEADER_NAME}.hpp"
                     "#pragma once\n#include \"json.hpp\"\n")
            endforeach()
            
            # Add the generated directory to include paths
            include_directories(BEFORE "${CMAKE_BINARY_DIR}/include")
            message(STATUS "Created component headers in ${JSON_HEADERS_DIR}")
        else()
            # Copy the entire nlohmann directory to the build directory if needed
            # Use CMake's file operations for better cross-platform support
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
            if(EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann")
                file(COPY "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann" DESTINATION "${CMAKE_BINARY_DIR}/include/")
                message(STATUS "Copied nlohmann directory to build directory")
                include_directories(BEFORE "${CMAKE_BINARY_DIR}/include")
            else()
                message(WARNING "nlohmann directory not found at ${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann. Build may fail.")
            endif()
        endif()
    else()
        message(FATAL_ERROR "Could not find nlohmann/json.hpp header file. Please install nlohmann_json library.")
    endif()
endif()

# FALLBACK: Use our custom single-header JSON implementation if the above fails
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/include/nlohmann/json.hpp")
    message(STATUS "Using fallback single-header JSON implementation from build/include")
    include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/build/include")
endif()

find_package(yaml-cpp CONFIG QUIET)
if(yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")
else()
    # Try to find yaml-cpp directly from common system paths
    find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_PREFIX_PATH}
    )

    find_library(YAML_CPP_LIBRARY
        NAMES yaml-cpp
        PATHS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            ${CMAKE_PREFIX_PATH}
    )

    if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
        message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
        message(STATUS "Found yaml-cpp library: ${YAML_CPP_LIBRARY}")

        # Create the target manually if find_package failed
        if(NOT TARGET yaml-cpp::yaml-cpp)
            add_library(yaml-cpp::yaml-cpp UNKNOWN IMPORTED)
            set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
                IMPORTED_LOCATION "${YAML_CPP_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
        endif()

        # Also create an unqualified target for compatibility
        if(NOT TARGET yaml-cpp)
            add_library(yaml-cpp UNKNOWN IMPORTED)
            set_target_properties(yaml-cpp PROPERTIES
                IMPORTED_LOCATION "${YAML_CPP_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR "Could not find yaml-cpp library or headers. Please install yaml-cpp.")
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    add_compile_definitions(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ─────────────────────────────────── Source sets ───────────────────────────────
set(CORE_SOURCES      src/core/igamestate.cpp src/core/game_export.cpp src/core/tensor_pool.cpp src/core/lock_free_tensor_pool.cpp)
set(CHESS_SOURCES     src/games/chess/chess_state.cpp src/games/chess/chess_rules.cpp src/games/chess/chess960.cpp)
set(GO_SOURCES        src/games/go/go_state.cpp   src/games/go/go_rules.cpp)
set(GOMOKU_SOURCES    src/games/gomoku/gomoku_state.cpp src/games/gomoku/gomoku_rules.cpp)
set(UTIL_SOURCES      src/utils/zobrist_hash.cpp src/utils/attack_defense_module.cpp src/utils/hash_specializations.cpp src/utils/debug_monitor.cpp src/utils/gamestate_pool.cpp src/utils/memory_tracker.cpp src/utils/logger.cpp src/utils/gpu_memory_manager.cpp src/utils/resource_monitor.cpp src/utils/advanced_memory_monitor.cpp)

# Conditionally define NN_SOURCES based on WITH_TORCH
set(NN_SOURCES "") # Initialize with non-Torch NN files if any, or empty
if(WITH_TORCH)
    list(APPEND NN_SOURCES
        src/nn/ddw_randwire_resnet.cpp
        src/nn/resnet_model.cpp
        src/nn/neural_network_factory.cpp
        src/nn/gpu_optimizer.cpp
    )
endif()

set(MCTS_SOURCES
    src/mcts/mcts_node.cpp
    src/mcts/mcts_node_methods.cpp
    src/mcts/mcts_node_pending_eval.cpp
    src/mcts/mcts_node_virtual_loss.cpp
    src/mcts/mcts_node_pool.cpp
    src/mcts/evaluation_types.cpp
    src/mcts/mcts_object_pool.cpp
    src/mcts/advanced_memory_pool.cpp
    src/mcts/mcts_engine_main.cpp
    src/mcts/mcts_engine_utils.cpp
    src/mcts/mcts_engine_tree.cpp
    src/mcts/mcts_engine_eval.cpp
    src/mcts/mcts_engine_common.cpp
    src/mcts/mcts_engine_search.cpp
    src/mcts/mcts_engine_simple_batch.cpp
    src/mcts/mcts_engine_methods.cpp
    src/mcts/mcts_engine_memory.cpp
    src/mcts/mcts_engine_parallel_batch.cpp
    src/mcts/mcts_engine_memory_pressure.cpp
    src/mcts/mcts_engine_true_parallel_search.cpp
    src/mcts/mcts_engine_taskflow.cpp
    src/mcts/aggressive_memory_manager.cpp
    src/mcts/transposition_table.cpp
    src/mcts/node_tracker.cpp
)
set(SELFPLAY_SOURCES
    src/selfplay/self_play_manager.cpp
)
set(TRAINING_SOURCES
    src/training/training_data_manager.cpp
    src/training/dataset.cpp
    src/training/data_loader.cpp
)
set(EVALUATION_SOURCES
)
set(CLI_SOURCES 
    src/cli/cli_manager.cpp
    src/cli/alphazero_pipeline.cpp
    src/cli/alphazero_cli_pipeline.cpp
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
    ${UTIL_SOURCES} ${NN_SOURCES} ${MCTS_SOURCES} ${CLI_SOURCES}
    ${SELFPLAY_SOURCES} ${TRAINING_SOURCES} ${EVALUATION_SOURCES}
)

# ───────────────────────────── alphazero library target ───────────────────────
add_library(alphazero ${ALL_SOURCES})

set_target_properties(alphazero PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(alphazero
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        /usr/include  # Add system include dir explicitly
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Define ALPHAZERO_EXPORTS when building the library to ensure symbols are exported
if(BUILD_SHARED_LIBS)
    target_compile_definitions(alphazero PRIVATE -DALPHAZERO_EXPORTS)
endif()

# Add CUDA synchronization definitions
target_compile_definitions(alphazero PRIVATE -DWITH_CUDA_SYNC)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero PUBLIC OpenMP::OpenMP_CXX)
endif()

# Linux compiler options
target_compile_options(alphazero PRIVATE -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-sign-compare)

# Link required libraries
target_link_libraries(alphazero PUBLIC Threads::Threads)

# Link JSON library (handle different target names)
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(alphazero PUBLIC nlohmann_json::nlohmann_json)
else()
    # Fallback in case the target has a different name
    target_include_directories(alphazero PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# Link YAML-CPP library (handle different target names)
if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(alphazero PUBLIC yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    target_link_libraries(alphazero PUBLIC yaml-cpp)
else()
    # Fallback in case the target has a different name
    target_include_directories(alphazero PUBLIC ${YAML_CPP_INCLUDE_DIR})
    if(YAML_CPP_LIBRARY)
        target_link_libraries(alphazero PUBLIC ${YAML_CPP_LIBRARY})
    endif()
endif()

if(WITH_TORCH)
    # Define WITH_TORCH compile definition for the library
    target_compile_definitions(alphazero PUBLIC -DWITH_TORCH)
    target_link_libraries(alphazero PUBLIC "${TORCH_LIBRARIES}")
    # Add CUDA libraries explicitly if needed
    if(CUDA_FOUND AND CUDA_CUDART_LIBRARY)
        target_link_libraries(alphazero PUBLIC ${CUDA_CUDART_LIBRARY})
    endif()
    
    # Link cuDNN library - check all possible paths
    set(CUDNN_LIB_CANDIDATES
        "${CUDNN_LIBRARY_PATH}"
        "${CUDNN_LIBRARY}"
        "${Torch_LIB_DIR}/libcudnn.so.9"
        "/opt/libtorch/lib/libcudnn.so.9"
        "/opt/libtorch-2.7.0-cu128/lib/libcudnn.so.9"
        "${CUDNN_LIBRARIES}"
    )

    set(CUDNN_LIB_FOUND FALSE)
    foreach(lib ${CUDNN_LIB_CANDIDATES})
        if(NOT CUDNN_LIB_FOUND AND EXISTS "${lib}")
            message(STATUS "Linking cuDNN library: ${lib}")
            target_link_libraries(alphazero PUBLIC ${lib})
            set(CUDNN_LIB_FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT CUDNN_LIB_FOUND)
        message(WARNING "No cuDNN library found in any of the candidate paths, but will continue building")
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero PUBLIC OpenMP::OpenMP_CXX)
endif()

# ───────────────────────── parallel-hashmap setup ─────────────────────────
message(STATUS "Setting up parallel-hashmap...")

# First try to find it on the system
find_package(phmap CONFIG QUIET)

if(phmap_FOUND)
    message(STATUS "Found system parallel-hashmap: ${phmap_VERSION}")
else()
    # Try to find parallel-hashmap headers directly
    find_path(PHMAP_INCLUDE_DIR parallel_hashmap/phmap.h
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_PREFIX_PATH}
    )
    
    if(PHMAP_INCLUDE_DIR)
        message(STATUS "Found parallel-hashmap headers at: ${PHMAP_INCLUDE_DIR}")
        
        # Create the interface target
        add_library(parallel_hashmap::parallel_hashmap INTERFACE IMPORTED)
        set_target_properties(parallel_hashmap::parallel_hashmap PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PHMAP_INCLUDE_DIR}")
    else()
        # Download parallel-hashmap header files
        message(STATUS "parallel-hashmap not found on system - downloading from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            phmap
            GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
            GIT_TAG v1.3.8  # Use a specific tag for stability
            GIT_SHALLOW TRUE
        )
        
        FetchContent_MakeAvailable(phmap)
        message(STATUS "Downloaded parallel-hashmap to ${phmap_SOURCE_DIR}")
        
        # Create interface target with download source
        add_library(parallel_hashmap::parallel_hashmap INTERFACE IMPORTED)
        set_target_properties(parallel_hashmap::parallel_hashmap PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${phmap_SOURCE_DIR}")
    endif()
endif()

# Add parallel_hashmap to the target link libraries
target_link_libraries(alphazero PUBLIC parallel_hashmap::parallel_hashmap)

# ───────────────────────── concurrentqueue setup ─────────────────────────
message(STATUS "Setting up concurrentqueue...")

# First try to find it on the system
find_package(concurrentqueue CONFIG QUIET)

if(concurrentqueue_FOUND)
    message(STATUS "Found system concurrentqueue: ${concurrentqueue_VERSION}")
    target_link_libraries(alphazero PUBLIC concurrentqueue::concurrentqueue)
else()
    # Try to find concurrentqueue headers directly
    find_path(CONCURRENTQUEUE_INCLUDE_DIR moodycamel/concurrentqueue.h
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_PREFIX_PATH}
    )
    
    if(CONCURRENTQUEUE_INCLUDE_DIR)
        message(STATUS "Found concurrentqueue headers at: ${CONCURRENTQUEUE_INCLUDE_DIR}")
        
        # Create the interface target
        add_library(moodycamel::concurrentqueue INTERFACE IMPORTED)
        set_target_properties(moodycamel::concurrentqueue PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CONCURRENTQUEUE_INCLUDE_DIR}")
            
        target_link_libraries(alphazero PUBLIC moodycamel::concurrentqueue)
    else()
        # Download concurrentqueue header files
        message(STATUS "concurrentqueue not found on system - downloading from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            concurrentqueue
            GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
            GIT_TAG v1.0.4  # Use a specific tag for stability
            GIT_SHALLOW TRUE
        )
        
        FetchContent_MakeAvailable(concurrentqueue)
        message(STATUS "Downloaded concurrentqueue to ${concurrentqueue_SOURCE_DIR}")
        
        # Create wrapper directory structure to match expected include path
        set(CONCURRENTQUEUE_WRAPPER_DIR "${CMAKE_BINARY_DIR}/concurrentqueue-wrapper")
        set(MOODYCAMEL_WRAPPER_DIR "${CONCURRENTQUEUE_WRAPPER_DIR}/moodycamel")
        file(MAKE_DIRECTORY "${MOODYCAMEL_WRAPPER_DIR}")
        
        # Copy headers to wrapper directory
        file(COPY "${concurrentqueue_SOURCE_DIR}/concurrentqueue.h" 
             DESTINATION "${MOODYCAMEL_WRAPPER_DIR}/")
        file(COPY "${concurrentqueue_SOURCE_DIR}/lightweightsemaphore.h" 
             DESTINATION "${MOODYCAMEL_WRAPPER_DIR}/")
        
        # Create interface target
        add_library(moodycamel::concurrentqueue INTERFACE IMPORTED)
        set_target_properties(moodycamel::concurrentqueue PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CONCURRENTQUEUE_WRAPPER_DIR}")
            
        target_link_libraries(alphazero PUBLIC moodycamel::concurrentqueue)
    endif()
endif()

# ───────────────────────────── mimalloc setup ─────────────────────────
message(STATUS "Setting up mimalloc...")

# First try to find it on the system
find_package(mimalloc CONFIG QUIET)

if(mimalloc_FOUND)
    message(STATUS "Found system mimalloc: ${mimalloc_VERSION}")
    target_link_libraries(alphazero PUBLIC mimalloc)
    target_compile_definitions(alphazero PUBLIC USE_MIMALLOC)
else()
    # Try to find mimalloc headers and library directly
    find_path(MIMALLOC_INCLUDE_DIR mimalloc.h
        PATHS
        /usr/include
        /usr/local/include
    )
    
    find_library(MIMALLOC_LIBRARY
        NAMES mimalloc
        PATHS
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib/x86_64-linux-gnu
    )
    
    if(MIMALLOC_INCLUDE_DIR AND MIMALLOC_LIBRARY)
        message(STATUS "Found mimalloc headers at: ${MIMALLOC_INCLUDE_DIR}")
        message(STATUS "Found mimalloc library at: ${MIMALLOC_LIBRARY}")
        
        # Create imported target
        add_library(mimalloc::mimalloc STATIC IMPORTED)
        set_target_properties(mimalloc::mimalloc PROPERTIES
            IMPORTED_LOCATION "${MIMALLOC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MIMALLOC_INCLUDE_DIR}")
            
        target_link_libraries(alphazero PUBLIC mimalloc::mimalloc)
        target_compile_definitions(alphazero PUBLIC USE_MIMALLOC)
    else()
        # Download mimalloc
        message(STATUS "mimalloc not found on system - downloading from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            mimalloc
            GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
            GIT_TAG        v2.1.2  # Latest stable release
            GIT_SHALLOW    TRUE
        )
        
        # Configure mimalloc build options
        set(MI_BUILD_SHARED OFF CACHE BOOL "Build shared library" FORCE)
        set(MI_BUILD_STATIC ON CACHE BOOL "Build static library" FORCE)
        set(MI_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
        set(MI_USE_CXX ON CACHE BOOL "Use C++ interface" FORCE)
        
        FetchContent_MakeAvailable(mimalloc)
        message(STATUS "Downloaded mimalloc to ${mimalloc_SOURCE_DIR}")
        
        target_link_libraries(alphazero PUBLIC mimalloc-static)
        target_compile_definitions(alphazero PUBLIC USE_MIMALLOC)
    endif()
endif()

# ───────────────────────────── Cpp-Taskflow setup ─────────────────────────
message(STATUS "Setting up Cpp-Taskflow...")

# First try to find it on the system
find_package(Taskflow CONFIG QUIET)

if(Taskflow_FOUND)
    message(STATUS "Found system Taskflow: ${Taskflow_VERSION}")
    target_link_libraries(alphazero PUBLIC Taskflow::Taskflow)
else()
    # Download Cpp-Taskflow header-only library
    message(STATUS "Taskflow not found on system - downloading from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        taskflow
        GIT_REPOSITORY https://github.com/taskflow/taskflow.git
        GIT_TAG        v3.7.0  # Latest stable release
        GIT_SHALLOW    TRUE
    )
    
    FetchContent_MakeAvailable(taskflow)
    message(STATUS "Downloaded Taskflow to ${taskflow_SOURCE_DIR}")
    
    # Taskflow is header-only, just need to add include directory
    target_include_directories(alphazero PUBLIC ${taskflow_SOURCE_DIR})
endif()

# ───────────────────────────── spdlog setup ─────────────────────────
message(STATUS "Setting up spdlog...")

# First try to find it on the system
find_package(spdlog CONFIG QUIET)

if(spdlog_FOUND)
    message(STATUS "Found system spdlog: ${spdlog_VERSION}")
    target_link_libraries(alphazero PUBLIC spdlog::spdlog)
else()
    # Download spdlog
    message(STATUS "spdlog not found on system - downloading from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1  # Latest stable release
        GIT_SHALLOW    TRUE
    )
    
    # Configure spdlog options
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Build the example" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "Install spdlog" FORCE)
    
    FetchContent_MakeAvailable(spdlog)
    message(STATUS "Downloaded spdlog to ${spdlog_SOURCE_DIR}")
    
    target_link_libraries(alphazero PUBLIC spdlog::spdlog)
endif()

# ───────────────────────────── Tracy Profiler setup ─────────────────────────
message(STATUS "Setting up Tracy Profiler...")

# Option to enable/disable Tracy profiling
option(ENABLE_TRACY "Enable Tracy profiler" ON)

if(ENABLE_TRACY)
    # Download Tracy
    message(STATUS "Tracy profiling enabled - downloading from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        v0.11.0  # Latest stable release
        GIT_SHALLOW    TRUE
    )
    
    FetchContent_MakeAvailable(tracy)
    message(STATUS "Downloaded Tracy to ${tracy_SOURCE_DIR}")
    
    # Check if TracyClient target already exists
    if(NOT TARGET TracyClient)
        # Add Tracy as a library
        add_library(TracyClient SHARED ${tracy_SOURCE_DIR}/public/TracyClient.cpp)
        target_include_directories(TracyClient PUBLIC ${tracy_SOURCE_DIR}/public ${tracy_SOURCE_DIR}/public/tracy)
        target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE)
    endif()
    
    # Link Tracy to our library
    target_link_libraries(alphazero PUBLIC TracyClient)
    target_compile_definitions(alphazero PUBLIC TRACY_ENABLE)
    target_include_directories(alphazero PUBLIC ${tracy_SOURCE_DIR}/public)
    
    # Optional: Enable GPU tracing
    if(WITH_TORCH AND CUDA_FOUND)
        target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE_GPU)
    endif()
else()
    message(STATUS "Tracy profiling disabled")
endif()

# ───────────────────────────── RAPIDS Memory Manager setup ─────────────────────────
# Enable RMM for memory management
option(ENABLE_RMM "Enable RAPIDS Memory Manager" ON)

if(ENABLE_RMM AND WITH_TORCH AND CUDA_FOUND)
    message(STATUS "Setting up RAPIDS Memory Manager (RMM)...")
    
    target_compile_definitions(alphazero PUBLIC USE_RAPIDS_RMM)
    
    # Try to find and link against system RMM library
    find_package(rmm CONFIG QUIET)
    
    if(rmm_FOUND)
        message(STATUS "Found system RMM: ${rmm_VERSION}")
        target_link_libraries(alphazero PUBLIC rmm::rmm)
    else()
        # Try to find RMM libraries directly
        find_library(RMM_LIBRARY
            NAMES rmm librmm
            PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64
            PATH_SUFFIXES lib)
            
        if(RMM_LIBRARY)
            message(STATUS "Found RMM library: ${RMM_LIBRARY}")
            target_link_libraries(alphazero PUBLIC ${RMM_LIBRARY})
        else()
            message(STATUS "RMM library not found, but continuing with USE_RAPIDS_RMM enabled")
            message(STATUS "Make sure RMM is in your LD_LIBRARY_PATH at runtime")
        endif()
        
        # Try to find RMM include directory
        find_path(RMM_INCLUDE_DIR
            NAMES rmm/device_buffer.hpp
            PATHS /usr/include /usr/local/include)
            
        if(RMM_INCLUDE_DIR)
            message(STATUS "Found RMM include directory: ${RMM_INCLUDE_DIR}")
            target_include_directories(alphazero PUBLIC ${RMM_INCLUDE_DIR})
        else()
            message(STATUS "RMM include directory not found, but continuing with USE_RAPIDS_RMM enabled")
            message(STATUS "Make sure your compiler can find RMM headers")
        endif()
    endif()
else()
    message(STATUS "RAPIDS RMM setup skipped (disabled or requires CUDA and Torch)")
endif()

# ───────────────────────────── Runtime Library Management ──────────────────────────
if(WITH_TORCH)
    # Linux: Set rpath for executables
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${Torch_LIB_DIR}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # Create a script to set LD_LIBRARY_PATH for running tests
    if(BUILD_TESTS)
        file(WRITE "${CMAKE_BINARY_DIR}/run_tests.sh" "#!/bin/bash\n")
        file(APPEND "${CMAKE_BINARY_DIR}/run_tests.sh" "export LD_LIBRARY_PATH=\"${Torch_LIB_DIR}:$LD_LIBRARY_PATH\"\n")
        file(APPEND "${CMAKE_BINARY_DIR}/run_tests.sh" "cd \"${CMAKE_BINARY_DIR}\"\n")
        file(APPEND "${CMAKE_BINARY_DIR}/run_tests.sh" "ctest \"$@\"\n")
        # Use CMake command to set permissions to avoid shell command issues
        file(CHMOD "${CMAKE_BINARY_DIR}/run_tests.sh"
             PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
        message(STATUS "Created test runner script: ${CMAKE_BINARY_DIR}/run_tests.sh")
    endif()
endif()

# ───────────────────────────── CLI executable ────────────────────────────────
# Main CLI executable conditionally built based on WITH_TORCH availability
if(WITH_TORCH)
    # Always build the CLI with TORCH
    add_executable(omoknuni_cli src/cli/omoknuni_cli.cpp)

    # Set RPATH to find shared libraries relative to executable location
    set_target_properties(omoknuni_cli PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib/Release:$ORIGIN/../lib:$ORIGIN/../../lib/Release:$ORIGIN/../../lib"
    )

    # Link against alphazero, yaml-cpp, and Torch
    target_link_libraries(omoknuni_cli PRIVATE
        alphazero
        "${TORCH_LIBRARIES}"
        dl  # For dlopen, dlclose, dlerror
        stdc++  # Explicitly link standard C++ library
    )

    # Prioritize system GCC library path to avoid Anaconda's libstdc++
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Assuming g++ version is 11.x based on linker logs
        # A more robust way would be to parse CMAKE_CXX_COMPILER_VERSION
        set(SYSTEM_GCC_MAJOR_VERSION 11) # Adjust if your /usr/bin/g++ is different
        set(SYSTEM_GCC_LIB_DIR "/usr/lib/gcc/x86_64-linux-gnu/${SYSTEM_GCC_MAJOR_VERSION}")
        if(EXISTS "${SYSTEM_GCC_LIB_DIR}")
            target_link_directories(omoknuni_cli BEFORE PRIVATE "${SYSTEM_GCC_LIB_DIR}")
            message(STATUS "Prepending system GCC lib path for omoknuni_cli: ${SYSTEM_GCC_LIB_DIR}")
        else()
            message(WARNING "System GCC lib path NOT found: ${SYSTEM_GCC_LIB_DIR}. Linker might still pick up incorrect libstdc++.")
        endif()
        # Also prepend the general system library path
        if(EXISTS "/usr/lib/x86_64-linux-gnu")
             target_link_directories(omoknuni_cli BEFORE PRIVATE "/usr/lib/x86_64-linux-gnu")
             message(STATUS "Prepending system library path for omoknuni_cli: /usr/lib/x86_64-linux-gnu")
        endif()
    endif()

    # Handle yaml-cpp linking - different distributions may have different target names
    if(TARGET yaml-cpp::yaml-cpp)
        target_link_libraries(omoknuni_cli PRIVATE yaml-cpp::yaml-cpp)
    elseif(TARGET yaml-cpp)
        target_link_libraries(omoknuni_cli PRIVATE yaml-cpp)
    elseif(YAML_CPP_LIBRARY)
        target_link_libraries(omoknuni_cli PRIVATE ${YAML_CPP_LIBRARY})
    else()
        message(WARNING "yaml-cpp target not found. Trying with -lyaml-cpp")
        target_link_libraries(omoknuni_cli PRIVATE -lyaml-cpp)
    endif()

    # Add compiler and linker flags to properly enable CUDA
    target_compile_definitions(omoknuni_cli PRIVATE
      PYTORCH_NO_CUDA_INIT_OVERRIDE=0
      USE_TORCH=1
      C10_CUDA_DRIVER_INIT=1
      USE_CUDNN=1
      CAFFE2_USE_CUDNN=1
      TORCH_USE_CUDA=1
      CUDA_VERSION=12080
      CUDNN_MAJOR=9
    )

    # Set RPATH for the CLI executable
    # Get the LibTorch lib directory
    file(REAL_PATH "${Torch_DIR}/../../../lib" Local_Torch_LIB_DIR)

    # Add the $ORIGIN based RPATH so the executable can find libraries
    # CMAKE_BUILD_TYPE will be e.g. "Release", "Debug"
    set(CLI_RPATH "$ORIGIN/../lib/${CMAKE_BUILD_TYPE}:${Local_Torch_LIB_DIR}")

    set_target_properties(omoknuni_cli PROPERTIES
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH TRUE # Uses INSTALL_RPATH as RPATH for build tree
        INSTALL_RPATH "${CLI_RPATH}" # RPATH for installation and build tree due to BUILD_WITH_INSTALL_RPATH
    )

    # Add debug output - prints a message when run
    target_compile_definitions(omoknuni_cli PRIVATE DEBUG_CLI=1)

    # Make sure WITH_TORCH is defined
    target_compile_definitions(omoknuni_cli PRIVATE WITH_TORCH)

    # Add include directories
    target_include_directories(omoknuni_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Install the CLI executable
    install(TARGETS omoknuni_cli DESTINATION ${CMAKE_INSTALL_BINDIR})

    message(STATUS "Building omoknuni_cli with PyTorch support")
else()
    message(STATUS "Skipping omoknuni_cli (requires WITH_TORCH=ON)")
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
    set(MCTS_TEST_SOURCES   tests/mcts/mcts_node_test.cpp
                            tests/mcts/mcts_engine_test.cpp tests/mcts/mcts_openmp_test.cpp
                            tests/mcts/progressive_widening_test.cpp
                            tests/mcts/virtual_loss_test.cpp)
    set(TRANSPOSITION_TEST_SOURCES tests/mcts/transposition_table_test.cpp
                                 tests/mcts/transposition_integration_test.cpp)
    set(CLI_TEST_SOURCES    tests/cli/cli_manager_test.cpp)
    set(NN_TEST_SOURCES     tests/nn/neural_network_test.cpp)
    set(SELFPLAY_TEST_SOURCES tests/selfplay/self_play_manager_test.cpp)
    set(TRAINING_TEST_SOURCES tests/training/training_data_manager_test.cpp)
    set(INTEGRATION_TEST_SOURCES tests/integration/mcts_with_nn_test.cpp
                               tests/integration/games/self_play_games_test.cpp)
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

        # Set RPATH to find shared libraries relative to executable location  
        set_target_properties(${name} PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN/../lib/Release:$ORIGIN/../lib:$ORIGIN/../../lib/Release:$ORIGIN/../../lib"
        )

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

        # Set RPATH for Linux builds to find libalphazero.so
        # Generate RPATH based on configuration
        set(TARGET_RPATH "${CMAKE_BINARY_DIR}/lib")

        # Add configuration-specific path if available
        if(CMAKE_CONFIGURATION_TYPES)
            set(TARGET_RPATH "${TARGET_RPATH}/$<CONFIG>")
        elseif(CMAKE_BUILD_TYPE)
            set(TARGET_RPATH "${TARGET_RPATH}/${CMAKE_BUILD_TYPE}")
        endif()

        # Add standard install lib path and Torch lib path if available
        list(APPEND TARGET_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        if(WITH_TORCH AND Torch_LIB_DIR)
            list(APPEND TARGET_RPATH "${Torch_LIB_DIR}")
        endif()

        # Convert the list to a semicolon-separated string (CMake separator for RPATH)
        string(REPLACE ";" ":" RPATH_STRING "${TARGET_RPATH}")

        set_target_properties(${name} PROPERTIES
            INSTALL_RPATH "${RPATH_STRING}"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )

        add_test(NAME ${name} COMMAND ${name})

        # Set up environment variables for Linux tests
        # For Linux, set LD_LIBRARY_PATH for shared libraries
        if(GTEST_IS_SHARED OR WITH_TORCH)
            # Default paths
            set(PROJECT_LIB_PATH "${CMAKE_BINARY_DIR}/lib")

            # Add configuration-specific path if using multi-config generator
            if(CMAKE_CONFIGURATION_TYPES)
                set(PROJECT_LIB_PATH "${PROJECT_LIB_PATH}/$<CONFIG>")
            elseif(CMAKE_BUILD_TYPE)
                set(PROJECT_LIB_PATH "${PROJECT_LIB_PATH}/${CMAKE_BUILD_TYPE}")
            endif()

            # Combine all paths
            set(TEST_LD_LIBRARY_PATH "${PROJECT_LIB_PATH}")

            # Add Torch library path if with Torch
            if(WITH_TORCH AND Torch_LIB_DIR)
                set(TEST_LD_LIBRARY_PATH "${TEST_LD_LIBRARY_PATH}:${Torch_LIB_DIR}")
            endif()

            # Add current LD_LIBRARY_PATH if exists
            if(DEFINED ENV{LD_LIBRARY_PATH})
                set(TEST_LD_LIBRARY_PATH "${TEST_LD_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}")
            endif()

            set_tests_properties(${name} PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=${TEST_LD_LIBRARY_PATH}")
        endif()
    endfunction()

    make_test(core_tests    tests/core_tests_main.cpp ${CORE_TEST_SOURCES})
    make_test(chess_tests   ${CHESS_TEST_SOURCES})
    make_test(go_tests      ${GO_TEST_SOURCES})
    make_test(gomoku_tests  ${GOMOKU_TEST_SOURCES})
    if(MCTS_TEST_SOURCES)
        make_test(mcts_tests ${MCTS_TEST_SOURCES})
    endif()
    make_test(transposition_tests ${TRANSPOSITION_TEST_SOURCES})
    make_test(nn_tests      ${NN_TEST_SOURCES})
    make_test(selfplay_tests ${SELFPLAY_TEST_SOURCES})
    make_test(training_tests ${TRAINING_TEST_SOURCES})
    make_test(integration_tests ${INTEGRATION_TEST_SOURCES})
    make_test(all_tests     tests/all_tests_main.cpp
                             ${CORE_TEST_SOURCES} ${CHESS_TEST_SOURCES}
                             ${GO_TEST_SOURCES} ${GOMOKU_TEST_SOURCES}
                             ${MCTS_TEST_SOURCES} ${TRANSPOSITION_TEST_SOURCES}
                             ${CLI_TEST_SOURCES} ${NN_TEST_SOURCES}
                             ${SELFPLAY_TEST_SOURCES} ${TRAINING_TEST_SOURCES}
                             ${INTEGRATION_TEST_SOURCES})
    
    # Add benchmark executable (not a test, but a performance tool)
    if(WITH_TORCH)
        add_executable(mcts_openmp_benchmark tests/mcts/mcts_openmp_benchmark.cpp)
        target_link_libraries(mcts_openmp_benchmark PRIVATE alphazero pthread)
        if(WITH_CUDA)
            target_compile_definitions(mcts_openmp_benchmark PRIVATE WITH_CUDA)
        endif()
        # Set rpath for benchmark
        set_target_properties(mcts_openmp_benchmark PROPERTIES
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${Torch_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
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