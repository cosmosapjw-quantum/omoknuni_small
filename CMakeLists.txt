# Force use of system gcc compiler (must be before project declaration)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# cmake_minimum_required(VERSION 3.20)          # 3.14 → 3.20 for generator-expr fixes
cmake_minimum_required(VERSION 3.20)
# Only set CMP0146 policy if CMake version supports it
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)             # keep your chosen policy
endif()

# Set CUDA architectures early to influence compiler detection for CUDA 12.8+
# This helps avoid nvcc trying to compile for deprecated default architectures like sm_52 during its ID check.
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures for compiler detection and build")
message(STATUS "Early setting CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES} to aid CUDA compiler ID.")

# ─────────────────────────────── vcpkg integration ─────────────────────────
# Windows vcpkg path
if(WIN32 AND EXISTS "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain (Windows): ${CMAKE_TOOLCHAIN_FILE}")

    list(APPEND CMAKE_PREFIX_PATH  "$ENV{USERPROFILE}/vcpkg/installed/x64-windows")
    list(APPEND CMAKE_LIBRARY_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/lib")
    list(APPEND CMAKE_INCLUDE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/include")
# Linux vcpkg path
elseif(UNIX AND NOT APPLE AND EXISTS "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain (Linux): ${CMAKE_TOOLCHAIN_FILE}")

    list(APPEND CMAKE_PREFIX_PATH  "$ENV{HOME}/vcpkg/installed/x64-linux")
    list(APPEND CMAKE_LIBRARY_PATH "$ENV{HOME}/vcpkg/installed/x64-linux/lib")
    list(APPEND CMAKE_INCLUDE_PATH "$ENV{HOME}/vcpkg/installed/x64-linux/include")
# Allow user to manually specify vcpkg path
elseif(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain from VCPKG_ROOT: ${CMAKE_TOOLCHAIN_FILE}")

    if(WIN32)
        list(APPEND CMAKE_PREFIX_PATH  "$ENV{VCPKG_ROOT}/installed/x64-windows")
        list(APPEND CMAKE_LIBRARY_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows/lib")
        list(APPEND CMAKE_INCLUDE_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows/include")
    elseif(UNIX AND NOT APPLE)
        list(APPEND CMAKE_PREFIX_PATH  "$ENV{VCPKG_ROOT}/installed/x64-linux")
        list(APPEND CMAKE_LIBRARY_PATH "$ENV{VCPKG_ROOT}/installed/x64-linux/lib")
        list(APPEND CMAKE_INCLUDE_PATH "$ENV{VCPKG_ROOT}/installed/x64-linux/include")
    endif()
endif()

project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)

# ────────────────────────────────────── Threads ──────────────────────────
find_package(Threads QUIET)
if(Threads_FOUND)
    if(WIN32)
        message(STATUS "Found Threads (pthreads via vcpkg): ${CMAKE_THREAD_LIBS_INIT}")
    else()
        message(STATUS "Found Threads: ${CMAKE_THREAD_LIBS_INIT}")
    endif()
else()
    if(WIN32)
        message(WARNING "pthreads:x64-windows not found – threading disabled.")
    else()
        message(WARNING "Threads not found – threading disabled.")
    endif()
endif()

# ────────────────────────────────── Standard / warnings ────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly use the new C++11 ABI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Use the old C++11 ABI to maintain compatibility with libstdc++
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

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
    
    # Set additional linker flags to fix ABI compatibility issues
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++") // Commented out as omoknuni_cli links stdc++ specifically
endif()

# ───────────────────────────────────── Torch / CUDA ────────────────────────────
if(WITH_TORCH)
    # --- Find CUDA first to detect available GPUs ---
    # Look for CUDA in standard paths before attempting PyTorch's detection
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

        # --- cuDNN hints (platform-specific) ----------------------------
        # User needs to ensure CUDNN_ROOT points to a cuDNN version compatible with the detected CUDA_VERSION
        if(WIN32)
            set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.7" CACHE PATH "cuDNN root directory (MUST match CUDA version used by Torch)")
        elseif(UNIX AND NOT APPLE)
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

            # If not found in libtorch, check for apt-installed cuDNN (commonly in /usr/lib/x86_64-linux-gnu/)
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
            # Try to find it in different locations
            if(WIN32)
                set(NVTOOLSEXT_PATHS
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/extras/CUPTI/lib64/nvperf_host.lib"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/extras/CUPTI/lib64/nvperf_host.lib"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/extras/CUPTI/lib64/nvperf_host.lib"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/extras/CUPTI/lib64/nvperf_host.lib"
                )
            elseif(UNIX AND NOT APPLE)
                set(NVTOOLSEXT_PATHS
                    "/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so"
                    "/usr/local/cuda/lib64/libnvToolsExt.so"
                    "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so"
                    "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1"
                )
            endif()
            
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

    # Platform-specific PyTorch paths
    if(WIN32)
        set(DEFAULT_TORCH_DIR "C:/libtorch/share/cmake/Torch")
    elseif(UNIX AND NOT APPLE)
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
    message(STATUS "NOTE: Python bindings are currently troublesome due to pybind11 compatibility issues.")
    message(STATUS "      If they fail to build, consider disabling BUILD_PYTHON_BINDINGS option.")
    
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
                    
                    # Handle suffix based on platform
                    if(WIN32)
                        set_target_properties(${target_name} PROPERTIES PREFIX "" SUFFIX ".pyd")
                    else()
                        set_target_properties(${target_name} PROPERTIES PREFIX "" SUFFIX ".so")
                    endif()
                    
                    # No need for lib prefix on most platforms
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
            # Use CMake's file operations instead of bash commands for better cross-platform support
            if(UNIX AND NOT APPLE)
                file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
                if(EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann")
                    file(COPY "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann" DESTINATION "${CMAKE_BINARY_DIR}/include/")
                    message(STATUS "Copied nlohmann directory to build directory")
                    include_directories(BEFORE "${CMAKE_BINARY_DIR}/include")
                else()
                    message(WARNING "nlohmann directory not found at ${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann. Build may fail.")
                endif()
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
set(CORE_SOURCES      src/core/igamestate.cpp src/core/game_export.cpp)
set(CHESS_SOURCES     src/games/chess/chess_state.cpp src/games/chess/chess_rules.cpp src/games/chess/chess960.cpp)
set(GO_SOURCES        src/games/go/go_state.cpp   src/games/go/go_rules.cpp)
set(GOMOKU_SOURCES    src/games/gomoku/gomoku_state.cpp src/games/gomoku/gomoku_rules.cpp)
set(UTIL_SOURCES      src/utils/zobrist_hash.cpp src/utils/attack_defense_module.cpp src/utils/hash_specializations.cpp src/utils/debug_monitor.cpp)

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
set(CLI_SOURCES src/cli/cli_manager.cpp)

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

if(MSVC)
    # Always use multithreaded DLL runtime (/MD) for both library and tests for consistency
    # This avoids linkage errors when mixing runtime libraries
    target_compile_options(alphazero PRIVATE "/MD$<$<CONFIG:Debug>:d>")
endif()

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

if(MSVC)
    target_compile_options(alphazero PRIVATE /W4 /WX-)
else()
    target_compile_options(alphazero PRIVATE -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-sign-compare)
endif()

# ───────────────────────────── Runtime Library Copy Helper ──────────────────────────
if(WITH_TORCH)
    if(WIN32)
        # Windows: Copy DLLs
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
    elseif(UNIX AND NOT APPLE)
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
endif()

# ───────────────────────────── CLI executable ────────────────────────────────
# Main CLI executable conditionally built based on WITH_TORCH availability
if(WITH_TORCH)
    # Always build the CLI with TORCH
        add_executable(omoknuni_cli src/cli/omoknuni_cli.cpp)

        # Link against alphazero, yaml-cpp, and Torch
        target_link_libraries(omoknuni_cli PRIVATE
            alphazero
            "${TORCH_LIBRARIES}"
            dl  # For dlopen, dlclose, dlerror
            stdc++  # Explicitly link standard C++ library
        )

        # Prioritize system GCC library path to avoid Anaconda's libstdc++
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND UNIX AND NOT APPLE)
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
        if(UNIX AND NOT APPLE)
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
        endif()

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
    set(MCTS_TEST_SOURCES   tests/mcts/mcts_node_test.cpp  tests/mcts/mcts_evaluator_test.cpp
                            tests/mcts/mcts_engine_test.cpp)
    set(CLI_TEST_SOURCES tests/cli/cli_manager_test.cpp)
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

        # Set RPATH for Linux builds to find libalphazero.so
        if(UNIX AND NOT APPLE)
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
        endif()

        add_test(NAME ${name} COMMAND ${name})

        # Set up platform-specific environment variables for tests
        if(WIN32 AND GTEST_IS_SHARED)
            # For Windows shared library builds, set the PATH environment variable
            # to include the vcpkg directory with the GTest DLLs and the test executable's directory

            # Set up vcpkg bin paths with fallbacks
            set(VCPKG_BIN_PATH "")
            if(EXISTS "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/bin")
                file(TO_NATIVE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/bin" VCPKG_BIN_NATIVE_PATH)
            elseif(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
                file(TO_NATIVE_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows/bin" VCPKG_BIN_NATIVE_PATH)
            else()
                # Set a default that might not exist
                file(TO_NATIVE_PATH "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/bin" VCPKG_BIN_NATIVE_PATH)
                message(WARNING "VCPKG bin directory not found. PATH environment may be incomplete.")
            endif()

            file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/bin" PROJECT_BIN_DIR_NATIVE_PART)
            # $<CONFIG> will be evaluated by CTest at test time.
            set(CTEST_CONFIG_SPECIFIC_BIN_PATH "${PROJECT_BIN_DIR_NATIVE_PART}/$<CONFIG>")

            set_tests_properties(${name} PROPERTIES
                ENVIRONMENT "PATH=${VCPKG_BIN_NATIVE_PATH};${CTEST_CONFIG_SPECIFIC_BIN_PATH};$ENV{PATH}")
        elseif(UNIX AND NOT APPLE)
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

                # Handle VCPKG paths
                set(VCPKG_LIB_PATHS "")
                if(EXISTS "$ENV{HOME}/vcpkg/installed/x64-linux/lib")
                    list(APPEND VCPKG_LIB_PATHS "$ENV{HOME}/vcpkg/installed/x64-linux/lib")
                endif()
                if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-linux/lib")
                    list(APPEND VCPKG_LIB_PATHS "$ENV{VCPKG_ROOT}/installed/x64-linux/lib")
                endif()

                # Combine all paths
                set(TEST_LD_LIBRARY_PATH "${PROJECT_LIB_PATH}")
                foreach(vcpkg_path ${VCPKG_LIB_PATHS})
                    set(TEST_LD_LIBRARY_PATH "${TEST_LD_LIBRARY_PATH}:${vcpkg_path}")
                endforeach()

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
                             ${MCTS_TEST_SOURCES} ${CLI_TEST_SOURCES})
    
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
