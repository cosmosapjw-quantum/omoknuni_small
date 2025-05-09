cmake_minimum_required(VERSION 3.20)          # 3.14 → 3.20 for generator-expr fixes
cmake_policy(SET CMP0146 OLD)                 # keep your chosen policy

# ─────────────────────────────────── vcpkg integration ─────────────────────────
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
    add_compile_options(/MP /W4 /WX- /arch:AVX2 /permissive-)
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    add_compile_options(-Wall -Wextra)
    if(NOT APPLE)
        add_compile_options(-mavx2)
    endif()
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

# ───────────────────────────────────── Options ─────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build Python bindings"   OFF)
option(BUILD_TESTS            "Build tests"            ON)
option(WITH_TORCH             "Build with PyTorch"     ON)
option(BUILD_SHARED_LIBS      "Build shared libs"      ON)
option(BUILD_EXAMPLES         "Build examples"         OFF)

include(GNUInstallDirs)

# ───────────────────────────────────── Torch / CUDA ────────────────────────────
if(WITH_TORCH)
    # --- cuDNN hints (edit if your layout differs) ----------------------------
    set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.7" CACHE PATH "cuDNN root")
    set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include/12.8")
    set(CUDNN_LIB_DIR     "${CUDNN_ROOT}/lib/12.8/x64")

    set(ENV{CUDNN_LIBRARY}      "${CUDNN_LIB_DIR}/cudnn.lib")
    set(ENV{CUDNN_INCLUDE_DIR}  "${CUDNN_INCLUDE_DIR}")
    set(ENV{CUDNN_ROOT_DIR}     "${CUDNN_ROOT}")

    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA toolkit ${CUDA_VERSION_STRING} found.")
        add_compile_definitions(TORCH_USE_CUDA=1)
        
        # Use nvperf_host.lib instead of nvToolsExt
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

    find_library(CUDNN_LIBRARY NAMES cudnn PATHS "${CUDNN_LIB_DIR}" NO_DEFAULT_PATH)
    if(CUDNN_LIBRARY)
        add_compile_definitions(USE_CUDNN=1)
    endif()

    # --- Torch discovery ------------------------------------------------------
    set(Torch_DIR "C:/libtorch/share/cmake/Torch" CACHE PATH "Torch_DIR")
    find_package(Torch REQUIRED)

    # record lib path for later copy
    file(REAL_PATH "${Torch_DIR}/../../../lib" Torch_LIB_DIR)
endif()

# ────────────────────────────────── Python / OpenMP ────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    add_compile_definitions(BUILD_PYTHON_BINDINGS)
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
set(UTIL_SOURCES      src/utils/zobrist_hash.cpp src/utils/attack_defense_module.cpp src/utils/hash_specializations.cpp)
set(NN_SOURCES        "")           # filled only if WITH_TORCH
set(MCTS_SOURCES      "")           # filled conditionally

if(WITH_TORCH)
    list(APPEND NN_SOURCES src/nn/ddw_randwire_resnet.cpp)
endif()

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
    ${UTIL_SOURCES} ${NN_SOURCES}   ${MCTS_SOURCES})

# ───────────────────────────── alphazero library target ───────────────────────
add_library(alphazero ${ALL_SOURCES})
if(BUILD_SHARED_LIBS)
    set_target_properties(alphazero PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(alphazero PRIVATE ALPHAZERO_EXPORTS)
endif()

# Add include directories
target_include_directories(alphazero PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WITH_TORCH)
    target_link_libraries(alphazero PUBLIC "${TORCH_LIBRARIES}")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero PUBLIC OpenMP::OpenMP_CXX)
endif()

# ───────────────────────────── Torch DLL copy helper ──────────────────────────
if(WIN32 AND WITH_TORCH)
    # Create the destination directories
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/MinSizeRel")
    
    # Copy at configuration time
    message(STATUS "Copying PyTorch runtime DLLs to output directories")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/Debug")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/Release")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
    file(COPY "${Torch_LIB_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/bin/MinSizeRel")
    
endif()

# ─────────────────────────── Python module (optional) ─────────────────────────
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
endif()

# ───────────────────────────── Example executables ────────────────────────────
if(BUILD_EXAMPLES)
    foreach(game IN ITEMS chess go gomoku)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${game}_self_play.cpp")
            add_executable(${game}_self_play "examples/${game}_self_play.cpp")
            target_link_libraries(${game}_self_play alphazero)
        endif()
    endforeach()
endif()

# ───────────────────────────────────── Tests ──────────────────────────────────
if(BUILD_TESTS)
    include(CTest)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # ── test source groups ──
    set(CORE_TEST_SOURCES   tests/core/igamestate_test.cpp  tests/core/game_export_test.cpp)
    set(CHESS_TEST_SOURCES  tests/games/chess/chess_test.cpp)
    set(GO_TEST_SOURCES     tests/games/go/go_test.cpp)
    set(GOMOKU_TEST_SOURCES tests/games/gomoku/gomoku_test.cpp)
    set(MCTS_TEST_SOURCES   "")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mcts/mcts_test.cpp")
        list(APPEND MCTS_TEST_SOURCES tests/mcts/mcts_test.cpp)
    endif()

    function(make_test name)
        add_executable(${name} ${ARGN})
        if (${name} STREQUAL "all_tests" OR ${name} STREQUAL "core_tests")
            target_link_libraries(${name} PRIVATE alphazero gtest)
            target_compile_definitions(${name} PRIVATE CUSTOM_MAIN_USED)
        else()
            target_link_libraries(${name} PRIVATE alphazero gtest gtest_main)
        endif()
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    make_test(core_tests    tests/core_tests_main.cpp ${CORE_TEST_SOURCES})
    make_test(chess_tests   ${CHESS_TEST_SOURCES})
    make_test(go_tests      ${GO_TEST_SOURCES})
    make_test(gomoku_tests  ${GOMOKU_TEST_SOURCES})
    if(MCTS_TEST_SOURCES)
        make_test(mcts_tests ${MCTS_TEST_SOURCES})
    endif()
    make_test(all_tests     tests/all_tests_main.cpp
                             ${CORE_TEST_SOURCES} ${CHESS_TEST_SOURCES}
                             ${GO_TEST_SOURCES} ${GOMOKU_TEST_SOURCES}
                             ${MCTS_TEST_SOURCES})
endif()

# ───────────────────────────────── Installation ───────────────────────────────
install(TARGETS alphazero
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION include)
