# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cosmos/omoknuni_small

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cosmos/omoknuni_small/build

# Include any dependencies generated for this target.
include CMakeFiles/omoknuni_cli_final.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/omoknuni_cli_final.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/omoknuni_cli_final.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/omoknuni_cli_final.dir/flags.make

CMakeFiles/omoknuni_cli_final.dir/codegen:
.PHONY : CMakeFiles/omoknuni_cli_final.dir/codegen

CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o: CMakeFiles/omoknuni_cli_final.dir/flags.make
CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o: /home/cosmos/omoknuni_small/src/cli/omoknuni_cli_final.cpp
CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o: CMakeFiles/omoknuni_cli_final.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/cosmos/omoknuni_small/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -O3 -march=native -fopenmp -MD -MT CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o -MF CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o.d -o CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o -c /home/cosmos/omoknuni_small/src/cli/omoknuni_cli_final.cpp

CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -O3 -march=native -fopenmp -E /home/cosmos/omoknuni_small/src/cli/omoknuni_cli_final.cpp > CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.i

CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -O3 -march=native -fopenmp -S /home/cosmos/omoknuni_small/src/cli/omoknuni_cli_final.cpp -o CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.s

# Object files for target omoknuni_cli_final
omoknuni_cli_final_OBJECTS = \
"CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o"

# External object files for target omoknuni_cli_final
omoknuni_cli_final_EXTERNAL_OBJECTS =

bin/Release/omoknuni_cli_final: CMakeFiles/omoknuni_cli_final.dir/src/cli/omoknuni_cli_final.cpp.o
bin/Release/omoknuni_cli_final: CMakeFiles/omoknuni_cli_final.dir/build.make
bin/Release/omoknuni_cli_final: CMakeFiles/omoknuni_cli_final.dir/compiler_depend.ts
bin/Release/omoknuni_cli_final: lib/Release/libalphazero.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libtorch.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libc10.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libkineto.a
bin/Release/omoknuni_cli_final: /usr/local/cuda/lib64/libnvrtc.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libc10_cuda.so
bin/Release/omoknuni_cli_final: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.8.0
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libc10_cuda.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libc10.so
bin/Release/omoknuni_cli_final: /usr/local/cuda/lib64/libnvToolsExt.so
bin/Release/omoknuni_cli_final: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
bin/Release/omoknuni_cli_final: /usr/local/cuda/lib64/libcudart.so
bin/Release/omoknuni_cli_final: /opt/libtorch-2.7.0-cu128/lib/libcudnn.so.9
bin/Release/omoknuni_cli_final: lib/Release/libmimalloc.a
bin/Release/omoknuni_cli_final: /usr/lib/x86_64-linux-gnu/libpthread.a
bin/Release/omoknuni_cli_final: /usr/lib/x86_64-linux-gnu/librt.a
bin/Release/omoknuni_cli_final: /usr/lib/x86_64-linux-gnu/libspdlog.so.1.12.0
bin/Release/omoknuni_cli_final: /usr/lib/x86_64-linux-gnu/libfmt.so.9.1.0
bin/Release/omoknuni_cli_final: lib/Release/libTracyClient.so.0.11.0
bin/Release/omoknuni_cli_final: /usr/local/cuda/lib64/libcudart.so
bin/Release/omoknuni_cli_final: /usr/local/lib/librapids_logger.so
bin/Release/omoknuni_cli_final: CMakeFiles/omoknuni_cli_final.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/cosmos/omoknuni_small/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/Release/omoknuni_cli_final"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/omoknuni_cli_final.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/omoknuni_cli_final.dir/build: bin/Release/omoknuni_cli_final
.PHONY : CMakeFiles/omoknuni_cli_final.dir/build

CMakeFiles/omoknuni_cli_final.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/omoknuni_cli_final.dir/cmake_clean.cmake
.PHONY : CMakeFiles/omoknuni_cli_final.dir/clean

CMakeFiles/omoknuni_cli_final.dir/depend:
	cd /home/cosmos/omoknuni_small/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cosmos/omoknuni_small /home/cosmos/omoknuni_small /home/cosmos/omoknuni_small/build /home/cosmos/omoknuni_small/build /home/cosmos/omoknuni_small/build/CMakeFiles/omoknuni_cli_final.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/omoknuni_cli_final.dir/depend

