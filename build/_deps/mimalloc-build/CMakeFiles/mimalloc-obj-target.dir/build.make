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

# Utility rule file for mimalloc-obj-target.

# Include any custom commands dependencies for this target.
include _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/progress.make

_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target: _deps/mimalloc-build/mimalloc.o

_deps/mimalloc-build/mimalloc.o:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/cosmos/omoknuni_small/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating mimalloc.o"
	cd /home/cosmos/omoknuni_small/build/_deps/mimalloc-build && /usr/bin/cmake -E copy /home/cosmos/omoknuni_small/build/_deps/mimalloc-build/CMakeFiles/mimalloc-obj.dir/src/static.c.o /home/cosmos/omoknuni_small/build/_deps/mimalloc-build/mimalloc.o

_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/codegen:
.PHONY : _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/codegen

mimalloc-obj-target: _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target
mimalloc-obj-target: _deps/mimalloc-build/mimalloc.o
mimalloc-obj-target: _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/build.make
.PHONY : mimalloc-obj-target

# Rule to build all files generated by this target.
_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/build: mimalloc-obj-target
.PHONY : _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/build

_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/clean:
	cd /home/cosmos/omoknuni_small/build/_deps/mimalloc-build && $(CMAKE_COMMAND) -P CMakeFiles/mimalloc-obj-target.dir/cmake_clean.cmake
.PHONY : _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/clean

_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/depend:
	cd /home/cosmos/omoknuni_small/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cosmos/omoknuni_small /home/cosmos/omoknuni_small/build/_deps/mimalloc-src /home/cosmos/omoknuni_small/build /home/cosmos/omoknuni_small/build/_deps/mimalloc-build /home/cosmos/omoknuni_small/build/_deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/mimalloc-build/CMakeFiles/mimalloc-obj-target.dir/depend

