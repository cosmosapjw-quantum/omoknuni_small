# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-src")
  file(MAKE_DIRECTORY "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-src")
endif()
file(MAKE_DIRECTORY
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-build"
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix"
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/tmp"
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/src/concurrentqueue-populate-stamp"
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/src"
  "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/src/concurrentqueue-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/src/concurrentqueue-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/cosmos/omoknuni_small/build/_deps/concurrentqueue-subbuild/concurrentqueue-populate-prefix/src/concurrentqueue-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
