# CMake generated Testfile for 
# Source directory: /home/cosmos/omoknuni_small
# Build directory: /home/cosmos/omoknuni_small/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[core_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/core_tests")
set_tests_properties([=[core_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1401;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[chess_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/chess_tests")
set_tests_properties([=[chess_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1402;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[go_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/go_tests")
set_tests_properties([=[go_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1403;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[gomoku_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/gomoku_tests")
set_tests_properties([=[gomoku_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1404;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[mcts_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/mcts_tests")
set_tests_properties([=[mcts_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1406;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[transposition_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/transposition_tests")
set_tests_properties([=[transposition_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1408;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[nn_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/nn_tests")
set_tests_properties([=[nn_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1409;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[selfplay_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/selfplay_tests")
set_tests_properties([=[selfplay_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1410;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[training_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/training_tests")
set_tests_properties([=[training_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1411;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[integration_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/integration_tests")
set_tests_properties([=[integration_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1412;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
add_test([=[all_tests]=] "/home/cosmos/omoknuni_small/build/bin/Release/all_tests")
set_tests_properties([=[all_tests]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/home/cosmos/omoknuni_small/build/lib/Release:/opt/libtorch-2.7.0-cu128/lib:/opt/libtorch/lib:/usr/local/cuda-12.8/lib64:" _BACKTRACE_TRIPLES "/home/cosmos/omoknuni_small/CMakeLists.txt;1368;add_test;/home/cosmos/omoknuni_small/CMakeLists.txt;1413;make_test;/home/cosmos/omoknuni_small/CMakeLists.txt;0;")
subdirs("_deps/concurrentqueue-build")
subdirs("_deps/mimalloc-build")
subdirs("_deps/tracy-build")
