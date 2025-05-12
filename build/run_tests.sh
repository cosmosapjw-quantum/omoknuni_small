#!/bin/bash
export LD_LIBRARY_PATH="/opt/libtorch-2.7.0-cu128/lib:$LD_LIBRARY_PATH"
cd "/home/cosmos/omoknuni_small/build"
ctest "$@"
