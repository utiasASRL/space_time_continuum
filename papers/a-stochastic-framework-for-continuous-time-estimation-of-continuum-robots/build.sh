#!/bin/bash
# Builds the results runner (runner/) against the installed spacetime package.
# Install spacetime first, from the spacetime repository:
#   ./build_release.sh && sudo cmake --install build
set -e
cd "$(dirname "$0")"
cmake -S runner -B runner/build -DCMAKE_BUILD_TYPE=Release
cmake --build runner/build -j"$(nproc)"
