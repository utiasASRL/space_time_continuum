#!/bin/bash
# Usage: ./build_release.sh [--python] [-D<cmake option>...]
#   --python  Also build the Python bindings (python/) and pip-install the spacetime package into
#             the project's .venv if it exists, otherwise into $PYTHON (default: python3)'s environment.
#   -D...     Passed on to cmake, e.g. -DCMAKE_PREFIX_PATH=$HOME/Qt/6.8.0/gcc_64 for a Qt (or VTK, GTSAM)
#             installed outside the default search path, or -DMarchNative=OFF.

python_flags=(-DPython=OFF)
cmake_flags=()
build_python=0
for arg in "$@"; do
    case "$arg" in
        --python) build_python=1 ;;
        -D*) cmake_flags+=("$arg") ;;
        *) echo "Unknown option: $arg (usage: $0 [--python] [-D<cmake option>...])"; exit 1 ;;
    esac
done
if [ $build_python -eq 1 ]; then
    if [ -z "$PYTHON" ]; then
        if [ -x .venv/bin/python ]; then PYTHON=$(pwd)/.venv/bin/python; else PYTHON=$(command -v python3); fi
    fi
    python_flags=(-DPython=ON -DPython_EXECUTABLE="$PYTHON")
fi

mkdir -p build
cd build
cmake -DDebug=OFF -DVisualizer=ON -DBuildTests=ON -DMarchNative=ON -DInterfaceROS2=ON "${python_flags[@]}" "${cmake_flags[@]}" .. || exit 1

cmake --build . -j$(nproc) || exit 1
if [ $build_python -eq 1 ]; then
    cmake --build . --target python-install || exit 1
fi
cd ..
