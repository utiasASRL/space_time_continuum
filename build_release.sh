#!/bin/bash
# Usage: ./build.sh [--python]
#   --python  Also build the Python bindings (python/) and pip-install the spacetime package into
#             the project's .venv if it exists, otherwise into $PYTHON (default: python3)'s environment.

python_flags=(-DPython=OFF)
build_python=0
for arg in "$@"; do
    case "$arg" in
        --python) build_python=1 ;;
        *) echo "Unknown option: $arg (usage: $0 [--python])"; exit 1 ;;
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
cmake -DDebug=OFF -DVisualizer=ON -DBuildTests=ON -DMarchNative=ON "${python_flags[@]}" .. || exit 1

cmake --build . -j$(nproc) || exit 1
if [ $build_python -eq 1 ]; then
    cmake --build . --target python-install || exit 1
fi
cd ..
