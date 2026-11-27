#!/bin/bash
# Usage: ./build_release.sh [--python] [-D<cmake option>...]
#   --python  Also build the Python bindings (python/) and pip-install the spacetime package into the
#             current Python environment: the python3 (or python) first on PATH, e.g. an activated venv,
#             conda or pyenv environment. Set $PYTHON to use another interpreter.
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
        PYTHON=$(command -v python3 || command -v python) # python only, e.g. on Windows
        [ -n "$PYTHON" ] || { echo "No python3 (or python) on PATH: activate a Python environment, or set PYTHON."; exit 1; }
    fi
    # The interpreter's real path, not a shim's (e.g. pyenv's ~/.pyenv/shims/python3, whose path stays the
    # same when the version behind it changes, so CMake would keep the previous version's cached headers).
    PYTHON=$("$PYTHON" -c 'import sys; print(sys.executable)') || exit 1
    echo "Python bindings for: $PYTHON"
    python_flags=(-DPython=ON -DPython_EXECUTABLE="$PYTHON")
fi

mkdir -p build
cd build
cmake -DDebug=OFF -DVisualizer=ON -DBuildTests=ON -DMarchNative=ON -DInterfaceROS2=ON "${python_flags[@]}" "${cmake_flags[@]}" .. || exit 1

# One job per core (nproc: Linux and Git Bash; sysctl: macOS). --config: multi-config generators (Visual Studio, Xcode).
jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config Release -j "$jobs" || exit 1
if [ $build_python -eq 1 ]; then
    cmake --build . --config Release --target python-install || exit 1
fi
cd ..
