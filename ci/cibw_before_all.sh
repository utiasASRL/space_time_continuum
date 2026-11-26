#!/bin/bash
# cibuildwheel before-all step (pyproject.toml [tool.cibuildwheel.linux]), run once per manylinux
# container: builds and installs into /usr/local the C++ dependencies of the spacetime wheel,
# Eigen, jsoncpp and GTSAM, and builds GTSAM's Python module for one Python version in
# /opt/gtsam-build/python/gtsam, which the wheel bundles (SPACETIME_BUNDLE_GTSAM_PY).
#
# Usage: cibw_before_all.sh <python version, e.g. 3.12> <project dir>
# Environment: CIBW_JOBS (default: nproc) build parallelism.
#
# GTSAM is built without TBB and Boost and without -march=native, so the wheel is portable and
# self-contained. The versions here are the ones spacetime is developed against.
set -euo pipefail
set -x

PYTHON_VERSION="${1:?usage: $0 <python version> <project dir>}"
PROJECT_DIR="${2:?usage: $0 <python version> <project dir>}"
JOBS="${CIBW_JOBS:-$(nproc)}"

EIGEN_VERSION=3.4.0
JSONCPP_VERSION=1.9.6
GTSAM_VERSION=4.3.0
PREFIX=/usr/local
SRC=/opt/src
GTSAM_BUILD=/opt/gtsam-build

PYTHON="$(command -v "python${PYTHON_VERSION}")"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install numpy pyparsing

mkdir -p "$SRC"
cd "$SRC"

# Eigen 3.4 (header-only; the system eigen3 of manylinux_2_28 is 3.3).
curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" | tar xz
cmake -S "eigen-${EIGEN_VERSION}" -B eigen-build -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
cmake --build eigen-build --target install

# jsoncpp, as a shared library (the static one is often built without -fPIC).
curl -fsSL "https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/${JSONCPP_VERSION}.tar.gz" | tar xz
cmake -S "jsoncpp-${JSONCPP_VERSION}" -B jsoncpp-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_OBJECT_LIBS=OFF \
    -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF -DJSONCPP_WITH_EXAMPLE=OFF
cmake --build jsoncpp-build -j"$JOBS" --target install

# GTSAM, with its Python module for $PYTHON_VERSION.
git clone --depth 1 --branch "$GTSAM_VERSION" https://github.com/borglab/gtsam.git gtsam
cmake -S gtsam -B "$GTSAM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_DOCS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF \
    -DGTSAM_UNSTABLE_BUILD_PYTHON=OFF \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
    -DGTSAM_BUILD_WITH_WERROR=OFF \
    -DGTSAM_WITH_TBB=OFF \
    -DGTSAM_USE_BOOST_FEATURES=OFF \
    -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_USE_QUATERNIONS=OFF \
    -DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON \
    -DGTSAM_SLOW_BUT_CORRECT_EXPMAP=ON \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_PYTHON_VERSION="$PYTHON_VERSION" \
    -DPYTHON_EXECUTABLE:FILEPATH="$PYTHON"
cmake --build "$GTSAM_BUILD" -j"$JOBS" --target install
# The generated wrapper's translation units need several GB each (GTSAM's own wheel build
# uses -j1): with one job per CPU, a 16 GB GitHub runner runs out of memory and the job is
# killed (exit 137). One job per GTSAM_WRAP_GB (default 6) of available memory, at most JOBS.
MEM_GB=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
WRAP_JOBS="${GTSAM_WRAP_JOBS:-$(( MEM_GB / ${GTSAM_WRAP_GB:-6} ))}"
WRAP_JOBS=$(( WRAP_JOBS < 1 ? 1 : (WRAP_JOBS > JOBS ? JOBS : WRAP_JOBS) ))
echo "Wrapper build: ${WRAP_JOBS} jobs (${MEM_GB} GB available)"
cmake --build "$GTSAM_BUILD" -j"$WRAP_JOBS" --target gtsam_py
test -f "$GTSAM_BUILD/python/gtsam/__init__.py"
# The module's RPATH points at the build tree's libgtsam, _spacetime's at the installed one:
# auditwheel would bundle both. Drop it, so both resolve to $PREFIX/lib (LD_LIBRARY_PATH).
for module in "$GTSAM_BUILD"/python/gtsam/*.so; do
    patchelf --remove-rpath "$module"
done
ls "$GTSAM_BUILD"/python/gtsam/*.so
