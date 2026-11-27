#!/bin/bash
# Builds the spacetime-continuum wheel(s) in Docker with cibuildwheel, using the
# [tool.cibuildwheel] settings of pyproject.toml (which runs ci/build_deps.py in the
# manylinux container to build Eigen, jsoncpp and GTSAM). One Python version per cibuildwheel
# run, because GTSAM's Python module is built for that version in before-all.
#
# Usage: ci/build_wheels.sh [python version...]   (default: 3.13, Google Colab's)
#   e.g. ci/build_wheels.sh 3.10 3.11 3.12 3.13 3.14 3.15
# Needs: Docker, git, python3 (with venv).
# Environment: WORK_DIR (default: <repo>/local_testing, gitignored) holds the outputs;
#   CIBW_JOBS (default: nproc) build parallelism; GTSAM_JOBS / GTSAM_WRAP_JOBS override the number
#   of parallel libgtsam / GTSAM wrapper compiles (otherwise picked from the available memory, see
#   ci/build_deps.py), and CMAKE_BUILD_PARALLEL_LEVEL bounds the extension's.
# The same build runs on GitHub (Linux, macOS, Windows) when a release is published: see
# .github/workflows/wheels.yml.
# Output: $WORK_DIR/wheelhouse/spacetime_continuum-*.whl, logs in $WORK_DIR/logs.
#
# The build runs on a clean copy of the repository (tracked + untracked-but-not-ignored files,
# i.e. what a fresh clone with your working-tree changes would contain) in $WORK_DIR/src, so
# local build/ directories never leak into the container. Upload the wheels to a GitHub release
# by hand; ci/test_fresh.sh tests one in a fresh Colab-like container first.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$REPO/local_testing}"
VERSIONS=("$@")
[ ${#VERSIONS[@]} -eq 0 ] && VERSIONS=(3.13)
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

# cibuildwheel in its own venv.
# Upgraded on every run, so an existing venv does not keep an older cibuildwheel (each release
# only knows the Python versions and manylinux images of its time).
if [ ! -x "$WORK_DIR/.venv/bin/python" ]; then
    python3 -m venv "$WORK_DIR/.venv"
    "$WORK_DIR/.venv/bin/pip" install -q --upgrade pip
fi
"$WORK_DIR/.venv/bin/pip" install -q --upgrade "cibuildwheel>=4.3,<5"

# Clean source snapshot.
rm -rf "$WORK_DIR/src"
mkdir -p "$WORK_DIR/src"
(cd "$REPO" && git ls-files -co --exclude-standard | while IFS= read -r f; do [ -e "$f" ] && printf '%s\n' "$f"; done \
    | tar -cf - -T -) | tar -xf - -C "$WORK_DIR/src"

mkdir -p "$WORK_DIR/wheelhouse" "$WORK_DIR/logs"
for version in "${VERSIONS[@]}"; do
    tag="cp${version/./}"
    echo "=== Building ${tag}-manylinux_x86_64 (log: $WORK_DIR/logs/build_${tag}.log)"
    # cibuildwheel's project directory is the current directory: run it from the snapshot.
    CIBW_BUILD="${tag}-manylinux_x86_64" CIBW_PYTHON="$version" CIBW_JOBS="${CIBW_JOBS:-$(nproc)}" \
        bash -c 'cd "$1/src" && "$1/.venv/bin/cibuildwheel" --platform linux --output-dir "$1/wheelhouse" .' _ "$WORK_DIR" \
        2>&1 | tee "$WORK_DIR/logs/build_${tag}.log"
done
ls -la "$WORK_DIR/wheelhouse"
