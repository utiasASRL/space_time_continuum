#!/bin/bash
# Tests a built wheel in a fresh Colab-like container (ci/Dockerfile.colab: root, no venv, no
# GTSAM). ci/colab_test.ipynb's first cell installs the wheel, as it would in Colab, and the
# notebook is executed headlessly; then the import time is measured and python/tests are run
# against the installed wheel.
#
# Usage: ci/test_fresh.sh [python version]   (default: 3.13, Google Colab's; after ci/build_wheels.sh)
# Needs: Docker.
# Environment: WORK_DIR (default: <repo>/local_testing), as for ci/build_wheels.sh.
# Output: $WORK_DIR/out/colab_test.executed.ipynb, $WORK_DIR/logs/test_fresh_<tag>.log
set -euo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$CI_DIR/.." && pwd)"
WORK_DIR="$(cd "${WORK_DIR:-$REPO/local_testing}" && pwd)"
VERSION="${1:-3.13}"
TAG="cp${VERSION/./}"
IMAGE="spacetime-colab-test:py${VERSION}"

shopt -s nullglob
wheels=("$WORK_DIR"/wheelhouse/spacetime_continuum-*-"$TAG"-*.whl)
[ ${#wheels[@]} -gt 0 ] || { echo "No $TAG wheel in $WORK_DIR/wheelhouse: run ci/build_wheels.sh $VERSION first"; exit 1; }
[ -d "$WORK_DIR/src" ] || { echo "No source snapshot in $WORK_DIR/src: run ci/build_wheels.sh first"; exit 1; }

docker build -q --build-arg PYTHON_VERSION="$VERSION" -t "$IMAGE" -f "$CI_DIR/Dockerfile.colab" "$CI_DIR"
mkdir -p "$WORK_DIR/out" "$WORK_DIR/logs"
rm -f "$WORK_DIR/out/"*

# The tests and examples come from the snapshot the wheel was built from ($WORK_DIR/src).
docker run --rm \
    -v "$WORK_DIR/wheelhouse:/wheelhouse:ro" \
    -v "$WORK_DIR/src:/repo:ro" \
    -v "$CI_DIR/colab_test.ipynb:/work/colab_test.ipynb:ro" \
    -v "$WORK_DIR/out:/out" \
    -e SPACETIME_EXAMPLES=/repo/python/examples \
    "$IMAGE" bash -euc '
        python -m jupyter nbconvert --to notebook --execute /work/colab_test.ipynb \
            --ExecutePreprocessor.timeout=600 --output-dir /out --output colab_test.executed.ipynb
        echo "--- notebook executed without errors"
        pip list 2>/dev/null | grep -Ei "^(spacetime|spacetime-continuum|gtsam|numpy) "
        cd /tmp
        python -X importtime -c "import spacetime" 2> /tmp/importtime.txt
        echo "--- import time (us, cumulative):"; tail -1 /tmp/importtime.txt
        python -m pytest -q -p no:cacheprovider /repo/python/tests
        chown -R '"$(id -u):$(id -g)"' /out
    ' 2>&1 | tee "$WORK_DIR/logs/test_fresh_${TAG}.log"
echo "Executed notebook: $WORK_DIR/out/colab_test.executed.ipynb"
