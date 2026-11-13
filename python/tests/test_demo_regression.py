"""The Python joining demos reproduce their C++ counterparts' estimates."""

import csv
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tests" / "regression" / "common"))
from compare import compare  # noqa: E402

DEMOS = ["robot_joining", "robot_joining_parallel", "stewart_gough"]
BUILD_EXAMPLES = REPO_ROOT / "build" / "examples"
PYTHON_EXAMPLES = REPO_ROOT / "python" / "examples"
RTOL, ATOL = 1e-9, 1e-12


def read_metrics(path, key_columns):
    """{'<key>/<column>': value} for every numeric cell of a demo output CSV."""
    metrics = {}
    with open(path) as file:
        for row in csv.DictReader(file):
            key = "/".join(row[c] for c in key_columns)
            for column, value in row.items():
                if column not in key_columns:
                    metrics[f"{key}/{column}"] = float(value)
    return metrics


def run_demo(command, output):
    result = subprocess.run(command + ["--headless", "--output", str(output)], capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("demo", DEMOS)
def test_python_demo_matches_cpp(demo, tmp_path):
    binary = BUILD_EXAMPLES / demo
    if not binary.exists():
        pytest.skip(f"{binary} is not built")
    run_demo([str(binary)], tmp_path / "cpp")
    run_demo([sys.executable, str(PYTHON_EXAMPLES / f"{demo}.py")], tmp_path / "python")

    for name, keys in (("estimates.csv", ["key"]), ("joints.csv", ["joint"])):
        reference = read_metrics(tmp_path / "cpp" / name, keys)
        candidate = read_metrics(tmp_path / "python" / name, keys)
        assert reference, f"{name} is empty"
        assert set(reference) == set(candidate), f"{name}: different rows or columns"
        rows, _ = compare(reference, candidate, rtol=RTOL, atol=ATOL)
        failures = [row for row in rows if not row["ok"]]
        assert not failures, failures[:10]
