# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Space-Time Continuum (`spacetime`): a C++20 library for continuous-time-and-space state estimation of continuum robots. It solves MAP estimation by factor-graph optimization on GTSAM. Dependencies: Eigen 3.4, GTSAM, jsoncpp, and optionally VTK 9.3 and Qt 6.8 for the visualizer. Developed on Ubuntu 24.04. macOS and Windows are not supported, because the Qt visualizer runs on a worker thread.

## Build

- `./build_release.sh [--python] [-D...]` configures `build/` with `-DVisualizer=ON -DBuildTests=ON -DMarchNative=ON` and builds it. `--python` also builds the bindings and pip-installs them into `.venv` if it exists, otherwise into `$PYTHON`.
- Manual build: `cmake -B build [options] && cmake --build build -j$(nproc)`.
- CMake options: `Debug` (`-Og -g3`, and turns on `SPACETIME_DEBUG_LOG`), `Profile` (`-pg`), `BuildTests`, `Visualizer`, `VisualizerBackend` (only `default`), `MarchNative`, `InterfaceStandalone` (default ON), `InterfaceROS2` (needs a sourced ROS2 distribution), `InterfaceMujoco` (not implemented; FATAL_ERROR), `Python`.
- `Visualizer`, `InterfaceROS2` and `Python` each require `InterfaceStandalone=ON`.
- `-march=native` changes Eigen's alignment, so `MarchNative` must match how GTSAM (and any downstream project) was built. A mismatch shows up as crashes at runtime, not as build errors.
- Dependencies (Qt6, VTK, GTSAM) are found via CMake's normal search; non-system installs go on `CMAKE_PREFIX_PATH`. Keep the repo free of machine-specific paths. `build_release.sh` passes extra `-D...` args through to cmake.
- Source files are picked up with `file(GLOB_RECURSE ...)`, so re-run cmake after adding a file.
- Docs: `cmake --build build --target docs` (Doxygen, if found).

## Tests

- C++ (GoogleTest, fetched by FetchContent): every `tests/**/*.cpp` is compiled into one `build/tests/tests` binary.
  - Run all: `ctest --test-dir build/tests` (not from `build/`: `enable_testing()` is only in tests/CMakeLists.txt) or `./build/tests/tests`. Visualizer tests need `QT_QPA_PLATFORM=offscreen` when headless.
  - Run one: `./build/tests/tests --gtest_filter='SuiteName.TestName'` (wildcards allowed), or `ctest -R <regex>`
  - Separate binaries: `visualizer_thread_tests` (needs its own QApplication thread) and `ros2_tests` (share one rclcpp context; `tests/ros2/main.cpp`).
  - Which tests are built depends on the options: `tests/visualizer/` only with Visualizer, `tests/standalone/` and `BatchEstimatorTests.cpp` only with InterfaceStandalone.
  - ctest runs each test in its own process concurrently: tests that write files must use a per-test path (e.g. `testing::TempDir()` + `current_test_info()->name()`).
  - `PROJECT_SOURCE_DIR` is defined for tests that read checked-in files. `tests/NumericalDifferentiation.hpp` provides helpers for checking Jacobians numerically.
- Python: `pip install -r python/requirements.txt`, `./build_release.sh --python`, then `python -m pytest python/tests` (or `-k <name>` for one test). GTSAM's own Python module must be installed from the same GTSAM build (`GTSAM_BUILD_PYTHON=ON`), and pybind11 is pinned to GTSAM's wrapper version. `test_demo_regression.py` compares the Python demos' output with the C++ demos' `--output <dir>` output.

## Demos

`build/examples/{robot_joining,stewart_gough,robot_joining_parallel,bucky_ball}` take `[--headless] [--output <dir>]`. Python ports are in `python/examples/`. The paper reproductions are self-contained under `papers/<paper>/`, each with its own build.sh, run scripts, configs and data, and they build against the installed `spacetime` CMake package.

## Architecture

CMake targets, exported as `spacetime::<name>` by `cmake --install`: `estimator` (core), `interface` (header-only abstractions), `standalone`, `ros2` + `rviz`, `interfaces` (every interface that was built), `visualizer`. Headers sit next to their sources under `spacetime/` and are included as `<spacetime/...>`.

### Estimator core (`spacetime/estimator/`)
- `Types::Model` (`types/Model.hpp`) is the base for everything that produces factors. Each `Estimator::estimate(time)` runs two phases over the models, in the order they were added:
  1. `extendState`: add new variables and initial guesses, and register their keys with the `KeyManager` so they can be marginalized.
  2. `getNewFactors`: add factors on variables that now exist.
  A model's `extendState` must not depend on another model's `extendState` in the same step. Anything that needs another model's new variables belongs in `getNewFactors`. Other hooks: `onMarginalization`, `lockedValues`, covariance key groups.
- `Types::StatefulModel<StateT>` is a model that owns a state, i.e. a prior. Only one stateful model may extend a given state. It resolves its state at any (time, arclength) via `getOuterFactor`, which wraps a sensor's inner factor with interpolation, and via `query`.
- Priors live in `prior_models/`: `WNOA`, `WNOV` and `WNOAQuasiStatic`, each with its own State, Prior, Interpolator and InterpolatedKeys. `WNOA::PriorOptions::factor_type` (config `prior.wnoa.factor_type`) selects the WNOA factors: `WNOA` (default) and `Legacy` use the custom group factors in `WNOA/` (`Legacy` adds the strain-velocity coupling term), and `Experimental` uses the GTSAM Lie-algebra factors built in `WNOA/experimental/`. A robot is a grid state (e.g. `ContinuumRobotState(length, nodes)`, which also holds locks such as a fixed base) plus the prior that extends it. The state does not own its prior.
- Sensors (`sensor_models/`, derived from `Types::SensorBase`) and actuators (`Types::ActuatorBase`) hold a `StatefulModel<>` and a `MeasurementQueue`. Joints and end effectors (`joint_models/`) join robots. An `EndEffector` owns its prior and joints, so one `addModel` registers all of them.
- `Estimator` modes come from `EstimatorOptions::smootherLag`. `Estimator::BATCH_LAG` (-1) selects batch mode: nothing is marginalized and `finalize()` runs one LM solve. Otherwise it runs a fixed-lag smoother, `Smoother::Batch` or `Smoother::Incremental` (iSAM2); a filter is a sliding window with a short lag. There is also a quasi-static mode, and optional posterior covariance extraction (`compute_covariance`).
- Results are published as thread-safe snapshots: `getSnapshot()` and `waitForSnapshotNewerThan()`.

### Interface layer (`spacetime/interface/`)
This layer starts no threads of its own; `tests/interface/InterfaceIsThreadFreeTests.cpp` enforces that. It provides `Clock`, `Dataloader` (with a delivery watermark), `EstimationSchedule`, `EstimatorDriver::step(time)` and `SnapshotHistory`. Each interface lives in `interface/<name>/` and declares a `Spacetime::Interfaces::<Name>` bundle that satisfies the `InterfaceType` concept (`InterfaceConcept.hpp` documents every hook). Downstream code picks one with a single line: `using SelectedInterface = Spacetime::Interfaces::Standalone;`.
- Standalone: the estimator thread (`EstimatorRunner` paced by a `PacingClock`), one `ThreadedDataloader` per source (e.g. `FileDataloader`), and lowered priority for the visualizer and query threads. With `SteppedClock` (offline) it waits for every dataloader, so results are deterministic. With `WallClock` (real time) it never waits for data.
- ROS2: an estimator node plus one node per sensor, with executors owning the threads. `visualizer/rviz` publishes snapshots as markers and TF.

### Visualizer (`spacetime/visualizer/`)
A backend-agnostic layer (`BackendConcept.hpp`, `Visualizable`, `PlaybackController`) plus the `default/` Qt/VTK backend, whose `VisualizerThread` owns all Qt and VTK objects. `QT_NO_KEYWORDS` is set publicly because Qt's `emit`/`signals`/`slots` clash with oneTBB identifiers that GTSAM pulls in. Use `Q_SLOTS` (and `Q_SIGNALS`/`Q_EMIT`) instead.

### Python (`python/`)
pybind11 bindings in `python/src/bind_*.cpp` build the `_spacetime` extension, and the pure-Python package in `python/spacetime/` is copied next to it. Every C++ base class can be subclassed from Python. Python models should build their factors from existing C++ or GTSAM factors so that factor evaluation stays in C++.
