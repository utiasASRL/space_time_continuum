# TRO final revision results

Everything needed to regenerate the results of the TRO paper with the refactored spacetime
estimator: Tables V and VI and the data of Figs. 11, 12, 13 and 14. Every run is a batch
estimate with the "wnoa" factor type, except the quasi-static benchmark of Fig. 14, which
uses the `WNOAQuasiStatic` prior (GTSAM WNOA unary and arclength factors, same weights).

## Build

1. Build and install spacetime, from the spacetime repository:

   ```bash
   ./build_release.sh
   sudo cmake --install build
   ```

2. Build the runner, from this folder:

   ```bash
   ./build.sh
   ```

The scripts in `scripts/` need Python 3 with numpy, scipy and matplotlib.

## Run

```bash
./run_all.sh       # Tables V, VI and Figs. 11, 12, 14 (a few minutes)
./run_runtime.sh   # Fig. 13: 64 runs sweeping N and K
```

## Layout

| Path | Contents |
|---|---|
| `runner/` | `tro_runner`: one batch estimate of one trial, written out for the scripts (see the top of `main.cpp`) |
| `configs/experiment_base.json` | Estimator and prior of the experiments (N = 5, nodes at 30 Hz) |
| `configs/simulation_base.json` | Estimator and prior of the simulations (N = 7, nodes at 30 Hz) |
| `configs/trials/` | One data section per trial and sensor configuration |
| `configs/holdout/` | The Q3 check on the hold-out data (see below) |
| `data/` | Trial data: `video5`-`video8` (experiments, tip Aurora sensor only), `simulation*` |
| `scripts/` | Metrics, tables and figure data |
| `results/` | Everything the scripts write |

Weights are in GTSAM order ([rotational, translational] per 6-vector).

## Outputs

| Paper item | File | Script |
|---|---|---|
| Table V | `results/table5.{md,tex,json}` | `make_tables.py` |
| Table VI | `results/table6.{md,tex,json}` | `make_tables.py` |
| Fig. 11 | `results/fig11/tikz_*.csv` | `export_fig11.py` |
| Fig. 12 | `results/fig12/chi2_coverage.{csv,pdf}` | `chi2_coverage.py` |
| Fig. 13 | `results/fig13/tikz_per_iter_runtime_data.csv` | `runtime_fig13.py` |
| Fig. 14 | `results/fig14/outofbounds_pose_*/tikz_*.csv`, `results/fig14/metrics.json` | `export_fig14.py` |
| Q3 check | `results/q3_holdout.json` | `q3_holdout.py` |
| Changes vs the paper | `results/comparison.json` | `compare_to_paper.py` (reads the paper repository's old result data) |

The figure CSVs have the columns the paper's TikZ figures read, with times relative to
the start of each trial.

## Metrics

- Position and rotation errors are those of `Log(T_est^-1 T_gt)` at every motion-capture (or
  simulated ground-truth) sample, with the estimate queried at the sample's time.
- "Tip" is the last frame; "Body" is every other frame. Experimental positions are in % of each
  frame's arclength (base frames left out), simulated ones in mm (velocities in mm/s).
- Average NEES is over every ground-truth frame, joint over pose (and, in simulation, velocity).
- The Fig. 14 RMSE uses only the ground-truth frames within 0.01 s of a quasi-static row,
  for both estimators.

The Q3 check runs the fused hold-out trajectory (the Out-of-Bounds recording, 25-35 s) with
the two candidate Q3 values (the original tuning file's and the paper appendix's) and reports
which gives an average NEES closer to 6.
