"""Fig. 14: the proposed estimator against the quasi-static benchmark on 'Out-of-Bounds'
(15-25 s, tip pose measurements only).

Writes, in results/fig14/ (the files figures/1D_2D_comparison_figure/1D_2D_comparison_figure.tex reads):
    outofbounds_pose_ours/tikz_estimate_data_tip.csv           ours, tip at every node
    outofbounds_pose_ours/tikz_estimate_data_T_5.000.csv       ours, shape at t = 5 s
    outofbounds_pose_ours/tikz_estimate_data_T_9.000.csv       ours, shape at t = 9 s
    outofbounds_pose_1D/tikz_estimate_data_tip.csv             quasi-static, tip at every row
    outofbounds_pose_1D/tikz_estimate_data_T_<t>.csv           quasi-static, shape at the row nearest t = 5 s
    outofbounds_pose_1D_prior/tikz_estimate_data_T_0.000.csv   the quasi-static prior alone (no data at t = 9 s)
and results/fig14/metrics.json: tip RMSE of both estimators over the ground-truth frames
within 0.01 s of a quasi-static row (Sec. III-D).
"""

import json
import os

import numpy as np

from common import DATA, RESULTS, errors, load_estimate, tip_positions, write_tikz

START, END, LENGTH = 15.0, 25.0, 0.466
MATCH_WINDOW = 0.01
ours = os.path.join(RESULTS, "out_of_bounds_pose")
quasi_static = os.path.join(RESULTS, "fig14_quasi_static")
prior_only = os.path.join(RESULTS, "fig14_quasi_static_prior")
out = os.path.join(RESULTS, "fig14")


def tip_rows(directory):
    return [[row[0] - START] + row[2:] for row in sorted(tip_positions(load_estimate(directory)))]


def shape_rows(directory):
    return [row[1:] for row in sorted(tip_positions(load_estimate(directory)), key=lambda r: r[1])]


write_tikz(os.path.join(out, "outofbounds_pose_ours", "tikz_estimate_data_tip.csv"), "time", tip_rows(os.path.join(ours, "tip_nodes")))
for t in (5, 9):
    write_tikz(os.path.join(out, "outofbounds_pose_ours", f"tikz_estimate_data_T_{t:.3f}.csv"), "arclength", shape_rows(os.path.join(ours, f"shape_{t}")))

# The quasi-static shape query at t = 5 s resolves to its nearest row; name the file by that row's time.
rows = np.loadtxt(os.path.join(quasi_static, "rows.txt"))
nearest_row = rows[np.argmin(np.abs(rows - (START + 5.0)))] - START
write_tikz(os.path.join(out, "outofbounds_pose_1D", "tikz_estimate_data_tip.csv"), "time", tip_rows(os.path.join(quasi_static, "tip_nodes")))
write_tikz(os.path.join(out, "outofbounds_pose_1D", f"tikz_estimate_data_T_{nearest_row:.3f}.csv"), "arclength", shape_rows(os.path.join(quasi_static, "shape_5")))
write_tikz(os.path.join(out, "outofbounds_pose_1D_prior", "tikz_estimate_data_T_0.000.csv"), "arclength", shape_rows(os.path.join(prior_only, "shape_0")))

# Tip RMSE over the same ground-truth frames for both: those within 0.01 s of a quasi-static row.
metrics = {"match_window_s": MATCH_WINDOW, "quasi_static_rows": int(len(rows))}
for name, directory in (("quasi_static", quasi_static), ("ours", ours)):
    tip = [x for x in errors(directory, os.path.join(DATA, "video5"), START, END) if x["type"] == "pose" and abs(x["s"] - LENGTH) < 1e-9]
    tip = [x for x in tip if np.min(np.abs(rows - x["t"])) <= MATCH_WINDOW]
    trans = np.array([x["trans"] / LENGTH * 100.0 for x in tip])
    rot = np.array([x["rot"] for x in tip])
    metrics[name] = {"frames": len(tip), "tip_position_rmse_pct": float(np.sqrt((trans ** 2).mean())),
                     "tip_rotation_rmse_rad": float(np.sqrt((rot ** 2).mean()))}
    print(f"{name}: {len(tip)} frames, tip RMSE {metrics[name]['tip_position_rmse_pct']:.3f} %, {metrics[name]['tip_rotation_rmse_rad']:.4f} rad")
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Wrote results/fig14/")
