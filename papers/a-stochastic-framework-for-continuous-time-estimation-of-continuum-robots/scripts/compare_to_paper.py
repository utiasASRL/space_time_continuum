"""How the regenerated results differ from the ones in the paper.

Tables V and VI are compared with the values printed in the paper; Figs. 11, 12, 13 and 14 with
the data the paper's figures were drawn from (assets/revisions/ and assets/demo_data/ of the paper
repository, read only). Run after run_all.sh (and run_runtime.sh for Fig. 13).

Writes results/comparison.json and prints a summary.
"""

import glob
import json
import os

import numpy as np
from scipy.stats import chi2

from common import RESULTS

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OLD = os.path.join(PAPER, "assets", "revisions", "experiment", "results")

COLUMNS = ["MAE tip lin", "MAE tip rot", "MAE body lin", "MAE body rot",
           "RMSE tip lin", "RMSE tip rot", "RMSE body lin", "RMSE body rot"]

# The values printed in the paper.
TABLE_VI = {  # (trial, configuration): 8 metrics (pos in %, rot in rad), average NEES
    ("Out-of-Bounds", "Pose-only"): [1.560, 0.043, 6.534, 0.128, 2.295, 0.059, 8.227, 0.173, 6.437],
    ("Out-of-Bounds", "Gyro-only"): [18.046, 0.500, 12.525, 0.186, 19.119, 0.508, 14.485, 0.238, 9.548],
    ("Out-of-Bounds", "Fused"): [1.489, 0.043, 6.151, 0.110, 2.031, 0.050, 7.776, 0.141, 10.219],
    ("Fast Contact", "Pose-only"): [1.169, 0.040, 5.800, 0.091, 1.276, 0.044, 7.065, 0.116, 4.056],
    ("Fast Contact", "Gyro-only"): [17.480, 0.285, 13.607, 0.134, 20.124, 0.303, 15.891, 0.165, 4.769],
    ("Fast Contact", "Fused"): [1.167, 0.048, 5.274, 0.068, 1.273, 0.066, 6.482, 0.085, 7.092],
    ("Impulse 1", "Pose-only"): [1.562, 0.043, 6.289, 0.107, 1.696, 0.048, 6.721, 0.130, 5.427],
    ("Impulse 1", "Gyro-only"): [21.399, 0.296, 16.750, 0.222, 21.665, 0.300, 17.516, 0.261, 7.216],
    ("Impulse 1", "Fused"): [1.562, 0.044, 6.112, 0.100, 1.695, 0.052, 6.465, 0.117, 8.860],
    ("Impulse 2", "Pose-only"): [1.320, 0.059, 6.877, 0.116, 1.405, 0.066, 8.488, 0.143, 6.255],
    ("Impulse 2", "Gyro-only"): [24.192, 0.525, 16.753, 0.168, 26.055, 0.536, 18.848, 0.213, 11.025],
    ("Impulse 2", "Fused"): [1.323, 0.076, 5.874, 0.086, 1.409, 0.091, 7.717, 0.111, 12.249],
    ("Slow Free Space", "Pose-only"): [1.102, 0.042, 5.004, 0.104, 1.165, 0.042, 5.347, 0.138, 4.759],
    ("Slow Free Space", "Gyro-only"): [14.357, 0.816, 7.834, 0.120, 14.592, 0.818, 8.005, 0.147, 5.739],
    ("Slow Free Space", "Fused"): [1.102, 0.042, 4.921, 0.103, 1.164, 0.042, 5.262, 0.135, 7.514],
}
TABLE_V = {  # (trial, data type): 8 metrics (mm or mm/s, rad or rad/s), average NEES (pose rows)
    ("Simulation", "Pose"): [1.521, 0.005, 1.216, 0.021, 1.778, 0.006, 1.741, 0.029, 8.45],
    ("Simulation", "Velocity"): [29.035, 0.049, 15.965, 0.032, 42.504, 0.066, 30.349, 0.055, None],
    ("Simulation High Noise", "Pose"): [1.961, 0.007, 1.403, 0.021, 2.283, 0.007, 1.963, 0.030, 8.66],
    ("Simulation High Noise", "Velocity"): [32.256, 0.071, 17.129, 0.056, 47.518, 0.071, 31.74, 0.056, None],
    ("Simulation Dropout", "Pose"): [2.020, 0.006, 1.441, 0.021, 2.630, 0.007, 2.274, 0.030, 8.41],
    ("Simulation Dropout", "Velocity"): [29.937, 0.050, 17.273, 0.032, 43.867, 0.067, 32.373, 0.055, None],
}


def table_changes(name, paper):
    """Per cell (old, new, change, relative change), and the largest and mean changes per group."""
    with open(os.path.join(RESULTS, name + ".json")) as f:
        new = {(row["trial"], row["configuration"]): [row[c] for c in COLUMNS] + [row["NEES"]] for row in json.load(f)}
    cells = []
    for key, old_values in paper.items():
        for column, old, value in zip(COLUMNS + ["NEES"], old_values, new[key]):
            if old is None or value is None:
                continue
            cells.append({"row": " / ".join(key), "column": column, "old": old, "new": value,
                          "change": value - old, "relative": (value - old) / old if old else float("nan")})

    def summary(selected):
        relative = np.abs([c["relative"] for c in selected])
        largest = max(selected, key=lambda c: abs(c["relative"]))
        return {"cells": len(selected), "mean_abs_relative_change": float(np.mean(relative)),
                "median_abs_relative_change": float(np.median(relative)), "mean_relative_change": float(np.mean([c["relative"] for c in selected])),
                "largest": largest}

    groups = {"all accuracy": [c for c in cells if c["column"] != "NEES"],
              "position": [c for c in cells if "lin" in c["column"]],
              "rotation": [c for c in cells if "rot" in c["column"]],
              "NEES": [c for c in cells if c["column"] == "NEES"]}
    for configuration in sorted({key[1] for key in paper}):
        groups[f"accuracy, {configuration}"] = [c for c in groups["all accuracy"] if c["row"].endswith(configuration)]
        nees = [c for c in groups["NEES"] if c["row"].endswith(configuration)]
        if nees:
            groups[f"NEES, {configuration}"] = nees
    return {"groups": {g: summary(c) for g, c in groups.items() if c}, "cells": cells}


def read_tikz(path):
    """A pgfplots table as {column: array} (the old files end rows with a comma)."""
    with open(path) as f:
        header = [h for h in f.readline().strip().split(",") if h]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    data = np.atleast_2d(data)[:, :len(header)]
    return {h: data[:, i] for i, h in enumerate(header)}


def curve_changes(old_path, new_path, axis):
    """Differences between two (axis, pos1..3 with 3-sigma bounds) curves, the new one resampled
    on the old one's `axis` values: mean distance, max distance, and the mean 3-sigma half width."""
    old, new = read_tikz(old_path), read_tikz(new_path)
    x = old[axis]
    position = lambda table, at: np.column_stack([np.interp(at, table[axis], table[f"pos{i}"]) for i in (1, 2, 3)])
    width = lambda table: np.mean([(table[f"pos{i}_ub"] - table[f"pos{i}_lb"]) / 2.0 for i in (1, 2, 3)])
    distance = np.linalg.norm(position(new, x) - position(old, x), axis=1)
    return {"samples_old": int(len(x)), "samples_new": int(len(new[axis])),
            "mean_distance_mm": float(1000 * distance.mean()), "max_distance_mm": float(1000 * distance.max()),
            "mean_3sigma_halfwidth_mm_old": float(1000 * width(old)), "mean_3sigma_halfwidth_mm_new": float(1000 * width(new))}


def old_coverage(mode):
    """The old Fig. 12 curve over the five trajectories (the hold-out run left out), with exact
    chi-square quantiles, as the new curve is computed."""
    nees = []
    for path in glob.glob(os.path.join(OLD, f"*_{mode}", "logs", "nees_cove*_data.txt")):
        if "tuning" in path:
            continue
        with open(path) as f:
            for line in f.readlines()[1:]:
                kind, _, _, value, dof = line.strip().split(",")
                if kind == "Joint":
                    nees.append((float(value), int(dof)))
    nominal = np.arange(0, 101)
    return np.array([100.0 * np.mean([v <= chi2.ppf(n / 100.0, d) for v, d in nees]) for n in nominal]), len(nees)


def figures():
    out = {}
    fig11_old = os.path.join(OLD, "fastcontact_both_time_interp")
    out["fig11"] = {
        "interpolated": curve_changes(os.path.join(fig11_old, "tikz_estimate_data_tip.csv"),
                                      os.path.join(RESULTS, "fig11", "tikz_estimate_data_tip.csv"), "time"),
        "estimated nodes": curve_changes(os.path.join(fig11_old, "tikz_estimate_data_tip_est.csv"),
                                         os.path.join(RESULTS, "fig11", "tikz_estimate_data_tip_est.csv"), "time"),
    }

    new_coverage = np.genfromtxt(os.path.join(RESULTS, "fig12", "chi2_coverage.csv"), delimiter=",", names=True)
    out["fig12"] = {}
    for old_mode, new_mode in (("pose", "pose"), ("gyro", "gyro"), ("both", "fused")):
        old, samples = old_coverage(old_mode)
        difference = new_coverage[new_mode] - old
        out["fig12"][new_mode] = {"samples_old": samples, "mean_abs_difference_pct": float(np.mean(np.abs(difference))),
                                  "max_abs_difference_pct": float(np.max(np.abs(difference))),
                                  "at_nominal_pct": {str(n): {"old": float(old[n]), "new": float(new_coverage[new_mode][n])} for n in (39, 68, 87, 95, 99)}}

    fig13_new = os.path.join(RESULTS, "fig13", "tikz_per_iter_runtime_data.csv")
    if os.path.exists(fig13_new):
        old = np.genfromtxt(os.path.join(PAPER, "assets", "demo_data", "results", "heatmap", "tikz_per_iter_runtime_data.csv"), delimiter=",", names=True)
        new = np.genfromtxt(fig13_new, delimiter=",", names=True)
        ratios = {name: [float(n / o) for o, n, k in zip(old[name], np.interp(old["K"], new["K"], new[name]), old["K"]) if o > 0]
                  for name in new.dtype.names[1:] if name in old.dtype.names}
        all_ratios = np.concatenate([r for r in ratios.values()])
        out["fig13"] = {"mean_runtime_ratio_new_over_old": float(np.mean(all_ratios)), "min_ratio": float(all_ratios.min()),
                        "max_ratio": float(all_ratios.max()), "per_N_mean_ratio": {k: float(np.mean(v)) for k, v in ratios.items()}}
    else:
        out["fig13"] = "not run (run_runtime.sh)"

    old14 = lambda folder, name: os.path.join(OLD, folder, name)
    new14 = lambda folder, name: os.path.join(RESULTS, "fig14", folder, name)
    out["fig14"] = {
        "ours, tip over time": curve_changes(old14("outofbounds_pose_ours", "tikz_estimate_data_tip.csv"),
                                             new14("outofbounds_pose_ours", "tikz_estimate_data_tip.csv"), "time"),
        "ours, shape at t = 5 s": curve_changes(old14("outofbounds_pose_ours", "tikz_estimate_data_T_5.000.csv"),
                                                new14("outofbounds_pose_ours", "tikz_estimate_data_T_5.000.csv"), "arclength"),
        "ours, shape at t = 9 s": curve_changes(old14("outofbounds_pose_ours", "tikz_estimate_data_T_9.000.csv"),
                                                new14("outofbounds_pose_ours", "tikz_estimate_data_T_9.000.csv"), "arclength"),
        "quasi-static, tip over time": curve_changes(old14("outofbounds_pose_1D", "tikz_estimate_data_tip.csv"),
                                                     new14("outofbounds_pose_1D", "tikz_estimate_data_tip.csv"), "time"),
        "quasi-static, shape at t = 5 s": curve_changes(old14("outofbounds_pose_1D", "tikz_estimate_data_T_5.002.csv"),
                                                        new14("outofbounds_pose_1D", "tikz_estimate_data_T_5.002.csv"), "arclength"),
        "quasi-static prior": curve_changes(old14("outofbounds_pose_1D_prior", "tikz_estimate_data_T_0.000.csv"),
                                            new14("outofbounds_pose_1D_prior", "tikz_estimate_data_T_0.000.csv"), "arclength"),
    }
    return out


if __name__ == "__main__":
    comparison = {"table5": table_changes("table5", TABLE_V), "table6": table_changes("table6", TABLE_VI), "figures": figures()}
    with open(os.path.join(RESULTS, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    for table in ("table5", "table6"):
        print(table)
        for group, s in comparison[table]["groups"].items():
            l = s["largest"]
            print(f"  {group:28s} mean |rel| {100 * s['mean_abs_relative_change']:6.2f} %, mean rel {100 * s['mean_relative_change']:+6.2f} %, "
                  f"largest {100 * l['relative']:+7.2f} % ({l['row']}, {l['column']}: {l['old']} -> {l['new']:.3f})")
    print(json.dumps(comparison["figures"], indent=1))
