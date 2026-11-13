"""Tables V and VI: error metrics of every simulation and experiment run.

Writes results/table5.{json,md,tex} and results/table6.{json,md,tex}.
Run after run_all.sh has produced results/<trial>/.
"""

import json
import os

from common import DATA, RESULTS, average_nees, errors, summarize

SIMULATIONS = [
    # (run, data folder, label)
    ("simulation", "simulation", "Simulation"),
    ("simulation_high_noise", "simulation_high_noise", "Simulation High Noise"),
    ("simulation_dropout", "simulation_dropout", "Simulation Dropout"),
]

EXPERIMENTS = [
    # (trial, data folder, start, end, label)
    ("out_of_bounds", "video5", 15, 25, "Out-of-Bounds"),
    ("fast_contact", "video7", 10, 20, "Fast Contact"),
    ("impulse_1", "video7", 20, 30, "Impulse 1"),
    ("impulse_2", "video8", 8, 18, "Impulse 2"),
    ("slow_free_space", "video6", 19, 29, "Slow Free Space"),
]
MODES = [("pose", "Pose-only"), ("gyro", "Gyro-only"), ("fused", "Fused")]

COLUMNS = ["MAE tip lin", "MAE tip rot", "MAE body lin", "MAE body rot",
           "RMSE tip lin", "RMSE tip rot", "RMSE body lin", "RMSE body rot"]


def cells(summary):
    tip, body = summary["tip"], summary["body"]
    return [tip["trans_mae"], tip["rot_mae"], body["trans_mae"], body["rot_mae"],
            tip["trans_rmse"], tip["rot_rmse"], body["trans_rmse"], body["rot_rmse"]]


def write_table(name, header, rows):
    """rows: [(trial label, configuration label, [8 metric cells], nees or None)]."""
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump([{"trial": t, "configuration": c, **dict(zip(COLUMNS, v)), "NEES": n} for t, c, v, n in rows], f, indent=2)

    with open(os.path.join(RESULTS, name + ".md"), "w") as f:
        f.write(f"| Trial | {header} | " + " | ".join(COLUMNS) + " | Average NEES |\n")
        f.write("|" + "---|" * (len(COLUMNS) + 3) + "\n")
        for trial, configuration, values, nees in rows:
            nees_cell = f"{nees:.3f}" if nees is not None else ""
            f.write(f"| {trial} | {configuration} | " + " | ".join(f"{v:.3f}" for v in values) + f" | {nees_cell} |\n")

    with open(os.path.join(RESULTS, name + ".tex"), "w") as f:
        for trial, configuration, values, nees in rows:
            nees_cell = f"{nees:.3f}" if nees is not None else ""
            f.write(f"{trial} & {configuration} & " + " & ".join(f"{v:.3f}" for v in values) + f" & {nees_cell} \\\\\n")
    print(f"Wrote results/{name}.{{json,md,tex}}")


def table5():
    rows = []
    for run, folder, label in SIMULATIONS:
        samples = errors(os.path.join(RESULTS, run), os.path.join(DATA, folder), 0, 20, locked_base=True)
        # Pose in mm and rad, velocity in mm/s and rad/s; NEES over pose and velocity (12 DoF).
        rows.append((label, "Pose", cells(summarize(samples, "pose", scale=1000.0)), average_nees(samples)))
        rows.append((label, "Velocity", cells(summarize(samples, "velocity", scale=1000.0)), None))
    write_table("table5", "Data Type", rows)


def table6():
    rows = []
    for trial, folder, start, end, label in EXPERIMENTS:
        for mode, mode_label in MODES:
            samples = errors(os.path.join(RESULTS, f"{trial}_{mode}"), os.path.join(DATA, folder), start, end)
            # Position in % of each frame's arclength, rotation in rad; NEES over the pose (6 DoF).
            rows.append((label, mode_label, cells(summarize(samples, "pose", percent=True)), average_nees(samples)))
    write_table("table6", "Configuration", rows)


if __name__ == "__main__":
    table5()
    table6()
