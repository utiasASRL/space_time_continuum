"""Shared helpers for the TRO result scripts: SE(3), reading the runner's outputs and the
ground truth, and the error metrics of the paper (Sec. III-E).

Every estimate file is the runner's export format (error_pred.txt / error_pred_cov.txt):
    time, arclength, Log(T_ib) [rho, phi], eps, varpi        (legacy [trans, rot] order)
    time, arclength, row-major 18 x 18 covariance of [pose, eps, varpi]
Entries a prior does not estimate are nan.
"""

import csv
import glob
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")


# ----------------------------------------------------------------------------
# SE(3), [rho, phi] ordering
# ----------------------------------------------------------------------------

def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_jacobian(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + 0.5 * hat(phi)
    axis = phi / angle
    s, c = math.sin(angle), math.cos(angle)
    return (s / angle) * np.eye(3) + (1.0 - s / angle) * np.outer(axis, axis) + ((1.0 - c) / angle) * hat(axis)


def so3_exp(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + hat(phi)
    axis = phi / angle
    return math.cos(angle) * np.eye(3) + (1.0 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * hat(axis)


def so3_log(C):
    angle = math.acos(np.clip((np.trace(C) - 1.0) / 2.0, -1.0, 1.0))
    w = np.array([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]])
    if angle < 1e-10:
        return w / 2.0
    return angle / (2.0 * math.sin(angle)) * w


def se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = so3_jacobian(xi[3:]) @ xi[:3]
    return T


def se3_log(T):
    phi = so3_log(T[:3, :3])
    return np.concatenate([np.linalg.solve(so3_jacobian(phi), T[:3, 3]), phi])


def se3_inverse(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

def key(t, s):
    """Lookup key of a (time, arclength) sample."""
    return (round(t, 9), round(s, 6))


def read_rows(path):
    with open(path, newline="") as f:
        return [[float(cell) for cell in row if cell.strip() != ""] for row in list(csv.reader(f))[1:]]


def load_estimate(directory):
    """{(t, s): (mean (18,), covariance (18, 18) or None)} of one runner export, in file order."""
    covariances = {}
    cov_path = os.path.join(directory, "error_pred_cov.txt")
    if os.path.exists(cov_path):
        for row in read_rows(cov_path):
            covariances[key(row[0], row[1])] = np.array(row[2:]).reshape(18, 18)
    estimate = {}
    for row in read_rows(os.path.join(directory, "error_pred.txt")):
        k = key(row[0], row[1])
        estimate[k] = (np.array(row[2:]), covariances.get(k))
    return estimate


def load_ground_truth(folder, start_time, end_time):
    """Ground-truth poses and velocities of a trial: [(type, t, s, value)] with type "pose"
    (value a 4 x 4 T_ib) or "velocity" (value a 6-vector, [trans, rot])."""
    samples = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        with open(path, newline="") as f:
            lines = list(csv.reader(f))
        sensor_type, ground_truth = lines[0][1].strip(), lines[1][1].strip()
        if ground_truth != "1":
            continue
        for line in lines[3:]:
            values = [float(v) for v in line if v.strip() != ""]
            t, s = values[0], values[1]
            if t < start_time or t > end_time:
                continue
            if sensor_type in ("vicon", "aurora"):
                samples.append(("pose", t, s, se3_exp(np.array(values[2:8]))))
            elif sensor_type == "velocity":
                samples.append(("velocity", t, s, np.array(values[2:8])))
    return samples


def tip_positions(estimate, n_std=3.0):
    """Rows (t, s, x, x_lb, x_ub, y, ..., z_ub) of an estimate: world-frame positions with
    n_std-sigma bounds (the body-frame translation covariance rotated into the world frame)."""
    rows = []
    for (t, s), (mean, cov) in estimate.items():
        T = se3_exp(mean[0:6])
        position = T[:3, 3]
        std = np.zeros(3)
        if cov is not None:
            R = T[:3, :3]
            std = np.sqrt(np.diag(R @ cov[:3, :3] @ R.T))
        row = [t, s]
        for i in range(3):
            row += [position[i], position[i] - n_std * std[i], position[i] + n_std * std[i]]
        rows.append(row)
    return rows


def write_tikz(path, first_column, rows):
    """A pgfplots table: `first_column`, pos{i}, pos{i}_lb, pos{i}_ub (as the paper's figures read)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(first_column + ",")
        f.write(",".join(f"pos{i},pos{i}_lb,pos{i}_ub" for i in (1, 2, 3)) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.9g}" for v in row) + "\n")


# ----------------------------------------------------------------------------
# Metrics (Sec. III-E)
# ----------------------------------------------------------------------------

def pose_residual(mean, T_gt):
    """Log(T_est^-1 T_gt), [trans, rot]."""
    return se3_log(se3_inverse(se3_exp(mean[0:6])) @ T_gt)


def errors(run_directory, data_folder, start_time, end_time, locked_base=False):
    """Per-sample errors of one run: a list of dicts with type, t, s, trans, rot, and the
    frame's joint NEES and its degrees of freedom (over the types estimated at that (t, s)).
    With locked_base, the base frames (s = 0, held fixed by the estimator) get no NEES."""
    estimate = load_estimate(os.path.join(run_directory, "ground_truth"))
    frames = {}
    samples = []
    for type_, t, s, value in load_ground_truth(data_folder, start_time, end_time):
        k = key(t, s)
        if k not in estimate:
            continue
        mean, cov = estimate[k]
        if type_ == "pose":
            residual, offset = pose_residual(mean, value), 0
        else:
            residual, offset = value - mean[12:18], 12
        samples.append({"type": type_, "t": t, "s": s,
                        "trans": float(np.linalg.norm(residual[:3])), "rot": float(np.linalg.norm(residual[3:]))})
        frame = frames.setdefault(k, {"residual": [], "indices": [], "cov": cov})
        frame["residual"].extend(residual)
        frame["indices"].extend(range(offset, offset + 6))

    nees = {}
    for k, frame in frames.items():
        if frame["cov"] is None or (locked_base and abs(k[1]) < 1e-9):
            continue
        e = np.array(frame["residual"])
        P = frame["cov"][np.ix_(frame["indices"], frame["indices"])]
        nees[k] = (float(e @ np.linalg.solve(P, e)), len(e))
    for sample in samples:
        sample["nees"], sample["dof"] = nees.get(key(sample["t"], sample["s"]), (None, None))
    return samples


def summarize(samples, type_, scale=1.0, percent=False):
    """MAE / RMSE at the tip (the largest arclength) and over the body (every other
    frame), as in the legacy printError: translation in `scale` units (or in % of each
    frame's arclength, frames at s = 0 left out), rotation in rad."""
    typed = [x for x in samples if x["type"] == type_]
    tip_s = max(x["s"] for x in typed)
    result = {}
    for scope in ("tip", "body"):
        chosen = [x for x in typed if (abs(x["s"] - tip_s) < 1e-9) == (scope == "tip")]
        if percent:
            trans = np.array([x["trans"] / x["s"] * 100.0 for x in chosen if x["s"] > 1e-9])
        else:
            trans = np.array([x["trans"] * scale for x in chosen])
        rot = np.array([x["rot"] for x in chosen])
        result[scope] = {"trans_mae": float(trans.mean()), "trans_rmse": float(np.sqrt((trans ** 2).mean())),
                         "rot_mae": float(rot.mean()), "rot_rmse": float(np.sqrt((rot ** 2).mean()))}
    return result


def average_nees(samples):
    """Mean joint NEES over every frame that has one (each (t, s) counted once)."""
    frames = {key(x["t"], x["s"]): x["nees"] for x in samples if x["nees"] is not None}
    return float(np.mean(list(frames.values())))
