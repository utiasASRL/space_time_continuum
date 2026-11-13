"""Appendix C: the R and Q weights of the simulations and experiments as a LaTeX table.

Writes results/appendix_parameters.tex. Q is scaled by the prior's "prior_factor" and R by the
trial's "measurement_factor", as spacetime's ConfigLoader does. The configs are in GTSAM order
([rotational, translational] per 6-vector); the paper lists [translational, rotational].
"""

import json
import os

from common import RESULTS, ROOT

CONFIGS = os.path.join(ROOT, "configs")

SECTIONS = [
    # (title, base config, trial config, footnote mark on R_pose)
    ("Simulation Parameters", "simulation_base.json", "simulation.json", True),
    ("Experiment Parameters", "experiment_base.json", "out_of_bounds_fused.json", False),
]

FOOTNOTE = r"\footnotetext{Note that $\boldsymbol{R}_{\text{pose}}$ is doubled for the `Simulation High Noise' trial.}"


def load(*path):
    with open(os.path.join(CONFIGS, *path)) as f:
        return json.load(f)


def paper_order(values):
    """Swap [rot, trans] to [trans, rot] in every 6-vector; 3-vectors are left as is."""
    if len(values) % 6:
        return list(values)
    out = []
    for i in range(0, len(values), 6):
        out += values[i + 3:i + 6] + values[i:i + 3]
    return out


def sci(x):
    """4e-05 -> 4\\mathrm{E}{-}5, 1500 -> 1.5\\mathrm{E}3."""
    mantissa, exponent = f"{x:.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    exponent = f"{{-}}{-exponent}" if exponent < 0 else str(exponent)
    return f"{mantissa}\\mathrm{{E}}{exponent}"


def rows(label, values):
    """One table row per 6 values, continuation rows with an empty label."""
    chunks = [values[i:i + 6] for i in range(0, len(values), 6)]
    lines = []
    for i, chunk in enumerate(chunks):
        cell = "$" + r",\,".join(sci(v) for v in chunk) + "$"
        if i < len(chunks) - 1:
            cell += ","
        lines.append(f"{label if i == 0 else '':<18} & {cell} \\\\")
    return lines


def section(title, base_file, trial_file, footnote):
    weights_q = load(base_file)["prior"]["wnoa"]["weights"]
    weights_r = load("trials", trial_file)["demo_data"]["weights"]
    prior_factor = weights_q.get("prior_factor", 1.0)
    measurement_factor = weights_r.get("measurement_factor", 1.0)

    lines = [r"\multicolumn{2}{c}{" + title + r"} \\", r"\hline"]
    r_pose_label = r"$\boldsymbol{R}_{\text{pose}}$" + (r"\footnotemark" if footnote else "")
    lines += rows(r_pose_label, paper_order([measurement_factor * v for v in weights_r["R_pose"]]))
    if "R_gyro" in weights_r:
        lines += rows(r"$\boldsymbol{R}_{\text{gyro}}$", [measurement_factor * v for v in weights_r["R_gyro"]])
    for k in range(5):
        values = weights_q.get(f"Q{k}", weights_q.get("Q3") if k == 4 else None)
        lines += rows(f"$\\boldsymbol{{Q}}_{k}$", paper_order([prior_factor * v for v in values]))
    lines.append(r"\hline")
    return lines


def main():
    lines = [
        r"\section{Estimation Parameters}\label{app:parameters}",
        "",
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l p{0.8\linewidth}}",
        r"\hline",
        r"Label & Diagonal Values \\",
        r"\hline",
    ]
    for title, base_file, trial_file, footnote in SECTIONS:
        lines += section(title, base_file, trial_file, footnote)
    lines += [r"\end{tabular}", r"\end{center}", "", FOOTNOTE, ""]

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "appendix_parameters.tex"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("Wrote results/appendix_parameters.tex")


if __name__ == "__main__":
    main()
