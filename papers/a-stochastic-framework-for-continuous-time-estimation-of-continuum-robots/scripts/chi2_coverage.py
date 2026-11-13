"""Fig. 12: chi-square coverage of the pose estimate over the five experimental trajectories,
for the pose-only, gyro-only and fused configurations.

For every ground-truth pose sample, the NEES of the 6-DoF pose error is compared with the
chi-square(6) quantile of each nominal coverage; the empirical coverage is the fraction of
samples below it.

Writes results/fig12/chi2_coverage.csv (nominal, gyro, pose, fused, in %) and
results/fig12/chi2_coverage.pdf.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from common import DATA, RESULTS, errors
from make_tables import EXPERIMENTS

def apply_pgfplots_like_style() -> None:
	"""Style matplotlib output to resemble default TikZ/pgfplots aesthetics."""
	print(plt.rcParams.keys())
	plt.rcParams.update(
		{
			"font.size": 9,
			"axes.labelsize": 9,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
			"legend.fontsize": 9,
			"figure.facecolor": "white",
			"axes.facecolor": "white",
			"axes.edgecolor": "black",
			"axes.linewidth": 0.2,
			"axes.grid": True,
			"grid.color": "0.85",
			"grid.linestyle": "-",
			"grid.linewidth": 0.2,
			"lines.linewidth": 0.2,
			"hatch.linewidth": 0.2,
			"font.family": "serif",
			"font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
			"mathtext.fontset": "cm",
			"xtick.direction": "in",
			"ytick.direction": "in",
			"xtick.major.size": 4,
			"ytick.major.size": 4,
			"xtick.minor.visible": False,
			"ytick.minor.visible": False,
			"legend.frameon": True,
			"legend.framealpha": 1.0,
			"legend.facecolor": "white",
			"legend.edgecolor": "black",
			"patch.linewidth": 0.2,
		}
	)
apply_pgfplots_like_style()

out = os.path.join(RESULTS, "fig12")
os.makedirs(out, exist_ok=True)

nominal = np.arange(0, 101)
coverage = {}
for mode in ("gyro", "pose", "fused"):
    nees = []
    for trial, folder, start, end, _ in EXPERIMENTS:
        samples = errors(os.path.join(RESULTS, f"{trial}_{mode}"), os.path.join(DATA, folder), start, end)
        nees += [x["nees"] for x in samples if x["nees"] is not None]
    nees = np.array(nees)
    thresholds = chi2.ppf(nominal / 100.0, df=6)
    coverage[mode] = np.array([100.0 * np.mean(nees <= q) for q in thresholds])
    print(f"{mode}: {len(nees)} samples")

with open(os.path.join(out, "chi2_coverage.csv"), "w") as f:
    f.write("nominal,gyro,pose,fused\n")
    for i, n in enumerate(nominal):
        f.write(f"{n},{coverage['gyro'][i]:.4f},{coverage['pose'][i]:.4f},{coverage['fused'][i]:.4f}\n")

width = 88.9 / 25.4
plt.figure(figsize=(width, width * 6.0 / 8.5), layout="constrained")
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
plt.plot(nominal, coverage["gyro"], color="#0000FF", linewidth=0.7, label="Gyro only")
plt.plot(nominal, coverage["pose"], color="#FF0000", linewidth=0.7, label="Pose only")
plt.plot(nominal, coverage["fused"], color="#00FF00", linewidth=0.7, label="Fused")
plt.plot([0, 100], [0, 100], "k--", linewidth=0.7, label="Ideal 1:1")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("Nominal Coverage [%]")
plt.ylabel("Empirical Coverage [%]")
plt.grid(True, linewidth=0.2)
plt.legend(loc="lower right")
plt.gcf().get_layout_engine().set(w_pad=0, h_pad=0)
plt.savefig(os.path.join(out, "chi2_coverage.pdf"))
print("Wrote results/fig12/")
