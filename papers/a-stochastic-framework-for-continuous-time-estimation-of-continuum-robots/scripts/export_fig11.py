"""Fig. 11: tip position over time on 'Fast Contact' (fused), with 3-sigma bounds, at the
estimated nodes and at 10 interpolated times per node interval, and the ground truth.

Writes, in results/fig11/ (the files figures/interpolation/time_interpolation_figure.tex reads):
    tikz_estimate_data_tip.csv       interpolated
    tikz_estimate_data_tip_est.csv   estimated nodes
    tikz_ground_truth_data_tip.csv   ground truth
Times are relative to the start of the trial (10 s).
"""

import os

from common import DATA, RESULTS, load_estimate, load_ground_truth, tip_positions, write_tikz

START, END, LENGTH = 10.0, 20.0, 0.466
run = os.path.join(RESULTS, "fast_contact_fused")
out = os.path.join(RESULTS, "fig11")


def relative(rows):
    return [[row[0] - START] + row[2:] for row in sorted(rows)]


write_tikz(os.path.join(out, "tikz_estimate_data_tip.csv"), "time", relative(tip_positions(load_estimate(os.path.join(run, "tip_samples")))))
write_tikz(os.path.join(out, "tikz_estimate_data_tip_est.csv"), "time", relative(tip_positions(load_estimate(os.path.join(run, "tip_nodes")))))

with open(os.path.join(out, "tikz_ground_truth_data_tip.csv"), "w") as f:
    f.write("time,pos1,pos2,pos3\n")
    for type_, t, s, T in load_ground_truth(os.path.join(DATA, "video7"), START, END):
        if type_ == "pose" and abs(s - LENGTH) < 1e-9:
            f.write(f"{t - START:.9g},{T[0, 3]:.9g},{T[1, 3]:.9g},{T[2, 3]:.9g}\n")
print("Wrote results/fig11/")
