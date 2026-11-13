"""Q3 check: which experiment Q3 (config file or paper appendix) is better tuned, on the fused
hold-out run (Out-of-Bounds data, 25-35 s). The tuning target is an average NEES of 6
(the pose DoF), with the tip position RMSE as the tie-break.

Writes results/q3_holdout.json.
"""

import json
import os

from common import DATA, RESULTS, average_nees, errors, summarize

results = {}
for variant in ("q3_config", "q3_appendix"):
    samples = errors(os.path.join(RESULTS, f"holdout_{variant}"), os.path.join(DATA, "video5"), 25, 35)
    tip = summarize(samples, "pose", percent=True)["tip"]
    results[variant] = {"NEES": average_nees(samples), "tip_rmse_pct": tip["trans_rmse"], "tip_rot_rmse": tip["rot_rmse"]}
    print(f"{variant}: NEES {results[variant]['NEES']:.3f}, tip RMSE {tip['trans_rmse']:.3f} %, {tip['rot_rmse']:.4f} rad")

best = min(results, key=lambda v: (abs(results[v]["NEES"] - 6.0), results[v]["tip_rmse_pct"]))
results["best"] = best
print(f"Closest to NEES = 6: {best}")
with open(os.path.join(RESULTS, "q3_holdout.json"), "w") as f:
    json.dump(results, f, indent=2)
