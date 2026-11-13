"""Fig. 13: mean-only batch solve time per iteration on 'Fast Contact' (fused), against the
number of time nodes K, for each number of spatial nodes N.

Reads results/fig13/N<N>_K<K>/runtime.json (written by run_runtime.sh) and writes
results/fig13/tikz_per_iter_runtime_data.csv (columns K, N5, N9, ..., N33), the table
figures/runtime_figure/runtime_figure.tex reads.
"""

import json
import os

from common import RESULTS

NS = [5, 9, 13, 17, 21, 25, 29, 33]
KS = [51, 101, 151, 201, 251, 301, 351, 401]

out = os.path.join(RESULTS, "fig13")
with open(os.path.join(out, "tikz_per_iter_runtime_data.csv"), "w") as f:
    f.write("K," + ",".join(f"N{n}" for n in NS) + "\n")
    for K in KS:
        cells = []
        for N in NS:
            with open(os.path.join(out, f"N{N}_K{K}", "runtime.json")) as runtime_file:
                runtime = json.load(runtime_file)
            cells.append(f"{runtime['solve'] / max(runtime['iterations'], 1):.6f}")
        f.write(f"{K}," + ",".join(cells) + "\n")
print("Wrote results/fig13/tikz_per_iter_runtime_data.csv")
