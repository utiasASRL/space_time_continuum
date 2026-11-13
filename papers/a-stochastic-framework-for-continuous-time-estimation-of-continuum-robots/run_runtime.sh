#!/bin/bash
# Fig. 13: mean-only batch solve time on 'Fast Contact' (fused, 10-20 s) for every number of
# spatial nodes N and time nodes K, then the per-iteration table (results/fig13/).
set -e
cd "$(dirname "$0")"
RUN=runner/build/tro_runner

for N in 5 9 13 17 21 25 29 33; do
    for K in 51 101 151 201 251 301 351 401; do
        $RUN configs/experiment_base.json configs/trials/fast_contact_fused.json \
            --N $N --K $K --no_covariance --output results/fig13/N${N}_K${K}
    done
done
python3 scripts/runtime_fig13.py
