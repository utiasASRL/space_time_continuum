#!/bin/bash
# Runs every estimate behind Tables V-VI and Figs. 11, 12 and 14, then the scripts that turn
# them into tables and figure data (in results/). The runtime sweep of Fig. 13 is separate:
# see run_runtime.sh.
set -e
cd "$(dirname "$0")"
RUN=runner/build/tro_runner

# --- Q3 check on the hold-out data (Out-of-Bounds recording, 25-35 s, fused) ---
$RUN configs/holdout/experiment_base_q3_config.json   configs/holdout/holdout_fused.json --output results/holdout_q3_config
$RUN configs/holdout/experiment_base_q3_appendix.json configs/holdout/holdout_fused.json --output results/holdout_q3_appendix
python3 scripts/q3_holdout.py

# --- Table V: simulation ---
for trial in simulation simulation_high_noise simulation_dropout; do
    $RUN configs/simulation_base.json configs/trials/$trial.json --output results/$trial
done

# --- Table VI and Fig. 12: experiments, pose-only / gyro-only / fused ---
for trial in out_of_bounds fast_contact impulse_1 impulse_2 slow_free_space; do
    for mode in pose gyro fused; do
        extra=""
        if [ "$trial" = fast_contact ] && [ "$mode" = fused ]; then extra="--time_samples 10"; fi   # Fig. 11
        if [ "$trial" = out_of_bounds ] && [ "$mode" = pose ]; then extra="--shape_times 5,9"; fi   # Fig. 14
        $RUN configs/experiment_base.json configs/trials/${trial}_${mode}.json --output results/${trial}_${mode} $extra
    done
done

# --- Fig. 14: the quasi-static benchmark, and its prior alone ---
$RUN configs/experiment_base.json configs/trials/out_of_bounds_pose.json --quasi_static --shape_times 5 --output results/fig14_quasi_static
$RUN configs/experiment_base.json configs/trials/out_of_bounds_pose.json --quasi_static --prior_only --shape_times 0 --output results/fig14_quasi_static_prior

# --- Tables and figure data ---
python3 scripts/make_tables.py
python3 scripts/export_fig11.py
python3 scripts/chi2_coverage.py
python3 scripts/export_fig14.py
