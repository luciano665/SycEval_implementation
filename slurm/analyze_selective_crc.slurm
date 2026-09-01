#!/bin/bash
#SBATCH --job-name=analyze_selective_crc
#SBATCH --output=logs/output_analyze_selective_crc_%j.txt
#SBATCH --error=logs/error_analyze_selective_crc_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_2day

# Retrofits the exact conformal-risk-control bound (conformal_v2/
# conformal_thresholds.py: fit_global_threshold_exact_crc, certifiable_fraction)
# onto the ALREADY-COLLECTED v9 calibration checkpoints. No model loading, no
# LLM calls, no GPU work -- pure arithmetic over the .calib.partial.jsonl
# files the real v9 suite already wrote. Only needs SLURM because the
# checkpoints live on the cluster filesystem, not because it needs compute;
# --cpus-per-task=1 and a 10-minute budget are generous for what this does.
#
# Run AFTER the real v9 suite (docs/V9_EXPERIMENT_TRACKER.md) has completed
# and results/v9_night_medquad/*.calib.partial.jsonl exist.
#
# What to check in the output log:
#   - Any model still showing "MISSING checkpoint" -> that run's
#     .calib.partial.jsonl isn't where this script expects it; verify the
#     path against docs/V9_EXPERIMENT_TRACKER.md's "Results locations".
#   - "RESCUED: bound-choice artifact, not a real ceiling" -> Wilson's
#     asymptotic bound was too conservative for that model; it certifies
#     under the exact bound with the calibration data already on disk, no
#     rerun needed.
#   - "STILL FAILS: real ceiling regardless of bound" -> confirms the
#     ceiling finding is robust to the choice of statistical bound.
#   - alpha_min per model: the loosest alpha at which any certifiable
#     region exists -- replaces the old "swept up to 0.25, still failed"
#     grid-search language with one exact number.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

echo "--- exact-CRC retrofit on v9 calibration checkpoints (alpha=0.10) ---"
srun python3 analyze_selective_crc.py --alpha 0.10 --dir results/v9_night_medquad
echo "--- done ---"
