#!/bin/bash
#SBATCH --job-name=analyze_rewrite_effect
#SBATCH --output=logs/output_analyze_rewrite_effect_%j.txt
#SBATCH --error=logs/error_analyze_rewrite_effect_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_2day

# Mentor-requested diagnostic (see docs/FINAL_DIAGNOSTICS_TRACKER.md):
# "compare directly against a rewrite-only baseline to see how much of the
# sycophancy reduction is really coming from rewriting." Zero new LLM
# calls -- reads run_conformal_<model>.json's individual_records already
# on disk. Same CPU-only rationale as analyze_selective_crc.slurm.
#
# Points at results/v9_night_medquad by default. If v10 has landed, run
# with --dir results/v10_exact_crc_medquad instead for the exact-CRC
# thresholded version (same underlying rewrite mechanism either way, so
# results should be very close, but v10 is the more current/citable run).
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

echo "--- rewrite-effect analysis on already-collected test records ---"
srun python3 analyze_rewrite_effect_v10.py --dir results/v9_night_medquad
echo "--- done ---"
