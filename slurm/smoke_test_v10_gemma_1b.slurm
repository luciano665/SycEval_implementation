#!/bin/bash
#SBATCH --job-name=smoke_v10_gemma_1b
#SBATCH --output=logs/output_smoke_v10_gemma_1b_%j.txt
#SBATCH --error=logs/error_smoke_v10_gemma_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# conformal_v10 smoke test (N=20, mode=both, threshold_method=exact_crc).
# Run this BEFORE submit_v10_suite.sh, same as smoke_test_v9_gemma_1b.slurm
# was run before the v9 overnight suite. Validates the new
# --threshold_method flag end-to-end against a real model:
#   1) CLI accepts --threshold_method exact_crc without error.
#   2) thresholds file contains "threshold_method": "exact_crc",
#      "xi_certified", and "alpha_min" (new fields; absent in v9n's files).
#   3) Pipeline behavior otherwise unchanged from v9n's smoke test (same
#      model, same flags apart from --threshold_method).
#
# What to check after it finishes:
#   - Job exits 0, no crashes (the new argparse choice/ValueError paths
#     didn't break anything).
#   - results/smoke_test_v10/smoke_thresholds_gemma_1b.json: confirm
#     "threshold_method": "exact_crc" is present, and xi_certified/
#     alpha_min are real numbers (not null).
#   - calibration_failed at N=20 is expected either way (too little data
#     regardless of bound) -- not a signal at this tiny N, same caveat as
#     the v9n smoke test.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test_v10

echo "--- conformal_v10 SMOKE (threshold_method=exact_crc): N=20, mode=both ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 20 \
  --enable_rewrite \
  --domain medquad \
  --threshold_method exact_crc \
  --thresholds_out results/smoke_test_v10/smoke_thresholds_gemma_1b.json \
  --out results/smoke_test_v10/smoke_conformal_gemma_1b.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE. Inspect: ---"
echo "  results/smoke_test_v10/smoke_conformal_gemma_1b.json"
echo "  results/smoke_test_v10/smoke_thresholds_gemma_1b.json  (check threshold_method/xi_certified/alpha_min)"
