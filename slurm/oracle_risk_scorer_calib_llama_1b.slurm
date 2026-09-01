#!/bin/bash
#SBATCH --job-name=oracle_calib_llama_1b
#SBATCH --output=logs/output_oracle_calib_llama_1b_%j.txt
#SBATCH --error=logs/error_oracle_calib_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_2day

# Mentor-requested diagnostic (see docs/FINAL_DIAGNOSTICS_TRACKER.md):
# "test an oracle or ground truth risk scorer. If calibration works there,
# the bottleneck is likely the risk scorer; if it still fails, that points
# to a more fundamental limitation."
#
# --oracle_risk_scorer_calib: risk scorer sees the reference answer during
# CALIBRATION ONLY (see calibration_collect's docstring -- test-time stays
# leak-free, this flag never touches it). mode=calibrate (not "both"):
# there is no deployable test-time counterpart to run against an oracle-
# assisted threshold, so this is calibration-only and skips the (much more
# expensive) test phase entirely -- cheaper than the v9/v10 "both" runs.
# Same alpha/threshold_method/data as v10 otherwise, so the tau_global /
# calibration_failed / alpha_min output is directly comparable to
# results/v10_exact_crc_medquad/thresholds_llama_1b.json (or v9n's, if v10
# hasn't landed yet).
#
# DO NOT treat this run's thresholds file as deployable. It exists to
# answer one yes/no question: does calibration succeed with the risk
# scorer handed maximum possible information. See the thresholds file's
# "oracle_risk_scorer_calib": true flag -- that marks it as diagnostic-only
# for anyone who finds the file later.
#
# --max_items 150 (not 300): matches v9n/v10's calibration slice size
# exactly (N=300, calib_frac=0.5 -> 150 calib items). mode=calibrate loads
# --max_items directly with no split, so this keeps calibration-set size
# identical to the real runs -- otherwise a bigger calibration set alone
# could tighten the statistical bound and confound the oracle comparison.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/oracle_diagnostic_medquad

echo "--- oracle-assisted risk scorer, CALIBRATION ONLY, llama_1b (N=150, alpha=0.10) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode calibrate \
  --backend hf \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --max_items 150 \
  --domain medquad \
  --threshold_method exact_crc \
  --oracle_risk_scorer_calib \
  --thresholds_out results/oracle_diagnostic_medquad/thresholds_llama_1b.json \
  --out results/oracle_diagnostic_medquad/calib_llama_1b.json
echo "--- oracle-calibration llama_1b exited 0 ---"
