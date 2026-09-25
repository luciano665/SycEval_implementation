#!/bin/bash
#SBATCH --job-name=oracle1khs_gemma_4b
#SBATCH --output=logs/output_oracle1khs_gemma_4b_%j.txt
#SBATCH --error=logs/error_oracle1khs_gemma_4b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_2day

# Oracle-assisted risk scorer, CALIBRATION ONLY, gemma_4b (HealthSearchQA, N=1000 scale).
# Full-suite version of the N=300 diagnostic in
# slurm/oracle_risk_scorer_calib_gemma_4b.slurm -- rerun at the same scale and on
# both datasets as the paper's primary results, so the oracle comparison no
# longer carries a "N=300, MedQuad only" caveat.
#
# --max_items 1000 --calib_frac 0.2 reproduces the final suite's calibration
# slice ITEM-FOR-ITEM (the same first 200 of the same seeded 1000-item sample).
# Loading 200 items directly would sample a different set -- rng.sample is not
# prefix-stable across different k -- and the comparison against
# results/final_n1000_*/thresholds_gemma_4b.json would be confounded by items
# rather than isolating the oracle. mode=calibrate honours --calib_frac as of
# the LM_corrections change to run_conformal_v2.py; verify by diffing
# data_split.calib_question_hashes against the deployable thresholds file.
#
# Calibration only: no test phase (the oracle threshold has no deployable
# test-time path by design -- building one would reintroduce the C1 leak), so
# this is ~1/5 the work of a full conformal run.
#
# DIAGNOSTIC ONLY. Do not report the resulting tau as a deployable result.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/oracle_n1000_healthsearch

echo "--- oracle-assisted risk scorer, CALIBRATION ONLY, gemma_4b, HealthSearchQA (200 calib items, alpha=0.10, exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode calibrate \
  --backend hf \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --max_items 1000 \
  --calib_frac 0.2 \
  --domain healthsearch \
  --threshold_method exact_crc \
  --oracle_risk_scorer_calib \
  --thresholds_out results/oracle_n1000_healthsearch/thresholds_gemma_4b.json \
  --out results/oracle_n1000_healthsearch/calib_gemma_4b.json
echo "--- oracle-calibration gemma_4b (HealthSearchQA) exited 0 ---"
