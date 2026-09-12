#!/bin/bash
#SBATCH --job-name=smoke_final1khs
#SBATCH --output=logs/output_smoke_final1khs_%j.txt
#SBATCH --error=logs/error_smoke_final1khs_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day

# Smoke test before committing multi-day compute to the HealthSearchQA
# N=1000 suite. The untested combination here is --domain healthsearch
# under the CORRECTED pipeline: every prior HealthSearchQA run in this
# repo is v6-era, i.e. before the leak-free rewrite, folding-only labels,
# and deployable-only changes. HealthSearchQA also pulls its ground truth
# from a different field than MedQuad (Free_form_answer vs Answer), so
# this confirms the loader, judge, and claim decomposition all behave on
# this dataset's format before scaling up.
#
# What to check when it finishes:
#   - exits 0
#   - data_split shows 4 calib / 20 loaded (calib_frac=0.2 of N=20)
#   - records contain real answers, not empty/parse-failure text
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/smoke_test_final1khs

echo "--- HSQ smoke: gemma_1b (N=20, calib_frac=0.2, exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.2 \
  --max_items 20 \
  --enable_rewrite \
  --domain healthsearch \
  --threshold_method exact_crc \
  --thresholds_out results/smoke_test_final1khs/smoke_thresholds_gemma_1b.json \
  --out results/smoke_test_final1khs/smoke_conformal_gemma_1b.json
echo "--- exited 0 ---"
