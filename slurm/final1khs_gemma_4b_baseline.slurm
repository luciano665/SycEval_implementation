#!/bin/bash
#SBATCH --job-name=final1khs_gemma_4b_base
#SBATCH --output=logs/output_final1khs_gemma_4b_base_%j.txt
#SBATCH --error=logs/error_final1khs_gemma_4b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu_2day

# Final experiment, HealthSearchQA (N=1000): BASELINE for gemma_4b.
# Companion to the MedQuad final suite (final1k_*) -- identical protocol,
# identical split (calib_frac=0.2 -> 200 calib / 800 test), only
# --domain changes. Gives the paper a second dataset under the corrected,
# leak-free pipeline; the old v6-era HealthSearchQA results are NOT
# reusable (review_findings.md C1: oracle truth leaked into the test-time
# intervention there).
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/final_n1000_healthsearch

echo "--- final N=1000 HSQ BASELINE gemma_4b ---"
srun python run_eval.py \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 1000 \
  --domain healthsearch \
  --out results/final_n1000_healthsearch/run_baseline_gemma_4b.json
echo "--- baseline gemma_4b exited 0 ---"
