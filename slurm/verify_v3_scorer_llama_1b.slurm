#!/bin/bash
#SBATCH --job-name=verify_v3_llama_1b
#SBATCH --output=logs/output_verify_v3_llama_1b_%j.txt
#SBATCH --error=logs/error_verify_v3_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_2day

# Verifies the v3 risk-scorer prompt (just deployed to production
# conformal_v2/syco_risk.py) actually improves AUC on real data, before
# committing to a full 300-item suite. N=60 (30 calib items -> 240
# calibration instances) -- bigger than a smoke test for a more reliable
# AUC estimate, still much cheaper than the full 300.
#
# llama_1b chosen because it's a borderline case: v9n's overall bad rate
# (11.4%) was only just above the 10% target, with AUC 0.302 (inverted)
# on the OLD scorer -- a working scorer might plausibly find a safe
# subset here where the old one couldn't.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/verify_v3_scorer

echo "--- v3 scorer verification: llama_1b (N=60) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 60 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/verify_v3_scorer/thresholds_llama_1b.json \
  --out results/verify_v3_scorer/run_conformal_llama_1b.json
echo "--- llama_1b exited 0 ---"
