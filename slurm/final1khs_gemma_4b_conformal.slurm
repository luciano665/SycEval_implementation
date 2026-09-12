#!/bin/bash
#SBATCH --job-name=final1khs_gemma_4b_conf
#SBATCH --output=logs/output_final1khs_gemma_4b_conf_%j.txt
#SBATCH --error=logs/error_final1khs_gemma_4b_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu_7day

# Final experiment, HealthSearchQA (N=1000): CONFORMAL (CCR) for gemma_4b.
# Identical to final1k_gemma_4b_conformal.slurm except --domain healthsearch:
# deployable-only, leak-free rewrite, folding-only labels, alpha=0.10,
# threshold_method=exact_crc, calib_frac=0.2 (200 calib / 800 test).
# Keeping every other knob fixed is the point -- it makes the MedQuad and
# HealthSearchQA results directly comparable, so any difference is the
# dataset and not a config change.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/final_n1000_healthsearch

echo "--- final N=1000 HSQ CONFORMAL gemma_4b (alpha=0.10, exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --enable_rewrite \
  --domain healthsearch \
  --threshold_method exact_crc \
  --thresholds_out results/final_n1000_healthsearch/thresholds_gemma_4b.json \
  --out results/final_n1000_healthsearch/run_conformal_gemma_4b.json
echo "--- conformal gemma_4b exited 0 ---"
