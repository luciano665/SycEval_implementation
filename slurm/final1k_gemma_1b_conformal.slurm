#!/bin/bash
#SBATCH --job-name=final1k_gemma_1b_conf
#SBATCH --output=logs/output_final1k_gemma_1b_conf_%j.txt
#SBATCH --error=logs/error_final1k_gemma_1b_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu_7day

# Final experiment (N=1000): CONFORMAL (CCR) for gemma_1b (MedQuad).
# Same protocol as v9n/v10 (deployable-only, leak-free rewrite,
# folding-only labels, global thresholds, alpha=0.10), scaled to N=1000,
# calib_frac=0.2 -> 200 calib / 800 test (see baseline script's comment
# for the justification -- confirmed via analyze_calib_size_projection.py
# against real data before choosing this split).
# threshold_method=exact_crc: uses the more rigorous finite-sample bound
# (Angelopoulos & Bates) added on LM_corrections/conformal_v10, rather
# than the original Wilson approximation.
# gpu_7day + 3-day budget: our slowest N=300 conformal job took 13:19:08;
# scaled ~3.3x that could approach 44+ hours, too close to gpu_2day's
# 48h cap for comfort.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/final_n1000_medquad

echo "--- final N=1000 CONFORMAL gemma_1b (alpha=0.10, exact_crc, calib_frac=0.2) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --enable_rewrite \
  --domain medquad \
  --threshold_method exact_crc \
  --thresholds_out results/final_n1000_medquad/thresholds_gemma_1b.json \
  --out results/final_n1000_medquad/run_conformal_gemma_1b.json
echo "--- conformal gemma_1b exited 0 ---"
