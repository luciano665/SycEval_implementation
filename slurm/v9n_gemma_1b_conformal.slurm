#!/bin/bash
#SBATCH --job-name=v9n_gemma_1b_conf
#SBATCH --output=logs/output_v9n_gemma_1b_conf_%j.txt
#SBATCH --error=logs/error_v9n_gemma_1b_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# conformal_v9 overnight suite -- CONFORMAL (CCR) for gemma_1b (MedQuad, N=300).
# Deployable-only (no oracle), leak-free rewrite, folding-only labels,
# GLOBAL thresholds (group thresholds dropped: they fragment calibration at
# this N). alpha=0.10, calib_frac=0.5 -> 150 calib / 150 test.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_night_medquad

echo "--- v9 CONFORMAL gemma_1b (N=300, alpha=0.10, deployable+folding-only) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 300 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/v9_night_medquad/thresholds_gemma_1b.json \
  --out results/v9_night_medquad/run_conformal_gemma_1b.json
echo "--- conformal gemma_1b exited 0 ---"
