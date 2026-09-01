#!/bin/bash
#SBATCH --job-name=v10_llama_3b_conf
#SBATCH --output=logs/output_v10_llama_3b_conf_%j.txt
#SBATCH --error=logs/error_v10_llama_3b_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# conformal_v10 -- CONFORMAL (CCR) for llama_3b (MedQuad, N=300).
# Identical to v9n except ONE variable: --threshold_method exact_crc
# instead of the Wilson-based default. Same seed, same judge/rebuttal
# models, same data split, so results are directly comparable to
# results/v9_night_medquad/run_conformal_llama_3b.json.
#
# No new baseline needed: results/v9_night_medquad/run_baseline_llama_3b.json
# (already on disk) is the correct, valid baseline to pair this against.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v10_exact_crc_medquad

echo "--- v10 CONFORMAL llama_3b (N=300, alpha=0.10, threshold_method=exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 300 \
  --enable_rewrite \
  --domain medquad \
  --threshold_method exact_crc \
  --thresholds_out results/v10_exact_crc_medquad/thresholds_llama_3b.json \
  --out results/v10_exact_crc_medquad/run_conformal_llama_3b.json
echo "--- conformal llama_3b exited 0 ---"
