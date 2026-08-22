#!/bin/bash
#SBATCH --job-name=v9n_gemma_1b_base
#SBATCH --output=logs/output_v9n_gemma_1b_base_%j.txt
#SBATCH --error=logs/error_v9n_gemma_1b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# conformal_v9 overnight suite -- BASELINE for gemma_1b (MedQuad, N=300).
# Same judge (Qwen2.5-7B) and seed as the conformal arm so both share an
# identical 300-item set; pairing is on question text at analysis time.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_night_medquad

echo "--- v9 BASELINE gemma_1b (N=300, MedQuad, Qwen judge) ---"
srun python run_eval.py \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v9_night_medquad/run_baseline_gemma_1b.json
echo "--- baseline gemma_1b exited 0 ---"
