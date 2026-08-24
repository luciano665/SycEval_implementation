#!/bin/bash
#SBATCH --job-name=v9j_gemma_4b_base
#SBATCH --output=logs/output_v9j_gemma_4b_base_%j.txt
#SBATCH --error=logs/error_v9j_gemma_4b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# Independent-judge suite -- BASELINE for gemma_4b (MedQuad, N=300).
# Same structure as v9n_gemma_4b_baseline.slurm, but judge_model swapped to
# Mistral-7B-Instruct-v0.3 (independent of rebuttal_model=Qwen2.5-7B) to
# check for judge self-preference bias. Same seed/N as the conformal arm
# so both share an identical 300-item set; pairing is on question text at
# analysis time.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_judge_split_medquad

echo "--- referee-split BASELINE gemma_4b (N=300, MedQuad, Mistral judge) ---"
srun python run_eval.py \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Mistral-7B-Instruct-v0.3 \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v9_judge_split_medquad/run_baseline_gemma_4b.json
echo "--- baseline gemma_4b exited 0 ---"
