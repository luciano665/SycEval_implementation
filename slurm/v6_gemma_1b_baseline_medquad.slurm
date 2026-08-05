#!/bin/bash
#SBATCH --job-name=v6_gemma_1b_baseline_mq
#SBATCH --output=logs/output_v6_gemma_1b_baseline_mq_%j.txt
#SBATCH --error=logs/error_v6_gemma_1b_baseline_mq_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_7day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/medDataset_v6_1000

echo "--- Gemma 1B V6 Baseline (N=1000, MedQuad) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --domain medquad --out results/medDataset_v6_1000/run_baseline_gemma_1b_v5.json
