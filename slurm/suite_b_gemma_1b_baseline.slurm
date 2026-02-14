#!/bin/bash
#SBATCH --job-name=b_gemma_1b_base
#SBATCH --output=logs/b_gemma_1b_base_%j.txt
#SBATCH --error=logs/err_b_gemma_1b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/suite_b_standardized

# Baseline Run (Gemma-1B)
# Judge: Qwen-7B, Rebuttal: Qwen-7B
srun python run_eval.py \
  --tested_model ./models/gemma-3-1b-it \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --backend hf \
  --out results/suite_b_standardized/gemma_1b_baseline.json
