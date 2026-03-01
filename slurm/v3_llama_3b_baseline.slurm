#!/bin/bash
#SBATCH --job-name=v3_baseline_3B
#SBATCH --output=logs/v3_baseline_3B_%j.txt
#SBATCH --error=logs/v3_baseline_3B_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_7day

# Load environment
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Debug
nvidia-smi

# HF Cache
export HF_HOME=$SCRATCH/hf_home
# Create results directory if it doesn't exist
mkdir -p results/final_experiment_v3

# Run V3 Pipeline (PURE BASELINE: No Conformal)
# Single Model (Tested vs Rebuttal)
srun --unbuffered python run_eval.py \
  --backend hf \
  --tested_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --out results/final_experiment_v3/suite_b_llama_3b_baseline_v3.json
