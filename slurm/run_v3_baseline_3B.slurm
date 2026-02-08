#!/bin/bash
#SBATCH --job-name=v3_baseline_3B
#SBATCH --output=logs/v3_baseline_3B_%j.txt
#SBATCH --error=logs/v3_baseline_3B_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_2day

# Load environment
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Debug
nvidia-smi

# HF Cache
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Run V3 Pipeline (PURE BASELINE: No Conformal)
# Single Model (Tested vs Rebuttal)
srun --unbuffered python run_eval.py \
  --backend hf \
  --tested_model /users/al00113/models/Llama-3.2-3B-Instruct \
  --rebuttal_model /users/al00113/models/Llama-3.2-1B-Instruct \
  --judge_model /users/al00113/models/Llama-3.2-3B-Instruct \
  --max_items 100 \
  --out results/v3_baseline_3B_results.json
