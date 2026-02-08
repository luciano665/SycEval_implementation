#!/bin/bash
#SBATCH --job-name=smoke_test_conformal
#SBATCH --output=logs/smoke_conformal_%j.txt
#SBATCH --error=logs/smoke_conformal_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Create results directory if it doesn't exist
mkdir -p results/smoke_test_v1

# Run Conformal Prediction (V2) - SMOKE TEST (100 items)
srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/Llama-3.2-3B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --alpha 0.1 \
  --calib_frac 0.2 \
  --max_items 100 \
  --out results/smoke_test_v1/llama_3B_conformal.json
