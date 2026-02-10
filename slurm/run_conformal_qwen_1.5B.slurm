#!/bin/bash
#SBATCH --job-name=conformal_qwen_1.5B
#SBATCH --output=logs/output_conformal_qwen_1.5B_%j.txt
#SBATCH --error=logs/error_conformal_qwen_1.5B_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Create results directory if it doesn't exist
mkdir -p results/final_run_v1

# Models
# Test: Qwen2.5-1.5B-Instruct
# Judge: Mistral-7B-Instruct-v0.2
echo "--- Qwen 1.5B Conformal ---"

# Run Conformal Prediction (V2)
srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/Qwen2.5-1.5B-Instruct \
  --judge_model ./models/Mistral-7B-Instruct-v0.2 \
  --rebuttal_model ./models/Mistral-7B-Instruct-v0.2 \
  --alpha 0.1 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --out results/final_run_v1/qwen_1.5B_conformal.json
