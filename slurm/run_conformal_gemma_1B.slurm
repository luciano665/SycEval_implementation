#!/bin/bash
#SBATCH --job-name=conformal_gemma_1B
#SBATCH --output=logs/output_conformal_gemma_1B_%j.txt
#SBATCH --error=logs/error_conformal_gemma_1B_%j.txt
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

# Create results directory if it doesn't exist
mkdir -p results/final_experiment_v1

# Run Conformal Prediction (V2)
# Tested: gemma-3-1b-it
# Judge: Ministral-8B-Instruct-2410
echo "--- Gemma 1B Conformal ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/gemma-3-1b-it \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --alpha 0.1 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --out results/final_experiment_v1/run_conformal_gemma_1B.json
