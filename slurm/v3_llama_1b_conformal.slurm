#!/bin/bash
#SBATCH --job-name=conformal_llama_1B
#SBATCH --output=logs/output_conformal_llama_1B_%j.txt
#SBATCH --error=logs/error_conformal_llama_1B_%j.txt
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
mkdir -p results/final_experiment_v3

# Run Conformal Prediction (V3 - Truth Grounded + Rewrite)
# Tested: Llama-3.2-1B-Instruct
# Judge: Qwen2.5-7B-Instruct
echo "--- Llama 1B Conformal v3 ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/Llama-3.2-1B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --alpha 0.1 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --enable_rewrite \
  --out results/final_experiment_v3/run_conformal_llama_1B_v3.json
