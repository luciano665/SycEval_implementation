#!/bin/bash
#SBATCH --job-name=conformal_v2
#SBATCH --output=logs/conformal_v2_%j.txt
#SBATCH --error=logs/conformal_v2_error_%j.txt
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
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Ensure cache dir exists
mkdir -p $HF_HOME

# Run Teammate's Conformal V2 (All-in-One Mode)
# Using python -m to handle imports correctly
srun --unbuffered python -m conformal_v2.run_conformal_v2 \
  --backend hf \
  --tested_model /users/al00113/models/Llama-3.2-3B-Instruct \
  --rebuttal_model /users/al00113/models/Llama-3.2-1B-Instruct \
  --judge_model /users/al00113/models/Llama-3.2-3B-Instruct \
  --max_items 100 \
  --mode both \
  --calib_frac 0.5 \
  --alpha 0.1 \
  --enable_rewrite \
  --out results/conformal_v2_results.json
