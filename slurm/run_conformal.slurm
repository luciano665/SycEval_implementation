#!/bin/bash
#SBATCH --job-name=syceval_conformal
#SBATCH --output=logs/output_conformal_%j.txt
#SBATCH --error=logs/error_conformal_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_2day

# Load environment
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Debugging info
nvidia-smi

# Set HF Cache to scratch to avoid home quota limits
export HF_HOME=$SCRATCH/hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Ensure cache dir exists
mkdir -p $HF_HOME

# Run Conformal Experiment
# Defaulting to 100 max items for valid stats in a reasonable time
# Using backend=hf since we are on HPC + GPUs
srun --unbuffered python -u run_conformal.py \
  --backend hf \
  --tested_model meta-llama/Llama-3.2-3B-Instruct \
  --rebuttal_model meta-llama/Llama-3.1-8B-Instruct \
  --judge_model meta-llama/Llama-3.1-8B-Instruct \
  --max_items 100 \
  --alpha 0.3 \
  --out results/conformal_results_100.json
