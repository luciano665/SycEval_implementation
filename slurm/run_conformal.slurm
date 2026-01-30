#!/bin/bash
#SBATCH --job-name=conformal_eval
#SBATCH --output=logs/conformal_%j.txt
#SBATCH --error=logs/conformal_error_%j.txt
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
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Ensure cache dir exists
mkdir -p $HF_HOME

# Default values if not provided via --export
TESTED_MODEL=${TESTED_MODEL:-"/users/al00113/models/Llama-3.2-3B-Instruct"}
DATASET_PATH=${DATASET_PATH:-"data/medDataset_processed.csv"}
OUT_FILE=${OUT_FILE:-"results/conformal_results.json"}
THRESHOLD=${CONFORMAL_THRESHOLD:-0.5}

# Run Conformal Eval
# Note: Using run_eval.py instead of distill_eval.py
srun --unbuffered python -u run_eval.py \
  --backend hf \
  --tested_model "$TESTED_MODEL" \
  --rebuttal_model /users/al00113/models/Llama-3.2-1B-Instruct \
  --judge_model /users/al00113/models/Llama-3.2-3B-Instruct \
  --max_items 500 \
  --enable_conformal \
  --conformal_threshold "$THRESHOLD" \
  --dataset_path "$DATASET_PATH" \
  --out "$OUT_FILE"
