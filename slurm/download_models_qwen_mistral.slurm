#!/bin/bash
#SBATCH --job-name=download_models_qwen_mistral
#SBATCH --output=logs/output_dl_qm_%j.txt
#SBATCH --error=logs/error_dl_qm_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=general

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

echo "--- Downloading Mistral Judge (7B) ---"
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir models/Mistral-7B-Instruct-v0.2

echo "--- Downloading Qwen 1.5B ---"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

echo "--- Downloading Qwen 3B ---"
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
