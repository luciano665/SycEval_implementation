#!/bin/bash
#SBATCH --job-name=download_models_qwen_mistral
#SBATCH --output=logs/output_dl_qm_%j.txt
#SBATCH --error=logs/error_dl_qm_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_7day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

echo "--- Downloading Mistral Judge (7B) ---"
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/Mistral-7B-Instruct-v0.3
