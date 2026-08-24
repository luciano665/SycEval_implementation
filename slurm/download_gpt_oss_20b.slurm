#!/bin/bash
#SBATCH --job-name=download_gpt_oss_20b
#SBATCH --output=logs/output_dl_gptoss_%j.txt
#SBATCH --error=logs/error_dl_gptoss_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

echo "--- Downloading gpt-oss-20b ---"
huggingface-cli download openai/gpt-oss-20b --local-dir models/gpt-oss-20b
