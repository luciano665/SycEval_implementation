#!/bin/bash
#SBATCH --job-name=download_ministral
#SBATCH --output=logs/download_ministral_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day
#SBATCH --mem=32G

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

echo "Downloading Ministral-8B-Instruct-2410..."

# Install hf_transfer for speed if feasible, but sticking to standard cli for safety
pip install -U "huggingface_hub[cli]"

# Download to the specific folder user requested
# HF ID: mistralai/Ministral-8B-Instruct-2410
MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
TARGET_DIR="/users/al00113/models/Ministral-8B-Instruct-2410"

echo "Downloading $MODEL_ID to $TARGET_DIR..."

huggingface-cli download $MODEL_ID \
    --local-dir $TARGET_DIR \
    --local-dir-use-symlinks False \
    --token $HF_TOKEN

echo "Download Complete."
echo "Path: $TARGET_DIR"
