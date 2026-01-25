#!/bin/bash
#SBATCH --job-name=conformal_calib
#SBATCH --output=logs/calib_%j.txt
#SBATCH --error=logs/calib_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_2day

# Load environment
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Debug
nvidia-smi

export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

echo "Running Calibration on Mixed Dataset..."

# We use the 3B model for calibration because it is the "weakest link"
# Calibrating on the weakest model ensures the threshold is strict enough for everyone.
srun --unbuffered python -u run_calibration.py \
  --n 50 \
  --alpha 0.1 \
  --backend hf \
  --tested_model "/users/al00113/models/Llama-3.2-3B-Instruct" \
  --rebuttal_model "/users/al00113/models/Llama-3.2-1B-Instruct" \
  --judge_model "/users/al00113/models/gpt-oss-20b"
