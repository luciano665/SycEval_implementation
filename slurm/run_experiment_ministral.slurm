#!/bin/bash
#SBATCH --job-name=syceval_ministral
#SBATCH --output=logs/output_ministral_%j.txt
#SBATCH --error=logs/error_ministral_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_7day

# Load environment
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Debugging info
nvidia-smi

# Set HF Cache to scratch to avoid home quota limits
# Using the system-defined $SCRATCH variable
export HF_HOME=$SCRATCH/hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Ensure cache dir exists
mkdir -p $HF_HOME
# Run experiment
# Using srun --unbuffered python -u allows for real-time log viewing 
srun --unbuffered python -u distill_eval.py \
  --backend hf \
  --teacher_model ./models/Ministral-3-8B-Instruct-2512 \
  --student_model ./models/Ministral-3-3B-Instruct-2512 \
  --judge_model ./models/Llama-3.1-8B-Instruct \
  --rebuttal_model ./models/Llama-3.1-8B-Instruct \
  --max_items 1000 \
  --out ministral_distill_results_1000.json
